"""多级串联盘管链核算测试。

钉住用户点名的判据：
* 单级链与单级核算逐项一致（浮点量级）；
* 两级链与"前级出口即后级进口"手工串联逐级吻合，级间状态逐位一致；
* 全链总冷量 == 各级之和、总去湿量 == 各级之和（望远镜守恒）；
* 沿链温度与含湿量单调；
* 透传级放行、全透传链判无效、空链拒绝；
* 某级装置露点高于该级进口湿球、出风比进风湿、级间气压不一致
  → 定位到级次的结构化错误；
* 全链目标反推：正算结果喂回去能还原各级 ADP/BF；欠定/过定/不可达
  分别拒绝；不收敛报级次、绝不返回试探值。
"""

import pytest

from app import adp
from app import chain as chain_mod
from app.chain import StageSpec, solve_chain, solve_chain_inverse, target_state_from_t_shr
from app.errors import PsychrometricError
from app.states import AirState, expand_state

P = 101325.0


def _specs(*args):
    return [StageSpec(t_adp_c=t, bf=b) for t, b in args]


# --------------------------------------------------------------------------
# 单级退化一致
# --------------------------------------------------------------------------

def test_single_stage_chain_matches_single_coil(demo_inlet):
    chain = solve_chain(demo_inlet, _specs((7.0, 0.2)), m_da=2.0)
    coil = adp.solve_direct(demo_inlet, 7.0, 0.2, m_da=2.0)
    st = chain.stages[0]
    assert st.mode == "direct"
    assert st.adp.t_db_c == pytest.approx(coil.adp.t_db_c, abs=1e-12)
    assert st.bf == pytest.approx(coil.bf, abs=1e-12)
    assert st.outlet.t_db_c == pytest.approx(coil.outlet.t_db_c, abs=1e-12)
    assert st.outlet.w == pytest.approx(coil.outlet.w, abs=1e-14)
    assert st.outlet.enthalpy == pytest.approx(coil.outlet.enthalpy, abs=1e-10)
    for name in ("q_total", "q_sensible", "q_latent"):
        assert getattr(chain.loads, name) == pytest.approx(
            getattr(coil.loads, name), abs=1e-10)
    assert chain.loads.shr == pytest.approx(coil.loads.shr, abs=1e-12)
    assert chain.fractions == pytest.approx([1.0], abs=1e-12)
    assert chain.delta_w == pytest.approx(
        demo_inlet.w - coil.outlet.w, abs=1e-14)


# --------------------------------------------------------------------------
# 两级链 vs 手工串联；级间逐位一致
# --------------------------------------------------------------------------

def test_two_stage_chain_matches_manual_chaining(demo_inlet):
    specs = _specs((12.0, 0.25), (6.0, 0.15))
    chain = solve_chain(demo_inlet, specs, m_da=1.5)

    r1 = adp.solve_direct(demo_inlet, 12.0, 0.25, m_da=1.5)
    r2 = adp.solve_direct(r1.outlet, 6.0, 0.15, m_da=1.5)

    s1, s2 = chain.stages
    # 每一级都与手工串联的单级核算一致
    assert s1.outlet.t_db_c == r1.outlet.t_db_c
    assert s1.outlet.w == r1.outlet.w
    assert s2.inlet.t_db_c == r2.inlet.t_db_c
    assert s2.outlet.t_db_c == r2.outlet.t_db_c
    assert s2.outlet.w == r2.outlet.w
    assert s2.outlet.enthalpy == r2.outlet.enthalpy
    assert s2.loads.q_total == r2.loads.q_total
    # 最终出口一致
    assert chain.outlet.t_db_c == r2.outlet.t_db_c
    assert chain.outlet.w == r2.outlet.w


def test_interstage_state_passed_bit_exact(demo_inlet):
    specs = _specs((14.0, 0.2), (9.0, 0.25), (5.0, 0.15))
    chain = solve_chain(demo_inlet, specs)
    for prev, nxt in zip(chain.stages, chain.stages[1:]):
        # 同一对象直传，不重新展开
        assert nxt.inlet is prev.outlet
        assert nxt.inlet.w == prev.outlet.w
        assert nxt.inlet.enthalpy == prev.outlet.enthalpy
        assert nxt.inlet.p_pa == prev.outlet.p_pa
    assert chain.inlet is chain.stages[0].inlet
    assert chain.outlet is chain.stages[-1].outlet


# --------------------------------------------------------------------------
# 守恒：合计 = 分级之和；单调性；份额
# --------------------------------------------------------------------------

def test_total_load_and_dehumidification_equal_stage_sums(demo_inlet):
    specs = _specs((14.0, 0.2), (9.0, 0.25), (5.0, 0.15))
    chain = solve_chain(demo_inlet, specs, m_da=3.0)
    q_sum = sum(s.loads.q_total for s in chain.stages)
    assert chain.loads.q_total == pytest.approx(q_sum, rel=1e-12)
    # 总冷量独立核对：m·(h_in − h_out)
    assert chain.loads.q_total == pytest.approx(
        3.0 * (demo_inlet.enthalpy - chain.outlet.enthalpy), rel=1e-12)
    dw_sum = sum(s.inlet.w - s.outlet.w for s in chain.stages)
    assert chain.delta_w == pytest.approx(dw_sum, abs=1e-15)
    assert chain.delta_w == pytest.approx(demo_inlet.w - chain.outlet.w, abs=1e-15)
    # 显热 + 潜热 = 总冷量
    assert chain.loads.q_sensible + chain.loads.q_latent == \
        pytest.approx(chain.loads.q_total, abs=1e-9)
    assert 0.0 < chain.loads.shr <= 1.0


def test_monotonic_t_and_w_along_chain(demo_inlet):
    specs = _specs((16.0, 0.15), (11.0, 0.2), (7.0, 0.25), (4.0, 0.3))
    chain = solve_chain(demo_inlet, specs)
    assert len(chain.stages) == 4
    for s in chain.stages:
        assert s.outlet.t_db_c <= s.inlet.t_db_c + 1e-12
        assert s.outlet.w <= s.inlet.w + 1e-15
    assert chain.outlet.t_db_c < demo_inlet.t_db_c
    assert chain.outlet.w < demo_inlet.w


def test_stage_fractions_sum_to_one(demo_inlet):
    specs = _specs((12.0, 0.25), (6.0, 0.15))
    chain = solve_chain(demo_inlet, specs)
    assert sum(chain.fractions) == pytest.approx(1.0, abs=1e-12)
    for s, f in zip(chain.stages, chain.fractions):
        assert f == pytest.approx(s.loads.q_total / chain.loads.q_total,
                                  rel=1e-12)


# --------------------------------------------------------------------------
# 透传级与无效链
# --------------------------------------------------------------------------

def test_bypass_stage_passthrough(demo_inlet):
    specs = [StageSpec(t_adp_c=12.0, bf=0.2), StageSpec(bf=1.0),
             StageSpec(t_adp_c=6.0, bf=0.2)]
    chain = solve_chain(demo_inlet, specs)
    bp = chain.stages[1]
    assert bp.mode == "bypass"
    assert bp.adp is None
    assert bp.bf == 1.0
    assert bp.outlet is bp.inlet
    assert bp.loads.q_total == pytest.approx(0.0, abs=1e-12)
    assert bp.loads.shr is None
    assert chain.fractions[1] == pytest.approx(0.0, abs=1e-12)
    # 透传级不打断守恒
    q_sum = sum(s.loads.q_total for s in chain.stages)
    assert chain.loads.q_total == pytest.approx(q_sum, rel=1e-12)


def test_direct_bf_one_also_passthrough(demo_inlet):
    chain = solve_chain(demo_inlet, _specs((10.0, 1.0), (6.0, 0.2)))
    assert chain.stages[0].loads.q_total == pytest.approx(0.0, abs=1e-12)
    assert chain.stages[1].loads.q_total > 0.0


def test_all_bypass_chain_rejected(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, [StageSpec(bf=1.0), StageSpec(bf=1.0)])
    assert ei.value.code == "invalid_chain"


def test_empty_chain_rejected(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, [])
    assert ei.value.code == "invalid_chain"


def test_forward_chain_with_undetermined_stage_rejected(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, [StageSpec(t_adp_c=10.0, bf=0.2), StageSpec()])
    assert ei.value.code == "invalid_request"
    assert ei.value.stage == 2


# --------------------------------------------------------------------------
# 非法链：定位到级次
# --------------------------------------------------------------------------

def test_stage_adp_above_its_inlet_wetbulb_located(demo_inlet):
    # 第 1 级把风冷到 ~10 °C，第 2 级进口湿球随之降到 ~10 °C；
    # 第 2 级装置露点 20 °C 高于它自己进口的湿球 → 必须定位到第 2 级
    specs = _specs((8.0, 0.1), (20.0, 0.2))
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert ei.value.stage == 2


def test_stage_outlet_wetter_than_its_inlet_located(demo_inlet):
    # 第 2 级目标出风比该级进口还湿
    specs = [StageSpec(t_adp_c=8.0, bf=0.2),
             StageSpec(outlet=expand_state(15.0, P, w=0.012))]
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert ei.value.stage == 2


def test_chain_target_wetter_than_inlet_rejected(demo_inlet):
    target = expand_state(20.0, P, w=0.020)  # 比进风 17.8 g/kg 还湿
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, _specs((7.0, 0.2)), target)
    assert ei.value.code == "not_dehumidifying"


def test_stage_outlet_pressure_mismatch_located(demo_inlet):
    bad = expand_state(15.0, 90000.0, rh=0.9)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain(demo_inlet, [StageSpec(t_adp_c=10.0, bf=0.2),
                                 StageSpec(outlet=bad)])
    assert ei.value.code == "invalid_chain"
    assert ei.value.stage == 2


def test_chain_target_pressure_mismatch_rejected(demo_inlet):
    target = expand_state(15.0, 90000.0, rh=0.8)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, _specs((7.0, 0.2)), target)
    assert ei.value.code == "invalid_chain"


# --------------------------------------------------------------------------
# 全链目标反推：还原
# --------------------------------------------------------------------------

def test_inverse_recovers_free_last_stage(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((12.0, 0.25), (6.0, 0.18)))
    inv_specs = [StageSpec(t_adp_c=12.0, bf=0.25), StageSpec()]
    inv = solve_chain_inverse(demo_inlet, inv_specs, fwd.outlet)
    assert inv.mode == "inverse_target"
    assert inv.stages[1].adp.t_db_c == pytest.approx(6.0, abs=1e-6)
    assert inv.stages[1].bf == pytest.approx(0.18, abs=1e-6)
    assert inv.outlet.t_db_c == pytest.approx(fwd.outlet.t_db_c, abs=1e-6)
    assert inv.outlet.w == pytest.approx(fwd.outlet.w, abs=1e-9)


def test_inverse_recovers_two_partial_stages(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((13.0, 0.3), (7.0, 0.2)))
    inv_specs = [StageSpec(bf=0.3), StageSpec(bf=0.2)]
    inv = solve_chain_inverse(demo_inlet, inv_specs, fwd.outlet)
    assert inv.stages[0].adp.t_db_c == pytest.approx(13.0, abs=1e-5)
    assert inv.stages[1].adp.t_db_c == pytest.approx(7.0, abs=1e-5)
    assert inv.outlet.w == pytest.approx(fwd.outlet.w, abs=1e-9)


def test_inverse_recovers_free_first_stage(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((15.0, 0.2), (9.0, 0.25), (5.0, 0.15)))
    inv_specs = [StageSpec(), StageSpec(t_adp_c=9.0, bf=0.25),
                 StageSpec(t_adp_c=5.0, bf=0.15)]
    inv = solve_chain_inverse(demo_inlet, inv_specs, fwd.outlet)
    assert inv.stages[0].adp.t_db_c == pytest.approx(15.0, abs=1e-5)
    assert inv.stages[0].bf == pytest.approx(0.2, abs=1e-5)


def test_inverse_single_unknown_recovers_adp(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((10.0, 0.22)))
    inv = solve_chain_inverse(demo_inlet, [StageSpec(bf=0.22)], fwd.outlet)
    assert inv.stages[0].adp.t_db_c == pytest.approx(10.0, abs=1e-6)
    assert inv.stages[0].bf == pytest.approx(0.22, abs=1e-12)


def test_inverse_single_stage_free_recovers_both(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((8.0, 0.25)))
    inv = solve_chain_inverse(demo_inlet, [StageSpec()], fwd.outlet)
    assert inv.stages[0].adp.t_db_c == pytest.approx(8.0, abs=1e-6)
    assert inv.stages[0].bf == pytest.approx(0.25, abs=1e-6)
    # 与单级 from_outlet 反算一致
    back = adp.solve_from_outlet(demo_inlet, fwd.outlet)
    assert inv.stages[0].adp.t_db_c == pytest.approx(back.adp.t_db_c, abs=1e-6)
    assert inv.stages[0].bf == pytest.approx(back.bf, abs=1e-6)


def test_inverse_with_t_and_shr_target(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((12.0, 0.25), (6.0, 0.15)))
    target = target_state_from_t_shr(
        demo_inlet, fwd.outlet.t_db_c, fwd.loads.shr)
    assert target.w == pytest.approx(fwd.outlet.w, abs=1e-9)
    inv_specs = [StageSpec(t_adp_c=12.0, bf=0.25), StageSpec()]
    inv = solve_chain_inverse(demo_inlet, inv_specs, target)
    assert inv.stages[1].adp.t_db_c == pytest.approx(6.0, abs=1e-6)
    assert inv.stages[1].bf == pytest.approx(0.15, abs=1e-6)
    assert inv.loads.shr == pytest.approx(fwd.loads.shr, abs=1e-6)


def test_inverse_all_stages_given_consistent_target(demo_inlet):
    specs = _specs((10.0, 0.2), (6.0, 0.2))
    fwd = solve_chain(demo_inlet, specs)
    inv = solve_chain_inverse(demo_inlet, specs, fwd.outlet)
    assert inv.outlet.t_db_c == pytest.approx(fwd.outlet.t_db_c, abs=1e-9)


# --------------------------------------------------------------------------
# 全链目标反推：欠定 / 过定 / 不可达 / 不收敛
# --------------------------------------------------------------------------

def test_inverse_underdetermined_rejected(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((12.0, 0.25), (6.0, 0.18)))
    # 两级全自由：4 个未知量 > 2 个方程
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, [StageSpec(), StageSpec()], fwd.outlet)
    assert ei.value.code == "invalid_chain"
    # 一级自由 + 一级只给 BF：3 个未知量
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, [StageSpec(), StageSpec(bf=0.2)],
                            fwd.outlet)
    assert ei.value.code == "invalid_chain"


def test_inverse_overdetermined_contradiction_rejected(demo_inlet):
    specs = _specs((8.0, 0.2))
    fwd = solve_chain(demo_inlet, specs)
    # 各级全给定，但目标与正算结果矛盾（温度差 2 °C）
    bad = AirState(t_db_c=fwd.outlet.t_db_c + 2.0, w=fwd.outlet.w, p_pa=P)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, specs, bad)
    assert ei.value.code == "invalid_chain"


def test_inverse_single_unknown_h_mismatch_rejected(demo_inlet):
    fwd = solve_chain(demo_inlet, _specs((10.0, 0.22)))
    # 只给 BF（1 个未知量），目标含湿量对、焓对不上 → 过定矛盾
    bad = AirState(t_db_c=fwd.outlet.t_db_c + 1.5, w=fwd.outlet.w, p_pa=P)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, [StageSpec(bf=0.22)], bad)
    assert ei.value.code == "invalid_chain"
    assert ei.value.stage == 1


def test_inverse_unreachable_target_rejected(demo_inlet):
    # 目标又暖又干：含湿量可达但焓无论如何都对不上
    target = expand_state(30.0, P, w=0.005)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet, [StageSpec(bf=0.2), StageSpec(bf=0.2)],
                            target)
    assert ei.value.code == "invalid_chain"


def test_inverse_nonconvergence_reports_stage(demo_inlet, monkeypatch):
    """二分被人为限制迭代次数时必须报不收敛并指明级次，绝不返回试探值。"""
    fwd = solve_chain(demo_inlet, _specs((12.0, 0.25), (6.0, 0.18)))
    real_bisection = chain_mod.bisection

    def starved(f, lo, hi, **kwargs):
        kwargs["max_iter"] = 2  # 两次迭代不可能收敛
        return real_bisection(f, lo, hi, **kwargs)

    monkeypatch.setattr(chain_mod, "bisection", starved)
    with pytest.raises(PsychrometricError) as ei:
        solve_chain_inverse(demo_inlet,
                            [StageSpec(t_adp_c=12.0, bf=0.25), StageSpec()],
                            fwd.outlet)
    assert ei.value.code == "no_convergence"
    assert ei.value.stage == 2


# --------------------------------------------------------------------------
# 混合模式与请求独立
# --------------------------------------------------------------------------

def test_forward_mixed_stage_modes(demo_inlet):
    specs = [
        StageSpec(t_adp_c=14.0, bf=0.2),                    # 正算
        StageSpec(target_t_out_c=15.0, target_shr=0.55),    # 目标温度+SHR
        StageSpec(bf=0.3, target_t_out_c=12.0),             # BF+目标温度
        StageSpec(bf=1.0),                                  # 纯透传
    ]
    chain = solve_chain(demo_inlet, specs)
    assert [s.mode for s in chain.stages] == [
        "direct", "from_outlet", "bf_target_t", "bypass"]
    assert chain.stages[1].outlet.t_db_c == pytest.approx(15.0, abs=1e-6)
    assert chain.stages[2].outlet.t_db_c == pytest.approx(12.0, abs=1e-6)
    q_sum = sum(s.loads.q_total for s in chain.stages)
    assert chain.loads.q_total == pytest.approx(q_sum, rel=1e-12)
    for s in chain.stages:
        assert s.outlet.t_db_c <= s.inlet.t_db_c + 1e-9
        assert s.outlet.w <= s.inlet.w + 1e-12


def test_request_independence_no_state_leak(demo_inlet):
    c1 = solve_chain(demo_inlet, _specs((12.0, 0.2), (6.0, 0.2)))
    c2 = solve_chain(demo_inlet, _specs((8.0, 0.4)))
    assert c1.outlet.t_db_c != pytest.approx(c2.outlet.t_db_c, abs=0.5)
    # 再算一遍 c1，结果不变
    c1b = solve_chain(demo_inlet, _specs((12.0, 0.2), (6.0, 0.2)))
    assert c1b.outlet.t_db_c == c1.outlet.t_db_c
    assert c1b.outlet.w == c1.outlet.w
