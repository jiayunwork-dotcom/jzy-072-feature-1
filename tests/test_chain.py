"""多级串联盘管链核算测试。

钉住用户点名的关系：
* 单级链与原单级核算逐项一致（浮点量级）；
* 两级链拆开用单级手工串，每一级与最终出口都吻合；
* 级间传递逐位一致（同一状态对象，不重新展开）；
* 全链总冷量 == 各级冷量之和、总去湿量 == 各级去湿量之和；
* 沿链温度不回升、含湿量不增加；
* 纯旁通透传级放行，全透传链判无效工况；
* 非法链（ADP 高于该级进口湿球、出风比进风还湿、级间气压不一致、
  空链、欠定、过定、不收敛）被结构化拒绝并定位到级次；
* 带全链目标反推能还原正算时各级的装置露点与旁通系数。
"""

import pytest

from app import adp, chain
from app.errors import PsychrometricError
from app.states import expand_state

# 温和工况四级链（各级出口均远离饱和线，BF=0.5）
ADPS_4 = (10.0, 8.0, 6.5, 5.5)
BFS_4 = (0.5, 0.5, 0.5, 0.5)


def _direct_specs(n):
    return [chain.StageSpec(t_adp_c=a, bf=b)
            for a, b in zip(ADPS_4[:n], BFS_4[:n])]


# --------------------------------------------------------------------------
# 单级退化：一条只有一级的链 == 原单级核算
# --------------------------------------------------------------------------

def test_single_stage_chain_matches_solve_coil_all_modes(demo_inlet):
    direct = adp.solve_direct(demo_inlet, 7.0, 0.2, m_da=2.0)
    cases = [
        (chain.StageSpec(t_adp_c=7.0, bf=0.2),
         dict(t_adp_c=7.0, bf=0.2)),
        (chain.StageSpec(outlet=direct.outlet),
         dict(outlet=direct.outlet)),
        (chain.StageSpec(target_t_out_c=20.0, target_shr=0.60),
         dict(target_t_out_c=20.0, target_shr=0.60)),
        (chain.StageSpec(bf=0.25, target_t_out_c=14.0),
         dict(bf=0.25, target_t_out_c=14.0)),
        (chain.StageSpec(bf=0.25, target_shr=0.6),
         dict(bf=0.25, target_shr=0.6)),
    ]
    for spec, kwargs in cases:
        ref = adp.solve_coil(demo_inlet, m_da=2.0, **kwargs)
        res = chain.solve_chain(demo_inlet, [spec], m_da=2.0)
        assert res.mode == "forward"
        assert len(res.stages) == 1
        st = res.stages[0]
        # 与原单级核算逐项一致（同一求解器，容差在浮点量级）
        assert st.coil.outlet.t_db_c == pytest.approx(ref.outlet.t_db_c, abs=1e-12)
        assert st.coil.outlet.w == pytest.approx(ref.outlet.w, abs=1e-14)
        assert st.coil.outlet.enthalpy == pytest.approx(ref.outlet.enthalpy, abs=1e-10)
        assert st.coil.adp.t_db_c == pytest.approx(ref.adp.t_db_c, abs=1e-9)
        assert st.coil.bf == pytest.approx(ref.bf, abs=1e-12)
        assert st.coil.loads.q_total == pytest.approx(ref.loads.q_total, abs=1e-9)
        assert st.coil.loads.q_sensible == pytest.approx(ref.loads.q_sensible, abs=1e-9)
        assert st.coil.loads.q_latent == pytest.approx(ref.loads.q_latent, abs=1e-9)
        assert st.coil.loads.shr == pytest.approx(ref.loads.shr, abs=1e-12)
        # 单级链的全链合计 == 本级冷量
        assert res.totals.q_total == pytest.approx(ref.loads.q_total, abs=1e-9)
        assert res.totals.shr == pytest.approx(ref.loads.shr, abs=1e-12)
        assert st.load_share == pytest.approx(1.0, abs=1e-12)
        assert not st.passthrough


def test_single_stage_mode_labels(demo_inlet):
    res = chain.solve_chain(
        demo_inlet, [chain.StageSpec(target_t_out_c=20.0, target_shr=0.6)]
    )
    assert res.stages[0].mode == "target_t_shr"
    res = chain.solve_chain(demo_inlet, [chain.StageSpec(t_adp_c=7.0, bf=0.2)])
    assert res.stages[0].mode == "direct"


# --------------------------------------------------------------------------
# 两级链拆开手工串：链结果 == 单级核算按「前级出口即后级进口」串一遍
# --------------------------------------------------------------------------

def test_two_stage_chain_matches_manual_serial(demo_inlet):
    specs = [
        chain.StageSpec(t_adp_c=12.0, bf=0.3),
        chain.StageSpec(bf=0.35, target_t_out_c=12.5),
    ]
    res = chain.solve_chain(demo_inlet, specs, m_da=1.5)

    # 手工串：第一级正算，出口原样喂给第二级
    s1 = adp.solve_direct(demo_inlet, 12.0, 0.3, m_da=1.5)
    s2 = adp.solve_coil(s1.outlet, bf=0.35, target_t_out_c=12.5, m_da=1.5)

    for got, want in zip(res.stages, (s1, s2)):
        assert got.coil.inlet.t_db_c == pytest.approx(want.inlet.t_db_c, abs=1e-12)
        assert got.coil.inlet.w == pytest.approx(want.inlet.w, abs=1e-14)
        assert got.coil.outlet.t_db_c == pytest.approx(want.outlet.t_db_c, abs=1e-12)
        assert got.coil.outlet.w == pytest.approx(want.outlet.w, abs=1e-14)
        assert got.coil.adp.t_db_c == pytest.approx(want.adp.t_db_c, abs=1e-9)
        assert got.coil.bf == pytest.approx(want.bf, abs=1e-12)
        assert got.coil.loads.q_total == pytest.approx(want.loads.q_total, abs=1e-9)
    # 最终出口吻合
    assert res.outlet.t_db_c == pytest.approx(s2.outlet.t_db_c, abs=1e-12)
    assert res.outlet.w == pytest.approx(s2.outlet.w, abs=1e-14)
    # 全链合计 == 手工串首末状态直接分解
    from app.cooling import breakdown_load
    manual = breakdown_load(demo_inlet, s2.outlet, 1.5)
    assert res.totals.q_total == pytest.approx(manual.q_total, abs=1e-9)
    assert res.totals.q_sensible == pytest.approx(manual.q_sensible, abs=1e-9)
    assert res.totals.q_latent == pytest.approx(manual.q_latent, abs=1e-9)


def test_interstage_state_passed_bit_exact(demo_inlet):
    """后一级的进口必须是前一级算出的出口对象本身，逐位一致。"""
    res = chain.solve_chain(demo_inlet, _direct_specs(3))
    for prev, nxt in zip(res.stages, res.stages[1:]):
        assert nxt.coil.inlet is prev.coil.outlet  # 同一对象，未重新展开
        assert nxt.coil.inlet.t_db_c == prev.coil.outlet.t_db_c
        assert nxt.coil.inlet.w == prev.coil.outlet.w
        assert nxt.coil.inlet.enthalpy == prev.coil.outlet.enthalpy
        assert nxt.coil.inlet.p_pa == prev.coil.outlet.p_pa
    # 全链共用同一个进口来风
    assert res.stages[0].coil.inlet is demo_inlet
    assert res.inlet is demo_inlet


# --------------------------------------------------------------------------
# 收口守恒与单调性
# --------------------------------------------------------------------------

def test_chain_total_equals_stage_sum_exactly(demo_inlet):
    for n in (2, 3, 4):
        res = chain.solve_chain(demo_inlet, _direct_specs(n), m_da=2.0)
        sum_q = sum(s.coil.loads.q_total for s in res.stages)
        assert res.totals.q_total == pytest.approx(sum_q, abs=1e-9)
        sum_dehum = sum(
            2.0 * (s.coil.inlet.w - s.coil.outlet.w) for s in res.stages
        )
        assert res.dehumidification == pytest.approx(sum_dehum, abs=1e-12)
        # 总去湿量 == m_da·(首级进口 W − 末级出口 W)
        assert res.dehumidification == pytest.approx(
            2.0 * (demo_inlet.w - res.outlet.w), abs=1e-12
        )
        # 份额归一
        assert sum(s.load_share for s in res.stages) == pytest.approx(1.0, abs=1e-12)
        # 显热 + 潜热 == 总冷量（全链合计同样自洽）
        assert res.totals.q_sensible + res.totals.q_latent == \
            pytest.approx(res.totals.q_total, abs=1e-9)


def test_temperature_and_humidity_monotonic_along_chain(demo_inlet):
    res = chain.solve_chain(demo_inlet, _direct_specs(4))
    ts = [res.inlet.t_db_c] + [s.coil.outlet.t_db_c for s in res.stages]
    ws = [res.inlet.w] + [s.coil.outlet.w for s in res.stages]
    hs = [res.inlet.enthalpy] + [s.coil.outlet.enthalpy for s in res.stages]
    assert all(a >= b for a, b in zip(ts, ts[1:])), "沿链温度不得回升"
    assert all(a >= b for a, b in zip(ws, ws[1:])), "沿链含湿量不得增加"
    assert all(a >= b for a, b in zip(hs, hs[1:])), "沿链焓不得回升"


def test_mixed_modes_four_stage_chain(demo_inlet):
    """四种求解方式混在一条四级链里，结果与全正算参考链一致。"""
    ref = chain.solve_chain(demo_inlet, _direct_specs(4), m_da=1.0)
    mixed_specs = [
        chain.StageSpec(t_adp_c=ADPS_4[0], bf=0.5),                    # 正算
        chain.StageSpec(outlet=ref.stages[1].coil.outlet),             # 完整出风
        chain.StageSpec(                                               # 目标温度+SHR
            target_t_out_c=ref.stages[2].coil.outlet.t_db_c,
            target_shr=ref.stages[2].coil.loads.shr,
        ),
        chain.StageSpec(                                               # BF+目标温度
            bf=0.5, target_t_out_c=ref.stages[3].coil.outlet.t_db_c
        ),
    ]
    res = chain.solve_chain(demo_inlet, mixed_specs)
    assert [s.mode for s in res.stages] == [
        "direct", "from_outlet", "target_t_shr", "bf_target_t"
    ]
    for got, want in zip(res.stages, ref.stages):
        assert got.coil.outlet.t_db_c == pytest.approx(
            want.coil.outlet.t_db_c, abs=1e-7)
        assert got.coil.outlet.w == pytest.approx(want.coil.outlet.w, abs=1e-9)
        assert got.coil.adp.t_db_c == pytest.approx(
            want.coil.adp.t_db_c, abs=1e-6)
        assert got.coil.bf == pytest.approx(want.coil.bf, abs=1e-6)
    assert res.totals.q_total == pytest.approx(ref.totals.q_total, abs=1e-7)


def test_mass_flow_scales_chain_totals(demo_inlet):
    r1 = chain.solve_chain(demo_inlet, _direct_specs(3), m_da=1.0)
    r5 = chain.solve_chain(demo_inlet, _direct_specs(3), m_da=5.0)
    assert r5.totals.q_total == pytest.approx(5.0 * r1.totals.q_total, rel=1e-12)
    assert r5.totals.q_sensible == pytest.approx(5.0 * r1.totals.q_sensible, rel=1e-12)
    assert r5.totals.q_latent == pytest.approx(5.0 * r1.totals.q_latent, rel=1e-12)
    assert r5.totals.shr == pytest.approx(r1.totals.shr, abs=1e-12)
    assert r5.dehumidification == pytest.approx(5.0 * r1.dehumidification, rel=1e-12)


# --------------------------------------------------------------------------
# 透传级与无效工况
# --------------------------------------------------------------------------

def test_passthrough_stage_is_legal(demo_inlet):
    specs = [
        chain.StageSpec(t_adp_c=10.0, bf=0.5),
        chain.StageSpec(t_adp_c=8.0, bf=1.0),   # 纯旁通透传级
        chain.StageSpec(t_adp_c=6.5, bf=0.5),
    ]
    res = chain.solve_chain(demo_inlet, specs)
    assert res.stages[1].passthrough
    assert res.stages[1].coil.loads.q_total == pytest.approx(0.0, abs=1e-12)
    assert res.stages[1].load_share == pytest.approx(0.0, abs=1e-12)
    # 透传级进出口一致
    assert res.stages[1].coil.outlet.t_db_c == \
        pytest.approx(res.stages[1].coil.inlet.t_db_c, abs=1e-12)
    assert not res.stages[0].passthrough
    assert not res.stages[2].passthrough


def test_all_passthrough_chain_rejected(demo_inlet):
    specs = [chain.StageSpec(t_adp_c=10.0, bf=1.0),
             chain.StageSpec(t_adp_c=8.0, bf=1.0)]
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert "没有任何一级" in ei.value.message


def test_empty_chain_rejected(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [])
    assert ei.value.code == "invalid_request"
    assert "空链" in ei.value.message


# --------------------------------------------------------------------------
# 非法链：结构化拒绝并定位到级次
# --------------------------------------------------------------------------

def test_stage_adp_above_wet_bulb_located(demo_inlet):
    """第二级的装置露点高于该级进口湿球：必须定位到第 2 级。"""
    specs = [
        chain.StageSpec(t_adp_c=10.0, bf=0.5),   # 出口湿球 ~19.3 °C
        chain.StageSpec(t_adp_c=19.5, bf=0.5),   # 高于该级进口湿球
    ]
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert ei.value.stage_index == 2
    assert "第 2 级" in ei.value.message


def test_stage_outlet_wetter_than_stage_inlet_located(demo_inlet):
    """某级给定出风比该级进口还湿：定位到该级。"""
    s1 = adp.solve_direct(demo_inlet, 10.0, 0.5)
    wetter = expand_state(s1.outlet.t_db_c, s1.outlet.p_pa,
                          w=s1.outlet.w + 0.002)
    specs = [chain.StageSpec(t_adp_c=10.0, bf=0.5),
             chain.StageSpec(outlet=wetter)]
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert ei.value.stage_index == 2


def test_supersaturated_interstage_state_located(demo_inlet):
    """某级极冷 ADP + 极小 BF 让出口（即下一级进口）落入饱和线外侧：
    必须在该级就拒绝并定位，绝不把超饱和状态往下传。"""
    specs = [
        chain.StageSpec(t_adp_c=4.0, bf=0.1),   # 混合弦穿入雾区
        chain.StageSpec(t_adp_c=8.0, bf=0.5),
    ]
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, specs)
    assert ei.value.code == "not_dehumidifying"
    assert ei.value.stage_index == 1
    assert "超饱和" in ei.value.message


def test_chain_target_wetter_than_inlet_rejected(demo_inlet):
    """链首尾给成出风比进风还湿：非冷却工况，结构化拒绝。"""
    wet_target = expand_state(34.0, 101325.0, w=demo_inlet.w + 0.003)
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(
            demo_inlet, _direct_specs(2),
            target=chain.ChainTarget(outlet=wet_target),
        )
    assert ei.value.code == "not_dehumidifying"
    assert "还湿" in ei.value.message


def test_chain_target_hotter_than_inlet_rejected(demo_inlet):
    hot_target = expand_state(36.0, 101325.0, rh=0.4)
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(
            demo_inlet, _direct_specs(2),
            target=chain.ChainTarget(outlet=hot_target),
        )
    assert ei.value.code == "not_dehumidifying"


def test_pressure_mismatch_between_stages_located(demo_inlet):
    """某级给定出风的气压与链上气压不一致：定位到该级。"""
    other_p = expand_state(15.0, 90000.0, rh=0.9)
    specs = [chain.StageSpec(t_adp_c=10.0, bf=0.5),
             chain.StageSpec(outlet=other_p)]
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, specs)
    assert ei.value.code == "invalid_request"
    assert ei.value.stage_index == 2


def test_chain_target_pressure_mismatch_rejected(demo_inlet):
    target = expand_state(12.0, 90000.0, rh=0.9)
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(
            demo_inlet, _direct_specs(2),
            target=chain.ChainTarget(outlet=target),
        )
    assert ei.value.code == "invalid_request"
    assert "气压" in ei.value.message


def test_invalid_stage_combinations_located(demo_inlet):
    # 正算参数与目标参数同给
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(t_adp_c=10.0, bf=0.5),
            chain.StageSpec(t_adp_c=8.0, bf=0.5, target_shr=0.5),
        ])
    assert ei.value.code == "invalid_request"
    assert ei.value.stage_index == 2
    # BF 越界
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(t_adp_c=10.0, bf=0.5),
            chain.StageSpec(t_adp_c=8.0, bf=1.2),
        ])
    assert ei.value.code == "invalid_bypass_factor"
    assert ei.value.stage_index == 2
    # SHR 越界
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(target_t_out_c=20.0, target_shr=1.5),
        ])
    assert ei.value.code == "invalid_shr"
    assert ei.value.stage_index == 1


# --------------------------------------------------------------------------
# 全链目标反推
# --------------------------------------------------------------------------

def _forward_reference(demo_inlet, n=3):
    return chain.solve_chain(demo_inlet, _direct_specs(n))


def test_inverse_recovers_free_stage(demo_inlet):
    """正算结果作为目标喂回去，自由级的 ADP/BF 必须还原。"""
    ref = _forward_reference(demo_inlet)
    final = ref.outlet
    for free_pos in (0, 1, 2):
        specs = _direct_specs(3)
        specs[free_pos] = chain.StageSpec()  # 自由级
        inv = chain.solve_chain(
            demo_inlet, specs, target=chain.ChainTarget(outlet=final)
        )
        assert inv.mode == "inverse"
        want = ref.stages[free_pos].coil
        got = inv.stages[free_pos].coil
        assert got.adp.t_db_c == pytest.approx(want.adp.t_db_c, abs=1e-5)
        assert got.bf == pytest.approx(want.bf, abs=1e-6)
        # 反推出的整条链落回目标
        assert inv.outlet.t_db_c == pytest.approx(final.t_db_c, abs=1e-6)
        assert inv.outlet.w == pytest.approx(final.w, abs=1e-9)


def test_inverse_recovers_two_half_specified_stages(demo_inlet):
    ref = _forward_reference(demo_inlet)
    final = ref.outlet

    # 两级各只给 BF：反解两级的 ADP
    inv = chain.solve_chain(demo_inlet, [
        chain.StageSpec(bf=0.5),
        chain.StageSpec(bf=0.5),
        chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
    ], target=chain.ChainTarget(outlet=final))
    assert inv.stages[0].coil.adp.t_db_c == pytest.approx(ADPS_4[0], abs=1e-5)
    assert inv.stages[1].coil.adp.t_db_c == pytest.approx(ADPS_4[1], abs=1e-5)

    # 两级各只给 ADP：反解两级的 BF
    inv = chain.solve_chain(demo_inlet, [
        chain.StageSpec(t_adp_c=ADPS_4[0]),
        chain.StageSpec(t_adp_c=ADPS_4[1]),
        chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
    ], target=chain.ChainTarget(outlet=final))
    assert inv.stages[0].coil.bf == pytest.approx(0.5, abs=1e-6)
    assert inv.stages[1].coil.bf == pytest.approx(0.5, abs=1e-6)

    # 混合：一级只给 BF、一级只给 ADP
    inv = chain.solve_chain(demo_inlet, [
        chain.StageSpec(bf=0.5),
        chain.StageSpec(t_adp_c=ADPS_4[1]),
        chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
    ], target=chain.ChainTarget(outlet=final))
    assert inv.stages[0].coil.adp.t_db_c == pytest.approx(ADPS_4[0], abs=1e-5)
    assert inv.stages[1].coil.bf == pytest.approx(0.5, abs=1e-6)


def test_inverse_with_t_shr_chain_target(demo_inlet):
    """全链目标给「最终出风温度 + 目标显热比」同样能还原。"""
    ref = _forward_reference(demo_inlet)
    specs = _direct_specs(3)
    specs[1] = chain.StageSpec()
    inv = chain.solve_chain(
        demo_inlet, specs,
        target=chain.ChainTarget(target_t_out_c=ref.outlet.t_db_c,
                                 target_shr=ref.totals.shr),
    )
    assert inv.mode == "inverse"
    got = inv.stages[1].coil
    assert got.adp.t_db_c == pytest.approx(ADPS_4[1], abs=1e-5)
    assert got.bf == pytest.approx(0.5, abs=1e-6)
    assert inv.totals.shr == pytest.approx(ref.totals.shr, abs=1e-7)


def test_inverse_with_stage_level_t_target(demo_inlet):
    """某级只给本级目标出风温度，与其余未知量一起被整体解出。"""
    ref = _forward_reference(demo_inlet)
    t1 = ref.stages[0].coil.outlet.t_db_c
    inv = chain.solve_chain(demo_inlet, [
        chain.StageSpec(target_t_out_c=t1),   # 本级两个自由度 + 一条方程
        chain.StageSpec(bf=0.5),              # 一个未知量
        chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
    ], target=chain.ChainTarget(outlet=ref.outlet))
    assert inv.stages[0].coil.adp.t_db_c == pytest.approx(ADPS_4[0], abs=1e-5)
    assert inv.stages[0].coil.bf == pytest.approx(0.5, abs=1e-6)
    assert inv.stages[1].coil.adp.t_db_c == pytest.approx(ADPS_4[1], abs=1e-5)
    assert inv.stages[0].coil.outlet.t_db_c == pytest.approx(t1, abs=1e-6)


def test_verify_mode_consistent_target(demo_inlet):
    """各级给全 + 目标一致：按 verify 放行，结果与纯 forward 一致。"""
    ref = _forward_reference(demo_inlet)
    res = chain.solve_chain(
        demo_inlet, _direct_specs(3),
        target=chain.ChainTarget(outlet=ref.outlet),
    )
    assert res.mode == "verify"
    assert res.totals.q_total == pytest.approx(ref.totals.q_total, abs=1e-9)


def test_verify_mode_contradictory_target_rejected(demo_inlet):
    """各级给全但目标互相矛盾：过定，拒绝。"""
    other = expand_state(11.5, 101325.0, rh=0.9)
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(
            demo_inlet, _direct_specs(3),
            target=chain.ChainTarget(outlet=other),
        )
    assert ei.value.code == "chain_overdetermined"


def test_inverse_underdetermined_rejected(demo_inlet):
    ref = _forward_reference(demo_inlet)
    # 两个自由级：4 个自由量、2 个方程
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(t_adp_c=ADPS_4[0], bf=0.5),
            chain.StageSpec(),
            chain.StageSpec(),
        ], target=chain.ChainTarget(outlet=ref.outlet))
    assert ei.value.code == "chain_underdetermined"
    assert ei.value.stage_index == 2
    # 只给一个目标温度、其余什么都不给：同样欠定
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [chain.StageSpec(), chain.StageSpec()],
                          target=chain.ChainTarget(outlet=ref.outlet))
    assert ei.value.code == "chain_underdetermined"


def test_partial_stage_without_chain_target_rejected(demo_inlet):
    """有未给足的级却不给全链目标：欠定。"""
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(t_adp_c=10.0, bf=0.5),
            chain.StageSpec(bf=0.5),
        ])
    assert ei.value.code == "chain_underdetermined"
    assert ei.value.stage_index == 2


def test_inverse_overdetermined_rejected(demo_inlet):
    ref = _forward_reference(demo_inlet)
    # 只剩 1 个自由量（BF 给定、ADP 未知），目标给 2 个方程：过定
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(bf=0.5),
            chain.StageSpec(t_adp_c=ADPS_4[1], bf=0.5),
            chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
        ], target=chain.ChainTarget(outlet=ref.outlet))
    assert ei.value.code == "chain_overdetermined"


def test_inverse_no_convergence_located(demo_inlet):
    """物理不可达的目标：报不收敛、指明卡在第几级，绝不返回临时值。"""
    unreachable = expand_state(2.0, 101325.0, w=0.002)  # 合法但够不着
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, [
            chain.StageSpec(t_adp_c=ADPS_4[0], bf=0.5),
            chain.StageSpec(),
            chain.StageSpec(t_adp_c=ADPS_4[2], bf=0.5),
        ], target=chain.ChainTarget(outlet=unreachable))
    assert ei.value.code == "no_convergence"
    assert ei.value.stage_index == 2
    assert "第 2 级" in ei.value.message


def test_chain_target_needs_both_t_and_shr(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, _direct_specs(2),
                          target=chain.ChainTarget(target_t_out_c=12.0))
    assert ei.value.code == "invalid_request"
    with pytest.raises(PsychrometricError) as ei:
        chain.solve_chain(demo_inlet, _direct_specs(2),
                          target=chain.ChainTarget(target_shr=0.6))
    assert ei.value.code == "invalid_request"


def test_chain_inverse_result_also_satisfies_conservation(demo_inlet):
    """反推出来的链同样要过收口守恒校验。"""
    ref = _forward_reference(demo_inlet)
    specs = _direct_specs(3)
    specs[1] = chain.StageSpec()
    inv = chain.solve_chain(
        demo_inlet, specs, target=chain.ChainTarget(outlet=ref.outlet)
    )
    sum_q = sum(s.coil.loads.q_total for s in inv.stages)
    assert inv.totals.q_total == pytest.approx(sum_q, abs=1e-9)
    ts = [inv.inlet.t_db_c] + [s.coil.outlet.t_db_c for s in inv.stages]
    assert all(a >= b for a, b in zip(ts, ts[1:]))


def test_request_independence_between_chains(demo_inlet):
    """连续两条不同的链互不串台。"""
    r1 = chain.solve_chain(demo_inlet, _direct_specs(2))
    r2 = chain.solve_chain(demo_inlet, _direct_specs(4))
    assert len(r1.stages) == 2 and len(r2.stages) == 4
    assert r1.outlet.t_db_c != pytest.approx(r2.outlet.t_db_c, abs=0.5)
    r1_again = chain.solve_chain(demo_inlet, _direct_specs(2))
    assert r1_again.outlet.t_db_c == pytest.approx(r1.outlet.t_db_c, abs=1e-12)
