"""盘管（ADP/旁通加权/冷量分解）测试。

钉住用户点名的关系：
* BF=0 出口精确等于 ADP 饱和状态；
* BF=1 出口等于进口、冷量为零；
* 固定 ADP，BF 0.1→0.3 出口 W、h 更靠近进口，冷量下降；
* 显热 + 潜热 == 总冷量（容差内）；
* RH 入口与露点入口展开一致时，下游出口结果不跳变；
* 五类非法输入被结构化拒绝；
* 五种求解模式彼此自洽（反算能还原正算结果）。
"""

import pytest

from app import adp
from app.cooling import breakdown_load
from app.errors import PsychrometricError
from app.states import expand_state, saturated_state

ADP_T = 7.0


# --------------------------------------------------------------------------
# 旁通两端行为
# --------------------------------------------------------------------------

def test_bf_zero_outlet_equals_adp_exactly(demo_inlet):
    res = adp.solve_direct(demo_inlet, ADP_T, 0.0)
    adp_sat = saturated_state(ADP_T, demo_inlet.p_pa)
    assert res.bf == 0.0
    assert res.outlet.t_db_c == pytest.approx(adp_sat.t_db_c, abs=1e-12)
    assert res.outlet.w == pytest.approx(adp_sat.w, abs=1e-12)
    assert res.outlet.enthalpy == pytest.approx(adp_sat.enthalpy, abs=1e-10)
    assert res.outlet.relative_humidity == pytest.approx(1.0, abs=1e-9)
    # 冷量为满负荷（进口到 ADP）
    assert res.loads.q_total == pytest.approx(
        demo_inlet.enthalpy - adp_sat.enthalpy, rel=1e-12
    )


def test_bf_one_outlet_equals_inlet_and_zero_load(demo_inlet):
    res = adp.solve_direct(demo_inlet, ADP_T, 1.0)
    assert res.outlet.t_db_c == pytest.approx(demo_inlet.t_db_c, abs=1e-12)
    assert res.outlet.w == pytest.approx(demo_inlet.w, abs=1e-14)
    assert res.outlet.enthalpy == pytest.approx(demo_inlet.enthalpy, abs=1e-10)
    assert res.loads.q_total == pytest.approx(0.0, abs=1e-12)
    assert res.loads.q_sensible == pytest.approx(0.0, abs=1e-12)
    assert res.loads.q_latent == pytest.approx(0.0, abs=1e-12)
    assert res.loads.shr is None


def test_bf_increase_moves_outlet_to_inlet_and_load_drops(demo_inlet):
    r1 = adp.solve_direct(demo_inlet, ADP_T, 0.1)
    r3 = adp.solve_direct(demo_inlet, ADP_T, 0.3)
    # 出口 W、h 更靠近进口（更大），但仍未越过进口
    assert demo_inlet.w > r3.outlet.w > r1.outlet.w
    assert demo_inlet.enthalpy > r3.outlet.enthalpy > r1.outlet.enthalpy
    # 出口温度也抬升
    assert r3.outlet.t_db_c > r1.outlet.t_db_c
    # 冷量（总/显热/潜热）随 BF 增大而下降
    assert r1.loads.q_total > r3.loads.q_total > 0.0
    assert r1.loads.q_sensible > r3.loads.q_sensible
    assert r1.loads.q_latent > r3.loads.q_latent


# --------------------------------------------------------------------------
# W 与 h 必须用同一个 BF 加权（不许温度单独算术平均）
# --------------------------------------------------------------------------

def test_w_and_h_share_single_bf(demo_inlet):
    res = adp.solve_direct(demo_inlet, ADP_T, 0.2)
    bf_w = (res.outlet.w - res.adp.w) / (demo_inlet.w - res.adp.w)
    bf_h = ((res.outlet.enthalpy - res.adp.enthalpy)
            / (demo_inlet.enthalpy - res.adp.enthalpy))
    assert bf_w == pytest.approx(0.2, abs=1e-10)
    assert bf_h == pytest.approx(0.2, abs=1e-10)
    # 出风温度必须由 (h_out, W_out) 反解，而不是 t 的算术平均
    t_arith = res.adp.t_db_c + 0.2 * (demo_inlet.t_db_c - res.adp.t_db_c)
    t_from_h = res.outlet.t_db_c
    assert t_from_h != pytest.approx(t_arith, abs=1e-6)
    h_check = 1.006 * t_from_h + res.outlet.w * (2501.0 + 1.86 * t_from_h)
    assert h_check == pytest.approx(res.outlet.enthalpy, abs=1e-10)


# --------------------------------------------------------------------------
# 冷量分解自洽
# --------------------------------------------------------------------------

def test_sensible_plus_latent_equals_total(demo_inlet):
    for bf in (0.0, 0.05, 0.2, 0.5, 0.9):
        res = adp.solve_direct(demo_inlet, ADP_T, bf)
        lb = res.loads
        assert lb.q_sensible + lb.q_latent == pytest.approx(lb.q_total, abs=1e-10)
        if bf < 1.0:
            assert 0.0 < lb.shr <= 1.0
            # 显热独立定义核对：cp·Δt（按出口 W 取湿空气比热）
            cp = 1.006 + 1.86 * res.outlet.w
            assert lb.q_sensible == pytest.approx(
                cp * (demo_inlet.t_db_c - res.outlet.t_db_c), rel=1e-12
            )
            # 总冷量独立定义核对：Δh
            assert lb.q_total == pytest.approx(
                demo_inlet.enthalpy - res.outlet.enthalpy, rel=1e-12
            )


def test_mass_flow_scales_loads(demo_inlet):
    r1 = adp.solve_direct(demo_inlet, ADP_T, 0.2, m_da=1.0)
    r5 = adp.solve_direct(demo_inlet, ADP_T, 0.2, m_da=5.0)
    for name in ("q_total", "q_sensible", "q_latent"):
        assert getattr(r5.loads, name) == pytest.approx(
            5.0 * getattr(r1.loads, name), rel=1e-12
        )
    assert r5.loads.shr == pytest.approx(r1.loads.shr, abs=1e-12)


def test_loads_endpoint_rejects_heating_and_pressure_mismatch():
    inlet = expand_state(30.0, 101325.0, rh=0.5)
    hotter = expand_state(32.0, 101325.0, rh=0.5)
    with pytest.raises(PsychrometricError) as ei:
        breakdown_load(inlet, hotter)
    assert ei.value.code == "invalid_shr"
    other_p = expand_state(20.0, 90000.0, rh=0.8)
    with pytest.raises(PsychrometricError):
        breakdown_load(inlet, other_p)
    with pytest.raises(PsychrometricError):
        breakdown_load(inlet, hotter, m_da=0.0)


# --------------------------------------------------------------------------
# 五种求解模式
# --------------------------------------------------------------------------

def test_mode2_recovers_direct_solution(demo_inlet):
    direct = adp.solve_direct(demo_inlet, ADP_T, 0.2)
    back = adp.solve_from_outlet(demo_inlet, direct.outlet)
    assert back.adp.t_db_c == pytest.approx(ADP_T, abs=1e-6)
    assert back.bf == pytest.approx(0.2, abs=1e-6)
    assert back.outlet.t_db_c == pytest.approx(direct.outlet.t_db_c, abs=1e-8)
    assert back.outlet.w == pytest.approx(direct.outlet.w, abs=1e-9)


def test_mode3_target_t_and_shr(demo_inlet):
    # 固定出风温度下 SHR 可行域较窄（过程线须在冻结区以上与饱和曲线相交），
    # 取物理可行的一对
    res = adp.solve_from_target_t_shr(demo_inlet, 20.0, 0.60)
    assert res.outlet.t_db_c == pytest.approx(20.0, abs=1e-6)
    assert res.loads.shr == pytest.approx(0.60, abs=1e-7)
    assert res.outlet.relative_humidity <= 1.0
    assert res.adp.t_db_c < demo_inlet.wet_bulb_c
    assert 0.0 < res.bf < 1.0
    # 反算出来的出口反推一遍 ADP/BF 应自洽
    again = adp.solve_from_outlet(demo_inlet, res.outlet)
    assert again.adp.t_db_c == pytest.approx(res.adp.t_db_c, abs=1e-6)


def test_mode4_bf_and_target_t(demo_inlet):
    res = adp.solve_from_bf_target_t(demo_inlet, 0.25, 14.0)
    assert res.bf == pytest.approx(0.25, abs=1e-12)
    assert res.outlet.t_db_c == pytest.approx(14.0, abs=1e-8)
    assert res.loads.shr is not None and 0.0 < res.loads.shr <= 1.0
    assert res.adp.t_db_c < demo_inlet.dewpoint_c


def test_mode5_bf_and_target_shr(demo_inlet):
    res = adp.solve_from_bf_shr(demo_inlet, 0.25, 0.6)
    assert res.bf == pytest.approx(0.25, abs=1e-12)
    assert res.loads.shr == pytest.approx(0.6, abs=1e-8)
    assert res.adp.t_db_c <= demo_inlet.dewpoint_c + 1e-8


def test_mode5_shr_one_is_dry_coil_endpoint(demo_inlet):
    # SHR=1 极限：ADP 取到进口露点（无潜热）
    res = adp.solve_from_bf_shr(demo_inlet, 0.2, 1.0)
    assert res.adp.t_db_c == pytest.approx(demo_inlet.dewpoint_c, abs=1e-7)
    assert res.loads.q_latent == pytest.approx(0.0, abs=1e-9)
    assert res.loads.shr == pytest.approx(1.0, abs=1e-9)


def test_underdetermined_single_targets_rejected(demo_inlet):
    for kwargs in (
        {"target_t_out_c": 12.0},
        {"target_shr": 0.5},
        {"bf": 0.2},
        {"t_adp_c": 7.0},
    ):
        with pytest.raises(PsychrometricError) as ei:
            adp.solve_coil(demo_inlet, **kwargs)
        assert ei.value.code == "invalid_request"


def test_conflicting_inputs_rejected(demo_inlet):
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_coil(demo_inlet, t_adp_c=7.0, bf=0.2, target_shr=0.5)
    assert ei.value.code == "invalid_request"
    outlet = adp.solve_direct(demo_inlet, ADP_T, 0.2).outlet
    with pytest.raises(PsychrometricError):
        adp.solve_coil(demo_inlet, outlet=outlet, bf=0.2)


def test_mode4_unreachable_target_rejected(demo_inlet):
    # BF 很大时盘管作用微弱，过冷的出风温度需要 ADP 低出物理范围
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_bf_target_t(demo_inlet, 0.95, 2.0)
    assert ei.value.code == "not_dehumidifying"
    # 比“ADP 取到进口露点时的最暖出风”还暖，同样不可达
    warmest = adp._mix(
        demo_inlet, saturated_state(demo_inlet.dewpoint_c, demo_inlet.p_pa), 0.95
    )
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_bf_target_t(demo_inlet, 0.95, warmest.t_db_c + 1.0)
    assert ei.value.code == "not_dehumidifying"


# --------------------------------------------------------------------------
# 边界卡死
# --------------------------------------------------------------------------

def test_bf_out_of_range_rejected(demo_inlet):
    for bad in (-0.01, 1.01):
        with pytest.raises(PsychrometricError) as ei:
            adp.solve_direct(demo_inlet, ADP_T, bad)
        assert ei.value.code == "invalid_bypass_factor"


def test_adp_must_be_below_wet_bulb(demo_inlet):
    twb = demo_inlet.wet_bulb_c
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_direct(demo_inlet, twb + 0.1, 0.2)
    assert ei.value.code == "not_dehumidifying"
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_direct(demo_inlet, twb, 0.2)
    assert ei.value.code == "not_dehumidifying"


def test_adp_must_not_be_wetter_than_inlet(demo_inlet):
    # ADP 高于进口露点 → 饱和含湿量超过进口含湿量
    tdp = demo_inlet.dewpoint_c
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_direct(demo_inlet, tdp + 0.5, 0.2)
    assert ei.value.code == "not_dehumidifying"


def test_shr_out_of_range_rejected(demo_inlet):
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(PsychrometricError):
            adp.solve_from_target_t_shr(demo_inlet, 12.0, bad)


def test_outlet_wetter_than_inlet_rejected(demo_inlet):
    wet = expand_state(34.0, 101325.0, w=0.019)
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_outlet(demo_inlet, wet)
    assert ei.value.code == "not_dehumidifying"


def test_humidifying_outlet_pair_rejected_by_loads(demo_inlet):
    # 等温加湿：W 更高、焓更高
    humidified = expand_state(35.0, 101325.0, w=0.020)
    with pytest.raises(PsychrometricError):
        breakdown_load(demo_inlet, humidified)


def test_fog_zone_supersaturated_mix_rejected(demo_inlet):
    """极冷 ADP + 极小 BF 会让混合弦穿入饱和雾区（RH>1），必须拒绝。"""
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_direct(demo_inlet, 4.0, 0.1)
    assert ei.value.code == "not_dehumidifying"


def test_mode5_too_low_shr_unreachable(demo_inlet):
    # 过低 SHR 需要冻结区以下的 ADP，物理不可达
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_bf_shr(demo_inlet, 0.2, 0.30)
    assert ei.value.code == "not_dehumidifying"


def test_mode3_infeasible_target_rejected(demo_inlet):
    # 目标温度低却给高 SHR：闭式解给出超饱和（或负 W），都必须结构化拒绝
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_target_t_shr(demo_inlet, 13.0, 0.9)
    assert ei.value.code in ("invalid_state", "not_dehumidifying")
    # 小温差 + 极低 SHR：闭式解给出负 W
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_target_t_shr(demo_inlet, 34.0, 0.01)
    assert ei.value.code == "not_dehumidifying"


def test_inconsistent_outlet_pair_rejected(demo_inlet):
    """手造一对不落在同一混合线上的进出口：不存在与饱和曲线的交点。

    取一个真实出口，再把温度抬高、保持 W，制造几何不一致：
    它仍是合法冷却状态，但不属于任何单 ADP + 单 BF 的盘管过程。
    """
    real = adp.solve_direct(demo_inlet, ADP_T, 0.2).outlet
    tampered = expand_state(real.t_db_c + 1.5, real.p_pa, w=real.w)
    with pytest.raises(PsychrometricError) as ei:
        adp.solve_from_outlet(demo_inlet, tampered)
    assert ei.value.code == "inconsistent_state"


# --------------------------------------------------------------------------
# 输入字段互推一致，下游不跳变
# --------------------------------------------------------------------------

def test_rh_vs_dewpoint_inlet_give_identical_coil_result():
    a = expand_state(35.0, 101325.0, rh=0.5)
    b = expand_state(35.0, 101325.0, t_dp_c=a.dewpoint_c)
    ra = adp.solve_direct(a, ADP_T, 0.2)
    rb = adp.solve_direct(b, ADP_T, 0.2)
    assert ra.outlet.t_db_c == pytest.approx(rb.outlet.t_db_c, abs=1e-9)
    assert ra.outlet.w == pytest.approx(rb.outlet.w, abs=1e-11)
    assert ra.outlet.enthalpy == pytest.approx(rb.outlet.enthalpy, abs=1e-9)
    assert ra.loads.q_total == pytest.approx(rb.loads.q_total, abs=1e-9)
    assert ra.adp.t_db_c == pytest.approx(rb.adp.t_db_c, abs=1e-12)


def test_request_independence_no_state_leak(demo_inlet):
    """连续两次不同请求，结果互不串台（不许复用上一次的中间值）。"""
    r1 = adp.solve_direct(demo_inlet, 5.0, 0.1)
    r2 = adp.solve_direct(demo_inlet, 10.0, 0.4)
    assert r1.adp.t_db_c == 5.0 and r2.adp.t_db_c == 10.0
    assert r1.bf == 0.1 and r2.bf == 0.4
    assert abs(r1.outlet.t_db_c - r2.outlet.t_db_c) > 1.0
