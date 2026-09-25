"""湿空气基础公式与状态展开测试。

钉住：
* Magnus 饱和蒸汽压在 0/100 °C 等锚点上的数值；
* RH / 露点 / 含湿量三种表示互换自洽；
* 超饱和（反算 RH>1）必须被结构化拒绝；
* 同一状态用 RH 与露点两种入口，展开出的含湿量一致。
"""

import math

import pytest

from app import psychrometrics as psy
from app.errors import PsychrometricError
from app.states import expand_state


# 参考值取自 ASHRAE 饱和蒸汽压表（Pa），允许 0.5% 偏差
PWS_ANCHORS = {
    0.0: 611.2,
    7.0: 1001.3,
    10.0: 1228.1,
    20.0: 2339.2,
    26.07: 3370.0,
    35.0: 5628.0,
    100.0: 101417.0,
}


@pytest.mark.parametrize("t,expected", list(PWS_ANCHORS.items()))
def test_saturation_pressure_anchors(t, expected):
    pws = psy.saturation_pressure_water(t)
    assert pws == pytest.approx(expected, rel=5e-3)


def test_saturation_pressure_monotonic():
    ts = [-20.0, -10.0, 0.0, 10.0, 25.0, 50.0, 80.0]
    vals = [psy.saturation_pressure_water(t) for t in ts]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_rh_to_w_and_back(demo_inlet):
    rh = psy.relative_humidity_from_w(demo_inlet.w, demo_inlet.t_db_c,
                                      demo_inlet.p_pa)
    assert rh == pytest.approx(0.5, abs=1e-12)


def test_dewpoint_roundtrip(demo_inlet):
    tdp = demo_inlet.dewpoint_c
    w_back = psy.humidity_ratio_from_dewpoint(tdp, demo_inlet.p_pa)
    assert w_back == pytest.approx(demo_inlet.w, rel=1e-9)
    # 35°C/50%RH 的露点约 23.0 °C
    assert tdp == pytest.approx(23.02, abs=0.05)


def test_wet_bulb_value(demo_inlet):
    # 35°C/50%RH 的热力学湿球约 26.1 °C
    assert demo_inlet.wet_bulb_c == pytest.approx(26.07, abs=0.05)


def test_saturated_state_rh_one():
    st = expand_state(7.0, 101325.0, rh=1.0)
    assert st.relative_humidity == pytest.approx(1.0, abs=1e-12)
    # 露点 == 干球
    assert st.dewpoint_c == pytest.approx(7.0, abs=1e-7)


def test_expand_rh_dewpoint_equivalence(demo_inlet):
    """同一进口分别用 RH 与露点表示，含湿量在容差内一致。"""
    via_dp = expand_state(35.0, 101325.0, t_dp_c=demo_inlet.dewpoint_c)
    via_w = expand_state(35.0, 101325.0, w=demo_inlet.w)
    assert via_dp.w == pytest.approx(demo_inlet.w, abs=1e-12)
    assert via_w.w == pytest.approx(demo_inlet.w, abs=1e-12)
    assert via_dp.enthalpy == pytest.approx(demo_inlet.enthalpy, abs=1e-9)


def test_expand_supersaturated_rejected():
    # 30 °C、常压下 W_s 约 0.0272 kg/kg；0.05 明显超饱和
    with pytest.raises(PsychrometricError) as ei:
        expand_state(30.0, 101325.0, w=0.05)
    assert ei.value.code == "invalid_state"


def test_expand_rh_above_one_rejected():
    with pytest.raises(PsychrometricError) as ei:
        expand_state(30.0, 101325.0, rh=1.1)
    assert ei.value.code == "invalid_state"


def test_expand_rh_negative_rejected():
    with pytest.raises(PsychrometricError):
        expand_state(30.0, 101325.0, rh=-0.1)


def test_exactly_one_humidity_spec_required():
    with pytest.raises(PsychrometricError) as ei:
        expand_state(30.0, 101325.0, rh=0.5, w=0.01)
    assert ei.value.code == "invalid_state"
    with pytest.raises(PsychrometricError):
        expand_state(30.0, 101325.0)


def test_dewpoint_above_drybulb_rejected():
    with pytest.raises(PsychrometricError):
        expand_state(20.0, 101325.0, t_dp_c=25.0)


def test_pressure_and_temperature_bounds():
    with pytest.raises(PsychrometricError):
        psy.p_ws(200.0)
    with pytest.raises(PsychrometricError):
        psy.humidity_ratio_from_vapor_pressure(2000.0, 1500.0)
    with pytest.raises(PsychrometricError):
        expand_state(30.0, 5_000.0, rh=0.5)


def test_enthalpy_formula_anchors():
    # 干空气（W=0）焓 = 1.006*t
    assert psy.enthalpy(30.0, 0.0) == pytest.approx(30.18, abs=1e-9)
    # 0 °C 饱和附近焓主要是潜热项
    h = psy.enthalpy(0.0, 0.00378)
    assert h == pytest.approx(0.00378 * 2501.0, rel=1e-9)


def test_temperature_from_enthalpy_inverse():
    t, w = 28.4, 0.0112
    h = psy.enthalpy(t, w)
    assert psy.temperature_from_enthalpy(h, w) == pytest.approx(t, abs=1e-12)
