"""湿空气基础公式 —— 全项目唯一的公式出处。

所有模块（状态展开、ADP 求解、冷量分解）都只从这里取饱和蒸汽压、
含湿量、焓等基本量，禁止在别处另抄一份。

约定（SI 单位）
---------------
* 温度 t: 摄氏度 °C（变量名统一带 ``_c`` 后缀），绝对温度仅在内部需要时换算
* 压力 P: Pa（大气压、水汽压、饱和水汽压均同）
* 含湿量 W: kg(水) / kg(干空气)
* 焓 h:     kJ / kg(干空气)

饱和水汽压
----------
采用 ASHRAE 手册（Fundamentals）给出的 Magnus 型公式，**系数固定写死**，
全实现不允许中途换公式：

    ln(p_ws) = c1 / T + c2 + c3*T + c4*T² + c5*T³ + c6*ln(T)

其中 T = t + 273.15 K，p_ws 单位为 Pa。水/冰各有一组系数；本服务的
盘管表面温度高于 0 °C，统一使用水面系数（含 0 °C 以下极小范围的回退
也走同一组，避免分界处跳变）。
"""

from __future__ import annotations

import math

from .errors import PsychrometricError, ERR_INVALID_STATE

# ---- 物理常数 -------------------------------------------------------------

#: 干空气定压比热，kJ/(kg干空气·K)
CP_AIR = 1.006
#: 水蒸气定压比热，kJ/(kg水蒸气·K)
CP_VAPOR = 1.86
#: 0 °C 液态水汽化潜热，kJ/kg
H_FG_0C = 2501.0
#: 水分子量 / 干空气分子量比
MW_RATIO = 0.621945

#: 允许的温度范围（°C），超出即视为非法输入
T_MIN_C = -60.0
T_MAX_C = 120.0
#: 允许的气压范围（Pa）
P_MIN = 20_000.0
P_MAX = 200_000.0

#: RH=1 的浮点容差：W 反算 RH 略超 1（<= 该值）视为饱和
RH_TOL = 1e-9

# ---- Magnus 公式系数（ASHRAE Fundamentals，水面，-40..120 °C）-----------
# 固定写进实现，禁止在其它模块替换。
_C1 = -5.800_220_6e3
_C2 = 1.391_499_3
_C3 = -4.864_023_9e-2
_C4 = 4.176_476_8e-5
_C5 = -1.445_209_3e-8
_C6 = 6.545_967_3


def saturation_pressure_water(t_c: float) -> float:
    """水面饱和水汽压 p_ws [Pa]，Magnus 公式（系数见模块文档）。"""
    if not (T_MIN_C <= t_c <= T_MAX_C):
        raise PsychrometricError(
            f"温度 {t_c:.3f} °C 超出允许范围 [{T_MIN_C}, {T_MAX_C}] °C",
            ERR_INVALID_STATE,
        )
    T = t_c + 273.15
    ln_pws = _C1 / T + _C2 + _C3 * T + _C4 * T**2 + _C5 * T**3 + _C6 * math.log(T)
    return math.exp(ln_pws)


# 短别名，内部模块统一用这个
p_ws = saturation_pressure_water


def humidity_ratio_from_vapor_pressure(p_v: float, p: float) -> float:
    """由水汽分压与大气压求含湿量 W = 0.621945·p_v / (P − p_v)。"""
    if p_v < -1e-9:
        raise PsychrometricError(f"水汽分压为负: {p_v!r} Pa", ERR_INVALID_STATE)
    if p_v >= p:
        raise PsychrometricError(
            f"水汽分压 {p_v:.1f} Pa 不小于大气压 {p:.1f} Pa，状态不成立",
            ERR_INVALID_STATE,
        )
    if not (P_MIN <= p <= P_MAX):
        raise PsychrometricError(
            f"大气压 {p:.0f} Pa 超出允许范围 [{P_MIN:.0f}, {P_MAX:.0f}] Pa",
            ERR_INVALID_STATE,
        )
    return MW_RATIO * max(p_v, 0.0) / (p - max(p_v, 0.0))


def humidity_ratio_sat(t_c: float, p: float) -> float:
    """饱和含湿量 W_s(t, P)。"""
    return humidity_ratio_from_vapor_pressure(saturation_pressure_water(t_c), p)


def relative_humidity_from_w(w: float, t_c: float, p: float) -> float:
    """由含湿量反算相对湿度 RH = p_v / p_ws（0~1）。"""
    p_v = vapor_pressure_from_humidity_ratio(w, p)
    pws = saturation_pressure_water(t_c)
    rh = p_v / pws
    if rh > 1.0 + RH_TOL:
        raise PsychrometricError(
            f"含湿量 {w:.6f} kg/kg 超过 {t_c:.2f} °C、{p:.0f} Pa 下的饱和上限 "
            f"（RH={rh:.4f} > 1）",
            ERR_INVALID_STATE,
        )
    return min(rh, 1.0)


def vapor_pressure_from_humidity_ratio(w: float, p: float) -> float:
    """含湿量反解水汽分压 p_v = W·P / (0.621945 + W)。"""
    if w < 0.0:
        raise PsychrometricError(f"含湿量为负: {w!r}", ERR_INVALID_STATE)
    return w * p / (MW_RATIO + w)


def humidity_ratio_from_rh(rh: float, t_c: float, p: float) -> float:
    """RH + 干球温度 → 含湿量。"""
    if not (0.0 <= rh <= 1.0 + RH_TOL):
        raise PsychrometricError(
            f"相对湿度 {rh!r} 不在 [0, 1]", ERR_INVALID_STATE
        )
    return humidity_ratio_from_vapor_pressure(min(rh, 1.0) * p_ws(t_c), p)


def humidity_ratio_from_dewpoint(t_dp_c: float, p: float) -> float:
    """露点温度 → 含湿量（露点定义：等压冷却到该温度恰好饱和，故取 W_s）。"""
    return humidity_ratio_sat(t_dp_c, p)


def dewpoint_from_humidity_ratio(w: float, p: float) -> float:
    """含湿量 → 露点温度 [°C]。

    露点处 p_ws(t_dp) = p_v，对饱和水汽压求反。用二分法解
    p_ws(t) − p_v = 0（p_ws 关于 t 单调）。
    """
    p_v = vapor_pressure_from_humidity_ratio(w, p)
    if p_v <= 0.0:
        return T_MIN_C

    def f(t: float) -> float:
        return p_ws(t) - p_v

    lo, hi = T_MIN_C, T_MAX_C
    if f(lo) > 0.0:
        return T_MIN_C
    # p_v 必然小于 p，而 p_ws(T_MAX) 远大于常规大气压，上界必然满足
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def enthalpy(t_c: float, w: float) -> float:
    """湿空气比焓 h = 1.006·t + W·(2501 + 1.86·t)，kJ/kg干空气。"""
    return CP_AIR * t_c + w * (H_FG_0C + CP_VAPOR * t_c)


def temperature_from_enthalpy(h: float, w: float) -> float:
    """焓与含湿量反解干球温度 t = (h − 2501·W) / (1.006 + 1.86·W)。"""
    return (h - H_FG_0C * w) / (CP_AIR + CP_VAPOR * w)


def wet_bulb_temperature(t_db_c: float, w: float, p: float) -> float:
    """热力学湿球温度 [°C]（ASHRAE 等焓饱和方程求根）。

    对任一试验温度 t* 定义（沿等焓线到饱和）：

        W_s(t*) = [(2501 − 2.326·t*)·W + 1.006·(t_db − t*)]
                  / [2501 + 1.86·t* − 4.186·t*]

    饱和方程的残差取 f(t*) = W_s(t*) − 上面的右端：
      * f(t_db) = W_s(t_db) − W >= 0（未饱和时严格为正）
      * f(t_dp) <= 0
    二分求根即得热力学湿球温度。
    """
    # 饱和状态：湿球 == 干球
    if relative_humidity_from_w_safe(w, t_db_c, p) >= 1.0:
        return t_db_c

    def f(t_star: float) -> float:
        w_s = humidity_ratio_sat(t_star, p)
        rhs = (
            (H_FG_0C - 2.326 * t_star) * w + CP_AIR * (t_db_c - t_star)
        ) / (H_FG_0C + CP_VAPOR * t_star - 4.186 * t_star)
        return w_s - rhs

    t_dp = dewpoint_from_humidity_ratio(w, p)
    lo, hi = max(T_MIN_C, t_dp), t_db_c
    flo = f(lo)
    if flo >= 0.0:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def relative_humidity_from_w_safe(w: float, t_c: float, p: float) -> float:
    """同 :func:`relative_humidity_from_w`，但超饱和时返回 >1 而不抛错。

    仅供内部（如湿球判定饱和）使用。
    """
    return vapor_pressure_from_humidity_ratio(w, p) / p_ws(t_c)
