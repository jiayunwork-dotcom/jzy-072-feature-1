"""装置露点（ADP）求解与旁通加权模块。

物理模型
========
盘管表面存在一点温度 ``T_adp``，该点空气恰好饱和（RH=1）。出风由两部分
按旁通系数 BF 混合而成，**含湿量与焓必须用同一个 BF 线性加权**：

    W_out = W_adp + BF·(W_in − W_adp)
    h_out = h_adp + BF·(h_in − h_adp)

出风干球温度由加权后的 (h_out, W_out) 反解，绝不允许单独做算术平均。

支持的调用组合
==============
1. 正算：给定 ADP 温度 + BF；
2. 给完整出风状态（t_db + 一种湿度表示），反解 ADP 与 BF；
3. 给目标出风干球温度 + 目标显热比 SHR；
4. 给 BF + 目标出风干球温度；
5. 给 BF + 目标显热比 SHR。

只给「目标出风温度」或只给「目标 SHR」时，两个混合方程里有 ADP 与 BF
两个未知量，欠定，本服务返回结构化错误（条件不足），不瞎猜解。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import psychrometrics as psy
from .cooling import LoadBreakdown, breakdown_load
from .errors import (
    PsychrometricError,
    ERR_INVALID_REQUEST,
    ERR_INVALID_BF,
    ERR_NOT_DEHUMIDIFYING,
    ERR_INCONSISTENT_STATE,
    ERR_NO_CONVERGENCE,
)
from .numerics import bisection
from .states import AirState, saturated_state

#: ADP 必须严格低于进口湿球的容差（°C）
WB_EPS = 1e-6
#: ADP 含湿量不得超过进口含湿量的容差（kg/kg）
W_EPS = 1e-9
#: 完整出风反解时，由 W 与由 h 推出的 BF 的最大允许偏差
BF_AGREE_TOL = 2e-3
#: 反算括号向下扩展的步长（°C）
BRACKET_STEP = 0.5
#: 温度求解容差（°C）
T_TOL = 1e-8


@dataclass(frozen=True)
class CoilResult:
    """盘管核算结果。"""

    inlet: AirState
    outlet: AirState
    adp: AirState
    bf: float
    loads: LoadBreakdown
    mode: str
    """实际采用的求解模式名（出参/调试用）"""


# --------------------------------------------------------------------------
# 基本工具
# --------------------------------------------------------------------------

def _mix(inlet: AirState, adp: AirState, bf: float) -> AirState:
    """用同一个 BF 对 W、h 加权，出风温度由 (h_out, W_out) 反解。"""
    w_out = adp.w + bf * (inlet.w - adp.w)
    h_out = adp.enthalpy + bf * (inlet.enthalpy - adp.enthalpy)
    t_out = psy.temperature_from_enthalpy(h_out, w_out)
    return AirState(t_db_c=t_out, w=w_out, p_pa=inlet.p_pa)


def _validate_bf(bf: float) -> None:
    if not (0.0 <= bf <= 1.0):
        raise PsychrometricError(
            f"旁通系数 BF={bf!r} 不在 [0, 1]", ERR_INVALID_BF
        )


def _validate_process(inlet: AirState, adp: AirState) -> None:
    """边界卡死：ADP 必须低于进口湿球，且 ADP 含湿量不超过进口含湿量。"""
    t_wb = inlet.wet_bulb_c
    if adp.t_db_c >= t_wb - WB_EPS:
        raise PsychrometricError(
            f"装置露点 {adp.t_db_c:.3f} °C 不低于进口湿球温度 {t_wb:.3f} °C，"
            "这不是冷却去湿工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    if adp.w > inlet.w + W_EPS:
        raise PsychrometricError(
            f"装置露点含湿量 {adp.w * 1000:.3f} g/kg 超过进口含湿量 "
            f"{inlet.w * 1000:.3f} g/kg（出风将比进口更湿），拒绝该工况",
            ERR_NOT_DEHUMIDIFYING,
        )


def _extend_down(f, hi: float, lo_min: float = psy.T_MIN_C + 1.0,
                 step: float = BRACKET_STEP, *,
                 miss_code: str = ERR_NO_CONVERGENCE,
                 miss_message: str | None = None) -> tuple[float, float]:
    """从 hi 起以固定步长向下扫描，找到第一个残差号变区间。

    找不到（到 lo_min 仍同号）抛领域错误。对于“目标物理不可达”类场景，
    调用方可传 ``miss_code`` 改写错误类别；绝不拿扫描中途的中间值充当结果。
    """
    x, fx = hi, f(hi)
    if fx == 0.0:
        return hi, hi
    while x > lo_min:
        x2 = max(lo_min, x - step)
        f2 = f(x2)
        if f2 == 0.0:
            return x2, x2
        if fx * f2 < 0.0:
            return x2, x
        x, fx = x2, f2
    raise PsychrometricError(
        miss_message
        or "ADP 迭代不收敛：向下扩展搜索区间至 "
           f"{lo_min:.1f} °C 仍未出现残差号变",
        miss_code,
    )


def _finish(inlet: AirState, adp: AirState, bf: float, mode: str,
            m_da: float, *, allow_zero_load: bool) -> CoilResult:
    """统一的加权出风 + 边界校验 + 冷量分解收口。"""
    _validate_process(inlet, adp)
    outlet = _mix(inlet, adp, bf)

    # 旁通混合弦可能在“极冷 ADP + 极小 BF”时穿入饱和曲线雾区（RH>1），
    # 这不是一个物理的出风状态，按超饱和结构化拒绝。
    try:
        psy.relative_humidity_from_w(outlet.w, outlet.t_db_c, outlet.p_pa)
    except PsychrometricError as exc:
        raise PsychrometricError(
            f"加权得到的出风状态超饱和（{exc.message}）；该 ADP/BF 组合会在"
            "气流中产生雾化，超出旁通模型适用范围",
            ERR_NOT_DEHUMIDIFYING,
        ) from exc

    loads = breakdown_load(inlet, outlet, m_da)
    if not allow_zero_load and loads.q_total <= 1e-8:
        raise PsychrometricError(
            "按给定条件算出的总冷量为零，没有实际冷却去湿，拒绝该工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    return CoilResult(inlet=inlet, outlet=outlet, adp=adp, bf=bf,
                      loads=loads, mode=mode)


# --------------------------------------------------------------------------
# 模式 1：正算（ADP + BF 已知）
# --------------------------------------------------------------------------

def solve_direct(inlet: AirState, t_adp_c: float, bf: float,
                 m_da: float = 1.0) -> CoilResult:
    _validate_bf(bf)
    adp = saturated_state(t_adp_c, inlet.p_pa)
    # BF=1 时出口==进口、冷量为零，这是合法端点（测试要求），予以放行
    return _finish(inlet, adp, bf, "direct", m_da, allow_zero_load=True)


# --------------------------------------------------------------------------
# 模式 2：完整出风状态 → ADP、BF
# --------------------------------------------------------------------------

def solve_from_outlet(inlet: AirState, outlet: AirState,
                      m_da: float = 1.0) -> CoilResult:
    if inlet.p_pa != outlet.p_pa:
        raise PsychrometricError(
            "进出口气压不一致，无法联立", ERR_INVALID_REQUEST
        )
    if outlet.enthalpy >= inlet.enthalpy - 1e-9:
        raise PsychrometricError(
            "给定出风焓不低于进口焓，不是冷却工况", ERR_NOT_DEHUMIDIFYING
        )
    if outlet.w > inlet.w + W_EPS:
        raise PsychrometricError(
            "给定出风含湿量高于进口（出风比进口还湿），拒绝该工况",
            ERR_NOT_DEHUMIDIFYING,
        )

    dt = inlet.t_db_c - outlet.t_db_c
    dh = inlet.enthalpy - outlet.enthalpy
    dw = inlet.w - outlet.w
    if dt <= 0.0:
        raise PsychrometricError(
            "给定出风干球温度不低于进风，不构成冷却", ERR_NOT_DEHUMIDIFYING
        )
    if dw <= 0.0:
        raise PsychrometricError(
            "给定出风含湿量不低于进风，不构成冷却去湿",
            ERR_NOT_DEHUMIDIFYING,
        )

    # 盘管过程线是进出口状态点在 (h, W) 平面上的直线（混合守恒，h 与 W
    # 对同一个混合比线性；注意 h 含 W·t 交叉项，不能在 (t, h) 平面拉直线）。
    # ADP 即该直线向冷端延伸与饱和曲线的交点：
    #   W_line(h) = W_out + (h − h_out)·ΔW/Δh，
    # 残差 g(t) = W_line(h_sat(t)) − W_sat(t)。
    slope = dw / dh

    def g(t: float) -> float:
        w_sat = psy.humidity_ratio_sat(t, inlet.p_pa)
        h_sat = psy.enthalpy(t, w_sat)
        return outlet.w + (h_sat - outlet.enthalpy) * slope - w_sat

    g_hi = g(outlet.t_db_c)
    if g_hi > 1e-9:
        # 出风状态已在饱和曲线之外（超饱和，正常展开入口应已拦截）
        raise PsychrometricError(
            "给定出风状态位于饱和曲线之外（RH>1）", ERR_NOT_DEHUMIDIFYING
        )
    if g_hi == 0.0:
        t_adp = outlet.t_db_c
    else:
        lo, hi = _extend_down(
            g, outlet.t_db_c,
            miss_code=ERR_INCONSISTENT_STATE,
            miss_message=(
                "过程线向低温端延伸至搜索下限仍不与饱和曲线相交，"
                "给定进出口状态不可能由同一 ADP 的旁通混合产生"
            ),
        )
        t_adp = bisection(g, lo, hi, tol=T_TOL, what="ADP（完整出风反算）")

    adp = saturated_state(t_adp, inlet.p_pa)
    _validate_process(inlet, adp)

    if inlet.w - adp.w <= 0.0:
        raise PsychrometricError(
            "过程线与饱和曲线的交点不含去湿段，无法反算旁通系数",
            ERR_NOT_DEHUMIDIFYING,
        )

    # 同一个 BF 必须同时满足 W 加权与 h 加权。交点按 (h,W) 混合线构造，
    # 两者代数上恒等，这里做一次浮点自洽把关。
    bf_w = (outlet.w - adp.w) / (inlet.w - adp.w)
    bf_h = (outlet.enthalpy - adp.enthalpy) / (inlet.enthalpy - adp.enthalpy)
    if not (0.0 <= bf_w <= 1.0) or abs(bf_w - bf_h) > BF_AGREE_TOL:
        raise PsychrometricError(
            f"进出口状态不自洽：由含湿量加权得 BF={bf_w:.5f}，"
            f"由焓加权得 BF={bf_h:.5f}，不存在单一旁通系数",
            ERR_INCONSISTENT_STATE,
        )
    bf = bf_w
    return _finish(inlet, adp, bf, "from_outlet", m_da, allow_zero_load=False)


# --------------------------------------------------------------------------
# 模式 3：目标出风干球温度 + 目标 SHR
# --------------------------------------------------------------------------

def solve_from_target_t_shr(inlet: AirState, t_out_c: float, shr: float,
                            m_da: float = 1.0) -> CoilResult:
    if not (0.0 < shr <= 1.0):
        raise PsychrometricError(
            f"目标显热比 SHR={shr!r} 不在 (0, 1]", "invalid_shr"
        )
    d_t = inlet.t_db_c - t_out_c
    if d_t <= 0.0:
        raise PsychrometricError(
            "目标出风温度不低于进风温度，不构成冷却",
            ERR_NOT_DEHUMIDIFYING,
        )

    # 由 SHR 定义显式解出口含湿量（cp 取出口湿空气比热）：
    #   SHR·(h_in − h_out) = (cp_a + cp_v·W_out)·Δt,   Δt = t_in − t_out
    # 代入 h_out = cp_a·t_out + W_out·(h_fg + cp_v·t_out)，把含 W 项
    # 移到同一边整理得
    #   W_out·[cp_v·Δt + SHR·(h_fg + cp_v·t_out)]
    #       = SHR·(h_in − cp_a·t_out) − cp_a·Δt
    denom = psy.CP_VAPOR * d_t + shr * (psy.H_FG_0C + psy.CP_VAPOR * t_out_c)
    w_out = (shr * (inlet.enthalpy - psy.CP_AIR * t_out_c)
             - psy.CP_AIR * d_t) / denom
    if w_out < -W_EPS:
        raise PsychrometricError(
            "按给定目标温度与 SHR 解出的出口含湿量为负，目标不可达",
            ERR_NOT_DEHUMIDIFYING,
        )
    w_out = max(w_out, 0.0)
    if w_out >= inlet.w - W_EPS and shr < 1.0:
        raise PsychrometricError(
            "按给定目标解出去湿量为零/为负（出口不比进口干），"
            "该目标温度与 SHR 组合不构成冷却去湿工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    # 构造目标出风状态前必须确认它在饱和曲线内侧
    psy.relative_humidity_from_w(w_out, t_out_c, inlet.p_pa)
    outlet = AirState(t_db_c=t_out_c, w=w_out, p_pa=inlet.p_pa)
    return solve_from_outlet(inlet, outlet, m_da)


# --------------------------------------------------------------------------
# 模式 4：BF + 目标出风干球温度
# --------------------------------------------------------------------------

def solve_from_bf_target_t(inlet: AirState, bf: float, t_out_c: float,
                           m_da: float = 1.0) -> CoilResult:
    _validate_bf(bf)
    if bf >= 1.0:
        raise PsychrometricError(
            "BF=1 时风全部旁通，出风恒等于进风，无法满足低于进风的"
            "目标出风温度",
            ERR_INVALID_REQUEST,
        )
    if t_out_c >= inlet.t_db_c:
        raise PsychrometricError(
            "目标出风温度不低于进风温度，不构成冷却",
            ERR_NOT_DEHUMIDIFYING,
        )

    def f(t_adp: float) -> float:
        adp = saturated_state(t_adp, inlet.p_pa)
        out = _mix(inlet, adp, bf)
        return out.t_db_c - t_out_c

    # t_adp 上界取进口露点（再高就没有去湿作用了）；此处给出最“暖”的出风。
    # 出风温度随 ADP 单调升高：最暖出风都比目标冷，才是真的够不着目标；
    # 若最暖出风已达到/超过目标，根在 [T_MIN, t_dp] 之间。
    t_hi = inlet.dewpoint_c
    if f(t_hi) < 0.0:
        raise PsychrometricError(
            f"BF={bf:.3f} 下即使盘管表面取到进口露点 {t_hi:.2f} °C，"
            f"出风温度仍低于目标 {t_out_c:.2f} °C，目标不可达",
            ERR_NOT_DEHUMIDIFYING,
        )
    if f(t_hi) == 0.0:
        t_adp = t_hi
    else:
        lo, hi = _extend_down(
            f, t_hi, miss_code=ERR_NOT_DEHUMIDIFYING,
            miss_message=(
                f"BF={bf:.3f} 下搜索到温度下限仍无法把出风拉到 "
                f"{t_out_c:.2f} °C，目标物理不可达（需 ADP 低于冻结区）"
            ),
        )
        t_adp = bisection(f, lo, hi, tol=T_TOL, what="ADP（BF+目标出风温度）")
    adp = saturated_state(t_adp, inlet.p_pa)
    return _finish(inlet, adp, bf, "bf_target_t", m_da, allow_zero_load=False)


# --------------------------------------------------------------------------
# 模式 5：BF + 目标 SHR
# --------------------------------------------------------------------------

def solve_from_bf_shr(inlet: AirState, bf: float, shr: float,
                      m_da: float = 1.0) -> CoilResult:
    _validate_bf(bf)
    if not (0.0 < shr <= 1.0):
        raise PsychrometricError(
            f"目标显热比 SHR={shr!r} 不在 (0, 1]", "invalid_shr"
        )
    if bf >= 1.0:
        raise PsychrometricError(
            "BF=1 时风全部旁通，出风恒等于进风，无法满足冷却目标",
            ERR_INVALID_REQUEST,
        )

    def f(t_adp: float) -> float:
        adp = saturated_state(t_adp, inlet.p_pa)
        out = _mix(inlet, adp, bf)
        lb = breakdown_load(inlet, out, 1.0)
        # 干盘管端点（t_adp=t_dp）SHR=1，可能 shr=None（零冷量）时按 1 处理
        shr_here = lb.shr if lb.shr is not None else 1.0
        return shr_here - shr

    t_hi = inlet.dewpoint_c
    f_hi = f(t_hi)
    if f_hi < -1e-10:
        raise PsychrometricError(
            f"BF={bf:.3f} 下即使盘管表面取到进口露点，SHR 仍低于 "
            f"{shr:.3f}，目标不可达",
            ERR_NOT_DEHUMIDIFYING,
        )
    if f_hi == 0.0:
        t_adp = t_hi  # SHR=1 干盘管端点
    else:
        lo, hi = _extend_down(
            f, t_hi, miss_code=ERR_NOT_DEHUMIDIFYING,
            miss_message=(
                f"BF={bf:.3f} 下搜索到温度下限仍无法达到目标 SHR={shr:.3f}，"
                "目标物理不可达"
            ),
        )
        t_adp = bisection(f, lo, hi, tol=T_TOL, what="ADP（BF+目标SHR）")
    adp = saturated_state(t_adp, inlet.p_pa)
    return _finish(inlet, adp, bf, "bf_shr", m_da, allow_zero_load=False)


# --------------------------------------------------------------------------
# 统一入口：由已给标量自动判定模式（HTTP 层只做字段映射）
# --------------------------------------------------------------------------

def solve_coil(
    inlet: AirState,
    *,
    bf: float | None = None,
    t_adp_c: float | None = None,
    target_t_out_c: float | None = None,
    target_shr: float | None = None,
    outlet: AirState | None = None,
    m_da: float = 1.0,
) -> CoilResult:
    """按调用方给出的已知量选择求解路径；条件不足/冲突即抛领域错误。"""
    has_target_t = target_t_out_c is not None
    has_target_shr = target_shr is not None
    has_outlet = outlet is not None
    has_adp = t_adp_c is not None
    has_bf = bf is not None

    if has_adp:
        if not has_bf:
            raise PsychrometricError(
                "正算模式必须同时给出装置露点温度 t_adp_c 与旁通系数 bf；"
                "只给 ADP 温度无法确定出风",
                ERR_INVALID_REQUEST,
            )
        if has_target_t or has_target_shr or has_outlet:
            raise PsychrometricError(
                "已给出 ADP+BF 正算时，不能再给目标出风温度/SHR/出风状态，"
                "条件互相冲突",
                ERR_INVALID_REQUEST,
            )
        return solve_direct(inlet, t_adp_c, bf, m_da)  # type: ignore[arg-type]

    if has_outlet:
        if has_target_t or has_target_shr or has_bf:
            raise PsychrometricError(
                "已给出完整出风状态时，不能再给目标出风温度/SHR/旁通系数，"
                "条件互相冲突",
                ERR_INVALID_REQUEST,
            )
        return solve_from_outlet(inlet, outlet, m_da)

    if has_target_t and has_target_shr:
        if has_bf:
            raise PsychrometricError(
                "已给出目标出风温度+目标SHR即可联立，不能再指定 BF",
                ERR_INVALID_REQUEST,
            )
        return solve_from_target_t_shr(
            inlet, target_t_out_c, target_shr, m_da  # type: ignore[arg-type]
        )

    if has_bf and has_target_t:
        if has_target_shr:
            raise PsychrometricError(
                "已知量冲突：BF、目标出风温度、目标SHR三者不能同时给",
                ERR_INVALID_REQUEST,
            )
        return solve_from_bf_target_t(
            inlet, bf, target_t_out_c, m_da  # type: ignore[arg-type]
        )

    if has_bf and has_target_shr:
        return solve_from_bf_shr(inlet, bf, target_shr, m_da)  # type: ignore[arg-type]

    raise PsychrometricError(
        "已知量不足，无法反算装置露点与旁通系数。支持的组合："
        "(1) ADP温度+BF 正算；(2) 完整出风状态；"
        "(3) 目标出风温度+目标SHR；(4) BF+目标出风温度；(5) BF+目标SHR。"
        "单独给目标出风温度或单独给目标SHR时欠定（两个混合方程含两个未知量），"
        "本服务不猜测解。",
        ERR_INVALID_REQUEST,
    )
