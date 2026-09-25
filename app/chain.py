"""多级串联盘管链核算。

在已验证的单级能力（:mod:`app.adp` 的五种求解组合、:mod:`app.cooling`
的冷量分解）之上，把“一条串联处理链”组织成有整体约束的对象，**不复制
任何单级求解逻辑**：

* **级间传递**：前一级的出口 :class:`AirState` 对象原样作为后一级的进口，
  不按字段重新展开，含湿量/焓/气压逐位一致；各级冷量因此望远镜求和，
  全链总冷量严格等于分级之和（浮点求和误差量级）。
* **混合模式**：每级各自给定已知量（ADP+BF 正算 / 完整出风 / 目标温度+SHR /
  BF+目标温度 / BF+目标SHR / 仅 BF / 仅 ADP / 纯旁通透传），一条链里混排。
* **全链目标反推**：只给最终目标出风状态（或最终目标温度+总体目标 SHR）
  加上部分级约束时，把未定级的装置露点/旁通系数解出来，使全链恰好落到
  目标。未知量 = 2 时嵌套二分求解（末级全自由则退化为单级"完整出风
  反算"精确求解）；> 2 判欠定、< 2 按过定做一致性核对；迭代不收敛报
  ``no_convergence`` 并指明卡在哪一级，**绝不把中途试探值当结果返回**
  （收敛后必须用完整链复算并核对目标才放行）。过程线与饱和曲线可能
  有两个交点（单级反算即如此），与单级保持一致：从暖端向冷端扫描，
  取装置露点最暖的那组解。
* **收口校验**：沿链温度不回升、含湿量不增加（纯透传级取等号）、
  总冷量 = 分级之和、总去湿量 = 分级去湿之和、至少一级实际做功；
  违反即 ``invalid_chain`` 并定位级次。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import adp as adp_mod
from . import psychrometrics as psy
from .cooling import LoadBreakdown, breakdown_load
from .errors import (
    PsychrometricError,
    ERR_INVALID_BF,
    ERR_INVALID_CHAIN,
    ERR_INVALID_REQUEST,
    ERR_NO_CONVERGENCE,
    ERR_NOT_DEHUMIDIFYING,
)
from .numerics import bisection
from .states import AirState

# ---- 容差 -----------------------------------------------------------------

#: 沿链温度不回升的容差（°C）
T_MONO_TOL = 1e-9
#: 沿链含湿量不增加的容差（kg/kg）
W_MONO_TOL = 1e-12
#: 总冷量 = 分级之和的收口容差（相对）
ENERGY_CLOSE_REL_TOL = 1e-9
#: 总去湿量 = 分级去湿之和的收口容差
W_CLOSE_TOL = 1e-12
#: 判定“本级不做功”的冷量阈值（kW），与单级一致
MIN_WORK_KW = 1e-8
#: 反推终验：出风温度 / 含湿量 / 焓 命中目标的容差
T_TARGET_TOL = 1e-6
W_TARGET_TOL = 1e-9
H_TARGET_TOL = 1e-6
#: 装置露点搜索下界（°C）
ADP_LO = psy.T_MIN_C + 1.0
#: 外层未知量扫描采样数（定位残差号变区间）
SCAN_POINTS = 41
#: 残差平坦判据：未知量对全链出口含湿量无影响即欠定
W_FLAT_TOL = 1e-15

# ---- 级的已知量分类 --------------------------------------------------------

FULL = "full"
"""已知量自足（0 个未知量）：五种单级组合之一。"""
BYPASS = "bypass"
"""纯透传级：仅给 BF=1.0，进出口一致、本级冷量为零。"""
PARTIAL_ADP = "partial_adp"
"""只给了 BF，装置露点待全链目标反推。"""
PARTIAL_BF = "partial_bf"
"""只给了装置露点，BF 待全链目标反推。"""
FREE = "free"
"""两级未知量都待全链目标反推。"""

_UNKNOWN_PARAMS = {
    PARTIAL_ADP: ("t_adp_c",),
    PARTIAL_BF: ("bf",),
    FREE: ("t_adp_c", "bf"),
}

_PARAM_LABEL = {"t_adp_c": "装置露点", "bf": "旁通系数"}

_KIND_LABEL = {
    PARTIAL_ADP: "只给了旁通系数，装置露点未知",
    PARTIAL_BF: "只给了装置露点，旁通系数未知",
    FREE: "装置露点与旁通系数都未知",
}


@dataclass(frozen=True)
class StageSpec:
    """一级的已知量（``outlet`` 已展开为 :class:`AirState`）。"""

    t_adp_c: float | None = None
    bf: float | None = None
    target_t_out_c: float | None = None
    target_shr: float | None = None
    outlet: AirState | None = None


@dataclass(frozen=True)
class StageResult:
    """一级的核算结果。"""

    index: int
    """级次（1 起，沿风向）。"""
    mode: str
    """本级实际求解模式（含透传级的 ``bypass``）。"""
    inlet: AirState
    outlet: AirState
    adp: AirState | None
    """装置露点饱和状态；纯透传级无装置露点，为 ``None``。"""
    bf: float
    loads: LoadBreakdown


@dataclass(frozen=True)
class ChainResult:
    """整条链的核算结果。"""

    mode: str
    """``forward``（逐级正算）或 ``inverse_target``（全链目标反推）。"""
    stages: list[StageResult]
    inlet: AirState
    outlet: AirState
    loads: LoadBreakdown
    """全链合计冷量分解（原始进口 → 最终出口）。"""
    fractions: list[float]
    """各级冷量占全链总冷量的份额。"""
    delta_w: float
    """全链总去湿量 kg/kg（= 各级去湿量之和，望远镜守恒）。"""


# --------------------------------------------------------------------------
# 级分类与单级求解
# --------------------------------------------------------------------------

def _classify(spec: StageSpec, index: int) -> str:
    """判定一级的已知量形态；组合不合法即定位级次拒绝。"""
    given = {name for name in ("t_adp_c", "bf", "target_t_out_c",
                               "target_shr", "outlet")
             if getattr(spec, name) is not None}
    if given in ({"t_adp_c", "bf"}, {"outlet"},
                 {"target_t_out_c", "target_shr"},
                 {"bf", "target_t_out_c"}, {"bf", "target_shr"}):
        return FULL
    if given == {"bf"}:
        if not (0.0 <= spec.bf <= 1.0):  # type: ignore[operator]
            raise PsychrometricError(
                f"第{index}级：旁通系数 BF={spec.bf!r} 不在 [0, 1]",
                ERR_INVALID_BF, stage=index,
            )
        return BYPASS if spec.bf == 1.0 else PARTIAL_ADP
    if given == {"t_adp_c"}:
        return PARTIAL_BF
    if not given:
        return FREE
    raise PsychrometricError(
        f"第{index}级已知量组合无法求解（给出："
        f"{', '.join(sorted(given))}）。每级支持的组合：ADP+BF 正算 / "
        "完整出风 / 目标温度+目标SHR / BF+目标温度 / BF+目标SHR / "
        "仅BF（BF=1 为纯透传级，否则待全链反推本级ADP）/ "
        "仅ADP（待全链反推本级BF）/ 全不给（待全链反推）",
        ERR_INVALID_REQUEST, stage=index,
    )


def _classify_all(specs: list[StageSpec]) -> list[str]:
    if not specs:
        raise PsychrometricError(
            "空链：至少需要一级盘管才能构成处理链", ERR_INVALID_CHAIN
        )
    return [_classify(spec, i) for i, spec in enumerate(specs, start=1)]


def _solve_stage(spec: StageSpec, kind: str, inlet: AirState, m_da: float,
                 index: int, assignments: dict[str, float]) -> StageResult:
    """解一级：透传级直接构造，其余一律委托 :func:`adp.solve_coil`。

    单级抛出的领域错误在这里补上级次后继续抛（保留原错误码与雾区标记）。
    """
    t_adp_c = assignments.get("t_adp_c", spec.t_adp_c)
    bf = assignments.get("bf", spec.bf)

    if kind == BYPASS:
        # 纯透传级：BF=1，出口原样等于进口、冷量为零，装置露点无意义
        loads = breakdown_load(inlet, inlet, m_da)
        return StageResult(index=index, mode="bypass", inlet=inlet,
                           outlet=inlet, adp=None, bf=1.0, loads=loads)

    if spec.outlet is not None and spec.outlet.p_pa != inlet.p_pa:
        raise PsychrometricError(
            f"第{index}级目标出风气压 {spec.outlet.p_pa:.0f} Pa 与链内气压 "
            f"{inlet.p_pa:.0f} Pa 不一致，无法串联",
            ERR_INVALID_CHAIN, stage=index,
        )

    try:
        res = adp_mod.solve_coil(
            inlet,
            bf=bf,
            t_adp_c=t_adp_c,
            target_t_out_c=spec.target_t_out_c,
            target_shr=spec.target_shr,
            outlet=spec.outlet,
            m_da=m_da,
        )
    except PsychrometricError as exc:
        raise PsychrometricError(
            f"第{index}级：{exc.message}", exc.code, stage=index, fog=exc.fog
        ) from exc
    return StageResult(index=index, mode=res.mode, inlet=res.inlet,
                       outlet=res.outlet, adp=res.adp, bf=res.bf,
                       loads=res.loads)


def _run_forward(inlet: AirState, specs: list[StageSpec], kinds: list[str],
                 m_da: float,
                 assignments: dict[int, dict[str, float]]) -> list[StageResult]:
    """从链首到链尾顺序求解；出口对象原样成为下一级进口，不重新展开。"""
    results: list[StageResult] = []
    current = inlet
    for i, (spec, kind) in enumerate(zip(specs, kinds), start=1):
        res = _solve_stage(spec, kind, current, m_da, i,
                           assignments.get(i, {}))
        results.append(res)
        current = res.outlet
    return results


def _stage_inlet(inlet: AirState, specs: list[StageSpec], kinds: list[str],
                 m_da: float, assignments: dict[int, dict[str, float]],
                 index: int) -> AirState:
    """只解第 1..index-1 级，返回第 ``index`` 级的进口状态。"""
    current = inlet
    for i in range(1, index):
        current = _solve_stage(specs[i - 1], kinds[i - 1], current, m_da, i,
                               assignments.get(i, {})).outlet
    return current


# --------------------------------------------------------------------------
# 收口：链级不变量校验 + 合计
# --------------------------------------------------------------------------

def _close_chain(inlet: AirState, results: list[StageResult], m_da: float,
                 mode: str) -> ChainResult:
    # 不变量 1：沿链温度不回升、含湿量不增加（纯透传级取等号）
    for r in results:
        if r.outlet.t_db_c > r.inlet.t_db_c + T_MONO_TOL:
            raise PsychrometricError(
                f"第{r.index}级出口温度 {r.outlet.t_db_c:.6f} °C 高于本级进口 "
                f"{r.inlet.t_db_c:.6f} °C，沿链温度回升，违反链内单调性",
                ERR_INVALID_CHAIN, stage=r.index,
            )
        if r.outlet.w > r.inlet.w + W_MONO_TOL:
            raise PsychrometricError(
                f"第{r.index}级出口含湿量 {r.outlet.w * 1000:.6f} g/kg 高于本级进口 "
                f"{r.inlet.w * 1000:.6f} g/kg，沿链含湿量增加，违反链内单调性",
                ERR_INVALID_CHAIN, stage=r.index,
            )

    q_sum = sum(r.loads.q_total for r in results)
    if q_sum <= MIN_WORK_KW:
        raise PsychrometricError(
            "整条链没有任何一级实际冷却（各级均为透传），无效工况",
            ERR_INVALID_CHAIN,
        )

    outlet = results[-1].outlet
    loads = breakdown_load(inlet, outlet, m_da)

    # 不变量 2：总冷量 = 分级冷量之和（级间对象直传保证望远镜求和）
    if abs(loads.q_total - q_sum) > ENERGY_CLOSE_REL_TOL * max(1.0, abs(q_sum)):
        raise PsychrometricError(
            f"全链总冷量 {loads.q_total:.9f} kW 与分级之和 {q_sum:.9f} kW "
            "不自洽，级间状态传递出现漂移",
            ERR_INVALID_CHAIN,
        )
    # 不变量 3：总去湿量 = 分级去湿量之和
    delta_w = inlet.w - outlet.w
    dw_sum = sum(r.inlet.w - r.outlet.w for r in results)
    if abs(dw_sum - delta_w) > W_CLOSE_TOL * max(1.0, abs(delta_w)):
        raise PsychrometricError(
            f"全链总去湿量 {delta_w:.9e} kg/kg 与分级去湿之和 {dw_sum:.9e} "
            "kg/kg 不自洽，级间状态传递出现漂移",
            ERR_INVALID_CHAIN,
        )

    fractions = [r.loads.q_total / loads.q_total for r in results]
    return ChainResult(mode=mode, stages=results, inlet=inlet, outlet=outlet,
                       loads=loads, fractions=fractions, delta_w=delta_w)


# --------------------------------------------------------------------------
# 正算：逐级各自给定已知量
# --------------------------------------------------------------------------

def solve_chain(inlet: AirState, specs: list[StageSpec],
                m_da: float = 1.0) -> ChainResult:
    """逐级正算：每级已知量必须自足（或 BF=1 纯透传），否则拒绝。"""
    kinds = _classify_all(specs)
    for i, kind in enumerate(kinds, start=1):
        if kind in _UNKNOWN_PARAMS:
            raise PsychrometricError(
                f"第{i}级已知量不足（{_KIND_LABEL[kind]}），且未提供全链目标，"
                "无法确定本级出风",
                ERR_INVALID_REQUEST, stage=i,
            )
    results = _run_forward(inlet, specs, kinds, m_da, {})
    return _close_chain(inlet, results, m_da, "forward")


# --------------------------------------------------------------------------
# 全链目标反推
# --------------------------------------------------------------------------

def target_state_from_t_shr(inlet: AirState, t_out_c: float,
                            shr: float) -> AirState:
    """全链目标：最终出风温度 + 总体 SHR → 最终出风状态（复用单级闭式解）。"""
    w_out = adp_mod.outlet_w_from_t_shr(inlet, t_out_c, shr)
    return AirState(t_db_c=float(t_out_c), w=w_out, p_pa=inlet.p_pa)


def _validate_target(inlet: AirState, target: AirState) -> None:
    if target.p_pa != inlet.p_pa:
        raise PsychrometricError(
            f"全链目标出风气压 {target.p_pa:.0f} Pa 与进风气压 "
            f"{inlet.p_pa:.0f} Pa 不一致，无法作为链目标",
            ERR_INVALID_CHAIN,
        )
    if target.w > inlet.w + adp_mod.W_EPS:
        raise PsychrometricError(
            f"全链目标出风含湿量 {target.w * 1000:.3f} g/kg 高于进风 "
            f"{inlet.w * 1000:.3f} g/kg（出风比进风还湿），不是冷却去湿工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    if target.t_db_c >= inlet.t_db_c - 1e-12:
        raise PsychrometricError(
            "全链目标出风温度不低于进风温度，不构成冷却", ERR_NOT_DEHUMIDIFYING
        )
    if target.enthalpy >= inlet.enthalpy - 1e-9:
        raise PsychrometricError(
            "全链目标出风焓不低于进风焓，不构成冷却", ERR_NOT_DEHUMIDIFYING
        )


def _assert_hits_target(final: AirState, target: AirState, context: str) -> None:
    if (abs(final.t_db_c - target.t_db_c) > T_TARGET_TOL
            or abs(final.w - target.w) > W_TARGET_TOL
            or abs(final.enthalpy - target.enthalpy) > H_TARGET_TOL):
        raise PsychrometricError(
            f"全链{context}：最终出风 (t={final.t_db_c:.6f} °C, "
            f"W={final.w * 1000:.6f} g/kg) 未命中目标 "
            f"(t={target.t_db_c:.6f} °C, W={target.w * 1000:.6f} g/kg)，"
            "给定目标与各级约束互相矛盾（过定/不可达）",
            ERR_INVALID_CHAIN,
        )


class _Infeasible(Exception):
    """内部信号：当前外层取值下，内层无解或链在该处不可评估。"""


def _param_bracket(param: str, inlet: AirState, specs: list[StageSpec],
                   kinds: list[str], m_da: float,
                   assignments: dict[int, dict[str, float]],
                   index: int) -> tuple[float, float]:
    """未知量的物理搜索区间：BF ∈ [0,1]；ADP ∈ [下限, 该级进口露点]。"""
    if param == "bf":
        return 0.0, 1.0
    stage_in = _stage_inlet(inlet, specs, kinds, m_da, assignments, index)
    return ADP_LO, stage_in.dewpoint_c - 1e-9


def _w_residual(inlet: AirState, specs: list[StageSpec], kinds: list[str],
                m_da: float, base: dict[int, dict[str, float]],
                idx: int, param: str, target_w: float):
    """构造“全链出口含湿量 − 目标含湿量”的残差函数（对未知量单调）。

    试探点落入雾区/非去湿区（冷侧）时按“过干”处理，把二分推向物理区；
    但若被试级自己的定值装置露点不合法（与试探无关的配置错误），原样抛出。
    """

    def f(u: float) -> float:
        stage_assign = {**base.get(idx, {}), param: u}
        try:
            final = _run_forward(inlet, specs, kinds, m_da,
                                 {**base, idx: stage_assign})[-1].outlet
        except PsychrometricError as exc:
            if exc.code == ERR_NOT_DEHUMIDIFYING and not (
                    param == "bf" and exc.stage == idx and not exc.fog):
                return -1.0
            raise
        return final.w - target_w

    return f


def _bisect_guarded(f, lo: float, hi: float, idx: int, param: str) -> float:
    """二分求解并把不收敛改写成带级次的领域错误，绝不返回试探中间值。"""
    try:
        return bisection(
            f, lo, hi,
            tol=1e-9 if param == "t_adp_c" else 1e-12,
            what=f"全链反推第{idx}级{_PARAM_LABEL[param]}",
        )
    except _Infeasible:
        raise
    except PsychrometricError as exc:
        if exc.code == ERR_NO_CONVERGENCE:
            raise PsychrometricError(
                f"全链反推不收敛：卡在求解第{idx}级{_PARAM_LABEL[param]}"
                f"（{exc.message}），不返回任何中途试探值",
                ERR_NO_CONVERGENCE, stage=idx,
            ) from exc
        raise


def _diagnostic_eval(inlet: AirState, specs: list[StageSpec], kinds: list[str],
                     m_da: float, unknowns: list[tuple[int, str]]) -> None:
    """用最不做功的未知量取值完整试算一遍。

    若某一级与未知量无关地坏掉（如给定装置露点高于该级进口湿球），
    让它的真实错误（带级次）直接抛出，而不是被笼统的“目标不可达”吞掉。
    """
    assignments: dict[int, dict[str, float]] = {}
    for idx, param in unknowns:
        if param == "bf":
            assignments.setdefault(idx, {})["bf"] = 1.0
        else:
            stage_in = _stage_inlet(inlet, specs, kinds, m_da, assignments, idx)
            assignments.setdefault(idx, {})["t_adp_c"] = \
                stage_in.dewpoint_c - 1e-6
    _run_forward(inlet, specs, kinds, m_da, assignments)


def _solve_single_unknown(inlet: AirState, specs: list[StageSpec],
                          kinds: list[str], unknown: tuple[int, str],
                          target: AirState, m_da: float
                          ) -> dict[int, dict[str, float]]:
    """1 个未知量 + 2 个目标分量：按含湿量解出，再核对焓（过定检验）。"""
    idx, param = unknown
    lo, hi = _param_bracket(param, inlet, specs, kinds, m_da, {}, idx)
    if hi <= lo:
        raise PsychrometricError(
            f"第{idx}级进口含湿量过低，无可调的装置露点区间，全链目标不可达",
            ERR_INVALID_CHAIN, stage=idx,
        )
    f = _w_residual(inlet, specs, kinds, m_da, {}, idx, param, target.w)
    flo, fhi = f(lo), f(hi)
    if abs(fhi - flo) < W_FLAT_TOL:
        raise PsychrometricError(
            f"第{idx}级{_PARAM_LABEL[param]}对全链出口含湿量无影响，"
            "目标无法约束它（欠定），拒绝猜测解",
            ERR_INVALID_CHAIN, stage=idx,
        )
    if flo > 0.0 or fhi < 0.0:
        _diagnostic_eval(inlet, specs, kinds, m_da, [unknown])
        raise PsychrometricError(
            f"无论第{idx}级{_PARAM_LABEL[param]}在 [{lo:.4f}, {hi:.4f}] 内如何取值，"
            f"全链出口含湿量都到不了目标 {target.w * 1000:.4f} g/kg，目标不可达",
            ERR_INVALID_CHAIN, stage=idx,
        )
    u = _bisect_guarded(f, lo, hi, idx, param)
    assignments = {idx: {param: u}}
    final = _run_forward(inlet, specs, kinds, m_da, assignments)[-1].outlet
    if abs(final.enthalpy - target.enthalpy) > H_TARGET_TOL:
        raise PsychrometricError(
            f"第{idx}级{_PARAM_LABEL[param]}按目标含湿量解出后，全链出口焓 "
            f"{final.enthalpy:.6f} kJ/kg 与目标焓 {target.enthalpy:.6f} kJ/kg "
            "对不上：未知量只有 1 个而目标有 2 个分量，约束互相矛盾（过定）",
            ERR_INVALID_CHAIN, stage=idx,
        )
    return assignments


def _solve_two_unknowns(inlet: AirState, specs: list[StageSpec],
                        kinds: list[str], unknowns: list[tuple[int, str]],
                        target: AirState, m_da: float
                        ) -> dict[int, dict[str, float]]:
    """2 个未知量 + 2 个目标分量：外层二分焓残差，内层二分含湿量残差。

    外层取较前级的未知量（同一级则先 ADP），内层取较后级的未知量。
    内层把全链出口含湿量钉到目标上，外层在该一维流形上找焓的零点。
    """
    (i1, p1), (i2, p2) = unknowns

    def inner(u1: float) -> float:
        a1 = {i1: {p1: u1}}
        try:
            lo2, hi2 = _param_bracket(p2, inlet, specs, kinds, m_da, a1, i2)
        except PsychrometricError as exc:
            if exc.code == ERR_NOT_DEHUMIDIFYING:
                raise _Infeasible from exc
            raise
        if hi2 <= lo2:
            raise _Infeasible
        f2 = _w_residual(inlet, specs, kinds, m_da, a1, i2, p2, target.w)
        flo, fhi = f2(lo2), f2(hi2)
        if abs(fhi - flo) < W_FLAT_TOL or flo > 0.0 or fhi < 0.0:
            raise _Infeasible
        return _bisect_guarded(f2, lo2, hi2, i2, p2)

    def outer_resid(u1: float) -> float:
        u2 = inner(u1)
        # 两个未知量可能同属一级（FREE 级），必须合并而不是覆盖
        assign: dict[int, dict[str, float]] = {i1: {p1: u1}}
        assign.setdefault(i2, {})[p2] = u2
        try:
            final = _run_forward(inlet, specs, kinds, m_da,
                                 assign)[-1].outlet
        except PsychrometricError as exc:
            if exc.code == ERR_NOT_DEHUMIDIFYING and not (
                    p1 == "bf" and exc.stage == i1 and not exc.fog):
                raise _Infeasible from exc
            raise
        return final.enthalpy - target.enthalpy

    lo1, hi1 = _param_bracket(p1, inlet, specs, kinds, m_da, {}, i1)

    # 先扫描定位可行域与残差号变区间，再在号变区间内二分
    prev: tuple[float, float] | None = None
    bracket: tuple[float, float] | None = None
    for k in range(SCAN_POINTS):
        x = lo1 + (hi1 - lo1) * k / (SCAN_POINTS - 1)
        try:
            r = outer_resid(x)
        except _Infeasible:
            prev = None
            continue
        if r == 0.0:
            bracket = (x, x)
            break
        if prev is not None and prev[1] * r < 0.0:
            bracket = (prev[0], x)
            break
        prev = (x, r)

    if bracket is None:
        # 某级若与未知量无关地坏掉，让它的真实错误先抛出来
        _diagnostic_eval(inlet, specs, kinds, m_da, unknowns)
        raise PsychrometricError(
            f"全链目标不可达：在第{i1}级{_PARAM_LABEL[p1]}的整个可行范围内，"
            "无论怎么取值都无法让全链出口同时命中目标含湿量与目标焓"
            "（目标与各级约束矛盾，或未知量对出口无影响导致欠定）",
            ERR_INVALID_CHAIN, stage=i1,
        )

    if bracket[0] == bracket[1]:
        u1 = bracket[0]
    else:
        try:
            u1 = _bisect_guarded(outer_resid, bracket[0], bracket[1], i1, p1)
        except _Infeasible as exc:
            raise PsychrometricError(
                f"全链反推不收敛：第{i1}级{_PARAM_LABEL[p1]}的搜索区间内部"
                "不可评估，不返回任何中途试探值",
                ERR_NO_CONVERGENCE, stage=i1,
            ) from exc
    u2 = inner(u1)
    assignments: dict[int, dict[str, float]] = {i1: {p1: u1}}
    assignments.setdefault(i2, {})[p2] = u2
    return assignments


def solve_chain_inverse(inlet: AirState, specs: list[StageSpec],
                        target: AirState, m_da: float = 1.0) -> ChainResult:
    """全链目标反推：给定最终目标出风状态，解出未定级的 ADP/BF。

    未知量（每个待定 ADP/BF 计 1 个）与目标提供的 2 个方程比较：

    * 恰好 2 个：嵌套二分唯一求解；
    * 1 个：按目标含湿量解出后必须再命中目标焓，否则过定拒绝；
    * 0 个：各级均已给定，正算结果必须命中目标，否则过定拒绝；
    * 超过 2 个：欠定，拒绝猜测解。
    """
    kinds = _classify_all(specs)
    _validate_target(inlet, target)

    unknowns: list[tuple[int, str]] = []
    for i, kind in enumerate(kinds, start=1):
        for param in _UNKNOWN_PARAMS.get(kind, ()):
            unknowns.append((i, param))

    if len(unknowns) > 2:
        desc = "、".join(f"第{i}级{_PARAM_LABEL[p]}" for i, p in unknowns)
        raise PsychrometricError(
            f"全链反推欠定：{len(unknowns)} 个未知量（{desc}），而最终目标"
            "只提供 2 个方程，凑不出唯一解，拒绝猜测。请补充级的约束"
            "（如给定部分级的 ADP/BF 或本级目标）",
            ERR_INVALID_CHAIN,
        )

    if not unknowns:
        results = _run_forward(inlet, specs, kinds, m_da, {})
        _assert_hits_target(results[-1].outlet, target,
                            "各级均已给定（过定核对）")
        return _close_chain(inlet, results, m_da, "inverse_target")

    if len(unknowns) == 1:
        assignments = _solve_single_unknown(inlet, specs, kinds,
                                            unknowns[0], target, m_da)
    else:
        assignments = _solve_two_unknowns(inlet, specs, kinds,
                                          unknowns, target, m_da)

    # 收敛后必须用完整链复算并核对目标，绝不把试探值当结果
    results = _run_forward(inlet, specs, kinds, m_da, assignments)
    _assert_hits_target(results[-1].outlet, target, "反推复算")
    return _close_chain(inlet, results, m_da, "inverse_target")
