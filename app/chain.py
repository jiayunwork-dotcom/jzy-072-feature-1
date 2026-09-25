"""多级串联盘管链核算。

在单级能力（:mod:`app.adp`）之上加一层「链」：若干级盘管顺序串在一根风道上，
前一级的出口**原样**（同一个 :class:`AirState` 对象逐位传递，不按字段重新
展开）成为后一级的进口，全链共用同一个进口来风与同一个干空气质量流量，
最后一级的出口就是整条链的出风。

本模块只承担链级耦合，不复制单级逻辑：

* 每一级的求解仍走 :func:`app.adp.solve_coil` / :func:`app.adp.solve_direct`；
* 冷量分解仍走 :func:`app.cooling.breakdown_load`；
* 饱和蒸汽压、含湿量、焓等基础量仍只来自 :mod:`app.psychrometrics`。

链级新增的三件事
================

1. **级间传递与不变量**：每过一级校验沿链温度不回升、含湿量不增加、级间
   气压一致（BF=1 的纯旁通透传级进出口一致，天然满足单调性，放行）。
2. **收口守恒校验**：全链总冷量（首级进口 → 末级出口）必须等于各级冷量
   之和，全链总去湿量必须等于各级去湿量之和（两者都是望远镜求和，数值
   容差内自洽）；整条链没有任何一级做功（全为透传级）判为无效工况。
3. **全链目标反推**：只给全链最终目标（完整出风状态，或目标出风温度 +
   目标 SHR）加上部分级的约束，反解未给足级的装置露点/旁通系数。先按
   「自由量数 vs 方程数」判定唯一可解 / 欠定 / 过定；可解时用阻尼牛顿
   迭代求解，不收敛报 ``no_convergence`` 并指明未知量卡在第几级，
   绝不把中途试探的临时装置露点当成结果返回。若同一组约束存在多个
   孤立解（过程线与饱和曲线可有两个交点，与单级完整出风反算同理），
   按与单级一致的确定性约定取**装置露点最暖**的解（旁通系数取较小者），
   保证同一请求永远得到同一答案。

各级已知量的给法（与单级一致，可在一条链里混用）
================================================

* ``t_adp_c`` + ``bf``：本级正算；
* ``outlet``：本级目标出风状态，反解本级 ADP/BF；
* ``target_t_out_c`` + ``target_shr``：本级目标出风温度 + 目标显热比；
* ``bf`` + ``target_t_out_c`` / ``bf`` + ``target_shr``。

仅当给了全链目标时，某一级还可以只给一半（只给 ``bf``、只给
``t_adp_c``、只给 ``target_t_out_c``、只给 ``target_shr``）或什么都不给
（自由级），由链级反推把缺的装置露点/旁通系数解出来。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import adp as adp_mod
from . import psychrometrics as psy
from .adp import CoilResult
from .cooling import LoadBreakdown, breakdown_load
from .errors import (
    PsychrometricError,
    ERR_INVALID_REQUEST,
    ERR_INVALID_STATE,
    ERR_INVALID_BF,
    ERR_INVALID_SHR,
    ERR_NOT_DEHUMIDIFYING,
    ERR_INCONSISTENT_STATE,
    ERR_NO_CONVERGENCE,
    ERR_CHAIN_UNDERDETERMINED,
    ERR_CHAIN_OVERDETERMINED,
)
from .states import AirState

# ---- 链级容差 ---------------------------------------------------------------

#: 判定一级为纯旁通透传级的总冷量阈值（kW），与单级零冷量判定一致
ZERO_LOAD_TOL = 1e-8
#: 沿链温度单调容差（°C）
T_MONO_TOL = 1e-9
#: 沿链含湿量单调容差（kg/kg）
W_MONO_TOL = 1e-9
#: 收口守恒校验的相对容差（望远镜求和的浮点累积远低于此）
CONSERVE_RTOL = 1e-9
#: 全量给定 + 全链目标时，目标一致性核验容差
VERIFY_T_TOL = 1e-6
VERIFY_W_TOL = 1e-9

# ---- 反推（阻尼牛顿）参数 ----------------------------------------------------

#: 残差归一化尺度：温度 °C、含湿量 g/kg、显热比 %
T_SCALE = 1.0
W_SCALE = 1e-3
S_SCALE = 1e-2
#: 牛顿收敛阈值（归一化残差无穷范数）
NEWTON_TOL = 1e-8
NEWTON_MAX_ITER = 60
LINE_SEARCH_MAX = 40
#: 试探点落到非法工况时返回的罚残差（让线搜索拒绝该方向）
PENALTY = 1e6
#: ADP 未知量在迭代中的活动范围（°C）
ADP_BOX = (psy.T_MIN_C + 1.0, 80.0)

#: 需要链级反推补全参数的级类型（本级两个自由度未给全）
_PARAMETERIZED_KINDS = ("free", "bf_only", "adp_only", "t_only", "shr_only")


# ---- 输入/结果数据结构 -------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """一级的已知量（与单级 :func:`app.adp.solve_coil` 的给法一致）。"""

    name: str | None = None
    """级名（如「预冷」「深度除湿」），仅标识用"""
    t_adp_c: float | None = None
    bf: float | None = None
    target_t_out_c: float | None = None
    target_shr: float | None = None
    outlet: AirState | None = None


@dataclass(frozen=True)
class ChainTarget:
    """全链最终目标：完整出风状态，或目标出风温度 + 目标显热比。"""

    outlet: AirState | None = None
    target_t_out_c: float | None = None
    target_shr: float | None = None


@dataclass(frozen=True)
class ChainStageResult:
    """一级的核算结果。"""

    index: int
    """级次，1 起计"""
    name: str | None
    mode: str
    """本级实际采用的求解模式名"""
    passthrough: bool
    """是否纯旁通透传级（进出口一致、本级冷量为零）"""
    coil: CoilResult
    load_share: float
    """本级冷量占全链总冷量的份额"""


@dataclass(frozen=True)
class ChainResult:
    """整条链的核算结果。"""

    mode: str
    """链级求解方式：forward（逐级正算）/ verify（全量给定+目标核验）/
    inverse（带全链目标反推）"""
    inlet: AirState
    stages: tuple[ChainStageResult, ...]
    totals: LoadBreakdown
    """全链合计冷量分解（首级进口 → 末级出口）"""
    dehumidification: float
    """全链总去湿量 m_da·(W_首 − W_末)，kg/s"""
    target: AirState | None
    """全链目标（解析成完整状态后）；无目标时为 None"""

    @property
    def outlet(self) -> AirState:
        """整条链的出风 = 最后一级的出口。"""
        return self.stages[-1].coil.outlet


# ---- 内部结构 -----------------------------------------------------------------


@dataclass(frozen=True)
class _StagePlan:
    pos: int
    """0 起计的级位置"""
    index: int
    """1 起计的级次（报错用）"""
    name: str | None
    kind: str
    spec: StageSpec


@dataclass(frozen=True)
class _Unknown:
    """链级反推的一个自由量：某级的装置露点或旁通系数。"""

    pos: int
    param: str  # "t_adp" | "bf"


@dataclass(frozen=True)
class _StageEq:
    """只给半个目标的级贡献的一条方程（本级出风温度或本级 SHR）。"""

    pos: int
    kind: str  # "t" | "shr"
    value: float


def _stage_label(plan: _StagePlan) -> str:
    return f"第 {plan.index} 级" + (f"（{plan.name}）" if plan.name else "")


# --------------------------------------------------------------------------
# 级分类：把一级的已知量映射到求解方式
# --------------------------------------------------------------------------

def _classify(spec: StageSpec, pos: int) -> _StagePlan:
    index = pos + 1
    label = f"第 {index} 级" + (f"（{spec.name}）" if spec.name else "")

    def reject(msg: str, code: str = ERR_INVALID_REQUEST) -> PsychrometricError:
        return PsychrometricError(f"{label}：{msg}", code, stage_index=index)

    has_adp = spec.t_adp_c is not None
    has_bf = spec.bf is not None
    has_t = spec.target_t_out_c is not None
    has_shr = spec.target_shr is not None
    has_outlet = spec.outlet is not None

    # 标量域校验提前到分类时做（带上级次）；完整组合在 solve_coil 里还会再查
    if has_bf and not (0.0 <= spec.bf <= 1.0):  # type: ignore[operator]
        raise reject(f"旁通系数 BF={spec.bf!r} 不在 [0, 1]", ERR_INVALID_BF)
    if has_shr and not (0.0 < spec.target_shr <= 1.0):  # type: ignore[operator]
        raise reject(f"目标显热比 SHR={spec.target_shr!r} 不在 (0, 1]",
                     ERR_INVALID_SHR)

    if has_outlet:
        if has_adp or has_bf or has_t or has_shr:
            raise reject(
                "已给出本级完整出风状态时，不能再给装置露点/旁通系数/"
                "目标出风温度/目标SHR，条件互相冲突"
            )
        kind = "from_outlet"
    elif has_adp and has_bf:
        if has_t or has_shr:
            raise reject(
                "装置露点+旁通系数正算与目标出风温度/目标SHR 不能同给，"
                "条件互相冲突"
            )
        kind = "direct"
    elif has_adp:
        if has_t or has_shr:
            raise reject("装置露点温度与目标出风温度/目标SHR 不能同给，"
                         "条件互相冲突")
        kind = "adp_only"
    elif has_bf and has_t and has_shr:
        raise reject("旁通系数、目标出风温度、目标SHR 三者不能同时给，"
                     "条件互相冲突")
    elif has_bf and has_t:
        kind = "bf_target_t"
    elif has_bf and has_shr:
        kind = "bf_target_shr"
    elif has_bf:
        kind = "bf_only"
    elif has_t and has_shr:
        kind = "target_t_shr"
    elif has_t:
        kind = "t_only"
    elif has_shr:
        kind = "shr_only"
    else:
        kind = "free"
    return _StagePlan(pos=pos, index=index, name=spec.name, kind=kind, spec=spec)


# --------------------------------------------------------------------------
# 全链目标解析与校验
# --------------------------------------------------------------------------

def _resolve_target(inlet: AirState, target: ChainTarget) -> AirState:
    """把全链目标解析成完整出风状态，并校验它相对链进口确为冷却工况。"""
    has_outlet = target.outlet is not None
    has_t = target.target_t_out_c is not None
    has_shr = target.target_shr is not None

    if has_outlet and (has_t or has_shr):
        raise PsychrometricError(
            "全链目标给了完整出风状态，又同时给目标出风温度/目标SHR，"
            "条件互相冲突",
            ERR_INVALID_REQUEST,
        )
    if has_outlet:
        state = target.outlet  # type: ignore[assignment]
    elif has_t and has_shr:
        # 与单级模式 3 共用同一处闭式解，不另抄公式
        w_out = adp_mod.w_out_from_t_shr(
            inlet, target.target_t_out_c, target.target_shr  # type: ignore[arg-type]
        )
        state = AirState(t_db_c=target.target_t_out_c, w=w_out,  # type: ignore[arg-type]
                         p_pa=inlet.p_pa)
    elif has_t or has_shr:
        raise PsychrometricError(
            "全链目标欠定：目标出风温度与目标SHR 必须同时给，"
            "或直接给完整目标出风状态；只给一个无法确定目标状态",
            ERR_INVALID_REQUEST,
        )
    else:
        raise PsychrometricError(
            "全链目标为空：请给完整目标出风状态，或目标出风温度+目标SHR",
            ERR_INVALID_REQUEST,
        )

    if state.p_pa != inlet.p_pa:
        raise PsychrometricError(
            f"全链目标出风气压 {state.p_pa:.0f} Pa 与链进口 "
            f"{inlet.p_pa:.0f} Pa 不一致",
            ERR_INVALID_REQUEST,
        )
    if state.w > inlet.w + W_MONO_TOL:
        raise PsychrometricError(
            f"全链目标出风含湿量 {state.w * 1000:.3f} g/kg 高于链进口 "
            f"{inlet.w * 1000:.3f} g/kg（出风比进风还湿），不是冷却去湿工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    if state.t_db_c > inlet.t_db_c + T_MONO_TOL:
        raise PsychrometricError(
            f"全链目标出风温度 {state.t_db_c:.3f} °C 高于链进口 "
            f"{inlet.t_db_c:.3f} °C，不是冷却工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    if state.enthalpy >= inlet.enthalpy - 1e-9:
        raise PsychrometricError(
            "全链目标出风焓不低于链进口焓，不是冷却工况",
            ERR_NOT_DEHUMIDIFYING,
        )
    return state


# --------------------------------------------------------------------------
# 前向传递：逐级求解 + 级间不变量
# --------------------------------------------------------------------------

def _solve_stage(plan: _StagePlan, stage_inlet: AirState,
                 assigned: dict[int, list[float]] | None,
                 m_da: float) -> tuple[CoilResult, str]:
    """解一级。完整给定的级走单级统一入口；反推补全的级按 ADP+BF 正算。"""
    spec = plan.spec
    try:
        if plan.kind in _PARAMETERIZED_KINDS:
            t_adp, bf = assigned[plan.pos]  # type: ignore[index]
            res = adp_mod.solve_direct(stage_inlet, t_adp, bf, m_da)
            return res, "direct"
        res = adp_mod.solve_coil(
            stage_inlet,
            bf=spec.bf,
            t_adp_c=spec.t_adp_c,
            target_t_out_c=spec.target_t_out_c,
            target_shr=spec.target_shr,
            outlet=spec.outlet,
            m_da=m_da,
        )
        # 与 /coil 路由一致：目标温度+SHR 内部复用了完整出风求解器，修正模式名
        mode = "target_t_shr" if plan.kind == "target_t_shr" else res.mode
        return res, mode
    except PsychrometricError as exc:
        raise PsychrometricError(
            f"{_stage_label(plan)}：{exc.message}",
            exc.code,
            stage_index=plan.index,
        ) from exc


def _check_stage_invariants(plan: _StagePlan, stage_inlet: AirState,
                            res: CoilResult) -> None:
    """级间不变量：气压一致、沿链温度不回升、含湿量不增加。"""
    out = res.outlet
    label = _stage_label(plan)
    if out.p_pa != stage_inlet.p_pa:
        raise PsychrometricError(
            f"{label}：级间气压不一致（本级进口 {stage_inlet.p_pa:.0f} Pa，"
            f"出口 {out.p_pa:.0f} Pa）",
            ERR_INVALID_STATE,
            stage_index=plan.index,
        )
    if out.t_db_c > stage_inlet.t_db_c + T_MONO_TOL:
        raise PsychrometricError(
            f"{label}：出风温度 {out.t_db_c:.3f} °C 高于本级进口 "
            f"{stage_inlet.t_db_c:.3f} °C，沿链温度回升，不是冷却工况",
            ERR_NOT_DEHUMIDIFYING,
            stage_index=plan.index,
        )
    if out.w > stage_inlet.w + W_MONO_TOL:
        raise PsychrometricError(
            f"{label}：出风含湿量 {out.w * 1000:.3f} g/kg 高于本级进口 "
            f"{stage_inlet.w * 1000:.3f} g/kg，含湿量沿链增加，"
            "不是冷却去湿工况",
            ERR_NOT_DEHUMIDIFYING,
            stage_index=plan.index,
        )


def _forward_pass(inlet: AirState, plans: list[_StagePlan], m_da: float,
                  assigned: dict[int, list[float]] | None = None,
                  ) -> list[tuple[CoilResult, str]]:
    """从头到尾把整条链算通：前一级的出口对象原样传给后一级。"""
    solved: list[tuple[CoilResult, str]] = []
    state = inlet
    for plan in plans:
        res, mode = _solve_stage(plan, state, assigned, m_da)
        _check_stage_invariants(plan, state, res)
        solved.append((res, mode))
        state = res.outlet  # 同一对象逐位传递，绝不按字段重新展开
    return solved


# --------------------------------------------------------------------------
# 反推：阻尼牛顿 + 多初值
# --------------------------------------------------------------------------

def _assign_params(plans: list[_StagePlan], unknowns: list[_Unknown],
                   x: list[float]) -> dict[int, list[float]]:
    """把自由量向量 x 铺回各级，得到每级的 (t_adp, bf)。"""
    assigned: dict[int, list[float]] = {}
    for plan in plans:
        if plan.kind in ("free", "t_only", "shr_only"):
            assigned[plan.pos] = [0.0, 0.0]
        elif plan.kind == "bf_only":
            assigned[plan.pos] = [0.0, plan.spec.bf]  # type: ignore[list-item]
        elif plan.kind == "adp_only":
            assigned[plan.pos] = [plan.spec.t_adp_c, 0.0]  # type: ignore[list-item]
    for unk, val in zip(unknowns, x):
        assigned[unk.pos][0 if unk.param == "t_adp" else 1] = val
    return assigned


def _make_residual(inlet: AirState, plans: list[_StagePlan],
                   unknowns: list[_Unknown], stage_eqs: list[_StageEq],
                   target: AirState, m_da: float):
    """构造归一化残差函数：级目标方程 + 全链目标（温度、含湿量）两条。"""
    n_eq = len(stage_eqs) + 2

    def residual(x: list[float]) -> list[float]:
        assigned = _assign_params(plans, unknowns, x)
        try:
            solved = _forward_pass(inlet, plans, m_da, assigned)
        except PsychrometricError:
            # 试探点落到非法工况（ADP 高于湿球、超饱和等）：给大罚残差，
            # 让线搜索退回，绝不把这个临时状态当成结果
            return [PENALTY] * n_eq
        r: list[float] = []
        for eq in stage_eqs:
            res = solved[eq.pos][0]
            if eq.kind == "t":
                r.append((res.outlet.t_db_c - eq.value) / T_SCALE)
            else:
                shr = res.loads.shr
                if shr is None:  # 零冷量时 SHR 无定义，按非法试探处理
                    return [PENALTY] * n_eq
                r.append((shr - eq.value) / S_SCALE)
        final = solved[-1][0].outlet
        r.append((final.t_db_c - target.t_db_c) / T_SCALE)
        r.append((final.w - target.w) / W_SCALE)
        return r

    return residual


def _clamp(value: float, param: str) -> float:
    if param == "t_adp":
        return min(max(value, ADP_BOX[0]), ADP_BOX[1])
    return min(max(value, 0.0), 1.0)


def _inf_norm(v: list[float]) -> float:
    return max(abs(x) for x in v)


def _jacobian(residual, x: list[float], r0: list[float],
              unknowns: list[_Unknown]) -> list[list[float]]:
    """中心差分数值雅可比（未知量个数很小，代价可忽略）。"""
    n, m = len(x), len(r0)
    jac = [[0.0] * n for _ in range(m)]
    for j, unk in enumerate(unknowns):
        h = 1e-5 * max(1.0, abs(x[j])) if unk.param == "t_adp" else 1e-6
        xp, xm = list(x), list(x)
        xp[j] = _clamp(x[j] + h, unk.param)
        xm[j] = _clamp(x[j] - h, unk.param)
        rp, rm = residual(xp), residual(xm)
        denom = xp[j] - xm[j]
        for i in range(m):
            jac[i][j] = (rp[i] - rm[i]) / denom
    return jac


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    """小规模高斯消元（部分主元）；奇异返回 None。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-13:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def _initial_starts(unknowns: list[_Unknown], target: AirState) -> list[list[float]]:
    """启发式初值：ADP 在目标温度上下取若干档、BF 取中小大三档，多组起步。

    多组初值既为收敛保险，也用于把多个孤立根都找出来，便于按确定性
    约定（最暖 ADP）选根。
    """
    cands: list[list[float]] = []
    for unk in unknowns:
        if unk.param == "t_adp":
            b = target.t_db_c
            cands.append([
                min(ADP_BOX[1], b + 2.0),
                max(1.0, b - 2.0),
                max(0.5, b - 6.0),
                max(0.5, b - 12.0),
            ])
        else:
            cands.append([0.2, 0.45, 0.7])
    starts = [[c[k % len(c)] for c in cands] for k in range(4)]
    if len(unknowns) == 2:  # 交叉组合，避免只走对角线漏掉另一支根
        c0, c1 = cands
        starts += [
            [c0[0], c1[1 % len(c1)]],
            [c0[1 % len(c0)], c1[0]],
            [c0[2 % len(c0)], c1[3 % len(c1)]],
            [c0[3 % len(c0)], c1[2 % len(c1)]],
        ]
    return starts


def _same_root(a: list[float], b: list[float], unknowns: list[_Unknown]) -> bool:
    for ua, ub, unk in zip(a, b, unknowns):
        tol = 1e-4 if unk.param == "t_adp" else 1e-5
        if abs(ua - ub) > tol:
            return False
    return True


def _root_preference_key(x: list[float], unknowns: list[_Unknown]):
    """确定性选根约定：装置露点越暖越优先；旁通系数越小越优先。"""
    return tuple(-v if unk.param == "t_adp" else v
                 for v, unk in zip(x, unknowns))


def _newton_roots(residual, unknowns: list[_Unknown],
                  starts: list[list[float]]) -> list[list[float]]:
    """阻尼牛顿 + 多初值，收集全部互异收敛根（不做取舍，取舍在调用方）。"""
    roots: list[list[float]] = []
    for x0 in starts:
        x = [_clamp(v, u.param) for v, u in zip(x0, unknowns)]
        r = residual(x)
        nrm = _inf_norm(r)
        for _ in range(NEWTON_MAX_ITER):
            if nrm <= NEWTON_TOL:
                break
            step = _solve_linear(_jacobian(residual, x, r, unknowns),
                                 [-v for v in r])
            if step is None:
                break
            accepted = False
            lam = 1.0
            for _ in range(LINE_SEARCH_MAX):
                xt = [_clamp(v + lam * d, u.param)
                      for v, d, u in zip(x, step, unknowns)]
                rt = residual(xt)
                nt = _inf_norm(rt)
                if nt < nrm:
                    x, r, nrm = xt, rt, nt
                    accepted = True
                    break
                lam *= 0.5
            if not accepted:
                break
        if nrm <= NEWTON_TOL and not any(
            _same_root(x, old, unknowns) for old in roots
        ):
            roots.append(x)
    return roots


def _inverse_pass(inlet: AirState, plans: list[_StagePlan],
                  unknowns: list[_Unknown], stage_eqs: list[_StageEq],
                  target: AirState, m_da: float) -> list[tuple[CoilResult, str]]:
    residual = _make_residual(inlet, plans, unknowns, stage_eqs, target, m_da)
    starts = _initial_starts(unknowns, target)
    roots = _newton_roots(residual, unknowns, starts)
    if not roots:
        idxs = sorted({u.pos + 1 for u in unknowns})
        where = "、".join(f"第 {i} 级" for i in idxs)
        raise PsychrometricError(
            f"链级反推不收敛：{where}的装置露点/旁通系数在 "
            f"{NEWTON_MAX_ITER} 次阻尼牛顿迭代、{len(starts)} 组初值下仍无法"
            "使全链命中目标，目标可能物理不可达；"
            "迭代中途试探的临时值不作为结果返回",
            ERR_NO_CONVERGENCE,
            stage_index=idxs[0],
        )
    # 多个孤立根按确定性约定选最暖 ADP 者（与单级反算的约定一致）
    x = min(roots, key=lambda rx: _root_preference_key(rx, unknowns))
    # 收敛后用严格前向通道重算一遍（所有领域校验全开），结果才准出
    return _forward_pass(inlet, plans, m_da, _assign_params(plans, unknowns, x))


# --------------------------------------------------------------------------
# 收口：全链合计 + 守恒校验
# --------------------------------------------------------------------------

def _close_chain(inlet: AirState, plans: list[_StagePlan],
                 solved: list[tuple[CoilResult, str]], m_da: float,
                 mode: str, target: AirState | None) -> ChainResult:
    final = solved[-1][0].outlet
    totals = breakdown_load(inlet, final, m_da)

    if totals.q_total <= ZERO_LOAD_TOL:
        raise PsychrometricError(
            "整条链没有任何一级实际冷却去湿（各级均为透传级），无效工况",
            ERR_NOT_DEHUMIDIFYING,
        )

    # 望远镜守恒：各级冷量之和必须等于首末状态算出的全链总冷量
    sum_q = sum(res.loads.q_total for res, _ in solved)
    if abs(sum_q - totals.q_total) > CONSERVE_RTOL * max(1.0, abs(totals.q_total)):
        raise PsychrometricError(
            f"链级收口校验失败：各级冷量之和 {sum_q:.9f} kW 与全链总冷量 "
            f"{totals.q_total:.9f} kW 在数值容差内不自洽",
            ERR_INCONSISTENT_STATE,
        )
    dehum = m_da * (inlet.w - final.w)
    sum_dehum = sum(m_da * (res.inlet.w - res.outlet.w) for res, _ in solved)
    if abs(sum_dehum - dehum) > CONSERVE_RTOL * max(1.0, abs(dehum)):
        raise PsychrometricError(
            f"链级收口校验失败：各级去湿量之和 {sum_dehum:.9f} kg/s 与全链"
            f"总去湿量 {dehum:.9f} kg/s 在数值容差内不自洽",
            ERR_INCONSISTENT_STATE,
        )

    stages = tuple(
        ChainStageResult(
            index=plan.index,
            name=plan.name,
            mode=mode_label,
            passthrough=res.loads.q_total <= ZERO_LOAD_TOL,
            coil=res,
            load_share=res.loads.q_total / totals.q_total,
        )
        for plan, (res, mode_label) in zip(plans, solved)
    )
    return ChainResult(mode=mode, inlet=inlet, stages=stages, totals=totals,
                       dehumidification=dehum, target=target)


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------

def solve_chain(inlet: AirState, stages: list[StageSpec], m_da: float = 1.0,
                target: ChainTarget | None = None) -> ChainResult:
    """把一条多级串联盘管链作为整体一次算完并核对。

    无全链目标：逐级正算（各级已知量必须各自给全）。
    有全链目标：按自由量数与方程数判定唯一可解 / 欠定 / 过定，
    可解则反推未给足级的装置露点与旁通系数。
    """
    if not stages:
        raise PsychrometricError(
            "空链：一条处理链至少需要一级盘管", ERR_INVALID_REQUEST
        )
    plans = [_classify(spec, pos) for pos, spec in enumerate(stages)]
    target_state = _resolve_target(inlet, target) if target is not None else None

    unknowns: list[_Unknown] = []
    stage_eqs: list[_StageEq] = []
    for plan in plans:
        if plan.kind in ("free", "t_only", "shr_only"):
            unknowns += [_Unknown(plan.pos, "t_adp"), _Unknown(plan.pos, "bf")]
        elif plan.kind == "bf_only":
            unknowns.append(_Unknown(plan.pos, "t_adp"))
        elif plan.kind == "adp_only":
            unknowns.append(_Unknown(plan.pos, "bf"))
        if plan.kind == "t_only":
            stage_eqs.append(_StageEq(plan.pos, "t", plan.spec.target_t_out_c))  # type: ignore[arg-type]
        elif plan.kind == "shr_only":
            stage_eqs.append(_StageEq(plan.pos, "shr", plan.spec.target_shr))  # type: ignore[arg-type]

    n_free = len(unknowns)
    n_eq = len(stage_eqs) + (2 if target_state is not None else 0)

    if target_state is None:
        if n_free:
            first = min(u.pos for u in unknowns) + 1
            raise PsychrometricError(
                f"存在已知量未给足的级（第 {first} 级等，缺装置露点/旁通系数），"
                "又未给全链目标可供反推，整条链欠定，拒绝猜测解",
                ERR_CHAIN_UNDERDETERMINED,
                stage_index=first,
            )
        solved = _forward_pass(inlet, plans, m_da)
        mode = "forward"
    elif n_free > n_eq:
        first = min(u.pos for u in unknowns) + 1
        raise PsychrometricError(
            f"整条链欠定：各级共有 {n_free} 个自由量（装置露点/旁通系数），"
            f"级目标与全链目标只提供 {n_eq} 个方程，凑不出唯一解，拒绝猜测解",
            ERR_CHAIN_UNDERDETERMINED,
            stage_index=first,
        )
    elif n_free == 0 and not stage_eqs:
        # 各级已给全 + 全链目标：正算后核验目标一致性（过定核验）
        solved = _forward_pass(inlet, plans, m_da)
        final = solved[-1][0].outlet
        if (abs(final.t_db_c - target_state.t_db_c) > VERIFY_T_TOL
                or abs(final.w - target_state.w) > VERIFY_W_TOL):
            raise PsychrometricError(
                f"各级已知量已唯一确定全链出风（{final.t_db_c:.4f} °C、"
                f"{final.w * 1000:.4f} g/kg），与给定全链目标"
                f"（{target_state.t_db_c:.4f} °C、"
                f"{target_state.w * 1000:.4f} g/kg）矛盾，约束过定，拒绝该工况",
                ERR_CHAIN_OVERDETERMINED,
            )
        mode = "verify"
    elif n_free < n_eq:
        raise PsychrometricError(
            f"整条链过定：各级已知量加全链目标共 {n_eq} 个约束，"
            f"自由量只有 {n_free} 个，约束彼此矛盾、无论怎么取都到不了目标",
            ERR_CHAIN_OVERDETERMINED,
        )
    else:
        solved = _inverse_pass(inlet, plans, unknowns, stage_eqs,
                               target_state, m_da)
        mode = "inverse"

    return _close_chain(inlet, plans, solved, m_da, mode, target_state)
