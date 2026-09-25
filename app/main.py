"""FastAPI 路由与校验。

三个对外接口：

* ``POST /coil``   —— 进口状态 + 已知 BF/目标出风，反算 ADP、BF、
  出口全部状态量与冷量；
* ``POST /state``  —— 只做单个湿空气状态展开，供手算核对；
* ``POST /loads``  —— 给定进出口状态，只反推 SHR 与冷量，不做 ADP 迭代；
* ``GET  /demo``   —— 内置示范工况（35 °C / 50%RH 常压进风）；
* ``GET  /``       —— 健康检查与服务说明。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import adp as adp_mod
from .cooling import breakdown_load
from .errors import PsychrometricError
from .schemas import (
    CoilOutput,
    ErrorResponse,
    LoadsOutput,
    LoadsRequest,
    CoilRequest,
    StateInput,
    StateOutput,
)
from .states import AirState, expand_state

app = FastAPI(
    title="冷却盘管选型核算服务",
    version="1.0.0",
    description=(
        "湿空气状态展开、装置露点（ADP）反算、旁通加权与冷量分解。"
        "饱和蒸汽压统一使用固定 Magnus 公式；无持久化，每次请求独立计算。"
    ),
)


# ---- 统一错误处理 ---------------------------------------------------------

@app.exception_handler(PsychrometricError)
async def psychrometric_error_handler(request: Request, exc: PsychrometricError):
    body = ErrorResponse(error={"code": exc.code, "message": exc.message})
    return JSONResponse(status_code=422, content=body.model_dump())


# ---- 小工具 ---------------------------------------------------------------

def _to_state_output(st: AirState) -> StateOutput:
    return StateOutput(**st.to_dict())


def _expand(inp: StateInput) -> AirState:
    return expand_state(
        inp.t_db_c,
        inp.p_pa,
        rh=inp.rh,
        t_dp_c=inp.t_dp_c,
        w=inp.w,
    )


def _coil_result_to_output(result: adp_mod.CoilResult, mode_label: str | None = None
                           ) -> CoilOutput:
    loads = result.loads
    return CoilOutput(
        mode=mode_label or result.mode,  # type: ignore[arg-type]
        inlet=_to_state_output(result.inlet),
        outlet=_to_state_output(result.outlet),
        adp=_to_state_output(result.adp),
        bf=result.bf,
        loads=LoadsOutput(
            q_total=loads.q_total,
            q_sensible=loads.q_sensible,
            q_latent=loads.q_latent,
            shr=loads.shr,
            m_da=loads.m_da,
        ),
    )


# ---- 路由 -----------------------------------------------------------------

@app.get("/", tags=["meta"])
def health() -> dict[str, str]:
    return {"service": "cooling-coil-service", "status": "ok"}


@app.post("/state", response_model=StateOutput, tags=["psychrometrics"])
def expand(input: StateInput) -> StateOutput:
    """只展开单个湿空气状态（三种湿度表示任选其一）。"""
    return _to_state_output(_expand(input))


@app.post("/coil", response_model=CoilOutput, tags=["coil"])
def coil(req: CoilRequest) -> CoilOutput:
    """装置露点核算：正算或反算，出口全部状态量 + 冷量分解。"""
    inlet = _expand(req.inlet)
    outlet = _expand(req.outlet) if req.outlet is not None else None

    result = adp_mod.solve_coil(
        inlet,
        bf=req.bf,
        t_adp_c=req.t_adp_c,
        target_t_out_c=req.target_t_out_c,
        target_shr=req.target_shr,
        outlet=outlet,
        m_da=req.m_da,
    )
    # 模式 3 内部复用了模式 2 的求解器，出口处修正对外模式名
    mode_label = result.mode
    if req.target_t_out_c is not None and req.target_shr is not None:
        mode_label = "target_t_shr"
    return _coil_result_to_output(result, mode_label)


@app.post("/loads", response_model=LoadsOutput, tags=["coil"])
def loads(req: LoadsRequest) -> LoadsOutput:
    """只核对报告数字：进出口状态 → 显热比与冷量，不做 ADP 迭代。"""
    inlet = _expand(req.inlet)
    outlet = _expand(req.outlet)
    lb = breakdown_load(inlet, outlet, req.m_da)
    return LoadsOutput(
        q_total=lb.q_total,
        q_sensible=lb.q_sensible,
        q_latent=lb.q_latent,
        shr=lb.shr,
        m_da=lb.m_da,
    )


@app.get("/demo", response_model=CoilOutput, tags=["coil"])
def demo(m_da: float = 1.0) -> CoilOutput:
    """示范工况：35 °C、50%RH、常压进风，ADP 7 °C，BF=0.2。

    手算量级：进口含湿量约 17.7 g/kg、焓约 80.7 kJ/kg；
    出口含湿量约 9.6 g/kg、焓约 39.5 kJ/kg；
    单位干空气总冷量约 41 kJ/kg，显热比约 0.45。
    """
    inlet = expand_state(35.0, 101325.0, rh=0.5)
    result = adp_mod.solve_direct(inlet, t_adp_c=7.0, bf=0.2, m_da=m_da)
    return _coil_result_to_output(result)
