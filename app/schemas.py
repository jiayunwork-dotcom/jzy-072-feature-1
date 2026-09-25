"""HTTP 请求/响应的 Pydantic 模型与字段校验。

领域校验（饱和、SHR 区间、模式条件不足等）放在算法层抛出
:class:`PsychrometricError`，路由层统一转成结构化错误响应。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StateInput(BaseModel):
    """湿空气状态输入：干球温度 + 气压，湿度三选一。"""

    model_config = ConfigDict(extra="forbid")

    t_db_c: float = Field(..., description="干球温度 °C")
    p_pa: float = Field(101325.0, description="大气压 Pa，缺省标准大气压")
    rh: float | None = Field(None, description="相对湿度，0~1 之间")
    t_dp_c: float | None = Field(None, description="露点温度 °C")
    w: float | None = Field(None, description="含湿量 kg(水)/kg(干空气)")


class StateOutput(BaseModel):
    t_db_c: float
    w: float
    w_g_per_kg: float
    rh: float
    t_dp_c: float
    t_wb_c: float
    enthalpy: float
    p_pa: float


class CoilRequest(BaseModel):
    """盘管核算请求。

    进口状态必填；再按求解模式给出下列任意一组已知量：

    * ``t_adp_c`` + ``bf``：正算；
    * ``outlet``：完整出风状态，反解 ADP/BF；
    * ``target_t_out_c`` + ``target_shr``；
    * ``bf`` + ``target_t_out_c``；
    * ``bf`` + ``target_shr``。

    只给目标出风温度、或只给目标 SHR 属于欠定，返回 422。
    """

    model_config = ConfigDict(extra="forbid")

    inlet: StateInput
    outlet: StateInput | None = Field(None, description="完整出风状态（模式2）")
    t_adp_c: float | None = Field(None, description="装置露点温度 °C（正算）")
    bf: float | None = Field(None, description="旁通系数，0~1")
    target_t_out_c: float | None = Field(None, description="目标出风干球温度 °C")
    target_shr: float | None = Field(None, description="目标显热比，(0,1]")
    m_da: float = Field(1.0, gt=0.0, description="干空气质量流量 kg/s")


class LoadsOutput(BaseModel):
    q_total: float = Field(..., description="总冷量 kW")
    q_sensible: float = Field(..., description="显热量 kW")
    q_latent: float = Field(..., description="潜热量 kW")
    shr: float | None = Field(..., description="显热比 (0,1]；零冷量时为 null")
    m_da: float


class CoilOutput(BaseModel):
    mode: Literal["direct", "from_outlet", "bf_target_t", "bf_shr", "target_t_shr"]
    inlet: StateOutput
    outlet: StateOutput
    adp: StateOutput
    bf: float
    loads: LoadsOutput


class LoadsRequest(BaseModel):
    """只核对选型报告：进出口状态 → SHR 与冷量，不做 ADP 迭代。"""

    model_config = ConfigDict(extra="forbid")

    inlet: StateInput
    outlet: StateInput
    m_da: float = Field(1.0, gt=0.0, description="干空气质量流量 kg/s")


# ---- 多级串联盘管链 ---------------------------------------------------------

class ChainStageInput(BaseModel):
    """链中一级的已知量。

    每级支持：``t_adp_c``+``bf`` 正算 / ``outlet`` 完整出风 /
    ``target_t_out_c``+``target_shr`` / ``bf``+``target_t_out_c`` /
    ``bf``+``target_shr`` / 仅 ``bf``（=1 为纯透传级，否则待全链反推 ADP）/
    仅 ``t_adp_c``（待全链反推 BF）/ 全不给（待全链反推）。
    """

    model_config = ConfigDict(extra="forbid")

    t_adp_c: float | None = Field(None, description="本级装置露点温度 °C")
    bf: float | None = Field(None, description="本级旁通系数 [0,1]；单独给 1.0 表示纯透传级")
    target_t_out_c: float | None = Field(None, description="本级目标出风干球温度 °C")
    target_shr: float | None = Field(None, description="本级目标显热比 (0,1]")
    outlet: StateInput | None = Field(None, description="本级完整目标出风状态")


class ChainRequest(BaseModel):
    """多级串联盘管链核算请求。

    不给全链目标 → 逐级正算（每级已知量必须自足）；
    给 ``target_outlet`` 或 ``target_t_out_c``+``target_shr`` → 全链目标反推，
    把未定级的装置露点/旁通系数解出来使全链恰好落到目标。
    """

    model_config = ConfigDict(extra="forbid")

    inlet: StateInput
    stages: list[ChainStageInput] = Field(..., description="按风向顺序的各级，至少一级")
    target_outlet: StateInput | None = Field(None, description="全链最终目标出风状态（反推模式）")
    target_t_out_c: float | None = Field(None, description="全链最终目标出风温度 °C（须与 target_shr 同给）")
    target_shr: float | None = Field(None, description="全链总体目标显热比 (0,1]（须与 target_t_out_c 同给）")
    m_da: float = Field(1.0, gt=0.0, description="干空气质量流量 kg/s（全链共用）")


class ChainStageOutput(BaseModel):
    index: int = Field(..., description="级次，1 起")
    mode: str = Field(..., description="本级求解模式；纯透传级为 bypass")
    inlet: StateOutput
    outlet: StateOutput
    adp: StateOutput | None = Field(..., description="装置露点状态；纯透传级为 null")
    bf: float
    loads: LoadsOutput
    q_fraction: float = Field(..., description="本级冷量占全链总冷量的份额")


class ChainSummaryOutput(BaseModel):
    inlet: StateOutput
    outlet: StateOutput
    q_total: float = Field(..., description="全链总冷量 kW（= 各级之和）")
    q_sensible: float = Field(..., description="全链总显热 kW")
    q_latent: float = Field(..., description="全链总潜热 kW")
    shr: float | None = Field(..., description="全链总体显热比")
    delta_w: float = Field(..., description="全链总去湿量 kg/kg（= 各级之和）")
    delta_w_g_per_kg: float = Field(..., description="全链总去湿量 g/kg")
    m_da: float


class ChainOutput(BaseModel):
    mode: Literal["forward", "inverse_target"]
    stages: list[ChainStageOutput]
    summary: ChainSummaryOutput


class ErrorDetail(BaseModel):
    code: str
    message: str
    stage: int | None = Field(None, description="多级链中出问题的级次（1 起）；非链场景缺省")


class ErrorResponse(BaseModel):
    """统一结构化错误信封。"""

    error: ErrorDetail
