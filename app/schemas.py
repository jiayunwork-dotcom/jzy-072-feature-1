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


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """统一结构化错误信封。"""

    error: ErrorDetail
