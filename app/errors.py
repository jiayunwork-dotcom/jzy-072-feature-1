"""冷却盘管核算服务的错误类型。

所有领域错误都抛出 :class:`PsychrometricError`，HTTP 层统一转成结构化 422 响应。
不同 ``code`` 便于调用方按错误类别处理。
"""

from __future__ import annotations


class PsychrometricError(ValueError):
    """湿空气/盘管计算领域错误。

    Parameters
    ----------
    message:
        人类可读的错误说明（中文）。
    code:
        机器可读的错误类别，取值见模块级常量。
    stage:
        多级串联链中出问题的级次（1 起）；非链场景为 ``None``。
    fog:
        是否为“旁通混合弦穿入饱和雾区”类拒绝。链反推的残差求值靠它
        区分“试探点太冷落入雾区”（按过干处理继续二分）与“定值装置
        露点本身不合法”（必须原样抛出）。
    """

    def __init__(self, message: str, code: str = "invalid_state", *,
                 stage: int | None = None, fog: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.stage = stage
        self.fog = fog


# ---- 错误类别常量 ---------------------------------------------------------

ERR_INVALID_STATE = "invalid_state"
"""状态本身不成立（含湿量超饱和、温度越界、湿度表示互相矛盾等）。"""

ERR_INVALID_REQUEST = "invalid_request"
"""请求组合不合法（已知量不足/互相冲突）。"""

ERR_NOT_DEHUMIDIFYING = "not_dehumidifying"
"""给定条件不构成冷却去湿工况（ADP 不低于进口露点/出风比进风还湿等）。"""

ERR_INVALID_BF = "invalid_bypass_factor"
"""旁通系数不在 [0, 1]。"""

ERR_INVALID_SHR = "invalid_shr"
"""显热比落在 (0, 1] 之外，或显热/潜热分解与总冷量不自洽。"""

ERR_NO_CONVERGENCE = "no_convergence"
"""ADP 数值迭代在限定步数内不收敛。"""

ERR_INCONSISTENT_STATE = "inconsistent_state"
"""给定的进出口状态无法用同一个旁通系数同时加权 W 与 h。"""

ERR_INVALID_CHAIN = "invalid_chain"
"""多级链整体不成立：空链、全链无一级做功、级间气压不一致、全链目标
欠定/过定/不可达、链级不变量（单调性、合计=分级之和）被破坏等。"""
