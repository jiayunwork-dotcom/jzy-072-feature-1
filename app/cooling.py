"""冷量分解模块：总冷量、显热量、潜热量与显热比 SHR。

只做进出口状态 + 干空气质量流量的算术分解，不涉及 ADP 迭代。
显热按干空气比热与进出口温差计算，潜热 = 总冷量 − 显热，
三者对同一组进出口状态必须自洽。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import psychrometrics as psy
from .errors import PsychrometricError, ERR_INVALID_SHR, ERR_INVALID_STATE
from .states import AirState

#: SHR 允许区间：(0, 1]。0 不接受（纯潜热、无显热冷却的盘管工况不纳入）。
SHR_LOW_TOL = 1e-9
SHR_HIGH_TOL = 1e-6


@dataclass(frozen=True)
class LoadBreakdown:
    """冷量分解结果（功率单位 kW）。"""

    q_total: float
    """总冷量 = m_da·(h_in − h_out)，kW"""
    q_sensible: float
    """显热量 = m_da·(1.006 + 1.86·W_out)·(t_in − t_out)，kW"""
    q_latent: float
    """潜热量 = 总冷量 − 显热量，kW"""
    shr: float | None
    """显热比 = 显热 / 总冷量；总冷量为 0（无换热）时为 None"""
    m_da: float
    """干空气质量流量 kg/s"""


def breakdown_load(inlet: AirState, outlet: AirState, m_da: float = 1.0) -> LoadBreakdown:
    """对给定进出口状态做冷量分解。

    自洽性校验：

    * 气压必须一致；
    * 必须确为冷却工况（h_out <= h_in），否则报错；
    * 显热量不得超过总冷量（数值噪声容差内）；
    * SHR 必须落在 (0, 1]；总冷量为 0 时 SHR 记为 None
      （调用方如需把“零冷量”当非法，应在盘管求解处拒绝）。
    """
    if inlet.p_pa != outlet.p_pa:
        raise PsychrometricError(
            f"进出口气压不一致: {inlet.p_pa:.0f} Pa vs {outlet.p_pa:.0f} Pa",
            ERR_INVALID_STATE,
        )
    if m_da <= 0.0:
        raise PsychrometricError(
            f"干空气质量流量必须为正，当前 m_da={m_da!r}", ERR_INVALID_STATE
        )

    q_total = m_da * (inlet.enthalpy - outlet.enthalpy)
    # 湿空气显热比热按出口含湿量取（干空气比热 + 水蒸气比热·W），
    # 温差则严格用进出口干球温度。
    cp_moist = psy.CP_AIR + psy.CP_VAPOR * outlet.w
    q_sensible = m_da * cp_moist * (inlet.t_db_c - outlet.t_db_c)
    q_latent = q_total - q_sensible

    if q_total < -1e-9:
        raise PsychrometricError(
            f"出口焓 {outlet.enthalpy:.3f} 高于进口焓 {inlet.enthalpy:.3f}，"
            "这不是冷却工况，无法分解冷量",
            ERR_INVALID_SHR,
        )
    if q_sensible > q_total + 1e-6 * max(1.0, abs(q_total)):
        raise PsychrometricError(
            f"显热量 {q_sensible:.6f} kW 大于总冷量 {q_total:.6f} kW，"
            "潜热量为负，SHR > 1，进出口状态不构成合法冷却盘管工况",
            ERR_INVALID_SHR,
        )

    shr: float | None
    if q_total <= 1e-9:
        # 无换热：SHR 无定义
        shr = None
    else:
        shr = q_sensible / q_total
        # 允许 SHR 因舍入极轻微越过 1
        if 1.0 < shr <= 1.0 + SHR_HIGH_TOL:
            shr = 1.0
        if not (SHR_LOW_TOL < shr <= 1.0):
            raise PsychrometricError(
                f"显热比 SHR={shr:.6f} 不在 (0, 1]，拒绝该工况",
                ERR_INVALID_SHR,
            )
        if q_latent < -1e-6 * max(1.0, abs(q_total)):
            raise PsychrometricError(
                f"潜热量 {q_latent:.6f} kW 为负，SHR 分解不自洽",
                ERR_INVALID_SHR,
            )

    return LoadBreakdown(
        q_total=q_total,
        q_sensible=q_sensible,
        q_latent=q_latent,
        shr=shr,
        m_da=m_da,
    )
