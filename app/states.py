"""湿空气状态展开模块。

对外只暴露：

* :class:`AirState` —— 统一状态表示（干球温度、气压、含湿量，
  以及按需缓存的 RH/露点/焓/湿球等导出量）；
* :func:`expand_state` —— 三种湿度表示（RH / 露点 / 含湿量）任选其一，
  展开到统一的 :class:`AirState`；
* :func:`saturated_state` —— 取某温度下的饱和状态。

基础公式全部来自 :mod:`app.psychrometrics`，本模块不重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import psychrometrics as psy
from .errors import PsychrometricError, ERR_INVALID_STATE


@dataclass(frozen=True)
class AirState:
    """湿空气状态（以 1 kg 干空气为基准）。

    两个基本量是干球温度和含湿量，其余全部由它们派生。
    """

    t_db_c: float
    """干球温度 °C"""
    w: float
    """含湿量 kg(水)/kg(干空气)"""
    p_pa: float
    """大气压 Pa"""

    # ---- 导出量（frozen dataclass 下缓存写入 __dict__）-------------------

    @property
    def enthalpy(self) -> float:
        """比焓 kJ/kg干空气。"""
        return psy.enthalpy(self.t_db_c, self.w)

    @property
    def relative_humidity(self) -> float:
        """相对湿度（0~1）；状态超饱和时由展开入口提前拒绝。"""
        return psy.relative_humidity_from_w(self.w, self.t_db_c, self.p_pa)

    @property
    def dewpoint_c(self) -> float:
        """露点温度 °C。"""
        return psy.dewpoint_from_humidity_ratio(self.w, self.p_pa)

    @property
    def wet_bulb_c(self) -> float:
        """热力学湿球温度 °C（惰性求值并缓存）。"""
        cache = self.__dict__.get("_wb_cache")
        if cache is None:
            cache = psy.wet_bulb_temperature(self.t_db_c, self.w, self.p_pa)
            self.__dict__["_wb_cache"] = cache
        return cache

    def to_dict(self) -> dict[str, float]:
        """出参用：全部状态量（含湿量以 kg/kg 与 g/kg 两种尺度给出）。"""
        return {
            "t_db_c": self.t_db_c,
            "w": self.w,
            "w_g_per_kg": self.w * 1000.0,
            "rh": self.relative_humidity,
            "t_dp_c": self.dewpoint_c,
            "t_wb_c": self.wet_bulb_c,
            "enthalpy": self.enthalpy,
            "p_pa": self.p_pa,
        }


def expand_state(
    t_db_c: float,
    p_pa: float,
    *,
    rh: float | None = None,
    t_dp_c: float | None = None,
    w: float | None = None,
) -> AirState:
    """进口/出口状态的统一展开入口。

    湿度三种表示必须恰好给一种；含湿量若超过该温度气压下的饱和上限
    （反算 RH > 1），直接抛领域错误，绝不当正常状态放行。
    """
    given = [name for name, val in (("rh", rh), ("t_dp_c", t_dp_c), ("w", w))
             if val is not None]
    if len(given) != 1:
        raise PsychrometricError(
            "湿度必须且只能用相对湿度 rh、露点 t_dp_c、含湿量 w 中的一种表示"
            + (f"，当前给出: {', '.join(given)}" if given else "，当前一个都没给"),
            ERR_INVALID_STATE,
        )

    if not (psy.T_MIN_C <= float(t_db_c) <= psy.T_MAX_C):
        raise PsychrometricError(
            f"干球温度 {t_db_c} °C 超出允许范围 "
            f"[{psy.T_MIN_C}, {psy.T_MAX_C}] °C",
            ERR_INVALID_STATE,
        )
    if not (psy.P_MIN <= float(p_pa) <= psy.P_MAX):
        raise PsychrometricError(
            f"大气压 {p_pa} Pa 超出允许范围 [{psy.P_MIN:.0f}, {psy.P_MAX:.0f}] Pa",
            ERR_INVALID_STATE,
        )

    if rh is not None:
        w_val = psy.humidity_ratio_from_rh(rh, t_db_c, p_pa)
    elif t_dp_c is not None:
        if t_dp_c > t_db_c + 1e-9:
            raise PsychrometricError(
                f"露点温度 {t_dp_c:.2f} °C 高于干球温度 {t_db_c:.2f} °C，"
                "状态不成立（冷却到露点以上不会结露）",
                ERR_INVALID_STATE,
            )
        w_val = psy.humidity_ratio_from_dewpoint(t_dp_c, p_pa)
    else:
        w_val = float(w)  # type: ignore[arg-type]

    # 统一的超饱和闸门：无论从哪条路进来都要过这一关
    psy.relative_humidity_from_w(w_val, t_db_c, p_pa)
    return AirState(t_db_c=float(t_db_c), w=w_val, p_pa=float(p_pa))


def saturated_state(t_c: float, p_pa: float) -> AirState:
    """温度 t、气压 P 下的饱和状态（RH = 1，即 ADP 表面那部分空气）。"""
    w_sat = psy.humidity_ratio_sat(t_c, p_pa)
    return AirState(t_db_c=float(t_c), w=w_sat, p_pa=float(p_pa))


def with_temperature_w(state: AirState, t_db_c: float, w: float) -> AirState:
    """同气压下换一组 (t, W) 构造新状态（内部加权用）。"""
    return AirState(t_db_c=float(t_db_c), w=float(w), p_pa=state.p_pa)
