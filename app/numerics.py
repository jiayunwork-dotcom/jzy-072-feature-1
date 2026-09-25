"""ADP 反算用到的通用数值工具（二分）。

不引入 numpy，全部用标准库实现，方便 python:3.12-slim 直接跑。
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import PsychrometricError, ERR_NO_CONVERGENCE

#: 迭代默认上限（二分 80 次可达 2^-80 量级，正常 40~60 次就收敛）
DEFAULT_MAX_ITER = 80


def bisection(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-8,
    max_iter: int = DEFAULT_MAX_ITER,
    what: str = "迭代",
) -> float:
    """在 ``[lo, hi]`` 上用二分法求 f(x)=0。

    端点处为 0（根在边界）也合法直接返回。要求 ``f(lo)`` 与 ``f(hi)``
    异号或其一为 0，同号则抛 :class:`PsychrometricError`。

    Parameters
    ----------
    tol:
        区间宽度收敛阈值（温度类问题即 °C）。
    what:
        错误信息里描述迭代对象的中文名。
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise PsychrometricError(
            f"{what}不收敛：搜索区间 [{lo:.4f}, {hi:.4f}] 两端残差同号 "
            f"({flo:.3e}, {fhi:.3e})，区间内无根",
            ERR_NO_CONVERGENCE,
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or 0.5 * (hi - lo) < tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    raise PsychrometricError(
        f"{what}不收敛：{max_iter} 次迭代后区间宽度仍大于 {tol:g}",
        ERR_NO_CONVERGENCE,
    )


def scan_sign_change(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    n: int = 240,
) -> tuple[float, float] | None:
    """在 ``[lo, hi]`` 上等分扫描，返回第一个残差号变（或命中 0）的子区间。

    用于残差函数非单调、无法预先给二分括号的场合。端点恰为 0 时返回
    零宽区间。找不到返回 ``None``，由调用方决定如何报错。
    """
    x0 = lo
    f0 = f(x0)
    if f0 == 0.0:
        return x0, x0
    for i in range(1, n + 1):
        x1 = lo + (hi - lo) * i / n
        f1 = f(x1)
        if f1 == 0.0:
            return x1, x1
        if f0 * f1 < 0.0:
            return x0, x1
        x0, f0 = x1, f1
    return None
