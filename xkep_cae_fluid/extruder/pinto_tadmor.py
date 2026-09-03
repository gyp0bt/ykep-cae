"""Pinto–Tadmor 1970 型の計量部 RTD（文献モデル、真値の供給源）.

Pinto, G. & Tadmor, Z. (1970) Polym. Eng. Sci. 10, 279. Tadmor & Gogos
*Principles of Polymer Processing* §7 に再掲。仮定: 無限幅平板（側壁無し）、
等温ニュートン、漏れ無し、フライトでの折返しは瞬時。ξ = y/H（0 根元、1 バレル）、
r = Q_p/Q_d ∈ (−1, 0]。

    下流  ŵ(ξ) = w/V_z = ξ + 3r·ξ(1−ξ)
    横断  û(ξ) = u/V_x = 3ξ² − 2ξ      （正味流量 0、ξ = 2/3 で符号反転）

上層 ξ ∈ (2/3, 1) の粒子はフライトで折り返し、横断流束の保存
∫_ξ^1 û = −∫_0^{ξ_c} û ⇔ g(ξ) = g(ξ_c), g(s) = s²(1−s) で決まる下層 ξ_c に移る。
1 周の時間重み（横断 1 回 ∝ 1/|û|）で平均した下流速度と滞留時間は

    w̄/V_z = (ŵ/û + ŵ_c/|û_c|) / (1/û + 1/|û_c|),   t = L/w̄

**普遍性の機構。** 3ξ(1−ξ) = ξ − (3ξ²−2ξ)、つまり圧力流れ分布は「引きずり分布 −
横断分布」に恒等的に等しい。横断速度は閉じた流線 1 周で変位ゼロなので周平均が
消え、どの流線でも w̄ = (1+r)·w̄_drag、流線対の流量も (1+r) 倍。t も t̄ も同じ因子で
割られて F(t/t̄) は r に依らない。t̄ = HWL/Q は流管の体積÷流束として厳密。

**数値評価は下層 ξ_c で標本化する。** 上層 ξ で標本化すると ξ→1 の粒子が根元
ξ_c ~ √(1−ξ) に張り付いて t ~ 1/√(1−ξ) と発散し、中点則が O(1/√n) にしか
収束しない（実測 1000→16000 点で 0.37%→0.09%）。下層 ξ_c ∈ (0, 2/3) の中点で
標本化し、流線対の流量 dQ = [ŵ(ξ_c) + ŵ(ξ)·|û_c|/û] dξ_c を重みにすると
被積分関数が滑らかになり、1000 点で t̄ が 1e-7、普遍性が 1e-6 で成立する。
r < −1/3 では根元付近の ŵ が負（逆流）になるが、流線対の正味流量は
(1+r)×引きずり対流量 > 0 なので重みは常に正。

t_min/t̄ = 3/4 は厳密値。最速の粒子はバレル面ではなく再循環の停留高さ ξ = 2/3 に
居て一度も折り返さない粒子で、t_min = L/(⅔V_z) = ¾·(2L/V_z)。

このモジュールは「真値の供給源」であり、ソルバーからは参照されない。
検証テスト（ゲート G5）とレポートだけが使う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

XI_SPLIT = 2.0 / 3.0
"""横断速度が符号反転する高さ。再循環の停留線."""

_BISECT_ITERS = 60
"""二分法の反復数。区間幅 1/3 × 2⁻⁶⁰ で倍精度に届く."""


@dataclass(frozen=True)
class PintoTadmorRTD:
    """縮約 RTD 曲線と代表値（すべて t̄ = HWL/Q で規格化）.

    Parameters
    ----------
    t_over_tbar : np.ndarray
        滞留時間 / 平均滞留時間、昇順
    F : np.ndarray
        累積分布（区間中点の累積割合、`weighted_quantile` と同じ流儀）
    t_min_ratio, t_p10_ratio, t_p50_ratio, t_p90_ratio : float
        最短・10/50/90 パーセンタイル（t̄ 規格化）
    tbar_over_L_Vz : float
        t̄·V_z/L。厳密値は 2/(1+r)
    r : float
        背圧比 Q_p/Q_d
    """

    t_over_tbar: np.ndarray
    F: np.ndarray
    t_min_ratio: float
    t_p10_ratio: float
    t_p50_ratio: float
    t_p90_ratio: float
    tbar_over_L_Vz: float
    r: float


def _g(s: np.ndarray) -> np.ndarray:
    """横断流束の積分 g(s) = s²(1−s)。(0, 2/3) で増加、(2/3, 1) で減少."""
    return s * s * (1.0 - s)


def _upper_partner(xi_lower: np.ndarray) -> np.ndarray:
    """g(ξ) = g(ξ_c) を満たす上層の高さ ξ ∈ (2/3, 1) を二分法で解く（ベクトル化）."""
    target = _g(xi_lower)
    lo = np.full_like(xi_lower, XI_SPLIT)
    hi = np.ones_like(xi_lower)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        above = _g(mid) > target  # g は (2/3, 1) で単調減少
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def pinto_tadmor_rtd(r: float = 0.0, n_xi: int = 4000) -> PintoTadmorRTD:
    """縮約 RTD 曲線 F(t/t̄) を返す.

    Parameters
    ----------
    r : float
        背圧比 Q_p/Q_d ∈ (−1, 0]。0 = 純引きずり。−1 は閉塞（Q = 0、t̄ 発散）で不可
    n_xi : int
        下層 ξ_c の標本数（中点則）
    """
    if not (-1.0 < r <= 0.0):
        msg = f"背圧比 Q_p/Q_d は (−1, 0] が必要: {r}"
        raise ValueError(msg)
    if n_xi < 16:
        msg = f"n_xi は 16 以上が必要: {n_xi}"
        raise ValueError(msg)

    edges = np.linspace(0.0, XI_SPLIT, n_xi + 1)
    xi_c = 0.5 * (edges[:-1] + edges[1:])
    d_xi = edges[1] - edges[0]
    xi = _upper_partner(xi_c)

    def w_hat(s: np.ndarray) -> np.ndarray:
        return s + 3.0 * r * s * (1.0 - s)

    u_up = 3.0 * xi * xi - 2.0 * xi  # û(ξ) > 0
    u_lo = 2.0 * xi_c - 3.0 * xi_c * xi_c  # |û(ξ_c)| > 0
    a, b = 1.0 / u_up, 1.0 / u_lo
    t = (a + b) / (w_hat(xi) * a + w_hat(xi_c) * b)  # t·V_z/L
    dq = (w_hat(xi_c) + w_hat(xi) * u_lo / u_up) * d_xi  # 流線対の流量（|dξ/dξ_c| = |û_c|/û）

    tbar = float(np.sum(t * dq) / np.sum(dq))
    order = np.argsort(t)
    t_red = t[order] / tbar
    q = dq[order]
    F = (np.cumsum(q) - 0.5 * q) / q.sum()
    p10, p50, p90 = np.interp([0.1, 0.5, 0.9], F, t_red)
    return PintoTadmorRTD(
        t_over_tbar=t_red,
        F=F,
        t_min_ratio=float(t_red[0]),
        t_p10_ratio=float(p10),
        t_p50_ratio=float(p50),
        t_p90_ratio=float(p90),
        tbar_over_L_Vz=tbar,
        r=float(r),
    )
