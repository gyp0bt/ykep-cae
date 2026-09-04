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

from xkep_cae_fluid.extruder.rtd import weighted_ecdf, weighted_quantile

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


@dataclass(frozen=True)
class RTDComparison:
    """ykep の追跡結果を文献曲線と同じ規格化で並べた比較.

    Parameters
    ----------
    t_over_tbar : np.ndarray
        ykep の滞留時間 / t_ref、昇順（脱出粒子のみ）
    F : np.ndarray
        その重み付き ECDF（区間中点）
    p10_ratio, p50_ratio, p90_ratio : float
        ykep の分位点 / t_ref
    dev_p10, dev_p50, dev_p90 : float
        \\|ykep 分位点 ÷ 文献分位点 − 1\\|
    curve_l1 : float
        分位関数の相対偏差 \\|t_ykep(F)/t_PT(F) − 1\\| を F ∈ f_range で平均したもの。
        **横方向（t 軸）で比べる。** 文献曲線は t_min 直上で F が 0 → 0.1 まで
        ほぼ垂直に立つので、縦方向 \\|F_ykep − F_PT\\| は t の 0.2% のずれで 0.1 を
        超えてしまい指標にならない。
    curve_max : float
        同じ偏差の最大値。種まきが 1 セル 1 粒子なので裾側（F > 0.6）では
        滞留時間が種の行ごとに束になり ECDF が階段状になる。最大値はその階段幅を
        拾うので判定には平均（curve_l1）を使い、最大値は観察として残す。
    f_range : tuple[float, float]
        曲線偏差を評価した F の範囲
    """

    t_over_tbar: np.ndarray
    F: np.ndarray
    p10_ratio: float
    p50_ratio: float
    p90_ratio: float
    dev_p10: float
    dev_p50: float
    dev_p90: float
    curve_l1: float
    curve_max: float
    f_range: tuple[float, float]


def compare_rtd(
    t_res: np.ndarray,
    weight: np.ndarray,
    t_ref: float,
    pt: PintoTadmorRTD,
    *,
    f_range: tuple[float, float] = (0.05, 0.9),
    n_f: int = 400,
) -> RTDComparison:
    """追跡結果 (t_res, weight) を t_ref で規格化して文献曲線 pt と比べる.

    t_ref には「側壁の無い」平均滞留時間 t̄_∞ = t̄_theory·F_d を渡す。
    側壁は流量を F_d 倍に減らして t̄_theory = V/Q を 1/F_d 倍に延ばすが、
    分位点を担う溝中央の流線は側壁を知らないので、絶対時間は文献の
    t̄_∞ = HWL/Q_∞ に対して決まる。H/W → 0 で F_d → 1 なので極限の主張は
    どちらの規格化でも同じだが、F_d を掛けた方が側壁の一次効果を先に除ける。
    """
    if t_ref <= 0.0:
        msg = f"t_ref は正が必要: {t_ref}"
        raise ValueError(msg)
    lo, hi = f_range
    if not (0.0 < lo < hi < 1.0):
        msg = f"f_range は 0 < lo < hi < 1 が必要: {f_range}"
        raise ValueError(msg)
    t_red = np.asarray(t_res, dtype=float) / t_ref
    v, F = weighted_ecdf(t_red, np.asarray(weight, dtype=float))
    p10, p50, p90 = (float(x) for x in weighted_quantile(t_red, weight, [0.1, 0.5, 0.9]))
    f_grid = np.linspace(lo, hi, n_f)
    dev = np.abs(np.interp(f_grid, F, v) / np.interp(f_grid, pt.F, pt.t_over_tbar) - 1.0)
    return RTDComparison(
        t_over_tbar=v,
        F=F,
        p10_ratio=p10,
        p50_ratio=p50,
        p90_ratio=p90,
        dev_p10=abs(p10 / pt.t_p10_ratio - 1.0),
        dev_p50=abs(p50 / pt.t_p50_ratio - 1.0),
        dev_p90=abs(p90 / pt.t_p90_ratio - 1.0),
        curve_l1=float(dev.mean()),
        curve_max=float(dev.max()),
        f_range=(lo, hi),
    )
