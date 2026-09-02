"""展開チャネル断面の格子生成 Process.

x 方向: フライトを中央 [W_t/2 - e/2, W_t/2 + e/2] に置き、チャネル部を
        周期境界（x=0 / x=W_t）で分割する。フライト側に周期境界を置くと
        両端の断面形状が一致せず周期条件が破綻するため。
y 方向: 隙間 delta に n_gap セルを等間隔で入れ、その下のバルクを上ほど細かい
        等比格子で埋める。delta=0 のときはバルクのみ（等間隔になる）。

隣接セル幅比は 1.3 以下に抑える。急な格子変化は 2 次精度を壊すため。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.extruder.data import ChannelGrid, ScrewSpec

MAX_CELL_RATIO = 1.3
"""隣接セル幅比の上限。これを超える等比格子は先頭セル幅を粗くして避ける."""


def _geometric_widths(
    L: float, n: int, w_first: float, max_ratio: float = MAX_CELL_RATIO
) -> np.ndarray:
    """先頭セル幅 w_first から等比で伸ばし、合計が L になる幅配列を返す.

    公比 r は Σ w_first·r^k = L を満たすものを二分法で求める。
    r が max_ratio を超える場合は w_first を「r = max_ratio となる値」まで
    粗くして、格子の急変を防ぐ。

    Parameters
    ----------
    L : float
        方向の全長 [m]
    n : int
        セル数
    w_first : float
        希望する先頭セル幅 [m]
    max_ratio : float
        隣接セル幅比の上限

    Returns
    -------
    np.ndarray
        セル幅配列 (n,)。単調増加。合計は厳密に L
    """
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.array([L])

    # r = max_ratio で許される最小の先頭幅。これより細かくは切らない
    w_min = L * (max_ratio - 1.0) / (max_ratio**n - 1.0)
    w_first = max(w_first, w_min)

    if w_first * n >= L:
        # 先頭幅が大きすぎる → 等間隔に落とす
        return np.full(n, L / n)

    def total(r: float) -> float:
        if abs(r - 1.0) < 1e-14:
            return w_first * n
        return w_first * (r**n - 1.0) / (r - 1.0)

    lo, hi = 1.0, 2.0
    while total(hi) < L:
        hi *= 2.0
        if hi > 1.0e6:
            msg = f"等比公比が発散: L={L}, n={n}, w_first={w_first}"
            raise ValueError(msg)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < L:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    w = w_first * r ** np.arange(n, dtype=np.float64)
    return w / w.sum() * L


class ScrewGeometryProcess(PreProcess["ScrewSpec", "ChannelGrid"]):
    """スクリュー諸元 → 展開チャネル断面の不等間隔格子 + 固体マスク.

    隙間 delta は 0.1mm 級、チャネル深さ H は 4mm 級で 40:1 の寸法比になる。
    1a/a02 のボクセルメッシュ品質ベンチの結論（誤差は最狭方向のセル数だけで
    決まる。1% なら 20 セル、0.1% なら 63 セル）に従い、隙間方向に n_gap セルを
    確保する。断面 2D なので隙間に 5μm セルを入れても総セル数は現実的に収まる。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ScrewGeometry",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [StructuredMeshProcess]

    def process(self, input_data: ScrewSpec) -> ChannelGrid:
        """諸元から断面格子・固体マスク・MeshData を作る."""
        s = input_data
        if s.delta < 0.0 or s.delta >= s.H:
            msg = f"隙間 delta は 0 <= delta < H が必要: delta={s.delta}, H={s.H}"
            raise ValueError(msg)
        if s.e < 0.0 or s.e >= s.W_t:
            msg = f"フライト幅 e は 0 <= e < W_t が必要: e={s.e}, W_t={s.W_t}"
            raise ValueError(msg)

        dx = self._build_dx(s)
        dy = self._build_dy(s)
        xc = np.cumsum(dx) - dx / 2.0
        yc = np.cumsum(dy) - dy / 2.0

        x_lo = 0.5 * (s.W_t - s.e)
        x_hi = 0.5 * (s.W_t + s.e)
        y_top = s.H - s.delta
        solid = (xc[:, None] > x_lo) & (xc[:, None] < x_hi) & (yc[None, :] < y_top)

        mesh_res = StructuredMeshProcess().process(
            StructuredMeshInput(
                Lx=s.W_t,
                Ly=s.H,
                Lz=1.0,
                nx=int(dx.shape[0]),
                ny=int(dy.shape[0]),
                nz=1,
                stretch_x=tuple(dx / dx.sum()),
                stretch_y=tuple(dy / dy.sum()),
            )
        )

        return ChannelGrid(dx=dx, dy=dy, xc=xc, yc=yc, solid=solid, spec=s, mesh=mesh_res.mesh)

    @staticmethod
    def _build_dx(s: ScrewSpec) -> np.ndarray:
        """x 方向: 半チャネル / ランド / 半チャネル。フライト角付近を細かくする.

        e = 0（フライト無し）は平行平板の 1D 検証用の正式なケースで、
        全幅を等間隔で切る。ランド区間を作ると幅ゼロのセルが出てしまう。
        """
        if s.e <= 0.0:
            n = max(2, s.nx_channel)
            return np.full(n, s.W_t / n)

        half = 0.5 * (s.W_t - s.e)
        n_half = max(2, s.nx_channel // 2)
        n_land = max(1, s.nx_land)
        w_fine = min(s.e / n_land, half / n_half)
        left = _geometric_widths(half, n_half, w_fine)[::-1]
        land = np.full(n_land, s.e / n_land)
        right = _geometric_widths(half, n_half, w_fine)
        return np.concatenate([left, land, right])

    @staticmethod
    def _build_dy(s: ScrewSpec) -> np.ndarray:
        """y 方向: バルク（下、上ほど細かい）+ 隙間（等間隔 n_gap セル）."""
        if s.delta <= 0.0 or s.n_gap <= 0:
            return _geometric_widths(s.H, s.ny_bulk, s.H / s.ny_bulk)[::-1]
        gap = np.full(s.n_gap, s.delta / s.n_gap)
        bulk = _geometric_widths(s.H - s.delta, s.ny_bulk, s.delta / s.n_gap)[::-1]
        return np.concatenate([bulk, gap])
