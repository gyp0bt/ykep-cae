"""色分け有限差分による厳密ヤコビアン（構造格子、半径 r のボックス型ステンシル）.

前処理行列に使う 1 次風上ヤコビアン J1（`BrinkmanDiscretization.jacobian_first_order`）は、
残差の 2 次風上 + Venkatakrishnan と RC 係数 d_f = V/a_P の速度依存を線形化していない。
高 Re（a_P が対流支配）ではこの差が速度列で O(1) になり、J1 を近似逆にした前処理が
真の作用素（JFNK の有限差分 matvec）と食い違って GMRES が壊れる（status-45）。

ここでは残差関数そのものを色分けして差分し、真の作用素と整合するヤコビアンを組む。
セル (i, j) の列は半径 r のボックス内のセル（u, v, p 全部）にしか非零を持たないので、
色 = var·a² + (i mod a)·a + (j mod a)、a = 2r + 1 とすれば同色の列は行を共有しない。
3a² 回（r=3 で 147 回）の残差評価で全列が求まる。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse


def colored_fd_jacobian(
    resid: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    nx: int,
    ny: int,
    radius: int = 3,
    r0: np.ndarray | None = None,
    eps_rel: float = 1.0e-7,
) -> sparse.csr_matrix:
    """J = ∂resid/∂x を前進差分で組む（3N×3N、ブロック順 [u, v, p]）.

    Parameters
    ----------
    resid : callable
        x (3N,) -> R (3N,)
    x : np.ndarray
        評価点
    nx, ny : int
        格子分割（セル k = i·ny + j）
    radius : int
        1 列が影響する行の最大セル距離（チェビシェフ距離）。2 次風上 + リミター + RC は 3
    r0 : np.ndarray | None
        resid(x) を既に持っていれば渡す
    eps_rel : float
        列 k の摂動 h_k = eps_rel · (1 + |x_k|)
    """
    n = nx * ny
    N = 3 * n
    x = np.asarray(x, dtype=float)
    if x.shape != (N,):
        raise ValueError(f"x の形状 {x.shape} が (3·nx·ny,)=({N},) と一致しません")
    if r0 is None:
        r0 = resid(x)
    a = 2 * radius + 1
    ii, jj = np.divmod(np.arange(n), ny)
    color_cell = (ii % a) * a + (jj % a)
    h = eps_rel * (1.0 + np.abs(x))

    # 列 k（var, i, j）→ 影響する行（var', i+di, j+dj）
    offsets = [(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    for var in range(3):
        for c in range(a * a):
            cells = np.nonzero(color_cell == c)[0]
            if cells.size == 0:
                continue
            cols = var * n + cells
            dx = np.zeros(N)
            dx[cols] = h[cols]
            q = resid(x + dx) - r0
            ci, cj = ii[cells], jj[cells]
            for di, dj in offsets:
                ri, rj = ci + di, cj + dj
                ok = (ri >= 0) & (ri < nx) & (rj >= 0) & (rj < ny)
                if not ok.any():
                    continue
                rcell = ri[ok] * ny + rj[ok]
                ck = cols[ok]
                for var_r in range(3):
                    rows = var_r * n + rcell
                    v = q[rows] / h[ck]
                    nz = v != 0.0
                    if nz.any():
                        rows_l.append(rows[nz])
                        cols_l.append(ck[nz])
                        vals_l.append(v[nz])
    J = sparse.coo_matrix(
        (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))), shape=(N, N)
    )
    return J.tocsr()
