"""局所基底の Galerkin 初期解: Stokes 解 + kNN 近傍の収束解を基底 V に、係数 c を残差最小化 min |R(Vc)| で決める.

[舞台] 未知数空間 R^{3n} の中の低次元アフィン部分空間 span(V)（Stokes 解 x_S と近傍解 y_1..y_k）。真の解はこの
  部分空間の外にある（大域 POD の n-width が 176 本でも 11〜30%）が、部分空間の中で残差が最小の点は
  Stokes 解より残差が小さい（x_S が基底に入っているので |R| ≤ |R(x_S)| が保証される）。
[解き方] Gauss–Newton: J V Δc = −R を最小 2 乗で解く（J は彩色 FD の厳密ヤコビアン、JV は k+1 列の密行列）。
  残差が減らなければ半分に刻む（最大 4 回）。残差が増える方向には進まないので最悪でも初期係数の場を返す。
[閉塞] 近傍解の速度はこの θ の閉塞セルで 0 にする（`mask_blocked` と同じ）。Stokes 解はこの θ の場なので不要。
[初期係数] c = e_Stokes（Stokes 解から出発）。kNN 重みから出発するより残差が単調に減る。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nsb.adjoint import colored_fd_jacobian
from nsb.assembly import BrinkmanDiscretization
from nsb.core import NSBInput
from nsb.linalg import pardiso_solve


def stokes_field(inp: NSBInput) -> tuple[np.ndarray, BrinkmanDiscretization]:
    """nsb の参照場（Stokes–Brinkman 解、静止場から線形 1 回）と離散化を返す."""
    disc = BrinkmanDiscretization(inp.to_flow_input())
    s = inp.settings
    x0 = np.zeros(3 * disc.n)
    st0 = disc.compute_state(x0, s.scheme, s.venkat_k)
    r0 = disc.residual_from_state(x0, st0, convection=False)
    J0 = disc.jacobian_first_order(st0, convection=False, x=x0).tocsc()
    return x0 + pardiso_solve(J0, -r0), disc


def galerkin_init(
    inp: NSBInput,
    neighbors: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    blocked: np.ndarray | None,
    steps: int = 6,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Stokes 解 + 近傍解の張る部分空間で残差最小の場を返す。info に残差ノルムの推移と係数."""
    s = inp.settings
    x_s, disc = stokes_field(inp)

    def resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k, None)

    cols = [x_s]
    for u, v, p in neighbors:
        u, v = np.asarray(u, float).copy(), np.asarray(v, float).copy()
        if blocked is not None:
            u[blocked] = 0.0
            v[blocked] = 0.0
        cols.append(np.concatenate([u.ravel(), v.ravel(), np.asarray(p, float).ravel()]))
    V = np.stack(cols, axis=1)  # (3n, k+1)
    c = np.zeros(V.shape[1])
    c[0] = 1.0
    x = V @ c
    r = resid(x)
    norms = [float(np.linalg.norm(r))]
    for _ in range(steps):
        J = colored_fd_jacobian(disc, resid, x)
        JV = np.asarray(J @ V)
        dc, *_ = np.linalg.lstsq(JV, -r, rcond=None)
        alpha = 1.0
        accepted = False
        for _h in range(5):
            x_new = V @ (c + alpha * dc)
            r_new = resid(x_new)
            n_new = float(np.linalg.norm(r_new))
            if np.isfinite(n_new) and n_new < norms[-1]:
                c, x, r = c + alpha * dc, x_new, r_new
                norms.append(n_new)
                accepted = True
                break
            alpha *= 0.5
        if not accepted or norms[-1] < 1e-3 * norms[0]:
            break
    u, v, p = disc.split(x)
    shape = (inp.nx, inp.ny)
    return (u.reshape(shape).copy(), v.reshape(shape).copy(), p.reshape(shape).copy()), {
        "r_norms": norms,
        "coef": c.tolist(),
        "r_stokes": float(norms[0]),
        "r_final": float(norms[-1]),
        "n_basis": int(V.shape[1]),
    }
