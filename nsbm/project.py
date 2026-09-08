"""予測場の Newton 射影: 初期場から厳密ヤコビアン（彩色 FD）で Newton 1 歩だけ進め、離散方程式に近づける.

[なぜ] 補間・予測で得た場の残差は高波数のごみ（発散・圧力の細かい誤差）が主で、Stokes 場の残差
  （滑らかな対流の不釣り合い）とは質が違う。SER は残差ノルムだけで出発 CFL を決めるので、
  ごみが残っている限り良い場でも小さな CFL から出発する。Newton の吸引域にある場なら、
  減衰なし（CFL = ∞）の 1 歩でごみが消え、残差比 |R|/|R_ref| が 1 を大きく割る。
[コスト] 彩色 FD で 3·5²·2 = 150 回の残差評価（numba で 0.5 ms）+ 疎 LU 1 回。Stokes 初期場と同程度。
[安全] 1 歩で残差が減らなければ元の場を返す（吸引域の外）。
"""

from __future__ import annotations

import numpy as np

from nsb.adjoint import colored_fd_jacobian
from nsb.assembly import BrinkmanDiscretization
from nsb.core import NSBInput
from nsb.linalg import pardiso_solve
from nsbm.features import blocked_mask_from_x


def newton_project(
    inp: NSBInput,
    init: tuple[np.ndarray, np.ndarray, np.ndarray],
    x_img: np.ndarray | None = None,
    steps: int = 1,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """init から Newton を steps 歩進めた場と、残差ノルムの推移を返す（減らなければ手前で止める）."""
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n

    def resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k, None)

    x = np.concatenate([np.asarray(f, dtype=float).ravel() for f in init])
    r = resid(x)
    norms = [float(np.linalg.norm(r))]
    blocked = None if x_img is None else blocked_mask_from_x(x_img).ravel()
    for _ in range(steps):
        J = colored_fd_jacobian(disc, resid, x)
        x_new = x + pardiso_solve(J.tocsc(), -r)
        if blocked is not None:  # 射影後も閉塞セルの速度は 0 に保つ
            x_new[:n][blocked] = 0.0
            x_new[n : 2 * n][blocked] = 0.0
        r_new = resid(x_new)
        n_new = float(np.linalg.norm(r_new))
        if not np.isfinite(n_new) or n_new >= norms[-1]:
            break
        x, r = x_new, r_new
        norms.append(n_new)
    u, v, p = disc.split(x)
    shape = (inp.nx, inp.ny)
    return (u.reshape(shape), v.reshape(shape), p.reshape(shape)), {
        "r_before": norms[0],
        "r_after": norms[-1],
        "steps_taken": float(len(norms) - 1),
    }
