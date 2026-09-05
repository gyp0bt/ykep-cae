"""設計変数 θ → 収束解 x(θ) の随伴感度（陰関数定理による VJP）.

外側の autodiff（JAX / PyTorch の custom VJP など）から呼ぶことを想定した最小 API:

    prob = ImplicitSolve(build_input)          # θ -> NSBInput を組む関数を渡す
    res, x = prob.forward(theta)               # 定常解
    theta_bar = prob.vjp(theta, x, x_bar)      # x_bar = ∂f/∂x に対する θ 方向の勾配 -(∂R/∂θ)ᵀ λ, Jᵀ λ = x_bar
    f, g = prob.gradient(theta, x, objective)  # スカラー目的関数の値と dθ

ヤコビアン J = ∂R/∂x は選択スキーム（2 次風上 + リミター）の残差そのものから
彩色有限差分（構造格子、ステンシル半径 radius）で厳密に組み、転置系は PARDISO で解く。1 次風上の解析ヤコビアン J1 は
RC 係数を凍結しているので感度には使わない。∂R/∂θ は θ が少数なので中心差分で求める。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from nsb.assembly import BrinkmanDiscretization
from nsb.core import NSBInput, NSBResult
from nsb.linalg import pardiso_solve
from nsb.solver import solve_steady

BuildFn = Callable[[np.ndarray], NSBInput]


def colored_fd_jacobian(
    disc: BrinkmanDiscretization,
    resid: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    radius: int = 2,
    rel_step: float = 1e-6,
) -> sparse.csr_matrix:
    """構造格子の彩色中心差分で疎ヤコビアン ∂R/∂x を組む.

    セル (i, j) を (i mod m, j mod m), m = 2·radius + 1 で彩色し、同色・同ブロック（u, v, p）の
    未知数をまとめて摂動する。残差の差分は摂動セルからチェビシェフ距離 radius 以内の
    セルの行にだけ帰属させる。評価回数は 3·m²·2。
    """
    nx, ny, n = disc.nx, disc.ny, disc.n
    m = 2 * radius + 1
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cells = (ii * ny + jj).ravel()
    color = ((ii % m) * m + (jj % m)).ravel()
    # 各セルの影響範囲（行）: 近傍セル id の配列
    di, dj = np.meshgrid(
        np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij"
    )
    di, dj = di.ravel(), dj.ravel()

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for block in range(3):
        for col in range(m * m):
            sel = cells[color == col]
            if sel.size == 0:
                continue
            k = block * n + sel
            h = rel_step * np.maximum(1.0, np.abs(x[k]))
            e = np.zeros_like(x)
            e[k] = h
            dr = (resid(x + e) - resid(x - e)) / 1.0  # 各列で h が違うので後で割る
            # 摂動セルごとに近傍行を集める
            ci = sel // ny
            cj = sel % ny
            ni = ci[:, None] + di[None, :]
            nj = cj[:, None] + dj[None, :]
            ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
            nb = np.where(ok, ni * ny + nj, -1)
            for b_row in range(3):
                r = b_row * n + nb
                r = np.where(ok, r, 0)
                v = dr[r] / (2.0 * h[:, None])
                rows.append(r[ok])
                cols.append(np.broadcast_to(k[:, None], r.shape)[ok])
                vals.append(v[ok])
    J = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n)
    )
    return J.tocsr()


@dataclass(frozen=True)
class Objective:
    """スカラー目的関数 f(x, θ) と ∂f/∂x（∂f/∂θ は固定 x での中心差分で求める）."""

    value: Callable[[np.ndarray, NSBInput], float]
    grad_x: Callable[[np.ndarray, NSBInput], np.ndarray]


def source_mean_pressure_objective() -> Objective:
    """注入マニホールド（q_src で重み付け）の平均圧力 = 圧力損失の代表値."""

    def weights(inp: NSBInput) -> np.ndarray:
        disc = BrinkmanDiscretization(inp.to_flow_input())
        w = disc.q_src.ravel()
        return w / w.sum()

    def value(x: np.ndarray, inp: NSBInput) -> float:
        n = inp.nx * inp.ny
        return float(weights(inp) @ x[2 * n :])

    def grad_x(x: np.ndarray, inp: NSBInput) -> np.ndarray:
        n = inp.nx * inp.ny
        g = np.zeros_like(x)
        g[2 * n :] = weights(inp)
        return g

    return Objective(value, grad_x)


class ImplicitSolve:
    """θ -> NSBInput を組む関数を受け取り、収束解の随伴感度を提供する."""

    def __init__(self, build_input: BuildFn, jac_radius: int = 2, theta_step: float = 1e-6) -> None:
        self.build_input = build_input
        self.jac_radius = jac_radius
        self.theta_step = theta_step

    def _disc(self, theta: np.ndarray) -> tuple[NSBInput, BrinkmanDiscretization]:
        inp = self.build_input(np.asarray(theta, dtype=float))
        return inp, BrinkmanDiscretization(inp.to_flow_input())

    def residual(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        inp, disc = self._disc(theta)
        s = inp.settings
        return disc.residual(x, s.scheme, s.venkat_k)

    def forward(
        self, theta: np.ndarray, init: NSBResult | None = None, log=None
    ) -> tuple[NSBResult, np.ndarray]:
        inp = self.build_input(np.asarray(theta, dtype=float))
        if init is not None:
            from dataclasses import replace

            inp = replace(inp, u0=init.u, v0=init.v, p0=init.p)
        res = solve_steady(inp, log=log)
        x = np.concatenate([res.u.ravel(), res.v.ravel(), res.p.ravel()])
        return res, x

    def jacobian(self, theta: np.ndarray, x: np.ndarray) -> sparse.csr_matrix:
        inp, disc = self._disc(theta)
        s = inp.settings
        return colored_fd_jacobian(
            disc, lambda xx: disc.residual(xx, s.scheme, s.venkat_k), x, self.jac_radius
        )

    def dR_dtheta(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """∂R/∂θ (3n × nθ)。θ は少数なので中心差分."""
        theta = np.asarray(theta, dtype=float)
        cols = []
        for k in range(theta.size):
            h = self.theta_step * max(1.0, abs(theta[k]))
            e = np.zeros_like(theta)
            e[k] = h
            cols.append((self.residual(theta + e, x) - self.residual(theta - e, x)) / (2 * h))
        return np.stack(cols, axis=1)

    def adjoint(self, theta: np.ndarray, x: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        """Jᵀ λ = x_bar を解く."""
        J = self.jacobian(theta, x)
        return pardiso_solve(J.T.tocsr(), np.asarray(x_bar, dtype=float))

    def vjp(self, theta: np.ndarray, x: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        """x(θ) の VJP: θ_bar = -(∂R/∂θ)ᵀ λ, Jᵀ λ = x_bar."""
        lam = self.adjoint(theta, x, x_bar)
        return -self.dR_dtheta(theta, x).T @ lam

    def gradient(
        self, theta: np.ndarray, x: np.ndarray, obj: Objective
    ) -> tuple[float, np.ndarray]:
        """f(x(θ), θ) の値と全微分 df/dθ = ∂f/∂θ + θ_bar(∂f/∂x)."""
        theta = np.asarray(theta, dtype=float)
        inp = self.build_input(theta)
        f = obj.value(x, inp)
        theta_bar = self.vjp(theta, x, obj.grad_x(x, inp))
        # ∂f/∂θ（x 固定）
        for k in range(theta.size):
            h = self.theta_step * max(1.0, abs(theta[k]))
            e = np.zeros_like(theta)
            e[k] = h
            fp = obj.value(x, self.build_input(theta + e))
            fm = obj.value(x, self.build_input(theta - e))
            theta_bar[k] += (fp - fm) / (2 * h)
        return f, theta_bar
