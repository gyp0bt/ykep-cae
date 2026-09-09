"""trama ケースの「最初の Newton 反復の線形系」を取り出す補助（診断用）.

solve_steady と同じ手順で Stokes–Brinkman 参照場を作り、その場での
  - 定常残差 R(x)（選択スキーム）、擬似時間対角 τ = ρV/Δτ（cfl_init）
  - 1 次風上ヤコビアン J1（+τ）
  - JFNK の有限差分 matvec（solve_linear と同じ式）
を返す。solve_steady の内部を再現しているので、solver.py を変えたらここも合わせること。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nsb.assembly import BrinkmanDiscretization, StateArrays  # noqa: E402
from nsb.core import NSBInput  # noqa: E402
from nsb.linalg import pardiso_solve  # noqa: E402
from nsb.solver import compute_dtau  # noqa: E402


@dataclass
class FirstStep:
    disc: BrinkmanDiscretization
    x: np.ndarray  # Stokes 参照場（= 初期場）
    r_ref: float  # |R(x_stokes)|
    cfl: float
    tau: np.ndarray  # (n,) ρV/Δτ
    diag_aug: np.ndarray  # (3n,) [τ, τ, 0]
    st: StateArrays
    J1: sparse.csr_matrix  # 1 次風上ヤコビアン（τ なし）
    steady_resid: Callable[[np.ndarray], np.ndarray]
    resid_tau: Callable[[np.ndarray], np.ndarray]  # R + τ(x − x_prev)、x_prev = x

    @property
    def n(self) -> int:
        return self.disc.n

    @property
    def rhs(self) -> np.ndarray:
        """solve_linear に渡る右辺 -R_τ(x)（x_prev = x なので = -R）."""
        return -self.resid_tau(self.x)

    @property
    def J1_tau(self) -> sparse.csr_matrix:
        """前処理が組む行列 J1 + diag_aug."""
        return (self.J1 + sparse.diags(self.diag_aug)).tocsr()

    def fd_matvec(self, vec: np.ndarray, extra_tau: bool = False) -> np.ndarray:
        """有限差分 J v（resid_tau に τ が入るので既定は diag_aug を足さない。extra_tau=True で status-45 以前の 2 重カウントを再現）."""
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0.0:
            return np.zeros_like(vec)
        x_norm = float(np.linalg.norm(self.x))
        eps = float(np.sqrt(np.finfo(float).eps)) * np.sqrt(1.0 + x_norm) / v_norm
        r0 = self.resid_tau(self.x)
        out = (self.resid_tau(self.x + eps * vec) - r0) / eps
        if extra_tau:
            out = out + self.diag_aug * vec
        return out


def stokes_field(disc: BrinkmanDiscretization, inp: NSBInput) -> np.ndarray:
    """静止場から Stokes–Brinkman（対流なし）を PARDISO で 1 回解く."""
    s = inp.settings
    x0 = np.zeros(3 * disc.n)
    st0 = disc.compute_state(x0, s.scheme, s.venkat_k)
    r0 = disc.residual_from_state(x0, st0, convection=False)
    J0 = disc.jacobian_first_order(st0, convection=False, x=x0).tocsr()
    return x0 + pardiso_solve(J0, -r0)


def first_step(inp: NSBInput, x: np.ndarray | None = None, cfl: float | None = None) -> FirstStep:
    """Stokes 参照場（または与えた x）での最初の線形系."""
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    if x is None:
        x = stokes_field(disc, inp)

    def steady_resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k, None)

    r_ref = float(np.linalg.norm(steady_resid(x)))
    cfl = s.cfl_init if cfl is None else cfl  # 初期場 = 参照場なら cfl_init·r_ref/r_ref
    uu, vv, _ = disc.split(x)
    u_floor = s.velocity_floor_ratio * disc.u_scale
    dtau = compute_dtau(uu, vv, disc.dx, disc.dy, cfl, s, u_floor)
    tau = (inp.rho * disc.vol / dtau).ravel()
    diag_aug = np.concatenate([tau, tau, np.zeros(n)])
    x_prev = x.copy()

    def resid_tau(xx: np.ndarray) -> np.ndarray:
        r = steady_resid(xx)
        if s.pseudo_time_in_residual:
            r = r.copy()
            r[: 2 * n] += np.concatenate([tau, tau]) * (xx[: 2 * n] - x_prev[: 2 * n])
        return r

    st = disc.compute_state(x, s.scheme, s.venkat_k, None)
    J1 = disc.jacobian_first_order(st, x=x).tocsr()
    return FirstStep(
        disc=disc,
        x=x,
        r_ref=r_ref,
        cfl=cfl,
        tau=tau,
        diag_aug=diag_aug,
        st=st,
        J1=J1,
        steady_resid=steady_resid,
        resid_tau=resid_tau,
    )
