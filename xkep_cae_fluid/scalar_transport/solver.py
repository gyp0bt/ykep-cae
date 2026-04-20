"""汎用スカラー輸送ソルバー (FDM).

3次元等間隔直交格子上で対流-拡散-ソース方程式を陰的 Euler + BiCGSTAB で解く。
速度場は外部（NaturalConvectionProcess 等）から与えられる前提。

Phase 6.1a（水槽 CAE ロードマップ）で新設。
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.scalar_transport.assembly import build_scalar_system
from xkep_cae_fluid.scalar_transport.data import (
    ScalarTransportInput,
    ScalarTransportResult,
)


def _solve_linear(
    A, b: np.ndarray, x0: np.ndarray | None = None, tol: float = 1e-8, maxiter: int = 200
) -> tuple[np.ndarray, float]:
    """ILU 前処理付き BiCGSTAB で線形方程式を解き、最終残差を返す."""
    try:
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-4)
        M = spla.LinearOperator(A.shape, matvec=ilu.solve)
    except Exception:
        M = None

    if x0 is None:
        x0 = np.zeros(b.shape[0])

    x, _info = spla.bicgstab(A, b, x0=x0, M=M, rtol=tol, maxiter=maxiter)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-30:
        resid = float(np.linalg.norm(b - A @ x))
    else:
        resid = float(np.linalg.norm(b - A @ x) / b_norm)
    return x, resid


class ScalarTransportProcess(SolverProcess["ScalarTransportInput", "ScalarTransportResult"]):
    """汎用スカラー輸送ソルバー (FDM, 陰的 Euler + BiCGSTAB).

    対流-拡散-ソース方程式を 3 次元等間隔直交格子上で解く。
    速度場は外部から与える。水槽 CAE（Phase 6）での CO2/O2 輸送が主目的。

    対応境界条件:
    - Dirichlet（φ 固定）
    - Neumann（Γ∂φ/∂n 指定）
    - Adiabatic（ゼロ勾配）
    - Robin（J = h_mass·(φ_inf - φ_surface)、ヘンリー則で使用）
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ScalarTransportFDM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/scalar-transport-fdm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ScalarTransportInput) -> ScalarTransportResult:
        """スカラー輸送解析を実行する."""
        t_start = time.perf_counter()
        inp = input_data

        phi = inp.field.phi0.astype(np.float64).copy()

        if inp.is_transient:
            return self._solve_transient(phi, inp, t_start)
        return self._solve_steady(phi, inp, t_start)

    def _solve_steady(
        self,
        phi: np.ndarray,
        inp: ScalarTransportInput,
        t_start: float,
    ) -> ScalarTransportResult:
        """定常解析: 1 回の線形求解で完了."""
        A, b = build_scalar_system(inp, phi_old_time=None)
        x, resid = _solve_linear(A, b, x0=phi.ravel(), tol=inp.tol, maxiter=inp.max_iter)
        phi_new = x.reshape(inp.nx, inp.ny, inp.nz)
        elapsed = time.perf_counter() - t_start
        return ScalarTransportResult(
            phi=phi_new,
            converged=bool(resid < max(inp.tol * 10.0, 1e-6)),
            n_timesteps=0,
            residual_history=(resid,),
            elapsed_seconds=elapsed,
        )

    def _solve_transient(
        self,
        phi: np.ndarray,
        inp: ScalarTransportInput,
        t_start: float,
    ) -> ScalarTransportResult:
        """非定常解析: 陰的 Euler で t_end まで進む."""
        t = 0.0
        residuals: list[float] = []
        n_steps = 0
        all_converged = True

        while t < inp.t_end - 1e-12:
            A, b = build_scalar_system(inp, phi_old_time=phi)
            x, resid = _solve_linear(A, b, x0=phi.ravel(), tol=inp.tol, maxiter=inp.max_iter)
            phi = x.reshape(inp.nx, inp.ny, inp.nz)
            residuals.append(resid)
            if resid >= max(inp.tol * 10.0, 1e-6):
                all_converged = False
            t += inp.dt
            n_steps += 1

        elapsed = time.perf_counter() - t_start
        return ScalarTransportResult(
            phi=phi,
            converged=all_converged,
            n_timesteps=n_steps,
            residual_history=tuple(residuals),
            elapsed_seconds=elapsed,
        )
