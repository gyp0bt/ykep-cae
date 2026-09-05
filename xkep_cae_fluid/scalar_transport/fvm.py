"""スカラー輸送ソルバー（面ベース FVM、``MeshData`` 上）.

:class:`ScalarTransportFVMProcess` は :mod:`xkep_cae_fluid.fvm` の低レイヤーを組み合わせた
薄い方程式ファミリー層で、構造格子 / polyMesh / .inp 由来の ``MeshData`` を同じ経路で解く。
構造格子上では既存の :class:`~xkep_cae_fluid.scalar_transport.solver.ScalarTransportProcess`
（FDM）と同じ解を返す（``tests/test_scalar_transport_fvm.py`` で回帰）。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm import (
    PatchBC,
    assemble_scalar_transport,
    face_mass_flux,
    make_linear_solver,
    relative_residual,
    resolve_boundary,
)


@dataclass(frozen=True)
class ScalarTransportFVMInput:
    """面ベース FVM スカラー輸送の入力.

    Parameters
    ----------
    mesh : MeshData
        面情報と ``boundary_patches`` を持つメッシュ
    phi0 : np.ndarray
        初期値 (n_cells,)
    diffusivity : float | np.ndarray
        拡散係数 Γ（スカラーかセル配列）
    rho : float
        密度（時間項・質量流束用）
    velocity : np.ndarray | None
        セル中心速度 (n_cells, ndim)。``mass_flux`` があればそちらを優先
    mass_flux : np.ndarray | None
        全面の質量流束 (n_faces,)（内部面は owner → neighbour、境界面は外向き正）
    bcs : Mapping[str, PatchBC]
        パッチ名 → 境界条件（未指定はゼロ勾配）
    source : np.ndarray | None
        体積あたりソース (n_cells,)
    solid_mask : np.ndarray | None
        固体セル (n_cells,) bool。接する面の対流をゼロにする
    dt, t_end : float
        非定常なら dt > 0
    linear_solver : str
        ``direct`` / ``bicgstab`` / ``amg``
    tol, max_iter : 反復解法の設定
    name : str
        スカラー名（ログ用）
    """

    mesh: MeshData
    phi0: np.ndarray
    diffusivity: float | np.ndarray
    rho: float = 1.0
    velocity: np.ndarray | None = None
    mass_flux: np.ndarray | None = None
    bcs: Mapping[str, PatchBC] = field(default_factory=dict)
    source: np.ndarray | None = None
    solid_mask: np.ndarray | None = None
    dt: float = 0.0
    t_end: float = 0.0
    linear_solver: str = "bicgstab"
    tol: float = 1e-8
    max_iter: int = 500
    name: str = "phi"

    @property
    def is_transient(self) -> bool:
        return self.dt > 0.0


@dataclass(frozen=True)
class ScalarTransportFVMResult:
    """面ベース FVM スカラー輸送の出力."""

    phi: np.ndarray  # (n_cells,)
    converged: bool
    n_timesteps: int = 0
    residual_history: tuple[float, ...] = ()
    elapsed_seconds: float = 0.0
    mass_flux: np.ndarray | None = None  # 使った面質量流束 (n_faces,)


class ScalarTransportFVMProcess(
    SolverProcess["ScalarTransportFVMInput", "ScalarTransportFVMResult"]
):
    """スカラー輸送（対流 1 次風上 + 拡散中心差分 + 陰的 Euler）を ``MeshData`` 上で解く."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ScalarTransportFVM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/fvm-layer.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ScalarTransportFVMInput) -> ScalarTransportFVMResult:
        t0 = time.perf_counter()
        inp = input_data
        mesh = inp.mesh
        phi = np.asarray(inp.phi0, dtype=np.float64).reshape(-1).copy()
        if phi.shape[0] != mesh.n_cells:
            raise ValueError(f"phi0 は長さ n_cells={mesh.n_cells} が必要: {phi.shape}")

        bfaces = resolve_boundary(mesh, inp.bcs)
        mass_flux: np.ndarray | None
        if inp.mass_flux is not None:
            mass_flux = np.asarray(inp.mass_flux, dtype=np.float64).copy()
            if inp.solid_mask is not None:
                mass_flux[self._touching(mesh, inp.solid_mask)] = 0.0
        elif inp.velocity is not None:
            mass_flux = face_mass_flux(mesh, inp.velocity, inp.rho, blocked_cells=inp.solid_mask)
        else:
            mass_flux = None

        solver = make_linear_solver(
            inp.linear_solver,
            **(
                {}
                if inp.linear_solver.lower() == "direct"
                else {"tol": inp.tol, "maxiter": inp.max_iter}
            ),
        )
        tol_ok = max(inp.tol * 10.0, 1e-6)

        def step(phi_old: np.ndarray | None) -> tuple[np.ndarray, float]:
            A, b = assemble_scalar_transport(
                mesh,
                gamma=inp.diffusivity,
                bfaces=bfaces,
                mass_flux=mass_flux,
                source=inp.source,
                rho=inp.rho,
                dt=inp.dt if phi_old is not None else 0.0,
                phi_old=phi_old,
            )
            x = solver.solve(A, b, x0=phi)
            return x, relative_residual(A, x, b)

        if not inp.is_transient:
            phi, resid = step(None)
            return ScalarTransportFVMResult(
                phi=phi,
                converged=bool(resid < tol_ok),
                n_timesteps=0,
                residual_history=(resid,),
                elapsed_seconds=time.perf_counter() - t0,
                mass_flux=mass_flux,
            )

        t = 0.0
        n_steps = 0
        residuals: list[float] = []
        all_ok = True
        while t < inp.t_end - 1e-12:
            phi, resid = step(phi)
            residuals.append(resid)
            all_ok &= resid < tol_ok
            t += inp.dt
            n_steps += 1
        return ScalarTransportFVMResult(
            phi=phi,
            converged=bool(all_ok),
            n_timesteps=n_steps,
            residual_history=tuple(residuals),
            elapsed_seconds=time.perf_counter() - t0,
            mass_flux=mass_flux,
        )

    @staticmethod
    def _touching(mesh: MeshData, solid_mask: np.ndarray) -> np.ndarray:
        """固体セルに接する面のマスク (n_faces,)."""
        blk = np.asarray(solid_mask, dtype=bool).reshape(-1)
        touch = blk[mesh.face_owner].copy()
        n_int = mesh.n_internal_faces
        touch[:n_int] |= blk[mesh.face_neighbour]
        return touch
