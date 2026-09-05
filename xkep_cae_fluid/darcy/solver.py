"""Darcy 流れソルバー（面ベース FVM の圧力ポアソン方程式）.

∇·(K/μ ∇p) = −S を :mod:`xkep_cae_fluid.fvm` の拡散組み立てで解き、
面の体積流量とそれから再構成した Darcy 速度、セルごとの質量不整合を返す。
構造格子・polyMesh・.inp のどの ``MeshData`` でも同じ経路。
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.darcy.data import DarcyBCKind, DarcyFlowInput, DarcyFlowResult, DarcyPatchBC
from xkep_cae_fluid.fvm import (
    BoundaryFaces,
    PatchBC,
    assemble_scalar_transport,
    face_diffusivity,
    make_linear_solver,
    relative_residual,
    resolve_boundary,
)


def _to_patch_bcs(bcs: dict[str, DarcyPatchBC] | object) -> dict[str, PatchBC]:
    """Darcy の境界条件をスカラー拡散の境界条件に写す.

    - PRESSURE → Dirichlet(p)
    - VELOCITY → Neumann。拡散流束 J = −(K/μ)∇p = u なので、流入速度 u_n がそのまま J の流入量
    - WALL → ゼロ勾配（J·n = 0）
    """
    out: dict[str, PatchBC] = {}
    for name, bc in dict(bcs).items():  # type: ignore[arg-type]
        if bc.kind == DarcyBCKind.PRESSURE:
            out[name] = PatchBC.dirichlet(bc.pressure)
        elif bc.kind == DarcyBCKind.VELOCITY:
            out[name] = PatchBC.neumann(bc.velocity)
        else:
            out[name] = PatchBC.zero_gradient()
    return out


def face_volume_flux(
    mesh: MeshData, p: np.ndarray, gamma: np.ndarray, bfaces: BoundaryFaces
) -> np.ndarray:
    """面の体積流量 q_f = −Γ_f A_f ∂p/∂n（内部面は owner → neighbour、境界面は外向き）."""
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    gamma_f = face_diffusivity(mesh, gamma)
    d = np.linalg.norm(mesh.cell_centers[nb] - mesh.cell_centers[owner], axis=1)
    q = np.zeros(mesh.n_faces)
    q[:n_int] = -gamma_f * mesh.face_areas[:n_int] * (p[nb] - p[owner]) / d

    g_p = gamma[bfaces.owner]
    q_b = np.zeros(bfaces.n)
    dir_ = bfaces.is_dirichlet
    safe_d = np.where(bfaces.distance > 0, bfaces.distance, 1.0)
    q_b[dir_] = (
        -g_p[dir_] * bfaces.area[dir_] * (bfaces.value[dir_] - p[bfaces.owner[dir_]]) / safe_d[dir_]
    )
    neu = bfaces.is_neumann
    q_b[neu] = -bfaces.flux[neu] * bfaces.area[neu]  # 流入（正）→ 外向きでは負
    q[bfaces.faces] = q_b
    return q


def cell_velocity_from_face_flux(mesh: MeshData, q: np.ndarray) -> np.ndarray:
    """面の体積流量からセル速度を再構成 u_P = (1/V_P) Σ_f q_f (x_f − x_P).

    定数速度場で厳密（Σ_f S_f (x_f − x_P)ᵀ = V_P I）。Green–Gauss 勾配 × 拡散係数と違い、
    透過率が不連続な界面セルでも面流束（調和平均）と整合した速度になる。
    """
    n_int = mesh.n_internal_faces
    nd = mesh.face_centers.shape[1]
    r_owner = mesh.face_centers - mesh.cell_centers[mesh.face_owner]
    r_nb = mesh.face_centers[:n_int] - mesh.cell_centers[mesh.face_neighbour]
    u = np.zeros((mesh.n_cells, nd))
    np.add.at(u, mesh.face_owner, q[:, None] * r_owner)
    np.add.at(u, mesh.face_neighbour, -q[:n_int, None] * r_nb)
    return u / mesh.cell_volumes[:, None]


class DarcyFlowProcess(SolverProcess["DarcyFlowInput", "DarcyFlowResult"]):
    """Darcy 流れ（∇·(K/μ ∇p) = −S）を ``MeshData`` 上で解く SolverProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="DarcyFlowFVM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/darcy-flow-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: DarcyFlowInput) -> DarcyFlowResult:
        t0 = time.perf_counter()
        inp = input_data
        mesh = inp.mesh
        n = mesh.n_cells
        if inp.viscosity <= 0.0:
            raise ValueError(f"viscosity は正の値が必要: {inp.viscosity}")
        k = (
            np.full(n, float(inp.permeability))
            if np.isscalar(inp.permeability)
            else np.asarray(inp.permeability, dtype=np.float64).reshape(-1)
        )
        if k.shape != (n,):
            raise ValueError(f"permeability は長さ n_cells={n} が必要: {k.shape}")
        if np.any(k <= 0.0):
            raise ValueError("permeability は正の値が必要（不透過は WALL 境界か極小値で表す）")
        gamma = k / float(inp.viscosity)

        patch_bcs = _to_patch_bcs(inp.bcs)
        bfaces = resolve_boundary(mesh, patch_bcs)
        if not np.any(bfaces.is_dirichlet):
            raise ValueError(
                "圧力の基準がありません（少なくとも 1 つのパッチに PRESSURE 境界が必要）"
            )

        source = None if inp.source is None else np.asarray(inp.source, dtype=np.float64)
        if source is not None and source.shape != (n,):
            raise ValueError(f"source は長さ n_cells={n} が必要: {source.shape}")
        A, b = assemble_scalar_transport(mesh, gamma=gamma, bfaces=bfaces, source=source)

        solver = make_linear_solver(
            inp.linear_solver,
            **(
                {}
                if inp.linear_solver.lower() == "direct"
                else {"tol": inp.tol, "maxiter": inp.max_iter}
            ),
        )
        x0 = None if inp.p0 is None else np.asarray(inp.p0, dtype=np.float64).reshape(-1)
        p = solver.solve(A, b, x0=x0)
        resid = relative_residual(A, p, b)

        q = face_volume_flux(mesh, p, gamma, bfaces)
        n_int = mesh.n_internal_faces
        velocity = cell_velocity_from_face_flux(mesh, q)
        if velocity.shape[1] < 3:
            velocity = np.hstack([velocity, np.zeros((n, 3 - velocity.shape[1]))])
        div = np.zeros(n)
        np.add.at(div, mesh.face_owner, q)
        np.add.at(div, mesh.face_neighbour, -q[:n_int])
        mass_residual = div - (0.0 if source is None else source * mesh.cell_volumes)
        q_b = q[n_int:]
        inflow = float(-q_b[q_b < 0].sum())
        outflow = float(q_b[q_b > 0].sum())
        return DarcyFlowResult(
            p=p,
            velocity=velocity,
            face_flux=q,
            mass_residual=mass_residual,
            converged=bool(resid < max(inp.tol * 10.0, 1e-8)),
            residual=resid,
            elapsed_seconds=time.perf_counter() - t0,
            inflow=inflow,
            outflow=outflow,
        )
