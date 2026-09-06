"""Darcy 流れソルバー（面ベース FVM の圧力ポアソン方程式）.

S_s ∂p/∂t − ∇·(Γ ∇p) = S、Γ = (K/μ)/(1 + β ρ K |u|/μ) を :mod:`xkep_cae_fluid.fvm` の
スカラー輸送組み立て（拡散 + 時間項 + ソース）で解き、面の体積流量とそれから再構成した
Darcy 速度、セルごとの質量不整合を返す。Forchheimer 項（β > 0）は |u| を固定して解く
Picard 反復、非定常は陰的 Euler。構造格子・polyMesh・.inp のどの ``MeshData`` でも同じ経路。
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
    diffusive_face_flux,
    make_linear_solver,
    neighbour_centers,
    resolve_boundary,
    solve_corrected,
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
    """面の体積流量 q_f = −Γ_f ∇p_f·S_f（内部面は owner → neighbour、境界面は外向き）.

    :func:`~xkep_cae_fluid.fvm.diffusive_face_flux` そのもの（非直交補正込み）。
    """
    return diffusive_face_flux(mesh, p, gamma, bfaces)


def cell_velocity_from_face_flux(mesh: MeshData, q: np.ndarray) -> np.ndarray:
    """面の体積流量からセル速度を再構成 u_P = (1/V_P) Σ_f q_f (x_f − x_P).

    定数速度場で厳密（Σ_f S_f (x_f − x_P)ᵀ = V_P I）。Green–Gauss 勾配 × 拡散係数と違い、
    透過率が不連続な界面セルでも面流束（調和平均）と整合した速度になる。
    """
    n_int = mesh.n_internal_faces
    nd = mesh.face_centers.shape[1]
    r_owner = mesh.face_centers[:, :nd] - mesh.cell_centers[mesh.face_owner, :nd]
    r_nb = mesh.face_centers[:n_int, :nd] - neighbour_centers(mesh)
    u = np.zeros((mesh.n_cells, nd))
    np.add.at(u, mesh.face_owner, q[:, None] * r_owner)
    np.add.at(u, mesh.face_neighbour, -q[:n_int, None] * r_nb)
    return u / mesh.cell_volumes[:, None]


class DarcyFlowProcess(SolverProcess["DarcyFlowInput", "DarcyFlowResult"]):
    """Darcy 流れ（S_s ∂p/∂t − ∇·(Γ∇p) = S、Forchheimer 補正付き）を ``MeshData`` 上で解く SolverProcess."""

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
        mobility = k / float(inp.viscosity)
        beta = (
            np.full(n, float(inp.forchheimer))
            if np.isscalar(inp.forchheimer)
            else np.asarray(inp.forchheimer, dtype=np.float64).reshape(-1)
        )
        if beta.shape != (n,) or np.any(beta < 0.0):
            raise ValueError("forchheimer は非負のスカラーか長さ n_cells の配列が必要")
        nonlinear = bool(np.any(beta > 0.0))
        storage = (
            np.full(n, float(inp.specific_storage))
            if np.isscalar(inp.specific_storage)
            else np.asarray(inp.specific_storage, dtype=np.float64).reshape(-1)
        )
        if storage.shape != (n,) or np.any(storage < 0.0):
            raise ValueError("specific_storage は非負のスカラーか長さ n_cells の配列が必要")
        transient = inp.is_transient
        if transient and not np.any(storage > 0.0):
            raise ValueError("非定常（dt > 0）には specific_storage > 0 が必要")

        patch_bcs = _to_patch_bcs(inp.bcs)
        bfaces = resolve_boundary(mesh, patch_bcs)
        if not np.any(bfaces.is_dirichlet) and not transient:
            raise ValueError(
                "圧力の基準がありません（少なくとも 1 つのパッチに PRESSURE 境界が必要）"
            )

        source = None if inp.source is None else np.asarray(inp.source, dtype=np.float64)
        if source is not None and source.shape != (n,):
            raise ValueError(f"source は長さ n_cells={n} が必要: {source.shape}")
        solver = make_linear_solver(
            inp.linear_solver,
            **(
                {}
                if inp.linear_solver.lower() == "direct"
                else {"tol": inp.tol, "maxiter": inp.max_iter}
            ),
        )
        p = np.zeros(n) if inp.p0 is None else np.asarray(inp.p0, dtype=np.float64).reshape(-1)
        rho = float(inp.density)
        n_int = mesh.n_internal_faces

        def mobility_of(vel: np.ndarray) -> np.ndarray:
            """Forchheimer の実効移動度 Γ = (K/μ)/(1 + β ρ K |u|/μ)（β = 0 なら K/μ）."""
            if not nonlinear:
                return mobility
            speed = np.linalg.norm(vel, axis=1)
            return mobility / (1.0 + beta * rho * mobility * speed)

        def solve_step(
            p_start: np.ndarray, p_old: np.ndarray | None, dt: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, bool]:
            """1 ステップ（定常なら全体）を Picard 反復で解く → (p, q, u, resid, n_corr, n_picard, ok)."""
            p_cur = p_start.copy()
            gamma = mobility_of(np.zeros((n, mesh.face_centers.shape[1])))
            if nonlinear and p_old is not None:
                gamma = mobility_of(
                    cell_velocity_from_face_flux(
                        mesh, face_volume_flux(mesh, p_cur, mobility, bfaces)
                    )
                )
            u_prev: np.ndarray | None = None
            resid = np.inf
            n_corr = 0
            ok = False
            n_picard = 0
            for it in range(max(int(inp.max_picard_iter), 1) if nonlinear else 1):
                n_picard = it + 1

                def build(p_corr: np.ndarray | None, gamma=gamma):
                    return assemble_scalar_transport(
                        mesh,
                        gamma=gamma,
                        bfaces=bfaces,
                        source=source,
                        rho=storage,
                        dt=dt,
                        phi_old=p_old,
                        phi_correction=p_corr,
                    )

                p_cur, resid, n_corr = solve_corrected(
                    mesh, build, solver, p_cur, max_iter=inp.max_nonorthogonal_iter, tol=inp.tol
                )
                q = face_volume_flux(mesh, p_cur, gamma, bfaces)
                u = cell_velocity_from_face_flux(mesh, q)
                if not nonlinear:
                    ok = True
                    break
                if u_prev is not None:
                    change = float(np.linalg.norm(u - u_prev))
                    if change <= inp.picard_tol * max(float(np.linalg.norm(u)), 1e-300):
                        ok = True
                        break
                u_prev = u
                gamma = mobility_of(u)
            return p_cur, q, u, resid, n_corr, n_picard, ok

        times: list[float] = []
        p_hist: list[np.ndarray] = []
        n_steps = 0
        all_ok = True
        if not transient:
            p, q, velocity, resid, n_corr, n_picard, all_ok = solve_step(p, None, 0.0)
        else:
            n_steps = int(np.ceil(inp.t_end / inp.dt))
            t = 0.0
            q = np.zeros(mesh.n_faces)
            velocity = np.zeros((n, mesh.face_centers.shape[1]))
            resid, n_corr, n_picard = 0.0, 0, 1
            for step in range(n_steps):
                t += inp.dt
                p, q, velocity, resid, n_corr, n_picard, ok = solve_step(p, p.copy(), inp.dt)
                all_ok &= ok
                if (step + 1) % max(inp.output_interval, 1) == 0 or step == n_steps - 1:
                    times.append(t)
                    p_hist.append(p.copy())
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
            converged=bool(resid < max(inp.tol * 10.0, 1e-8)) and all_ok,
            residual=resid,
            elapsed_seconds=time.perf_counter() - t0,
            inflow=inflow,
            outflow=outflow,
            n_nonorthogonal_iter=n_corr,
            n_picard_iter=n_picard,
            n_timesteps=n_steps,
            time_history=tuple(times),
            p_history=tuple(p_hist),
        )
