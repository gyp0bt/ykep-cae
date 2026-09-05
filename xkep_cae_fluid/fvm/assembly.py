"""面ベース FVM の係数行列組み立て（拡散・対流・時間項・ソース項）.

すべて体積積分形（行は Σ_f F_f = S V_P）で組む。構造格子の既存 FDM（体積で割った形）
とは行ごとの定数倍の違いしかなく、解は一致する。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.boundary import BoundaryFaces
from xkep_cae_fluid.fvm.geometry import _require_faces, face_diffusivity


def _internal_distance(mesh: MeshData) -> np.ndarray:
    n_int = mesh.n_internal_faces
    d = mesh.cell_centers[mesh.face_neighbour] - mesh.cell_centers[mesh.face_owner[:n_int]]
    return np.linalg.norm(d, axis=1)


def assemble_diffusion(
    mesh: MeshData,
    gamma: float | np.ndarray,
    bfaces: BoundaryFaces,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """−∇·(Γ∇φ) の係数行列と境界寄与の右辺を返す（A φ = b の形、A は対角優位 SPD）.

    内部面: a_f = Γ_f A_f / d_PN（Γ_f は調和平均）
    境界面: :mod:`xkep_cae_fluid.fvm.boundary` の規則
    """
    _require_faces(mesh)
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    neighbour = mesh.face_neighbour

    gamma_f = face_diffusivity(mesh, gamma)
    a_f = gamma_f * mesh.face_areas[:n_int] / _internal_distance(mesh)

    diag = np.zeros(n)
    np.add.at(diag, owner, a_f)
    np.add.at(diag, neighbour, a_f)
    rhs = np.zeros(n)

    # 境界
    g_cell = np.full(n, float(gamma)) if np.isscalar(gamma) else np.asarray(gamma, dtype=np.float64)
    g_p = g_cell[bfaces.owner]
    d_b = bfaces.distance
    safe_d = np.where(d_b > 0, d_b, 1.0)

    dir_ = bfaces.is_dirichlet
    a_b = np.where(dir_ & (d_b > 0), g_p * bfaces.area / safe_d, 0.0)
    np.add.at(diag, bfaces.owner, a_b)
    np.add.at(rhs, bfaces.owner, a_b * bfaces.value)

    neu = bfaces.is_neumann
    np.add.at(rhs, bfaces.owner, np.where(neu, bfaces.flux * bfaces.area, 0.0))

    rob = bfaces.is_robin
    denom = g_p + bfaces.h * d_b
    u_eff = np.where(rob & (denom > 0), g_p * bfaces.h / np.where(denom > 0, denom, 1.0), 0.0)
    np.add.at(diag, bfaces.owner, u_eff * bfaces.area)
    np.add.at(rhs, bfaces.owner, u_eff * bfaces.area * bfaces.phi_inf)

    rows = np.concatenate([owner, neighbour, np.arange(n)])
    cols = np.concatenate([neighbour, owner, np.arange(n)])
    vals = np.concatenate([-a_f, -a_f, diag])
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A, rhs


def assemble_convection(
    mesh: MeshData,
    mass_flux: np.ndarray,
    bfaces: BoundaryFaces,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """∇·(ṁ φ) の 1 次風上の係数行列と境界寄与の右辺を返す.

    Parameters
    ----------
    mass_flux : np.ndarray
        全面の質量流束 (n_faces,)。内部面は owner → neighbour 向き、境界面は外向きが正

    境界面: 流出（ṁ_b > 0）は φ_P、流入は Dirichlet なら φ_b（右辺）、それ以外は φ_P。
    """
    _require_faces(mesh)
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    neighbour = mesh.face_neighbour
    mf = np.asarray(mass_flux, dtype=np.float64)
    if mf.shape != (mesh.n_faces,):
        raise ValueError(f"mass_flux は長さ n_faces={mesh.n_faces} が必要: {mf.shape}")

    mf_int = mf[:n_int]
    mf_pos = np.maximum(mf_int, 0.0)
    mf_neg = np.minimum(mf_int, 0.0)
    diag = np.zeros(n)
    np.add.at(diag, owner, mf_pos)
    np.add.at(diag, neighbour, -mf_neg)
    rhs = np.zeros(n)

    mf_b = mf[bfaces.faces]
    outflow = mf_b > 0
    inflow = ~outflow
    np.add.at(diag, bfaces.owner, np.where(outflow, mf_b, 0.0))
    fixed_in = inflow & bfaces.is_dirichlet
    np.add.at(rhs, bfaces.owner, np.where(fixed_in, -mf_b * bfaces.value, 0.0))
    np.add.at(diag, bfaces.owner, np.where(inflow & ~bfaces.is_dirichlet, mf_b, 0.0))

    rows = np.concatenate([owner, neighbour, np.arange(n)])
    cols = np.concatenate([neighbour, owner, np.arange(n)])
    vals = np.concatenate([mf_neg, -mf_pos, diag])
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A, rhs


def assemble_scalar_transport(
    mesh: MeshData,
    *,
    gamma: float | np.ndarray,
    bfaces: BoundaryFaces,
    mass_flux: np.ndarray | None = None,
    source: np.ndarray | None = None,
    rho: float = 1.0,
    dt: float = 0.0,
    phi_old: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """∂(ρφ)/∂t + ∇·(ṁφ) − ∇·(Γ∇φ) = S の陰的 Euler 系 (A, b) を組む.

    Parameters
    ----------
    source : np.ndarray | None
        体積あたりソース S (n_cells,)。右辺に S V_P
    dt, phi_old : 非定常なら dt > 0 と前ステップ値を与える
    """
    A, b = assemble_diffusion(mesh, gamma, bfaces)
    if mass_flux is not None:
        A_c, b_c = assemble_convection(mesh, mass_flux, bfaces)
        A = (A + A_c).tocsr()
        b = b + b_c
    if source is not None:
        b = b + np.asarray(source, dtype=np.float64) * mesh.cell_volumes
    if dt > 0.0 and phi_old is not None:
        coeff = rho * mesh.cell_volumes / dt
        A = (A + sparse.diags(coeff)).tocsr()
        b = b + coeff * np.asarray(phi_old, dtype=np.float64)
    return A, b
