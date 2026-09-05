"""面ベース FVM の幾何演算（補間重み・面物性・面フラックス・勾配）."""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.boundary import BoundaryFaces


def _require_faces(mesh: MeshData) -> None:
    if (
        mesh.face_owner is None
        or mesh.face_neighbour is None
        or mesh.face_areas is None
        or mesh.face_normals is None
        or mesh.face_centers is None
        or mesh.cell_centers is None
    ):
        raise ValueError("MeshData に面情報（owner/neighbour/areas/normals/centers）がありません")


def face_interpolation_weights(mesh: MeshData) -> np.ndarray:
    """内部面の線形補間重み w_f（φ_f = w_f φ_P + (1 − w_f) φ_N）.

    重みは面中心から neighbour 中心までの距離の比（距離重み付き線形補間）。
    等間隔格子では 0.5。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    neighbour = mesh.face_neighbour
    fc = mesh.face_centers[:n_int]
    d_pf = np.linalg.norm(fc - mesh.cell_centers[owner], axis=1)
    d_fn = np.linalg.norm(mesh.cell_centers[neighbour] - fc, axis=1)
    denom = d_pf + d_fn
    w = np.where(denom > 0, d_fn / np.where(denom > 0, denom, 1.0), 0.5)
    return w


def internal_face_values(
    mesh: MeshData, phi: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """内部面のセル値補間 φ_f (n_internal_faces,)."""
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    if weights is None:
        weights = face_interpolation_weights(mesh)
    owner = mesh.face_owner[:n_int]
    return weights * phi[owner] + (1.0 - weights) * phi[mesh.face_neighbour]


def face_diffusivity(mesh: MeshData, gamma: float | np.ndarray) -> np.ndarray:
    """内部面の拡散係数（セル値の調和平均）(n_internal_faces,)."""
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    if np.isscalar(gamma):
        return np.full(n_int, float(gamma))
    g = np.asarray(gamma, dtype=np.float64)
    gp = g[mesh.face_owner[:n_int]]
    gn = g[mesh.face_neighbour]
    denom = gp + gn
    out = np.zeros(n_int)
    ok = denom > 0
    out[ok] = 2.0 * gp[ok] * gn[ok] / denom[ok]
    return out


def boundary_face_values(
    phi: np.ndarray, bfaces: BoundaryFaces, gamma_owner: np.ndarray | None = None
) -> np.ndarray:
    """境界面の φ_b を境界条件から評価する (n_boundary_faces,).

    - Dirichlet: 指定値
    - Neumann: φ_P + flux·d_b/Γ_P（Γ_P 未指定なら φ_P）
    - Robin: φ_b = (Γ_P φ_P/d_b + h φ_inf) / (Γ_P/d_b + h)（Γ_P 未指定なら φ_P）
    - zero-gradient: φ_P
    """
    phi_p = phi[bfaces.owner]
    out = phi_p.copy()
    out[bfaces.is_dirichlet] = bfaces.value[bfaces.is_dirichlet]
    if gamma_owner is not None:
        g = np.asarray(gamma_owner, dtype=np.float64)
        neu = bfaces.is_neumann & (g > 0)
        out[neu] = phi_p[neu] + bfaces.flux[neu] * bfaces.distance[neu] / g[neu]
        rob = bfaces.is_robin
        if np.any(rob):
            a = np.where(
                bfaces.distance[rob] > 0,
                g[rob] / np.where(bfaces.distance[rob] > 0, bfaces.distance[rob], 1.0),
                0.0,
            )
            denom = a + bfaces.h[rob]
            ok = denom > 0
            vals = phi_p[rob].copy()
            vals[ok] = (
                a[ok] * phi_p[rob][ok] + bfaces.h[rob][ok] * bfaces.phi_inf[rob][ok]
            ) / denom[ok]
            out[rob] = vals
    return out


def face_mass_flux(
    mesh: MeshData,
    velocity: np.ndarray,
    rho: float = 1.0,
    *,
    blocked_cells: np.ndarray | None = None,
    boundary_normal_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """セル中心速度から全面の質量流束 ṁ_f = ρ (u_f·n_f) A_f を作る (n_faces,).

    内部面は距離重み付き線形補間、境界面は owner の速度（``boundary_normal_velocity``
    を与えればその外向き法線速度で上書き）。``blocked_cells``（固体など）に接する
    面の流束はゼロにする。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    n_faces = mesh.n_faces
    u = np.asarray(velocity, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != mesh.n_cells:
        raise ValueError(f"velocity は (n_cells, ndim) が必要: {u.shape}")
    nd = mesh.face_normals.shape[1]
    if u.shape[1] < nd:
        pad = np.zeros((u.shape[0], nd - u.shape[1]))
        u = np.hstack([u, pad])
    w = face_interpolation_weights(mesh)
    owner = mesh.face_owner
    u_face = np.zeros((n_faces, nd))
    u_face[:n_int] = w[:, None] * u[owner[:n_int]] + (1.0 - w)[:, None] * u[mesh.face_neighbour]
    u_face[n_int:] = u[owner[n_int:]]
    un = np.sum(u_face * mesh.face_normals[:, :nd], axis=1)
    if boundary_normal_velocity is not None:
        bnv = np.asarray(boundary_normal_velocity, dtype=np.float64)
        if bnv.shape != (n_faces - n_int,):
            raise ValueError("boundary_normal_velocity は長さ n_boundary_faces が必要")
        un[n_int:] = bnv
    flux = rho * un * mesh.face_areas
    if blocked_cells is not None:
        blk = np.asarray(blocked_cells, dtype=bool)
        touch = blk[owner].copy()
        touch[:n_int] |= blk[mesh.face_neighbour]
        flux[touch] = 0.0
    return flux


def cell_gradient(
    mesh: MeshData,
    phi: np.ndarray,
    bfaces: BoundaryFaces,
    gamma: float | np.ndarray | None = None,
) -> np.ndarray:
    """Green–Gauss のセル勾配 ∇φ_P = (1/V_P) Σ_f φ_f S_f (n_cells, ndim).

    内部面は距離重み付き線形補間、境界面は :func:`boundary_face_values`。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    phi_int = internal_face_values(mesh, phi)
    gamma_owner: np.ndarray | None = None
    if gamma is not None:
        g = np.full(mesh.n_cells, float(gamma)) if np.isscalar(gamma) else np.asarray(gamma)
        gamma_owner = g[bfaces.owner]
    phi_b = boundary_face_values(phi, bfaces, gamma_owner)
    phi_f = np.concatenate([phi_int, phi_b])
    s_f = mesh.face_normals[:, :nd] * mesh.face_areas[:, None]
    contrib = phi_f[:, None] * s_f
    grad = np.zeros((mesh.n_cells, nd))
    np.add.at(grad, mesh.face_owner, contrib)
    np.add.at(grad, mesh.face_neighbour, -contrib[:n_int])
    return grad / mesh.cell_volumes[:, None]
