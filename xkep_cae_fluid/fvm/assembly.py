"""面ベース FVM の係数行列組み立て（拡散・対流・時間項・ソース項・非直交補正）.

すべて体積積分形（行は Σ_f F_f = S V_P）で組む。構造格子の既存 FDM（体積で割った形）
とは行ごとの定数倍の違いしかなく、解は一致する。

拡散項は over-relaxed 分解（:func:`~xkep_cae_fluid.fvm.geometry.face_decomposition`）で
E_f 成分を陰的に、T_f 成分を :func:`nonorthogonal_correction` の遅延補正（右辺）として扱う。
非直交メッシュでは :func:`solve_corrected` で補正を数回反復して収束させる。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.boundary import BoundaryFaces
from xkep_cae_fluid.fvm.geometry import (
    _require_faces,
    cell_gradient,
    face_decomposition,
    face_diffusivity,
    face_gradient,
    is_orthogonal,
)
from xkep_cae_fluid.fvm.linear import relative_residual


def boundary_tangent(mesh: MeshData, bfaces: BoundaryFaces) -> np.ndarray:
    """境界面の owner 中心 → 面中心ベクトルの接線成分 t_b (n_boundary_faces, ndim).

    傾いた境界面（面法線とセル中心からのベクトルが平行でない）では、法線勾配を
    (φ_b − φ_P − ∇φ_P·t_b)/d_b と評価する必要がある。直交メッシュではゼロ。
    """
    nd = mesh.face_normals.shape[1]
    r_b = mesh.face_centers[bfaces.faces, :nd] - mesh.cell_centers[bfaces.owner, :nd]
    n_b = mesh.face_normals[bfaces.faces, :nd]
    return r_b - np.sum(r_b * n_b, axis=1)[:, None] * n_b


def _boundary_correction_coefficient(
    mesh: MeshData, gamma: float | np.ndarray, bfaces: BoundaryFaces
) -> np.ndarray:
    """境界面の接線補正の係数 c_b (n_boundary_faces,): J_b に c_b (∇φ_P·t_b) が加わる.

    Dirichlet: Γ_P A_b/d_b、Robin: U A_b、Neumann / ゼロ勾配: 0
    """
    g_p = _cell_gamma(mesh, gamma)[bfaces.owner]
    d_b = bfaces.distance
    safe_d = np.where(d_b > 0, d_b, 1.0)
    c = np.zeros(bfaces.n)
    dir_ = bfaces.is_dirichlet & (d_b > 0)
    c[dir_] = g_p[dir_] * bfaces.area[dir_] / safe_d[dir_]
    rob = bfaces.is_robin
    denom = g_p + bfaces.h * d_b
    u_eff = np.where(rob & (denom > 0), g_p * bfaces.h / np.where(denom > 0, denom, 1.0), 0.0)
    c[rob] = u_eff[rob] * bfaces.area[rob]
    return c


def _cell_gamma(mesh: MeshData, gamma: float | np.ndarray) -> np.ndarray:
    n = mesh.n_cells
    return np.full(n, float(gamma)) if np.isscalar(gamma) else np.asarray(gamma, dtype=np.float64)


def assemble_diffusion(
    mesh: MeshData,
    gamma: float | np.ndarray,
    bfaces: BoundaryFaces,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """−∇·(Γ∇φ) の係数行列と境界寄与の右辺を返す（A φ = b の形、A は対角優位 SPD）.

    内部面: a_f = Γ_f |E_f| / d_PN（Γ_f は調和平均、|E_f| = A_f/(n_f·e_f) は over-relaxed 分解の
    陰的成分。直交メッシュでは A_f）。非直交成分は :func:`nonorthogonal_correction` で右辺に足す
    境界面: :mod:`xkep_cae_fluid.fvm.boundary` の規則
    """
    _require_faces(mesh)
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    neighbour = mesh.face_neighbour

    gamma_f = face_diffusivity(mesh, gamma)
    e_mag, _t_vec, d_pn = face_decomposition(mesh)
    a_f = gamma_f * e_mag / d_pn

    diag = np.zeros(n)
    np.add.at(diag, owner, a_f)
    np.add.at(diag, neighbour, a_f)
    rhs = np.zeros(n)

    # 境界
    g_p = _cell_gamma(mesh, gamma)[bfaces.owner]
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


def nonorthogonal_correction(
    mesh: MeshData,
    phi: np.ndarray,
    gamma: float | np.ndarray,
    bfaces: BoundaryFaces,
    grad: np.ndarray | None = None,
) -> np.ndarray:
    """拡散項の非直交（遅延）補正の右辺寄与 (n_cells,).

    内部面 f の陽的フラックス Γ_f (∇φ)_f·T_f を owner に +、neighbour に − で足す
    （行の左辺が Σ_f a_f (φ_P − φ_N) なので、右辺に足すと J_f = −Γ_f ∇φ_f·S_f が完成する）。
    傾いた境界面（Dirichlet / Robin）では法線勾配の評価点を法線の足に移す接線補正
    −c_b (∇φ_P·t_b) を足す（:func:`boundary_tangent`）。
    ``grad`` を省略すると :func:`cell_gradient` で評価する。直交メッシュではゼロ。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    rhs = np.zeros(mesh.n_cells)
    t_b = boundary_tangent(mesh, bfaces)
    _e_mag, t_vec, _d = face_decomposition(mesh) if n_int else (None, np.zeros((0, 1)), None)
    need_int = bool(np.any(np.abs(t_vec) > 1e-14))
    need_b = bool(np.any(np.abs(t_b) > 1e-14))
    if not (need_int or need_b):
        return rhs
    if grad is None:
        grad = cell_gradient(mesh, phi, bfaces, gamma)
    if need_int:
        gamma_f = face_diffusivity(mesh, gamma)
        corr = gamma_f * np.sum(face_gradient(mesh, grad) * t_vec, axis=1)
        np.add.at(rhs, mesh.face_owner[:n_int], corr)
        np.add.at(rhs, mesh.face_neighbour, -corr)
    if need_b:
        c_b = _boundary_correction_coefficient(mesh, gamma, bfaces)
        np.add.at(rhs, bfaces.owner, -c_b * np.sum(grad[bfaces.owner] * t_b, axis=1))
    return rhs


def diffusive_face_flux(
    mesh: MeshData,
    phi: np.ndarray,
    gamma: float | np.ndarray,
    bfaces: BoundaryFaces,
    corrected: bool = True,
) -> np.ndarray:
    """全面の拡散フラックス J_f = −Γ_f ∇φ_f·S_f (n_faces,)（owner から出る向きが正）.

    内部面: −Γ_f [ |E_f|(φ_N − φ_P)/d_PN + (∇φ)_f·T_f ]（``corrected=False`` なら第 2 項なし）
    境界面: Dirichlet Γ_P A_b (φ_P − φ_b)/d_b、Neumann −flux·A_b、Robin U A_b (φ_P − φ_inf)、
    ゼロ勾配 0（``corrected`` なら Dirichlet / Robin に接線補正 +c_b ∇φ_P·t_b）。
    :func:`assemble_diffusion` + :func:`nonorthogonal_correction` と同じ係数なので、
    収束解では Σ_f J_f = S V_P。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    gamma_f = face_diffusivity(mesh, gamma)
    e_mag, t_vec, d_pn = face_decomposition(mesh)
    t_b = boundary_tangent(mesh, bfaces)
    flux = np.zeros(mesh.n_faces)
    flux[:n_int] = -gamma_f * e_mag * (phi[nb] - phi[owner]) / d_pn
    grad: np.ndarray | None = None
    if corrected and (np.any(np.abs(t_vec) > 1e-14) or np.any(np.abs(t_b) > 1e-14)):
        grad = cell_gradient(mesh, phi, bfaces, gamma)
        flux[:n_int] -= gamma_f * np.sum(face_gradient(mesh, grad) * t_vec, axis=1)

    g_p = _cell_gamma(mesh, gamma)[bfaces.owner]
    phi_p = phi[bfaces.owner]
    d_b = bfaces.distance
    safe_d = np.where(d_b > 0, d_b, 1.0)
    fb = np.zeros(bfaces.n)
    dir_ = bfaces.is_dirichlet & (d_b > 0)
    fb[dir_] = g_p[dir_] * bfaces.area[dir_] * (phi_p[dir_] - bfaces.value[dir_]) / safe_d[dir_]
    neu = bfaces.is_neumann
    fb[neu] = -bfaces.flux[neu] * bfaces.area[neu]
    rob = bfaces.is_robin
    denom = g_p + bfaces.h * d_b
    u_eff = np.where(rob & (denom > 0), g_p * bfaces.h / np.where(denom > 0, denom, 1.0), 0.0)
    fb[rob] = u_eff[rob] * bfaces.area[rob] * (phi_p[rob] - bfaces.phi_inf[rob])
    if grad is not None:
        c_b = _boundary_correction_coefficient(mesh, gamma, bfaces)
        fb += c_b * np.sum(grad[bfaces.owner] * t_b, axis=1)
    flux[bfaces.faces] = fb
    return flux


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
    rho: float | np.ndarray = 1.0,
    dt: float = 0.0,
    phi_old: np.ndarray | None = None,
    phi_correction: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """∂(ρφ)/∂t + ∇·(ṁφ) − ∇·(Γ∇φ) = S の陰的 Euler 系 (A, b) を組む.

    Parameters
    ----------
    source : np.ndarray | None
        体積あたりソース S (n_cells,)。右辺に S V_P
    rho : float | np.ndarray
        時間項の係数（スカラーかセル配列。伝熱なら ρC）
    dt, phi_old : 非定常なら dt > 0 と前ステップ値を与える
    phi_correction : np.ndarray | None
        非直交補正を評価する現在の φ（与えなければ補正なし）。:func:`solve_corrected` が使う
    """
    A, b = assemble_diffusion(mesh, gamma, bfaces)
    if phi_correction is not None:
        b = b + nonorthogonal_correction(mesh, phi_correction, gamma, bfaces)
    if mass_flux is not None:
        A_c, b_c = assemble_convection(mesh, mass_flux, bfaces)
        A = (A + A_c).tocsr()
        b = b + b_c
    if source is not None:
        b = b + np.asarray(source, dtype=np.float64) * mesh.cell_volumes
    if dt > 0.0 and phi_old is not None:
        coeff = np.asarray(rho, dtype=np.float64) * mesh.cell_volumes / dt
        A = (A + sparse.diags(coeff)).tocsr()
        b = b + coeff * np.asarray(phi_old, dtype=np.float64)
    return A, b


def solve_corrected(
    mesh: MeshData,
    build: Callable[[np.ndarray | None], tuple[sparse.csr_matrix, np.ndarray]],
    solver: object,
    phi0: np.ndarray,
    *,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float, int]:
    """非直交補正を遅延補正として反復し、1 つの線形系を収束させる.

    Parameters
    ----------
    build : Callable
        ``build(phi_correction) -> (A, b)``。``None`` なら補正なし
    solver : LinearSolverStrategy
        ``solve(A, b, x0) -> x``
    phi0 : np.ndarray
        初期値（補正の評価にも使う）
    max_iter, tol :
        直交メッシュなら 1 回で終える。非直交では ‖Δφ‖/‖φ‖ < tol まで最大 ``max_iter`` 回

    Returns
    -------
    (phi, residual, n_iter) : 解、最終系の相対残差、反復回数
    """
    phi = np.asarray(phi0, dtype=np.float64).copy()
    if is_orthogonal(mesh):
        A, b = build(None)
        phi = solver.solve(A, b, x0=phi)  # type: ignore[attr-defined]
        return phi, relative_residual(A, phi, b), 1
    n_done = 0
    resid = np.inf
    for it in range(max(int(max_iter), 1)):
        A, b = build(phi)
        new = solver.solve(A, b, x0=phi)  # type: ignore[attr-defined]
        resid = relative_residual(A, new, b)
        change = float(np.linalg.norm(new - phi))
        scale = float(np.linalg.norm(new))
        phi = new
        n_done = it + 1
        if change <= tol * max(scale, 1e-30):
            break
    return phi, resid, n_done
