"""面ベース FVM の幾何演算（補間重み・面物性・面フラックス・勾配・非直交分解）.

非直交メッシュでは内部面の面ベクトル S_f = n_f A_f を over-relaxed 分解
S_f = E_f + T_f（E_f ∥ セル中心間ベクトル、|E_f| = A_f/(n_f·e_f)）し、
E_f 成分を陰的（係数行列）、T_f 成分を陽的（遅延補正）に扱う。直交メッシュでは T_f = 0。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

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


def neighbour_centers(mesh: MeshData) -> np.ndarray:
    """内部面の neighbour セル中心を owner から見た位置で返す (n_internal_faces, ndim).

    通常の面は ``cell_centers[neighbour]``。周期面（``mesh.face_offset``）は並進で戻した
    ``cell_centers[neighbour] + face_offset``。P–N ベクトル・補間重み・スキュー・非直交分解・
    最小二乗勾配・TVD の上流距離・Rhie–Chow は全てこれを使う。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    xn = mesh.cell_centers[mesh.face_neighbour, :nd]
    if mesh.face_offset is None:
        return xn
    off = np.asarray(mesh.face_offset, dtype=np.float64)
    if off.shape[0] != n_int:
        raise ValueError(f"face_offset は (n_internal_faces={n_int}, ndim) が必要: {off.shape}")
    return xn + off[:, :nd]


def face_interpolation_weights(mesh: MeshData) -> np.ndarray:
    """内部面の線形補間重み w_f（φ_f = w_f φ_P + (1 − w_f) φ_N）.

    重みは面中心から neighbour 中心までの距離の比（距離重み付き線形補間）。
    等間隔格子では 0.5。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nd = mesh.face_normals.shape[1]
    fc = mesh.face_centers[:n_int, :nd]
    d_pf = np.linalg.norm(fc - mesh.cell_centers[owner, :nd], axis=1)
    d_fn = np.linalg.norm(neighbour_centers(mesh) - fc, axis=1)
    denom = d_pf + d_fn
    w = np.where(denom > 0, d_fn / np.where(denom > 0, denom, 1.0), 0.5)
    return w


def face_skewness(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """内部面のスキュー: セル中心を結ぶ直線と面平面の交点 x'_f と、面中心からのずれ.

    Returns
    -------
    t : np.ndarray
        交点の直線パラメータ x'_f = x_P + t (x_N − x_P) (n_internal_faces,)
    skew : np.ndarray
        x_f − x'_f (n_internal_faces, ndim)。直交・非スキューなメッシュではゼロ。
        四面体など面中心が P–N 直線から外れるメッシュで、面値の補間を x'_f で行い
        ∇φ_f·skew を足す（スキュー補正）ために使う
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    owner = mesh.face_owner[:n_int]
    xp = mesh.cell_centers[owner, :nd]
    d = neighbour_centers(mesh) - xp
    n_f = mesh.face_normals[:n_int, :nd]
    xf = mesh.face_centers[:n_int, :nd]
    denom = np.sum(n_f * d, axis=1)
    safe = np.where(np.abs(denom) > 1e-300, denom, 1.0)
    t = np.where(np.abs(denom) > 1e-300, np.sum(n_f * (xf - xp), axis=1) / safe, 0.5)
    t = np.clip(t, 0.0, 1.0)
    skew = xf - (xp + t[:, None] * d)
    return t, skew


def face_decomposition(mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """内部面の over-relaxed 分解 S_f = E_f + T_f.

    Returns
    -------
    e_mag : np.ndarray
        |E_f| = A_f / (n_f·e_f) (n_internal_faces,)。直交なら A_f
    t_vec : np.ndarray
        T_f = S_f − E_f (n_internal_faces, ndim)。直交ならゼロ
    d_pn : np.ndarray
        セル中心間距離 |x_N − x_P| (n_internal_faces,)
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    d_vec = neighbour_centers(mesh) - mesh.cell_centers[mesh.face_owner[:n_int], :nd]
    d_pn = np.linalg.norm(d_vec, axis=1)
    e = d_vec / np.where(d_pn > 0, d_pn, 1.0)[:, None]
    n = mesh.face_normals[:n_int, :nd]
    cos = np.sum(n * e, axis=1)
    if np.any(cos <= 1e-6):
        raise ValueError("内部面の法線とセル中心間ベクトルがほぼ直交しています（メッシュが不正）")
    area = mesh.face_areas[:n_int]
    e_mag = area / cos
    t_vec = n * area[:, None] - e * e_mag[:, None]
    return e_mag, t_vec, d_pn


def max_nonorthogonality_deg(mesh: MeshData) -> float:
    """内部面の最大非直交角 [deg]（n_f と e_f のなす角）。直交メッシュなら 0."""
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    if n_int == 0:
        return 0.0
    nd = mesh.face_normals.shape[1]
    d_vec = neighbour_centers(mesh) - mesh.cell_centers[mesh.face_owner[:n_int], :nd]
    d_pn = np.linalg.norm(d_vec, axis=1)
    e = d_vec / np.where(d_pn > 0, d_pn, 1.0)[:, None]
    cos = np.clip(np.sum(mesh.face_normals[:n_int, :nd] * e, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos.min())))


def is_orthogonal(mesh: MeshData, tol_deg: float = 1e-6) -> bool:
    """非直交補正が不要（最大非直交角が ``tol_deg`` 以下）か."""
    return max_nonorthogonality_deg(mesh) <= tol_deg


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


def face_gradient(
    mesh: MeshData, grad: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """内部面の勾配（セル勾配の距離重み付き線形補間）(n_internal_faces, ndim)."""
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    if weights is None:
        weights = face_interpolation_weights(mesh)
    owner = mesh.face_owner[:n_int]
    return weights[:, None] * grad[owner] + (1.0 - weights)[:, None] * grad[mesh.face_neighbour]


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
    n_iter: int = 30,
    tol: float = 1e-10,
) -> np.ndarray:
    """Green–Gauss のセル勾配 ∇φ_P = (1/V_P) Σ_f φ_f S_f (n_cells, ndim).

    内部面は P–N 直線と面平面の交点 x'_f での線形補間にスキュー補正 ∇φ_f·(x_f − x'_f) を
    加えた値（:func:`face_skewness`。六面体の箱格子・せん断格子ではゼロ、四面体では効く）、
    境界面は :func:`boundary_face_values`。Dirichlet 以外の境界面では φ_b に接線方向の外挿
    ∇φ_P·t_b（t_b は面中心へのベクトルの接線成分）を加える。どちらも勾配自身に依存するので
    変化が ``tol``（相対）を切るまで最大 ``n_iter`` 回反復する（線形場では反復ごとにスキュー比
    だけ誤差が縮む。Kuhn 分割の四面体で 1 回あたり約 0.15 倍。直交メッシュでは 1 回）。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    t_line, skew = face_skewness(mesh)
    phi_int0 = (1.0 - t_line) * phi[owner] + t_line * phi[nb]
    gamma_owner: np.ndarray | None = None
    if gamma is not None:
        g = np.full(mesh.n_cells, float(gamma)) if np.isscalar(gamma) else np.asarray(gamma)
        gamma_owner = g[bfaces.owner]
    phi_b0 = boundary_face_values(phi, bfaces, gamma_owner)
    s_f = mesh.face_normals[:, :nd] * mesh.face_areas[:, None]

    r_b = mesh.face_centers[bfaces.faces, :nd] - mesh.cell_centers[bfaces.owner, :nd]
    n_b = mesh.face_normals[bfaces.faces, :nd]
    t_b = r_b - np.sum(r_b * n_b, axis=1)[:, None] * n_b
    extrapolate = ~bfaces.is_dirichlet
    needs_b = bool(np.any(np.abs(t_b[extrapolate]) > 1e-14)) if np.any(extrapolate) else False
    needs_skew = bool(np.any(np.abs(skew) > 1e-14))

    def gauss(phi_int: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
        phi_f = np.concatenate([phi_int, phi_b])
        contrib = phi_f[:, None] * s_f
        grad = np.zeros((mesh.n_cells, nd))
        np.add.at(grad, mesh.face_owner, contrib)
        np.add.at(grad, nb, -contrib[:n_int])
        return grad / mesh.cell_volumes[:, None]

    grad = gauss(phi_int0, phi_b0)
    if not (needs_b or needs_skew):
        return grad
    for _ in range(max(int(n_iter) - 1, 0)):
        phi_int = phi_int0
        if needs_skew:
            g_f = (1.0 - t_line)[:, None] * grad[owner] + t_line[:, None] * grad[nb]
            phi_int = phi_int0 + np.sum(g_f * skew, axis=1)
        phi_b = phi_b0.copy()
        if needs_b:
            phi_b[extrapolate] += np.sum(grad[bfaces.owner[extrapolate]] * t_b[extrapolate], axis=1)
        new = gauss(phi_int, phi_b)
        change = float(np.linalg.norm(new - grad))
        grad = new
        if change <= tol * max(float(np.linalg.norm(grad)), 1e-300):
            break
    return grad


def lsq_gradient_operator(
    mesh: MeshData, bfaces: BoundaryFaces | None = None
) -> tuple[list[sparse.csr_matrix], np.ndarray]:
    """:func:`cell_gradient_lsq` の線形作用素形 (∇φ)_c = G_c φ + g0_c.

    Returns
    -------
    (G, g0) : 成分ごとの疎行列 G_c (n_cells × n_cells) のリスト（長さ ndim）と、Dirichlet 境界値の
        定数寄与 g0 (n_cells, ndim)。速度–圧力の連成組み立て（:func:`~xkep_cae_fluid.fvm.momentum.assemble_coupled`）
        が圧力勾配と Rhie–Chow 流束を陰的に書くために使う
    """
    _require_faces(mesh)
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    xc = mesh.cell_centers[:, :nd]
    r = neighbour_centers(mesh) - xc[owner]
    w = 1.0 / np.maximum(np.sum(r * r, axis=1), 1e-300)
    G = np.zeros((n, nd, nd))
    outer = w[:, None, None] * r[:, :, None] * r[:, None, :]
    np.add.at(G, owner, outer)
    np.add.at(G, nb, outer)
    has_dir = bfaces is not None and bool(np.any(bfaces.is_dirichlet))
    if has_dir:
        d = bfaces.is_dirichlet
        own_b = bfaces.owner[d]
        r_b = mesh.face_centers[bfaces.faces[d], :nd] - xc[own_b]
        w_b = 1.0 / np.maximum(np.sum(r_b * r_b, axis=1), 1e-300)
        np.add.at(G, own_b, w_b[:, None, None] * r_b[:, :, None] * r_b[:, None, :])
    scale = np.maximum(np.max(np.abs(G).reshape(n, -1), axis=1), 1e-300)
    G_inv = np.linalg.pinv(G / scale[:, None, None], rcond=1e-10) / scale[:, None, None]
    # 内部面: owner 行に c_P (φ_N − φ_P)、neighbour 行に c_N (φ_N − φ_P)
    wr = w[:, None] * r  # (n_int, nd)
    c_own = np.einsum("nij,nj->ni", G_inv[owner], wr)
    c_nb = np.einsum("nij,nj->ni", G_inv[nb], wr)
    rows = np.concatenate([owner, owner, nb, nb])
    cols = np.concatenate([nb, owner, nb, owner])
    g0 = np.zeros((n, nd))
    mats: list[sparse.csr_matrix] = []
    for c in range(nd):
        vals = np.concatenate([c_own[:, c], -c_own[:, c], c_nb[:, c], -c_nb[:, c]])
        r_c = rows
        c_c = cols
        if has_dir:
            cb = np.einsum("nij,nj->ni", G_inv[own_b], w_b[:, None] * r_b)
            r_c = np.concatenate([rows, own_b])
            c_c = np.concatenate([cols, own_b])
            vals = np.concatenate([vals, -cb[:, c]])
            np.add.at(g0[:, c], own_b, cb[:, c] * bfaces.value[d])
        mats.append(sparse.coo_matrix((vals, (r_c, c_c)), shape=(n, n)).tocsr())
    return mats, g0


def cell_gradient_lsq(
    mesh: MeshData,
    phi: np.ndarray,
    bfaces: BoundaryFaces | None = None,
    gamma: float | np.ndarray | None = None,
) -> np.ndarray:
    """重み付き最小二乗のセル勾配 (n_cells, ndim).

    内部面の隣接セル（両向き）と Dirichlet 境界面（面中心の既知値）を点集合にして
    min Σ w (∇φ·r − Δφ)²、w = 1/|r|² を解く。線形場では境界セルでも厳密
    （Green–Gauss は非 Dirichlet 境界に接するセルで法線方向の勾配を過小評価する）。
    情報の無い方向（例: 1 セル厚の押し出しメッシュの z）は擬似逆行列でゼロになる。
    """
    _require_faces(mesh)
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    xc = mesh.cell_centers[:, :nd]
    r = neighbour_centers(mesh) - xc[owner]
    dphi = phi[nb] - phi[owner]
    w = 1.0 / np.maximum(np.sum(r * r, axis=1), 1e-300)
    G = np.zeros((n, nd, nd))
    rhs = np.zeros((n, nd))
    outer = w[:, None, None] * r[:, :, None] * r[:, None, :]
    np.add.at(G, owner, outer)
    np.add.at(G, nb, outer)
    np.add.at(rhs, owner, (w * dphi)[:, None] * r)
    np.add.at(rhs, nb, (w * dphi)[:, None] * r)
    if bfaces is not None and np.any(bfaces.is_dirichlet):
        d = bfaces.is_dirichlet
        own_b = bfaces.owner[d]
        r_b = mesh.face_centers[bfaces.faces[d], :nd] - xc[own_b]
        dphi_b = bfaces.value[d] - phi[own_b]
        w_b = 1.0 / np.maximum(np.sum(r_b * r_b, axis=1), 1e-300)
        np.add.at(G, own_b, w_b[:, None, None] * r_b[:, :, None] * r_b[:, None, :])
        np.add.at(rhs, own_b, (w_b * dphi_b)[:, None] * r_b)
    scale = np.maximum(np.max(np.abs(G).reshape(n, -1), axis=1), 1e-300)
    G_inv = np.linalg.pinv(G / scale[:, None, None], rcond=1e-10)
    return np.einsum("nij,nj->ni", G_inv, rhs / scale[:, None])
