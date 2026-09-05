"""面ベース FVM の運動量・圧力連成カーネル（同位置格子 SIMPLE / Rhie–Chow）.

速度の境界パッチ条件（:class:`VelocityPatchBC`）を境界面配列に展開し、

- 運動量方程式の成分ごとの係数行列（対流 1 次風上 + 拡散 + 時間項 + 圧力勾配 + 抵抗 + 緩和）
- Rhie–Chow 補間の面質量流束
- 圧力補正方程式
- 速度・圧力・面流束の修正

を方程式ファミリー非依存に提供する。:mod:`xkep_cae_fluid.incompressible` の
:class:`~xkep_cae_fluid.incompressible.solver.NavierStokesFVMProcess` がこれを組み合わせる。

境界の扱い（面 b、owner P、外向き法線 n）:

- WALL: 速度 Dirichlet（既定 0、動く壁は ``velocity``）、質量流束 0、圧力ゼロ勾配
- INLET: 速度 Dirichlet、質量流束 ρ u_in·S、圧力ゼロ勾配
- OUTLET: 速度ゼロ勾配、質量流束 ρ u_P·S（圧力補正で修正）、圧力 Dirichlet
- SLIP（対称面）: 法線速度 0、接線ゼロ勾配。成分ごとの行列では owner 速度の接線射影を
  Dirichlet 値にする遅延評価（軸に平行な面では法線成分 0・接線成分ゼロ勾配と同じ）
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import sparse

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.assembly import (
    assemble_convection,
    assemble_diffusion,
    nonorthogonal_correction,
)
from xkep_cae_fluid.fvm.boundary import BCKind, BoundaryFaces, PatchBC, resolve_boundary
from xkep_cae_fluid.fvm.geometry import (
    _require_faces,
    face_decomposition,
    face_interpolation_weights,
)


class VelocityBCKind(Enum):
    """速度境界条件の種別."""

    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"
    SLIP = "slip"  # 対称面も同じ


_VKIND_CODE: dict[VelocityBCKind, int] = {
    VelocityBCKind.WALL: 0,
    VelocityBCKind.INLET: 1,
    VelocityBCKind.OUTLET: 2,
    VelocityBCKind.SLIP: 3,
}


@dataclass(frozen=True)
class VelocityPatchBC:
    """1 パッチの速度境界条件.

    Parameters
    ----------
    kind : VelocityBCKind
    velocity : tuple[float, float, float]
        WALL の壁速度（動く蓋など）、INLET の流入速度ベクトル
    pressure : float
        OUTLET の圧力
    """

    kind: VelocityBCKind = VelocityBCKind.WALL
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pressure: float = 0.0

    @staticmethod
    def wall(velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.WALL, velocity=velocity)

    @staticmethod
    def inlet(velocity: tuple[float, float, float]) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.INLET, velocity=velocity)

    @staticmethod
    def outlet(pressure: float = 0.0) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.OUTLET, pressure=pressure)

    @staticmethod
    def slip() -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.SLIP)


@dataclass(frozen=True)
class VelocityBoundaryFaces:
    """境界面ごとに展開した速度境界条件（長さは全て n_boundary_faces）."""

    faces: np.ndarray
    owner: np.ndarray
    kind: np.ndarray
    velocity: np.ndarray  # (n_b, 3)
    pressure: np.ndarray
    area: np.ndarray
    distance: np.ndarray
    normal: np.ndarray  # (n_b, 3) 外向き

    @property
    def n(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_wall(self) -> np.ndarray:
        return self.kind == _VKIND_CODE[VelocityBCKind.WALL]

    @property
    def is_inlet(self) -> np.ndarray:
        return self.kind == _VKIND_CODE[VelocityBCKind.INLET]

    @property
    def is_outlet(self) -> np.ndarray:
        return self.kind == _VKIND_CODE[VelocityBCKind.OUTLET]

    @property
    def is_slip(self) -> np.ndarray:
        return self.kind == _VKIND_CODE[VelocityBCKind.SLIP]

    @property
    def is_fixed_velocity(self) -> np.ndarray:
        return self.is_wall | self.is_inlet


def resolve_velocity_boundary(
    mesh: MeshData,
    bcs: Mapping[str, VelocityPatchBC],
    *,
    default: VelocityPatchBC | None = None,
) -> VelocityBoundaryFaces:
    """パッチ別の速度境界条件を境界面配列に展開する（未指定パッチは ``default``、既定 WALL）."""
    _require_faces(mesh)
    patches = dict(mesh.boundary_patches or {})
    unknown = sorted(set(bcs) - set(patches))
    if unknown:
        raise KeyError(f"メッシュに無いパッチ名: {unknown}（定義済み: {sorted(patches)}）")
    n_b = mesh.n_boundary_faces
    n_int = mesh.n_internal_faces
    faces = mesh.boundary_faces
    if default is None:
        default = VelocityPatchBC.wall()
    kind = np.full(n_b, _VKIND_CODE[default.kind], dtype=np.int64)
    vel = np.tile(np.asarray(default.velocity, dtype=np.float64), (n_b, 1))
    pres = np.full(n_b, float(default.pressure))
    for name, bc in bcs.items():
        local = np.asarray(patches[name], dtype=np.int64) - n_int
        kind[local] = _VKIND_CODE[bc.kind]
        vel[local] = np.asarray(bc.velocity, dtype=np.float64)
        pres[local] = float(bc.pressure)
    owner = mesh.face_owner[faces]
    normals = mesh.face_normals[faces]
    d_vec = mesh.face_centers[faces] - mesh.cell_centers[owner]
    distance = np.abs(np.sum(d_vec * normals, axis=1))
    return VelocityBoundaryFaces(
        faces=faces,
        owner=owner,
        kind=kind,
        velocity=vel,
        pressure=pres,
        area=mesh.face_areas[faces],
        distance=distance,
        normal=normals,
    )


def _boundary_faces_from_arrays(
    vb: VelocityBoundaryFaces, kind: np.ndarray, value: np.ndarray
) -> BoundaryFaces:
    n = vb.n
    return BoundaryFaces(
        faces=vb.faces,
        owner=vb.owner,
        kind=kind,
        value=value,
        flux=np.zeros(n),
        h=np.zeros(n),
        phi_inf=np.zeros(n),
        area=vb.area,
        distance=vb.distance,
    )


_DIRICHLET = 1
_ZERO_GRAD = 0


def component_boundary(vb: VelocityBoundaryFaces, u: np.ndarray, component: int) -> BoundaryFaces:
    """運動量成分 ``component`` のスカラー境界条件（Dirichlet / ゼロ勾配）を作る.

    WALL / INLET: Dirichlet（指定速度の成分）。SLIP: owner 速度の接線射影の成分を Dirichlet
    （遅延評価）。OUTLET: ゼロ勾配。
    """
    kind = np.full(vb.n, _ZERO_GRAD, dtype=np.int64)
    value = np.zeros(vb.n)
    fixed = vb.is_fixed_velocity
    kind[fixed] = _DIRICHLET
    value[fixed] = vb.velocity[fixed, component]
    slip = vb.is_slip
    if np.any(slip):
        u_p = u[vb.owner[slip]]
        n = vb.normal[slip]
        tangential = u_p - np.sum(u_p * n, axis=1)[:, None] * n
        kind[slip] = _DIRICHLET
        value[slip] = tangential[:, component]
    return _boundary_faces_from_arrays(vb, kind, value)


def pressure_boundary(vb: VelocityBoundaryFaces, *, correction: bool = False) -> BoundaryFaces:
    """圧力（``correction=False``）/ 圧力補正（``True``、値 0）の境界条件: OUTLET だけ Dirichlet."""
    kind = np.where(vb.is_outlet, _DIRICHLET, _ZERO_GRAD).astype(np.int64)
    value = np.zeros(vb.n) if correction else np.where(vb.is_outlet, vb.pressure, 0.0)
    return _boundary_faces_from_arrays(vb, kind, value)


def boundary_mass_flux(
    mesh: MeshData, vb: VelocityBoundaryFaces, u: np.ndarray, rho: float
) -> np.ndarray:
    """境界面の質量流束（外向き正）(n_boundary_faces,): INLET は指定速度、OUTLET は owner 速度、他は 0."""
    s_b = vb.normal * vb.area[:, None]
    out = np.zeros(vb.n)
    inl = vb.is_inlet
    out[inl] = rho * np.sum(vb.velocity[inl] * s_b[inl], axis=1)
    o = vb.is_outlet
    out[o] = rho * np.sum(u[vb.owner[o]] * s_b[o], axis=1)
    return out


def _touching_faces(mesh: MeshData, blocked: np.ndarray | None) -> np.ndarray:
    touch = np.zeros(mesh.n_faces, dtype=bool)
    if blocked is None:
        return touch
    blk = np.asarray(blocked, dtype=bool)
    touch = blk[mesh.face_owner].copy()
    n_int = mesh.n_internal_faces
    touch[:n_int] |= blk[mesh.face_neighbour]
    return touch


def assemble_momentum(
    mesh: MeshData,
    *,
    component: int,
    u: np.ndarray,
    mass_flux: np.ndarray,
    mu: float | np.ndarray,
    vb: VelocityBoundaryFaces,
    grad_p: np.ndarray,
    rho: float,
    alpha: float = 1.0,
    source: np.ndarray | None = None,
    drag: np.ndarray | None = None,
    dt: float = 0.0,
    u_old: np.ndarray | None = None,
    blocked: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """運動量方程式（成分 ``component``）の陰的緩和付き係数行列を組む.

    ρ ∂u/∂t + ∇·(ṁ u) − ∇·(μ∇u) = −∂p/∂x_c + S − D u（D: 抵抗係数 [kg/(m³ s)]、Brinkman なら μ/K）

    Parameters
    ----------
    u : np.ndarray
        現在の速度 (n_cells, 3)（緩和項・SLIP の接線射影・非直交補正に使う）
    mass_flux : np.ndarray
        面質量流束 (n_faces,)
    grad_p : np.ndarray
        セル圧力勾配 (n_cells, 3)
    alpha : float
        陰的緩和係数（対角を α で割り、(1−α)/α a_P u_prev を右辺へ）
    blocked : np.ndarray | None
        固体セル (n_cells,) bool。速度 0 を強制し、接する面の流束は無視

    Returns
    -------
    (A, b, a_p, sum_nb) : 係数行列、右辺、緩和後の対角 (n_cells,)、隣接係数の絶対値和 (n_cells,)
    """
    _require_faces(mesh)
    n = mesh.n_cells
    bf = component_boundary(vb, u, component)
    mf = np.asarray(mass_flux, dtype=np.float64).copy()
    touch = _touching_faces(mesh, blocked)
    mf[touch] = 0.0
    A_d, b_d = assemble_diffusion(mesh, mu, bf)
    A_c, b_c = assemble_convection(mesh, mf, bf)
    A = (A_d + A_c).tocsr()
    b = b_d + b_c + nonorthogonal_correction(mesh, u[:, component], mu, bf)
    vol = mesh.cell_volumes
    b = b - grad_p[:, component] * vol
    if source is not None:
        b = b + np.asarray(source, dtype=np.float64) * vol
    diag_extra = np.zeros(n)
    if drag is not None:
        diag_extra += np.asarray(drag, dtype=np.float64) * vol
    if dt > 0.0 and u_old is not None:
        coeff = rho * vol / dt
        diag_extra += coeff
        b = b + coeff * np.asarray(u_old, dtype=np.float64)[:, component]
    A = (A + sparse.diags(diag_extra)).tocsr()
    diag = np.asarray(A.diagonal(), dtype=np.float64)
    off = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
    if alpha < 1.0:
        a_p = diag / alpha
        b = b + (1.0 - alpha) / alpha * diag * u[:, component]
        A = (A + sparse.diags(a_p - diag)).tocsr()
    else:
        a_p = diag
    if blocked is not None:
        blk = np.asarray(blocked, dtype=bool)
        if np.any(blk):
            A = A.tolil()
            idx = np.flatnonzero(blk)
            for i in idx:
                A.rows[i] = [int(i)]
                A.data[i] = [1.0]
            A = A.tocsr()
            b[idx] = 0.0
            a_p[idx] = np.inf
            off[idx] = 0.0
    return A, b, a_p, off


def rhie_chow_mass_flux(
    mesh: MeshData,
    u: np.ndarray,
    p: np.ndarray,
    d_cells: np.ndarray,
    grad_p: np.ndarray,
    vb: VelocityBoundaryFaces,
    rho: float,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    """Rhie–Chow 補間の面質量流束 (n_faces,).

    ṁ_f = ρ [ ū_f·S_f − D_f ( (p_N − p_P)|E_f|/d_PN − (∇p)_f·E_f ) ]、D_f = interp(V_P/a_P)。
    境界面は :func:`boundary_mass_flux`。固体に接する面は 0。
    """
    _require_faces(mesh)
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    w = face_interpolation_weights(mesh)
    nd = mesh.face_normals.shape[1]
    s_f = mesh.face_normals[:n_int, :nd] * mesh.face_areas[:n_int, None]
    u_f = w[:, None] * u[owner, :nd] + (1.0 - w)[:, None] * u[nb, :nd]
    e_mag, _t, d_pn = face_decomposition(mesh)
    d_vec = mesh.cell_centers[nb, :nd] - mesh.cell_centers[owner, :nd]
    e_vec = d_vec / d_pn[:, None]
    d_f = w * d_cells[owner] + (1.0 - w) * d_cells[nb]
    grad_f = w[:, None] * grad_p[owner, :nd] + (1.0 - w)[:, None] * grad_p[nb, :nd]
    corr = d_f * ((p[nb] - p[owner]) * e_mag / d_pn - e_mag * np.sum(grad_f * e_vec, axis=1))
    flux = np.zeros(mesh.n_faces)
    flux[:n_int] = rho * (np.sum(u_f * s_f, axis=1) - corr)
    flux[vb.faces] = boundary_mass_flux(mesh, vb, u, rho)
    flux[_touching_faces(mesh, blocked)] = 0.0
    return flux


def pressure_correction_coefficients(
    mesh: MeshData,
    d_cells: np.ndarray,
    vb: VelocityBoundaryFaces,
    rho: float,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """圧力補正の面係数 (内部面 (n_int,), 境界面 (n_b,)): ṁ'_f = −a_f (p'_N − p'_P)."""
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    w = face_interpolation_weights(mesh)
    e_mag, _t, d_pn = face_decomposition(mesh)
    d_f = w * d_cells[owner] + (1.0 - w) * d_cells[nb]
    a_int = rho * d_f * e_mag / d_pn
    a_b = np.zeros(vb.n)
    o = vb.is_outlet
    safe = np.where(vb.distance > 0, vb.distance, 1.0)
    a_b[o] = rho * d_cells[vb.owner[o]] * vb.area[o] / safe[o]
    touch = _touching_faces(mesh, blocked)
    a_int[touch[:n_int]] = 0.0
    a_b[touch[vb.faces]] = 0.0
    return a_int, a_b


def assemble_pressure_correction(
    mesh: MeshData,
    mass_flux: np.ndarray,
    a_int: np.ndarray,
    a_b: np.ndarray,
    vb: VelocityBoundaryFaces,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """圧力補正方程式 Σ_f a_f (p'_P − p'_N) = −Σ_f ṁ_f を組む.

    OUTLET 面は p' = 0 の Dirichlet（係数 a_b）。Dirichlet 面が無い（閉じた領域）ときは
    セル 0 を基準（p' = 0）にする。

    Returns
    -------
    (A, b, imbalance) : 行列、右辺、各セルの質量不整合 Σ_f ṁ_f (n_cells,)
    """
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    imbalance = np.zeros(n)
    np.add.at(imbalance, mesh.face_owner, mass_flux)
    np.add.at(imbalance, nb, -mass_flux[:n_int])
    diag = np.zeros(n)
    np.add.at(diag, owner, a_int)
    np.add.at(diag, nb, a_int)
    np.add.at(diag, vb.owner, a_b)
    rows = np.concatenate([owner, nb, np.arange(n)])
    cols = np.concatenate([nb, owner, np.arange(n)])
    vals = np.concatenate([-a_int, -a_int, diag])
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    b = -imbalance
    if not np.any(a_b > 0):
        A = A.tolil()
        A.rows[0] = [0]
        A.data[0] = [1.0]
        A = A.tocsr()
        b = b.copy()
        b[0] = 0.0
    zero = np.flatnonzero(diag == 0.0)
    if zero.size:
        A = A.tolil()
        for i in zero:
            A.rows[i] = [int(i)]
            A.data[i] = [1.0]
        A = A.tocsr()
        b = b.copy()
        b[zero] = 0.0
    return A, b, imbalance


def correct_mass_flux(
    mesh: MeshData,
    mass_flux: np.ndarray,
    p_prime: np.ndarray,
    a_int: np.ndarray,
    a_b: np.ndarray,
    vb: VelocityBoundaryFaces,
) -> np.ndarray:
    """面質量流束を p' で修正する（内部面 −a_f (p'_N − p'_P)、OUTLET 面 +a_b p'_P）."""
    n_int = mesh.n_internal_faces
    out = np.asarray(mass_flux, dtype=np.float64).copy()
    out[:n_int] -= a_int * (p_prime[mesh.face_neighbour] - p_prime[mesh.face_owner[:n_int]])
    out[vb.faces] += a_b * p_prime[vb.owner]
    return out


def velocity_patch_from_kind(kind: str, **kwargs: object) -> VelocityPatchBC:
    """文字列（wall / inlet / outlet / slip / symmetry）から :class:`VelocityPatchBC` を作る."""
    key = kind.strip().lower()
    if key == "wall":
        return VelocityPatchBC.wall(kwargs.get("velocity", (0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    if key == "inlet":
        return VelocityPatchBC.inlet(kwargs["velocity"])  # type: ignore[arg-type]
    if key == "outlet":
        return VelocityPatchBC.outlet(float(kwargs.get("pressure", 0.0)))  # type: ignore[arg-type]
    if key in ("slip", "symmetry"):
        return VelocityPatchBC.slip()
    raise ValueError(f"未知の速度境界 {kind!r}")


def thermal_boundary(mesh: MeshData, bcs: Mapping[str, PatchBC | None]) -> BoundaryFaces:
    """温度の境界条件（None は断熱）を展開する."""
    return resolve_boundary(mesh, {k: v for k, v in bcs.items() if v is not None})


__all__ = [
    "VelocityBCKind",
    "VelocityPatchBC",
    "VelocityBoundaryFaces",
    "resolve_velocity_boundary",
    "component_boundary",
    "pressure_boundary",
    "boundary_mass_flux",
    "assemble_momentum",
    "rhie_chow_mass_flux",
    "pressure_correction_coefficients",
    "assemble_pressure_correction",
    "correct_mass_flux",
    "velocity_patch_from_kind",
    "thermal_boundary",
    "BCKind",
]
