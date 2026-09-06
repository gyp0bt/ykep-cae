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
- OUTFLOW（対流流出）: 速度・圧力ともゼロ勾配。質量流束は ρ u_P·S を他の境界の流出入と
  釣り合うようスケーリング（全体質量保存を境界で強制、p' はゼロ勾配。圧力の基準は別途）
- SLIP（対称面）: 法線速度 0、接線ゼロ勾配。軸に平行な面では成分ごとに陰的（法線成分 Dirichlet 0、
  接線成分ゼロ勾配）、傾いた面では owner 速度の接線射影を Dirichlet 値にする遅延評価

対流は 1 次風上（陰的）+ TVD 遅延補正（``convection="tvd"``）、時間項は Euler / BDF2
（``u_old2``）。内部セルの固定速度（``fixed_mask`` / ``fixed_velocity``）と圧力補正の
ピン留め（``pinned``）で、領域内部の吐出・吸入（InternalCellBC 相当）を表す。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import sparse

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.assembly import (
    assemble_convection,
    assemble_diffusion,
    convection_correction,
    nonorthogonal_correction,
    time_derivative_terms,
)
from xkep_cae_fluid.fvm.boundary import BCKind, BoundaryFaces, PatchBC, resolve_boundary
from xkep_cae_fluid.fvm.geometry import (
    _require_faces,
    face_decomposition,
    face_interpolation_weights,
    neighbour_centers,
)

logger = logging.getLogger(__name__)


class VelocityBCKind(Enum):
    """速度境界条件の種別."""

    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"  # 圧力指定の流出
    SLIP = "slip"  # 対称面も同じ
    OUTFLOW = "outflow"  # 対流流出（速度・圧力ゼロ勾配、流束を流入と釣り合わせる）


_VKIND_CODE: dict[VelocityBCKind, int] = {
    VelocityBCKind.WALL: 0,
    VelocityBCKind.INLET: 1,
    VelocityBCKind.OUTLET: 2,
    VelocityBCKind.SLIP: 3,
    VelocityBCKind.OUTFLOW: 4,
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
    angular_velocity : tuple[float, float, float]
        剛体回転の角速度 ω [rad/s]。面ごとの速度が ``velocity + ω × (x_f − center)`` になる
        （回転するバレル・インペラ。``.inp`` では参照節点の自由度 4-6 + ``*MPC``）
    center : tuple[float, float, float]
        回転中心（参照節点の座標）[m]
    """

    kind: VelocityBCKind = VelocityBCKind.WALL
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pressure: float = 0.0
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_rotating(self) -> bool:
        return any(w != 0.0 for w in self.angular_velocity)

    @staticmethod
    def wall(velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.WALL, velocity=velocity)

    @staticmethod
    def rotating_wall(
        angular_velocity: tuple[float, float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> VelocityPatchBC:
        """剛体回転する壁 u(x) = velocity + ω × (x − center)（no-slip）."""
        return VelocityPatchBC(
            VelocityBCKind.WALL,
            velocity=velocity,
            angular_velocity=angular_velocity,
            center=center,
        )

    @staticmethod
    def inlet(velocity: tuple[float, float, float]) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.INLET, velocity=velocity)

    @staticmethod
    def outlet(pressure: float = 0.0) -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.OUTLET, pressure=pressure)

    @staticmethod
    def slip() -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.SLIP)

    @staticmethod
    def outflow() -> VelocityPatchBC:
        return VelocityPatchBC(VelocityBCKind.OUTFLOW)


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
    def is_outflow(self) -> np.ndarray:
        return self.kind == _VKIND_CODE[VelocityBCKind.OUTFLOW]

    @property
    def is_fixed_velocity(self) -> np.ndarray:
        return self.is_wall | self.is_inlet


def resolve_velocity_boundary(
    mesh: MeshData,
    bcs: Mapping[str, VelocityPatchBC],
    *,
    default: VelocityPatchBC | None = None,
) -> VelocityBoundaryFaces:
    """パッチ別の速度境界条件を境界面配列に展開する（未指定パッチは ``default``、既定 WALL）.

    ``angular_velocity`` を持つパッチは面ごとに ``velocity + ω × (x_f − center)`` を割り当てる
    （剛体回転する壁）。回転面の法線速度が接線速度に対して無視できないときは警告する
    （回転軸まわりの回転面でなければ壁が「吹く」ことになり、質量流束 0 と矛盾するため）。
    """
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
    rotating: list[tuple[str, np.ndarray]] = []
    for name, bc in bcs.items():
        local = np.asarray(patches[name], dtype=np.int64) - n_int
        kind[local] = _VKIND_CODE[bc.kind]
        vel[local] = np.asarray(bc.velocity, dtype=np.float64)
        pres[local] = float(bc.pressure)
        if bc.is_rotating:
            nd_ = mesh.face_centers.shape[1]
            r = np.zeros((local.size, 3))
            r[:, :nd_] = (
                mesh.face_centers[faces[local], :nd_]
                - np.asarray(bc.center, dtype=np.float64)[:nd_]
            )
            vel[local] = vel[local] + np.cross(np.asarray(bc.angular_velocity, dtype=np.float64), r)
            rotating.append((name, local))
    owner = mesh.face_owner[faces]
    normals = mesh.face_normals[faces]
    for name, local in rotating:
        u_n = np.abs(np.sum(vel[local] * normals[local], axis=1))
        u_mag = np.linalg.norm(vel[local], axis=1)
        bad = u_n > 1e-6 * np.maximum(u_mag, 1e-300)
        if np.any(bad):
            logger.warning(
                "回転パッチ %s: 面 %d 枚で法線速度が接線速度の 1e-6 倍を超えます"
                "（最大 %.3e m/s）。回転軸まわりの回転面になっているか確認してください",
                name,
                int(np.count_nonzero(bad)),
                float(np.max(u_n)),
            )
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

    WALL / INLET: Dirichlet（指定速度の成分）。OUTLET / OUTFLOW: ゼロ勾配。
    SLIP: 面法線が軸 ``component`` に平行なら Dirichlet 0、直交ならゼロ勾配（どちらも陰的）。
    傾いた対称面では owner 速度の接線射影の成分を Dirichlet にする遅延評価。
    """
    kind = np.full(vb.n, _ZERO_GRAD, dtype=np.int64)
    value = np.zeros(vb.n)
    fixed = vb.is_fixed_velocity
    kind[fixed] = _DIRICHLET
    value[fixed] = vb.velocity[fixed, component]
    slip = vb.is_slip
    if np.any(slip):
        n = vb.normal[slip]
        nc = np.abs(n[:, component])
        normal_axis = nc > 1.0 - 1e-12
        tangent_axis = nc < 1e-12
        tilted = ~(normal_axis | tangent_axis)
        u_p = u[vb.owner[slip]]
        tangential = u_p - np.sum(u_p * n, axis=1)[:, None] * n
        k_s = np.where(tangent_axis, _ZERO_GRAD, _DIRICHLET)
        v_s = np.where(tilted, tangential[:, component], 0.0)
        kind[slip] = k_s
        value[slip] = v_s
    return _boundary_faces_from_arrays(vb, kind, value)


def pressure_boundary(vb: VelocityBoundaryFaces, *, correction: bool = False) -> BoundaryFaces:
    """圧力（``correction=False``）/ 圧力補正（``True``、値 0）の境界条件: OUTLET だけ Dirichlet."""
    kind = np.where(vb.is_outlet, _DIRICHLET, _ZERO_GRAD).astype(np.int64)
    value = np.zeros(vb.n) if correction else np.where(vb.is_outlet, vb.pressure, 0.0)
    return _boundary_faces_from_arrays(vb, kind, value)


def boundary_mass_flux(
    mesh: MeshData, vb: VelocityBoundaryFaces, u: np.ndarray, rho: float
) -> np.ndarray:
    """境界面の質量流束（外向き正）(n_boundary_faces,).

    INLET は指定速度、OUTLET は owner 速度、壁・対称面は 0。OUTFLOW（対流流出）は owner 速度の
    流束を、他の全境界の正味流入 −Σ_{other} ṁ_b と釣り合うようスケーリングする（流束が
    まだ無い初期状態では面積比で配分）。OUTFLOW 面が無ければ質量の釣り合いは圧力補正に任せる。
    """
    s_b = vb.normal * vb.area[:, None]
    out = np.zeros(vb.n)
    inl = vb.is_inlet
    out[inl] = rho * np.sum(vb.velocity[inl] * s_b[inl], axis=1)
    o = vb.is_outlet
    out[o] = rho * np.sum(u[vb.owner[o]] * s_b[o], axis=1)
    cv = vb.is_outflow
    if np.any(cv):
        raw = rho * np.sum(u[vb.owner[cv]] * s_b[cv], axis=1)
        target = -float(np.sum(out[~cv]))
        total = float(np.sum(raw))
        if total > 0.0 and target > 0.0:
            out[cv] = raw * (target / total)
        elif target > 0.0:
            out[cv] = target * vb.area[cv] / float(np.sum(vb.area[cv]))
        else:
            out[cv] = 0.0
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
    u_old2: np.ndarray | None = None,
    convection: str = "upwind",
    limiter: str = "van_leer",
    fixed_mask: np.ndarray | None = None,
    fixed_velocity: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """運動量方程式（成分 ``component``）の陰的緩和付き係数行列を組む.

    ρ ∂u/∂t + ∇·(ṁ u) − ∇·(μ∇u) = −∂p/∂x_c + S − D u（D: 抵抗係数 [kg/(m³ s)]、Brinkman なら μ/K）

    対流は有界形（``assemble_convection(bounded=True)``: 途中反復や内部吐出・吸入セルの
    質量不整合を対角から差し引く）。

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
    u_old2 : np.ndarray | None
        前々ステップの速度 (n_cells, 3)。与えると時間項が BDF2
    convection, limiter :
        ``upwind`` / ``tvd``（現在の速度 ``u`` で遅延補正）とリミッタ。``none`` は対流項なし
        （Stokes 流れ。Re ≪ 1 の押出・クリープ流れで反復を線形にする）
    fixed_mask, fixed_velocity :
        速度を固定する内部セル (n_cells,) bool とその値 (n_cells, 3)（吐出口など）。
        固体セルと違い、接する面の流束は消さない

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
    b = b_d + nonorthogonal_correction(mesh, u[:, component], mu, bf)
    if convection.lower() == "none":
        # Stokes（クリープ流れ）: 対流項を落とす。面流束は圧力補正だけが使う
        A = A_d.tocsr()
    else:
        A_c, b_c = assemble_convection(mesh, mf, bf, bounded=True)
        A = (A_d + A_c).tocsr()
        b = b + b_c + convection_correction(mesh, u[:, component], mf, bf, convection, limiter)
    vol = mesh.cell_volumes
    b = b - grad_p[:, component] * vol
    if source is not None:
        b = b + np.asarray(source, dtype=np.float64) * vol
    diag_extra = np.zeros(n)
    if drag is not None:
        diag_extra += np.asarray(drag, dtype=np.float64) * vol
    if dt > 0.0 and u_old is not None:
        old2 = None if u_old2 is None else np.asarray(u_old2, dtype=np.float64)[:, component]
        coeff, rhs_t = time_derivative_terms(
            mesh, rho, dt, np.asarray(u_old, dtype=np.float64)[:, component], old2
        )
        diag_extra += coeff
        b = b + rhs_t
    A = (A + sparse.diags(diag_extra)).tocsr()
    diag = np.asarray(A.diagonal(), dtype=np.float64)
    off = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
    if alpha < 1.0:
        a_p = diag / alpha
        b = b + (1.0 - alpha) / alpha * diag * u[:, component]
        A = (A + sparse.diags(a_p - diag)).tocsr()
    else:
        a_p = diag
    fix = np.zeros(n, dtype=bool)
    fix_val = np.zeros(n)
    if blocked is not None:
        fix |= np.asarray(blocked, dtype=bool)
    if fixed_mask is not None:
        fm = np.asarray(fixed_mask, dtype=bool)
        if fixed_velocity is None:
            raise ValueError("fixed_mask には fixed_velocity (n_cells, 3) が必要")
        fix_val[fm] = np.asarray(fixed_velocity, dtype=np.float64)[fm, component]
        fix |= fm
    if np.any(fix):
        idx = np.flatnonzero(fix)
        A = fix_rows(A, idx)
        b = b.copy()
        b[idx] = fix_val[idx]
        a_p[idx] = np.inf
        off[idx] = 0.0
    return A, b, a_p, off


def fix_rows(A: sparse.spmatrix, idx: np.ndarray) -> sparse.csr_matrix:
    """行 ``idx`` を単位行（対角 1、非対角 0）に置き換えた CSR 行列を返す（右辺は呼び出し側で）."""
    if idx.size == 0:
        return sparse.csr_matrix(A)
    n = A.shape[0]
    mask = np.ones(n)
    mask[idx] = 0.0
    out = sparse.diags(mask) @ sparse.csr_matrix(A)
    out = out + sparse.coo_matrix((np.ones(idx.size), (idx, idx)), shape=(n, n))
    return sparse.csr_matrix(out)


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
    d_vec = neighbour_centers(mesh) - mesh.cell_centers[owner, :nd]
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


def pressure_correction_nonorthogonal(
    mesh: MeshData,
    d_cells: np.ndarray,
    grad_pp: np.ndarray,
    rho: float,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    """圧力補正の非直交（遅延）補正の面流束 c_f = ρ D_f (∇p')_f·T_f (n_internal_faces,).

    圧力補正の面流束 ṁ'_f = −ρ D_f ∇p'_f·S_f を over-relaxed 分解 S_f = E_f + T_f で
    −a_f (p'_N − p'_P) − c_f に分け、E_f 部分を陰的（:func:`pressure_correction_coefficients`）、
    T_f 部分 c_f を前回の p' の勾配（:func:`~xkep_cae_fluid.fvm.geometry.cell_gradient_lsq`）で
    陽的に評価する。直交メッシュではゼロ。固体に接する面はゼロ。
    """
    n_int = mesh.n_internal_faces
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    w = face_interpolation_weights(mesh)
    _e_mag, t_vec, _d_pn = face_decomposition(mesh)
    nd = t_vec.shape[1]
    d_f = w * d_cells[owner] + (1.0 - w) * d_cells[nb]
    grad_f = w[:, None] * grad_pp[owner, :nd] + (1.0 - w)[:, None] * grad_pp[nb, :nd]
    c = rho * d_f * np.sum(grad_f * t_vec, axis=1)
    c[_touching_faces(mesh, blocked)[:n_int]] = 0.0
    return c


def assemble_pressure_correction(
    mesh: MeshData,
    mass_flux: np.ndarray,
    a_int: np.ndarray,
    a_b: np.ndarray,
    vb: VelocityBoundaryFaces,
    pinned: np.ndarray | None = None,
    explicit_flux: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """圧力補正方程式 Σ_f a_f (p'_P − p'_N) = −Σ_f ṁ_f + Σ_f c_f を組む.

    OUTLET 面は p' = 0 の Dirichlet（係数 a_b）。``pinned`` のセル（内部の吐出・吸入セル）は
    p' = 0 に固定（質量の湧き出し・吸い込みを許す）。Dirichlet 面もピン留めセルも無い
    （閉じた領域）ときはセル 0 を基準（p' = 0）にする。
    ``explicit_flux`` は内部面の非直交補正流束 c_f（:func:`pressure_correction_nonorthogonal`。
    owner に +、neighbour に − で右辺へ。:func:`correct_mass_flux` にも同じ配列を渡す）。

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
    b = (-imbalance).copy()
    if explicit_flux is not None:
        np.add.at(b, owner, explicit_flux)
        np.add.at(b, nb, -explicit_flux)
    pin = np.zeros(n, dtype=bool)
    if pinned is not None:
        pin |= np.asarray(pinned, dtype=bool)
    if not np.any(a_b > 0) and not np.any(pin):
        pin[0] = True
    pin |= diag == 0.0
    idx = np.flatnonzero(pin)
    if idx.size:
        A = fix_rows(A, idx)
        b[idx] = 0.0
    return A, b, imbalance


def correct_mass_flux(
    mesh: MeshData,
    mass_flux: np.ndarray,
    p_prime: np.ndarray,
    a_int: np.ndarray,
    a_b: np.ndarray,
    vb: VelocityBoundaryFaces,
    explicit_flux: np.ndarray | None = None,
) -> np.ndarray:
    """面質量流束を p' で修正する（内部面 −a_f (p'_N − p'_P) − c_f、OUTLET 面 +a_b p'_P）.

    ``explicit_flux`` は :func:`assemble_pressure_correction` に渡した非直交補正流束 c_f
    （同じ配列を渡すと修正後の流束の発散が解いた線形系と厳密に整合する）。
    """
    n_int = mesh.n_internal_faces
    out = np.asarray(mass_flux, dtype=np.float64).copy()
    out[:n_int] -= a_int * (p_prime[mesh.face_neighbour] - p_prime[mesh.face_owner[:n_int]])
    if explicit_flux is not None:
        out[:n_int] -= explicit_flux
    out[vb.faces] += a_b * p_prime[vb.owner]
    return out


def velocity_patch_from_kind(kind: str, **kwargs: object) -> VelocityPatchBC:
    """文字列（wall / inlet / outlet / slip / symmetry / outflow）から :class:`VelocityPatchBC` を作る."""
    key = kind.strip().lower()
    if key == "wall":
        return VelocityPatchBC.wall(kwargs.get("velocity", (0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    if key == "inlet":
        return VelocityPatchBC.inlet(kwargs["velocity"])  # type: ignore[arg-type]
    if key == "outlet":
        return VelocityPatchBC.outlet(float(kwargs.get("pressure", 0.0)))  # type: ignore[arg-type]
    if key in ("slip", "symmetry"):
        return VelocityPatchBC.slip()
    if key in ("outflow", "convective"):
        return VelocityPatchBC.outflow()
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
    "pressure_correction_nonorthogonal",
    "correct_mass_flux",
    "fix_rows",
    "velocity_patch_from_kind",
    "thermal_boundary",
    "BCKind",
]


# ---------------------------------------------------------------------------
# 速度–圧力の連成組み立て（coupled）
# ---------------------------------------------------------------------------


def assemble_coupled(
    mesh: MeshData,
    *,
    u: np.ndarray,
    p: np.ndarray,
    mass_flux: np.ndarray,
    mu: float | np.ndarray,
    vb: VelocityBoundaryFaces,
    bp: BoundaryFaces,
    rho: float,
    source: np.ndarray | None = None,
    drag: np.ndarray | None = None,
    dt: float = 0.0,
    u_old: np.ndarray | None = None,
    blocked: np.ndarray | None = None,
    u_old2: np.ndarray | None = None,
    convection: str = "upwind",
    limiter: str = "van_leer",
    fixed_mask: np.ndarray | None = None,
    fixed_velocity: np.ndarray | None = None,
    pinned: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[tuple[sparse.csr_matrix, np.ndarray]]]:
    """速度 nd 成分と圧力を 1 つの線形系 [A  V∇; ρ Div(Rhie–Chow)] にまとめる（緩和なし）.

    運動量は :func:`assemble_momentum` と同じ係数（対流は前回の面流束で線形化、TVD と非直交補正は
    現在の速度で遅延補正、圧力勾配は最小二乗作用素 :func:`~xkep_cae_fluid.fvm.geometry.lsq_gradient_operator`
    で陰的）、連続は Rhie–Chow 流束（D_f = interp(V/a_P)、a_P は緩和前の対角）を u と p の両方について
    陰的に書く。Stokes（``convection="none"``）なら 1 回の直接解で厳密解に達し、対流があっても
    流束の Picard だけで収束する。OUTFLOW 面は未対応。

    Returns
    -------
    (A, b, d_cells, systems) : 全体行列（未知数の並びは u_0.. u_{nd−1}, p の順で各 n_cells）、右辺、
        Rhie–Chow の D_P（面流束の再評価に使う）、成分ごとの運動量 (A_c, b_c)（残差評価用。圧力勾配は含まない）
    """
    from xkep_cae_fluid.fvm.geometry import lsq_gradient_operator

    _require_faces(mesh)
    if np.any(vb.is_outflow):
        raise ValueError("coupled では OUTFLOW（対流流出）境界は使えません（PRESSURE 流出を使う）")
    n = mesh.n_cells
    n_int = mesh.n_internal_faces
    nd = mesh.face_normals.shape[1]
    vol = mesh.cell_volumes
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    touch = _touching_faces(mesh, blocked)
    mf = np.asarray(mass_flux, dtype=np.float64).copy()
    mf[touch] = 0.0
    fix = np.zeros(n, dtype=bool)
    fix_val = np.zeros((n, 3))
    if blocked is not None:
        fix |= np.asarray(blocked, dtype=bool)
    if fixed_mask is not None:
        fm = np.asarray(fixed_mask, dtype=bool)
        if fixed_velocity is None:
            raise ValueError("fixed_mask には fixed_velocity (n_cells, 3) が必要")
        fix_val[fm] = np.asarray(fixed_velocity, dtype=np.float64)[fm]
        fix |= fm

    # --- 運動量ブロック（成分ごと、緩和なし）---
    systems: list[tuple[sparse.csr_matrix, np.ndarray]] = []
    a_p = np.zeros((n, nd))
    for c in range(nd):
        bf = component_boundary(vb, u, c)
        A_d, b_d = assemble_diffusion(mesh, mu, bf)
        b = b_d + nonorthogonal_correction(mesh, u[:, c], mu, bf)
        if convection.lower() == "none":
            A = A_d.tocsr()
        else:
            A_c, b_c = assemble_convection(mesh, mf, bf, bounded=True)
            A = (A_d + A_c).tocsr()
            b = b + b_c + convection_correction(mesh, u[:, c], mf, bf, convection, limiter)
        if source is not None:
            b = b + np.asarray(source, dtype=np.float64)[:, c] * vol
        diag_extra = np.zeros(n)
        if drag is not None:
            diag_extra += np.asarray(drag, dtype=np.float64) * vol
        if dt > 0.0 and u_old is not None:
            old2 = None if u_old2 is None else np.asarray(u_old2, dtype=np.float64)[:, c]
            coeff, rhs_t = time_derivative_terms(
                mesh, rho, dt, np.asarray(u_old, dtype=np.float64)[:, c], old2
            )
            diag_extra += coeff
            b = b + rhs_t
        A = (A + sparse.diags(diag_extra)).tocsr()
        a_p[:, c] = np.asarray(A.diagonal(), dtype=np.float64)
        systems.append((A, b))
    ap_mean = np.mean(a_p, axis=1)
    d_cells = np.where(ap_mean > 0, vol / np.where(ap_mean > 0, ap_mean, 1.0), 0.0)
    if blocked is not None:
        d_cells = np.where(np.asarray(blocked, dtype=bool), 0.0, d_cells)

    # --- 圧力勾配作用素と Rhie–Chow 流束 ---
    G, g0 = lsq_gradient_operator(mesh, bp)
    w = face_interpolation_weights(mesh)
    e_mag, _t, d_pn = face_decomposition(mesh)
    d_vec = neighbour_centers(mesh) - mesh.cell_centers[owner, :nd]
    e_vec = d_vec / d_pn[:, None]
    s_f = mesh.face_normals[:n_int, :nd] * mesh.face_areas[:n_int, None]
    d_f = w * d_cells[owner] + (1.0 - w) * d_cells[nb]
    ok = ~touch[:n_int]
    rows_f = np.arange(n_int)
    sel_own = sparse.coo_matrix((w * ok, (rows_f, owner)), shape=(n_int, n)).tocsr()
    sel_nb = sparse.coo_matrix(((1.0 - w) * ok, (rows_f, nb)), shape=(n_int, n)).tocsr()
    # 内部面流束 ṁ_f = ρ[ ū_f·S_f − D_f (p_N − p_P)|E_f|/d_PN + D_f |E_f| (∇p)_f·e_f ]
    F_u = [
        sparse.coo_matrix(
            (
                np.concatenate([rho * w * s_f[:, c] * ok, rho * (1.0 - w) * s_f[:, c] * ok]),
                (np.concatenate([rows_f, rows_f]), np.concatenate([owner, nb])),
            ),
            shape=(n_int, n),
        ).tocsr()
        for c in range(nd)
    ]
    coef = rho * d_f * e_mag / d_pn * ok
    F_p = sparse.coo_matrix(
        (
            np.concatenate([coef, -coef]),
            (np.concatenate([rows_f, rows_f]), np.concatenate([owner, nb])),
        ),
        shape=(n_int, n),
    ).tocsr()
    rhs_f = np.zeros(n_int)
    scale_g = rho * d_f * e_mag * ok
    for c in range(nd):
        interp = sel_own @ G[c] + sel_nb @ G[c]
        F_p = F_p + sparse.diags(scale_g * e_vec[:, c]) @ interp
        rhs_f += scale_g * e_vec[:, c] * (sel_own @ g0[:, c] + sel_nb @ g0[:, c])
    # 発散（owner +、neighbour −）
    div = sparse.coo_matrix(
        (
            np.concatenate([np.ones(n_int), -np.ones(n_int)]),
            (np.concatenate([owner, nb]), np.concatenate([rows_f, rows_f])),
        ),
        shape=(n, n_int),
    ).tocsr()
    # 境界流束: INLET は既知（右辺）、OUTLET は ρ S·u_P（陰的）、壁・対称面は 0
    s_b = vb.normal[:, :nd] * vb.area[:, None]
    ok_b = ~touch[vb.faces]
    b_cont = -div @ rhs_f
    inl = vb.is_inlet & ok_b
    np.add.at(b_cont, vb.owner[inl], -rho * np.sum(vb.velocity[inl, :nd] * s_b[inl], axis=1))
    out_ = vb.is_outlet & ok_b
    B_u = [
        sparse.coo_matrix(
            (rho * s_b[out_, c], (vb.owner[out_], vb.owner[out_])), shape=(n, n)
        ).tocsr()
        for c in range(nd)
    ]

    # --- 全体行列 ---
    blocks: list[list[sparse.spmatrix | None]] = []
    b_full = np.zeros((nd + 1) * n)
    for c in range(nd):
        row: list[sparse.spmatrix | None] = [None] * (nd + 1)
        A_c, b_c = systems[c]
        row[c] = A_c
        row[nd] = sparse.diags(vol) @ G[c]
        blocks.append(row)
        b_full[c * n : (c + 1) * n] = b_c - vol * g0[:, c]
    last: list[sparse.spmatrix | None] = [div @ F_u[c] + B_u[c] for c in range(nd)]
    last.append(div @ F_p)
    blocks.append(last)
    A_full = sparse.bmat(blocks, format="csr")
    b_full[nd * n :] = b_cont

    # --- 固定行: 固体 / 内部固定セルの速度、圧力のピン留め ---
    pin = np.zeros(n, dtype=bool)
    if pinned is not None:
        pin |= np.asarray(pinned, dtype=bool)
    if blocked is not None:
        pin |= np.asarray(blocked, dtype=bool)
    if not np.any(bp.is_dirichlet) and not np.any(pin):
        pin[0] = True
    idx_list = []
    vals_list = []
    for c in range(nd):
        idx = np.flatnonzero(fix)
        idx_list.append(idx + c * n)
        vals_list.append(fix_val[idx, c])
    idx_list.append(np.flatnonzero(pin) + nd * n)
    vals_list.append(np.zeros(int(pin.sum())))
    idx_all = np.concatenate(idx_list)
    if idx_all.size:
        A_full = fix_rows(A_full, idx_all)
        b_full[idx_all] = np.concatenate(vals_list)
    return A_full, b_full, d_cells, systems


__all__.append("assemble_coupled")
