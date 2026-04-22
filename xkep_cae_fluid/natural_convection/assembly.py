"""SIMPLE法の方程式アセンブリ (FDM 疎行列).

3次元等間隔直交格子上で運動量方程式・圧力補正方程式・エネルギー方程式の
係数行列と右辺ベクトルを疎行列として組み立てる。

対流項: 1次風上差分 / TVD（van Leer, Superbee）遅延補正法
拡散項: 中心差分
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    InternalFaceBC,
    InternalFaceBCKind,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)

_INTERNAL_BC_PENALTY = 1e25
"""内部セル BC のペナルティ係数.

`solid_mask` で使っている 1e30 より小さく、運動量行列の条件数を過度に悪化
させない範囲で Dirichlet 拘束として機能する値。固体ペナルティと明確に
区別するため別定数とする。
"""


def _flat_index(i: np.ndarray, j: np.ndarray, k: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """(i, j, k) → フラットインデックス."""
    return i * (ny * nz) + j * nz + k


def _build_meshgrid(nx: int, ny: int, nz: int) -> tuple[np.ndarray, ...]:
    """全セルのメッシュグリッドを生成."""
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return ii.ravel(), jj.ravel(), kk.ravel()


def _is_solid(
    solid_mask: np.ndarray | None, i: np.ndarray, j: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """固体セルかどうかを判定."""
    if solid_mask is None:
        return np.zeros(len(i), dtype=bool)
    return solid_mask[i, j, k]


def _limiter_van_leer(r: np.ndarray) -> np.ndarray:
    """van Leer リミッター: ψ(r) = (r + |r|) / (1 + |r|)."""
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _limiter_superbee(r: np.ndarray) -> np.ndarray:
    """Superbee リミッター: ψ(r) = max(0, min(2r,1), min(r,2))."""
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def _tvd_deferred_correction(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    rho_factor: float,
    limiter: str,
    solid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """TVD 対流スキームの遅延補正ソース項を計算.

    1次風上差分を行列に保持したまま、TVD補正をRHSソースとして追加する。
    各面の補正: ΔF = F * 0.5 * ψ(r) * (φ_D - φ_U)

    Parameters
    ----------
    phi : np.ndarray
        輸送されるスカラー場 (nx, ny, nz)
    u, v, w : np.ndarray
        速度場 (nx, ny, nz)
    rho_factor : float
        密度係数（運動量: rho, エネルギー: rho*Cp）
    limiter : str
        "van_leer" or "superbee"

    Returns
    -------
    np.ndarray
        補正ソース項 (nx*ny*nz,)
    """
    psi_fn = _limiter_van_leer if limiter == "van_leer" else _limiter_superbee
    correction = np.zeros((nx, ny, nz))

    # x方向面 (i と i+1 の間)
    if nx > 1:
        u_face = 0.5 * (u[:-1] + u[1:])  # (nx-1, ny, nz)
        F = rho_factor * u_face / dx
        delta = phi[1:] - phi[:-1]  # φ_right - φ_left
        safe_delta = np.where(np.abs(delta) > 1e-30, delta, np.sign(delta) * 1e-30)
        safe_delta = np.where(safe_delta == 0, 1e-30, safe_delta)

        # F > 0: upwind=left(i), UU=left-1(i-1)
        delta_up_pos = np.zeros_like(delta)
        if nx > 2:
            delta_up_pos[1:] = phi[1:-1] - phi[:-2]  # φ[i] - φ[i-1] for face i

        # F < 0: upwind=right(i+1), UU=right+1(i+2)
        delta_up_neg = np.zeros_like(delta)
        if nx > 2:
            delta_up_neg[:-1] = phi[1:-1] - phi[2:]  # φ[i+1] - φ[i+2] for face i

        r = np.where(F > 0, delta_up_pos / safe_delta, delta_up_neg / (-safe_delta))
        psi = psi_fn(r)
        phi_D_minus_U = np.where(F > 0, delta, -delta)
        face_corr = F * 0.5 * psi * phi_D_minus_U
        correction[:-1] += face_corr
        correction[1:] -= face_corr

    # y方向面 (j と j+1 の間)
    if ny > 1:
        v_face = 0.5 * (v[:, :-1] + v[:, 1:])
        F = rho_factor * v_face / dy
        delta = phi[:, 1:] - phi[:, :-1]
        safe_delta = np.where(np.abs(delta) > 1e-30, delta, np.sign(delta) * 1e-30)
        safe_delta = np.where(safe_delta == 0, 1e-30, safe_delta)

        delta_up_pos = np.zeros_like(delta)
        if ny > 2:
            delta_up_pos[:, 1:] = phi[:, 1:-1] - phi[:, :-2]

        delta_up_neg = np.zeros_like(delta)
        if ny > 2:
            delta_up_neg[:, :-1] = phi[:, 1:-1] - phi[:, 2:]

        r = np.where(F > 0, delta_up_pos / safe_delta, delta_up_neg / (-safe_delta))
        psi = psi_fn(r)
        phi_D_minus_U = np.where(F > 0, delta, -delta)
        face_corr = F * 0.5 * psi * phi_D_minus_U
        correction[:, :-1] += face_corr
        correction[:, 1:] -= face_corr

    # z方向面 (k と k+1 の間)
    if nz > 1:
        w_face = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        F = rho_factor * w_face / dz
        delta = phi[:, :, 1:] - phi[:, :, :-1]
        safe_delta = np.where(np.abs(delta) > 1e-30, delta, np.sign(delta) * 1e-30)
        safe_delta = np.where(safe_delta == 0, 1e-30, safe_delta)

        delta_up_pos = np.zeros_like(delta)
        if nz > 2:
            delta_up_pos[:, :, 1:] = phi[:, :, 1:-1] - phi[:, :, :-2]

        delta_up_neg = np.zeros_like(delta)
        if nz > 2:
            delta_up_neg[:, :, :-1] = phi[:, :, 1:-1] - phi[:, :, 2:]

        r = np.where(F > 0, delta_up_pos / safe_delta, delta_up_neg / (-safe_delta))
        psi = psi_fn(r)
        phi_D_minus_U = np.where(F > 0, delta, -delta)
        face_corr = F * 0.5 * psi * phi_D_minus_U
        correction[:, :, :-1] += face_corr
        correction[:, :, 1:] -= face_corr

    # 固体セル: 補正なし
    if solid_mask is not None:
        correction[solid_mask] = 0.0

    return correction.ravel()


def _apply_internal_bc_momentum(
    bcs: tuple[InternalFaceBC, ...],
    component: str,
    flat_idx: np.ndarray,
    diag: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """運動量方程式に InternalFaceBC のペナルティを適用.

    INLET セルに対して `diag += BIG`, `rhs += BIG * velocity[comp_idx]` を加え
    Dirichlet 条件として `u = velocity` を強制する。OUTLET は何もしない
    （ゼロ勾配相当、圧力補正で圧力が決まる）。
    """
    if not bcs:
        return
    comp_idx = {"u": 0, "v": 1, "w": 2}[component]
    for bc in bcs:
        if bc.kind != InternalFaceBCKind.INLET:
            continue
        if bc.mask is None or not np.any(bc.mask):
            continue
        cells = flat_idx.reshape(bc.mask.shape)[bc.mask]
        diag[cells] += _INTERNAL_BC_PENALTY
        rhs[cells] += _INTERNAL_BC_PENALTY * bc.velocity[comp_idx]


def _apply_internal_bc_pressure(
    bcs: tuple[InternalFaceBC, ...],
    flat_idx: np.ndarray,
    diag: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """圧力補正方程式に InternalFaceBC のペナルティを適用.

    INLET/OUTLET ともに `p' = 0` ピン留めする。INLET では速度が強制されている
    ため補正を禁じ、OUTLET は基準圧力位置として使う（p は実際の値ではなく、
    Boussinesq 近似下での相対圧なのでゼロピン留めで問題ない）。
    """
    if not bcs:
        return
    for bc in bcs:
        if bc.mask is None or not np.any(bc.mask):
            continue
        cells = flat_idx.reshape(bc.mask.shape)[bc.mask]
        diag[cells] = 1e30
        rhs[cells] = 0.0


def _apply_internal_bc_energy(
    bcs: tuple[InternalFaceBC, ...],
    flat_idx: np.ndarray,
    diag: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """エネルギー方程式に InternalFaceBC のペナルティを適用.

    INLET で `temperature` が指定されていれば、対角ペナルティで強制する。
    OUTLET と `temperature=None` の INLET は変更しない。
    """
    if not bcs:
        return
    for bc in bcs:
        if bc.kind != InternalFaceBCKind.INLET:
            continue
        if bc.temperature is None:
            continue
        if bc.mask is None or not np.any(bc.mask):
            continue
        cells = flat_idx.reshape(bc.mask.shape)[bc.mask]
        diag[cells] += _INTERNAL_BC_PENALTY
        rhs[cells] += _INTERNAL_BC_PENALTY * float(bc.temperature)


def _inlet_mask_union(
    bcs: tuple[InternalFaceBC, ...],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """全 INLET BC のマスクを論理和で結合."""
    combined = np.zeros(shape, dtype=bool)
    for bc in bcs:
        if bc.kind != InternalFaceBCKind.INLET or bc.mask is None:
            continue
        combined |= bc.mask
    return combined


def build_momentum_system(
    inp: NaturalConvectionInput,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    component: str,
    u_old: np.ndarray | None = None,
    v_old: np.ndarray | None = None,
    w_old: np.ndarray | None = None,
    u_old_old: np.ndarray | None = None,
    v_old_old: np.ndarray | None = None,
    w_old_old: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """運動量方程式の疎行列を組み立てる.

    Parameters
    ----------
    component : str
        "u", "v", "w" のいずれか
    u_old, v_old, w_old : np.ndarray | None
        前タイムステップの速度場（非定常時）
    u_old_old, v_old_old, w_old_old : np.ndarray | None
        前々タイムステップの速度場（BDF2時間積分時）

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray, np.ndarray]
        (係数行列 A, 右辺ベクトル b, 対角係数 a_P)
        a_P は圧力補正方程式で使用する。
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho, mu = inp.rho, inp.mu

    ii_f, jj_f, kk_f = _build_meshgrid(nx, ny, nz)
    flat_idx = _flat_index(ii_f, jj_f, kk_f, ny, nz)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    # 固体マスク
    is_solid_cell = _is_solid(inp.solid_mask, ii_f, jj_f, kk_f)

    # 各方向の隣接面処理
    directions = [
        ("xm", -1, 0, 0, dx, "bc_xm"),
        ("xp", +1, 0, 0, dx, "bc_xp"),
        ("ym", 0, -1, 0, dy, "bc_ym"),
        ("yp", 0, +1, 0, dy, "bc_yp"),
        ("zm", 0, 0, -1, dz, "bc_zm"),
        ("zp", 0, 0, +1, dz, "bc_zp"),
    ]

    for _dir_name, di, dj, dk, d, bc_attr in directions:
        # 内部面
        if di != 0:
            if di < 0:
                mask = ii_f > 0
            else:
                mask = ii_f < nx - 1
            nb_i = ii_f[mask] + di
            nb_j = jj_f[mask]
            nb_k = kk_f[mask]
        elif dj != 0:
            if dj < 0:
                mask = jj_f > 0
            else:
                mask = jj_f < ny - 1
            nb_i = ii_f[mask]
            nb_j = jj_f[mask] + dj
            nb_k = kk_f[mask]
        else:
            if dk < 0:
                mask = kk_f > 0
            else:
                mask = kk_f < nz - 1
            nb_i = ii_f[mask]
            nb_j = jj_f[mask]
            nb_k = kk_f[mask] + dk

        i_c = flat_idx[mask]
        i_nb = _flat_index(nb_i, nb_j, nb_k, ny, nz)

        # 面法線方向の速度（対流速度）
        if di != 0:
            u_face = 0.5 * (u[ii_f[mask], jj_f[mask], kk_f[mask]] + u[nb_i, nb_j, nb_k])
            face_vel = u_face * di
        elif dj != 0:
            v_face = 0.5 * (v[ii_f[mask], jj_f[mask], kk_f[mask]] + v[nb_i, nb_j, nb_k])
            face_vel = v_face * dj
        else:
            w_face = 0.5 * (w[ii_f[mask], jj_f[mask], kk_f[mask]] + w[nb_i, nb_j, nb_k])
            face_vel = w_face * dk

        # 対流 (1次風上)
        F = rho * face_vel / d  # 対流フラックス / d
        # 拡散
        D = mu / (d * d)

        # 風上: max(F, 0) を自セル、max(-F, 0) を隣接セル
        a_nb = -(D + np.maximum(-F, 0.0))
        a_c_contrib = D + np.maximum(F, 0.0)

        diag[i_c] += a_c_contrib
        rows.append(i_c)
        cols.append(i_nb)
        vals.append(a_nb)

        # 境界面
        bc: FluidBoundarySpec = getattr(inp, bc_attr)
        if di != 0:
            if di < 0:
                mask_bd = ii_f == 0
            else:
                mask_bd = ii_f == nx - 1
        elif dj != 0:
            if dj < 0:
                mask_bd = jj_f == 0
            else:
                mask_bd = jj_f == ny - 1
        else:
            if dk < 0:
                mask_bd = kk_f == 0
            else:
                mask_bd = kk_f == nz - 1

        bd_idx = flat_idx[mask_bd]
        comp_idx = {"u": 0, "v": 1, "w": 2}[component]

        if bc.condition == FluidBoundaryCondition.NO_SLIP:
            # 壁面: 速度=0、ゴーストセル法で Dirichlet
            coeff = 2.0 * mu / (d * d)
            diag[bd_idx] += coeff
            rhs[bd_idx] += coeff * 0.0  # 壁面速度=0
        elif bc.condition == FluidBoundaryCondition.INLET_VELOCITY:
            coeff = 2.0 * mu / (d * d)
            u_bc = bc.velocity[comp_idx]
            diag[bd_idx] += coeff
            rhs[bd_idx] += coeff * u_bc
            # 対流流入
            # 面法線速度
            if di != 0:
                vn = bc.velocity[0] * di
            elif dj != 0:
                vn = bc.velocity[1] * dj
            else:
                vn = bc.velocity[2] * dk
            F_bc = rho * max(vn, 0.0) / d
            diag[bd_idx] += F_bc
            rhs[bd_idx] += F_bc * u_bc
        elif bc.condition == FluidBoundaryCondition.OUTLET_PRESSURE:
            # 出口: ゼロ勾配（何もしない = 自然外挿）
            pass
        elif bc.condition == FluidBoundaryCondition.OUTLET_CONVECTIVE:
            # 対流流出（非反射）: 境界面の対流フラックスを陽的に処理
            # ∂u/∂n = 0 + 流出対流フラックス
            # 境界面での法線速度を使い、流出フラックスを対角に追加
            vel_comp = {"u": u, "v": v, "w": w}[component]
            bd_vel = vel_comp[ii_f[mask_bd], jj_f[mask_bd], kk_f[mask_bd]]
            if di != 0:
                vn = u[ii_f[mask_bd], jj_f[mask_bd], kk_f[mask_bd]] * di
            elif dj != 0:
                vn = v[ii_f[mask_bd], jj_f[mask_bd], kk_f[mask_bd]] * dj
            else:
                vn = w[ii_f[mask_bd], jj_f[mask_bd], kk_f[mask_bd]] * dk
            # 流出フラックスのみ（max(vn, 0)）
            F_out = rho * np.maximum(vn, 0.0) / d
            diag[bd_idx] += F_out
            rhs[bd_idx] += F_out * bd_vel
        elif bc.condition == FluidBoundaryCondition.SYMMETRY:
            # 対称面: 法線速度=0、接線速度勾配=0
            # 法線方向の成分にはDirichlet=0
            is_normal = (
                (di != 0 and component == "u")
                or (dj != 0 and component == "v")
                or (dk != 0 and component == "w")
            )
            if is_normal:
                coeff = 2.0 * mu / (d * d)
                diag[bd_idx] += coeff
                rhs[bd_idx] += 0.0  # 法線速度=0
            # 接線: ゼロ勾配（何もしない）
        elif bc.condition == FluidBoundaryCondition.SLIP:
            is_normal = (
                (di != 0 and component == "u")
                or (dj != 0 and component == "v")
                or (dk != 0 and component == "w")
            )
            if is_normal:
                coeff = 2.0 * mu / (d * d)
                diag[bd_idx] += coeff

    # 圧力勾配ソース項
    gx, gy, gz = inp.gravity
    buoyancy_map = {"u": gx, "v": gy, "w": gz}
    g_comp = buoyancy_map[component]

    # Boussinesq浮力: -ρ₀β(T - T_ref) * g
    buoyancy = -inp.rho * inp.beta * (T.ravel() - inp.T_ref) * g_comp
    rhs += buoyancy

    # 圧力勾配: -∂p/∂x（ベクトル化中心差分、境界は片側差分）
    p_grad_3d = np.zeros((nx, ny, nz))
    if component == "u":
        if nx > 2:
            p_grad_3d[1:-1, :, :] = -(p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
        if nx > 1:
            p_grad_3d[0, :, :] = -(p[1, :, :] - p[0, :, :]) / dx
            p_grad_3d[-1, :, :] = -(p[-1, :, :] - p[-2, :, :]) / dx
    elif component == "v":
        if ny > 2:
            p_grad_3d[:, 1:-1, :] = -(p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dy)
        if ny > 1:
            p_grad_3d[:, 0, :] = -(p[:, 1, :] - p[:, 0, :]) / dy
            p_grad_3d[:, -1, :] = -(p[:, -1, :] - p[:, -2, :]) / dy
    else:  # w
        if nz > 2:
            p_grad_3d[:, :, 1:-1] = -(p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dz)
        if nz > 1:
            p_grad_3d[:, :, 0] = -(p[:, :, 1] - p[:, :, 0]) / dz
            p_grad_3d[:, :, -1] = -(p[:, :, -1] - p[:, :, -2]) / dz

    rhs += p_grad_3d.ravel()

    # 時間項（非定常時）
    if inp.is_transient and u_old is not None:
        vel_old_map = {"u": u_old, "v": v_old, "w": w_old}
        vel_old_old_map = {"u": u_old_old, "v": v_old_old, "w": w_old_old}
        vel_old = vel_old_map[component]
        vel_old2 = vel_old_old_map[component]
        if vel_old is not None:
            if inp.time_scheme == "bdf2" and vel_old2 is not None:
                # BDF2: (3u^{n+1} - 4u^n + u^{n-1}) / (2dt)
                time_coeff = 1.5 * rho / inp.dt
                diag += time_coeff
                rhs += (2.0 * rho / inp.dt) * vel_old.ravel()
                rhs -= (0.5 * rho / inp.dt) * vel_old2.ravel()
            else:
                # Euler後退: (u^{n+1} - u^n) / dt
                time_coeff = rho / inp.dt
                diag += time_coeff
                rhs += time_coeff * vel_old.ravel()

    # TVD 遅延補正（1次風上を行列に保持し、高次補正をRHSに追加）
    if inp.convection_scheme in ("van_leer", "superbee"):
        vel_map = {"u": u, "v": v, "w": w}
        phi_conv = vel_map[component]
        tvd_src = _tvd_deferred_correction(
            phi_conv,
            u,
            v,
            w,
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            rho,
            inp.convection_scheme,
            inp.solid_mask,
        )
        rhs += tvd_src

    # 固体セル: 速度=0 を強制
    solid_cells = is_solid_cell
    if np.any(solid_cells):
        solid_idx = flat_idx[solid_cells]
        # 大きな対角係数で速度=0 を強制
        big = 1e30
        diag[solid_idx] = big
        rhs[solid_idx] = 0.0

    # 内部 BC (INLET): 強制速度
    _apply_internal_bc_momentum(inp.internal_face_bcs, component, flat_idx, diag, rhs)

    # 行列組み立て
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
    return A, rhs, diag


def _cell_pressure_gradient(
    p: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """セル中心の圧力勾配を中心差分で計算."""
    dp_dx = np.zeros((nx, ny, nz))
    if nx > 2:
        dp_dx[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
    if nx > 1:
        dp_dx[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx
        dp_dx[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / dx

    dp_dy = np.zeros((nx, ny, nz))
    if ny > 2:
        dp_dy[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dy)
    if ny > 1:
        dp_dy[:, 0, :] = (p[:, 1, :] - p[:, 0, :]) / dy
        dp_dy[:, -1, :] = (p[:, -1, :] - p[:, -2, :]) / dy

    dp_dz = np.zeros((nx, ny, nz))
    if nz > 2:
        dp_dz[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dz)
    if nz > 1:
        dp_dz[:, :, 0] = (p[:, :, 1] - p[:, :, 0]) / dz
        dp_dz[:, :, -1] = (p[:, :, -1] - p[:, :, -2]) / dz

    return dp_dx, dp_dy, dp_dz


def compute_rhie_chow_face_velocity(
    inp: NaturalConvectionInput,
    u_star: np.ndarray,
    v_star: np.ndarray,
    w_star: np.ndarray,
    p: np.ndarray,
    a_P_u: np.ndarray,
    a_P_v: np.ndarray,
    a_P_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rhie-Chow 補間による面速度を計算.

    コロケーション格子のチェッカーボード圧力振動を抑制するため、
    面速度に圧力勾配の補正項を追加する。

    u_f = (u_P + u_N)/2 - d_f * [(p_N - p_P)/d - (grad_p_P + grad_p_N)/2]

    Parameters
    ----------
    u_star, v_star, w_star : np.ndarray
        予測速度場 (nx, ny, nz)
    p : np.ndarray
        圧力場 (nx, ny, nz)
    a_P_u, a_P_v, a_P_w : np.ndarray
        各運動量方程式の対角係数 (n,)

    Returns
    -------
    tuple of 6 np.ndarray
        (u_face_xp, v_face_yp, w_face_zp,
         u_face_xm, v_face_ym, w_face_zm)
        各方向 +/- 面の Rhie-Chow 補間速度
        xp: i+1/2 面 (shape: nx-1, ny, nz)
        xm: i-1/2 面 (同形状、反転参照用)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho

    a_P_u_3d = a_P_u.reshape(nx, ny, nz)
    a_P_v_3d = a_P_v.reshape(nx, ny, nz)
    a_P_w_3d = a_P_w.reshape(nx, ny, nz)

    # セル中心圧力勾配
    dp_dx, dp_dy, dp_dz = _cell_pressure_gradient(p, nx, ny, nz, dx, dy, dz)

    # --- x方向面 (i と i+1 の間) ---
    if nx > 1:
        # d_f = rho * 0.5 * (1/a_P_i + 1/a_P_{i+1})
        safe_u_l = np.where(a_P_u_3d[:-1] > 0, a_P_u_3d[:-1], 1.0)
        safe_u_r = np.where(a_P_u_3d[1:] > 0, a_P_u_3d[1:], 1.0)
        d_f_x = rho * 0.5 * (1.0 / safe_u_l + 1.0 / safe_u_r)

        # 面速度 = 線形補間 - Rhie-Chow 補正
        u_interp = 0.5 * (u_star[:-1] + u_star[1:])
        # コンパクト勾配: (p_{i+1} - p_i) / dx
        dp_compact = (p[1:] - p[:-1]) / dx
        # 補間勾配: 0.5*(grad_p_i + grad_p_{i+1})
        dp_interp = 0.5 * (dp_dx[:-1] + dp_dx[1:])
        # Rhie-Chow 補正
        u_face_xp = u_interp - d_f_x * (dp_compact - dp_interp)
    else:
        u_face_xp = np.zeros((0, ny, nz))

    # --- y方向面 (j と j+1 の間) ---
    if ny > 1:
        safe_v_l = np.where(a_P_v_3d[:, :-1] > 0, a_P_v_3d[:, :-1], 1.0)
        safe_v_r = np.where(a_P_v_3d[:, 1:] > 0, a_P_v_3d[:, 1:], 1.0)
        d_f_y = rho * 0.5 * (1.0 / safe_v_l + 1.0 / safe_v_r)

        v_interp = 0.5 * (v_star[:, :-1] + v_star[:, 1:])
        dp_compact = (p[:, 1:] - p[:, :-1]) / dy
        dp_interp = 0.5 * (dp_dy[:, :-1] + dp_dy[:, 1:])
        v_face_yp = v_interp - d_f_y * (dp_compact - dp_interp)
    else:
        v_face_yp = np.zeros((nx, 0, nz))

    # --- z方向面 (k と k+1 の間) ---
    if nz > 1:
        safe_w_l = np.where(a_P_w_3d[:, :, :-1] > 0, a_P_w_3d[:, :, :-1], 1.0)
        safe_w_r = np.where(a_P_w_3d[:, :, 1:] > 0, a_P_w_3d[:, :, 1:], 1.0)
        d_f_z = rho * 0.5 * (1.0 / safe_w_l + 1.0 / safe_w_r)

        w_interp = 0.5 * (w_star[:, :, :-1] + w_star[:, :, 1:])
        dp_compact = (p[:, :, 1:] - p[:, :, :-1]) / dz
        dp_interp = 0.5 * (dp_dz[:, :, :-1] + dp_dz[:, :, 1:])
        w_face_zp = w_interp - d_f_z * (dp_compact - dp_interp)
    else:
        w_face_zp = np.zeros((nx, ny, 0))

    return u_face_xp, v_face_yp, w_face_zp


def build_pressure_correction_system_rc(
    inp: NaturalConvectionInput,
    u_star: np.ndarray,
    v_star: np.ndarray,
    w_star: np.ndarray,
    p: np.ndarray,
    a_P_u: np.ndarray,
    a_P_v: np.ndarray,
    a_P_w: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Rhie-Chow 補間付き圧力補正方程式の疎行列を組み立てる.

    従来の中心差分による発散計算の代わりに、Rhie-Chow 面速度を使い
    面ごとの質量流束から発散を計算する。チェッカーボード圧力振動を抑制。

    Parameters
    ----------
    u_star, v_star, w_star : np.ndarray
        予測速度場 (nx, ny, nz)
    p : np.ndarray
        現在の圧力場 (nx, ny, nz)
    a_P_u, a_P_v, a_P_w : np.ndarray
        各運動量方程式の対角係数 (n,)

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        (係数行列 A, 右辺ベクトル b)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho

    ii_f, jj_f, kk_f = _build_meshgrid(nx, ny, nz)
    flat_idx = _flat_index(ii_f, jj_f, kk_f, ny, nz)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)

    is_solid_cell = _is_solid(inp.solid_mask, ii_f, jj_f, kk_f)

    # d係数
    d_u = np.where(a_P_u > 0, rho / a_P_u, 0.0)
    d_v = np.where(a_P_v > 0, rho / a_P_v, 0.0)
    d_w = np.where(a_P_w > 0, rho / a_P_w, 0.0)

    # Rhie-Chow 面速度で質量残差を計算
    u_face_xp, v_face_yp, w_face_zp = compute_rhie_chow_face_velocity(
        inp, u_star, v_star, w_star, p, a_P_u, a_P_v, a_P_w
    )

    # 面ベースの発散 (面積は dy*dz, dx*dz, dx*dy で等間隔格子)
    div_3d = np.zeros((nx, ny, nz))

    # x方向面
    if nx > 1:
        # i+1/2 面の質量流束 → cell i に +、cell i+1 に -
        flux_x = rho * u_face_xp  # (nx-1, ny, nz)
        div_3d[:-1] += flux_x / dx
        div_3d[1:] -= flux_x / dx

    # y方向面
    if ny > 1:
        flux_y = rho * v_face_yp  # (nx, ny-1, nz)
        div_3d[:, :-1] += flux_y / dy
        div_3d[:, 1:] -= flux_y / dy

    # z方向面
    if nz > 1:
        flux_z = rho * w_face_zp  # (nx, ny, nz-1)
        div_3d[:, :, :-1] += flux_z / dz
        div_3d[:, :, 1:] -= flux_z / dz

    rhs = -div_3d.ravel()

    # 圧力補正方程式のラプラシアン: Σ d_face * (p'_nb - p'_P) / d²
    directions_info = [
        (1, 0, 0, dx, d_u),
        (-1, 0, 0, dx, d_u),
        (0, 1, 0, dy, d_v),
        (0, -1, 0, dy, d_v),
        (0, 0, 1, dz, d_w),
        (0, 0, -1, dz, d_w),
    ]

    for di, dj, dk, d, d_coeff in directions_info:
        if di != 0:
            if di > 0:
                mask = ii_f < nx - 1
            else:
                mask = ii_f > 0
            nb_i = ii_f[mask] + di
            nb_j = jj_f[mask]
            nb_k = kk_f[mask]
        elif dj != 0:
            if dj > 0:
                mask = jj_f < ny - 1
            else:
                mask = jj_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask] + dj
            nb_k = kk_f[mask]
        else:
            if dk > 0:
                mask = kk_f < nz - 1
            else:
                mask = kk_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask]
            nb_k = kk_f[mask] + dk

        i_c = flat_idx[mask]
        i_nb = _flat_index(nb_i, nb_j, nb_k, ny, nz)

        d_face = 0.5 * (d_coeff[i_c] + d_coeff[i_nb])
        coeff = d_face / (d * d)

        diag[i_c] += coeff
        rows.append(i_c)
        cols.append(i_nb)
        vals.append(-coeff)

    # 固体セル: 圧力補正=0
    if np.any(is_solid_cell):
        solid_idx = flat_idx[is_solid_cell]
        big = 1e30
        diag[solid_idx] = big
        rhs[solid_idx] = 0.0

    # 内部 BC (INLET/OUTLET): 圧力をピン留め
    _apply_internal_bc_pressure(inp.internal_face_bcs, flat_idx, diag, rhs)

    # 圧力の基準点固定（内部 BC に OUTLET がある場合はそれが基準なので重複固定しない）
    has_outlet = any(
        b.kind == InternalFaceBCKind.OUTLET and b.mask is not None and np.any(b.mask)
        for b in inp.internal_face_bcs
    )
    if not has_outlet:
        fluid_cells = ~is_solid_cell
        if np.any(fluid_cells):
            ref_idx = flat_idx[fluid_cells][0]
            diag[ref_idx] = 1e30
            rhs[ref_idx] = 0.0

    # 行列組み立て
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
    return A, rhs


def build_pressure_correction_system(
    inp: NaturalConvectionInput,
    u_star: np.ndarray,
    v_star: np.ndarray,
    w_star: np.ndarray,
    a_P_u: np.ndarray,
    a_P_v: np.ndarray,
    a_P_w: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """圧力補正方程式の疎行列を組み立てる.

    SIMPLE法の圧力補正方程式:
      Σ a_nb * p'_nb = b
    ここで b = 質量残差（連続の式の不整合）

    Parameters
    ----------
    u_star, v_star, w_star : np.ndarray
        予測速度場 (nx, ny, nz)
    a_P_u, a_P_v, a_P_w : np.ndarray
        各運動量方程式の対角係数 (n,)

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        (係数行列 A, 右辺ベクトル b)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho

    ii_f, jj_f, kk_f = _build_meshgrid(nx, ny, nz)
    flat_idx = _flat_index(ii_f, jj_f, kk_f, ny, nz)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    # 固体マスク
    is_solid_cell = _is_solid(inp.solid_mask, ii_f, jj_f, kk_f)

    # d係数: d_u = Volume / a_P_u, etc.
    # コロケーション配置では d = A_face / a_P
    d_u = np.where(a_P_u > 0, rho / a_P_u, 0.0)
    d_v = np.where(a_P_v > 0, rho / a_P_v, 0.0)
    d_w = np.where(a_P_w > 0, rho / a_P_w, 0.0)

    # 質量残差（連続の式）ベクトル化中心差分
    div_3d = np.zeros((nx, ny, nz))
    if nx > 2:
        div_3d[1:-1, :, :] += (u_star[2:, :, :] - u_star[:-2, :, :]) / (2.0 * dx)
    if nx > 1:
        div_3d[0, :, :] += (u_star[1, :, :] - u_star[0, :, :]) / dx
        div_3d[-1, :, :] += (u_star[-1, :, :] - u_star[-2, :, :]) / dx

    if ny > 2:
        div_3d[:, 1:-1, :] += (v_star[:, 2:, :] - v_star[:, :-2, :]) / (2.0 * dy)
    if ny > 1:
        div_3d[:, 0, :] += (v_star[:, 1, :] - v_star[:, 0, :]) / dy
        div_3d[:, -1, :] += (v_star[:, -1, :] - v_star[:, -2, :]) / dy

    if nz > 2:
        div_3d[:, :, 1:-1] += (w_star[:, :, 2:] - w_star[:, :, :-2]) / (2.0 * dz)
    if nz > 1:
        div_3d[:, :, 0] += (w_star[:, :, 1] - w_star[:, :, 0]) / dz
        div_3d[:, :, -1] += (w_star[:, :, -1] - w_star[:, :, -2]) / dz

    rhs = -(rho * div_3d).ravel()

    # 圧力補正方程式のラプラシアン: ∇·(d∇p') = -∇·u*
    # 離散化: Σ d_face * (p'_nb - p'_P) / d² = mass_residual
    directions_info = [
        (1, 0, 0, dx, d_u),  # x+
        (-1, 0, 0, dx, d_u),  # x-
        (0, 1, 0, dy, d_v),  # y+
        (0, -1, 0, dy, d_v),  # y-
        (0, 0, 1, dz, d_w),  # z+
        (0, 0, -1, dz, d_w),  # z-
    ]

    for di, dj, dk, d, d_coeff in directions_info:
        if di != 0:
            if di > 0:
                mask = ii_f < nx - 1
            else:
                mask = ii_f > 0
            nb_i = ii_f[mask] + di
            nb_j = jj_f[mask]
            nb_k = kk_f[mask]
        elif dj != 0:
            if dj > 0:
                mask = jj_f < ny - 1
            else:
                mask = jj_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask] + dj
            nb_k = kk_f[mask]
        else:
            if dk > 0:
                mask = kk_f < nz - 1
            else:
                mask = kk_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask]
            nb_k = kk_f[mask] + dk

        i_c = flat_idx[mask]
        i_nb = _flat_index(nb_i, nb_j, nb_k, ny, nz)

        # 面の d 係数（隣接セル平均）
        d_face = 0.5 * (d_coeff[i_c] + d_coeff[i_nb])
        coeff = d_face / (d * d)

        diag[i_c] += coeff
        rows.append(i_c)
        cols.append(i_nb)
        vals.append(-coeff)

    # 固体セル: 圧力補正=0
    if np.any(is_solid_cell):
        solid_idx = flat_idx[is_solid_cell]
        big = 1e30
        diag[solid_idx] = big
        rhs[solid_idx] = 0.0

    # 内部 BC (INLET/OUTLET): 圧力をピン留め
    _apply_internal_bc_pressure(inp.internal_face_bcs, flat_idx, diag, rhs)

    # 圧力の基準点固定（圧力の一意性のため）
    # 最初の流体セルの圧力補正をゼロに固定。内部 OUTLET があればそれが基準なので省略。
    has_outlet = any(
        b.kind == InternalFaceBCKind.OUTLET and b.mask is not None and np.any(b.mask)
        for b in inp.internal_face_bcs
    )
    if not has_outlet:
        fluid_cells = ~is_solid_cell
        if np.any(fluid_cells):
            ref_idx = flat_idx[fluid_cells][0]
            diag[ref_idx] = 1e30
            rhs[ref_idx] = 0.0

    # 行列組み立て
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
    return A, rhs


def build_energy_system(
    inp: NaturalConvectionInput,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    T_old_time: np.ndarray | None = None,
    rc_face_velocities: (tuple[np.ndarray, np.ndarray, np.ndarray] | None) = None,
    T_old_old_time: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """エネルギー方程式の疎行列を組み立てる.

    固体-流体練成に対応:
    - 流体領域: 対流+拡散
    - 固体領域: 拡散のみ
    - 界面: 調和平均熱伝導率

    Parameters
    ----------
    u, v, w : np.ndarray
        速度場 (nx, ny, nz)
    T_old_time : np.ndarray | None
        前タイムステップの温度場（非定常時）
    rc_face_velocities : tuple | None
        Rhie-Chow 補間済み面速度 (u_face_xp, v_face_yp, w_face_zp)。
        指定時はこの面速度で対流フラックスを計算し、チェッカーボード抑制。
        None の場合は従来通りセル中心速度の線形補間を使用。
    T_old_old_time : np.ndarray | None
        前々タイムステップの温度場（BDF2時間積分時）

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        (係数行列 A, 右辺ベクトル b)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho, Cp, k_f = inp.rho, inp.Cp, inp.k_fluid

    ii_f, jj_f, kk_f = _build_meshgrid(nx, ny, nz)
    flat_idx = _flat_index(ii_f, jj_f, kk_f, ny, nz)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    # セルごとの熱伝導率
    k_cell = np.full((nx, ny, nz), k_f)
    if inp.solid_mask is not None and inp.k_solid is not None:
        k_cell[inp.solid_mask] = inp.k_solid[inp.solid_mask]

    directions = [
        (1, 0, 0, dx, "bc_xp"),
        (-1, 0, 0, dx, "bc_xm"),
        (0, 1, 0, dy, "bc_yp"),
        (0, -1, 0, dy, "bc_ym"),
        (0, 0, 1, dz, "bc_zp"),
        (0, 0, -1, dz, "bc_zm"),
    ]

    for di, dj, dk, d, bc_attr in directions:
        # 内部面
        if di != 0:
            if di > 0:
                mask = ii_f < nx - 1
            else:
                mask = ii_f > 0
            nb_i = ii_f[mask] + di
            nb_j = jj_f[mask]
            nb_k = kk_f[mask]
        elif dj != 0:
            if dj > 0:
                mask = jj_f < ny - 1
            else:
                mask = jj_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask] + dj
            nb_k = kk_f[mask]
        else:
            if dk > 0:
                mask = kk_f < nz - 1
            else:
                mask = kk_f > 0
            nb_i = ii_f[mask]
            nb_j = jj_f[mask]
            nb_k = kk_f[mask] + dk

        i_c = flat_idx[mask]
        i_nb = _flat_index(nb_i, nb_j, nb_k, ny, nz)

        # 面間熱伝導率（調和平均）
        k_c_vals = k_cell[ii_f[mask], jj_f[mask], kk_f[mask]]
        k_nb_vals = k_cell[nb_i, nb_j, nb_k]
        k_sum = k_c_vals + k_nb_vals
        k_face = np.where(k_sum > 0, 2.0 * k_c_vals * k_nb_vals / k_sum, 0.0)

        # 拡散
        D = k_face / (d * d)

        # 対流（流体セルのみ）
        cell_is_fluid = ~_is_solid(inp.solid_mask, ii_f[mask], jj_f[mask], kk_f[mask])

        # 面速度: RC面速度があればそれを使用（チェッカーボード抑制）
        if rc_face_velocities is not None:
            u_rc, v_rc, w_rc = rc_face_velocities
            if di != 0:
                # xp面 (di=+1): セルi→i+1, 面index=i
                # xm面 (di=-1): セルi→i-1, 面index=i-1
                if di > 0:
                    face_vel = u_rc[ii_f[mask], jj_f[mask], kk_f[mask]]
                else:
                    face_vel = -u_rc[nb_i, nb_j, nb_k]
            elif dj != 0:
                if dj > 0:
                    face_vel = v_rc[ii_f[mask], jj_f[mask], kk_f[mask]]
                else:
                    face_vel = -v_rc[nb_i, nb_j, nb_k]
            else:
                if dk > 0:
                    face_vel = w_rc[ii_f[mask], jj_f[mask], kk_f[mask]]
                else:
                    face_vel = -w_rc[nb_i, nb_j, nb_k]
        else:
            if di != 0:
                u_face = 0.5 * (u[ii_f[mask], jj_f[mask], kk_f[mask]] + u[nb_i, nb_j, nb_k])
                face_vel = u_face * di
            elif dj != 0:
                v_face = 0.5 * (v[ii_f[mask], jj_f[mask], kk_f[mask]] + v[nb_i, nb_j, nb_k])
                face_vel = v_face * dj
            else:
                w_face = 0.5 * (w[ii_f[mask], jj_f[mask], kk_f[mask]] + w[nb_i, nb_j, nb_k])
                face_vel = w_face * dk

        F = np.where(cell_is_fluid, rho * Cp * face_vel / d, 0.0)

        a_nb = -(D + np.maximum(-F, 0.0))
        a_c_contrib = D + np.maximum(F, 0.0)

        diag[i_c] += a_c_contrib
        rows.append(i_c)
        cols.append(i_nb)
        vals.append(a_nb)

        # 境界面
        bc: FluidBoundarySpec = getattr(inp, bc_attr)
        if di != 0:
            if di < 0:
                mask_bd = ii_f == 0
            else:
                mask_bd = ii_f == nx - 1
        elif dj != 0:
            if dj < 0:
                mask_bd = jj_f == 0
            else:
                mask_bd = jj_f == ny - 1
        else:
            if dk < 0:
                mask_bd = kk_f == 0
            else:
                mask_bd = kk_f == nz - 1

        bd_idx = flat_idx[mask_bd]
        bd_k = k_cell[ii_f[mask_bd], jj_f[mask_bd], kk_f[mask_bd]]

        if bc.thermal == ThermalBoundaryCondition.DIRICHLET:
            coeff = 2.0 * bd_k / (d * d)
            diag[bd_idx] += coeff
            rhs[bd_idx] += coeff * bc.temperature
        elif bc.thermal == ThermalBoundaryCondition.NEUMANN:
            rhs[bd_idx] += bc.heat_flux / d
        # ADIABATIC: 何もしない

    # 体積熱生成ソース項
    if inp.q_vol is not None:
        rhs += inp.q_vol.ravel()

    # 時間項
    if inp.is_transient and T_old_time is not None:
        if inp.time_scheme == "bdf2" and T_old_old_time is not None:
            # BDF2: (3T^{n+1} - 4T^n + T^{n-1}) / (2dt)
            time_coeff = 1.5 * rho * Cp / inp.dt
            diag += time_coeff
            rhs += (2.0 * rho * Cp / inp.dt) * T_old_time.ravel()
            rhs -= (0.5 * rho * Cp / inp.dt) * T_old_old_time.ravel()
        else:
            # Euler後退: (T^{n+1} - T^n) / dt
            time_coeff = rho * Cp / inp.dt
            diag += time_coeff
            rhs += time_coeff * T_old_time.ravel()

    # 内部 BC (INLET): 温度強制（temperature が指定されている場合）
    _apply_internal_bc_energy(inp.internal_face_bcs, flat_idx, diag, rhs)

    # TVD 遅延補正（エネルギー方程式）
    if inp.convection_scheme in ("van_leer", "superbee"):
        T_for_tvd = T_old_time if T_old_time is not None else np.zeros((nx, ny, nz))
        tvd_src = _tvd_deferred_correction(
            T_for_tvd,
            u,
            v,
            w,
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            rho * Cp,
            inp.convection_scheme,
            inp.solid_mask,
        )
        rhs += tvd_src

    # 行列組み立て
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
    return A, rhs


def compute_face_mass_residual(
    inp: NaturalConvectionInput,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    a_P_u: np.ndarray,
    a_P_v: np.ndarray,
    a_P_w: np.ndarray,
) -> float:
    """Rhie-Chow 面速度による質量残差を計算.

    圧力補正方程式のRHSと同じ面ベース発散を使うことで、
    SIMPLE反復の収束を正しく評価する。
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho

    u_face_xp, v_face_yp, w_face_zp = compute_rhie_chow_face_velocity(
        inp, u, v, w, p, a_P_u, a_P_v, a_P_w
    )

    div_3d = np.zeros((nx, ny, nz))

    if nx > 1:
        flux_x = rho * u_face_xp
        div_3d[:-1] += flux_x / dx
        div_3d[1:] -= flux_x / dx

    if ny > 1:
        flux_y = rho * v_face_yp
        div_3d[:, :-1] += flux_y / dy
        div_3d[:, 1:] -= flux_y / dy

    if nz > 1:
        flux_z = rho * w_face_zp
        div_3d[:, :, :-1] += flux_z / dz
        div_3d[:, :, 1:] -= flux_z / dz

    if inp.solid_mask is not None:
        div_3d[inp.solid_mask] = 0.0

    return float(np.linalg.norm(div_3d.ravel()))
