"""スカラー輸送方程式のアセンブリ (FDM 疎行列).

3次元等間隔直交格子上で対流-拡散-ソース方程式の係数行列と右辺ベクトルを
疎行列として組み立てる。

離散化:
- 対流項: 1次風上差分
- 拡散項: 中心差分（面間拡散係数は調和平均、ただし現在はスカラー拡散係数のみ対応）
- 時間項: 陰的 Euler
- 境界条件: Dirichlet / Neumann / Adiabatic / Robin

設計方針は natural_convection/assembly.py:build_energy_system() と同系列だが、
スカラー単体の輸送に簡略化している。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from xkep_cae_fluid.scalar_transport.data import (
    ScalarBoundaryCondition,
    ScalarBoundarySpec,
    ScalarTransportInput,
)


def _flat_index(i: np.ndarray, j: np.ndarray, k: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """(i, j, k) → フラットインデックス."""
    return i * (ny * nz) + j * nz + k


def _build_meshgrid(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_scalar_system(
    inp: ScalarTransportInput,
    phi_old_time: np.ndarray | None = None,
    rc_face_velocities: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """対流-拡散方程式の疎行列を組み立てる.

    支配方程式:
        ∂(ρφ)/∂t + ∇·(ρu φ) = ∇·(Γ∇φ) + S

    Parameters
    ----------
    inp : ScalarTransportInput
        入力データ（速度場・境界条件・スカラー仕様）
    phi_old_time : np.ndarray | None
        前タイムステップの φ 場 (nx, ny, nz)。
        非定常時に必要（陰的 Euler の時間項）。
        定常時（inp.dt == 0）は無視。
    rc_face_velocities : tuple | None
        Rhie-Chow 補間済み面速度 (u_face_xp, v_face_yp, w_face_zp)。
        NaturalConvection 統合時にエネルギー方程式と整合的な面速度を
        渡すことでチェッカーボード抑制とスカラー質量保存を両立する。
        None の場合はセル中心速度の線形補間を使用。

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        (係数行列 A, 右辺ベクトル b)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho
    Gamma = inp.field.diffusivity

    ii_f, jj_f, kk_f = _build_meshgrid(nx, ny, nz)
    flat_idx = _flat_index(ii_f, jj_f, kk_f, ny, nz)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    u, v, w = inp.u, inp.v, inp.w

    directions = [
        (1, 0, 0, dx, "bc_xp"),
        (-1, 0, 0, dx, "bc_xm"),
        (0, 1, 0, dy, "bc_yp"),
        (0, -1, 0, dy, "bc_ym"),
        (0, 0, 1, dz, "bc_zp"),
        (0, 0, -1, dz, "bc_zm"),
    ]

    for di, dj, dk, d, bc_attr in directions:
        # 内部面マスク
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

        # 拡散係数（現状はスカラー定数）
        D = Gamma / (d * d)

        # 対流速度（面法線方向）
        cell_is_fluid = ~_is_solid(inp.solid_mask, ii_f[mask], jj_f[mask], kk_f[mask])

        # RC面速度がある場合は energy 方程式と同じ取り回しで取得する
        if rc_face_velocities is not None:
            u_rc, v_rc, w_rc = rc_face_velocities
            if di != 0:
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
        elif di != 0:
            u_face = 0.5 * (u[ii_f[mask], jj_f[mask], kk_f[mask]] + u[nb_i, nb_j, nb_k])
            face_vel = u_face * di
        elif dj != 0:
            v_face = 0.5 * (v[ii_f[mask], jj_f[mask], kk_f[mask]] + v[nb_i, nb_j, nb_k])
            face_vel = v_face * dj
        else:
            w_face = 0.5 * (w[ii_f[mask], jj_f[mask], kk_f[mask]] + w[nb_i, nb_j, nb_k])
            face_vel = w_face * dk

        # 流体セルのみ対流有効
        F = np.where(cell_is_fluid, rho * face_vel / d, 0.0)

        # 1次風上: max(F, 0) を自セル、max(-F, 0) を隣接セル
        a_nb = -(D + np.maximum(-F, 0.0))
        a_c_contrib = D + np.maximum(F, 0.0)

        diag[i_c] += a_c_contrib
        rows.append(i_c)
        cols.append(i_nb)
        vals.append(a_nb)

        # 境界面
        bc: ScalarBoundarySpec = getattr(inp, bc_attr)
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

        if bc.condition == ScalarBoundaryCondition.DIRICHLET:
            # ゴーストセル法: 壁面φ=value
            coeff = 2.0 * Gamma / (d * d)
            diag[bd_idx] += coeff
            rhs[bd_idx] += coeff * bc.value
        elif bc.condition == ScalarBoundaryCondition.NEUMANN:
            # Γ∂φ/∂n = flux → rhs に加算（正=流入）
            rhs[bd_idx] += bc.flux / d
        elif bc.condition == ScalarBoundaryCondition.ROBIN:
            # J = h_mass·(phi_inf - phi_surface)
            # 合成抵抗: U_eff = 2·Γ·h / (2·Γ + h·d)
            Gh = bc.h_mass
            denom = 2.0 * Gamma + Gh * d
            if denom > 0.0:
                U_eff = 2.0 * Gamma * Gh / denom
            else:
                U_eff = 0.0
            coeff = U_eff / d
            diag[bd_idx] += coeff
            rhs[bd_idx] += coeff * bc.phi_inf
        # ADIABATIC: 何もしない（ゼロ勾配）

    # 体積ソース項
    if inp.field.source is not None:
        rhs += inp.field.source.ravel()

    # 時間項（陰的 Euler）
    if inp.is_transient and phi_old_time is not None:
        time_coeff = rho / inp.dt
        diag += time_coeff
        rhs += time_coeff * phi_old_time.ravel()

    # 行列組み立て
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
    return A, rhs
