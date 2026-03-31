"""NumPy ベクトル化 伝熱解析ソルバー (FDM, ヤコビ法).

Python 3重ループを排除し、NumPy 配列演算のみで反復を実行する。
ヤコビ法（旧値ベース更新）を採用し、完全ベクトル化を実現。
"""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
)


def _face_conductivity(k: np.ndarray, k_neighbor: np.ndarray) -> np.ndarray:
    """面間熱伝導率の調和平均（ベクトル化版）."""
    s = k + k_neighbor
    safe = np.where(s > 0.0, s, 1.0)
    return np.where(s > 0.0, 2.0 * k * k_neighbor / safe, 0.0)


def _apply_bc_vectorized(
    bc: BoundarySpec,
    k_boundary: np.ndarray,
    d: float,
    d2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """境界条件の係数を配列で返す.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (対角係数への寄与, ソース項への寄与)
    """
    shape = k_boundary.shape
    if bc.condition == BoundaryCondition.DIRICHLET:
        coeff = 2.0 * k_boundary / d2
        return coeff, coeff * bc.value
    elif bc.condition == BoundaryCondition.NEUMANN:
        return np.zeros(shape), np.full(shape, bc.value / d)
    elif bc.condition == BoundaryCondition.ROBIN:
        h = bc.h_conv
        if h <= 0.0:
            return np.zeros(shape), np.zeros(shape)
        u_eff = 2.0 * k_boundary * h / (2.0 * k_boundary + h * d)
        coeff = u_eff / d
        return coeff, coeff * bc.T_inf
    else:
        return np.zeros(shape), np.zeros(shape)


def solve_jacobi_step_vectorized(
    T: np.ndarray,
    T_old_time: np.ndarray,
    inp: HeatTransferInput,
    is_transient: bool,
) -> tuple[np.ndarray, float]:
    """ヤコビ法の1反復（完全ベクトル化版）.

    Parameters
    ----------
    T : np.ndarray
        現在の温度場 (nx, ny, nz)
    T_old_time : np.ndarray
        前タイムステップの温度場（非定常時）
    inp : HeatTransferInput
        入力パラメータ
    is_transient : bool
        非定常解析かどうか

    Returns
    -------
    tuple[np.ndarray, float]
        (更新後の温度場, 残差 L2 ノルム)
    """
    nx, ny, nz = T.shape
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    k = inp.k

    # 係数の合計と流束の合計を蓄積
    a_sum = np.zeros_like(T)
    flux_sum = np.zeros_like(T)

    # --- x方向 ---
    # x- 隣接（内部セル i>0）
    if nx > 1:
        k_face = _face_conductivity(k[1:, :, :], k[:-1, :, :])
        coeff = k_face / dx2
        a_sum[1:, :, :] += coeff
        flux_sum[1:, :, :] += coeff * T[:-1, :, :]

    # x- 境界（i=0）
    bc_a, bc_f = _apply_bc_vectorized(inp.bc_xm, k[0, :, :], dx, dx2)
    a_sum[0, :, :] += bc_a
    flux_sum[0, :, :] += bc_f

    # x+ 隣接（内部セル i<nx-1）
    if nx > 1:
        k_face = _face_conductivity(k[:-1, :, :], k[1:, :, :])
        coeff = k_face / dx2
        a_sum[:-1, :, :] += coeff
        flux_sum[:-1, :, :] += coeff * T[1:, :, :]

    # x+ 境界（i=nx-1）
    bc_a, bc_f = _apply_bc_vectorized(inp.bc_xp, k[-1, :, :], dx, dx2)
    a_sum[-1, :, :] += bc_a
    flux_sum[-1, :, :] += bc_f

    # --- y方向 ---
    if ny > 1:
        k_face = _face_conductivity(k[:, 1:, :], k[:, :-1, :])
        coeff = k_face / dy2
        a_sum[:, 1:, :] += coeff
        flux_sum[:, 1:, :] += coeff * T[:, :-1, :]

    bc_a, bc_f = _apply_bc_vectorized(inp.bc_ym, k[:, 0, :], dy, dy2)
    a_sum[:, 0, :] += bc_a
    flux_sum[:, 0, :] += bc_f

    if ny > 1:
        k_face = _face_conductivity(k[:, :-1, :], k[:, 1:, :])
        coeff = k_face / dy2
        a_sum[:, :-1, :] += coeff
        flux_sum[:, :-1, :] += coeff * T[:, 1:, :]

    bc_a, bc_f = _apply_bc_vectorized(inp.bc_yp, k[:, -1, :], dy, dy2)
    a_sum[:, -1, :] += bc_a
    flux_sum[:, -1, :] += bc_f

    # --- z方向 ---
    if nz > 1:
        k_face = _face_conductivity(k[:, :, 1:], k[:, :, :-1])
        coeff = k_face / dz2
        a_sum[:, :, 1:] += coeff
        flux_sum[:, :, 1:] += coeff * T[:, :, :-1]

    bc_a, bc_f = _apply_bc_vectorized(inp.bc_zm, k[:, :, 0], dz, dz2)
    a_sum[:, :, 0] += bc_a
    flux_sum[:, :, 0] += bc_f

    if nz > 1:
        k_face = _face_conductivity(k[:, :, :-1], k[:, :, 1:])
        coeff = k_face / dz2
        a_sum[:, :, :-1] += coeff
        flux_sum[:, :, :-1] += coeff * T[:, :, 1:]

    bc_a, bc_f = _apply_bc_vectorized(inp.bc_zp, k[:, :, -1], dz, dz2)
    a_sum[:, :, -1] += bc_a
    flux_sum[:, :, -1] += bc_f

    # 時間項
    if is_transient:
        time_coeff = inp.C / inp.dt
        a_sum += time_coeff
        flux_sum += time_coeff * T_old_time

    # 発熱量
    flux_sum += inp.q

    # 対角係数
    a_p = a_sum
    safe_a_p = np.where(a_p > 0.0, a_p, 1.0)

    # ヤコビ更新
    T_new = np.where(a_p > 0.0, flux_sum / safe_a_p, T)

    # 残差
    n_cells = nx * ny * nz
    residual = np.sqrt(np.sum((T_new - T) ** 2) / n_cells) if n_cells > 0 else 0.0

    return T_new, residual
