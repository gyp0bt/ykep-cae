"""SciPy 疎行列ベース伝熱ソルバー.

FDM 離散化の係数行列を CSR 疎行列として組み立て、
直接解法 (spsolve) または前処理付き反復法 (BiCGSTAB + ILU) で解く。
ヤコビ/ガウスザイデル法に比べ、特に異種材料（熱伝導率の大きな差）を
含む問題で大幅な収束改善が期待できる。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
)


def _bc_diag_source(
    bc: BoundarySpec,
    k_boundary: np.ndarray,
    d: float,
    d2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """境界条件の対角寄与とソース寄与を返す."""
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


def _face_conductivity(k_a: np.ndarray, k_b: np.ndarray) -> np.ndarray:
    """面間調和平均."""
    s = k_a + k_b
    safe = np.where(s > 0.0, s, 1.0)
    return np.where(s > 0.0, 2.0 * k_a * k_b / safe, 0.0)


def build_sparse_system(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """FDM 離散化の疎行列 A と右辺ベクトル b を組み立てる.

    Ax = b の形で温度場を求める。x は (nx*ny*nz,) のフラットベクトル。

    Parameters
    ----------
    inp : HeatTransferInput
        入力パラメータ
    T_old_time : np.ndarray | None
        前タイムステップの温度場（非定常時）
    is_transient : bool
        非定常解析フラグ

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        (係数行列 A, 右辺ベクトル b)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    k = inp.k

    # COO 形式で行列要素を蓄積
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    # 対角項とソース項
    diag = np.zeros(n)
    rhs = np.zeros(n)

    # フラットインデックス関数
    def idx(i: np.ndarray, j: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
        return i * (ny * nz) + j * nz + k_idx

    # 全セルのインデックス
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()
    kk_flat = kk.ravel()
    flat_idx = idx(ii_flat, jj_flat, kk_flat)

    # --- x方向 ---
    # x- 内部 (i > 0)
    mask = ii_flat > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask] - 1, jj_flat[mask], kk_flat[mask])
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask] - 1, jj_flat[mask], kk_flat[mask]],
    )
    coeff = k_face / dx2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    # x- 境界 (i == 0)
    mask_bd = ii_flat == 0
    bd_k = k[0, jj_flat[mask_bd], kk_flat[mask_bd]]
    bc_a, bc_f = _bc_diag_source(inp.bc_xm, bd_k, dx, dx2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # x+ 内部 (i < nx-1)
    mask = ii_flat < nx - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask] + 1, jj_flat[mask], kk_flat[mask])
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask] + 1, jj_flat[mask], kk_flat[mask]],
    )
    coeff = k_face / dx2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    # x+ 境界 (i == nx-1)
    mask_bd = ii_flat == nx - 1
    bd_k = k[nx - 1, jj_flat[mask_bd], kk_flat[mask_bd]]
    bc_a, bc_f = _bc_diag_source(inp.bc_xp, bd_k, dx, dx2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # --- y方向 ---
    mask = jj_flat > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask], jj_flat[mask] - 1, kk_flat[mask])
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask], jj_flat[mask] - 1, kk_flat[mask]],
    )
    coeff = k_face / dy2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = jj_flat == 0
    bd_k = k[ii_flat[mask_bd], 0, kk_flat[mask_bd]]
    bc_a, bc_f = _bc_diag_source(inp.bc_ym, bd_k, dy, dy2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    mask = jj_flat < ny - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask], jj_flat[mask] + 1, kk_flat[mask])
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask], jj_flat[mask] + 1, kk_flat[mask]],
    )
    coeff = k_face / dy2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = jj_flat == ny - 1
    bd_k = k[ii_flat[mask_bd], ny - 1, kk_flat[mask_bd]]
    bc_a, bc_f = _bc_diag_source(inp.bc_yp, bd_k, dy, dy2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # --- z方向 ---
    mask = kk_flat > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask], jj_flat[mask], kk_flat[mask] - 1)
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask] - 1],
    )
    coeff = k_face / dz2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = kk_flat == 0
    bd_k = k[ii_flat[mask_bd], jj_flat[mask_bd], 0]
    bc_a, bc_f = _bc_diag_source(inp.bc_zm, bd_k, dz, dz2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    mask = kk_flat < nz - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_flat[mask], jj_flat[mask], kk_flat[mask] + 1)
    k_face = _face_conductivity(
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask]],
        k[ii_flat[mask], jj_flat[mask], kk_flat[mask] + 1],
    )
    coeff = k_face / dz2
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = kk_flat == nz - 1
    bd_k = k[ii_flat[mask_bd], jj_flat[mask_bd], nz - 1]
    bc_a, bc_f = _bc_diag_source(inp.bc_zp, bd_k, dz, dz2)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # 時間項
    if is_transient and T_old_time is not None:
        time_coeff = inp.C.ravel() / inp.dt
        diag += time_coeff
        rhs += time_coeff * T_old_time.ravel()

    # 発熱量
    rhs += inp.q.ravel()

    # 対角項を COO に追加
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    # CSR 行列を構築
    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsc()
    return A, rhs


def solve_sparse_direct(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[np.ndarray, int]:
    """疎行列直接解法 (SuperLU) で温度場を求める.

    Returns
    -------
    tuple[np.ndarray, int]
        (温度場 (nx,ny,nz), 反復数=1)
    """
    A, b = build_sparse_system(inp, T_old_time, is_transient)
    x = spla.spsolve(A, b)
    T = x.reshape(inp.nx, inp.ny, inp.nz)
    return T, 1


def solve_sparse_amg(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    T_init: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[np.ndarray, int, float]:
    """PyAMG マルチグリッド前処理付き CG で温度場を求める.

    代数的マルチグリッド (AMG) を前処理に使用し、
    大規模問題でも効率的に収束する。
    伝熱方程式の係数行列は SPD（対称正定値）なので CG が使用可能。

    Parameters
    ----------
    T_init : np.ndarray | None
        初期推定値。None の場合はゼロベクトル。

    Returns
    -------
    tuple[np.ndarray, int, float]
        (温度場 (nx,ny,nz), 反復数, 最終残差)

    Raises
    ------
    ImportError
        pyamg がインストールされていない場合
    """
    try:
        import pyamg
    except ImportError:
        msg = "PyAMG が必要です。pip install 'xkep-cae-fluid[amg]' でインストールしてください。"
        raise ImportError(msg) from None

    A, b = build_sparse_system(inp, T_old_time, is_transient)

    # CSR 形式に変換（PyAMG は CSR を想定）
    A_csr = A.tocsr()

    # AMG ソルバーの構築（Ruge-Stüben法）
    ml = pyamg.ruge_stuben_solver(A_csr)
    M = ml.aspreconditioner()

    x0 = T_init.ravel() if T_init is not None else np.zeros(b.shape[0])

    # コールバックで反復数をカウント
    iter_count = [0]

    def _callback(xk: np.ndarray) -> None:
        iter_count[0] += 1

    x, info = spla.cg(A_csr, b, x0=x0, M=M, rtol=inp.tol, maxiter=inp.max_iter, callback=_callback)

    T = x.reshape(inp.nx, inp.ny, inp.nz)
    residual = float(np.linalg.norm(A_csr @ x - b) / max(np.linalg.norm(b), 1e-30))

    return T, iter_count[0], residual


def solve_sparse_iterative(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    T_init: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[np.ndarray, int, float]:
    """ILU 前処理付き BiCGSTAB で温度場を求める.

    Parameters
    ----------
    T_init : np.ndarray | None
        初期推定値。None の場合はゼロベクトル。

    Returns
    -------
    tuple[np.ndarray, int, float]
        (温度場 (nx,ny,nz), 反復数, 最終残差)
    """
    A, b = build_sparse_system(inp, T_old_time, is_transient)

    # ILU 前処理
    ilu = spla.spilu(A, drop_tol=1e-4)
    M = spla.LinearOperator(A.shape, matvec=ilu.solve)

    x0 = T_init.ravel() if T_init is not None else None

    # コールバックで反復数をカウント
    iter_count = [0]

    def _callback(xk: np.ndarray) -> None:
        iter_count[0] += 1

    x, info = spla.bicgstab(
        A, b, x0=x0, M=M, rtol=inp.tol, maxiter=inp.max_iter, callback=_callback
    )

    if info > 0:
        # 収束せず — info は反復回数
        pass
    elif info < 0:
        # 不正入力
        pass

    T = x.reshape(inp.nx, inp.ny, inp.nz)

    # 残差計算
    residual = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-30))

    return T, iter_count[0], residual
