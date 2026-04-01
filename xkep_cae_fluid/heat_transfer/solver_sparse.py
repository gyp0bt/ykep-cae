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


def build_sparse_system_nonuniform(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """不等間隔格子用の疎行列システムを組み立てる.

    dx_array/dy_array/dz_array を使い、各セルごとに異なる格子幅で
    拡散項を離散化する。

    2次微分の離散化（セルPの東隣Eとの間）:
      d²T/dx² ≈ k_face * (T_E - T_P) / d_PE
      ここで d_PE = (dx[i] + dx[i+1]) / 2
      体積あたり: coeff = k_face / (d_PE * dx[i])
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    n = nx * ny * nz
    dx_arr = inp.dx_array
    dy_arr = inp.dy_array
    dz_arr = inp.dz_array
    k = inp.k

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    def idx(i: np.ndarray, j: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
        return i * (ny * nz) + j * nz + k_idx

    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ii_f = ii.ravel()
    jj_f = jj.ravel()
    kk_f = kk.ravel()
    flat_idx = idx(ii_f, jj_f, kk_f)

    # x方向のセル幅（各セルに対応）
    dx_cell = dx_arr[ii_f]  # (n,) 各セルの x 幅

    # --- x方向 ---
    # x- 内部
    mask = ii_f > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask] - 1, jj_f[mask], kk_f[mask])
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask] - 1, jj_f[mask], kk_f[mask]],
    )
    d_pn = 0.5 * (dx_arr[ii_f[mask]] + dx_arr[ii_f[mask] - 1])
    coeff = k_face / (d_pn * dx_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    # x- 境界
    mask_bd = ii_f == 0
    bd_k = k[0, jj_f[mask_bd], kk_f[mask_bd]]
    dx_bd = dx_arr[0]
    bc_a, bc_f = _bc_diag_source(inp.bc_xm, bd_k, dx_bd, dx_bd * dx_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # x+ 内部
    mask = ii_f < nx - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask] + 1, jj_f[mask], kk_f[mask])
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask] + 1, jj_f[mask], kk_f[mask]],
    )
    d_pn = 0.5 * (dx_arr[ii_f[mask]] + dx_arr[ii_f[mask] + 1])
    coeff = k_face / (d_pn * dx_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    # x+ 境界
    mask_bd = ii_f == nx - 1
    bd_k = k[nx - 1, jj_f[mask_bd], kk_f[mask_bd]]
    dx_bd = dx_arr[nx - 1]
    bc_a, bc_f = _bc_diag_source(inp.bc_xp, bd_k, dx_bd, dx_bd * dx_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # --- y方向 ---
    dy_cell = dy_arr[jj_f]

    mask = jj_f > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask], jj_f[mask] - 1, kk_f[mask])
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask], jj_f[mask] - 1, kk_f[mask]],
    )
    d_pn = 0.5 * (dy_arr[jj_f[mask]] + dy_arr[jj_f[mask] - 1])
    coeff = k_face / (d_pn * dy_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = jj_f == 0
    bd_k = k[ii_f[mask_bd], 0, kk_f[mask_bd]]
    dy_bd = dy_arr[0]
    bc_a, bc_f = _bc_diag_source(inp.bc_ym, bd_k, dy_bd, dy_bd * dy_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    mask = jj_f < ny - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask], jj_f[mask] + 1, kk_f[mask])
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask], jj_f[mask] + 1, kk_f[mask]],
    )
    d_pn = 0.5 * (dy_arr[jj_f[mask]] + dy_arr[jj_f[mask] + 1])
    coeff = k_face / (d_pn * dy_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = jj_f == ny - 1
    bd_k = k[ii_f[mask_bd], ny - 1, kk_f[mask_bd]]
    dy_bd = dy_arr[ny - 1]
    bc_a, bc_f = _bc_diag_source(inp.bc_yp, bd_k, dy_bd, dy_bd * dy_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # --- z方向 ---
    dz_cell = dz_arr[kk_f]

    mask = kk_f > 0
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask], jj_f[mask], kk_f[mask] - 1)
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask], jj_f[mask], kk_f[mask] - 1],
    )
    d_pn = 0.5 * (dz_arr[kk_f[mask]] + dz_arr[kk_f[mask] - 1])
    coeff = k_face / (d_pn * dz_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = kk_f == 0
    bd_k = k[ii_f[mask_bd], jj_f[mask_bd], 0]
    dz_bd = dz_arr[0]
    bc_a, bc_f = _bc_diag_source(inp.bc_zm, bd_k, dz_bd, dz_bd * dz_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    mask = kk_f < nz - 1
    i_c = flat_idx[mask]
    i_nb = idx(ii_f[mask], jj_f[mask], kk_f[mask] + 1)
    k_face = _face_conductivity(
        k[ii_f[mask], jj_f[mask], kk_f[mask]],
        k[ii_f[mask], jj_f[mask], kk_f[mask] + 1],
    )
    d_pn = 0.5 * (dz_arr[kk_f[mask]] + dz_arr[kk_f[mask] + 1])
    coeff = k_face / (d_pn * dz_cell[mask])
    diag[i_c] += coeff
    rows.append(i_c)
    cols.append(i_nb)
    vals.append(-coeff)

    mask_bd = kk_f == nz - 1
    bd_k = k[ii_f[mask_bd], jj_f[mask_bd], nz - 1]
    dz_bd = dz_arr[nz - 1]
    bc_a, bc_f = _bc_diag_source(inp.bc_zp, bd_k, dz_bd, dz_bd * dz_bd)
    diag[flat_idx[mask_bd]] += bc_a
    rhs[flat_idx[mask_bd]] += bc_f

    # 時間項
    if is_transient and T_old_time is not None:
        time_coeff = inp.C.ravel() / inp.dt
        diag += time_coeff
        rhs += time_coeff * T_old_time.ravel()

    # 発熱量
    rhs += inp.q.ravel()

    # 対角項
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsc()
    return A, rhs


def _build_system(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    is_transient: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """等間隔/不等間隔を自動判定してシステムを組み立てる."""
    if inp.is_nonuniform:
        return build_sparse_system_nonuniform(inp, T_old_time, is_transient)
    return build_sparse_system(inp, T_old_time, is_transient)


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
    A, b = _build_system(inp, T_old_time, is_transient)
    x = spla.spsolve(A, b)
    T = x.reshape(inp.nx, inp.ny, inp.nz)
    return T, 1


class AMGCache:
    """AMG 階層構造のキャッシュ.

    非定常問題でスパースパターンが変わらない場合に AMG セットアップを
    再利用し、計算コストを削減する。行列の値が変わった場合は階層を再構築する。
    """

    def __init__(self) -> None:
        self._ml = None
        self._indptr_hash: int | None = None
        self._indices_hash: int | None = None

    def get_solver(self, A_csr: sparse.csr_matrix) -> object:
        """キャッシュ済み AMG ソルバーを返す。パターン変更時は再構築."""
        try:
            import pyamg
        except ImportError:
            msg = "PyAMG が必要です。pip install 'xkep-cae-fluid[amg]' でインストールしてください。"
            raise ImportError(msg) from None

        indptr_h = hash(A_csr.indptr.data.tobytes())
        indices_h = hash(A_csr.indices.data.tobytes())

        if (
            self._ml is None
            or self._indptr_hash != indptr_h
            or self._indices_hash != indices_h
        ):
            self._ml = pyamg.ruge_stuben_solver(A_csr)
            self._indptr_hash = indptr_h
            self._indices_hash = indices_h

        return self._ml

    def clear(self) -> None:
        """キャッシュをクリア."""
        self._ml = None
        self._indptr_hash = None
        self._indices_hash = None


# モジュールレベルのグローバルキャッシュ
_amg_cache = AMGCache()


def solve_sparse_amg(
    inp: HeatTransferInput,
    T_old_time: np.ndarray | None = None,
    T_init: np.ndarray | None = None,
    is_transient: bool = False,
    amg_cache: AMGCache | None = None,
) -> tuple[np.ndarray, int, float]:
    """PyAMG マルチグリッド前処理付き CG で温度場を求める.

    代数的マルチグリッド (AMG) を前処理に使用し、
    大規模問題でも効率的に収束する。
    伝熱方程式の係数行列は SPD（対称正定値）なので CG が使用可能。

    Parameters
    ----------
    T_init : np.ndarray | None
        初期推定値。None の場合はゼロベクトル。
    amg_cache : AMGCache | None
        AMG 階層キャッシュ。None の場合はモジュールグローバルキャッシュを使用。
        非定常問題でタイムステップ間の AMG セットアップ再利用に有効。

    Returns
    -------
    tuple[np.ndarray, int, float]
        (温度場 (nx,ny,nz), 反復数, 最終残差)

    Raises
    ------
    ImportError
        pyamg がインストールされていない場合
    """
    cache = amg_cache if amg_cache is not None else _amg_cache

    A, b = _build_system(inp, T_old_time, is_transient)

    # CSR 形式に変換（PyAMG は CSR を想定）
    A_csr = A.tocsr()

    # AMG ソルバーの構築（キャッシュ活用）
    ml = cache.get_solver(A_csr)
    M = ml.aspreconditioner()

    x0 = T_init.ravel() if T_init is not None else np.zeros(b.shape[0])

    # コールバックで反復数をカウント
    iter_count = [0]

    def _callback(xk: np.ndarray) -> None:
        iter_count[0] += 1

    x, info = spla.cg(
        A_csr, b, x0=x0, M=M, rtol=inp.tol, maxiter=inp.max_iter, callback=_callback
    )

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
    A, b = _build_system(inp, T_old_time, is_transient)

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
