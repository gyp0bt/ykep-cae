"""疎 LU（PARDISO / MKL）の薄いラッパ. nsb の線形ソルバーは pypardiso 前提（scipy splu へのフォールバックは無い）.

    pip install pypardiso        # MKL 付属。見つからない場合は PYPARDISO_MKL_RT=/path/to/libmkl_rt.so

Newton 反復間で前処理 LU を使い回す（遅延更新）ため、分解と解を分けた `PardisoLU` を提供する。
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from ctypes.util import find_library

import numpy as np
from scipy import sparse


def _ensure_mkl_rt() -> None:
    """pypardiso が libmkl_rt を見つけられない配置（例: システム pip の /usr/local/lib）を補う."""
    if os.environ.get("PYPARDISO_MKL_RT") or find_library("mkl_rt") or find_library("mkl_rt.2"):
        return
    patterns = (
        f"{sys.prefix}/local/lib/libmkl_rt*",
        f"{sys.prefix}/lib/libmkl_rt*",
        f"{sys.base_prefix}/lib/libmkl_rt*",
        "/usr/local/lib/libmkl_rt*",
    )
    for pat in patterns:
        hits = sorted(glob.glob(pat), key=len)
        if hits:
            os.environ["PYPARDISO_MKL_RT"] = hits[0]
            return


_ensure_mkl_rt()
# MKL/OpenMP スレッドの spin 待ち時間（既定 200 ms）。GMRES 内で numpy と MKL 三角解が交互に走るので、
# spin 中のスレッドが numpy の CPU を奪って三角解が 4〜10 倍遅くなる（status-32 で実測）。0 で即休眠
os.environ.setdefault("KMP_BLOCKTIME", "0")

try:
    from pypardiso import PyPardisoSolver
except ImportError as exc:  # pragma: no cover - 環境依存
    raise ImportError(
        "nsb は pypardiso（Intel MKL PARDISO）を必要とします: pip install pypardiso "
        "（libmkl_rt が見つからない場合は環境変数 PYPARDISO_MKL_RT にパスを指定）"
    ) from exc


class PardisoLU:
    """1 つの疎行列の LU 分解を保持し、右辺を何度でも解く（scipy `splu` の代替）.

    `factorize(A)` で分解（PARDISO phase 12: 解析 + 数値分解）、`solve(b)` で三角解（phase 33）。
    分解済み行列は CSR で保持し、`solve` では pypardiso の同一性チェックを通すためにそれを渡す。
    使い終わったら `free()` で MKL 内部メモリを解放する（`with` 文でも可）。

    スレッド数は分解と三角解で分ける（`MKL_Set_Num_Threads_Local`）:
      - factor_threads: 分解。既定は MKL の最大スレッド数（環境変数 MKL_NUM_THREADS で制御可）
      - solve_threads: 三角解。既定 1。前処理として GMRES 内で数十回呼ばれる小さな処理なので、
        並列化の利得よりスレッド同期のコストが大きい（4 スレッド 15 ms vs 1 スレッド 5.7 ms、72×48）
    """

    def __init__(self, factor_threads: int | None = None, solve_threads: int = 1) -> None:
        self._solver = PyPardisoSolver(mtype=11)  # 実非対称
        self._A: sparse.csr_matrix | None = None
        lib = self._solver.libmkl
        # 小文字の mkl_set_num_threads* は mkl_rt 経由で segfault する環境があるので大文字 API を使う
        self._get_max = lib.MKL_Get_Max_Threads
        self._get_max.restype = ctypes.c_int
        self._set_local = lib.MKL_Set_Num_Threads_Local
        self._set_local.argtypes = [ctypes.c_int]
        self._set_local.restype = ctypes.c_int
        self.factor_threads = int(factor_threads or self._get_max())
        self.solve_threads = max(1, int(solve_threads))

    @property
    def max_threads(self) -> int:
        """MKL のグローバル最大スレッド数."""
        return int(self._get_max())

    @property
    def is_factorized(self) -> bool:
        return self._A is not None

    @property
    def shape(self) -> tuple[int, int]:
        if self._A is None:
            raise RuntimeError("未分解です")
        return self._A.shape

    def factorize(self, A: sparse.spmatrix) -> PardisoLU:
        """A を LU 分解する（以前の分解は捨てる）。A は正方・float64."""
        A_csr = sparse.csr_matrix(A, dtype=np.float64)
        A_csr.sort_indices()
        self._set_local(self.factor_threads)
        try:
            self._solver.factorize(A_csr)
        finally:
            self._set_local(0)  # 0 でグローバル設定に戻す
        self._A = A_csr
        return self

    def solve(self, b: np.ndarray) -> np.ndarray:
        """分解済み行列で A x = b を解く（b は (n,) または (n, k)）."""
        if self._A is None:
            raise RuntimeError("factorize() を先に呼んでください")
        self._set_local(self.solve_threads)
        try:
            x = self._solver.solve(self._A, np.asarray(b, dtype=np.float64))
        finally:
            self._set_local(0)
        return x.reshape(np.shape(b))

    def free(self) -> None:
        """MKL 内部メモリを解放する（以後 solve は不可）."""
        if self._A is not None:
            self._solver.free_memory(everything=True)
            self._A = None

    def __enter__(self) -> PardisoLU:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.free()


def pardiso_solve(A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
    """A x = b を 1 回だけ解く（分解は使い捨て）."""
    with PardisoLU() as lu:
        return lu.factorize(A).solve(b)
