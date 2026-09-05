"""線形ソルバー Strategy（:class:`LinearSolverStrategy` の具象実装）.

各パッケージに散っていた spsolve / spilu+BiCGSTAB / PyAMG（構築キャッシュ付き）を
1 か所に集める。すべて ``solve(A, b, x0=None) -> x`` を持ち、
``xkep_cae_fluid.core.strategies.protocols.LinearSolverStrategy`` を満たす。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla


def relative_residual(A: sp.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    """‖b − A x‖ / ‖b‖（‖b‖ が極小なら絶対値）."""
    r = np.linalg.norm(b - A @ x)
    nb = np.linalg.norm(b)
    return float(r / nb) if nb > 1e-30 else float(r)


@dataclass
class DirectSolver:
    """疎 LU 直接法（``scipy.sparse.linalg.spsolve``）."""

    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(spla.spsolve(sp.csc_matrix(A), b), dtype=np.float64)


@dataclass
class BiCGSTABSolver:
    """ILU 前処理付き BiCGSTAB."""

    tol: float = 1e-8
    maxiter: int = 500
    drop_tol: float = 1e-4
    last_info: int = field(default=0, init=False)

    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        A_csc = sp.csc_matrix(A)
        try:
            ilu = spla.spilu(A_csc, drop_tol=self.drop_tol)
            M: spla.LinearOperator | None = spla.LinearOperator(A.shape, matvec=ilu.solve)
        except RuntimeError:
            M = None
        x, info = spla.bicgstab(A_csc, b, x0=x0, M=M, rtol=self.tol, maxiter=self.maxiter)
        self.last_info = int(info)
        return np.asarray(x, dtype=np.float64)


@dataclass
class AMGSolver:
    """PyAMG（Ruge–Stüben）前処理付き CG。構築した階層は行列の構造で再利用する.

    PyAMG が無ければ ``ImportError``。
    """

    tol: float = 1e-8
    maxiter: int = 500
    cache: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    last_info: int = field(default=0, init=False)

    @staticmethod
    def _key(A: sp.csr_matrix) -> str:
        h = hashlib.md5(usedforsecurity=False)
        h.update(A.indptr.tobytes())
        h.update(A.indices.tobytes())
        h.update(A.data.tobytes())
        return h.hexdigest()

    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        try:
            import pyamg
        except ImportError as exc:  # pragma: no cover - 環境依存
            raise ImportError(
                "PyAMG が必要です。pip install 'xkep-cae-fluid[amg]' でインストールしてください。"
            ) from exc
        A_csr = sp.csr_matrix(A)
        ml = None
        key = self._key(A_csr) if self.cache else ""
        if self.cache:
            ml = self._cache.get(key)
        if ml is None:
            ml = pyamg.ruge_stuben_solver(A_csr)
            if self.cache:
                self._cache.clear()
                self._cache[key] = ml
        M = ml.aspreconditioner(cycle="V")
        x, info = spla.cg(A_csr, b, x0=x0, M=M, rtol=self.tol, maxiter=self.maxiter)
        self.last_info = int(info)
        return np.asarray(x, dtype=np.float64)


_SOLVERS: dict[str, type] = {
    "direct": DirectSolver,
    "bicgstab": BiCGSTABSolver,
    "amg": AMGSolver,
}


def make_linear_solver(name: str, **kwargs: Any) -> DirectSolver | BiCGSTABSolver | AMGSolver:
    """名前（direct / bicgstab / amg）から線形ソルバーを作る."""
    key = name.strip().lower()
    if key not in _SOLVERS:
        raise ValueError(f"未知の線形ソルバー {name!r}（{sorted(_SOLVERS)}）")
    cls = _SOLVERS[key]
    if cls is DirectSolver:
        return DirectSolver()
    return cls(**kwargs)
