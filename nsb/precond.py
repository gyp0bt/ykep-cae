"""[前処理] SIMPLE 型ブロック前処理: 運動量ブロックの近似解 + 圧力 Schur 補元の AMG.

PARDISO の疎 LU（`nsb.linalg.PardisoLU`）と同じインターフェース（`factorize` / `solve` / `free`）で、
3N×3N ヤコビアン J（ブロック順 [u, v, p]）を

    J = [[A, B],      A: 速度 2N×2N（対流・拡散・抵抗 + 擬似時間対角）
         [C, D]]      B: 圧力勾配、C: 発散、D: Rhie–Chow の圧力項（+ マニホールド吸出）

と分け、Elman らの SIMPLE 型前処理 M^{-1} r を

    1. A u* = r_u             （運動量: ILU（既定）または Jacobi で近似解）
    2. Ŝ δp = r_p − C u*      （圧力: Ŝ = D − C diag(A)^{-1} B を pyamg smoothed aggregation の V サイクルで近似解）
    3. u = u* − diag(A)^{-1} B δp

で適用する。Ŝ は compact 5 点 Poisson が支配的だが、Rhie–Chow の compact/wide 勾配差と Newton 項から
±2〜3 セルの符号混在の遠方項（対角の 5〜10%）が乗る。この遠方項で Ruge–Stüben の収束率が 1 サイクル
0.76 まで落ちる（288×192 で GMRES 199 反復）一方、smoothed aggregation（非対称モード）は 43 反復で、
運動量・Schur とも厳密解にした SIMPLE（64 反復）より少ない（status-38 の切り分け）。
運動量ブロックは ILU（scipy `spilu`）が最良で、Gauss–Seidel と運動量 AMG は高 CFL で発散する（採用しない）。

LU 分解と違って組立コストが O(N)（pyamg の階層構築 + ILU）なので、Newton 反復ごとに組み直す前提。
"""

from __future__ import annotations

import time

import numpy as np
import pyamg
from scipy import sparse
from scipy.sparse import linalg as spla

MomentumSolverType = str  # "jacobi" | "ilu"


class SimpleBlockPreconditioner:
    """SIMPLE 型ブロック前処理（`PardisoLU` 互換の factorize / solve / free）.

    Parameters
    ----------
    n : int
        セル数 N（未知数は 3N）
    momentum : str
        運動量ブロック A の近似解法。"ilu"（scipy `spilu`、ilu_drop_tol / ilu_fill_factor）/ "jacobi"（対角）
    schur_cycles : int
        Schur 補元 Ŝ に当てる AMG V サイクル数
    """

    def __init__(
        self,
        n: int,
        momentum: MomentumSolverType = "ilu",
        schur_cycles: int = 1,
        ilu_drop_tol: float = 1.0e-2,
        ilu_fill_factor: float = 1.5,
    ) -> None:
        if momentum not in ("jacobi", "ilu"):
            raise ValueError(f"momentum は jacobi / ilu のいずれか: {momentum!r}")
        self.n = int(n)
        self.momentum = momentum
        self.schur_cycles = max(1, int(schur_cycles))
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill_factor = float(ilu_fill_factor)
        self._J: sparse.csr_matrix | None = None
        self._A: sparse.csr_matrix | None = None
        self._B: sparse.csr_matrix | None = None
        self._C: sparse.csr_matrix | None = None
        self._inv_dA: np.ndarray | None = None
        self._S: sparse.csr_matrix | None = None
        self._schur_sign = 1.0
        self._ml_schur: pyamg.multilevel.MultilevelSolver | None = None
        self._ilu: spla.SuperLU | None = None
        self.setup_time = 0.0

    # ------------------------------------------------------------------
    @property
    def is_factorized(self) -> bool:
        return self._J is not None

    @property
    def shape(self) -> tuple[int, int]:
        if self._J is None:
            raise RuntimeError("未分解です")
        return self._J.shape

    @property
    def J(self) -> sparse.csr_matrix:
        """直近に組んだ 3N×3N 行列（defect correction の matvec 用）."""
        if self._J is None:
            raise RuntimeError("factorize() を先に呼んでください")
        return self._J

    @property
    def schur(self) -> sparse.csr_matrix:
        """近似 Schur 補元 Ŝ = D − C diag(A)^{-1} B（符号調整前）."""
        if self._S is None:
            raise RuntimeError("factorize() を先に呼んでください")
        return self._S

    # ------------------------------------------------------------------
    def factorize(self, J: sparse.spmatrix) -> SimpleBlockPreconditioner:
        """J をブロック分解し、運動量近似解と Schur 補元 AMG の階層を組む."""
        t0 = time.perf_counter()
        n, n2 = self.n, 2 * self.n
        J_csr = sparse.csr_matrix(J, dtype=np.float64)
        if J_csr.shape != (3 * n, 3 * n):
            raise ValueError(f"J の形状 {J_csr.shape} が 3N={3 * n} と合いません")
        J_csr.sort_indices()
        A = J_csr[:n2, :n2].tocsr()
        B = J_csr[:n2, n2:].tocsr()
        C = J_csr[n2:, :n2].tocsr()
        D = J_csr[n2:, n2:].tocsr()
        dA = A.diagonal()
        if not np.all(np.isfinite(dA)) or np.any(dA == 0.0):
            raise RuntimeError("運動量ブロックの対角にゼロまたは非有限値があります")
        inv_dA = 1.0 / dA
        S = (D - C @ sparse.diags(inv_dA) @ B).tocsr()
        S.sum_duplicates()
        S.sort_indices()
        # AMG は正の対角を前提にするので符号を揃える（Ŝ は −div(ρ d ∇p) 型で通常は正）
        sign = 1.0 if float(np.mean(S.diagonal())) > 0.0 else -1.0
        self._ml_schur = pyamg.smoothed_aggregation_solver(
            (sign * S).tocsr(), symmetry="nonsymmetric", max_coarse=50
        )

        self._ilu = None
        if self.momentum == "ilu":
            self._ilu = spla.spilu(
                A.tocsc(), drop_tol=self.ilu_drop_tol, fill_factor=self.ilu_fill_factor
            )
        self._J, self._A, self._B, self._C = J_csr, A, B, C
        self._inv_dA, self._S, self._schur_sign = inv_dA, S, sign
        self.setup_time = time.perf_counter() - t0
        return self

    def _solve_momentum(self, r: np.ndarray) -> np.ndarray:
        assert self._A is not None and self._inv_dA is not None
        if self.momentum == "jacobi":
            return self._inv_dA * r
        assert self._ilu is not None
        return self._ilu.solve(r)

    def _solve_schur(self, r: np.ndarray) -> np.ndarray:
        assert self._ml_schur is not None
        return self._ml_schur.solve(
            self._schur_sign * r, tol=1e-12, maxiter=self.schur_cycles, cycle="V", accel=None
        )

    def solve(self, b: np.ndarray) -> np.ndarray:
        """M^{-1} b を返す（b は (3N,) または (3N, 1)）."""
        if self._J is None or self._B is None or self._C is None or self._inv_dA is None:
            raise RuntimeError("factorize() を先に呼んでください")
        n2 = 2 * self.n
        bb = np.asarray(b, dtype=np.float64).reshape(-1)
        u_star = self._solve_momentum(bb[:n2])
        dp = self._solve_schur(bb[n2:] - self._C @ u_star)
        u = u_star - self._inv_dA * (self._B @ dp)
        return np.concatenate([u, dp]).reshape(np.shape(b))

    def free(self) -> None:
        self._J = self._A = self._B = self._C = None
        self._inv_dA = self._S = None
        self._ml_schur = None
        self._ilu = None

    def __enter__(self) -> SimpleBlockPreconditioner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.free()
