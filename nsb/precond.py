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

組立の内訳（288×192、20 コア、status-39）は SA 階層構築 0.1〜1.9 s（pyamg のスペクトル半径推定が乱数依存で振れる）、
ILU 0.24 s。SA の**集約（P, R）は最初の 1 回だけ作り、以後は細格子行列 Ŝ を差し替えて Galerkin 積 R Ŝ P で
粗格子行列だけ組み直す**（`reuse_hierarchy=True`、10 ms）。CFL が 1.4 → 27 と動いても GMRES 反復数は
毎回作り直す場合と同等以下（66 → 44 反復になった例もある: 集約が変わらない方が前処理が安定する）。
V サイクルは `MultilevelSolver.solve` を経由せず（残差ノルム評価の spmv が 1 回余計に入り 6.8 ms → 4.6 ms）、
階層の `presmoother` / `R` / `P` / `postsmoother` / `coarse_solver` を直接辿る。
"""

from __future__ import annotations

import time

import numpy as np
import pyamg
from pyamg.multilevel import coarse_grid_solver
from pyamg.relaxation.smoothing import change_smoothers
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
    reuse_hierarchy : bool
        True なら SA の集約（P, R）を最初の factorize で固定し、以後は Galerkin 積で粗格子行列だけ組み直す
    """

    def __init__(
        self,
        n: int,
        momentum: MomentumSolverType = "ilu",
        schur_cycles: int = 1,
        ilu_drop_tol: float = 1.0e-3,
        ilu_fill_factor: float = 3.0,
        reuse_hierarchy: bool = True,
    ) -> None:
        if momentum not in ("jacobi", "ilu"):
            raise ValueError(f"momentum は jacobi / ilu のいずれか: {momentum!r}")
        self.n = int(n)
        self.momentum = momentum
        self.schur_cycles = max(1, int(schur_cycles))
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill_factor = float(ilu_fill_factor)
        self.reuse_hierarchy = bool(reuse_hierarchy)
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
        self.n_ilu_retries = 0  # 零ピボットで ILU を組み直した回数（累積）
        self.n_hierarchy_builds = 0  # SA の集約を作った回数（累積。reuse_hierarchy なら 1 のまま）

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
        S_pos = (sign * S).tocsr()
        if self.reuse_hierarchy and self._ml_schur is not None:
            self._regalerkin(self._ml_schur, S_pos)
        else:
            self._ml_schur = self._build_hierarchy(S_pos)
            self.n_hierarchy_builds += 1

        self._ilu = None
        if self.momentum == "ilu":
            self._ilu = self._build_ilu(A)
        self._J, self._A, self._B, self._C = J_csr, A, B, C
        self._inv_dA, self._S, self._schur_sign = inv_dA, S, sign
        self.setup_time = time.perf_counter() - t0
        return self

    _SMOOTHER = ("block_gauss_seidel", {"sweep": "symmetric"})

    @staticmethod
    def _build_hierarchy(S: sparse.csr_matrix) -> pyamg.multilevel.MultilevelSolver:
        """SA 階層を組む。pyamg のスペクトル半径推定は numpy のグローバル乱数を使うので、組立の間だけ固定シードにして
        同じ行列から同じ階層が出るようにする（前処理の微差で Newton の経路が変わり反復数が 22〜37 と振れた）."""
        state = np.random.get_state()
        try:
            np.random.seed(0)
            return pyamg.smoothed_aggregation_solver(S, symmetry="nonsymmetric", max_coarse=50)
        finally:
            np.random.set_state(state)

    @staticmethod
    def _regalerkin(ml: pyamg.multilevel.MultilevelSolver, S: sparse.csr_matrix) -> None:
        """集約（P, R）を固定したまま細格子行列を S に差し替え、粗格子行列を Galerkin 積で組み直す."""
        A_l: sparse.csr_matrix = S
        for lv in ml.levels[:-1]:
            lv.A = A_l
            A_l = (lv.R @ A_l @ lv.P).tocsr()
        ml.levels[-1].A = A_l
        sm = SimpleBlockPreconditioner._SMOOTHER  # smoothed_aggregation_solver の既定と同じ
        change_smoothers(ml, sm, sm)
        ml.coarse_solver = coarse_grid_solver(
            "pinv"
        )  # 最粗行列の擬似逆行列は遅延生成（キャッシュを捨てる）

    def _build_ilu(self, A: sparse.csr_matrix) -> spla.SuperLU:
        """運動量ブロックの ILU。零ピボット（"Factor is exactly singular"）なら drop_tol を 1/10、
        fill_factor を 2 倍にして最大 3 回組み直す（288×192 の Newton 途中で drop_tol 1e-2 が落ちた実績）."""
        tol, fill = self.ilu_drop_tol, self.ilu_fill_factor
        A_csc = A.tocsc()
        last: RuntimeError | None = None
        for _attempt in range(3):
            try:
                return spla.spilu(A_csc, drop_tol=tol, fill_factor=fill)
            except RuntimeError as exc:
                last = exc
                self.n_ilu_retries += 1
                tol, fill = tol * 0.1, fill * 2.0
        raise RuntimeError(f"運動量 ILU が 3 回とも零ピボット: {last}") from last

    def _solve_momentum(self, r: np.ndarray) -> np.ndarray:
        assert self._A is not None and self._inv_dA is not None
        if self.momentum == "jacobi":
            return self._inv_dA * r
        assert self._ilu is not None
        return self._ilu.solve(r)

    def _vcycle(self, lvl: int, x: np.ndarray, b: np.ndarray) -> None:
        """pyamg の V サイクル 1 回（`MultilevelSolver.solve` の残差ノルム評価を省いた直接版）."""
        ml = self._ml_schur
        assert ml is not None
        level = ml.levels[lvl]
        A = level.A
        level.presmoother(A, x, b)
        coarse_b = level.R @ (b - A @ x)
        coarse_x = np.zeros_like(coarse_b)
        if lvl == len(ml.levels) - 2:
            coarse_x[:] = ml.coarse_solver(ml.levels[-1].A, coarse_b)
        else:
            self._vcycle(lvl + 1, coarse_x, coarse_b)
        x += level.P @ coarse_x
        level.postsmoother(A, x, b)

    def _solve_schur(self, r: np.ndarray) -> np.ndarray:
        assert self._ml_schur is not None
        b = self._schur_sign * r
        x = np.zeros_like(b)
        for _ in range(self.schur_cycles):
            self._vcycle(0, x, b)
        return x

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
