"""右前処理 FGMRES（flexible GMRES）.

scipy `gmres` を置き換える理由（status-39 の実測、288×192 の J1 + SIMPLE 前処理）:

- scipy は内側の打ち切り許容を「前サイクルで真の残差が届かなかった」度合いで自動的に締める（`ptol_max_factor`）ため、
  `rtol=1e-2` 指定でも真の残差 2e-3 まで 37 反復回る。inexact Newton では **指定した許容で止まる**ことが要で、
  本実装は同じ系を 16 反復（真の残差 9.8e-3）で返す
- Gram–Schmidt を 1 反復ずつの Python ループではなく Krylov 基底行列との gemv（BLAS、CGS2 = 古典 Gram–Schmidt
  2 回）で行い、1 反復あたりの overhead が 2.8 ms → 1.4 ms（n3 = 166k）
- 前処理ベクトル Z_j = M^{-1} v_j を保持する flexible 版なので、前処理が反復ごとに変わっても（内側を Krylov 加速
  する等）正しい。メモリは (2·restart + 1)·n·8 バイト（restart=40、n3=166k で 108 MB）

収束判定は Givens 回転で更新される最小二乗残差 |g_{j+1}|（右前処理では真の残差に等しい）で、
restart ごとに真の残差 b − A x を組み直して確認する（`check_true_residual=True`）。
JFNK の有限差分 matvec は線形写像から 1e-4〜1e-3 ずれる（風上切替・リミターの折れ点を跨ぐ）ので Arnoldi 関係が
崩れ、Givens 推定 8e-4 に対し真の残差 5e-3 という食い違いが出る。その場合は真の残差で再出発しても FD 雑音の
床を割れず「not converged」で空回りするだけなので、JFNK では `check_true_residual=False` で Givens 推定を信じて
止める（inexact Newton の許容は FD ヤコビアン自体の近似度と同程度でよい）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg as sla

Op = Callable[[np.ndarray], np.ndarray]


def fgmres(
    matvec: Op,
    b: np.ndarray,
    precond: Op | None = None,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
    restart: int = 40,
    maxiter: int = 5,
    x0: np.ndarray | None = None,
    callback: Callable[[float], None] | None = None,
    check_true_residual: bool = True,
    info: dict[str, float] | None = None,
) -> tuple[np.ndarray, int, bool]:
    """A x = b を右前処理 FGMRES で解く。戻り値 (x, 反復数, 収束フラグ).

    Parameters
    ----------
    matvec : callable
        v -> A v
    precond : callable | None
        v -> M^{-1} v（None なら前処理なし）
    rtol, atol : float
        |b − A x| ≤ max(rtol·|b|, atol) で収束
    restart : int
        Krylov 基底の最大次元（これを超えたら x を更新して再出発）
    maxiter : int
        再出発の最大回数（外側反復数）。scipy `gmres(maxiter=…)` と同じ意味
    callback : callable | None
        反復ごとに残差ノルムを受ける（scipy の `callback_type="pr_norm"` 相当）
    check_true_residual : bool
        True: Givens 推定が許容に届いたら真の残差 b − A x を計算して確認し、届いていなければ再出発する。
        False: Givens 推定を信じて止める（matvec が厳密に線形でない JFNK 用。matvec 1 回分も節約）
    info : dict | None
        与えると最後に評価した残差比 |b − A x| / |b| を "resid_ratio" に書き込む
        （check_true_residual=False で Givens 推定のまま止めたときは推定値）
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = b.size
    if restart < 1 or maxiter < 1:
        raise ValueError(f"restart={restart}, maxiter={maxiter} は 1 以上が必要です")
    identity: Op = lambda v: v  # noqa: E731
    M = precond if precond is not None else identity
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64).reshape(-1)
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return np.zeros(n), 0, True
    tol = max(rtol * bnorm, atol)
    r = b.copy() if x0 is None else b - matvec(x)
    beta = float(np.linalg.norm(r))
    n_iter = 0
    for _outer in range(maxiter):
        if beta <= tol:
            return x, n_iter, True
        V = np.empty((restart + 1, n))
        Z = np.empty((restart, n))
        H = np.zeros((restart + 1, restart))
        cs = np.zeros(restart)
        sn = np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = beta
        V[0] = r / beta
        resid = beta
        k = 0
        for j in range(restart):
            Z[j] = M(V[j])
            w = np.asarray(matvec(Z[j]), dtype=np.float64).reshape(-1)
            Vj = V[: j + 1]
            h = Vj @ w
            w -= Vj.T @ h
            h2 = Vj @ w  # 再直交化（CGS2）
            w -= Vj.T @ h2
            h += h2
            hn = float(np.linalg.norm(w))
            H[: j + 1, j] = h
            H[j + 1, j] = hn
            if hn > 0.0:
                V[j + 1] = w / hn
            # 既存の Givens 回転を新しい列に適用
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            d = float(np.hypot(H[j, j], H[j + 1, j]))
            if d == 0.0:  # 完全な breakdown（A が特異など）
                k = j
                break
            cs[j], sn[j] = H[j, j] / d, H[j + 1, j] / d
            H[j, j], H[j + 1, j] = d, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            resid = abs(float(g[j + 1]))
            n_iter += 1
            k = j + 1
            if callback is not None:
                callback(resid)
            if resid <= tol or hn == 0.0:
                break
        if k > 0:
            y = sla.solve_triangular(H[:k, :k], g[:k], lower=False, check_finite=False)
            x = x + Z[:k].T @ y
        if resid <= tol and not check_true_residual:
            if info is not None:
                info["resid_ratio"] = resid / bnorm
            return x, n_iter, True
        r = b - matvec(x)
        beta = float(np.linalg.norm(r))
        if info is not None:
            info["resid_ratio"] = beta / bnorm
        if beta <= tol:
            return x, n_iter, True
        if k == 0:
            break
    return x, n_iter, False
