"""メイン: Newton + 擬似時間 + GMRES(LU 前処理) で定常解を求める.

手元構成との比較のため、制御則の「線」をすべて明示的なブロックとして書き下す:
  [Δτ]      局所/大域、速度下限の有無
  [残差]    擬似時間項を残差に含めるか（dual-time 型）
  [反復]    1 擬似時間ステップあたりの Newton 反復数（sub_iters）
  [RC]      Rhie–Chow 係数に擬似時間項を含めるか
  [線形]    JFNK（GMRES + LU(J1)）か LU 直接（defect correction）。LU は PARDISO（pypardiso）
  [前処理]  LU(J1) の遅延更新: 1 回の分解を precond_lag 反復まで使い回す（GMRES 不収束なら即再分解）
  [SER]     残差比で CFL を増減
  [初期場]  静止場 / Stokes–Brinkman 解
  [棄却]    残差が増えた更新を棄却して CFL を下げる backtracking
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from nsb.assembly import BrinkmanDiscretization, StateArrays
from nsb.core import NSBInput, NSBResult, NSBSettings
from nsb.linalg import PardisoLU

LogFn = Callable[[str], None]


def compute_dtau(
    u: np.ndarray, v: np.ndarray, dx: float, dy: float, cfl: float, s: NSBSettings
) -> np.ndarray:
    """[Δτ] セルごとの擬似時間増分 (nx, ny).

    Δτ_P = CFL·min(Δx,Δy) / max(|u_P|+|v_P|, velocity_floor)。
    velocity_floor=0 の静止セルでは Δτ→∞（1e30 で打ち切り、対角補強は実質ゼロ）。
    local_dtau=False なら全セル最小値を一律に使う。
    """
    speed = np.abs(u) + np.abs(v)
    speed = np.maximum(speed, s.velocity_floor)
    with np.errstate(divide="ignore"):
        dtau = np.where(speed > 0.0, cfl * min(dx, dy) / np.maximum(speed, 1e-300), 1e30)
    dtau = np.minimum(dtau, 1e30)
    if not s.local_dtau:
        dtau = np.full_like(dtau, float(dtau.min()))
    return dtau


class LaggedPreconditioner:
    """[前処理] LU(J1 + diag) を Newton 反復間で使い回す（遅延更新）.

    JFNK では前処理は近似でよいので、毎反復の分解（実測で全体の 70〜81%）を
    `precond_lag` 反復に 1 回へ減らす。再分解の条件:
      - まだ分解していない / age >= precond_lag
      - 直前の GMRES 反復数が precond_refresh_gmres を超えた（古くなった兆候）
      - 分解時から CFL が precond_cfl_ratio 倍以上変わった（擬似時間対角の不整合）
      - 呼び出し側の force（棄却後など）
    GMRES が収束しなかった場合は `solve_linear` 内で即再分解して解き直す。
    """

    def __init__(self, s: NSBSettings) -> None:
        self.s = s
        self.lu = PardisoLU()
        self.age = 0  # この分解で解いた Newton 反復数
        self.last_gmres = 0
        self.n_factorizations = 0
        self.cfl_at_factorization = float("nan")
        self.cfl = float("nan")  # 現在の CFL（solve_steady が擬似時間ステップ開始時に更新）

    def needs_refresh(self, force: bool = False) -> bool:
        if force or not self.lu.is_factorized or self.s.linear_solver == "lu":
            return True
        if self.age >= max(1, self.s.precond_lag):
            return True
        if self.last_gmres > self.s.precond_refresh_gmres:
            return True
        if self.s.precond_cfl_ratio > 0.0 and np.isfinite(self.cfl_at_factorization):
            ratio = self.cfl / self.cfl_at_factorization
            if ratio >= self.s.precond_cfl_ratio or ratio <= 1.0 / self.s.precond_cfl_ratio:
                return True
        return False

    def refresh(self, J1: sparse.spmatrix) -> None:
        self.lu.factorize(J1)
        self.age = 0
        self.n_factorizations += 1
        self.cfl_at_factorization = self.cfl

    def free(self) -> None:
        self.lu.free()


def solve_linear(
    disc: BrinkmanDiscretization,
    st: StateArrays,
    x: np.ndarray,
    rhs_resid: np.ndarray,
    diag_aug: np.ndarray,
    resid_fn: Callable[[np.ndarray], np.ndarray],
    s: NSBSettings,
    pc: LaggedPreconditioner,
    force_refresh: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """[線形] (J + diag_aug) δ = -R を解く。戻り値 (δ, GMRES 反復数, 収束フラグ).

    前処理（"lu" では解そのもの）は pc が保持する PARDISO 分解。必要なときだけ J1 を組んで再分解する。
    """
    n3 = x.size

    def assemble() -> sparse.csr_matrix:
        return (disc.jacobian_first_order(st, x=x) + sparse.diags(diag_aug)).tocsr()

    if pc.needs_refresh(force_refresh):
        pc.refresh(assemble())
    if s.linear_solver == "lu":
        pc.age += 1
        return pc.lu.solve(-rhs_resid), 0, True

    x_norm = float(np.linalg.norm(x))
    sqrt_eps = float(np.sqrt(np.finfo(float).eps))

    def matvec(vec: np.ndarray) -> np.ndarray:
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0.0:
            return np.zeros_like(vec)
        eps = sqrt_eps * np.sqrt(1.0 + x_norm) / v_norm
        return (resid_fn(x + eps * vec) - rhs_resid) / eps + diag_aug * vec

    A = spla.LinearOperator((n3, n3), matvec=matvec, dtype=np.float64)

    def run_gmres() -> tuple[np.ndarray, int, bool]:
        count = [0]

        def cb(_: float) -> None:
            count[0] += 1

        M = spla.LinearOperator((n3, n3), matvec=pc.lu.solve, dtype=np.float64)
        delta, info = spla.gmres(
            A,
            -rhs_resid,
            M=M,
            rtol=s.gmres_tol,
            atol=0.0,
            restart=s.gmres_restart,
            maxiter=s.gmres_maxiter,
            callback=cb,
            callback_type="pr_norm",
        )
        return delta, count[0], info == 0

    delta, n_gmres, ok = run_gmres()
    if not ok and pc.age > 0:
        # 古い前処理で収束しなかった: 現在の J1 で再分解して解き直す
        pc.refresh(assemble())
        delta, n_retry, ok = run_gmres()
        n_gmres += n_retry
    pc.age += 1
    pc.last_gmres = n_gmres
    return delta, n_gmres, ok


def solve_steady(inp: NSBInput, log: LogFn | None = print) -> NSBResult:
    """定常 Brinkman-NS を Newton + 擬似時間で解く（収束しなければ converged=False）."""
    t0 = time.perf_counter()
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    shape = (inp.nx, inp.ny)
    emit = log if log is not None else (lambda _m: None)

    u = np.zeros(shape) if inp.u0 is None else inp.u0.astype(float)
    v = np.zeros(shape) if inp.v0 is None else inp.v0.astype(float)
    p = np.zeros(shape) if inp.p0 is None else inp.p0.astype(float)
    x = np.concatenate([u.ravel(), v.ravel(), p.ravel()])

    # [初期場] Stokes–Brinkman 解: 運動量の対流項（inlet の運動量流束を含む）を落とした線形問題を
    # ゼロ場から 1 回の LU で解く。対流項込みの残差をゼロ場で解くと inlet 運動量流束が
    # ソースとして残り、流速が U_in の 10 倍超の非物理的な噴流になるので注意
    pc = LaggedPreconditioner(s)
    n_gmres_total = 0
    if s.init_field == "stokes":
        st0 = disc.compute_state(x, s.scheme, s.venkat_k)
        r_init = disc.residual_from_state(x, st0, convection=False)
        J0 = disc.jacobian_first_order(st0, convection=False, x=x).tocsr()
        pc.refresh(J0)
        x = x + pc.lu.solve(-r_init)
        pc.age = 10**9  # Stokes 行列は前処理として使わない（次で必ず再分解）
        u0_, v0_, _ = disc.split(x)
        emit(
            f"[nsb] stokes init: |R_stokes(0)|={np.linalg.norm(r_init):.4e} "
            f"speed_max={np.hypot(u0_, v0_).max():.3g} m/s"
        )

    # 擬似時間ステップ内で凍結する量
    rc_diag: np.ndarray | None = None  # [RC] RC 係数に含める ρV/Δτ
    tau_diag = np.zeros(n)  # ρV/Δτ（u, v 各 n 要素分）
    x_prev = x.copy()  # [残差] 擬似時間項の基準（前ステップの場）

    def state(xx: np.ndarray) -> StateArrays:
        return disc.compute_state(xx, s.scheme, s.venkat_k, rc_diag)

    def steady_resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_from_state(xx, state(xx))

    def resid(xx: np.ndarray) -> np.ndarray:
        """[残差] 擬似時間項込みの残差 R_τ = R + ρV(u - u_prev)/Δτ（u, v 成分のみ）."""
        r = steady_resid(xx)
        if s.pseudo_time_in_residual:
            r = r.copy()
            r[: 2 * n] += np.concatenate([tau_diag, tau_diag]) * (xx[: 2 * n] - x_prev[: 2 * n])
        return r

    cfl = s.cfl_init
    r = steady_resid(x)
    r_norm = float(np.linalg.norm(r))
    r0 = max(r_norm, 1e-300)
    r_steady0 = r0
    hist = [r_norm]
    hist_steady = [r_norm]
    cfl_hist: list[float] = []
    converged = False
    failure = ""
    n_iter = 0
    n_rejected = 0
    n_rej_step = 0
    force_refresh = False
    emit(f"[nsb] it=0 |R|={r_norm:.4e} cfl={cfl:.3g}")

    while n_iter < s.newton_max_iter:
        if not np.isfinite(r_norm):
            failure = "nan"
            break
        if r_norm / r0 < s.newton_tol:
            converged = True
            break
        if r_norm / r0 > s.divergence_ratio:
            failure = "diverged"
            break

        # ---- 擬似時間ステップ開始: Δτ を決めて凍結 ----
        uu, vv, _ = disc.split(x)
        dtau = compute_dtau(uu, vv, disc.dx, disc.dy, cfl, s)
        tau_diag = (inp.rho * disc.vol / dtau).ravel()
        rc_diag = tau_diag.reshape(shape) if s.rc_with_pseudo_time else None
        x_prev = x.copy()
        pc.cfl = cfl

        step_ok = True
        for _sub in range(s.sub_iters):
            st = state(x)
            r_tau = resid(x)
            relax = (1.0 - s.alpha_u) / s.alpha_u * st.a_p.ravel()
            diag_aug = np.concatenate([tau_diag + relax, tau_diag + relax, np.zeros(n)])
            try:
                delta, n_gmres, lin_ok = solve_linear(
                    disc, st, x, r_tau, diag_aug, resid, s, pc, force_refresh
                )
            except (RuntimeError, ValueError) as exc:
                failure = f"lu_failed: {exc}"
                step_ok = False
                break
            force_refresh = False
            n_gmres_total += n_gmres
            if not np.all(np.isfinite(delta)):
                failure = "gmres_breakdown"
                step_ok = False
                break
            x_new = x + delta

            # ステップ終了時の残差（擬似時間項込み: 収束判定・SER に使う）と定常残差
            r_new = float(np.linalg.norm(resid(x_new)))

            # [棄却] 残差が reject_growth 倍を超えて増えたら更新を捨て、CFL を半分にして Δτ を組み直す
            if (
                s.reject_growth > 0.0
                and n_iter >= 1  # 静止初期場からの 1 歩目は残差が必ず増えるので棄却しない
                and n_rej_step < s.max_rejects
                and cfl > s.cfl_min
                and (not np.isfinite(r_new) or r_new > s.reject_growth * r_norm)
            ):
                n_rej_step += 1
                n_rejected += 1
                cfl *= 0.5
                emit(
                    f"[nsb]   reject: |R_tau|={r_new:.4e} > {s.reject_growth:g}×{r_norm:.4e} "
                    f"-> cfl={cfl:.3g} (rejects={n_rej_step})"
                )
                dtau = compute_dtau(uu, vv, disc.dx, disc.dy, cfl, s)
                tau_diag = (inp.rho * disc.vol / dtau).ravel()
                rc_diag = tau_diag.reshape(shape) if s.rc_with_pseudo_time else None
                force_refresh = True  # Δτ が変わるので前処理も組み直す
                step_ok = False  # 擬似時間ステップをやり直す（SER をスキップ）
                break

            x = x_new
            n_iter += 1
            r_steady_new = float(np.linalg.norm(steady_resid(x)))
            hist.append(r_new)
            hist_steady.append(r_steady_new)
            emit(
                f"[nsb] it={n_iter} |R_tau|={r_new:.4e} rel={r_new / r0:.3e} "
                f"|R_steady|/|R0|={r_steady_new / r_steady0:.3e} cfl={cfl:.3g} "
                f"dtau=[{dtau.min():.2e},{dtau.max():.2e}] gmres={n_gmres} "
                f"pc_age={pc.age} fact={pc.n_factorizations}"
                f"{'' if lin_ok else ' (gmres not converged)'}"
            )
            if not np.isfinite(r_new):
                break
        if not step_ok:
            if failure:
                break
            continue  # 棄却: 同じ x から縮めた CFL で再試行
        n_rej_step = 0

        # ---- [SER] 残差比で CFL を更新 ----
        ratio = r_norm / r_new if r_new > 0.0 and np.isfinite(r_new) else 0.1
        cfl = float(min(s.cfl_max, cfl * float(np.clip(ratio, 0.1, s.ser_growth))))
        cfl_hist.append(cfl)
        r_norm = r_new
    else:
        if np.isfinite(r_norm) and r_norm / r0 < s.newton_tol:
            converged = True
        elif not np.isfinite(r_norm):
            failure = "nan"
        else:
            failure = "max_iter"

    pc.free()
    u, v, p = disc.split(x)
    rc_diag = None
    m_in, m_out = disc.mass_flow(state(x), x)
    elapsed = time.perf_counter() - t0
    emit(
        f"[nsb] done converged={converged} reason='{failure}' it={n_iter} "
        f"m_in={m_in:.4e} m_out={m_out:.4e} rejected={n_rejected} "
        f"factorizations={pc.n_factorizations} gmres_total={n_gmres_total} elapsed={elapsed:.1f}s"
    )
    return NSBResult(
        u=u.copy(),
        v=v.copy(),
        p=p.copy(),
        converged=converged,
        failure_reason=failure,
        n_iter=n_iter,
        residual_history=tuple(hist),
        steady_residual_history=tuple(hist_steady),
        cfl_history=tuple(cfl_hist),
        mass_in=m_in,
        mass_out=m_out,
        elapsed=elapsed,
        n_rejected=n_rejected,
        n_factorizations=pc.n_factorizations,
        n_gmres_total=n_gmres_total,
    )
