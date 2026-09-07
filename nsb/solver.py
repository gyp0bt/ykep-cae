"""メイン: Newton + 擬似時間 + GMRES(LU 前処理) で定常解を求める.

手元構成との比較のため、制御則の「線」をすべて明示的なブロックとして書き下す:
  [Δτ]      局所/大域、速度下限の有無
  [残差]    擬似時間項を残差に含めるか（dual-time 型）
  [反復]    1 擬似時間ステップあたりの Newton 反復数（sub_iters）
  [RC]      Rhie–Chow 係数に擬似時間項を含めるか
  [線形]    JFNK（GMRES + LU(J1)）か LU 直接（defect correction）。LU は PARDISO（pypardiso）。
            GMRES は自作の右前処理 FGMRES（`nsb.krylov`。scipy gmres は指定 rtol より深く解いてしまう）。
            SIMPLE 型ブロック前処理（ILU + Schur 補元 AMG、`nsb.precond`）を使う
            jfnk_simple（有限差分 matvec）/ dc_simple（J1 matvec の defect correction）
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

from nsb.assembly import BrinkmanDiscretization, StateArrays
from nsb.core import NSBInput, NSBResult, NSBSettings
from nsb.krylov import fgmres
from nsb.linalg import PardisoLU, pardiso_solve
from nsb.precond import SimpleBlockPreconditioner

LogFn = Callable[[str], None]
SIMPLE_MODES = ("jfnk_simple", "dc_simple")
LINEAR_SOLVERS = ("jfnk", "lu") + SIMPLE_MODES


def uses_simple_preconditioner(s: NSBSettings) -> bool:
    return s.linear_solver in SIMPLE_MODES


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
    """[前処理] LU(J1 + diag) または SIMPLE 型ブロック前処理を Newton 反復間で使い回す（遅延更新）.

    `fac` は `PardisoLU`（"jfnk" / "lu"）か `SimpleBlockPreconditioner`（"jfnk_simple" / "dc_simple"）で、
    どちらも factorize / solve / free の同じインターフェースを持つ。
    JFNK では前処理は近似でよいので、毎反復の分解（実測で全体の 70〜81%）を
    `precond_lag` 反復に 1 回へ減らす。再分解の条件:
      - まだ分解していない / age >= precond_lag
      - 直前の GMRES 反復数が precond_refresh_gmres を超えた（古くなった兆候）
      - 分解時から CFL が precond_cfl_ratio 倍以上変わった（擬似時間対角の不整合）
      - 呼び出し側の force（棄却後など）
    GMRES が収束しなかった場合は `solve_linear` 内で即再分解して解き直す。
    """

    def __init__(self, s: NSBSettings, n: int | None = None) -> None:
        if s.linear_solver not in LINEAR_SOLVERS:
            raise ValueError(f"linear_solver は {LINEAR_SOLVERS} のいずれか: {s.linear_solver!r}")
        self.s = s
        self.fac: PardisoLU | SimpleBlockPreconditioner
        if uses_simple_preconditioner(s):
            if n is None:
                raise ValueError("SIMPLE 型前処理にはセル数 n が必要です")
            self.fac = SimpleBlockPreconditioner(
                n,
                momentum=s.simple_momentum,
                schur_cycles=s.simple_schur_cycles,
                ilu_drop_tol=s.simple_ilu_drop_tol,
                ilu_fill_factor=s.simple_ilu_fill_factor,
            )
        else:
            self.fac = PardisoLU()
        self.age = 0  # この分解で解いた Newton 反復数
        self.last_gmres = 0
        self.n_factorizations = 0
        self.cfl_at_factorization = float("nan")
        self.cfl = float("nan")  # 現在の CFL（solve_steady が擬似時間ステップ開始時に更新）

    def needs_refresh(self, force: bool = False) -> bool:
        if force or not self.fac.is_factorized or self.s.linear_solver == "lu":
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
        self.fac.factorize(J1)
        self.age = 0
        self.n_factorizations += 1
        self.cfl_at_factorization = self.cfl

    def free(self) -> None:
        self.fac.free()


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

    前処理（"lu" では解そのもの）は pc が保持する PARDISO 分解または SIMPLE 型前処理。
    必要なときだけ J1 を組んで組み直す。"dc_simple" は matvec も現在の J1 + diag_aug で行う
    （defect correction）ので、前処理の遅延更新に関わらず毎反復 J1 を組む。
    """

    def assemble() -> sparse.csr_matrix:
        return (disc.jacobian_first_order(st, x=x) + sparse.diags(diag_aug)).tocsr()

    J_cur: sparse.csr_matrix | None = None
    if pc.needs_refresh(force_refresh):
        J_cur = assemble()
        pc.refresh(J_cur)
    if s.linear_solver == "lu":
        pc.age += 1
        return pc.fac.solve(-rhs_resid), 0, True

    matvec: Callable[[np.ndarray], np.ndarray]
    if s.linear_solver == "dc_simple":
        if J_cur is None:
            J_cur = assemble()
        J_dc = J_cur

        def matvec(vec: np.ndarray) -> np.ndarray:
            return J_dc @ vec

    else:
        x_norm = float(np.linalg.norm(x))
        sqrt_eps = float(np.sqrt(np.finfo(float).eps))

        def matvec(vec: np.ndarray) -> np.ndarray:
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0.0:
                return np.zeros_like(vec)
            eps = sqrt_eps * np.sqrt(1.0 + x_norm) / v_norm
            return (resid_fn(x + eps * vec) - rhs_resid) / eps + diag_aug * vec

    def run_gmres() -> tuple[np.ndarray, int, bool]:
        # JFNK の FD matvec は厳密に線形でないので Givens 推定と真の残差が食い違い、再出発が空回りして
        # 「not converged」→ 前処理を組み直して解き直す経路に入ることがある（status-39 §3.1）。
        # Givens 推定だけで止める（check_true_residual=False）と PARDISO 側の Newton が収束しなくなった
        # 走行があったので、真の残差確認 + 不収束時の組み直しは残す
        return fgmres(
            matvec,
            -rhs_resid,
            precond=pc.fac.solve,
            rtol=s.gmres_tol,
            atol=0.0,
            restart=s.gmres_restart,
            maxiter=s.gmres_maxiter,
        )

    delta, n_gmres, ok = run_gmres()
    if not ok and pc.age > 0:
        # 古い前処理で収束しなかった: 現在の J1 で再分解して解き直す
        pc.refresh(J_cur if J_cur is not None else assemble())
        delta, n_retry, ok = run_gmres()
        n_gmres += n_retry
    pc.age += 1
    pc.last_gmres = n_gmres
    return delta, n_gmres, ok


def _gmres_exact(
    A: sparse.spmatrix,
    b: np.ndarray,
    precond: Callable[[np.ndarray], np.ndarray],
    s: NSBSettings,
    rtol: float = 1.0e-10,
    max_outer: int = 20,
) -> tuple[np.ndarray, int, bool]:
    """線形系 A x = b を前処理付き GMRES で厳密（rtol）に解く（Stokes 初期場用）."""
    A_csr = sparse.csr_matrix(A)
    x, n_iter, ok = fgmres(
        lambda v: A_csr @ v,
        b,
        precond=precond,
        rtol=rtol,
        atol=0.0,
        restart=s.gmres_restart,
        maxiter=max_outer,
    )
    ok = ok and bool(np.all(np.isfinite(x)))
    if ok:
        ok = float(np.linalg.norm(A_csr @ x - b)) <= 1e-6 * float(np.linalg.norm(b))
    return x, n_iter, ok


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
    pc = LaggedPreconditioner(s, n)
    n_gmres_total = 0
    if s.init_field == "stokes":
        st0 = disc.compute_state(x, s.scheme, s.venkat_k)
        r_init = disc.residual_from_state(x, st0, convection=False)
        J0 = disc.jacobian_first_order(st0, convection=False, x=x).tocsr()
        pc.refresh(J0)
        if uses_simple_preconditioner(s):
            # 線形問題なので SIMPLE 前処理付き GMRES で厳密に解く（収束しなければ PARDISO 1 回）
            dx0, n_g0, ok0 = _gmres_exact(J0, -r_init, pc.fac.solve, s)
            n_gmres_total += n_g0
            how = f"gmres={n_g0}"
            if not ok0:
                dx0 = pardiso_solve(J0, -r_init)
                how = f"gmres={n_g0} (not converged) -> pardiso"
        else:
            dx0 = pc.fac.solve(-r_init)
            how = "pardiso"
        x = x + dx0
        pc.age = 10**9  # Stokes 行列は前処理として使わない（次で必ず再分解）
        u0_, v0_, _ = disc.split(x)
        emit(
            f"[nsb] stokes init ({how}): |R_stokes(0)|={np.linalg.norm(r_init):.4e} "
            f"speed_max={np.hypot(u0_, v0_).max():.3g} m/s"
        )

    # 擬似時間ステップ内で凍結する量
    rc_diag: np.ndarray | None = None  # [RC] RC 係数に含める ρV/Δτ
    tau_diag = np.zeros(n)  # ρV/Δτ（u, v 各 n 要素分）
    x_prev = x.copy()  # [残差] 擬似時間項の基準（前ステップの場）

    def state(xx: np.ndarray) -> StateArrays:
        return disc.compute_state(xx, s.scheme, s.venkat_k, rc_diag)

    def steady_resid(xx: np.ndarray) -> np.ndarray:
        if s.fast_residual:
            return disc.residual_fast(xx, s.scheme, s.venkat_k, rc_diag)
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
