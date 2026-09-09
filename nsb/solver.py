"""メイン: Newton + 擬似時間 + GMRES(LU 前処理) で定常解を求める.

手元構成との比較のため、制御則の「線」をすべて明示的なブロックとして書き下す:
  [Δτ]      局所/大域、速度下限の有無
  [残差]    擬似時間項を残差に含めるか（dual-time 型）
  [反復]    1 擬似時間ステップあたりの Newton 反復数（sub_iters）
  [RC]      Rhie–Chow 係数に擬似時間項を含めるか
  [線形]    JFNK: 有限差分 J v を自作の右前処理 FGMRES（`nsb.krylov`）で解く。前処理は
            jfnk_simple（SIMPLE 型ブロック前処理 ILU + Schur 補元 AMG、`nsb.precond`）か jfnk（PARDISO 疎 LU(J1)）
  [前処理]  遅延更新: 1 回の組立を precond_lag 反復まで使い回す（GMRES 不収束なら即組み直し）
  [参照場]  Stokes–Brinkman 解を常に解き、収束判定の基準 r0 と初期 CFL の基準にする
  [初期場]  NSBInput.u0/v0/p0（粗格子解の補間など）があればそれ、無ければ Stokes 解
  [SER]     残差比で CFL を増減（出発は cfl_init·|R_ref|/|R_init|）

status-40 で静止場発進・LU 直接・defect correction・CFL backtracking・速度下限なしを落とした
（いずれも実験で常に劣った。status-30 / 38 / 39）。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy import sparse

from nsb.assembly import BrinkmanDiscretization, StateArrays
from nsb.core import NSBInput, NSBResult, NSBSettings
from nsb.data import ConvectionSchemeType
from nsb.fdjac import colored_fd_jacobian
from nsb.krylov import fgmres
from nsb.linalg import PardisoLU, pardiso_solve
from nsb.precond import SimpleBlockPreconditioner

LogFn = Callable[[str], None]
LINEAR_SOLVERS = ("jfnk_simple", "jfnk")


def uses_simple_preconditioner(s: NSBSettings) -> bool:
    return s.linear_solver == "jfnk_simple"


def compute_dtau(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    cfl: float,
    s: NSBSettings,
    u_floor: float,
) -> np.ndarray:
    """[Δτ] セルごとの擬似時間増分 (nx, ny).

    Δτ_P = CFL·min(Δx,Δy) / max(|u_P|+|v_P|, u_floor)、u_floor = velocity_floor_ratio × u_scale。
    u_floor=0 の静止セルでは Δτ→∞（1e30 で打ち切り、対角補強は実質ゼロ）。
    local_dtau=False なら全セル最小値を一律に使う。
    """
    speed = np.abs(u) + np.abs(v)
    speed = np.maximum(speed, u_floor)
    with np.errstate(divide="ignore"):
        dtau = np.where(speed > 0.0, cfl * min(dx, dy) / np.maximum(speed, 1e-300), 1e30)
    dtau = np.minimum(dtau, 1e30)
    if not s.local_dtau:
        dtau = np.full_like(dtau, float(dtau.min()))
    return dtau


class LaggedPreconditioner:
    """[前処理] LU(J1 + diag) または SIMPLE 型ブロック前処理を Newton 反復間で使い回す（遅延更新）.

    `fac` は `PardisoLU`（"jfnk"）か `SimpleBlockPreconditioner`（"jfnk_simple"）で、
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
        if force or not self.fac.is_factorized:
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
    fd_diag: np.ndarray | None = None,
    steady_resid_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, int, bool, float]:
    """[線形] (J + diag_aug) δ = -R を JFNK（有限差分 matvec + FGMRES）で解く.

    戻り値 (δ, GMRES 反復数, 収束フラグ, 最終の真の残差比 |b − A δ|/|b|)。

    前処理は pc が保持する PARDISO 分解または SIMPLE 型前処理。必要なときだけ行列（J1 または
    色分け有限差分の厳密ヤコビアン、`s.jacobian`）を組んで組み直す。

    fd_diag は有限差分 matvec に足す対角。resid_fn が擬似時間項 τ(x − x_prev) を既に含むなら
    τ を二度足さないよう diag_aug から τ を除いたもの（緩和分だけ）を渡す。None なら diag_aug
    （status-45 以前の挙動: τ が 2 重に入り、前処理行列 J1+τ と作用素 J+2τ が食い違っていた）。
    """
    if fd_diag is None:
        fd_diag = diag_aug

    def assemble() -> sparse.csr_matrix:
        if s.jacobian == "fd":
            fn = steady_resid_fn if steady_resid_fn is not None else resid_fn
            J = colored_fd_jacobian(fn, x, disc.nx, disc.ny, radius=s.fd_jacobian_radius)
            if steady_resid_fn is None:
                # resid_fn に τ が入っている場合は差分にも τ が入るので diag_aug の τ を足さない
                return (J + sparse.diags(fd_diag)).tocsr()
        else:
            J = disc.jacobian_first_order(st, x=x)
        return (J + sparse.diags(diag_aug)).tocsr()

    J_cur: sparse.csr_matrix | None = None
    if pc.needs_refresh(force_refresh):
        J_cur = assemble()
        pc.refresh(J_cur)

    x_norm = float(np.linalg.norm(x))
    sqrt_eps = float(np.sqrt(np.finfo(float).eps))

    def matvec(vec: np.ndarray) -> np.ndarray:
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0.0:
            return np.zeros_like(vec)
        eps = sqrt_eps * np.sqrt(1.0 + x_norm) / v_norm
        return (resid_fn(x + eps * vec) - rhs_resid) / eps + fd_diag * vec

    def run_gmres() -> tuple[np.ndarray, int, bool]:
        # JFNK の FD matvec は厳密に線形でないので Givens 推定と真の残差が食い違い、再出発が空回りして
        # 「not converged」→ 前処理を組み直して解き直す経路に入ることがある（status-39 §3.1）。
        # Givens 推定だけで止める（check_true_residual=False）と PARDISO 側の Newton が収束しなくなった
        # 走行があったので、真の残差確認 + 不収束時の組み直しは残す
        lin_info.clear()
        return fgmres(
            matvec,
            -rhs_resid,
            precond=pc.fac.solve,
            rtol=s.gmres_tol,
            atol=0.0,
            restart=s.gmres_restart,
            maxiter=s.gmres_maxiter,
            info=lin_info,
        )

    lin_info: dict[str, float] = {}
    delta, n_gmres, ok = run_gmres()
    if not ok and pc.age > 0:
        # 古い前処理で収束しなかった: 現在の J1 で組み直して解き直す
        pc.refresh(J_cur if J_cur is not None else assemble())
        delta, n_retry, ok = run_gmres()
        n_gmres += n_retry
    pc.age += 1
    pc.last_gmres = n_gmres
    return delta, n_gmres, ok, float(lin_info.get("resid_ratio", np.nan))


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


def stokes_reference(
    disc: BrinkmanDiscretization,
    s: NSBSettings,
    pc: LaggedPreconditioner,
    steady_resid: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, float, float, str, int]:
    """[参照場] Stokes–Brinkman 解と収束判定の基準残差 |R_ref| を返す.

    運動量の対流項（inlet の運動量流束を含む）を落とした線形問題を静止場から解く。
    対流項込みの残差を静止場で解くと inlet 運動量流束がソースとして残り、流速が U_in の
    10 倍超の非物理的な噴流になるので注意。r_ref はこの場で評価した完全 NS の定常残差
    （対流の不釣り合い ρU²Δy のスケール）で、初期場の良し悪しに依らない。
    戻り値 (x_stokes, r_ref, |R_stokes(0)|, 解法の説明, GMRES 反復数)。
    """
    n = disc.n
    n_gmres = 0
    x_zero = np.zeros(3 * n)
    st0 = disc.compute_state(x_zero, s.scheme, s.venkat_k)
    r_init = disc.residual_from_state(x_zero, st0, convection=False)
    J0 = disc.jacobian_first_order(st0, convection=False, x=x_zero).tocsr()
    pc.refresh(J0)
    if uses_simple_preconditioner(s):
        # 線形問題なので SIMPLE 前処理付き GMRES で厳密に解く（収束しなければ PARDISO 1 回）
        dx0, n_g0, ok0 = _gmres_exact(J0, -r_init, pc.fac.solve, s)
        n_gmres += n_g0
        how = f"gmres={n_g0}"
        if not ok0:
            dx0 = pardiso_solve(J0, -r_init)
            how = f"gmres={n_g0} (not converged) -> pardiso"
    else:
        dx0 = pc.fac.solve(-r_init)
        how = "pardiso"
    x_stokes = x_zero + dx0
    pc.age = 10**9  # Stokes 行列は前処理として使わない（次で必ず組み直す）
    r_ref = float(np.linalg.norm(steady_resid(x_stokes)))
    if r_ref == 0.0:  # Re → 0 の極限（Stokes 場が厳密解）: 静止場の残差を基準にする
        r_ref = float(np.linalg.norm(steady_resid(x_zero)))
    r_ref = max(r_ref, 1e-300)
    return x_stokes, r_ref, float(np.linalg.norm(r_init)), how, n_gmres


def solve_steady(inp: NSBInput, log: LogFn | None = print) -> NSBResult:
    """定常 Brinkman-NS を Newton + 擬似時間で解く（収束しなければ converged=False）."""
    t0 = time.perf_counter()
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    shape = (inp.nx, inp.ny)
    emit = log if log is not None else (lambda _m: None)
    u_floor = s.velocity_floor_ratio * disc.u_scale

    # 擬似時間ステップ内で凍結する量
    rc_diag: np.ndarray | None = None  # [RC] RC 係数に含める ρV/Δτ
    tau_diag = np.zeros(n)  # ρV/Δτ（u, v 各 n 要素分）
    cp_diag = np.zeros(n)  # [残差] 圧力の擬似時間対角 τ/(ρ (β u_scale)²)（人工圧縮性、β=0 なら 0）
    x_prev = np.zeros(3 * n)  # [残差] 擬似時間項の基準（前ステップの場）
    psi_frozen: tuple[np.ndarray, np.ndarray] | None = None  # [リミター凍結] 凍結した (ψ_u, ψ_v)

    def state(xx: np.ndarray) -> StateArrays:
        return disc.compute_state(xx, s.scheme, s.venkat_k, rc_diag, psi=psi_frozen)

    def steady_resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k, rc_diag, psi=psi_frozen)

    def resid(xx: np.ndarray) -> np.ndarray:
        """[残差] 擬似時間項込みの残差 R_τ = R + ρV(u - u_prev)/Δτ（u, v 成分のみ）."""
        r = steady_resid(xx)
        if s.pseudo_time_in_residual:
            r = r.copy()
            r[: 2 * n] += np.concatenate([tau_diag, tau_diag]) * (xx[: 2 * n] - x_prev[: 2 * n])
            if s.pseudo_compressibility > 0.0:
                r[2 * n :] += cp_diag * (xx[2 * n :] - x_prev[2 * n :])
        return r

    # [参照場] Stokes–Brinkman 解: 運動量の対流項（inlet の運動量流束を含む）を落とした線形問題を
    # 静止場から解く。対流項込みの残差を静止場で解くと inlet 運動量流束がソースとして残り、
    # 流速が U_in の 10 倍超の非物理的な噴流になるので注意。
    # 収束判定の基準 r_ref はこの場で評価した完全 NS の定常残差（対流の不釣り合い ρU²Δy のスケール）で、
    # 初期場の良し悪しに依らない
    pc = LaggedPreconditioner(s, n)
    x_stokes, r_ref, r_init_norm, how, n_gmres_total = stokes_reference(disc, s, pc, steady_resid)

    # [初期場] u0/v0/p0 があればそれ（粗格子解の補間など）、無ければ Stokes 解
    if inp.u0 is not None or inp.v0 is not None or inp.p0 is not None:
        u = np.zeros(shape) if inp.u0 is None else inp.u0.astype(float)
        v = np.zeros(shape) if inp.v0 is None else inp.v0.astype(float)
        p = np.zeros(shape) if inp.p0 is None else inp.p0.astype(float)
        x = np.concatenate([u.ravel(), v.ravel(), p.ravel()])
        init_how = "u0/v0/p0"
    else:
        x = x_stokes.copy()
        init_how = "stokes"
    x_prev = x.copy()

    r = steady_resid(x)
    r_norm = float(np.linalg.norm(r))
    r0 = r_ref
    # [SER] 古典形 CFL = cfl_init·|R_ref|/|R|: 初期場が参照場より良ければその分だけ大きく出発する
    cfl = float(min(s.cfl_max, s.cfl_init * r0 / max(r_norm, 1e-300)))
    hist = [r_norm]
    hist_steady = [r_norm]
    cfl_hist: list[float] = []
    converged = False
    failure = ""
    n_iter = 0
    u0_, v0_, _ = disc.split(x_stokes)
    emit(
        f"[nsb] stokes ref ({how}): |R_stokes(0)|={r_init_norm:.4e} "
        f"|R_ref|={r_ref:.4e} speed_max={np.hypot(u0_, v0_).max():.3g} m/s"
    )
    emit(f"[nsb] it=0 init={init_how} |R|={r_norm:.4e} rel={r_norm / r0:.3e} cfl={cfl:.3g}")

    # [リミター凍結] 判定用の状態
    freeze_on = (
        s.limiter_freeze_rel > 0.0 and s.scheme is not ConvectionSchemeType.FIRST_ORDER_UPWIND
    )
    psi_last: tuple[np.ndarray, np.ndarray] | None = None
    r_steady_cur = r_norm  # 直近の定常残差
    r_freeze_min = float("inf")  # 凍結以降の定常残差の最小値（解凍判定）
    frozen_at = -1
    freeze_cooldown = 0
    n_above = 0

    n_refreeze = 0

    while n_iter < s.newton_max_iter:
        if not np.isfinite(r_norm):
            failure = "nan"
            break
        if r_norm / r0 < s.newton_tol:
            if psi_frozen is None or n_refreeze >= s.limiter_refreeze_max:
                # 凍結中なら「凍結問題の収束」。解凍した真の残差は末尾で residual_unfrozen に報告する
                converged = True
                break
            # [リミター凍結] 凍結問題が収束した: 解凍して真の残差を見る。判定を満たさなければ
            # 現在の場の ψ で凍結し直して続ける（ψ の Picard 反復。凍結時点の ψ は解の ψ と違うので
            # 凍結問題の解の真の残差は 1e-4〜1e-3 程度に留まる）
            psi_frozen = None
            r_true = float(np.linalg.norm(steady_resid(x)))
            if r_true / r0 < s.newton_tol:
                r_norm = r_true
                converged = True
                break
            n_refreeze += 1
            psi_frozen = disc.limiter(x, s.venkat_k)
            r_norm = r_true
            r_steady_cur = r_true
            r_freeze_min = r_true
            pc.age = 10**9
            emit(
                f"[nsb] it={n_iter} frozen problem converged; unfrozen |R_steady|/|R_ref|="
                f"{r_true / r0:.3e} -> re-frozen with current psi (#{n_refreeze})"
            )
        if r_norm / r0 > s.divergence_ratio:
            failure = "diverged"
            break

        # ---- [リミター凍結] 残差が落ちて ψ が動かなくなったら凍結、凍結後に残差が跳ねたら解凍 ----
        if freeze_on and psi_frozen is None:
            psi_now = disc.limiter(x, s.venkat_k)
            if (
                psi_last is not None
                and freeze_cooldown <= 0
                and r_steady_cur / r0 < s.limiter_freeze_rel
            ):
                moved = float(
                    np.mean(
                        (np.abs(psi_now[0] - psi_last[0]) > s.limiter_freeze_delta)
                        | (np.abs(psi_now[1] - psi_last[1]) > s.limiter_freeze_delta)
                    )
                )
                if moved < s.limiter_freeze_stable_frac:
                    psi_frozen = psi_now
                    frozen_at = n_iter
                    r_freeze_min = r_steady_cur
                    pc.age = 10**9  # 残差関数が変わるので前処理を組み直す
                    emit(
                        f"[nsb] it={n_iter} limiter frozen (|R_steady|/|R_ref|={r_steady_cur / r0:.2e}, "
                        f"moved={moved:.3%}, psi<1 in {np.mean(psi_now[0] < 0.999):.1%} cells)"
                    )
            psi_last = psi_now
            freeze_cooldown -= 1

        # ---- 擬似時間ステップ開始: Δτ を決めて凍結 ----
        uu, vv, _ = disc.split(x)
        dtau = compute_dtau(uu, vv, disc.dx, disc.dy, cfl, s, u_floor)
        tau_diag = (inp.rho * disc.vol / dtau).ravel()
        if s.pseudo_compressibility > 0.0:
            cp_diag = tau_diag / (inp.rho * (s.pseudo_compressibility * disc.u_scale) ** 2)
        rc_diag = tau_diag.reshape(shape) if s.rc_with_pseudo_time else None
        x_prev = x.copy()
        pc.cfl = cfl

        r_new = r_norm
        lin_rejected = False
        for _sub in range(s.sub_iters):
            st = state(x)
            r_tau = resid(x)
            relax = (1.0 - s.alpha_u) / s.alpha_u * st.a_p.ravel()
            diag_aug = np.concatenate([tau_diag + relax, tau_diag + relax, cp_diag])
            # resid に τ(x − x_prev) が入っているときは有限差分 matvec にも τ が出るので、足すのは緩和分だけ
            fd_diag = (
                np.concatenate([relax, relax, np.zeros(n)])
                if s.pseudo_time_in_residual
                else diag_aug
            )
            try:
                delta, n_gmres, lin_ok, lin_ratio = solve_linear(
                    disc,
                    st,
                    x,
                    r_tau,
                    diag_aug,
                    resid,
                    s,
                    pc,
                    fd_diag=fd_diag,
                    steady_resid_fn=steady_resid,
                )
            except (RuntimeError, ValueError) as exc:
                failure = f"lu_failed: {exc}"
                break
            n_gmres_total += n_gmres
            if not np.all(np.isfinite(delta)):
                failure = "gmres_breakdown"
                break
            if s.reject_lin_ratio > 0.0 and not (lin_ratio <= s.reject_lin_ratio):
                # 線形解が実質失敗（真の残差比が閾値超）: 修正量は捨てて CFL を下げ、同じ場からやり直す
                n_iter += 1
                x = x_prev
                cfl = float(cfl * s.ser_shrink)
                emit(
                    f"[nsb] it={n_iter} linear solve failed (gmres={n_gmres}, "
                    f"|b-Ax|/|b|={lin_ratio:.2e}), rejected, cfl -> {cfl:.3g}"
                )
                lin_rejected = True
                break
            if s.line_search_halvings > 0:
                # [ラインサーチ] 定常残差が減る最初の α を採る。どの α でも減らなければ修正量を捨てて
                # CFL を ser_shrink 倍にし、同じ場からやり直す（線形解の棄却と同じ扱い）
                r_cur = float(np.linalg.norm(steady_resid(x)))
                alpha = 1.0
                found = False
                for _k in range(s.line_search_halvings + 1):
                    if float(np.linalg.norm(steady_resid(x + alpha * delta))) < r_cur:
                        found = True
                        break
                    alpha *= 0.5
                if not found:
                    n_iter += 1
                    x = x_prev
                    cfl = float(cfl * s.ser_shrink)
                    emit(
                        f"[nsb] it={n_iter} line search failed (alpha down to {alpha * 2:g}), "
                        f"rejected, cfl -> {cfl:.3g}"
                    )
                    lin_rejected = True
                    break
                if alpha < 1.0:
                    emit(f"[nsb] line search: alpha={alpha:g}")
                delta = alpha * delta
            x = x + delta
            n_iter += 1
            # ステップ終了時の残差（擬似時間項込み: 収束判定・SER に使う）と定常残差
            r_new = float(np.linalg.norm(resid(x)))
            r_steady_new = float(np.linalg.norm(steady_resid(x)))
            hist.append(r_new)
            hist_steady.append(r_steady_new)
            r_steady_cur = r_steady_new
            if psi_frozen is not None:
                r_freeze_min = min(r_freeze_min, r_steady_new)
                # 解凍は「最小値の unfreeze_ratio 倍超」が patience 回続いたとき（1 反復の跳ねは次で戻る
                # ことが多く、そのたびに解凍すると前処理と CFL がリセットされて育たない: trama 内部ポート）
                if r_steady_new > s.limiter_unfreeze_ratio * r_freeze_min:
                    n_above += 1
                else:
                    n_above = 0
                if n_above >= s.limiter_unfreeze_patience:
                    psi_frozen = None
                    psi_last = None
                    n_above = 0
                    freeze_cooldown = 5
                    pc.age = 10**9
                    emit(
                        f"[nsb] it={n_iter} limiter unfrozen (|R_steady| stayed above "
                        f"{s.limiter_unfreeze_ratio:g}x the post-freeze minimum for "
                        f"{s.limiter_unfreeze_patience} iterations)"
                    )
            emit(
                f"[nsb] it={n_iter} |R_tau|={r_new:.4e} rel={r_new / r0:.3e} "
                f"|R_steady|/|R_ref|={r_steady_new / r0:.3e} cfl={cfl:.3g} "
                f"dtau=[{dtau.min():.2e},{dtau.max():.2e}] gmres={n_gmres} "
                f"pc_age={pc.age} fact={pc.n_factorizations}"
                f"{'' if lin_ok else f' (gmres not converged: {lin_ratio:.1e})'}"
            )
            if not np.isfinite(r_new):
                break
        if failure:
            break
        if lin_rejected:
            cfl_hist.append(cfl)
            continue

        # ---- [SER] 残差比で CFL を更新（乗法形: 減少率下限 ser_shrink、成長率上限 ser_growth）----
        ratio = r_norm / r_new if r_new > 0.0 and np.isfinite(r_new) else s.ser_shrink
        cfl = float(min(s.cfl_max, cfl * float(np.clip(ratio, s.ser_shrink, s.ser_growth))))
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
    residual_unfrozen = -1.0
    if frozen_at >= 0:
        psi_frozen = None
        residual_unfrozen = float(np.linalg.norm(steady_resid(x)))
        emit(
            f"[nsb] limiter was frozen at it={frozen_at} (re-frozen {n_refreeze}x); "
            f"unfrozen |R_steady|/|R_ref|={residual_unfrozen / r0:.3e}"
        )
    elapsed = time.perf_counter() - t0
    emit(
        f"[nsb] done converged={converged} reason='{failure}' it={n_iter} "
        f"m_in={m_in:.4e} m_out={m_out:.4e} "
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
        residual_ref=r_ref,
        n_factorizations=pc.n_factorizations,
        n_gmres_total=n_gmres_total,
        limiter_frozen_at=frozen_at,
        residual_unfrozen=residual_unfrozen,
    )
