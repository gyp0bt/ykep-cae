"""メイン: Newton + 擬似時間 + GMRES(LU 前処理) で定常解を求める.

手元構成との比較のため、制御則の「線」をすべて明示的なブロックとして書き下す:
  [Δτ]      局所/大域、速度下限の有無
  [残差]    擬似時間項を残差に含めるか（dual-time 型）
  [反復]    1 擬似時間ステップあたりの Newton 反復数（sub_iters）
  [RC]      Rhie–Chow 係数に擬似時間項を含めるか
  [線形]    JFNK（GMRES + LU(J1)）か LU 直接（defect correction）
  [SER]     残差比で CFL を増減
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from nsb.core import NSBInput, NSBResult, NSBSettings
from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization, StateArrays

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


def solve_linear(
    disc: BrinkmanDiscretization,
    st: StateArrays,
    x: np.ndarray,
    rhs_resid: np.ndarray,
    diag_aug: np.ndarray,
    resid_fn: Callable[[np.ndarray], np.ndarray],
    s: NSBSettings,
) -> tuple[np.ndarray, int, bool]:
    """[線形] (J + diag_aug) δ = -R を解く。戻り値 (δ, GMRES 反復数, 収束フラグ)."""
    n3 = x.size
    J1 = disc.jacobian_first_order(st) + sparse.diags(diag_aug)
    lu = spla.splu(J1.tocsc())
    if s.linear_solver == "lu":
        return lu.solve(-rhs_resid), 0, True

    x_norm = float(np.linalg.norm(x))
    sqrt_eps = float(np.sqrt(np.finfo(float).eps))

    def matvec(vec: np.ndarray) -> np.ndarray:
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0.0:
            return np.zeros_like(vec)
        eps = sqrt_eps * np.sqrt(1.0 + x_norm) / v_norm
        return (resid_fn(x + eps * vec) - rhs_resid) / eps + diag_aug * vec

    count = [0]

    def cb(_: float) -> None:
        count[0] += 1

    A = spla.LinearOperator((n3, n3), matvec=matvec, dtype=np.float64)
    M = spla.LinearOperator((n3, n3), matvec=lu.solve, dtype=np.float64)
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

        step_ok = True
        for _sub in range(s.sub_iters):
            st = state(x)
            r_tau = resid(x)
            relax = (1.0 - s.alpha_u) / s.alpha_u * st.a_p.ravel()
            diag_aug = np.concatenate([tau_diag + relax, tau_diag + relax, np.zeros(n)])
            try:
                delta, n_gmres, lin_ok = solve_linear(disc, st, x, r_tau, diag_aug, resid, s)
            except RuntimeError as exc:
                failure = f"lu_failed: {exc}"
                step_ok = False
                break
            if not np.all(np.isfinite(delta)):
                failure = "gmres_breakdown"
                step_ok = False
                break
            x = x + delta
            n_iter += 1

            # ステップ終了時の残差（擬似時間項込み: 収束判定・SER に使う）と定常残差
            r_tau_new = resid(x)
            r_new = float(np.linalg.norm(r_tau_new))
            r_steady_new = float(np.linalg.norm(steady_resid(x)))
            hist.append(r_new)
            hist_steady.append(r_steady_new)
            emit(
                f"[nsb] it={n_iter} |R_tau|={r_new:.4e} rel={r_new / r0:.3e} "
                f"|R_steady|/|R0|={r_steady_new / r_steady0:.3e} cfl={cfl:.3g} "
                f"dtau=[{dtau.min():.2e},{dtau.max():.2e}] gmres={n_gmres}"
                f"{'' if lin_ok else ' (gmres not converged)'}"
            )
            if not np.isfinite(r_new):
                break
        if not step_ok:
            break

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

    u, v, p = disc.split(x)
    rc_diag = None
    m_in, m_out = disc.mass_flow(state(x))
    elapsed = time.perf_counter() - t0
    emit(
        f"[nsb] done converged={converged} reason='{failure}' it={n_iter} "
        f"m_in={m_in:.4e} m_out={m_out:.4e} elapsed={elapsed:.1f}s"
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
    )
