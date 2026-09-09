"""物理時間の陰的非定常計算（後退 Euler + 各時間ステップを Newton で解く dual-time 型）.

`solve_steady` の擬似時間 Newton は「局所 Δτ・SER で CFL を伸ばす」ので、Stokes 出発点が遠い
高 Re の蛇行流路では CFL 大で出口 sink セルの非線形性にステップが飛び、CFL 小で前進 Euler 型の
時間発展になって降下方向にならない（status-45）。ここでは大域の物理時間刻み Δt を固定し、
各時間ステップで
    R(x) + ρV (u − u^n)/Δt = 0
を Newton（`solve_linear`、JFNK + 前処理）で解いて進める。目的は 2 つ:
  1. 定常解が存在するか（|R_steady| が時間とともに落ちるか、振動するか）の診断
  2. 定常 Newton が飛ぶ場でも時間精度を持った経路で到達点を得る
出力は各ステップの定常残差・Newton 反復数・プローブ（最大速度・運動エネルギー・出口圧）の時系列と
一定間隔の場のスナップショット。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from nsb.assembly import BrinkmanDiscretization, StateArrays
from nsb.core import NSBInput
from nsb.solver import LaggedPreconditioner, solve_linear, stokes_reference

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class UnsteadyResult:
    """非定常計算の結果.

    Parameters
    ----------
    u, v, p : np.ndarray
        最終時刻の場 (nx, ny)
    times : tuple[float, ...]
        各ステップ終了時刻 [s]
    steady_residual : tuple[float, ...]
        各ステップ終了時の定常残差 |R| / |R_ref|（時間項を含まない）
    step_residual : tuple[float, ...]
        各ステップの Newton 終了時の時間項込み残差 |R_τ| / |R_ref|
    newton_iters, gmres_iters : tuple[int, ...]
        各ステップの Newton 反復数・GMRES 反復数
    probes : dict[str, tuple[float, ...]]
        プローブ時系列（speed_max, kinetic_energy, p_range, mass_out）
    snapshots : tuple[tuple[float, np.ndarray, np.ndarray, np.ndarray], ...]
        (t, u, v, p) のスナップショット
    reached_steady : bool
        定常残差が settings.newton_tol を下回った
    n_steps_hit_max_newton : int
        Newton が上限で打ち切られたステップ数（時間精度の欠落を示す）
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    dt: float
    times: tuple[float, ...]
    steady_residual: tuple[float, ...]
    step_residual: tuple[float, ...]
    newton_iters: tuple[int, ...]
    gmres_iters: tuple[int, ...]
    probes: dict[str, tuple[float, ...]]
    snapshots: tuple[tuple[float, np.ndarray, np.ndarray, np.ndarray], ...]
    reached_steady: bool
    n_steps_hit_max_newton: int
    residual_ref: float
    elapsed: float
    failure_reason: str = ""
    extra: dict[str, float] = field(default_factory=dict)
    dt_history: tuple[float, ...] = ()
    n_dt_backoffs: int = 0


def solve_unsteady(  # noqa: PLR0913
    inp: NSBInput,
    dt: float,
    n_steps: int,
    log: LogFn | None = print,
    newton_max: int = 8,
    step_tol: float = 1.0e-3,
    step_tol_abs: float = 1.0e-5,
    save_every: int = 0,
    stop_at_steady: bool = True,
    dt_backoff_max: int = 6,
) -> UnsteadyResult:
    """後退 Euler の陰的非定常計算.

    Parameters
    ----------
    dt : float
        物理時間刻み [s]
    n_steps : int
        時間ステップ数
    newton_max : int
        1 ステップあたりの Newton 反復上限
    step_tol : float
        ステップ内の Newton 打ち切り: |R_τ| がステップ開始時の |R_τ| の step_tol 倍を下回るか、
        |R_τ|/|R_ref| < step_tol_abs
    save_every : int
        場のスナップショット間隔（0 で最終場のみ）
    stop_at_steady : bool
        定常残差 |R|/|R_ref| < settings.newton_tol になったら打ち切る
    dt_backoff_max : int
        線形解が壊れた（真の残差比 > reject_lin_ratio）・ステップ内残差が 10 倍に増えた・NaN のとき、
        同じ場から Δt を半分にしてやり直す回数の上限（2^-6 まで）。成功したステップの後は 2 倍ずつ戻す
    """
    t0 = time.perf_counter()
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    shape = (inp.nx, inp.ny)
    emit = log if log is not None else (lambda _m: None)

    tau_diag = (inp.rho * disc.vol / dt) * np.ones(n)  # ρV/Δt（大域・一定）
    x_prev = np.zeros(3 * n)

    def state(xx: np.ndarray) -> StateArrays:
        return disc.compute_state(xx, s.scheme, s.venkat_k)

    def steady_resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k)

    def resid(xx: np.ndarray) -> np.ndarray:
        r = steady_resid(xx).copy()
        r[: 2 * n] += np.concatenate([tau_diag, tau_diag]) * (xx[: 2 * n] - x_prev[: 2 * n])
        return r

    pc = LaggedPreconditioner(s, n)
    pc.cfl = 1.0  # 一定（CFL 比による前処理の組み直しは起きない）
    x_stokes, r_ref, r_init_norm, how, _ = stokes_reference(disc, s, pc, steady_resid)
    if inp.u0 is not None or inp.v0 is not None or inp.p0 is not None:
        u = np.zeros(shape) if inp.u0 is None else inp.u0.astype(float)
        v = np.zeros(shape) if inp.v0 is None else inp.v0.astype(float)
        p = np.zeros(shape) if inp.p0 is None else inp.p0.astype(float)
        x = np.concatenate([u.ravel(), v.ravel(), p.ravel()])
        init_how = "u0/v0/p0"
    else:
        x = x_stokes.copy()
        init_how = "stokes"
    r0 = r_ref
    r_steady = float(np.linalg.norm(steady_resid(x)))
    emit(
        f"[nsb-t] stokes ref ({how}): |R_stokes(0)|={r_init_norm:.4e} |R_ref|={r_ref:.4e}; "
        f"init={init_how} |R|/|R_ref|={r_steady / r0:.3e} dt={dt:g} s n_steps={n_steps} "
        f"newton_max={newton_max}"
    )

    times: list[float] = []
    hist_steady: list[float] = []
    hist_step: list[float] = []
    newton_hist: list[int] = []
    gmres_hist: list[int] = []
    dt_hist: list[float] = []
    probes: dict[str, list[float]] = {
        "speed_max": [],
        "kinetic_energy": [],
        "p_range": [],
        "mass_out": [],
    }
    snapshots: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    reached = False
    n_hit_max = 0
    failure = ""
    t = 0.0
    fd_diag = np.zeros(3 * n)  # resid に τ が入っているので matvec には足さない
    dt_cur = dt
    dt_min = dt / 2.0**dt_backoff_max
    n_backoff = 0
    step = 0
    while step < n_steps:
        # ---- 1 ステップ（Δt_cur）。線形解が壊れたら Δt を半分にして同じ場からやり直す ----
        tau_diag = (inp.rho * disc.vol / dt_cur) * np.ones(n)
        pc.cfl = dt_cur / dt  # Δt が 2 倍変わったら前処理を組み直す（precond_cfl_ratio）
        x_prev = x.copy()
        r_tau0 = float(np.linalg.norm(resid(x)))  # = 定常残差（x = x_prev）
        target = max(step_tol * r_tau0, step_tol_abs * r0)
        n_newton = 0
        n_gmres_step = 0
        r_tau = r_tau0
        bad = ""
        while n_newton < newton_max and r_tau > target:
            st = state(x)
            r_vec = resid(x)
            diag_aug = np.concatenate([tau_diag, tau_diag, np.zeros(n)])
            try:
                delta, n_g, _ok, lin_ratio = solve_linear(
                    disc,
                    st,
                    x,
                    r_vec,
                    diag_aug,
                    resid,
                    s,
                    pc,
                    fd_diag=fd_diag,
                    steady_resid_fn=steady_resid,
                )
            except (RuntimeError, ValueError) as exc:
                bad = f"lu_failed: {exc}"
                break
            n_gmres_step += n_g
            if not np.all(np.isfinite(delta)):
                bad = "gmres_breakdown"
                break
            if s.reject_lin_ratio > 0.0 and not (lin_ratio <= s.reject_lin_ratio):
                bad = f"linear solve failed (|b-Ax|/|b|={lin_ratio:.2e})"
                break
            if s.line_search_halvings > 0:
                # [ラインサーチ] ステップ内の残差 |R_τ| が減る最初の α。どの α でも減らなければ
                # この Δt では Newton が降下しないとみなして Δt を後退させる
                alpha = 1.0
                found = False
                for _k in range(s.line_search_halvings + 1):
                    if float(np.linalg.norm(resid(x + alpha * delta))) < r_tau:
                        found = True
                        break
                    alpha *= 0.5
                if not found:
                    bad = f"line search failed (alpha down to {alpha * 2:g})"
                    break
                delta = alpha * delta
            x = x + delta
            n_newton += 1
            r_tau = float(np.linalg.norm(resid(x)))
            if not np.isfinite(r_tau):
                bad = "nan"
                break
        if not bad and r_tau > 10.0 * r_tau0:
            bad = f"step residual grew {r_tau / r_tau0:.1f}x"
        if not bad and r_tau > target and dt_cur / 2.0 >= dt_min:
            # Newton が上限で打ち切られた: Δt を半分にしてやり直す（Δt 下限に達していれば受け入れて数える）
            bad = f"newton not converged in {newton_max} (|R_tau|/|R_ref|={r_tau / r0:.2e})"
        if bad:
            x = x_prev
            pc.age = 10**9
            if dt_cur / 2.0 < dt_min:
                failure = f"dt backoff exhausted ({bad})"
                break
            dt_cur *= 0.5
            n_backoff += 1
            emit(f"[nsb-t] step={step + 1} {bad}; retry with dt={dt_cur:.3g}")
            continue
        step += 1
        if n_newton >= newton_max and r_tau > target:
            n_hit_max += 1
        t += dt_cur
        r_steady = float(np.linalg.norm(steady_resid(x)))
        uu, vv, pp = disc.split(x)
        _m_in, m_out = disc.mass_flow(state(x), x)
        times.append(t)
        hist_steady.append(r_steady / r0)
        hist_step.append(r_tau / r0)
        newton_hist.append(n_newton)
        gmres_hist.append(n_gmres_step)
        dt_hist.append(dt_cur)
        probes["speed_max"].append(float(np.hypot(uu, vv).max()))
        probes["kinetic_energy"].append(float(0.5 * inp.rho * ((uu**2 + vv**2) * disc.vol).sum()))
        probes["p_range"].append(float(np.ptp(pp)))
        probes["mass_out"].append(float(m_out))
        if save_every > 0 and step % save_every == 0:
            snapshots.append((t, uu.copy(), vv.copy(), pp.copy()))
        emit(
            f"[nsb-t] step={step} t={t:.4f} dt={dt_cur:.3g} |R_steady|/|R_ref|={r_steady / r0:.3e} "
            f"|R_tau|/|R_ref|={r_tau / r0:.2e} (start {r_tau0 / r0:.2e}) newton={n_newton} "
            f"gmres={n_gmres_step} speed_max={probes['speed_max'][-1]:.3g} "
            f"KE={probes['kinetic_energy'][-1]:.4e} dp={probes['p_range'][-1]:.4e} "
            f"m_out={m_out:.4e} fact={pc.n_factorizations}"
        )
        if stop_at_steady and r_steady / r0 < s.newton_tol:
            reached = True
            break
        if r_steady / r0 > s.divergence_ratio:
            failure = "diverged"
            break
        dt_cur = min(dt, 2.0 * dt_cur)  # 成功したら Δt を戻していく
    pc.free()
    uu, vv, pp = disc.split(x)
    if not snapshots or snapshots[-1][0] != t:
        snapshots.append((t, uu.copy(), vv.copy(), pp.copy()))
    elapsed = time.perf_counter() - t0
    emit(
        f"[nsb-t] done reached_steady={reached} reason='{failure}' steps={len(times)} t={t:.4f} "
        f"final |R_steady|/|R_ref|={(hist_steady[-1] if hist_steady else float('nan')):.3e} "
        f"steps_hit_max_newton={n_hit_max} dt_backoffs={n_backoff} "
        f"factorizations={pc.n_factorizations} elapsed={elapsed:.1f}s"
    )
    return UnsteadyResult(
        u=uu.copy(),
        v=vv.copy(),
        p=pp.copy(),
        dt=dt,
        times=tuple(times),
        steady_residual=tuple(hist_steady),
        step_residual=tuple(hist_step),
        newton_iters=tuple(newton_hist),
        gmres_iters=tuple(gmres_hist),
        probes={k: tuple(v) for k, v in probes.items()},
        snapshots=tuple(snapshots),
        reached_steady=reached,
        n_steps_hit_max_newton=n_hit_max,
        residual_ref=r_ref,
        elapsed=elapsed,
        failure_reason=failure,
        dt_history=tuple(dt_hist),
        n_dt_backoffs=n_backoff,
    )
