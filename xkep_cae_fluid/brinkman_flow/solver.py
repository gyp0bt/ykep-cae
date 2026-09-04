"""2D Brinkman 補正 Navier-Stokes ソルバー (FVM, Newton–Krylov).

外側 Newton、内側 GMRES（LU(J1) 前処理）、擬似時間増分 + 陰的緩和。
残差は選択した対流スキーム、ヤコビアンは常に 1 次風上で組む。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization
from xkep_cae_fluid.brinkman_flow.data import (
    BrinkmanFlowInput,
    BrinkmanFlowResult,
    JacobianMode,
)
from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess

LogFn = Callable[[str], None]


class _GmresCounter:
    """GMRES 反復数カウンタ（callback）."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, _pr_norm: float) -> None:
        self.n += 1


def _make_jfnk_matvec(
    resid: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    r: np.ndarray,
    aug: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """有限差分 Jacobian-vector 積 (J v ≈ (R(x+εv)-R(x))/ε) + 対角補強."""
    x_norm = float(np.linalg.norm(x))
    sqrt_eps = float(np.sqrt(np.finfo(float).eps))

    def matvec(vec: np.ndarray) -> np.ndarray:
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0.0:
            return np.zeros_like(vec)
        eps = sqrt_eps * np.sqrt(1.0 + x_norm) / v_norm
        return (resid(x + eps * vec) - r) / eps + aug * vec

    return matvec


class BrinkmanFlowFVMProcess(SolverProcess["BrinkmanFlowInput", "BrinkmanFlowResult"]):
    """Brinkman 補正付き 2D 定常 Navier-Stokes を Newton–Krylov で解く.

    収束しない場合は converged=False と failure_reason を返す（例外は投げない）。
    `log` に関数を渡すと反復ごとの残差・CFL・GMRES 反復数を出力する。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="BrinkmanFlowFVM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/brinkman-flow-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def __init__(self, log: LogFn | None = None) -> None:
        self._log = log

    def _emit(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def process(self, input_data: BrinkmanFlowInput) -> BrinkmanFlowResult:
        t0 = time.perf_counter()
        inp = input_data
        s = inp.settings
        disc = BrinkmanDiscretization(inp)
        n = disc.n
        shape = (inp.nx, inp.ny)

        u = np.zeros(shape) if inp.u0 is None else inp.u0.astype(np.float64)
        v = np.zeros(shape) if inp.v0 is None else inp.v0.astype(np.float64)
        p = np.zeros(shape) if inp.p0 is None else inp.p0.astype(np.float64)
        x = np.concatenate([u.ravel(), v.ravel(), p.ravel()])

        def resid(xx: np.ndarray) -> np.ndarray:
            return disc.residual(xx, s.convection_scheme, s.venkat_k)

        def norms(r: np.ndarray) -> tuple[float, float, float, float]:
            ru, rv, rp = r[:n], r[n : 2 * n], r[2 * n :]
            return (
                float(np.linalg.norm(r)),
                float(np.linalg.norm(ru)),
                float(np.linalg.norm(rv)),
                float(np.linalg.norm(rp)),
            )

        st = disc.compute_state(x, s.convection_scheme, s.venkat_k)
        r = disc.residual_from_state(x, st)
        r_norm, ru, rv, rp = norms(r)
        r0 = max(r_norm, 1e-300)

        hist: list[float] = [r_norm]
        comps: list[tuple[float, float, float]] = [(ru, rv, rp)]
        cfl_hist: list[float] = []
        gmres_hist: list[int] = []
        cfl = s.cfl_init
        converged = False
        failure = ""
        n_newton = 0
        u_floor = 0.1 * abs(inp.u_inlet)
        h_min = min(disc.dx, disc.dy)

        self._emit(
            f"[newton] it=0 |R|={r_norm:.4e} (u={ru:.2e} v={rv:.2e} p={rp:.2e}) cfl={cfl:.3g}"
        )

        for it in range(1, s.newton_max_iter + 1):
            if not np.isfinite(r_norm):
                failure = "nan"
                break
            if r_norm / r0 < s.newton_tol or r_norm < s.newton_abs_tol:
                converged = True
                break
            if r_norm / r0 > s.divergence_ratio:
                failure = "diverged"
                break

            # 擬似時間 + 陰的緩和の対角補強
            uu, vv, _ = disc.split(x)
            speed = np.maximum(np.abs(uu) + np.abs(vv), u_floor)
            dtau = cfl * h_min / speed
            aug = inp.rho * disc.vol / dtau + (1.0 - s.alpha_u) / s.alpha_u * st.a_p
            aug_vec = np.concatenate([aug.ravel(), aug.ravel(), np.zeros(n)])

            J1 = disc.jacobian_first_order(st) + sparse.diags(aug_vec)
            try:
                lu = spla.splu(J1.tocsc())
            except RuntimeError as exc:
                failure = f"lu_failed: {exc}"
                break

            n_gmres = 0
            if s.jacobian_mode is JacobianMode.DEFECT_CORRECTION:
                delta = lu.solve(-r)
                gmres_ok = True
            else:
                matvec = _make_jfnk_matvec(resid, x, r, aug_vec)
                counter = _GmresCounter()

                A = spla.LinearOperator((3 * n, 3 * n), matvec=matvec, dtype=np.float64)
                M = spla.LinearOperator((3 * n, 3 * n), matvec=lu.solve, dtype=np.float64)
                delta, info = spla.gmres(
                    A,
                    -r,
                    M=M,
                    rtol=s.gmres_tol,
                    atol=0.0,
                    restart=s.gmres_restart,
                    maxiter=s.gmres_maxiter,
                    callback=counter,
                    callback_type="pr_norm",
                )
                n_gmres = counter.n
                gmres_ok = info == 0
                if not np.all(np.isfinite(delta)):
                    failure = "gmres_breakdown"
                    break

            if s.line_search:
                lam = 1.0
                for _ in range(8):
                    x_try = x + lam * delta
                    r_try = resid(x_try)
                    if (
                        np.isfinite(r_try).all()
                        and np.linalg.norm(r_try) < (1.0 - 1e-4 * lam) * r_norm
                    ):
                        break
                    lam *= 0.5
                x = x + lam * delta
            else:
                x = x + delta

            st = disc.compute_state(x, s.convection_scheme, s.venkat_k)
            r = disc.residual_from_state(x, st)
            r_new, ru, rv, rp = norms(r)
            ratio = r_norm / r_new if r_new > 0.0 and np.isfinite(r_new) else 0.1
            cfl = float(min(s.cfl_max, cfl * float(np.clip(ratio, 0.1, s.ser_growth))))
            r_norm = r_new
            n_newton = it
            hist.append(r_norm)
            comps.append((ru, rv, rp))
            cfl_hist.append(cfl)
            gmres_hist.append(n_gmres)
            self._emit(
                f"[newton] it={it} |R|={r_norm:.4e} rel={r_norm / r0:.3e} "
                f"(u={ru:.2e} v={rv:.2e} p={rp:.2e}) cfl={cfl:.3g} gmres={n_gmres}"
                f"{'' if gmres_ok else ' (gmres not converged)'}"
            )
        else:
            if np.isfinite(r_norm) and (r_norm / r0 < s.newton_tol or r_norm < s.newton_abs_tol):
                converged = True
            elif not np.isfinite(r_norm):
                failure = "nan"
            else:
                failure = "max_iter"

        u, v, p = disc.split(x)
        m_in, m_out = disc.mass_flow(st)
        elapsed = time.perf_counter() - t0
        self._emit(
            f"[newton] done converged={converged} reason='{failure}' it={n_newton} "
            f"m_in={m_in:.4e} m_out={m_out:.4e} elapsed={elapsed:.1f}s"
        )
        return BrinkmanFlowResult(
            u=u.copy(),
            v=v.copy(),
            p=p.copy(),
            converged=converged,
            failure_reason=failure,
            n_newton=n_newton,
            residual_history=tuple(hist),
            residual_components=tuple(comps),
            cfl_history=tuple(cfl_hist),
            gmres_iterations=tuple(gmres_hist),
            mass_in=m_in,
            mass_out=m_out,
            elapsed_seconds=elapsed,
        )
