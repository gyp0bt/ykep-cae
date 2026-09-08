"""nsbp ベンチ: 1 ケースを PETSc で解いて JSON 1 行を出す（mpiexec -n R で並列）.

例:
  $PETSC_DIR/bin/mpiexec -n 4 python experiments/nsbp/bench.py --model flat --refine 4 --u 0.1
  python experiments/nsbp/bench.py --model uturn --refine 2 --u 1 --compare-nsb   # 逐次で nsb と解を比較
  ... --nested 1,2,4        # 入れ子反復（粗い順、各段 tol 1e-4、双一次補間）
  ... --opts "-fieldsplit_0_sub_pc_factor_levels 0"   # PETSc オプションを上書き（prefix nsbp_ は付けなくてよい）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np

from nsb.geo import make_case
from nsb.nested import PROLONGATIONS
from nsbp.solver import NSBPSettings, NSBPSolver


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flat")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--u", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--pc", default="asm")
    ap.add_argument("--schur-fact", default="full")
    ap.add_argument("--amg", default="hypre")
    ap.add_argument("--ksp-rtol", type=float, default=1e-3)
    ap.add_argument("--jac-lag", type=int, default=1)
    ap.add_argument("--pc-lag", type=int, default=1)
    ap.add_argument("--cfl-init", type=float, default=0.25)
    ap.add_argument("--convection", default="sou")
    ap.add_argument("--venkat-k", type=float, default=5.0)
    ap.add_argument("--opts", default="")
    ap.add_argument(
        "--gopts", default="", help="prefix なしの PETSc オプション（-mat_fd_coloring_err 等）"
    )
    ap.add_argument(
        "--nested", default="", help="粗い順の refine 列 例 1,2,4（最後が --refine と一致）"
    )
    ap.add_argument("--coarse-tol", type=float, default=1e-4)
    ap.add_argument("--prolong", default="bilinear", help="bilinear / inject")
    ap.add_argument("--compare-nsb", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def prolong(u: np.ndarray, v: np.ndarray, p: np.ndarray, nx: int, ny: int, kind: str = "bilinear"):
    P = PROLONGATIONS[kind]
    while u.shape != (nx, ny):
        u, v, p = P(u), P(v), P(p)
    return u, v, p


def main() -> int:
    a = parse()
    from petsc4py import PETSc

    rank = PETSc.COMM_WORLD.getRank()
    log = (lambda m: print(m, flush=True)) if (rank == 0 and not a.quiet) else None
    settings = NSBPSettings(
        convection=a.convection,
        venkat_k=a.venkat_k,
        newton_tol=a.tol,
        newton_max_iter=a.max_iter,
        pc=a.pc,
        schur_fact=a.schur_fact,
        schur_amg=a.amg,
        ksp_rtol=a.ksp_rtol,
        jacobian_lag=a.jac_lag,
        pc_lag=a.pc_lag,
        cfl_init=a.cfl_init,
        petsc_options=a.opts,
        petsc_options_global=a.gopts,
    )
    refines = [int(t) for t in a.nested.split(",")] if a.nested else [a.refine]
    if refines[-1] != a.refine:
        raise ValueError(f"--nested の最後 {refines[-1]} は --refine {a.refine} と一致させる")
    t0 = time.perf_counter()
    prev = None
    levels = []
    res = None
    for k, r in enumerate(refines):
        tol = a.tol if k == len(refines) - 1 else a.coarse_tol
        inp = make_case(a.model, r, u_in=a.u)
        if prev is not None:
            u0, v0, p0 = prolong(prev.u, prev.v, prev.p, inp.nx, inp.ny, a.prolong)
            inp = replace(inp, u0=u0, v0=v0, p0=p0)
        solver = NSBPSolver(inp, replace(settings, newton_tol=tol))
        if log:
            log(f"[bench] level {k + 1}/{len(refines)} {inp.nx}x{inp.ny} tol={tol:.0e}")
        res = solver.solve(log)
        solver.destroy()
        levels.append(
            {
                "nx": inp.nx,
                "ny": inp.ny,
                "it": res.n_iter,
                "elapsed": res.elapsed,
                "converged": res.converged,
            }
        )
        if not res.converged:
            break
        prev = res
    total = time.perf_counter() - t0
    assert res is not None
    out = {
        "tag": a.tag,
        "model": a.model,
        "refine": a.refine,
        "u": a.u,
        "ranks": res.n_ranks,
        "pc": f"{a.pc}/{a.schur_fact}/{a.amg}",
        "ksp_rtol": a.ksp_rtol,
        "jac_lag": a.jac_lag,
        "pc_lag": a.pc_lag,
        "opts": (a.opts + " " + a.gopts).strip(),
        "nested": a.nested,
        "prolong": a.prolong if a.nested else "",
        "converged": res.converged,
        "reason": res.failure_reason,
        "it": res.n_iter,
        "it_total": int(sum(lv["it"] for lv in levels)),
        "ksp_total": res.n_ksp_total,
        "jacobians": res.n_jacobians,
        "pc_setups": res.n_pc_setups,
        "elapsed_final": res.elapsed,
        "elapsed_total": total,
        "timings": {k: round(v, 3) for k, v in res.timings.items()},
        "levels": levels,
        "mass_in": res.mass_in,
        "mass_out": res.mass_out,
        "speed_max": float(np.hypot(res.u, res.v).max()),
        "rel_steady": res.rel_steady_residual,
    }
    if a.compare_nsb and rank == 0:
        from nsb.core import NSBSettings
        from nsb.solver import solve_steady as nsb_solve

        inp_nsb = make_case(
            a.model,
            a.refine,
            u_in=a.u,
            settings=NSBSettings(newton_tol=a.tol, convection=a.convection),
        )
        t1 = time.perf_counter()
        rn = nsb_solve(inp_nsb, log)
        out["nsb"] = {
            "converged": rn.converged,
            "it": rn.n_iter,
            "elapsed": time.perf_counter() - t1,
            "gmres_total": rn.n_gmres_total,
            "rel_steady": rn.rel_steady_residual,
            "du_rel": float(np.abs(rn.u - res.u).max() / max(np.abs(rn.u).max(), 1e-300)),
            "dp_rel": float(np.abs(rn.p - res.p).max() / max(np.abs(rn.p).max(), 1e-300)),
        }
    if rank == 0:
        print("[bench-json] " + json.dumps(out), flush=True)
    return 0 if res.converged else 1


if __name__ == "__main__":
    sys.exit(main())
