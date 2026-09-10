"""trama 蛇行流路を nsbp（PETSc）で解く — nsb と同じ継続法で反復数と時間を比べる.

離散化は nsb と同一（`NSBInput` をそのまま渡す）。違うのはソルバー部品だけ:

| | nsb | nsbp |
|---|---|---|
| ヤコビアン | 1 次風上 J1（前処理用）+ JFNK 差分 matvec | SNES の FD カラーリングで**厳密 J** |
| 前処理 | J1+τ の疎直接 LU（PARDISO）または SIMPLE 型ブロック | ASM(重なり 2) + ILU(2) |
| 並列 | なし（MKL のスレッドのみ） | DMDA の MPI 分割 |

nsb の三角解（279600 自由度で 1 回 180 ms）が費用を支配していたので、ILU(2) に置き換えて
効くかどうかを測るのが目的。高抗力コントラスト（流路と閉塞で 1.4e5 倍）で ILU が保つかは未知。

    python experiments/nsb/trama_nsbp.py --port wall --continuation 0.0015,0.005,0.015 \
        2>&1 | tee experiments/nsb/logs/trama-nsbp-$(date +%s).log
    $(python -c "from nsbp.launch import mpiexec_path; print(mpiexec_path())") -n 8 \
        python experiments/nsb/trama_nsbp.py --port wall --continuation 0.0015,0.005,0.015
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trama_case import DEFAULT_PATTERN, load_trama, make_trama_input  # noqa: E402

from nsb.core import NSBInput  # noqa: E402
from nsbp.solver import NSBPSettings, NSBPSolver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", nargs="?", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--mass", type=float, default=0.015)
    ap.add_argument("--port", default="wall", choices=["interior", "wall"])
    ap.add_argument("--continuation", default="", help="カンマ区切りの質量流量")
    ap.add_argument("--freeze", type=float, default=1.0e-3, help="limiter_freeze_tol")
    ap.add_argument("--pc", default="asm", choices=["asm", "bjacobi", "schur", "lu"])
    ap.add_argument("--ksp-rtol", type=float, default=1.0e-3)
    ap.add_argument("--jac-lag", type=int, default=1)
    ap.add_argument("--pc-lag", type=int, default=1)
    ap.add_argument("--cfl-init", type=float, default=0.25)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--tol", type=float, default=1.0e-6)
    ap.add_argument("--opts", default="")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_nsbp.json")
    a = ap.parse_args()

    from petsc4py import PETSc

    rank = PETSc.COMM_WORLD.getRank()
    log = (lambda m: print(m, flush=True)) if rank == 0 else None

    def emit(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    geo = load_trama(a.pattern)
    settings = NSBPSettings(
        newton_tol=a.tol,
        newton_max_iter=a.max_iter,
        pc=a.pc,
        ksp_rtol=a.ksp_rtol,
        jacobian_lag=a.jac_lag,
        pc_lag=a.pc_lag,
        cfl_init=a.cfl_init,
        limiter_freeze_tol=a.freeze,
        petsc_options=a.opts,
    )
    masses = [float(v) for v in a.continuation.split(",")] if a.continuation else [a.mass]
    stages = []
    prev = None
    prev_m = masses[0]
    t_all = time.perf_counter()
    for m_k in masses:
        inp = make_trama_input(geo, m_k, dx_mm=a.dx, port=a.port)
        if prev is not None:
            r = m_k / prev_m
            inp = NSBInput(
                **{k: v for k, v in inp.__dict__.items() if k not in ("u0", "v0", "p0")},
                u0=prev.u * r,
                v0=prev.v * r,
                p0=prev.p * r,
            )
        emit(f"[nsbp-trama] === stage mass={m_k:g} ({inp.nx}x{inp.ny}, init from {prev_m:g}) ===")
        solver = NSBPSolver(inp, replace(settings))
        try:
            res = solver.solve(log)
        finally:
            solver.destroy()
        rel = (
            res.steady_residual_history[-1] / res.residual_ref if res.residual_ref else float("nan")
        )
        stages.append(
            {
                "mass": m_k,
                "converged": bool(res.converged),
                "reason": res.failure_reason,
                "n_iter": int(res.n_iter),
                "rel_steady_final": float(rel),
                "elapsed": float(res.elapsed),
                "ksp_total": int(res.n_ksp_total),
                "jacobians": int(res.n_jacobians),
                "pc_setups": int(res.n_pc_setups),
                "n_ranks": int(res.n_ranks),
                "timings": {k: round(v, 3) for k, v in res.timings.items()},
            }
        )
        emit(
            f"[nsbp-trama] stage mass={m_k:g}: converged={res.converged} it={res.n_iter} "
            f"rel={rel:.2e} ksp={res.n_ksp_total} {res.elapsed:.1f}s"
        )
        if not res.converged:
            break
        prev, prev_m = res, m_k
    out = {
        "port": a.port,
        "dx_mm": a.dx,
        "pc": a.pc,
        "ranks": stages[-1]["n_ranks"] if stages else 0,
        "elapsed_total": time.perf_counter() - t_all,
        "it_total": sum(s["n_iter"] for s in stages),
        "stages": stages,
    }
    if rank == 0:
        Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
