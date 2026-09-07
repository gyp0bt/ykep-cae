"""線形ソルバー方式の比較: PARDISO LU 前処理（jfnk）vs SIMPLE 型ブロック前処理（jfnk_simple）.

    python experiments/nsb/bench_precond.py [refine ...] 2>&1 | tee experiments/nsb/logs/bench-precond-$(date +%s).log

flat、U=1、推奨構成（velocity_floor_ratio=0.1 U、Stokes 初期場、alpha_u=1）で refine=1/2/4（72×48 / 144×96 / 288×192）を
解き、収束・Newton 反復数・GMRES 総反復・前処理組立回数・所要時間・段別内訳を出す。解は jfnk（PARDISO）を基準に
最大差で照合する。status-38 の結果（4 コア、scipy gmres）: experiments/nsb/logs/bench-precond-flat-r124.log、
status-39 の結果（20 コア、FGMRES + SA 階層再利用 + numba 残差）: experiments/nsb/logs/bench-precond-flat-r124-status39.log
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.sparse import linalg as spla

from nsb import NSBSettings, krylov, make_case, solve_steady
from nsb import assembly as asm
from nsb import solver as nsolver
from nsb.linalg import PardisoLU
from nsb.precond import SimpleBlockPreconditioner

T: dict[str, float] = {}
C: dict[str, int] = {}


def wrap(obj: object, name: str, key: str) -> None:
    f = getattr(obj, name)

    def g(*a, **k):
        t = time.perf_counter()
        r = f(*a, **k)
        T[key] = T.get(key, 0.0) + time.perf_counter() - t
        C[key] = C.get(key, 0) + 1
        return r

    setattr(obj, name, g)


wrap(asm.BrinkmanDiscretization, "compute_state", "compute_state")
wrap(asm.BrinkmanDiscretization, "residual_from_state", "residual")
wrap(asm.BrinkmanDiscretization, "residual_fast", "residual_fast")
wrap(asm.BrinkmanDiscretization, "jacobian_first_order", "jacobian")
wrap(PardisoLU, "factorize", "pardiso factorize")
wrap(PardisoLU, "solve", "pardiso solve")
wrap(SimpleBlockPreconditioner, "factorize", "simple setup")
wrap(SimpleBlockPreconditioner, "solve", "simple apply")
wrap(spla, "spilu", "spilu")  # 呼び出し回数 > 組立回数なら零ピボットの組み直しが起きている
wrap(krylov, "fgmres", "gmres(total)")
nsolver.fgmres = krylov.fgmres

CONFIGS: tuple[tuple[str, dict[str, object]], ...] = (
    ("jfnk (pardiso, lag=4)", {"linear_solver": "jfnk", "precond_lag": 4}),
    ("jfnk_simple lag=1", {"linear_solver": "jfnk_simple", "precond_lag": 1}),
    ("jfnk_simple lag=4", {"linear_solver": "jfnk_simple", "precond_lag": 4}),
    (
        "jfnk_simple lag=4 gmres_tol=1e-2",
        {"linear_solver": "jfnk_simple", "precond_lag": 4, "gmres_tol": 1e-2},
    ),
    (
        "jfnk_simple lag=4 schur_cycles=2",
        {"linear_solver": "jfnk_simple", "precond_lag": 4, "simple_schur_cycles": 2},
    ),
)


def run(refine: int, u_in: float = 1.0) -> None:
    base = NSBSettings(
        velocity_floor_ratio=0.1,
        alpha_u=1.0,
        newton_max_iter=120,
        precond_cfl_ratio=2.0,
    )
    ref = None
    for label, kw in CONFIGS:
        T.clear()
        C.clear()
        inp = make_case("flat", refine, u_in, settings=replace(base, **kw))
        t0 = time.perf_counter()
        res = solve_steady(inp, log=None)
        tot = time.perf_counter() - t0
        if ref is None:
            ref = res
        du = float(np.abs(res.u - ref.u).max() / np.abs(ref.u).max())
        dp = float(np.abs(res.p - ref.p).max() / np.abs(ref.p).max())
        print(
            f"\n== flat refine={refine} ({inp.nx}x{inp.ny}, n3={3 * inp.nx * inp.ny}) {label}: "
            f"converged={res.converged} reason='{res.failure_reason}' it={res.n_iter} "
            f"setups={res.n_factorizations} gmres={res.n_gmres_total} total={tot:.2f}s "
            f"per_newton={tot / max(res.n_iter, 1):.2f}s  du={du:.1e} dp={dp:.1e}"
        )
        for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
            print(
                f"  {k:18s} {v:8.2f}s {100 * v / tot:5.1f}%  calls={C[k]:5d}  per={1e3 * v / C[k]:7.1f}ms"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    refines = [int(a) for a in sys.argv[1:]] or [1, 2, 4]
    print(f"cores={os.cpu_count()} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    for r in refines:
        run(r)
