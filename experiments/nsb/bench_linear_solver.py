"""同じ 1 次風上ヤコビアン J1 + 擬似時間対角で scipy splu(COLAMD) と pypardiso の分解・解法時間を比較する.

    pip install pypardiso   # MKL が見つからない環境では PYPARDISO_MKL_RT=/path/to/libmkl_rt.so
    python experiments/nsb/bench_linear_solver.py 2>&1 | tee experiments/nsb/logs/bench-linear-solver-$(date +%s).log

status-32 の結果: experiments/nsb/logs/bench-linear-solver-flat-r124.log
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pypardiso
from scipy import sparse
from scipy.sparse import linalg as spla

from nsb import NSBSettings, make_case
from nsb.assembly import BrinkmanDiscretization


def bench(refine: int) -> None:
    s = NSBSettings(velocity_floor=0.1, init_field="stokes", alpha_u=1.0)
    inp = make_case("flat", refine, 1.0, settings=s)
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    x = np.zeros(3 * n)
    st0 = disc.compute_state(x, s.scheme, s.venkat_k)
    J0 = disc.jacobian_first_order(st0, convection=False, x=x).tocsc()
    x = x + spla.splu(J0).solve(-disc.residual_from_state(x, st0, convection=False))
    st = disc.compute_state(x, s.scheme, s.venkat_k)
    tau = (
        inp.rho
        * disc.vol
        / (0.5 * min(disc.dx, disc.dy) / np.maximum(np.hypot(*disc.split(x)[:2]), 0.1))
    ).ravel()
    J = (disc.jacobian_first_order(st, x=x) + sparse.diags(np.r_[tau, tau, np.zeros(n)])).tocsc()
    b = disc.residual_from_state(x, st)
    print(f"\n== refine={refine} n3={3 * n} nnz={J.nnz} ({J.nnz / (3 * n):.1f}/row)")
    for spec in ("COLAMD",):
        t = time.perf_counter()
        lu = spla.splu(J, permc_spec=spec)
        tf = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(10):
            y = lu.solve(b)
        ts = (time.perf_counter() - t) / 10
        print(
            f"  splu[{spec:14s}] factor={tf:7.3f}s solve={1e3 * ts:6.1f}ms fill={lu.L.nnz + lu.U.nnz:,} res={np.linalg.norm(J @ y - b) / np.linalg.norm(b):.1e}"
        )
    Jr = J.tocsr()
    ps = pypardiso.PyPardisoSolver()
    t = time.perf_counter()
    ps.factorize(Jr)
    tf = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(10):
        y = ps.solve(Jr, b)
    ts = (time.perf_counter() - t) / 10
    print(
        f"  pypardiso ({os.cpu_count()} cores)  factor={tf:7.3f}s solve={1e3 * ts:6.1f}ms res={np.linalg.norm(J @ y - b) / np.linalg.norm(b):.1e}"
    )
    ps.free_memory(everything=True)


for r in (1, 2, 4):
    bench(r)
