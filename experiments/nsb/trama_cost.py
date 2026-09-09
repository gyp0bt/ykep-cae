"""nsb の 1 ステップの費用を部品ごとに測る（OpenFOAM との速度差の内訳）.

非定常 1 ステップ = Newton 数回 × (残差評価 + 状態量 + 線形解)、線形解 = 前処理の組み立て
（LU 分解 or SIMPLE 型ブロック）+ GMRES 反復 × (前処理適用 + 差分 matvec)。
どれが効いているかを、同じ trama ケースの格子を変えて測る。

格子を粗くするのは「閉塞セルを外したら何が起きるか」の代理。Δx 3 mm（23400 セル）は
流路だけを解いたときのセル数（24054）とほぼ同じで、行列の疎パターンも同じなので、
分解の費用が自由度に対してどれだけ超線形かがそのまま出る。

    python experiments/nsb/trama_cost.py 2>&1 | tee experiments/nsb/logs/trama-cost-$(date +%s).log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import load_trama, make_trama_input  # noqa: E402
from trama_lin import first_step  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402

from nsb.core import NSBSettings  # noqa: E402
from nsb.krylov import fgmres  # noqa: E402
from nsb.linalg import PardisoLU  # noqa: E402
from nsb.precond import SimpleBlockPreconditioner  # noqa: E402


def timeit(fn, repeat: int = 3) -> tuple[float, object]:
    """最小時間と最後の戻り値."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def measure(dx_mm: float, mass: float, dt: float, pattern: Path) -> dict:
    geo = load_trama(pattern)
    inp = make_trama_input(geo, mass, dx_mm=dx_mm, settings=NSBSettings(linear_solver="jfnk"))
    fs = first_step(inp)
    n = fs.n
    print(f"\n=== Δx {dx_mm} mm: {inp.nx}×{inp.ny} = {n} セル, 自由度 {3 * n} ===", flush=True)

    # 非定常 1 ステップの τ = ρV/Δt（CFL 由来ではなく物理時間刻み）
    tau = inp.rho * np.full(n, fs.disc.vol) / dt
    diag_aug = np.concatenate([tau, tau, np.zeros(n)])
    A = (fs.J1 + sparse.diags(diag_aug)).tocsr()
    rhs = fs.rhs
    out: dict[str, float] = {"nx": inp.nx, "ny": inp.ny, "n_cells": n, "ndof": 3 * n}
    out["nnz"] = int(A.nnz)

    t_res, _ = timeit(lambda: fs.steady_resid(fs.x), 5)
    out["residual_s"] = t_res
    t_state, _ = timeit(lambda: fs.disc.compute_state(fs.x, inp.settings.convection, 0.3), 3)
    out["state_s"] = t_state
    t_j1, _ = timeit(lambda: fs.disc.jacobian_first_order(fs.st, x=fs.x), 3)
    out["assemble_J1_s"] = t_j1

    lu = PardisoLU()
    t_fac, _ = timeit(lambda: lu.factorize(A), 3)
    out["lu_factorize_s"] = t_fac
    t_sol, _ = timeit(lambda: lu.solve(rhs), 5)
    out["lu_solve_s"] = t_sol

    # GMRES（差分 matvec、LU 前処理）
    def mv(v):
        return fs.fd_matvec(v) + diag_aug * v

    t0 = time.perf_counter()
    _sol, n_it, ok = fgmres(mv, rhs, lu.solve, rtol=1e-2, restart=50, maxiter=4)
    out["gmres_lu_s"] = time.perf_counter() - t0
    out["gmres_lu_iters"] = int(n_it)
    out["gmres_lu_ok"] = int(ok)
    lu.free()

    # SIMPLE 型ブロック前処理（O(N)）
    pc = SimpleBlockPreconditioner(n)
    t_pc, _ = timeit(lambda: pc.factorize(A), 2)
    out["simple_build_s"] = t_pc
    t_app, _ = timeit(lambda: pc.solve(rhs), 5)
    out["simple_apply_s"] = t_app
    t0 = time.perf_counter()
    _sol2, n_it2, ok2 = fgmres(mv, rhs, pc.solve, rtol=1e-2, restart=50, maxiter=4)
    out["gmres_simple_s"] = time.perf_counter() - t0
    out["gmres_simple_iters"] = int(n_it2)
    out["gmres_simple_ok"] = int(ok2)
    pc.free()

    for k, v in out.items():
        if k.endswith("_s"):
            print(f"  {k:20s} {v * 1e3:9.1f} ms")
        else:
            print(f"  {k:20s} {v:9d}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", nargs="?", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--dx", nargs="*", type=float, default=[1.5, 3.0])
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_cost.json")
    a = ap.parse_args()
    rows = [measure(dx, a.mass, a.dt, a.pattern) for dx in a.dx]
    Path(a.out_json).write_text(json.dumps(rows, indent=1) + "\n")
    if len(rows) == 2:
        fine, coarse = rows[0], rows[1]
        print("\n=== 細 / 粗 の比（自由度比 %.2f）===" % (fine["ndof"] / coarse["ndof"]))
        for k in fine:
            if k.endswith("_s"):
                print(f"  {k:20s} {fine[k] / coarse[k]:6.2f}×")
    print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
