"""早期やり直し規則の評価用: test θ を Stokes 発進 × cfl_init で解き、定常残差の履歴（相対値）を JSON に残す.

    nohup ~/.claude/hooks/memcap -m 8G -- python experiments/nsbm/cfl_histories.py --cfls 4,8 --workers 4 \
        > experiments/nsbm/logs/cfl-hist-$(date +%s).log 2>&1 &
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def _job(args):
    theta_d, cfl, max_iter = args
    from nsb.core import NSBSettings
    from nsb.solver import solve_steady
    from nsbm.families import Theta, build_input

    theta = Theta.from_dict(theta_d)
    res = solve_steady(
        build_input(theta, NSBSettings(newton_max_iter=max_iter, cfl_init=cfl)), log=None
    )
    return {
        "seed": theta.seed,
        "family": theta.family,
        "cfl_init": cfl,
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        "hist": [float(h / res.residual_ref) for h in res.steady_residual_history],
        "cfl_hist": [float(c) for c in res.cfl_history],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", type=Path, default=HERE / "runs" / "unet-a" / "split.json")
    ap.add_argument("--cfls", type=str, default="4,8")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "cfl_histories.json")
    args = ap.parse_args()
    from nsbm.dataset import load_shards
    from nsbm.train import seeds_to_split

    samples = load_shards(args.data)
    split = seeds_to_split(samples, json.loads(args.split.read_text()))
    cfls = [float(c) for c in args.cfls.split(",")]
    jobs = [(samples[i].theta.to_dict(), c, args.max_iter) for i in split["test"] for c in cfls]
    print(f"jobs={len(jobs)}", flush=True)
    t0 = time.perf_counter()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (n + 1) % 100 == 0:
                print(f"[{n + 1}/{len(jobs)}] {time.perf_counter() - t0:.0f}s", flush=True)
    args.out.write_text(json.dumps(rows))
    print(f"done -> {args.out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
