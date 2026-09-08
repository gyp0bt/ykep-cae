"""cfl_init 選択器のラベル生成: 収束済みの全 θ について Stokes 発進 × cfl_init の掃引で反復数と収束可否を取る.

    nohup ~/.claude/hooks/memcap -m 8G -- python experiments/nsbm/cfl_labels.py --workers 4 \
        > experiments/nsbm/logs/cfl-labels-$(date +%s).log 2>&1 &

出力 results/cfl_labels.csv: seed, family, u_in, h0, inlet, outlet, cfl_init, n_iter, converged, r0_ratio。
θ ごとの最適 cfl_init（収束したうち最少反復）と、それを当てたときの反復数が選択器の上限になる。
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def _job(args):
    theta_d, cfl, max_iter = args
    from nsb.core import NSBSettings
    from nsbm.evaluate import run_with_init
    from nsbm.families import Theta

    theta = Theta.from_dict(theta_d)
    r = run_with_init(theta, None, NSBSettings(newton_max_iter=max_iter, cfl_init=cfl))
    return theta_d, cfl, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "cfl_labels.csv")
    ap.add_argument("--cfls", type=str, default="1,2,4,8,16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from nsbm.dataset import load_shards

    samples = [s for s in load_shards(args.data) if s.converged]
    if args.limit:
        samples = samples[: args.limit]
    cfls = [float(c) for c in args.cfls.split(",")]
    jobs = [(s.theta.to_dict(), c, args.max_iter) for s in samples for c in cfls]
    print(f"thetas={len(samples)} cfls={cfls} jobs={len(jobs)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with (
        args.out.open("w", newline="") as f,
        mp.get_context("spawn").Pool(args.workers) as pool,
    ):
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "family",
                "u_in",
                "h0",
                "inlet",
                "outlet",
                "cfl_init",
                "n_iter",
                "converged",
                "r0_ratio",
            ]
        )
        for n, (td, c, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            w.writerow(
                [
                    td["seed"],
                    td["family"],
                    td["u_in"],
                    td["h0"],
                    td["inlet"]["wall"],
                    td["outlet"]["wall"],
                    c,
                    r["n_iter"],
                    r["converged"],
                    r["r0_ratio"],
                ]
            )
            f.flush()
            if (n + 1) % 100 == 0:
                print(f"[{n + 1}/{len(jobs)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done -> {args.out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
