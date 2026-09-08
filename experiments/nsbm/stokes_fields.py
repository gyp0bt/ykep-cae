"""床（Stokes–Brinkman 解）の生成: 全 θ について nsb の参照場を 1 回の線形解で作り、正規化して npz に保存する.

    nohup ~/.claude/hooks/memcap -m 8G -- python experiments/nsbm/stokes_fields.py --workers 4 \
        > experiments/nsbm/logs/stokes-fields-$(date +%s).log 2>&1 &

出力 data/stokes.npz: seed (N,), ys (N, 3, 72, 48) float32 = (u_S/u_in, v_S/u_in, p_S/p_ref)。`nsbm.floor.load_stokes` が読む。
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def _job(theta_d):
    import numpy as np

    from nsbm.families import Theta, build_input
    from nsbm.features import normalize_y
    from nsbm.galerkin import stokes_field

    theta = Theta.from_dict(theta_d)
    inp = build_input(theta)
    x, disc = stokes_field(inp)
    u, v, p = disc.split(x)
    shape = (inp.nx, inp.ny)
    return theta.seed, normalize_y(
        theta, u.reshape(shape), v.reshape(shape), p.reshape(shape)
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import numpy as np

    from nsbm.dataset import load_shards

    samples = load_shards(args.data)
    jobs = [s.theta.to_dict() for s in samples]
    t0 = time.perf_counter()
    seeds, ys = [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, (sd, y) in enumerate(pool.imap(_job, jobs, chunksize=8)):
            seeds.append(sd)
            ys.append(y)
            if (n + 1) % 500 == 0:
                print(f"[{n + 1}/{len(jobs)}] {time.perf_counter() - t0:.0f}s", flush=True)
    out = args.data / "stokes.npz"
    np.savez(out, seed=np.array(seeds), ys=np.stack(ys))
    print(f"done -> {out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
