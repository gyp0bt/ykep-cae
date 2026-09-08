"""スパイク: kNN 局所基底の Galerkin 初期解 vs Stokes 発進の Newton 反復数（テスト N 件）."""

import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402


def job(args):
    seed_idx, theta_d, x_img, nbrs, k = args
    from nsb.core import NSBSettings
    from nsbm.evaluate import run_with_init
    from nsbm.families import Theta, build_input
    from nsbm.features import blocked_mask_from_x
    from nsbm.galerkin import galerkin_init

    theta = Theta.from_dict(theta_d)
    st = NSBSettings(newton_max_iter=200)
    inp = build_input(theta, st)
    t0 = time.perf_counter()
    init, info = galerkin_init(inp, nbrs[:k], blocked_mask_from_x(x_img), steps=6)
    t_g = time.perf_counter() - t0
    r_st = run_with_init(theta, None, st)
    r_ga = run_with_init(theta, init, st)
    return seed_idx, {
        "stokes": r_st,
        "galerkin": r_ga,
        "r_ratio": info["r_final"] / info["r_stokes"],
        "n_gn": len(info["r_norms"]) - 1,
        "t_galerkin": t_g,
        "family": theta.family,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from nsbm.dataset import load_shards
    from nsbm.evaluate import channel_stats
    from nsbm.features import denormalize_y
    from nsbm.train import seeds_to_split

    samples = load_shards("experiments/nsbm/data")
    split = seeds_to_split(
        samples, json.loads(open("experiments/nsbm/runs/unet-a/split.json").read())
    )
    x_tr = np.stack([samples[i].x for i in split["train"]])
    tr_idx = split["train"]
    mean, std = channel_stats(x_tr)
    A = ((x_tr - mean) / std).reshape(len(x_tr), -1)
    rng = np.random.default_rng(0)
    test = rng.choice(split["test"], a.n, replace=False)
    jobs = []
    for i in test:
        s = samples[i]
        b = ((s.x[None] - mean) / std).reshape(1, -1)
        d = np.sqrt(((A - b) ** 2).sum(1))
        nn = np.argsort(d)[: a.k]
        # 近傍解をこの θ のスケールに直す（正規化 y は u/u_in, p/p_ref なので denormalize_y(theta_test, y_nn)）
        nbrs = [denormalize_y(s.theta, samples[tr_idx[j]].y) for j in nn]
        jobs.append((int(i), s.theta.to_dict(), s.x, nbrs, a.k))
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for m, (_i, r) in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(r)
            print(
                f"[{m + 1}/{len(jobs)}] {r['family']:10s} stokes {r['stokes']['n_iter']:3d} galerkin {r['galerkin']['n_iter']:3d} r_ratio {r['r_ratio']:.3f} gn {r['n_gn']} r0 {r['galerkin']['r0_ratio']:.3f} cfl0 {r['galerkin']['cfl0']:.3g} t {r['t_galerkin']:.1f}s",
                flush=True,
            )
    st = np.array([r["stokes"]["n_iter"] for r in rows])
    ga = np.array([r["galerkin"]["n_iter"] for r in rows])
    print(
        f"N={len(rows)} stokes median {np.median(st)} galerkin median {np.median(ga)} wins {(ga < st).sum()} ties {(ga == st).sum()} losses {(ga > st).sum()} r_ratio median {np.median([r['r_ratio'] for r in rows]):.3f}"
    )
    print(
        f"converged stokes {sum(r['stokes']['converged'] for r in rows)} galerkin {sum(r['galerkin']['converged'] for r in rows)}"
    )
    json.dump(rows, open(f"experiments/nsbm/results/galerkin-knn-k{a.k}.json", "w"))


if __name__ == "__main__":
    main()
