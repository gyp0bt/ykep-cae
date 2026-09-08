"""POD の n-width: 収束解（正規化 y）の train 集合で SVD、val/test のファミリ別射影誤差を m 本で測る."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from nsbm.dataset import load_shards  # noqa: E402
from nsbm.train import seeds_to_split  # noqa: E402


def main():
    samples = load_shards("experiments/nsbm/data")
    split = seeds_to_split(
        samples, json.loads(open("experiments/nsbm/runs/unet-a/split.json").read())
    )
    Y = np.stack([samples[i].y.reshape(-1) for i in split["train"]]).astype(np.float64)  # (N, 3*n)
    mean = Y.mean(0)
    U, S, Vt = np.linalg.svd(Y - mean, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    print(
        "energy captured m=10,20,44,88,176:",
        [f"{energy[m - 1]:.4f}" for m in (10, 20, 44, 88, 176)],
    )
    test = split["test"]
    fams = sorted({samples[i].theta.family for i in test})
    for m in (20, 44, 88, 176):
        V = Vt[:m].T
        print(
            f"--- m={m}: 相対射影誤差 |y - P y|/|y - mean| のファミリ別中央値 / q90（マスク無し）"
        )
        row = []
        for f in fams:
            errs = []
            for i in test:
                if samples[i].theta.family != f:
                    continue
                y = samples[i].y.reshape(-1).astype(np.float64) - mean
                c = V.T @ y
                errs.append(np.linalg.norm(y - V @ c) / max(np.linalg.norm(y), 1e-12))
            errs = np.array(errs)
            row.append(f"{f}:{np.median(errs):.3f}/{np.percentile(errs, 90):.3f}")
        print("  " + "  ".join(row))
    np.savez("experiments/nsbm/results/pod_basis.npz", mean=mean, Vt=Vt[:176], S=S[:176])


if __name__ == "__main__":
    main()
