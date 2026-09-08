"""評価の入口: テスト集合で Stokes / kNN / UNet の Newton 反復数を比較し YAML と CSV に書く.

python experiments/nsbm/eval.py --run experiments/nsbm/runs/unet-a 2>&1 | tee experiments/nsbm/logs/eval-$(date +%s).log
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--run", type=Path, default=HERE / "runs" / "unet-a")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="テスト件数の上限（0 で全件）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml

    from nsb.core import NSBSettings
    from nsbm.dataset import load_shards
    from nsbm.evaluate import evaluate, summarize
    from nsbm.train import load_model, seeds_to_split

    torch.set_num_threads(4)
    samples = load_shards(args.data)
    split = seeds_to_split(samples, json.loads((args.run / "split.json").read_text()))
    if args.limit:
        split["test"] = split["test"][: args.limit]
    net = load_model(args.run / "best.pt")

    def predict(s):
        with torch.no_grad():
            return net(torch.from_numpy(s.x[None])).numpy()[0]

    print(f"test={len(split['test'])} train={len(split['train'])} run={args.run}", flush=True)
    rows = evaluate(
        predict,
        samples,
        split,
        k=args.k,
        n_workers=args.workers,
        settings=NSBSettings(newton_max_iter=args.max_iter),
        log=lambda m: print(m, flush=True),
    )
    summary = summarize(rows)
    out = args.out or (HERE / "results" / f"eval-{args.run.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    )
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(yaml.safe_dump(summary["all"], sort_keys=False), flush=True)
    print(f"-> {out.with_suffix('.yaml')}", flush=True)
    _ = np


if __name__ == "__main__":
    main()
