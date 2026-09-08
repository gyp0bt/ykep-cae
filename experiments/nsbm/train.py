"""学習の入口.

    python experiments/nsbm/train.py --data experiments/nsbm/data --out experiments/nsbm/runs/unet-a --epochs 200 \
        2>&1 | tee experiments/nsbm/logs/train-$(date +%s).log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "unet-a")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", type=str, default="32,64,128,256")
    ap.add_argument(
        "--div-weight", type=float, default=0.0, help="離散連続式ペナルティの重み（0 で MSE のみ）"
    )
    args = ap.parse_args()

    from nsbm.dataset import load_shards
    from nsbm.train import split_by_family, train

    samples = load_shards(args.data)
    n_conv = sum(s.converged for s in samples)
    print(f"samples={len(samples)} converged={n_conv} ({n_conv / len(samples):.1%})", flush=True)
    split = split_by_family(samples, args.seed)
    print({k: len(v) for k, v in split.items()}, flush=True)
    res = train(
        samples,
        args.out,
        split=split,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        widths=tuple(int(w) for w in args.widths.split(",")),
        seed=args.seed,
        threads=args.threads,
        div_weight=args.div_weight,
        log=lambda m: print(m, flush=True),
    )
    print(f"best epoch {res.best_epoch} val {res.best_val:.3e} -> {res.best_path}", flush=True)


if __name__ == "__main__":
    main()
