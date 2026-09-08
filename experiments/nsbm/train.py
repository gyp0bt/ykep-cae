"""学習の入口.

    # 場だけ（MSE）
    python experiments/nsbm/train.py --out experiments/nsbm/runs/unet-a --epochs 200 2>&1 | tee experiments/nsbm/logs/train-$(date +%s).log
    # 場 + cfl_init、残差損失で微調整（unet-a から引き継ぎ）
    ~/.claude/hooks/memcap -m 24G -- python experiments/nsbm/train.py --out experiments/nsbm/runs/unet-r \
        --init-from experiments/nsbm/runs/unet-a/best.pt --res-weight 1e-4 --res-workers 16 --epochs 30 --lr 3e-4 \
        2>&1 | tee experiments/nsbm/logs/train-unet-r-$(date +%s).log
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(
        _k, "1"
    )  # 残差損失のワーカー（spawn 子）に継承させる。torch のスレッド数は --threads

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

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
    ap.add_argument(
        "--init-from", type=Path, default=None, help="引き継ぐ best.pt（widths はそちらに従う）"
    )
    ap.add_argument("--res-weight", type=float, default=0.0, help="残差損失の重み λ（0 で無効）")
    ap.add_argument("--res-steps", type=int, default=5, help="残差損失で展開する Newton 歩数 K")
    ap.add_argument("--res-transform", type=str, default="ratio", choices=("ratio", "log"))
    ap.add_argument("--res-workers", type=int, default=8, help="残差損失のワーカープール")
    ap.add_argument(
        "--res-frac", type=float, default=1.0, help="各バッチで残差損失を評価するサンプルの割合"
    )
    ap.add_argument(
        "--split-from",
        type=Path,
        default=None,
        help="split.json を流用する run（既定: --init-from の run）",
    )
    args = ap.parse_args()

    import json

    from nsbm.dataset import load_shards
    from nsbm.train import seeds_to_split, split_by_family, train

    samples = load_shards(args.data)
    n_conv = sum(s.converged for s in samples)
    print(f"samples={len(samples)} converged={n_conv} ({n_conv / len(samples):.1%})", flush=True)
    split_src = args.split_from or (args.init_from.parent if args.init_from else None)
    if split_src is not None and (split_src / "split.json").exists():
        split = seeds_to_split(samples, json.loads((split_src / "split.json").read_text()))
        print(f"split from {split_src / 'split.json'}", flush=True)
    else:
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
        init_from=args.init_from,
        res_weight=args.res_weight,
        res_steps=args.res_steps,
        res_transform=args.res_transform,
        res_workers=args.res_workers,
        res_frac=args.res_frac,
        log=lambda m: print(m, flush=True),
    )
    print(f"best epoch {res.best_epoch} val {res.best_val:.3e} -> {res.best_path}", flush=True)


if __name__ == "__main__":
    main()
