"""データ生成の入口: 空きコア分のワーカーで θ を解いて npz シャードへ.

    nohup ~/.claude/hooks/memcap -- python experiments/nsbm/gen.py --n 2000 --workers 16 \
        2>&1 | tee experiments/nsbm/logs/gen-$(date +%s).log &
"""

from __future__ import annotations

import os

# ワーカーは 1 スレッド（numpy / MKL / numba の import 前に決める。spawn した子に継承される）
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--shard", type=int, default=256)
    ap.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="生成時の Newton 反復上限（既定 80 より緩め、遅いケースも解に到達させる）",
    )
    args = ap.parse_args()

    from nsb.core import NSBSettings
    from nsbm.dataset import generate

    print(
        f"generate n={args.n} seed0={args.seed0} workers={args.workers} out={args.out}", flush=True
    )
    paths = generate(
        range(args.seed0, args.seed0 + args.n),
        args.out,
        args.workers,
        args.shard,
        log=lambda m: print(m, flush=True),
        settings=NSBSettings(newton_max_iter=args.max_iter),
    )
    print(f"done: {len(paths)} shards -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
