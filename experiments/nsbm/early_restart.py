"""早期やり直し規則の評価: 高い cfl_init で出発し、k 反復後の相対残差が閾値を超えていたら 0.25 でやり直す.

    python experiments/nsbm/early_restart.py 2>&1 | tee experiments/nsbm/logs/early-restart-$(date +%s).log

[入力] results/cfl_histories.json（test θ × cfl 4/8/16 の定常残差履歴 |R_k|/|R_ref|）と、
  0.25 の反復数（データセットの Sample.n_iter、split.json の test）。
[規則] 反復 k で hist[k] > τ なら打ち切って 0.25 でやり直す。費用 = k + n_iter(0.25)。
  打ち切らなければ費用 = n_iter（収束）または max_iter + n_iter(0.25)（未収束）。
[出力] (cfl, k, τ) の格子で平均・中央値・q90・未収束数。最良の規則と、固定 0.25・固定 cfl・オラクルの比較。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", type=Path, default=HERE / "results" / "cfl_histories.json")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", type=Path, default=HERE / "runs" / "unet-a" / "split.json")
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "early_restart.yaml")
    args = ap.parse_args()
    from nsbm.dataset import load_shards
    from nsbm.train import seeds_to_split

    samples = load_shards(args.data)
    split = seeds_to_split(samples, json.loads(args.split.read_text()))
    n025 = {samples[i].theta.seed: samples[i].n_iter for i in split["test"]}
    rows = json.load(args.hist.open())
    cfls = sorted({r["cfl_init"] for r in rows})
    by = {c: [r for r in rows if r["cfl_init"] == c] for c in cfls}
    out = {
        "n_test": len(n025),
        "fixed_0.25": {
            "mean": float(np.mean(list(n025.values()))),
            "median": float(np.median(list(n025.values()))),
        },
    }
    print(
        f"fixed 0.25: mean {out['fixed_0.25']['mean']:.2f} median {out['fixed_0.25']['median']:.1f}",
        flush=True,
    )

    def cost_of(r, k, tau):
        h = r["hist"]
        if k < len(h) and h[k] > tau:  # k 反復後に打ち切り
            return k + n025[r["seed"]], "restart"
        if r["converged"]:
            return r["n_iter"], "ok"
        return args.max_iter + n025[r["seed"]], "fail"

    grid = {}
    best = None
    for c in cfls:
        rs = by[c]
        base = np.array(
            [args.max_iter + n025[r["seed"]] if not r["converged"] else r["n_iter"] for r in rs]
        )
        grid[f"fixed_{c:g}"] = {
            "mean": float(base.mean()),
            "median": float(np.median(base)),
            "fails": int(sum(not r["converged"] for r in rs)),
        }
        print(
            f"fixed {c:g}: mean {base.mean():.2f} median {np.median(base):.1f} fails {grid[f'fixed_{c:g}']['fails']}",
            flush=True,
        )
        for k in (2, 3, 4, 5, 6, 8, 10):
            for tau in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
                cs = [cost_of(r, k, tau) for r in rs]
                cost = np.array([x[0] for x in cs])
                n_restart = sum(x[1] == "restart" for x in cs)
                n_fail = sum(x[1] == "fail" for x in cs)
                d = {
                    "mean": float(cost.mean()),
                    "median": float(np.median(cost)),
                    "q90": float(np.percentile(cost, 90)),
                    "restarts": n_restart,
                    "fails": n_fail,
                }
                grid[f"cfl{c:g}_k{k}_tau{tau:g}"] = d
                if best is None or d["mean"] < best[1]["mean"]:
                    best = (f"cfl{c:g}_k{k}_tau{tau:g}", d)
        # 失敗ケースは k 反復後にどこにいるか（見切れるか）
        fails = [r for r in rs if not r["converged"]]
        oks = [r for r in rs if r["converged"]]
        for k in (3, 5, 8):
            f_h = [r["hist"][k] for r in fails if k < len(r["hist"])]
            o_h = [r["hist"][k] for r in oks if k < len(r["hist"])]
            if f_h and o_h:
                print(
                    f"  cfl {c:g} k={k}: 失敗 {len(f_h)} 件の残差比 中央値 {np.median(f_h):.2g} min {np.min(f_h):.2g} | "
                    f"収束 {len(o_h)} 件（まだ走っている）の中央値 {np.median(o_h):.2g} q90 {np.percentile(o_h, 90):.2g}",
                    flush=True,
                )
    out["best_rule"] = {"name": best[0], **best[1]}
    out["grid"] = grid
    print(f"best rule {best[0]}: {best[1]}", flush=True)
    # 上位 8 規則
    top = sorted(
        ((k, v) for k, v in grid.items() if k.startswith("cfl")), key=lambda kv: kv[1]["mean"]
    )[:8]
    for k, v in top:
        print(
            f"  {k:22s} mean {v['mean']:6.2f} median {v['median']:5.1f} q90 {v['q90']:5.1f} restarts {v['restarts']:3d} fails {v['fails']}",
            flush=True,
        )
    args.out.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
