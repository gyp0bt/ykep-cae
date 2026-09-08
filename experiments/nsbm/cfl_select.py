"""cfl_init 選択器: θ の特徴から「各 cfl_init で収束するか」を予測し、収束すると見込める最大の cfl_init を選ぶ.

    python experiments/nsbm/cfl_select.py 2>&1 | tee experiments/nsbm/logs/cfl-select-$(date +%s).log

[入力] results/cfl_labels.csv（`cfl_labels.py`: Stokes 発進 × cfl_init ∈ {1,2,4,8,16} の反復数・収束）と
  データセットの cfl_init 0.25 の反復数（Sample.n_iter）。分割は runs/unet-a/split.json（θ 単位）。
[特徴] log u_in, log h0, log Re_h = ρ u_in h0/μ, log Re_L = ρ u_in LX/μ, ファミリ one-hot, inlet/outlet 壁 one-hot,
  同一壁フラグ, ポート幅 / 壁長, 閉塞セル率, 開口部の h の幾何平均と分散（log h/h0）。
[モデル] 小さな MLP（特徴 → 5 つの logit = 各 cfl_init で収束する確率）、BCE。
[方策] val で閾値 τ を選び（期待反復数最小）、test で P(収束|cfl) > τ の最大 cfl を採用（無ければ 0.25）。
  比較: 固定 cfl_init（0.25, 1, 2, 4, 8, 16）、オラクル（θ ごとに収束した最少反復の cfl）。
  失敗の費用は「上限まで回して 0.25 でやり直す」= max_iter + n_iter(0.25) とする。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

FAMS = ("uniform", "sin2d", "quad", "grf", "uturn", "serpentine", "pins", "blobs")
WALLS = ("west", "east", "south", "north")
RHO, MU = 1000.0, 1.0e-3


def features(sample) -> np.ndarray:
    from nsbm.families import LX, LY, WALL_LENGTH
    from nsbm.features import blocked_mask_from_x

    th = sample.theta
    x = sample.x
    blocked = blocked_mask_from_x(x)
    open_logh = x[0][~blocked]
    f = [
        np.log(th.u_in),
        np.log(th.h0 / 1e-3),
        np.log(RHO * th.u_in * th.h0 / MU),
        np.log(RHO * th.u_in * LX / MU),
        float(th.inlet.wall == th.outlet.wall),
        (th.inlet.s1 - th.inlet.s0) / WALL_LENGTH[th.inlet.wall],
        (th.outlet.s1 - th.outlet.s0) / WALL_LENGTH[th.outlet.wall],
        float(blocked.mean()),
        float(open_logh.mean()) if open_logh.size else 0.0,
        float(open_logh.std()) if open_logh.size else 0.0,
        LY / LX,
    ]
    f += [float(th.family == g) for g in FAMS]
    f += [float(th.inlet.wall == w) for w in WALLS]
    f += [float(th.outlet.wall == w) for w in WALLS]
    return np.array(f, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=HERE / "results" / "cfl_labels.csv")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", type=Path, default=HERE / "runs" / "unet-a" / "split.json")
    ap.add_argument(
        "--max-iter", type=int, default=120, help="ラベル生成時の反復上限（失敗の費用）"
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "cfl_select.yaml")
    args = ap.parse_args()

    import torch
    import yaml

    from nsbm.dataset import load_shards

    torch.set_num_threads(2)
    torch.manual_seed(0)
    rows = list(csv.DictReader(args.labels.open()))
    lab: dict[int, dict[float, tuple[int, bool]]] = defaultdict(dict)
    for r in rows:
        lab[int(r["seed"])][float(r["cfl_init"])] = (int(r["n_iter"]), r["converged"] == "True")
    cfls = sorted({float(r["cfl_init"]) for r in rows})
    seeds_full = [s for s, d in lab.items() if len(d) == len(cfls)]
    samples = {s.theta.seed: s for s in load_shards(args.data) if s.converged}
    seeds_full = [s for s in seeds_full if s in samples]
    split_seeds = json.loads(args.split.read_text())
    sets = {k: [s for s in v if s in set(seeds_full)] for k, v in split_seeds.items()}
    print(
        f"labelled θ: {len(seeds_full)}  cfls={cfls}  split sizes: { {k: len(v) for k, v in sets.items()} }",
        flush=True,
    )

    def table(seeds):
        X = np.stack([features(samples[s]) for s in seeds])
        conv = np.array([[lab[s][c][1] for c in cfls] for s in seeds], dtype=np.float32)
        nit = np.array([[lab[s][c][0] for c in cfls] for s in seeds], dtype=np.float32)
        n025 = np.array([samples[s].n_iter for s in seeds], dtype=np.float32)
        return X, conv, nit, n025

    Xtr, Ctr, Ntr, N0tr = table(sets["train"])
    Xva, Cva, Nva, N0va = table(sets["val"])
    Xte, Cte, Nte, N0te = table(sets["test"])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: torch.from_numpy((X - mu) / sd)  # noqa: E731
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, len(cfls)),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, ct = norm(Xtr), torch.from_numpy(Ctr)
    xv, cv = norm(Xva), torch.from_numpy(Cva)
    best, best_state = 1e9, None
    for _ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(perm), 64):
            b = perm[k : k + 64]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]), ct[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = torch.nn.functional.binary_cross_entropy_with_logits(net(xv), cv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    print(f"val BCE best {best:.4f}", flush=True)

    def policy_cost(P, conv, nit, n025, tau):
        """P(収束|cfl) > τ の最大 cfl を選ぶ。費用: 収束なら n_iter、失敗なら max_iter + n_iter(0.25)。"""
        cost = np.empty(len(P))
        chosen = np.empty(len(P))
        for i in range(len(P)):
            ok = np.where(P[i] > tau)[0]
            if ok.size == 0:
                cost[i], chosen[i] = n025[i], 0.25
            else:
                j = ok.max()
                chosen[i] = cfls[j]
                cost[i] = nit[i, j] if conv[i, j] else args.max_iter + n025[i]
        return cost, chosen

    with torch.no_grad():
        Pva = torch.sigmoid(net(xv)).numpy()
        Pte = torch.sigmoid(net(norm(Xte))).numpy()
    taus = np.linspace(0.3, 0.99, 70)
    tau = taus[int(np.argmin([policy_cost(Pva, Cva, Nva, N0va, t)[0].mean() for t in taus]))]
    res = {
        "n_test": int(len(Xte)),
        "cfls": cfls,
        "tau": float(tau),
        "val_bce": float(best),
        "policies": {},
    }

    def summarize(name, cost, chosen=None, fails=None):
        d = {
            "mean": float(cost.mean()),
            "median": float(np.median(cost)),
            "q90": float(np.percentile(cost, 90)),
        }
        if fails is not None:
            d["fails"] = int(fails)
        if chosen is not None:
            d["chosen_hist"] = {str(c): int((chosen == c).sum()) for c in [0.25] + cfls}
        res["policies"][name] = d
        print(
            f"{name:14s} mean {d['mean']:6.2f} median {d['median']:5.1f} q90 {d['q90']:5.1f}"
            + (f" fails {fails}" if fails is not None else "")
            + (f" chosen {d['chosen_hist']}" if chosen is not None else ""),
            flush=True,
        )

    summarize("fixed 0.25", N0te, fails=0)
    for j, c in enumerate(cfls):
        cost = np.where(Cte[:, j] > 0, Nte[:, j], args.max_iter + N0te)
        summarize(f"fixed {c:g}", cost, fails=int((Cte[:, j] == 0).sum()))
    # オラクル: 収束した中で最少反復（0.25 を含む）
    allN = np.concatenate([N0te[:, None], np.where(Cte > 0, Nte, np.inf)], axis=1)
    summarize("oracle", allN.min(1))
    cost, chosen = policy_cost(Pte, Cte, Nte, N0te, tau)
    fails = sum(
        1 for i in range(len(Pte)) if chosen[i] != 0.25 and not Cte[i, cfls.index(chosen[i])]
    )
    summarize("selector", cost, chosen, fails)
    # 選択器の精度: 各 cfl の収束予測 AUC 相当（正例率と閾値 0.5 の正答率）
    acc = ((Pte > 0.5) == (Cte > 0)).mean(0)
    res["per_cfl_accuracy"] = {str(c): float(a) for c, a in zip(cfls, acc, strict=True)}
    res["per_cfl_converged_rate_test"] = {
        str(c): float(Cte[:, j].mean()) for j, c in enumerate(cfls)
    }
    print(
        "per-cfl accuracy",
        res["per_cfl_accuracy"],
        "converged rate",
        res["per_cfl_converged_rate_test"],
        flush=True,
    )
    args.out.write_text(yaml.safe_dump(res, sort_keys=False))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
