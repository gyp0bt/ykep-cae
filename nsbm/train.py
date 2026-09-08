"""学習ループ（CPU torch）: 収束したサンプルだけをファミリ層化で train/val/test に分け、UNet を MSE で学習する.

[分割] θ 単位。ファミリごとに seed を並べ替えて 80/10/10。未収束サンプルは学習・検証に使わない
  （評価側では別枠で数える）。分割は `split.json` に seed のリストとして保存し、評価が同じ分割を読む。
[損失] 正規化した (u, v, p) の MSE、3 チャネル等重み。
[保存] val 最良の `best.pt`（state_dict, widths, epoch, val_loss）と `history.csv`。
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from nsbm.dataset import Sample
from nsbm.model import UNet

LogFn = Callable[[str], None]


def split_by_family(
    samples: Sequence[Sample], seed: int = 0, frac: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[int]]:
    """収束サンプルの index をファミリ層化で train/val/test に分ける."""
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[int]] = {}
    for k, s in enumerate(samples):
        if s.converged:
            by_family.setdefault(s.theta.family, []).append(k)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for fam in sorted(by_family):
        idx = np.array(by_family[fam])
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        out["train"] += idx[:n_tr].tolist()
        out["val"] += idx[n_tr : n_tr + n_va].tolist()
        out["test"] += idx[n_tr + n_va :].tolist()
    return out


def split_to_seeds(samples: Sequence[Sample], split: dict[str, list[int]]) -> dict[str, list[int]]:
    return {k: [samples[i].theta.seed for i in v] for k, v in split.items()}


def seeds_to_split(samples: Sequence[Sample], seeds: dict[str, list[int]]) -> dict[str, list[int]]:
    pos = {s.theta.seed: k for k, s in enumerate(samples)}
    return {k: [pos[sd] for sd in v if sd in pos] for k, v in seeds.items()}


def divergence_loss(x: Tensor, yhat: Tensor) -> Tensor:
    """離散連続式のペナルティ: 正規化速度 (u/u_in, v/u_in) と h/h0 = exp(x0) から面流束 h_f u_f の発散を作り 2 乗平均.

    nsb の p 方程式（質量保存）の残差は予測速度の発散で決まる。面値は隣接セル平均（Rhie–Chow なし）で近似する。
    無次元化: 発散 × (dx·dy)/(h0 u_in) 相当になるよう、面流束を h/h0·u/u_in、格子幅 dx = LX/NX, dy = LY/NY で組む。
    """
    from nsbm.families import DX, DY

    h = torch.exp(x[:, 0:1])
    u, v = yhat[:, 0:1], yhat[:, 1:2]
    fx = h * u  # (B,1,NX,NY)
    fy = h * v
    fe = 0.5 * (fx[:, :, 1:, :] + fx[:, :, :-1, :])  # x 方向内部面 (NX-1, NY)
    fn = 0.5 * (fy[:, :, :, 1:] + fy[:, :, :, :-1])  # y 方向内部面 (NX, NY-1)
    div = torch.zeros_like(u)
    div[:, :, 1:, :] += fe * DY
    div[:, :, :-1, :] -= fe * DY
    div[:, :, :, 1:] += fn * DX
    div[:, :, :, :-1] -= fn * DX
    # 境界面（inlet/outlet/wall）は流束不明なので内部面だけで評価し、境界セルは除く
    inner = div[:, :, 1:-1, 1:-1] / (DX * DY)
    return (inner**2).mean() * (DX * DY)


def to_tensors(samples: Sequence[Sample], idx: Sequence[int]) -> tuple[Tensor, Tensor]:
    x = torch.from_numpy(np.stack([samples[i].x for i in idx]))
    y = torch.from_numpy(np.stack([samples[i].y for i in idx]))
    return x, y


@dataclass
class TrainResult:
    best_path: Path
    best_epoch: int
    best_val: float
    history: list[tuple[int, float, float, float]]


def load_model(path: Path) -> UNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = UNet(widths=tuple(ck["widths"]))
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def train(
    samples: Sequence[Sample],
    out_dir: Path,
    split: dict[str, list[int]] | None = None,
    epochs: int = 200,
    batch: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    widths: tuple[int, ...] = (32, 64, 128, 256),
    seed: int = 0,
    threads: int | None = None,
    div_weight: float = 0.0,
    log: LogFn | None = print,
) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    split = split or split_by_family(samples, seed)
    (out_dir / "split.json").write_text(json.dumps(split_to_seeds(samples, split)))
    x_tr, y_tr = to_tensors(samples, split["train"])
    x_va, y_va = to_tensors(samples, split["val"])
    net = UNet(widths=widths)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    best_val, best_epoch, history = float("inf"), -1, []
    best_path = out_dir / "best.pt"
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        tot = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k : k + batch]
            yhat = net(x_tr[b])
            loss = torch.nn.functional.mse_loss(yhat, y_tr[b])
            if div_weight > 0:
                loss = loss + div_weight * divergence_loss(x_tr[b], yhat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        tr = tot / len(perm)
        net.eval()
        with torch.no_grad():
            va = float(
                sum(
                    torch.nn.functional.mse_loss(
                        net(x_va[k : k + 128]), y_va[k : k + 128], reduction="sum"
                    )
                    for k in range(0, len(x_va), 128)
                )
                / y_va.numel()
            )
        history.append((ep, tr, va, time.perf_counter() - t0))
        if va < best_val:
            best_val, best_epoch = va, ep
            torch.save(
                {"state_dict": net.state_dict(), "widths": widths, "epoch": ep, "val_loss": va},
                best_path,
            )
        if log is not None:
            log(
                f"epoch {ep:4d} train {tr:.3e} val {va:.3e} best {best_val:.3e}@{best_epoch} lr {sched.get_last_lr()[0]:.2e} {time.perf_counter() - t0:7.1f}s"
            )
    with (out_dir / "history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse", "elapsed"])
        w.writerows(history)
    return TrainResult(best_path, best_epoch, best_val, history)
