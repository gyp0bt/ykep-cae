"""学習ループ（CPU torch）: 収束したサンプルだけをファミリ層化で train/val/test に分け、UNet を学習する.

[分割] θ 単位。ファミリごとに seed を並べ替えて 80/10/10。未収束サンプルは学習・検証に使わない
  （評価側では別枠で数える）。分割は `split.json` に seed のリストとして保存し、評価が同じ分割を読む。
[損失] MSE(正規化した (u, v, p)、3 チャネル等重み) + res_weight × 残差損失（`nsbm.residual_loss`:
  予測場を初期解・予測 cfl_init で nsb を res_steps 歩回した定常残差比 Σ_k |R(x_k)|/|R_ref| のバッチ平均）。
  残差損失はワーカープールで並列に評価し、勾配は直通（straight-through）で torch に渡す。
  res_weight=0 なら MSE のみ（cfl ヘッドは学習されず既定 0.25 のまま）。
[選抜] val の目的関数 MSE + res_weight × 残差損失が最小の epoch を `best.pt` に保存
  （state_dict, widths, epoch, val_loss, val_mse, val_res）。`history.csv` に train/val の両項を残す。
[引き継ぎ] init_from に既存の best.pt（場だけの unet-a など）を渡すと重みを読んで微調整から始める。
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from nsbm.dataset import Sample
from nsbm.features import p_ref
from nsbm.floor import floor_input, floor_loss, floor_predict
from nsbm.model import UNet
from nsbm.residual_loss import ResidualLossPool, straight_through

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


def to_physical(x: Tensor, yhat: Tensor, u_in: Tensor, pref: Tensor) -> Tensor:
    """正規化した予測 (B,3,H,W) → 物理量の初期場。閉塞セル（x[0] から復元）の速度は 0（`mask_blocked` と同じ）."""
    scale = torch.stack([u_in, u_in, pref], dim=1)[:, :, None, None]
    fields = yhat * scale
    blocked = (x[:, 0:1] < float(np.log(1.5 / 100.0))).to(fields.dtype)
    keep = torch.cat([1.0 - blocked, 1.0 - blocked, torch.ones_like(blocked)], dim=1)
    return fields * keep


@dataclass
class TrainResult:
    best_path: Path
    best_epoch: int
    best_val: float
    history: list[tuple]


def load_model(path: Path) -> UNet:
    """best.pt を読む。cfl ヘッドの無い旧チェックポイント（unet-a）はヘッドをゼロ初期化（cfl_init=0.25）のまま読む."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = UNet(in_ch=int(ck.get("in_ch", 8)), widths=tuple(ck["widths"]))
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("cfl_head.")] + list(unexpected)
    if bad:
        raise KeyError(f"state_dict の不一致: {bad}")
    return net.eval()


def _sample_meta(samples: Sequence[Sample], idx: Sequence[int]) -> dict[str, Any]:
    return {
        "theta": [samples[i].theta.to_dict() for i in idx],
        "u_in": torch.tensor([samples[i].theta.u_in for i in idx], dtype=torch.float32),
        "p_ref": torch.tensor([p_ref(samples[i].theta) for i in idx], dtype=torch.float32),
        "r_ref": np.array([samples[i].residual_ref for i in idx], dtype=float),
    }


def _floor_tensors(
    samples: Sequence[Sample], idx: Sequence[int], floor: dict[int, np.ndarray]
) -> dict[str, Tensor]:
    xs, ss, ms, ys = [], [], [], []
    for i in idx:
        s = samples[i]
        x_ext, sc, open_mask = floor_input(s.x, floor[s.theta.seed])
        xs.append(x_ext)
        ss.append(sc)
        ms.append(open_mask[None])
        ys.append(floor[s.theta.seed])
    return {
        "x": torch.from_numpy(np.stack(xs)),
        "s": torch.from_numpy(np.stack(ss)),
        "open": torch.from_numpy(np.stack(ms)),
        "ys": torch.from_numpy(np.stack(ys)),
    }


def _predict_floor(net: UNet, xb: Tensor, fl: dict[str, Tensor], b) -> tuple[Tensor, Tensor]:
    """床モードの予測: (y_pred 正規化単位, log cfl)."""
    c, logcfl = net(xb)
    return floor_predict(fl["ys"][b], fl["s"][b], c, fl["open"][b]), logcfl


def _residual_term(
    pool: ResidualLossPool,
    x: Tensor,
    yhat: Tensor,
    logcfl: Tensor,
    meta: dict[str, Any],
    sel: Sequence[int],
    with_grad: bool,
    cfl_gain: float = 1.0,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """バッチの sel 位置のサンプルについて残差損失を評価し、(torch 損失, ワーカー出力) を返す."""
    sel_t = torch.tensor(list(sel))
    fields = to_physical(x[sel_t], yhat[sel_t], meta["u_in"][sel_t], meta["p_ref"][sel_t])
    lc = logcfl[sel_t]
    outs = pool.compute(
        [meta["theta"][b] for b in sel],
        fields.detach().double().numpy(),
        torch.exp(lc.detach()).double().numpy(),
        meta["r_ref"][list(sel)],
        with_grad=with_grad,
    )
    if not with_grad:  # 検証用: 値だけ（勾配は持たない）
        ok = [o["loss"] for o in outs if np.isfinite(o["loss"])]
        return torch.tensor(float(np.mean(ok)) if ok else 0.0), outs
    loss, _n_ok = straight_through(fields, lc, outs, cfl_gain)
    return loss, outs


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
    init_from: Path | None = None,
    res_weight: float = 0.0,
    res_steps: int = 5,
    res_transform: str = "ratio",
    res_workers: int = 8,
    res_frac: float = 1.0,
    res_cfl_gain: float = 1.0,
    grad_clip: float = 1.0,
    floor: dict[int, np.ndarray] | None = None,
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
    meta_tr = _sample_meta(samples, split["train"])
    meta_va = _sample_meta(samples, split["val"])
    fl_tr = fl_va = None
    if floor is not None:  # 床モード: 入力 11ch、予測 y = y_S + s·c、損失は開きセルの (Δ/s)²
        fl_tr = _floor_tensors(samples, split["train"], floor)
        fl_va = _floor_tensors(samples, split["val"], floor)
        x_tr, x_va = fl_tr["x"], fl_va["x"]
    in_ch = int(x_tr.shape[1])
    if init_from is not None:
        net = load_model(init_from).train()
    else:
        net = UNet(in_ch=in_ch, widths=widths)
        if floor is not None:  # 出発点を床（c = 0）にする
            torch.nn.init.zeros_(net.head.weight)
            torch.nn.init.zeros_(net.head.bias)
    widths = net.widths
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    best_val, best_epoch, history = float("inf"), -1, []
    best_path = out_dir / "best.pt"
    use_res = res_weight > 0.0
    pool = ResidualLossPool(res_workers, res_steps, res_transform) if use_res else None
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    def val_pass() -> tuple[float, float, dict[str, float]]:
        net.eval()
        mse_sum, res_vals, cfls, fails = 0.0, [], [], 0
        fl_sum = 0.0
        with torch.no_grad():
            for k in range(0, len(x_va), 128):
                xb, yb = x_va[k : k + 128], y_va[k : k + 128]
                if fl_va is not None:
                    bb = torch.arange(k, min(k + 128, len(x_va)))
                    yhat, logcfl = _predict_floor(net, xb, fl_va, bb)
                    fl_sum += floor_loss(yhat, yb, fl_va["s"][bb], fl_va["open"][bb]).item() * len(
                        bb
                    )
                else:
                    yhat, logcfl = net(xb)
                mse_sum += torch.nn.functional.mse_loss(yhat, yb, reduction="sum").item()
                cfls += torch.exp(logcfl).tolist()
                if pool is not None:
                    sub = {
                        kk: (v[k : k + 128] if kk != "theta" else v[k : k + 128])
                        for kk, v in meta_va.items()
                    }
                    _, outs = _residual_term(
                        pool, xb[:, :8], yhat, logcfl, sub, range(len(xb)), with_grad=False
                    )
                    res_vals += [o["loss"] for o in outs if np.isfinite(o["loss"])]
                    fails += sum(not np.isfinite(o["loss"]) for o in outs)
        mse = mse_sum / y_va.numel()
        res = float(np.mean(res_vals)) if res_vals else 0.0
        info = {
            "floor_loss": fl_sum / len(x_va),
            "cfl_median": float(np.median(cfls)),
            "cfl_min": float(np.min(cfls)),
            "cfl_max": float(np.max(cfls)),
            "res_fail": float(fails),
        }
        return mse, res, info

    try:
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(x_tr), generator=gen)
            tot_mse, tot_res, n_res = 0.0, 0.0, 0
            g_mse_norm = g_res_norm = float("nan")
            for k in range(0, len(perm), batch):
                b = perm[k : k + batch]
                xb, yb = x_tr[b], y_tr[b]
                if fl_tr is not None:
                    yhat, logcfl = _predict_floor(net, xb, fl_tr, b)
                    mse = torch.nn.functional.mse_loss(yhat, yb)  # 記録用（unet-a と同じ単位）
                    loss = floor_loss(yhat, yb, fl_tr["s"][b], fl_tr["open"][b])
                else:
                    yhat, logcfl = net(xb)
                    mse = torch.nn.functional.mse_loss(yhat, yb)
                    loss = mse
                if div_weight > 0:
                    loss = loss + div_weight * divergence_loss(xb, yhat)
                if pool is not None:
                    n_sel = max(1, int(round(res_frac * len(b))))
                    sel = sorted(rng.choice(len(b), n_sel, replace=False).tolist())
                    sub = {
                        kk: ([v[i] for i in b.tolist()] if kk == "theta" else v[b])
                        for kk, v in meta_tr.items()
                    }
                    res_loss, outs = _residual_term(
                        pool,
                        xb[:, :8],
                        yhat,
                        logcfl,
                        sub,
                        sel,
                        with_grad=True,
                        cfl_gain=res_cfl_gain,
                    )
                    if k == 0:  # 最初のバッチで両項の勾配ノルムを測る（res_weight の目安）
                        g1 = (
                            torch.autograd.grad(mse, net.head.weight, retain_graph=True)[0]
                            .norm()
                            .item()
                        )
                        g2 = (
                            torch.autograd.grad(res_loss, net.head.weight, retain_graph=True)[0]
                            .norm()
                            .item()
                        )
                        g_mse_norm, g_res_norm = g1, g2
                        if log is not None:
                            log(
                                f"epoch {ep:4d} first batch: |g|head mse {g1:.2e} res {g2:.2e} "
                                f"(res/mse {g2 / max(g1, 1e-30):.1e}) res mean {np.mean([o['loss'] for o in outs if np.isfinite(o['loss'])]):.2f}"
                            )
                    loss = loss + res_weight * res_loss
                    ok = [o["loss"] for o in outs if np.isfinite(o["loss"])]
                    tot_res += float(np.sum(ok))
                    n_res += len(ok)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()
                tot_mse += mse.item() * len(b)
            sched.step()
            tr_mse = tot_mse / len(perm)
            tr_res = tot_res / max(1, n_res)
            va_mse, va_res, info = val_pass()
            va_obj = (info["floor_loss"] if fl_va is not None else va_mse) + res_weight * va_res
            history.append(
                (
                    ep,
                    tr_mse,
                    tr_res,
                    va_mse,
                    va_res,
                    va_obj,
                    info["cfl_median"],
                    time.perf_counter() - t0,
                )
            )
            if va_obj < best_val:
                best_val, best_epoch = va_obj, ep
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "widths": widths,
                        "epoch": ep,
                        "val_loss": va_obj,
                        "val_mse": va_mse,
                        "val_res": va_res,
                        "res_weight": res_weight,
                        "res_steps": res_steps,
                        "res_transform": res_transform,
                        "res_cfl_gain": res_cfl_gain,
                        "in_ch": in_ch,
                        "floor": floor is not None,
                    },
                    best_path,
                )
            if log is not None:
                log(
                    f"epoch {ep:4d} train mse {tr_mse:.3e} res {tr_res:.3f} | val mse {va_mse:.3e} "
                    f"floor {info['floor_loss']:.3e} res {va_res:.3f} "
                    f"obj {va_obj:.3e} best {best_val:.3e}@{best_epoch} | cfl med {info['cfl_median']:.3g} "
                    f"[{info['cfl_min']:.3g},{info['cfl_max']:.3g}] fail {info['res_fail']:.0f} "
                    f"| |g|head mse {g_mse_norm:.2e} res {g_res_norm:.2e} lr {sched.get_last_lr()[0]:.2e} "
                    f"{time.perf_counter() - t0:7.1f}s"
                )
            with (out_dir / "history.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "epoch",
                        "train_mse",
                        "train_res",
                        "val_mse",
                        "val_res",
                        "val_obj",
                        "val_cfl_median",
                        "elapsed",
                    ]
                )
                w.writerows(history)
    finally:
        if pool is not None:
            pool.close()
    return TrainResult(best_path, best_epoch, best_val, history)
