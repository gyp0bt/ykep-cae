"""報告用の図: ファミリ見本、Stokes vs UNet/kNN の反復数散布、ファミリ別箱ひげ、場の比較、学習曲線.

python experiments/nsbm/report_figs.py --run experiments/nsbm/runs/unet-a
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAPGothic", "IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from nsbm.families import FAMILIES, LX, LY, build_h, sample_theta  # noqa: E402

FAM_COLORS = dict(zip(FAMILIES, plt.cm.tab10.colors, strict=False))


def fig_families(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.2))
    for ax, fam in zip(axes.ravel(), FAMILIES, strict=True):
        th = sample_theta(7, [fam])
        h = build_h(th)
        ax.imshow(
            np.log10(h / th.h0).T,
            origin="lower",
            extent=(0, LX, 0, LY),
            cmap="viridis",
            vmin=-2.2,
            vmax=0.9,
        )
        ax.set_title(f"{fam}  ({th.inlet.wall}→{th.outlet.wall})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("h 場ファミリの見本（色 = log10(h/h0)、seed 7）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k.endswith(
                ("n_iter", "n_iter_dataset", "r0_ratio", "cfl0", "elapsed", "u_in", "h0")
            ):
                r[k] = float(v)
            elif k.endswith("converged"):
                r[k] = v == "True"
    return rows


def fig_scatter(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    lim = max(max(r["stokes_n_iter"] for r in rows), max(r["unet_n_iter"] for r in rows)) * 1.05
    for ax, m in zip(axes, ("knn", "unet"), strict=True):
        for fam in FAMILIES:
            sub = [r for r in rows if r["family"] == fam]
            ax.scatter(
                [r["stokes_n_iter"] for r in sub],
                [r[f"{m}_n_iter"] for r in sub],
                s=14,
                alpha=0.7,
                color=FAM_COLORS[fam],
                label=fam,
            )
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.plot([0, lim], [0, lim / 2], "k:", lw=0.8, label="半分")
        ax.set_xlabel("Newton 反復数（Stokes 発進）")
        ax.set_title(f"{m} 初期場")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Newton 反復数（学習/補間の初期場）")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_box(rows: list[dict], out: Path) -> None:
    fams = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.26
    for k, (m, c) in enumerate((("stokes", "0.5"), ("knn", "tab:orange"), ("unet", "tab:blue"))):
        data = [[r[f"{m}_n_iter"] for r in rows if r["family"] == f] for f in fams]
        bp = ax.boxplot(
            data,
            positions=np.arange(len(fams)) + (k - 1) * w,
            widths=w * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "k"},
        )
        for b in bp["boxes"]:
            b.set_facecolor(c)
            b.set_alpha(0.6)
        ax.plot([], [], color=c, lw=8, alpha=0.6, label=m)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylabel("Newton 反復数")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("ファミリ別の Newton 反復数（箱: 四分位、外れ値非表示）")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_history(run: Path, out: Path) -> None:
    with (run / "history.csv").open() as f:
        h = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in h]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(ep, [float(r["train_mse"]) for r in h], label="train")
    ax.semilogy(ep, [float(r["val_mse"]) for r in h], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE（正規化した u, v, p）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_fields(run: Path, data: Path, rows: list[dict], out: Path, n_cases: int = 4) -> None:
    import torch

    from nsbm.dataset import load_shards
    from nsbm.features import denormalize_y
    from nsbm.train import load_model

    samples = {s.theta.seed: s for s in load_shards(data)}
    net = load_model(run / "best.pt")
    # 反復数の削減が大きい順から、ファミリが重ならないように選ぶ
    picked, seen = [], set()
    for r in sorted(rows, key=lambda r: r["unet_n_iter"] / max(r["stokes_n_iter"], 1)):
        if r["family"] not in seen and int(r["seed"]) in samples:
            picked.append(r)
            seen.add(r["family"])
        if len(picked) >= n_cases:
            break
    fig, axes = plt.subplots(len(picked), 4, figsize=(13, 2.6 * len(picked)))
    for row_ax, r in zip(np.atleast_2d(axes), picked, strict=True):
        s = samples[int(r["seed"])]
        with torch.no_grad():
            yhat = net(torch.from_numpy(s.x[None])).numpy()[0]
        u, v, p = s.fields()
        uh, vh, ph = denormalize_y(s.theta, yhat)
        sp, sph = np.hypot(u, v), np.hypot(uh, vh)
        vmax = sp.max()
        ims = [
            (np.log10(np.exp(s.x[0])).T, "log10(h/h0)", "viridis", None),
            (sp.T, "|u| 正解", "magma", vmax),
            (sph.T, "|u| UNet", "magma", vmax),
            ((sph - sp).T, "差 UNet−正解", "coolwarm", 0.3 * vmax),
        ]
        for ax, (im, title, cmap, vm) in zip(row_ax, ims, strict=True):
            kw = (
                {"vmin": -vm, "vmax": vm}
                if (vm is not None and cmap == "coolwarm")
                else ({"vmin": 0, "vmax": vm} if vm else {})
            )
            ax.imshow(im, origin="lower", extent=(0, LX, 0, LY), cmap=cmap, **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        row_ax[0].set_ylabel(
            f"{s.theta.family} u_in={s.theta.u_in:.2f}\nNewton {int(r['stokes_n_iter'])}→{int(r['unet_n_iter'])}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_dataset(data: Path, out: Path) -> dict:
    """データセット統計: ファミリ別の収束率と Stokes 発進の Newton 反復数分布、u_in・h0 依存."""
    from nsbm.dataset import load_shards

    S = load_shards(data)
    fams = [f for f in FAMILIES if any(s.theta.family == f for s in S)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    conv = {f: np.mean([s.converged for s in S if s.theta.family == f]) for f in fams}
    axes[0].bar(range(len(fams)), [conv[f] for f in fams], color=[FAM_COLORS[f] for f in fams])
    axes[0].set_xticks(range(len(fams)))
    axes[0].set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("収束率（Stokes 発進、200 反復以内）")
    axes[0].grid(axis="y", alpha=0.3)
    data_it = [[s.n_iter for s in S if s.theta.family == f and s.converged] for f in fams]
    axes[1].boxplot(data_it, showfliers=False, patch_artist=True)
    for b, f in zip(axes[1].patches, fams, strict=False):
        b.set_facecolor(FAM_COLORS[f])
        b.set_alpha(0.7)
    axes[1].set_xticks(range(1, len(fams) + 1))
    axes[1].set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Newton 反復数（収束例）")
    axes[1].grid(axis="y", alpha=0.3)
    u = np.array([s.theta.u_in for s in S])
    h0 = np.array([s.theta.h0 for s in S])
    c = np.array([s.converged for s in S])
    sc = axes[2].scatter(u, h0 * 1e3, c=np.where(c, "tab:blue", "tab:red"), s=4, alpha=0.5)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("u_in [m/s]")
    axes[2].set_ylabel("h0 [mm]")
    axes[2].set_title("青: 収束 / 赤: 未収束", fontsize=9)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    _ = sc
    return {
        "n": len(S),
        "converged": int(c.sum()),
        "by_family": {
            f: {
                "n": int(sum(s.theta.family == f for s in S)),
                "conv": float(conv[f]),
                "newton_median": float(np.median(d)),
            }
            for f, d in zip(fams, data_it, strict=True)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "runs" / "unet-a")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--rows", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_families(args.out / "families.png")
    if args.data.exists():
        stats = fig_dataset(args.data, args.out / "dataset.png")
        (args.out / "dataset_stats.json").write_text(
            json.dumps(stats, indent=1, ensure_ascii=False)
        )
    rows_path = args.rows or (HERE / "results" / f"eval-{args.run.name}.csv")
    if rows_path.exists():
        rows = load_rows(rows_path)
        fig_scatter(rows, args.out / "scatter.png")
        fig_box(rows, args.out / "box.png")
        fig_fields(args.run, args.data, rows, args.out / "fields.png")
    if (args.run / "history.csv").exists():
        fig_history(args.run, args.out / "history.png")
    print(f"-> {args.out}")
    _ = json


if __name__ == "__main__":
    main()
