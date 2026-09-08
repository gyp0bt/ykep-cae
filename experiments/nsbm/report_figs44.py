"""status-44 の図: 局所 Galerkin の残差比 vs 反復差、cfl_init 掃引の分布、unet-r の学習履歴、選択器の比較.

python experiments/nsbm/report_figs44.py            # あるものだけ描く（results/figs44/*.png）
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = HERE / "results" / "figs44"


def fig_galerkin() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k in zip(axes, (6, 16), strict=True):
        p = HERE / "results" / f"galerkin-knn-k{k}.json"
        if not p.exists():
            continue
        rows = json.load(open(p))
        rr = np.array([r["r_ratio"] for r in rows])
        d = np.array([r["galerkin"]["n_iter"] - r["stokes"]["n_iter"] for r in rows])
        fams = sorted({r["family"] for r in rows})
        for f in fams:
            m = np.array([r["family"] == f for r in rows])
            ax.scatter(rr[m], d[m], s=28, label=f, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("初期残差比 |R(x₀)|/|R(Stokes)|（Galerkin、1 以下が保証）")
        ax.set_ylabel("Newton 反復数の差（Galerkin − Stokes 発進）")
        ax.set_title(f"局所基底 Galerkin（Stokes + kNN {k} 件）")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "galerkin_scatter.png", dpi=130)
    plt.close(fig)


def fig_cfl_sweep() -> dict | None:
    p = HERE / "results" / "cfl_labels.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    by = defaultdict(list)
    conv = defaultdict(list)
    for r in rows:
        c = float(r["cfl_init"])
        by[c].append(int(r["n_iter"]))
        conv[c].append(r["converged"] == "True")
    cfls = sorted(by)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [np.array(by[c])[np.array(conv[c])] for c in cfls]
    ax.boxplot(data, tick_labels=[f"{c:g}" for c in cfls], showfliers=False)
    for i, c in enumerate(cfls):
        n_fail = int((~np.array(conv[c])).sum())
        ax.text(
            i + 1,
            ax.get_ylim()[1] * 0.92,
            f"未収束 {n_fail}\n({n_fail / len(conv[c]):.1%})",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("cfl_init（Stokes 発進、全 θ）")
    ax.set_ylabel("Newton 反復数（収束したもの）")
    ax.set_title("SER の出発係数 cfl_init の掃引: 反復数の分布と未収束率")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "cfl_sweep_box.png", dpi=130)
    plt.close(fig)
    return {
        f"{c:g}": {
            "median": float(np.median(d)),
            "q90": float(np.percentile(d, 90)),
            "fails": int((~np.array(conv[c])).sum()),
            "n": len(conv[c]),
        }
        for c, d in zip(cfls, data, strict=True)
    }


def fig_history() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for run, ls in (("unet-r0", "--"), ("unet-r", "-")):
        p = HERE / "runs" / run / "history.csv"
        if not p.exists():
            continue
        h = list(csv.DictReader(p.open()))
        ep = [int(r["epoch"]) for r in h]
        axes[0].plot(ep, [float(r["train_mse"]) for r in h], ls, color="C0", label=f"{run} train")
        axes[0].plot(ep, [float(r["val_mse"]) for r in h], ls, color="C1", label=f"{run} val")
        axes[1].plot(ep, [float(r["train_res"]) for r in h], ls, color="C0", label=f"{run} train")
        axes[1].plot(ep, [float(r["val_res"]) for r in h], ls, color="C1", label=f"{run} val")
        axes[2].plot(ep, [float(r["val_cfl_median"]) for r in h], ls, color="C2", label=run)
    axes[0].set_yscale("log")
    axes[0].set_title("場の MSE（unet-a の val 最良は 2.2e-2）")
    axes[1].set_title("残差項 Σ_k log(|R(x_k)|/|R_ref|)（k=0..5）")
    axes[2].set_title("予測 cfl_init の中央値（val）")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "unet_r_history.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_galerkin()
    stats = fig_cfl_sweep()
    fig_history()
    if stats:
        print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
