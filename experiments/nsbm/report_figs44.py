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
    """unet-r2（残差損失 + cfl ヘッド）と unet-s（Stokes 床 + 補正）の学習履歴."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    p = HERE / "runs" / "unet-r2" / "history.csv"
    if p.exists():
        h = list(csv.DictReader(p.open()))
        ep = [int(r["epoch"]) for r in h]
        axes[0].plot(ep, [float(r["val_mse"]) for r in h], "-o", ms=3, label="unet-r2 val MSE")
        axes[0].axhline(0.02187, color="gray", ls="--", lw=0.8, label="unet-a val 最良 2.19e-2")
        axes[0].set_title("残差損失: 場の MSE（旧単位）")
        axes[1].plot(ep, [float(r["val_res"]) for r in h], "-o", ms=3, color="C1", label="val")
        axes[1].plot(
            ep, [float(r["train_res"]) for r in h], "--", color="C1", alpha=0.6, label="train"
        )
        axes[1].set_title("残差項 Σ_k log(|R(x_k)|/|R_ref|)（k=0..5）")
        ax2 = axes[2]
        ax2.plot(
            ep,
            [float(r["val_cfl_median"]) for r in h],
            "-o",
            ms=3,
            color="C2",
            label="予測 cfl_init 中央値",
        )
        ax2.set_yscale("log")
        ax2.set_title("cfl ヘッド")
        ax3 = axes[3]
        ax3.plot(
            ep,
            [float(r["val_n_iter_mean"]) for r in h],
            "-o",
            ms=3,
            color="C3",
            label="平均（失敗は 120）",
        )
        ax3.plot(
            ep, [float(r["val_n_iter_median"]) for r in h], "-s", ms=3, color="C4", label="中央値"
        )
        ax3.axhline(13, color="gray", ls="--", lw=0.8, label="Stokes 発進 中央値 13")
        ax3.set_title("val 96 件を本当に解いた反復数（選抜指標）")
    p = HERE / "runs" / "unet-s" / "history.csv"
    if p.exists():
        h = list(csv.DictReader(p.open()))
        ep = [int(r["epoch"]) for r in h]
        axes[0].plot(
            ep,
            [float(r["val_mse"]) for r in h],
            "-",
            color="C5",
            label="unet-s val MSE（床 + 補正）",
        )
    axes[0].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "unet_r_history.png", dpi=130)
    plt.close(fig)


def fig_early_restart() -> None:
    p = HERE / "results" / "cfl_histories.json"
    if not p.exists():
        return
    rows = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, c in zip(axes, (4.0, 8.0, 16.0), strict=True):
        rs = [r for r in rows if r["cfl_init"] == c]
        for conv, color, label in (
            (True, "C0", "収束"),
            (False, "C3", "未収束（120 反復で打ち切り）"),
        ):
            for r in rs:
                if r["converged"] != conv:
                    continue
                h = np.array(r["hist"][:40])
                ax.plot(
                    np.arange(len(h)),
                    np.maximum(h, 1e-8),
                    color=color,
                    alpha=0.25 if conv else 0.9,
                    lw=0.8,
                )
            ax.plot([], [], color=color, label=label)
        ax.axhline(0.3, color="k", ls="--", lw=0.8)
        ax.axvline(10, color="k", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Newton 反復")
        ax.set_title(f"Stokes 発進、cfl_init {c:g}（test 357 件）")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("定常残差比 |R|/|R_ref|")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "早期やり直し規則: 10 反復後に残差比 0.3 を超えていれば 0.25 でやり直す（点線）",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "early_restart_hist.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_galerkin()
    stats = fig_cfl_sweep()
    fig_history()
    fig_early_restart()
    if stats:
        print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
