"""trama 収束性調査の図（results/trama_* から PNG を作る）.

python experiments/nsb/trama_plots.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "trama_figs"
FIG.mkdir(parents=True, exist_ok=True)

# dataviz 参照パレット（カテゴリは固定順、逐次は青 1 色、発散は青/橙 + 灰）
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#f3a07a", "#c94a1c"]
)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "legend.frameon": False,
        "font.family": ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"],
    }
)


def imshow_field(ax, f: np.ndarray, extent, title: str, cmap=SEQ, norm=None, **kw):
    im = ax.imshow(f.T, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal", **kw)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def stair_overlay(ax, stair: np.ndarray, extent, color=CAT[1]):
    nx, ny = stair.shape
    xs = (np.arange(nx) + 0.5) * (extent[1] / nx) * 1e3
    ys = (np.arange(ny) + 0.5) * (extent[3] / ny) * 1e3
    ii, jj = np.nonzero(stair)
    ax.scatter(xs[ii], ys[jj], s=1.5, c=color, marker="s", linewidths=0)


def load_diag(label: str):
    p = RES / f"trama_diag_{label}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    j = json.loads((RES / f"trama_diag_{label}.json").read_text())
    return d, j


def cell_classes(d) -> dict[str, np.ndarray]:
    chan = d["h"] > 1e-4
    src, snk = d["q_src"] > 0, d["c_sink"] > 0
    stair, wall = d["stair"], d["wall_adj"] & ~d["stair"]
    return {
        "source": src,
        "sink": snk,
        "stair": stair & ~src & ~snk,
        "wall_adj": wall & ~src & ~snk,
        "chan_int": chan & ~d["wall_adj"] & ~src & ~snk,
        "blocked": ~chan,
    }


# ----------------------------------------------------------------------------
def fig_geometry():
    d, _ = load_diag("orig-interior-m0.15-dx1.5-cfl0.25-sou")
    if d is None:
        return
    nx, ny = int(d["nx"]), int(d["ny"])
    ext = (0, 600, 0, 350)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
    im = imshow_field(
        axs[0],
        np.where(d["h"] > 1e-4, 1.0, 0.0),
        ext,
        "流路（h=3.8mm）と階段セル（橙）",
        cmap=SEQ,
        vmin=0,
        vmax=1.6,
    )
    stair_overlay(axs[0], d["stair"], (0, 0.6, 0, 0.35))
    axs[0].text(
        5, 5, f"格子 {nx}×{ny}, Δx=1.5mm, 階段セル {int(d['stair'].sum())}", fontsize=8, color=INK2
    )
    im = imshow_field(
        axs[1], d["speed"], ext, "Stokes 参照場の速さ |u| [m/s]（0.15 kg/s）", norm=LogNorm(1e-3, 8)
    )
    plt.colorbar(im, ax=axs[1], fraction=0.03)
    im = imshow_field(axs[2], d["p"], ext, "Stokes 参照場の圧力 p [Pa]", cmap=SEQ)
    plt.colorbar(im, ax=axs[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "f1_geometry.png", dpi=130)
    plt.close(fig)


def fig_regime():
    ext = (0, 600, 0, 350)
    labels = [
        ("orig-interior-m0.0015-dx1.5-cfl0.25-sou", "0.0015 kg/s"),
        ("orig-interior-m0.15-dx1.5-cfl0.25-sou", "0.15 kg/s"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.2))
    for k, (lab, name) in enumerate(labels):
        d, _ = load_diag(lab)
        if d is None:
            continue
        chan = d["h"] > 1e-4
        cf = np.where(chan, d["conv_frac"], np.nan)
        im = imshow_field(
            axs[0, k],
            cf,
            ext,
            f"a_P の対流割合 a_conv/(a_conv+a_diff+a_drag)  {name}",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axs[0, k], fraction=0.03)
        re = d["re_cell"][chan]
        axs[1, k].hist(np.log10(np.maximum(re, 1e-4)), bins=60, color=CAT[0], linewidth=0)
        axs[1, k].axvline(np.log10(2), color=CAT[1], lw=1.2)
        axs[1, k].text(
            np.log10(2) + 0.05,
            axs[1, k].get_ylim()[1] * 0.9,
            "Re_cell = 2",
            color=CAT[1],
            fontsize=8,
        )
        axs[1, k].set_xlabel("log10 セル Re = ρ|u|Δx/μ（流路セルのみ）")
        axs[1, k].set_title(
            f"セル Re の分布  {name}  中央値 {np.median(re):.3g}",
            loc="left",
            fontsize=10,
            color=INK,
        )
    fig.tight_layout()
    fig.savefig(FIG / "f2_regime.png", dpi=130)
    plt.close(fig)


def fig_spectrum():
    rows = []
    for p in glob.glob(str(RES / "trama_sv_*.json")):
        for r in json.loads(Path(p).read_text()):
            lab = r["label"]
            port = "wall" if "-wall-" in lab else "interior"
            m = float(lab.split("-m")[1].split("-")[0])
            rows.append(
                (
                    port,
                    m,
                    r["sigma_max"],
                    r["sigma_min"][0],
                    r["kappa"],
                    r["v_min"][0]["mismatch"],
                    r["step"]["true_resid_ratio"],
                    r["step"]["norm_d"] / r["step"]["norm_x"],
                )
            )
    if not rows:
        return
    rows.sort()
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for port, c in (("interior", CAT[0]), ("wall", CAT[1])):
        rr = [r for r in rows if r[0] == port]
        if not rr:
            continue
        ms = [r[1] for r in rr]
        axs[0].plot(ms, [r[4] for r in rr], "o-", color=c, lw=2, ms=6, label=f"{port} ports")
        axs[1].plot(ms, [r[5] for r in rr], "o-", color=c, lw=2, ms=6, label=f"{port}")
        axs[2].plot(ms, [r[6] for r in rr], "o-", color=c, lw=2, ms=6, label=f"{port}")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_title("κ₂(J1+τ) = σ_max/σ_min（cfl 0.25）", loc="left", color=INK)
    axs[0].set_xlabel("質量流量 [kg/s]")
    axs[0].legend()
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_title("最小特異方向 v_min での |A_fd v − J1τ v| / |J1τ v|", loc="left", color=INK)
    axs[1].set_xlabel("質量流量 [kg/s]")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].axhline(0.3, color=CAT[7], lw=1, ls="--")
    axs[2].text(ms[0], 0.33, "棄却閾値 0.3", color=CAT[7], fontsize=8)
    axs[2].axhline(1.0, color=MUTED, lw=1)
    axs[2].set_title("厳密な J1τ⁻¹ b 一歩の真の残差比 |b − A_fd d|/|b|", loc="left", color=INK)
    axs[2].set_xlabel("質量流量 [kg/s]")
    fig.tight_layout()
    fig.savefig(FIG / "f3_spectrum.png", dpi=130)
    plt.close(fig)
    # v_min の空間分布（0.15 interior）
    p = RES / "trama_sv_orig-interior-m0.15-dx1.5-cfl0.25.npz"
    if p.exists():
        sv = np.load(p)
        h = sv["h"]
        nx, ny = h.shape
        n = nx * ny
        ext = (0, 600, 0, 350)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
        for k in range(3):
            vp = sv["v_min"][k][2 * n :].reshape(nx, ny)
            vmax = np.abs(vp).max()
            im = imshow_field(
                axs[k],
                vp,
                ext,
                f"v_min[{k}] の圧力成分（σ={sv['sigma_min'][k]:.2e}）",
                cmap=DIV,
                norm=TwoSlopeNorm(0, -vmax, vmax),
            )
            axs[k].contour(
                np.linspace(0.75, 599.25, nx),
                np.linspace(0.75, 349.25, ny),
                (h > 1e-4).T,
                levels=[0.5],
                colors=[INK2],
                linewidths=0.4,
            )
            plt.colorbar(im, ax=axs[k], fraction=0.03)
        fig.suptitle(
            "J1+τ の最小特異ベクトル（u, v 成分は 0）: 閉塞領域と出口円板の圧力レベル",
            x=0.01,
            ha="left",
            fontsize=10,
            color=INK,
        )
        fig.tight_layout()
        fig.savefig(FIG / "f3b_vmin.png", dpi=130)
        plt.close(fig)


def fig_gmres():
    cases = [
        ("orig-interior-m0.0015-dx1.5-cfl0.25-sou", "内部ポート 0.0015"),
        ("orig-interior-m0.015-dx1.5-cfl0.25-sou", "内部ポート 0.015"),
        ("orig-interior-m0.05-dx1.5-cfl0.25-sou", "内部ポート 0.05"),
        ("orig-interior-m0.15-dx1.5-cfl0.25-sou", "内部ポート 0.15"),
        ("orig-wall-m0.15-dx1.5-cfl0.25-sou", "壁ポート 0.15"),
        ("orig-interior-m0.15-dx1.5-cfl0.25-fou", "内部 0.15, 残差も 1 次風上"),
        ("straight0-interior-m0.15-dx1.5-cfl0.25-sou", "直線流路 0°, 0.15"),
        ("straight30-interior-m0.15-dx1.5-cfl0.25-sou", "直線流路 30°, 0.15"),
        ("orig-interior-m0.15-dx1.5-cfl1e-06-sou", "内部ポート 0.15, cfl 1e-6"),
    ]
    combos = [
        ("j1-simple", CAT[0], "J1 厳密 × SIMPLE"),
        ("fd-pardiso", CAT[2], "有限差分 × LU(J1)"),
        ("fd-simple", CAT[1], "有限差分 × SIMPLE（本番）"),
        ("j1-pardiso", MUTED, "J1 × LU（検算）"),
    ]
    avail = [(lab, name) for lab, name in cases if (RES / f"trama_diag_{lab}.json").exists()]
    if not avail:
        return
    ncol = 4
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (lab, name) in zip(axs.ravel(), avail, strict=False):
        j = json.loads((RES / f"trama_diag_{lab}.json").read_text())["gmres"]
        for key, c, nm in combos:
            if key not in j:
                continue
            h = j[key]["hist"]
            ax.plot(np.arange(1, len(h) + 1), h, color=c, lw=1.6, label=nm)
            ax.plot([len(h)], [j[key]["true_ratio"]], marker="o", ms=7, mfc="none", mec=c, mew=1.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 1e3)
        ax.axhline(0.3, color=CAT[7], lw=0.8, ls="--")
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Arnoldi 反復")
    axs[0, 0].set_ylabel("残差比（線: Givens 推定, ○: 最後の真の残差）")
    axs[0, 0].legend(fontsize=8, loc="lower left")
    for ax in axs.ravel()[len(avail) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "f4_gmres.png", dpi=130)
    plt.close(fig)


def fig_ritz():
    cases = [
        ("orig-interior-m0.0015-dx1.5-cfl0.25-sou", "0.0015"),
        ("orig-interior-m0.015-dx1.5-cfl0.25-sou", "0.015"),
        ("orig-interior-m0.05-dx1.5-cfl0.25-sou", "0.05"),
        ("orig-interior-m0.15-dx1.5-cfl0.25-sou", "0.15"),
    ]
    avail = [(lab, name) for lab, name in cases if (RES / f"trama_diag_{lab}.json").exists()]
    if not avail:
        return
    fig, axs = plt.subplots(2, len(avail), figsize=(3.6 * len(avail), 6.4), squeeze=False)
    for k, (lab, name) in enumerate(avail):
        j = json.loads((RES / f"trama_diag_{lab}.json").read_text())["gmres"]
        for r, (key, c, nm) in enumerate(
            [("j1-simple", CAT[0], "J1 厳密 × SIMPLE"), ("fd-simple", CAT[1], "有限差分 × SIMPLE")]
        ):
            ax = axs[r, k]
            re, im = np.array(j[key]["ritz_re"]), np.array(j[key]["ritz_im"])
            mag = np.hypot(re, im)
            ang = np.arctan2(im, re)
            ax.scatter(np.log10(mag), ang, s=14, c=c, linewidths=0)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_xlim(-5, 8)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_title(f"{nm}  {name} kg/s", loc="left", fontsize=9, color=INK)
            ax.set_xlabel("log10 |λ|（Ritz 値）")
            if k == 0:
                ax.set_ylabel("arg λ [rad]")
    fig.suptitle(
        "前処理付き作用素 A M⁻¹ の Ritz 値: 1（log=0）に集まるほど GMRES は速い",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f5_ritz.png", dpi=130)
    plt.close(fig)


def fig_residual_maps():
    for lab, name in [
        ("orig-interior-m0.15-dx1.5-cfl0.25-sou", "0.15"),
        ("orig-interior-m0.0015-dx1.5-cfl0.25-sou", "0.0015"),
    ]:
        d, j = load_diag(lab)
        if d is None:
            continue
        nx, ny = int(d["nx"]), int(d["ny"])
        n = nx * ny
        ext = (0, 600, 0, 350)
        r = d["resid_fd-simple"]
        b = d["b"]
        cls = cell_classes(d)
        fig, axs = plt.subplots(1, 4, figsize=(18, 3.4))
        for k, bn in enumerate(("u 運動量", "v 運動量", "連続")):
            rb = np.abs(r[k * n : (k + 1) * n]).reshape(nx, ny)
            vmax = rb.max()
            im = imshow_field(
                axs[k],
                np.maximum(rb, vmax * 1e-6),
                ext,
                f"|b − A x| の {bn} 成分（有限差分×SIMPLE, {name} kg/s）",
                norm=LogNorm(vmax * 1e-5, vmax),
            )
            plt.colorbar(im, ax=axs[k], fraction=0.03)
        # 分担
        ax = axs[3]
        names = list(cls)
        width = 0.25
        for k, (bn, c) in enumerate(zip(("u", "v", "p"), CAT[:3], strict=True)):
            rb = (r[k * n : (k + 1) * n] ** 2).reshape(nx, ny)
            tot = rb.sum()
            shares = [rb[cls[m]].sum() / tot for m in names]
            ax.bar(
                np.arange(len(names)) + (k - 1) * width,
                shares,
                width=width * 0.92,
                color=c,
                label=bn,
                linewidth=0,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_title("残差二乗和のセル種別分担（|b−Ax|²）", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8)
        cnt = {m: int(cls[m].sum()) for m in names}
        ax.text(
            0.02,
            0.95,
            "セル数: " + ", ".join(f"{m} {v}" for m, v in cnt.items()),
            transform=ax.transAxes,
            fontsize=7,
            color=INK2,
            va="top",
        )
        fig.tight_layout()
        fig.savefig(FIG / f"f6_resid_map_m{name}.png", dpi=130)
        plt.close(fig)
        # rhs b の分布
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
        for k, bn in enumerate(("u 運動量", "v 運動量", "連続")):
            rb = np.abs(b[k * n : (k + 1) * n]).reshape(nx, ny)
            vmax = rb.max()
            im = imshow_field(
                axs[k],
                np.maximum(rb, vmax * 1e-6),
                ext,
                f"|R(x_stokes)| の {bn} 成分（{name} kg/s）",
                norm=LogNorm(vmax * 1e-5, vmax),
            )
            stair_overlay(axs[k], d["stair"], (0, 0.6, 0, 0.35))
            plt.colorbar(im, ax=axs[k], fraction=0.03)
        fig.tight_layout()
        fig.savefig(FIG / f"f6b_rhs_map_m{name}.png", dpi=130)
        plt.close(fig)


def fig_histories():
    runs = [
        ("trama_m0.15-dx1.5-sou-jfnk_simple.yaml", "蛇行 内部ポート 0.15（本番）", CAT[1]),
        ("trama_orig-wall-m0.15-dx1.5-sou-jfnk_simple.yaml", "壁ポート 0.15", CAT[3]),
        ("trama_ortho-interior-m0.15-dx1.5-sou-jfnk_simple.yaml", "斜め区間を直交化 0.15", CAT[4]),
        ("trama_orig-interior-m0.15-dx1.5-fou-jfnk_simple.yaml", "残差も 1 次風上 0.15", CAT[6]),
        ("trama_orig-interior-m0.15-dx1.5-sou-jfnk.yaml", "前処理 LU(J1) PARDISO 0.15", CAT[2]),
        ("trama_m0.0015-dx1.5-sou-jfnk_simple.yaml", "内部ポート 0.0015", CAT[0]),
        ("trama_orig-wall-m0.0015-dx1.5-sou-jfnk_simple.yaml", "壁ポート 0.0015", CAT[5]),
    ]
    straight = [
        (
            f"trama_straight{a}-interior-m0.15-dx1.5-sou-jfnk_simple.yaml",
            f"直線流路 {a}°, 0.15",
            CAT[k],
        )
        for k, a in enumerate((0, 15, 30, 45))
    ]
    for fname, group in (("f7_histories.png", runs), ("f7b_straight.png", straight)):
        avail = [(f, nm, c) for f, nm, c in group if (RES / f).exists()]
        if not avail:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
        for f, nm, c in avail:
            y = yaml.safe_load((RES / f).read_text())
            rr = np.array(y["steady_residual_history"]) / (y["steady_residual_history"][0] / 1.0)
            axs[0].plot(
                rr,
                color=c,
                lw=1.6,
                marker="o",
                ms=2.5,
                label=f"{nm}  {'収束' if y['converged'] else '未収束: ' + y['reason']} it={y['n_iter']}",
            )
            axs[1].plot(y["cfl_history"], color=c, lw=1.6)
        axs[0].set_yscale("log")
        axs[0].set_title("定常残差 |R|/|R(Stokes)|（Newton 反復）", loc="left", color=INK)
        axs[0].axhline(1e-6, color=MUTED, lw=0.8, ls="--")
        axs[0].legend(fontsize=7.5)
        axs[0].set_xlabel("Newton 反復")
        axs[1].set_yscale("log")
        axs[1].set_title("CFL（SER）", loc="left", color=INK)
        axs[1].set_xlabel("Newton 反復")
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    for fn in (
        fig_geometry,
        fig_regime,
        fig_spectrum,
        fig_gmres,
        fig_ritz,
        fig_residual_maps,
        fig_histories,
    ):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("FAILED", fn.__name__, exc)
