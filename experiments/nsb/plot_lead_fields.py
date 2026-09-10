"""助走脚つき trama の時間平均場を図にする（流速・圧力・変動 RMS）と、剥離剪断層が崩れる位置を測る.

図は 3 枚:
  f20_lead_fields.png  全域の 2×3（流速 nsb / OF / 差、圧力 nsb / OF / 差）
  f22_lead_zoom.png    曲がりを出た直後の拡大（流速・圧力・RMS を nsb / OF で並べる）
  f21_lead_onset.png   断面の最大変動 RMS を x に沿って描き、しきい値を超える位置を比べる

    python experiments/nsb/plot_lead_fields.py \
        --nsb experiments/nsb/results/trama_LEAD-carve-m01-B_fields.npz --of /tmp/.../walls-lead2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "extruder"))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import load_trama, make_trama_h  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402
from trama_of_compare import to_grid  # noqa: E402
from trama_solid_compare import load_of_mean  # noqa: E402

# 上側水平区間（90 度の曲がりを出た直後）。剥離泡と剪断層の崩れがここに出る
ZOOM = (90.0, 520.0, 255.0, 325.0)  # x0, x1, y0, y1 [mm]
BAND = (0.278, 0.320)  # 遷移位置を測る y 帯 [m]
ONSET_X = (0.132, 0.300)  # 測る x の範囲 [m]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsb", required=True)
    ap.add_argument("--of", required=True)
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_lead_onset.json")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    case = Path(a.of)
    spec = json.loads((case / "case.json").read_text())
    geo = load_trama(a.pattern, variant=spec.get("geo_variant", "orig"))
    z = np.load(a.nsb)
    un, vn, pn = z["mean_u"], z["mean_v"], z["mean_p"]
    rn = np.hypot(z["rms_u"], z["rms_v"])
    nx, ny = un.shape
    f = load_of_mean(case)
    g = to_grid(f, nx, ny, geo.lx, geo.ly)
    uo, vo = g["u"], g["v"]
    po = g["p"] * spec["rho"]
    dx, dy = geo.lx / nx, geo.ly / ny
    ci = f["C"]
    i = np.rint(ci[:, 0] / dx - 0.5).astype(int)
    j = np.rint(ci[:, 1] / dy - 0.5).astype(int)
    ro = np.full((nx, ny), np.nan)
    ro[i, j] = np.sqrt(np.maximum(f["R"][:, 0], 0.0) + np.maximum(f["R"][:, 3], 0.0))

    h = make_trama_h(
        geo,
        nx,
        ny,
        float(np.sqrt(12.0 / spec["d_channel"])),
        float(np.sqrt(12.0 / spec["d_blocked"])),
    )
    fluid = (h > 1e-4) & g["mask"]
    keep = fluid & np.isfinite(un) & np.isfinite(uo)
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    for cx, cy in (geo.inlet, geo.outlet):
        keep &= np.hypot(X - cx, Y - cy) > spec["port_radius"] + 3.0 * dx

    def m(arr: np.ndarray) -> np.ndarray:
        return np.where(keep, arr, np.nan).T

    spn, spo = np.hypot(un, vn), np.hypot(uo, vo)
    ext = (0.0, geo.lx * 1e3, 0.0, geo.ly * 1e3)
    vmax = float(np.nanmax(m(spn)))
    pmin, pmax = float(np.nanmin(m(po))), float(np.nanmax(m(po)))
    rmax = float(np.nanmax(m(ro)))
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)

    def panel(ax, arr, ttl, cmap, lo, hi, unit):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmin=lo, vmax=hi)
        plt.colorbar(im, ax=ax, shrink=0.78, label=unit)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    # --- f20: 全域の流速と圧力 ---
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.6))
    panel(ax[0, 0], m(spn), "nsb 時間平均 流速 |Ū|", "viridis", 0, vmax, "m/s")
    panel(ax[0, 1], m(spo), "OpenFOAM 時間平均 流速 |Ū|", "viridis", 0, vmax, "m/s")
    panel(ax[0, 2], m(np.abs(spn - spo)), "流速の差 |Δ|", "magma", 0, 0.25 * vmax, "m/s")
    panel(ax[1, 0], m(pn), "nsb 時間平均 圧力 p̄（出口 = 0）", "coolwarm", pmin, pmax, "Pa")
    panel(ax[1, 1], m(po), "OpenFOAM 時間平均 圧力 p̄", "coolwarm", pmin, pmax, "Pa")
    d = 0.08 * (pmax - pmin)
    panel(ax[1, 2], m(pn - po), "圧力の差", "RdBu_r", -d, d, "Pa")
    fig.suptitle(
        f"trama 助走脚つき {spec['flow_rate'] * spec['rho']:g} kg/s の時間平均: 流速と圧力",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(figs / "f20_lead_fields.png", dpi=105)

    # --- f22: 曲がり直後の拡大 ---
    x0, x1, y0, y1 = ZOOM
    rows = (
        (spn, spo, "時間平均 流速 |Ū|", "viridis", 0.0, vmax, "m/s"),
        (pn, po, "時間平均 圧力 p̄", "coolwarm", pmin, pmax, "Pa"),
        (rn, ro, "変動 RMS √(u′²+v′²)", "inferno", 0.0, rmax, "m/s"),
    )
    fig3, ax3 = plt.subplots(3, 2, figsize=(16, 9.0))
    for r, (an, ao, ttl, cmap, lo, hi, unit) in enumerate(rows):
        for c, (arr, who) in enumerate(((an, "nsb"), (ao, "OpenFOAM"))):
            axx = ax3[r, c]
            im = axx.imshow(
                m(arr), origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, aspect="equal"
            )
            plt.colorbar(im, ax=axx, shrink=0.9, label=unit)
            axx.set_xlim(x0, x1)
            axx.set_ylim(y0, y1)
            axx.set_title(f"{who} — {ttl}", fontsize=11)
            axx.set_xlabel("x [mm]")
    fig3.suptitle("曲がりを出た直後（上側水平区間）— 剪断層が崩れる位置", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(figs / "f22_lead_zoom.png", dpi=105)

    # --- f21: 剥離剪断層が崩れ始める位置 ---
    band = (np.arange(ny) * dy > BAND[0]) & (np.arange(ny) * dy < BAND[1])
    xs, r_n, r_o = [], [], []
    for ii in range(int(ONSET_X[0] / dx), int(ONSET_X[1] / dx)):
        sel = fluid[ii, :] & band
        if sel.sum() < 10:
            continue
        xs.append((ii + 0.5) * dx * 1e3)
        r_n.append(float(np.nanmax(rn[ii, sel])))
        r_o.append(float(np.nanmax(ro[ii, sel])))
    xs, r_n, r_o = np.array(xs), np.array(r_n), np.array(r_o)
    u_mean = spec["flow_rate"] / (geo.width * float(np.sqrt(12.0 / spec["d_channel"])))

    def onset(r: np.ndarray, thr: float) -> float:
        k = np.where(r > thr)[0]
        return float(xs[k[0]]) if k.size else float("nan")

    table = []
    print("剥離剪断層が崩れ始める位置（断面の最大変動 RMS がしきい値を超える x）")
    print(" しきい値            nsb        OF       ずれ")
    for frac in (0.05, 0.10, 0.15, 0.20, 0.25):
        thr = frac * u_mean
        p_n, p_o = onset(r_n, thr), onset(r_o, thr)
        table.append({"frac": frac, "threshold": thr, "nsb_mm": p_n, "of_mm": p_o})
        print(
            f"  {frac * 100:2.0f}% u_mean ({thr:.3f})   {p_n:6.1f}    {p_o:6.1f}   {p_n - p_o:+6.1f} mm"
        )

    fig2, ax2 = plt.subplots(figsize=(11, 4.4))
    ax2.plot(xs, r_n, color="#c0392b", lw=2, label="nsb 刳り抜きポート")
    ax2.plot(xs, r_o, color="#1f6fb4", lw=2, ls="--", label="OpenFOAM walls")
    for frac in (0.10, 0.20):
        ax2.axhline(frac * u_mean, color="#999", lw=1, ls=":")
        ax2.annotate(f"{frac * 100:.0f}% u_mean", (xs[0] + 2, frac * u_mean + 0.005), fontsize=8)
    pn10, po10 = onset(r_n, 0.1 * u_mean), onset(r_o, 0.1 * u_mean)
    for x, c, lab in ((po10, "#1f6fb4", "OF"), (pn10, "#c0392b", "nsb")):
        ax2.axvline(x, color=c, lw=1.4, alpha=0.6)
        ax2.annotate(f"{lab} {x:.0f} mm", (x + 3, 0.9 * r_o.max()), color=c, fontsize=10)
    ax2.annotate(
        "",
        xy=(po10, 0.82 * r_o.max()),
        xytext=(pn10, 0.82 * r_o.max()),
        arrowprops={"arrowstyle": "<->", "color": "#333"},
    )
    ax2.annotate(
        f"ずれ {pn10 - po10:.0f} mm ≈ 流路幅 {(pn10 - po10) / (geo.width * 1e3):.1f} 個ぶん",
        (0.5 * (pn10 + po10), 0.85 * r_o.max()),
        ha="center",
        fontsize=10,
    )
    ax2.set_xlabel(f"x [mm]（90 度の曲がりは x = {ONSET_X[0] * 1e3:.0f} mm）")
    ax2.set_ylabel("断面の最大変動 RMS [m/s]")
    ax2.set_title("剥離剪断層が崩れ始める位置は nsb のほうが下流にずれる", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(figs / "f21_lead_onset.png", dpi=110)

    Path(a.out_json).write_text(
        json.dumps(
            {
                "u_mean": u_mean,
                "onset": table,
                "x_mm": xs.tolist(),
                "rms_max_nsb": r_n.tolist(),
                "rms_max_of": r_o.tolist(),
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {figs}/f20_lead_fields.png, f21_lead_onset.png, f22_lead_zoom.png")


if __name__ == "__main__":
    main()
