"""壁セル版 nsb の時間平均場を OpenFOAM（walls 変種）の fieldAverage と突き合わせる.

0.15 kg/s の蛇行流路には定常解が無い（status-47）ので、比べるのは瞬間場ではなく
**時間平均場**にする。両者とも
  - 流路だけを解く（nsb は h <= h_solid を壁セルに、OpenFOAM は subsetMesh で刳り抜く）
  - 階段状の側壁は no-slip
  - 隙間の抗力は 12μ/h²（nsb は Brinkman 項、OpenFOAM は DarcyForchheimer）
なので、残る違いはポートの与え方（nsb はセル内の質量ソース／圧力シンク、OpenFOAM は
円板を刳り抜いた実パッチ）と時間平均の窓だけ。ポート周りは比較から外す。

    python experiments/nsb/trama_solid_compare.py \
        --nsb experiments/nsb/results/trama_SOLID-unsteady-dx1.5-dt1ms_fields.npz \
        --of /tmp/.../of-trama/walls-t2
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

from foam_io import latest_time, read_internal_field, run_of  # noqa: E402
from trama_case import load_trama, make_trama_h  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402
from trama_of_compare import to_grid  # noqa: E402


def load_of_mean(case: Path, *, of_mem: str = "4g") -> dict[str, np.ndarray]:
    """最新時刻の UMean / pMean / UPrime2Mean とセル中心を読む."""
    t = latest_time(str(case))
    if not (case / t / "C").exists():
        import os

        os.environ["OF_MEM"] = of_mem
        run_of(
            str(case),
            "postProcess",
            "-func",
            "writeCellCentres",
            "-latestTime",
            log=str(case / "log.post"),
        )
        t = latest_time(str(case))
    need = ("UMean", "pMean", "UPrime2Mean")
    missing = [f for f in need if not (case / t / f).exists()]
    if missing:
        raise FileNotFoundError(f"{case}/{t} に {missing} がありません（fieldAverage が未出力）")
    return {
        "t": t,
        "C": read_internal_field(str(case / t / "C")),
        "U": read_internal_field(str(case / t / "UMean")),
        "p": read_internal_field(str(case / t / "pMean")),
        "R": read_internal_field(str(case / t / "UPrime2Mean")),
    }


def arclength(geo, x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, float]:
    """比較セルの、流路中心線に沿った弧長 s [m] と全長を返す（最寄りの区間へ射影）."""
    poly = geo.polyline
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s0 = np.r_[0.0, np.cumsum(seg_len)]
    pts = np.stack([x[keep], y[keep]], axis=1)
    best_s = np.full(len(pts), np.nan)
    best_d = np.full(len(pts), np.inf)
    for k in range(len(seg)):
        a0, d = poly[k], seg[k] / seg_len[k]
        tp = np.clip((pts - a0) @ d, 0.0, seg_len[k])
        dist = np.linalg.norm(pts - (a0 + tp[:, None] * d), axis=1)
        upd = dist < best_d
        best_d[upd], best_s[upd] = dist[upd], s0[k] + tp[upd]
    return best_s, float(s0[-1])


def summarize(a: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    v = a[keep]
    return {
        "mean": float(v.mean()),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsb", required=True, help="nsb の *_fields.npz（mean_u/mean_v/mean_p 入り）")
    ap.add_argument("--of", required=True, help="OpenFOAM の非定常ケース")
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_solid_compare.json")
    ap.add_argument("--exclude-cells", type=float, default=3.0, help="ポート周りの除外 [セル]")
    ap.add_argument("--arc-bins", type=int, default=16, help="流路に沿った分割数")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    geo = load_trama(a.pattern)
    z = np.load(a.nsb)
    if "mean_u" not in z:
        raise KeyError(f"{a.nsb} に時間平均場がありません（--avg-start を付けて回す）")
    un, vn, pn = z["mean_u"], z["mean_v"], z["mean_p"]
    rn_u, rn_v = z["rms_u"], z["rms_v"]
    nx, ny = un.shape
    case = Path(a.of)
    spec = json.loads((case / "case.json").read_text())
    if (spec["nx"], spec["ny"]) != (nx, ny):
        raise ValueError(f"格子が違う: nsb {un.shape} vs OpenFOAM {(spec['nx'], spec['ny'])}")
    f = load_of_mean(case)
    g = to_grid(f, nx, ny, geo.lx, geo.ly)
    # UPrime2Mean の対角 (xx, yy) → 変動 RMS
    ci = f["C"]
    i = np.rint(ci[:, 0] / (geo.lx / nx) - 0.5).astype(int)
    j = np.rint(ci[:, 1] / (geo.ly / ny) - 0.5).astype(int)
    ro_u = np.full((nx, ny), np.nan)
    ro_v = np.full((nx, ny), np.nan)
    ro_u[i, j] = np.sqrt(np.maximum(f["R"][:, 0], 0.0))
    ro_v[i, j] = np.sqrt(np.maximum(f["R"][:, 3], 0.0))

    dx = geo.lx / nx
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * geo.ly / ny
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    h = make_trama_h(
        geo,
        nx,
        ny,
        float(np.sqrt(12.0 / spec["d_channel"])),
        float(np.sqrt(12.0 / spec["d_blocked"])),
    )
    keep = (h > 1e-4) & g["mask"] & np.isfinite(un)
    for cx, cy in (geo.inlet, geo.outlet):
        keep &= np.hypot(X - cx, Y - cy) > spec["port_radius"] + a.exclude_cells * dx

    uo, vo = g["u"], g["v"]
    po = g["p"] * spec["rho"]

    # 入口→出口の圧力ヘッド: ポート円周のすぐ外の輪帯（両者で同じ定義。OpenFOAM は円板を
    # 刳り抜いてあるので円板内にセルが無く、nsb の体積ソースとも直接は比べられない）
    r0 = spec["port_radius"] + a.exclude_cells * dx
    ring = {}
    for name, (cx, cy) in (("in", geo.inlet), ("out", geo.outlet)):
        d = np.hypot(X - cx, Y - cy)
        ring[name] = (h > 1e-4) & g["mask"] & np.isfinite(un) & (d > r0) & (d < r0 + 2.0 * dx)
    head_nsb = float(pn[ring["in"]].mean() - pn[ring["out"]].mean())
    head_of = float(po[ring["in"]].mean() - po[ring["out"]].mean())
    sp_n, sp_o = np.hypot(un, vn), np.hypot(uo, vo)
    rms_n, rms_o = np.hypot(rn_u, rn_v), np.hypot(ro_u, ro_v)
    ref = float(np.sqrt((sp_n[keep] ** 2).sum()))
    dif = float(np.sqrt(((un[keep] - uo[keep]) ** 2 + (vn[keep] - vo[keep]) ** 2).sum()))
    dpn = pn[keep] - pn[keep].mean()
    dpo = po[keep] - po[keep].mean()
    out = {
        "of_time": f["t"],
        "nsb_avg_window": [float(v) for v in z["avg_window"]] if "avg_window" in z else None,
        "n_cells": int(keep.sum()),
        "l2_rel_mean_velocity": dif / ref,
        "l2_rel_mean_pressure": float(np.sqrt(((dpn - dpo) ** 2).sum()) / np.sqrt((dpn**2).sum())),
        "mean_speed_nsb": summarize(sp_n, keep),
        "mean_speed_of": summarize(sp_o, keep),
        "rms_nsb": summarize(rms_n, keep),
        "rms_of": summarize(rms_o, keep),
        "turbulence_intensity_nsb": float(rms_n[keep].mean() / sp_n[keep].mean()),
        "turbulence_intensity_of": float(rms_o[keep].mean() / sp_o[keep].mean()),
        "dp_span_nsb_pa": float(pn[keep].max() - pn[keep].min()),
        "dp_span_of_pa": float(po[keep].max() - po[keep].min()),
        "head_ring_nsb_pa": head_nsb,
        "head_ring_of_pa": head_of,
        "head_ring_cells": [int(ring["in"].sum()), int(ring["out"].sum())],
    }
    # ---- 流路に沿った差の分布（ポートの与え方の違いがどこまで届くか）----
    sarc, path_len = arclength(geo, X, Y, keep)
    du, dv = (un - uo)[keep], (vn - vo)[keep]
    spn_k, rn_k, ro_k = sp_n[keep], rms_n[keep], rms_o[keep]
    edges = np.linspace(0.0, path_len, a.arc_bins + 1)
    prof: list[dict[str, float]] = []
    for k in range(a.arc_bins):
        m = (sarc >= edges[k]) & (sarc < edges[k + 1])
        if m.sum() < 50:
            continue
        prof.append(
            {
                "s_m": float(0.5 * (edges[k] + edges[k + 1])),
                "n": int(m.sum()),
                "l2_rel": float(
                    np.sqrt((du[m] ** 2 + dv[m] ** 2).sum()) / np.sqrt((spn_k[m] ** 2).sum())
                ),
                "rms_nsb": float(rn_k[m].mean()),
                "rms_of": float(ro_k[m].mean()),
                "speed_nsb": float(spn_k[m].mean()),
                "speed_of": float(sp_o[keep][m].mean()),
            }
        )
    sc = np.array([q["s_m"] for q in prof])
    rel_arc = np.array([q["l2_rel"] for q in prof])
    slope = np.polyfit(sc, np.log(rel_arc), 1)[0]
    decay = float(-1.0 / slope) if slope < 0 else float("inf")
    # 抗力長 L_drag = ρ u h²/(12 μ)（§11.4 の N の分子）
    # case.json は d = 12/h² と体積流量 Q [m³/s]、動粘度 nu を持つ
    h_ch = float(np.sqrt(12.0 / spec["d_channel"]))
    mu = spec["nu"] * spec["rho"]
    u_mean = spec["flow_rate"] / (geo.width * h_ch)
    l_drag = spec["rho"] * u_mean * h_ch**2 / (12.0 * mu)
    out["arc_profile"] = prof
    out["arc_decay_length_m"] = decay
    out["l_drag_m"] = float(l_drag)
    out["decay_over_l_drag"] = decay / float(l_drag)
    out["path_length_m"] = path_len

    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    ext = (0, geo.lx * 1e3, 0, geo.ly * 1e3)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4))
    vmax = float(np.nanmax(np.where(keep, sp_n, np.nan)))
    rmax = float(np.nanmax(np.where(keep, rms_o, np.nan)))
    panels = (
        (axes[0, 0], sp_n, "nsb 壁セル 時間平均 |U|", "viridis", vmax),
        (axes[0, 1], sp_o, "OpenFOAM walls 時間平均 |U|", "viridis", vmax),
        (
            axes[0, 2],
            np.abs(sp_n - sp_o),
            f"平均速度の差（L2 相対 {out['l2_rel_mean_velocity']:.1%}）",
            "magma",
            0.2 * vmax,
        ),
        (axes[1, 0], rms_n, "nsb 変動 RMS", "inferno", rmax),
        (axes[1, 1], rms_o, "OpenFOAM 変動 RMS", "inferno", rmax),
        (axes[1, 2], np.abs(rms_n - rms_o), "変動 RMS の差", "magma", 0.5 * rmax),
    )
    for ax, arr, ttl, cmap, vm in panels:
        im = ax.imshow(
            np.where(keep, arr, np.nan).T,
            origin="lower",
            extent=ext,
            cmap=cmap,
            aspect="equal",
            vmin=0,
            vmax=vm,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="m/s")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.suptitle(
        "0.15 kg/s（N = 13.3、定常解なし）の時間平均場: nsb 壁セル版 ↔ OpenFOAM", fontsize=13
    )
    fig.tight_layout()
    out_png = figs / "f16_solid_transient.png"
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.4))
    sm = sc * 1e3
    ax2[0].semilogy(sm, rel_arc, "o-", color="#c0392b", label="時間平均速度の差の L2 相対")
    ax2[0].semilogy(
        sm,
        rel_arc[0] * np.exp(-(sc - sc[0]) / decay),
        "--",
        color="#555",
        label=f"exp(−s / {decay * 1e3:.0f} mm) 当てはめ",
    )
    ax2[0].axvline(l_drag * 1e3, color="#2c6fa8", lw=1.2, ls=":")
    ax2[0].annotate(
        f"抗力長 L_drag = {l_drag * 1e3:.0f} mm",
        xy=(l_drag * 1e3, rel_arc.max()),
        xytext=(l_drag * 1e3 + 60, rel_arc.max()),
        color="#2c6fa8",
        fontsize=9,
    )
    ax2[0].set_xlabel("入口ポートからの流路に沿った距離 s [mm]")
    ax2[0].set_ylabel("時間平均速度の L2 相対差")
    ax2[0].set_title("ポートの与え方の違いは抗力長で消える", fontsize=11)
    ax2[0].legend(fontsize=9)
    ax2[0].grid(alpha=0.3)

    ax2[1].plot(
        [q["s_m"] * 1e3 for q in prof], [q["rms_nsb"] for q in prof], "o-", label="nsb 壁セル"
    )
    ax2[1].plot(
        [q["s_m"] * 1e3 for q in prof], [q["rms_of"] for q in prof], "s-", label="OpenFOAM walls"
    )
    ax2[1].set_xlabel("入口ポートからの流路に沿った距離 s [mm]")
    ax2[1].set_ylabel("変動 RMS √(u′² + v′²) [m/s]")
    ax2[1].set_title("変動が立ち上がる位置が違う（総量は同じ）", fontsize=11)
    ax2[1].legend(fontsize=9)
    ax2[1].grid(alpha=0.3)
    fig2.tight_layout()
    out2 = figs / "f17_solid_arc.png"
    fig2.savefig(out2, dpi=110)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
