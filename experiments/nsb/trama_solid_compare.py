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
    h = make_trama_h(geo, nx, ny, spec.get("h_channel", 3.8e-3), spec.get("h_blocked", 1.0e-5))
    keep = (h > 1e-4) & g["mask"] & np.isfinite(un)
    for cx, cy in (geo.inlet, geo.outlet):
        keep &= np.hypot(X - cx, Y - cy) > spec["port_radius"] + a.exclude_cells * dx

    uo, vo = g["u"], g["v"]
    po = g["p"] * spec["rho"]
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
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    ext = (0, geo.lx * 1e3, 0, geo.ly * 1e3)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4))
    vmax = float(np.nanmax(np.where(keep, sp_n, np.nan)))
    rmax = float(np.nanmax(np.where(keep, rms_o, np.nan)))
    panels = (
        (axes[0, 0], sp_n, "nsb 壁セル ⟨|U|⟩", "viridis", vmax),
        (axes[0, 1], sp_o, "OpenFOAM walls ⟨|U|⟩", "viridis", vmax),
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


if __name__ == "__main__":
    main()
