"""nsb の解と OpenFOAM の解を同じ格子で突き合わせる（オラクル検算）.

定常解が両方で求まる流量（0.005 kg/s、N = 0.44）で、nsb（内部ポート、リミター凍結、
継続法）と OpenFOAM（`porous` 変種 = 閉塞を抗力の栓で表した nsb と同じ離散化）の
速度場・圧力場を比べる。

ポートの与え方だけは違う（nsb はセル内の質量ソース／圧力シンク、OpenFOAM は円板を
刳り抜いた実パッチ）ので、両ポートから 半径 + 3 セル 以内は比較から外す。
圧力は基準が違う（nsb の出口シンクは q = C·p で p ≈ ṁ/(h·C) の下駄を履く）ので、
比較マスク上の平均を引いてから比べる。

    python experiments/nsb/trama_of_verify.py --nsb experiments/nsb/results/trama_X_fields.npz \
        --of /tmp/of-trama/porous-m0005
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
from trama_of_compare import load_of_fields, to_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsb", required=True, help="nsb の *_fields.npz")
    ap.add_argument("--of", required=True, help="OpenFOAM ケース")
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_of_verify.json")
    ap.add_argument("--exclude-cells", type=float, default=3.0, help="ポート周りの除外 [セル]")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    geo = load_trama(a.pattern)
    z = np.load(a.nsb)
    un, vn, pn = z["u"], z["v"], z["p"]
    nx, ny = un.shape
    case = Path(a.of)
    spec = json.loads((case / "case.json").read_text())
    if (spec["nx"], spec["ny"]) != (nx, ny):
        raise ValueError(f"格子が違う: nsb {un.shape} vs OpenFOAM {(spec['nx'], spec['ny'])}")
    g = to_grid(load_of_fields(case), nx, ny, geo.lx, geo.ly)

    dx = geo.lx / nx
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * geo.ly / ny
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    h = make_trama_h(geo, nx, ny, 3.8e-3, 1.0e-5)
    keep = (h > 1e-4) & g["mask"]
    for cx, cy in (geo.inlet, geo.outlet):
        keep &= np.hypot(X - cx, Y - cy) > spec["port_radius"] + a.exclude_cells * dx

    uo, vo = g["u"], g["v"]
    po = g["p"] * spec["rho"]
    sp_n = np.hypot(un, vn)
    sp_o = np.hypot(uo, vo)
    ref = float(np.sqrt((sp_n[keep] ** 2).sum()))
    dif = float(np.sqrt(((un[keep] - uo[keep]) ** 2 + (vn[keep] - vo[keep]) ** 2).sum()))
    dpn = pn[keep] - pn[keep].mean()
    dpo = po[keep] - po[keep].mean()
    out = {
        "n_cells": int(keep.sum()),
        "l2_rel_velocity": dif / ref,
        "l2_rel_pressure": float(np.sqrt(((dpn - dpo) ** 2).sum()) / np.sqrt((dpn**2).sum())),
        "speed_mean_nsb": float(sp_n[keep].mean()),
        "speed_mean_of": float(sp_o[keep].mean()),
        "speed_p90_nsb": float(np.percentile(sp_n[keep], 90)),
        "speed_p90_of": float(np.percentile(sp_o[keep], 90)),
        "speed_max_nsb": float(sp_n[keep].max()),
        "speed_max_of": float(sp_o[keep].max()),
        "dp_span_nsb_pa": float(pn[keep].max() - pn[keep].min()),
        "dp_span_of_pa": float(po[keep].max() - po[keep].min()),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ext = (0, geo.lx * 1e3, 0, geo.ly * 1e3)
    vmax = float(np.nanmax(np.where(keep, sp_n, np.nan)))
    for ax, arr, ttl, cmap, vm in (
        (axes[0], np.where(keep, sp_n, np.nan), "nsb 流速", "viridis", vmax),
        (axes[1], np.where(keep, sp_o, np.nan), "OpenFOAM 流速", "viridis", vmax),
        (
            axes[2],
            np.where(keep, np.abs(sp_n - sp_o), np.nan),
            f"差の大きさ（L2 相対 {out['l2_rel_velocity']:.2%}）",
            "magma",
            0.1 * vmax,
        ),
    ):
        im = ax.imshow(
            arr.T, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmin=0, vmax=vm
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="m/s")
        ax.set_title(ttl)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.suptitle("定常解がある流量（0.005 kg/s、N = 0.44）での突き合わせ", fontsize=13)
    fig.tight_layout()
    out_png = figs / "f15_of_verify.png"
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
