"""助走脚（variant="lead" で継ぎ足した入口の直線区間）の横断面プロファイルを nsb ↔ OpenFOAM で比べる.

この区間は変動 RMS がほぼ 0 なので、時間平均場の比較が統計ではなく**決定論的な場の比較**になる。
刳り抜きポートは流路幅 34.5 mm に直径 28.5 mm の円板を置くので、流れは両脇 3 mm の隙間を
通って壁沿いの 2 本の噴流になる。その噴流の形と減衰がそのままポート境界条件の再現性になる。

    python experiments/nsb/trama_lead_profile.py \
        --nsb experiments/nsb/results/trama_LEAD-carve-m01-B_fields.npz \
        --of /tmp/.../of-trama/walls-lead2 --figs experiments/nsb/results/trama_figs
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

from trama_case import load_trama  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402
from trama_of_compare import to_grid  # noqa: E402
from trama_solid_compare import load_of_mean  # noqa: E402


def sample_line(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, lx: float, ly: float):
    """格子配列を点列 (xs, ys) で最近傍サンプリング（範囲外・NaN は NaN）."""
    nx, ny = arr.shape
    i = np.floor(xs / (lx / nx)).astype(int)
    j = np.floor(ys / (ly / ny)).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    out = np.full(xs.shape, np.nan)
    out[ok] = arr[i[ok], j[ok]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsb", required=True)
    ap.add_argument("--of", required=True)
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_lead_profile.json")
    ap.add_argument(
        "--stations", default="25,60,100,150", help="入口ポート中心からの距離 [mm]（カンマ区切り）"
    )
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    case = Path(a.of)
    spec = json.loads((case / "case.json").read_text())
    geo = load_trama(a.pattern, variant=spec.get("geo_variant", "orig"))
    if geo.variant != "lead":
        raise ValueError(f"助走脚のあるケース（variant=lead）用です: {geo.variant!r}")
    z = np.load(a.nsb)
    if "mean_u" not in z:
        raise KeyError(f"{a.nsb} に時間平均場がありません（--avg-start を付けて回す）")
    un, vn = z["mean_u"], z["mean_v"]
    nx, ny = un.shape
    f = load_of_mean(case)
    g = to_grid(f, nx, ny, geo.lx, geo.ly)
    sp_n = np.hypot(un, vn)
    sp_o = np.hypot(g["u"], g["v"])
    sp_n = np.where(g["mask"], sp_n, np.nan)
    sp_o = np.where(g["mask"], sp_o, np.nan)

    # 助走脚 = 中心線の第 1 区間（入口ポート → 曲がり角）
    p0, p1 = geo.polyline[0], geo.polyline[1]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([d[1], -d[0]])  # 流れ方向の右手法線（t > 0 が流れの右側）
    leg = float(np.linalg.norm(p1 - p0))
    w = geo.width
    dx = geo.lx / nx
    tt = np.linspace(-w / 2 + 0.5 * dx, w / 2 - 0.5 * dx, int(w / dx))

    stations = [float(v) * 1e-3 for v in a.stations.split(",")]
    prof = []
    fig, axes = plt.subplots(1, len(stations) + 1, figsize=(4.0 * (len(stations) + 1), 4.0))
    for ax, s in zip(axes[:-1], stations, strict=True):
        q = p0 + d * s
        xs, ys = q[0] + n[0] * tt, q[1] + n[1] * tt
        yn = sample_line(sp_n, xs, ys, geo.lx, geo.ly)
        yo = sample_line(sp_o, xs, ys, geo.lx, geo.ly)
        ok = np.isfinite(yn) & np.isfinite(yo)
        ax.plot(tt * 1e3, yn, "o-", ms=3, color="#c0392b", label="nsb 刳り抜きポート")
        ax.plot(tt * 1e3, yo, "s--", ms=3, color="#1f6fb4", label="OpenFOAM walls")
        ax.set_title(f"ポートから s = {s * 1e3:.0f} mm", fontsize=11)
        ax.set_xlabel("横位置 [mm]（+ が流れの右手）")
        ax.set_ylabel("時間平均流速 |Ū| [m/s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        prof.append(
            {
                "s_mm": s * 1e3,
                "peak_nsb": float(np.nanmax(yn)),
                "peak_of": float(np.nanmax(yo)),
                "core_nsb": float(np.nanmin(yn[ok])),
                "core_of": float(np.nanmin(yo[ok])),
                "l2_rel": float(
                    np.sqrt(((yn[ok] - yo[ok]) ** 2).sum()) / np.sqrt((yo[ok] ** 2).sum())
                ),
            }
        )
    # 助走脚に沿ったピークの減衰
    ss = np.arange(15.0e-3, leg - 5.0e-3, dx)
    pk_n, pk_o = [], []
    for s in ss:
        q = p0 + d * s
        xs, ys = q[0] + n[0] * tt, q[1] + n[1] * tt
        pk_n.append(np.nanmax(sample_line(sp_n, xs, ys, geo.lx, geo.ly)))
        pk_o.append(np.nanmax(sample_line(sp_o, xs, ys, geo.lx, geo.ly)))
    ax = axes[-1]
    ax.plot(ss * 1e3, pk_n, "-", color="#c0392b", label="nsb")
    ax.plot(ss * 1e3, pk_o, "--", color="#1f6fb4", label="OpenFOAM")
    u_mean = spec["flow_rate"] / (w * float(np.sqrt(12.0 / spec["d_channel"])))
    ax.axhline(u_mean, color="#555", lw=1.0, ls=":", label=f"流路平均 {u_mean:.2f} m/s")
    ax.set_xlabel("ポートからの距離 s [mm]")
    ax.set_ylabel("横断面のピーク流速 [m/s]")
    ax.set_title("噴流のピークは抗力長で減衰する", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"助走脚の横断面プロファイル（{spec['flow_rate'] * spec['rho']:g} kg/s、変動 RMS ≈ 0 の区間）",
        fontsize=13,
    )
    fig.tight_layout()
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    out_png = figs / "f18_lead_profile.png"
    fig.savefig(out_png, dpi=110)

    out = {
        "of_time": f["t"],
        "leg_length_mm": leg * 1e3,
        "u_mean": float(u_mean),
        "stations": prof,
        "peak_decay_nsb": [float(v) for v in pk_n],
        "peak_decay_of": [float(v) for v in pk_o],
        "s_mm": [float(v) * 1e3 for v in ss],
    }
    print(f"助走脚 {leg * 1e3:.0f} mm、流路平均 {u_mean:.3f} m/s、OF 時刻 {f['t']}")
    print(" s [mm]   ピーク nsb / OF (比)      淀み核 nsb / OF     断面の L2 相対差")
    for q in prof:
        print(
            f" {q['s_mm']:6.0f}   {q['peak_nsb']:.3f} / {q['peak_of']:.3f}"
            f" ({q['peak_nsb'] / q['peak_of']:.3f})   {q['core_nsb']:.3f} / {q['core_of']:.3f}"
            f"      {q['l2_rel']:.2%}"
        )
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
