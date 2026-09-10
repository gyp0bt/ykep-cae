"""非定常（pimpleFoam）の結果をまとめる: 時間平均場・変動場・入口圧の時系列.

定常解が無い流量では「答え」は 1 枚の場ではなく、時間平均場と変動の大きさになる。
`fieldAverage` が書く UMean / pMean / UPrime2Mean を読み、nsb と同じ格子に載せる。

    python experiments/nsb/trama_of_transient.py --case /tmp/of-trama/walls-t \
        --figs experiments/nsb/results/trama_figs
"""

from __future__ import annotations

import argparse
import json
import re
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


def inlet_pressure(log_path: Path, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """入口パッチの面積平均圧 [Pa] の時系列（サンプル間隔は 10 ステップ）."""
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    p = np.array([float(x) for x in re.findall(r"areaAverage\(inlet\) of p = ([-+0-9.eE]+)", txt)])
    t = np.array([float(x) for x in re.findall(r"^Time = ([-+0-9.eE]+)", txt, re.M)])
    # サンプルは 10 ステップごと。時刻は最後の n サンプルに対応づける
    if t.size >= 10 * p.size:
        ts = t[9::10][: p.size]
    else:
        ts = np.linspace(t[0] if t.size else 0.0, t[-1] if t.size else 1.0, p.size)
    n = min(ts.size, p.size)
    return ts[:n], p[:n] * rho


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_of_transient.json")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    case = Path(a.case)
    spec = json.loads((case / "case.json").read_text())
    geo = load_trama(a.pattern, variant=spec.get("geo_variant", "orig"))
    t = latest_time(str(case))
    if not (case / t / "C").exists():
        run_of(
            str(case),
            "postProcess",
            "-func",
            "writeCellCentres",
            "-latestTime",
            log=str(case / "log.post"),
        )
        t = latest_time(str(case))
    need = ["C", "UMean", "pMean", "UPrime2Mean"]
    missing = [f for f in need if not (case / t / f).exists()]
    if missing:
        raise FileNotFoundError(f"{case / t} に {missing} が無い（fieldAverage が書いたか確認）")

    fields = {
        "C": read_internal_field(str(case / t / "C")),
        "U": read_internal_field(str(case / t / "UMean")),
        "p": read_internal_field(str(case / t / "pMean")),
    }
    g = to_grid(fields, spec["nx"], spec["ny"], geo.lx, geo.ly)
    prime = read_internal_field(str(case / t / "UPrime2Mean"))  # (N, 6) 対称テンソル
    rms = np.sqrt(np.maximum(prime[:, 0] + prime[:, 3], 0.0))  # √(u'u' + v'v')
    ci = np.rint(fields["C"][:, 0] / (geo.lx / spec["nx"]) - 0.5).astype(int)
    cj = np.rint(fields["C"][:, 1] / (geo.ly / spec["ny"]) - 0.5).astype(int)
    rms_g = np.full((spec["nx"], spec["ny"]), np.nan)
    rms_g[ci, cj] = rms

    h = make_trama_h(geo, spec["nx"], spec["ny"], 3.8e-3, 1.0e-5)
    chan = (h > 1e-4) & g["mask"]
    speed = np.hypot(g["u"], g["v"])
    p_pa = g["p"] * spec["rho"]

    log = case / "log.pimpleFoam"
    ts, pin = inlet_pressure(log, spec["rho"])
    half = pin[pin.size // 3 :]
    summary = {
        "time": t,
        "speed_max_mean": float(np.nanmax(speed[chan])),
        "speed_mean_channel": float(np.nanmean(speed[chan])),
        "dp_mean_pa": float(np.nanmax(p_pa[chan]) - np.nanmin(p_pa[chan])),
        "p_inlet_mean_pa": float(half.mean()),
        "p_inlet_std_pa": float(half.std()),
        "p_inlet_min_pa": float(half.min()),
        "p_inlet_max_pa": float(half.max()),
        "p_inlet_rel_swing": float(half.std() / half.mean()),
        "rms_max": float(np.nanmax(rms_g[chan])),
        "rms_mean_channel": float(np.nanmean(rms_g[chan])),
        "rms_over_umean": float(np.nanmean(rms_g[chan]) / np.nanmean(speed[chan])),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")

    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    panels = [
        (np.where(chan, speed, np.nan), "時間平均の流速 |Ū|", "viridis", "m/s"),
        (np.where(chan, p_pa, np.nan), "時間平均の圧力 p̄", "coolwarm", "Pa"),
        (np.where(chan, rms_g, np.nan), "速度変動 √(u′²+v′²)", "magma", "m/s"),
    ]
    for ax, (arr, ttl, cmap, unit) in zip(axes.ravel()[:3], panels, strict=True):
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=(0, geo.lx * 1e3, 0, geo.ly * 1e3),
            cmap=cmap,
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, shrink=0.82, label=unit)
        ax.set_title(ttl)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    ax = axes.ravel()[3]
    ax.plot(ts, pin * 1e-3, color="#B24714", lw=1.2)
    ax.axhline(half.mean() * 1e-3, color="#1F6FB4", ls="--", lw=1.2, label="時間平均")
    ax.fill_between(
        ts,
        (half.mean() - half.std()) * 1e-3,
        (half.mean() + half.std()) * 1e-3,
        color="#1F6FB4",
        alpha=0.15,
        label="±1σ",
    )
    ax.set_xlabel("時刻 [s]")
    ax.set_ylabel("入口の面積平均圧 [kPa]")
    ax.set_title(
        f"必要な圧力ヘッドは一定にならない（{half.mean() * 1e-3:.1f} ± {half.std() * 1e-3:.1f} kPa）"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("OpenFOAM pimpleFoam — trama 蛇行流路 0.15 kg/s の時間平均と変動", fontsize=13)
    fig.tight_layout()
    out = figs / "f14_of_transient.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
