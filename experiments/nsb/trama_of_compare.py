"""OpenFOAM の trama 解を nsb の格子に載せ替えて突き合わせ、図を作る.

`run_trama_of.py` が回した `porous` / `walls` の最新時刻を読み、セル中心座標から
nsb と同じ (nx, ny) 配列に戻す（両者とも一様直交格子なので添字に直せる）。

    python experiments/nsb/trama_of_compare.py --work /tmp/of-trama \
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


def load_of_fields(case: Path, *, of_mem: str = "4g") -> dict[str, np.ndarray]:
    """最新時刻の U, p とセル中心を読む（C が無ければ postProcess で作る）."""
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
    return {
        "t": t,
        "C": read_internal_field(str(case / t / "C")),
        "U": read_internal_field(str(case / t / "U")),
        "p": read_internal_field(str(case / t / "p")),
    }


def to_grid(
    fields: dict[str, np.ndarray], nx: int, ny: int, lx: float, ly: float
) -> dict[str, np.ndarray]:
    """セル中心座標 → (nx, ny) 配列（欠けたセルは NaN）."""
    c = fields["C"]
    i = np.rint(c[:, 0] / (lx / nx) - 0.5).astype(int)
    j = np.rint(c[:, 1] / (ly / ny) - 0.5).astype(int)
    if i.min() < 0 or i.max() >= nx or j.min() < 0 or j.max() >= ny:
        raise ValueError("セル中心が格子の外: 格子幅か領域が食い違っている")
    out = {}
    for key, ncomp in (("u", 0), ("v", 1)):
        a = np.full((nx, ny), np.nan)
        a[i, j] = fields["U"][:, ncomp]
        out[key] = a
    a = np.full((nx, ny), np.nan)
    a[i, j] = fields["p"]
    out["p"] = a
    m = np.zeros((nx, ny), dtype=bool)
    m[i, j] = True
    out["mask"] = m
    return out


def residual_history(log_path: Path) -> dict[str, np.ndarray]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    steps, ux, uy, pp = [], [], [], []
    for m in re.finditer(r"^Time = (\d+)$(.*?)(?=^Time = |\Z)", text, re.M | re.S):
        body = m.group(2)
        got = {}
        for f in ("Ux", "Uy", "p"):
            mm = re.search(rf"Solving for {f}, Initial residual = ([-+0-9.eE]+)", body)
            if mm:
                got[f] = float(mm.group(1))
        if len(got) == 3:
            steps.append(int(m.group(1)))
            ux.append(got["Ux"])
            uy.append(got["Uy"])
            pp.append(got["p"])
    return {
        "step": np.array(steps),
        "Ux": np.array(ux),
        "Uy": np.array(uy),
        "p": np.array(pp),
    }


def transient_probes(log_path: Path) -> dict[str, np.ndarray]:
    """pimpleFoam ログから時刻・Δt・最大流速・出口平均圧を拾う（振動の証拠）."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    times, dts, umax, pout = [], [], [], []
    cur_t = None
    cur_dt = None
    for line in text.splitlines():
        m = re.match(r"^Time = ([-+0-9.eE]+)", line)
        if m:
            cur_t = float(m.group(1))
            continue
        m = re.match(r"^deltaT = ([-+0-9.eE]+)", line)
        if m:
            cur_dt = float(m.group(1))
            continue
        m = re.search(r"maxMag\(U\) = ([-+0-9.eE]+)", line)
        if m and cur_t is not None:
            times.append(cur_t)
            dts.append(cur_dt if cur_dt is not None else np.nan)
            umax.append(float(m.group(1)))
            continue
        m = re.search(r"areaAverage\(inlet\) of p = ([-+0-9.eE]+)", line)
        if m and umax:
            while len(pout) < len(umax) - 1:
                pout.append(np.nan)
            pout.append(float(m.group(1)))
    n = min(len(times), len(umax))
    while len(pout) < n:
        pout.append(np.nan)
    return {
        "t": np.array(times[:n]),
        "dt": np.array(dts[:n]),
        "umax": np.array(umax[:n]),
        "p_in": np.array(pout[:n]),
    }


def summarize(g: dict[str, np.ndarray], spec: dict, h: np.ndarray) -> dict[str, float]:
    """代表量: 最大流速、流路内の平均流速、圧力差（運動学的 → Pa に戻す）."""
    speed = np.hypot(g["u"], g["v"])
    chan = (h > 1e-4) & g["mask"]
    rho = spec["rho"]
    p_pa = g["p"] * rho
    return {
        "speed_max": float(np.nanmax(speed[chan])),
        "speed_mean_channel": float(np.nanmean(speed[chan])),
        "p_max_pa": float(np.nanmax(p_pa[chan])),
        "p_min_pa": float(np.nanmin(p_pa[chan])),
        "dp_pa": float(np.nanmax(p_pa[chan]) - np.nanmin(p_pa[chan])),
        "n_cells": int(chan.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True, help="porous / walls を含むディレクトリ")
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--variants", nargs="*", default=["porous", "walls"])
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    work = Path(a.work)
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    geo = load_trama(a.pattern)

    data, summ, hist = {}, {}, {}
    for v in a.variants:
        case = work / v
        if not case.exists():
            print(f"[skip] {case} が無い")
            continue
        spec = json.loads((case / "case.json").read_text())
        f = load_of_fields(case)
        g = to_grid(f, spec["nx"], spec["ny"], geo.lx, geo.ly)
        h = make_trama_h(geo, spec["nx"], spec["ny"], 3.8e-3, 1.0e-5)
        data[v] = (g, spec, h)
        summ[v] = summarize(g, spec, h) | {"time": f["t"]}
        if (case / "log.simpleFoam").exists():
            hist[v] = residual_history(case / "log.simpleFoam")
        print(f"[{v}] t={f['t']} {summ[v]}")

    if not data:
        raise SystemExit("読める OpenFOAM ケースが無い")

    # --- 図: 流速・圧力 ------------------------------------------------
    n = len(data)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False)
    for r, (v, (g, spec, h)) in enumerate(data.items()):
        speed = np.hypot(g["u"], g["v"])
        chan = h > 1e-4
        sp = np.where(chan & g["mask"], speed, np.nan)
        pp = np.where(chan & g["mask"], g["p"] * spec["rho"], np.nan)
        for c, (arr, ttl, cmap, unit) in enumerate(
            [(sp, "流速", "viridis", "m/s"), (pp, "圧力", "coolwarm", "Pa")]
        ):
            ax = axes[r][c]
            im = ax.imshow(
                arr.T,
                origin="lower",
                extent=(0, geo.lx * 1e3, 0, geo.ly * 1e3),
                cmap=cmap,
                aspect="equal",
            )
            fig.colorbar(im, ax=ax, shrink=0.8, label=unit)
            ax.set_title(f"{v}: {ttl}（t={summ[v]['time']}）")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
    fig.suptitle("OpenFOAM simpleFoam — trama 蛇行流路 0.15 kg/s", fontsize=13)
    fig.tight_layout()
    fig.savefig(figs / "f10_of_fields.png", dpi=110)
    plt.close(fig)

    # --- 図: 残差履歴 --------------------------------------------------
    if hist:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        colors = {"porous": "#B24714", "walls": "#1F6FB4"}
        for v, hh in hist.items():
            ax.semilogy(hh["step"], hh["p"], color=colors.get(v, None), label=f"{v}: p")
            ax.semilogy(
                hh["step"],
                hh["Ux"],
                color=colors.get(v, None),
                ls="--",
                alpha=0.7,
                label=f"{v}: Ux",
            )
        ax.axhline(1e-6, color="0.5", ls=":", lw=1)
        ax.text(ax.get_xlim()[1], 1e-6, " 判定 1e-6", va="center", fontsize=9, color="0.4")
        ax.set_xlabel("SIMPLE 反復")
        ax.set_ylabel("初期残差")
        ax.set_title("OpenFOAM の収束履歴（trama 0.15 kg/s）")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / "f11_of_residuals.png", dpi=110)
        plt.close(fig)

    out = {v: summ[v] for v in summ}
    if len(data) == 2 and "porous" in data and "walls" in data:
        gp, sp_, hp = data["porous"]
        gw, _, _ = data["walls"]
        both = gp["mask"] & gw["mask"] & (hp > 1e-4)
        du = gp["u"][both] - gw["u"][both]
        dv = gp["v"][both] - gw["v"][both]
        ref = np.hypot(gw["u"][both], gw["v"][both])
        out["porous_vs_walls"] = {
            "l2_rel_speed": float(np.sqrt((du**2 + dv**2).sum()) / np.sqrt((ref**2).sum())),
            "n_common_cells": int(both.sum()),
        }
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    print(txt)
    if a.out_json:
        Path(a.out_json).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
