"""非定常計算の |R_steady|(t) から擾乱の時間増幅率を測る（縮小ジオメトリの掃引用）.

定常残差 |R|/|R_ref| は「いまの場が定常解からどれだけ離れているか」なので、定常解が無い流れでは
擾乱の全体振幅の代理になる。同じ初期場から出発した掃引ならこの立ち上がりの傾き
d ln|R| / dt [1/s] が、スキームごとの「擾乱が育つ速さ」をそのまま与える。

RMS の空間包絡（`trama_shear_diag.py`）は飽和してからでないと測れないが、こちらは**線形成長段階**で
測れるので短い走行で済む。

使用例::

    python experiments/nsb/trama_growth_rate.py \
        base=experiments/nsb/results/trama_STUB-A-base.yaml \
        k50=experiments/nsb/results/trama_STUB-B-k50.yaml

[README](../../README.md)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = yaml.safe_load(path.read_text())
    return np.asarray(d["times"], float), np.asarray(d["steady_residual"], float)


def fit(t: np.ndarray, r: np.ndarray, t0: float, t1: float) -> dict[str, float]:
    m = (t >= t0) & (t <= t1) & np.isfinite(r) & (r > 0)
    if m.sum() < 10:
        return {"rate_per_s": float("nan"), "n": int(m.sum())}
    c = np.polyfit(t[m], np.log(r[m]), 1)
    return {
        "rate_per_s": float(c[0]),
        "doubling_ms": float(np.log(2.0) / c[0] * 1e3) if c[0] > 0 else float("inf"),
        "r_start": float(r[m][0]),
        "r_end": float(r[m][-1]),
        "ratio": float(r[m][-1] / r[m][0]),
        "n": int(m.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", help="ラベル=results/trama_TAG.yaml")
    ap.add_argument("--t0", type=float, default=0.05, help="回帰の開始時刻 [s]")
    ap.add_argument("--t1", type=float, default=1e9, help="回帰の終了時刻 [s]")
    ap.add_argument("--fig", default="experiments/nsb/results/trama_figs/f25_stub_growth.png")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_stub_growth.json")
    a = ap.parse_args()

    out = {}
    print(f"{'':>12} {'d ln|R|/dt':>11} {'倍加 [ms]':>10} {'|R| 始':>9} {'|R| 終':>9} {'比':>7}")
    series = {}
    for spec in a.cases:
        lab, _, path = spec.partition("=")
        lab = lab or Path(path).stem
        t, r = load(Path(path))
        series[lab] = (t, r)
        f = fit(t, r, a.t0, min(a.t1, float(t[-1])))
        out[lab] = f
        print(
            f"{lab:>12} {f['rate_per_s']:11.3f} {f.get('doubling_ms', float('nan')):10.1f} "
            f"{f.get('r_start', float('nan')):9.3e} {f.get('r_end', float('nan')):9.3e} "
            f"{f.get('ratio', float('nan')):7.2f}"
        )
    Path(a.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for lab, (t, r) in series.items():
        ax.semilogy(t, r, label=f"{lab}  {out[lab]['rate_per_s']:+.2f} /s")
    ax.set(
        xlabel="物理時間 [s]",
        ylabel="定常残差 |R| / |R_ref|",
        title="擾乱の育ち方（同じ初期場から、スキームだけを変えた掃引）",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=130)
    print(f"\n[growth] {a.out_json}, {a.fig}")


if __name__ == "__main__":
    main()
