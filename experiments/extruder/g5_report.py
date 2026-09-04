"""G5 の result.json から docs/reports/extruder/g5-literature-rtd.md を組み立てる.

PYTHONPATH=. .venv/bin/python experiments/extruder/g5_report.py --work /tmp/of-g5 --out docs/reports/extruder/g5-literature-rtd.md

図はすべて inline SVG（軸・文字は currentColor、ykep の系列だけ色を持つ）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections.abc import Sequence

import numpy as np

TOL_P10 = TOL_P50 = 0.03
TOL_P90 = 0.05
TOL_CURVE = 0.05
TOL_MEAN = 0.02

ACCENT = "#B24714"
SERIES_COLORS = {0.004: "#7A9CC6", 0.002: "#4D6FA8", 0.001: ACCENT}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _mark(r: float) -> str:
    return "✅" if r < 1.0 else "❌"


def _ratio(dev: float, tol: float) -> str:
    return f"**{dev / tol:.2f}** {_mark(dev / tol)}"


def _pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%"


# ---------------------------------------------------------------- SVG 部品


def _polyline(xs, ys, sx, sy, **attrs) -> str:
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True))
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polyline points="{pts}" fill="none" {a}/>'


def line_chart(
    series: list[dict],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    xticks: list[float],
    yticks: list[float],
    caption: str,
    width: int = 720,
    height: int = 380,
    hlines: Sequence[tuple[float, str]] = (),
    vlines: Sequence[tuple[float, str]] = (),
    legend_pos: tuple[int, int] | None = None,
) -> str:
    """折れ線図。series: {label, x, y, color, width, dash}。"""
    ml, mr, mt, mb = 60, 20, 16, 48
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x):
        return ml + (x - xlim[0]) / (xlim[1] - xlim[0]) * pw

    def sy(y):
        return mt + ph - (y - ylim[0]) / (ylim[1] - ylim[0]) * ph

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{caption}" '
        'style="max-width:100%;height:auto;font-family:sans-serif;font-size:12px">'
    ]
    # 格子と軸
    for xt in xticks:
        out.append(
            f'<line x1="{sx(xt):.1f}" y1="{mt}" x2="{sx(xt):.1f}" y2="{mt + ph}" '
            'stroke="currentColor" stroke-opacity="0.15"/>'
        )
        out.append(
            f'<text x="{sx(xt):.1f}" y="{mt + ph + 16}" text-anchor="middle" fill="currentColor">{xt:g}</text>'
        )
    for yt in yticks:
        out.append(
            f'<line x1="{ml}" y1="{sy(yt):.1f}" x2="{ml + pw}" y2="{sy(yt):.1f}" '
            'stroke="currentColor" stroke-opacity="0.15"/>'
        )
        out.append(
            f'<text x="{ml - 6}" y="{sy(yt) + 4:.1f}" text-anchor="end" fill="currentColor">{yt:g}</text>'
        )
    out.append(
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="currentColor"/>'
    )
    out.append(
        f'<text x="{ml + pw / 2:.1f}" y="{height - 8}" text-anchor="middle" fill="currentColor">{xlabel}</text>'
    )
    out.append(
        f'<text transform="translate(14,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" fill="currentColor">{ylabel}</text>'
    )
    for yv, lab in hlines:
        out.append(
            f'<line x1="{ml}" y1="{sy(yv):.1f}" x2="{ml + pw}" y2="{sy(yv):.1f}" '
            'stroke="currentColor" stroke-dasharray="2 4"/>'
        )
        out.append(
            f'<text x="{ml + pw - 4}" y="{sy(yv) - 4:.1f}" text-anchor="end" fill="currentColor">{lab}</text>'
        )
    for xv, lab in vlines:
        out.append(
            f'<line x1="{sx(xv):.1f}" y1="{mt}" x2="{sx(xv):.1f}" y2="{mt + ph}" '
            'stroke="currentColor" stroke-dasharray="2 4"/>'
        )
        out.append(f'<text x="{sx(xv) + 4:.1f}" y="{mt + 12}" fill="currentColor">{lab}</text>')
    # 系列（範囲外の点は落とす）
    for s in series:
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        keep = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        attrs = {"stroke": s.get("color", "currentColor"), "stroke_width": s.get("width", 1.5)}
        if s.get("dash"):
            attrs["stroke_dasharray"] = s["dash"]
        if s.get("markers"):
            for xi, yi in zip(x[keep], y[keep], strict=True):
                out.append(
                    f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="3.5" fill="{attrs["stroke"]}"/>'
                )
        out.append(_polyline(x[keep], y[keep], sx, sy, **attrs))
    # 凡例
    lx, ly = legend_pos or (ml + pw - 200, mt + 12)
    for k, s in enumerate(series):
        yy = ly + 16 * k
        attrs = f'stroke="{s.get("color", "currentColor")}" stroke-width="{s.get("width", 1.5)}"'
        if s.get("dash"):
            attrs += f' stroke-dasharray="{s["dash"]}"'
        out.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 26}" y2="{yy}" {attrs}/>')
        out.append(f'<text x="{lx + 32}" y="{yy + 4}" fill="currentColor">{s["label"]}</text>')
    out.append("</svg>")
    return "<figure>\n" + "\n".join(out) + f"\n<figcaption>{caption}</figcaption>\n</figure>"


OVERVIEW_SVG = """
<figure>
<svg viewBox="0 0 800 330" role="img" aria-label="文献モデルの三つの仮定を ykep 側の三つの条件で満たし、同じ物差し compare_rtd で分位点と曲線を比べる" style="max-width:100%;height:auto;font-family:sans-serif;font-size:12px">
  <defs><marker id="ah5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0,0 8,4 0,8" fill="currentColor"/></marker></defs>

  <rect x="20" y="20" width="250" height="130" rx="6" fill="none" stroke="currentColor"/>
  <text x="145" y="42" text-anchor="middle" font-weight="bold">Pinto–Tadmor 1970（真値）</text>
  <text x="32" y="64">無限幅 → 側壁なし、Q = Q_∞</text>
  <text x="32" y="82">フライトでの折返しは瞬時</text>
  <text x="32" y="100">周回を無限回 → 周回平均の速度</text>
  <text x="32" y="126" font-style="italic">F_PT(t/t̄)、r によらない普遍曲線</text>

  <rect x="20" y="180" width="250" height="130" rx="6" fill="none" stroke="#B24714" stroke-width="1.5"/>
  <text x="145" y="202" text-anchor="middle" font-weight="bold" fill="#B24714">ykep 2.5D + 粒子追跡</text>
  <text x="32" y="224">有限幅（側壁で Q = F_d·Q_∞）</text>
  <text x="32" y="242">隙間 δ、有限長 z_axial</text>
  <text x="32" y="260">ψ 双一次補間 + RK4（刻み cfl）</text>
  <text x="32" y="286" font-style="italic">t_res, weight（流束重み）</text>

  <rect x="330" y="100" width="190" height="130" rx="6" fill="none" stroke="currentColor"/>
  <text x="425" y="122" text-anchor="middle" font-weight="bold">仮定に寄せる 3 条件</text>
  <text x="342" y="146">δ = 0（隙間の速い経路を消す）</text>
  <text x="342" y="166">z = 0.5 m（周回 ≈ 5 回）</text>
  <text x="342" y="186">cfl = 0.1（流線ドリフトを消す）</text>
  <text x="342" y="214">規格化 t̄_∞ = t̄_theory·F_d</text>

  <line x1="270" y1="245" x2="330" y2="200" stroke="#B24714" marker-end="url(#ah5)"/>

  <rect x="580" y="100" width="200" height="130" rx="6" fill="none" stroke="currentColor"/>
  <text x="680" y="122" text-anchor="middle" font-weight="bold">compare_rtd</text>
  <text x="592" y="146">p10 / p50 / p90 の比 − 1</text>
  <text x="592" y="166">分位関数 t(F) の相対偏差</text>
  <text x="592" y="186">F ∈ [0.05, 0.9] 平均（横方向）</text>
  <text x="592" y="214" font-style="italic">閾値 3% / 3% / 5% / 5%</text>

  <line x1="270" y1="85" x2="580" y2="140" stroke="currentColor" marker-end="url(#ah5)"/>
  <text x="430" y="70" text-anchor="middle">F_PT</text>
  <line x1="520" y1="165" x2="580" y2="165" stroke="#B24714" marker-end="url(#ah5)"/>
</svg>
<figcaption>文献モデルの三つの仮定（無限幅・折返し瞬時・周回無限回）を ykep 側で三つの条件に翻訳し、同じ物差しで分位点と曲線を比べる。</figcaption>
</figure>
"""


def profile_figure() -> str:
    """1D 速度分布: ŵ（引きずり）、û（横断）、3ξ(1−ξ)（圧力 = 引きずり − 横断）."""
    xi = np.linspace(0.0, 1.0, 101)
    w_drag = xi
    u_cross = 3 * xi * xi - 2 * xi
    p_prof = 3 * xi * (1 - xi)
    return line_chart(
        [
            {"label": "ŵ_drag = ξ（下流・引きずり）", "x": w_drag, "y": xi, "width": 2},
            {"label": "û = 3ξ² − 2ξ（横断、正味 0）", "x": u_cross, "y": xi, "dash": "6 3"},
            {
                "label": "3ξ(1−ξ) = ξ − û（圧力流れ ＝ 軸方向速度の形）",
                "x": p_prof,
                "y": xi,
                "color": ACCENT,
                "width": 2,
            },
        ],
        xlim=(-0.4, 1.0),
        ylim=(0.0, 1.0),
        xlabel="無次元速度",
        ylabel="ξ = y/H（0 根元、1 バレル）",
        xticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
        yticks=[0, 0.25, 0.5, 2 / 3, 0.75, 1.0],
        hlines=[(2 / 3, "ξ = 2/3 停留線（û = 0）"), (0.5, "ξ = 1/2 軸方向速度の最大")],
        caption="恒等式 3ξ(1−ξ) = ξ − (3ξ² − 2ξ)。圧力流れの分布は引きずり分布から横断分布を引いたものに等しく、横断分布は周回平均で消えるので、どの流線でも周回平均速度は (1+r) 倍になるだけ。同じ 3ξ(1−ξ) が軸方向速度 u cosφ + w sinφ の形でもある。",
        width=720,
        height=400,
        legend_pos=(70, 28),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R = _load(os.path.join(args.work, "result.json"))
    meta, pt, cases = R["meta"], R["pt"], R["cases"]
    by = {}
    for c in cases:
        by.setdefault(c["label"], []).append(c)
    series = by["series"]  # H = 4, 2, 1
    shallow = series[-1]
    length = sorted([*by["length"], shallow], key=lambda c: c["z_axial"])
    gap = by["gap"][0]
    cfl1 = by["cfl"][0]
    press = by["pressure"][0]
    today = dt.date.today().isoformat()

    gate = [
        (
            "p10/t̄_∞ の文献比",
            shallow["dev_p10"],
            TOL_P10,
            f"{shallow['p10_over_ref']:.4f} vs {pt['t_p10_ratio']:.4f}",
        ),
        (
            "p50 同上",
            shallow["dev_p50"],
            TOL_P50,
            f"{shallow['p50_over_ref']:.4f} vs {pt['t_p50_ratio']:.4f}",
        ),
        (
            "p90 同上",
            shallow["dev_p90"],
            TOL_P90,
            f"{shallow['p90_over_ref']:.4f} vs {pt['t_p90_ratio']:.4f}",
        ),
        (
            "曲線: 分位関数の相対偏差の F∈[0.05,0.9] 平均",
            shallow["curve_l1"],
            TOL_CURVE,
            f"最大 {shallow['curve_max']:.3f}（階段幅、観察）",
        ),
        (
            "⟨t⟩ = V/Q（全 H の最大ずれ）",
            max(abs(c["t_mean_over_theory"] - 1) for c in series),
            TOL_MEAN,
            "流線ドリフトの残りが裾に出ればここに現れる",
        ),
        (
            "t_min/t̄_∞ → 3/4（全 H の最大ずれ）",
            max(abs(c["t_min_over_ref"] / 0.75 - 1) for c in series),
            0.05,
            "種はセル中心なので停留線に乗れない",
        ),
    ]
    mono = all(
        series[k][key] >= series[k + 1][key]
        for key in ("dev_p50", "dev_p90", "curve_l1")
        for k in range(len(series) - 1)
    )
    passed = all(dev / tol < 1.0 for _, dev, tol, _ in gate) and mono

    gate_rows = "\n".join(
        f"| {name} | {_ratio(dev, tol)} | {dev:.4f} / {tol} | {note} |"
        for name, dev, tol, note in gate
    )
    gate_rows += f"\n| 単調接近（p50・p90・曲線が H/W とともに減る） | **{'成立' if mono else '不成立'}** {'✅' if mono else '❌'} | — | 下表 |"

    series_rows = "\n".join(
        f"| {c['H'] * 1e3:g} | {c['H_over_W']:.3f} | {c['F_d']:.3f} | {c['n_loops']:.1f} | "
        f"{c['t_mean_over_theory']:.3f} | {c['t_min_over_ref']:.3f} | "
        f"{c['p10_over_ref']:.4f} ({_pct(c['p10_over_ref'] / pt['t_p10_ratio'] - 1)}) | "
        f"{c['p50_over_ref']:.4f} ({_pct(c['p50_over_ref'] / pt['t_p50_ratio'] - 1)}) | "
        f"{c['p90_over_ref']:.4f} ({_pct(c['p90_over_ref'] / pt['t_p90_ratio'] - 1)}) | "
        f"{c['curve_l1']:.4f} | {c['curve_max']:.3f} | {c['wall_s']:.0f} |"
        for c in series
    )
    length_rows = "\n".join(
        f"| {c['z_axial']:g} | {c['z_axial'] / meta['base_spec']['D']:.1f} D | {c['n_loops']:.1f} | "
        f"{c['t_min_over_ref']:.3f} | {c['p10_over_ref']:.4f} ({_pct(c['p10_over_ref'] / pt['t_p10_ratio'] - 1)}) | "
        f"{c['p50_over_ref']:.4f} ({_pct(c['p50_over_ref'] / pt['t_p50_ratio'] - 1)}) | "
        f"{c['p90_over_ref']:.4f} ({_pct(c['p90_over_ref'] / pt['t_p90_ratio'] - 1)}) | {c['curve_l1']:.4f} |"
        for c in length
    )

    def cmp_row(name, c):
        return (
            f"| {name} | {c['t_mean_over_theory']:.3f} | {c['t_min_over_ref']:.3f} | "
            f"{c['p10_over_ref']:.4f} ({_pct(c['p10_over_ref'] / pt['t_p10_ratio'] - 1)}) | "
            f"{c['p50_over_ref']:.4f} ({_pct(c['p50_over_ref'] / pt['t_p50_ratio'] - 1)}) | "
            f"{c['p90_over_ref']:.4f} ({_pct(c['p90_over_ref'] / pt['t_p90_ratio'] - 1)}) | "
            f"{c['curve_l1']:.4f} | {c['curve_max']:.3f} | {100 * c['extrapolated_weight_fraction']:.1f}% | {c['wall_s']:.0f} |"
        )

    cmp_head = (
        "| ケース | ⟨t⟩/t̄_theory | t_min/t̄_∞ | p10 | p50 | p90 | 曲線 L1 | 曲線 max | 外挿粒子 | 追跡 [s] |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    # 図: F(t/t̄∞) 重ね描き
    pt_series = {
        "label": "Pinto–Tadmor（文献）",
        "x": pt["curve_t"],
        "y": pt["curve_F"],
        "width": 2.5,
    }
    overlay = line_chart(
        [pt_series]
        + [
            {
                "label": f"ykep H = {c['H'] * 1e3:g} mm（H/W = {c['H_over_W']:.3f}）",
                "x": c["curve_t"],
                "y": c["curve_F"],
                "color": SERIES_COLORS[c["H"]],
                "width": 1.5,
            }
            for c in series
        ]
        + [
            {
                "label": "ykep H = 1 mm, r = −0.3（背圧）",
                "x": press["curve_t"],
                "y": press["curve_F"],
                "color": ACCENT,
                "dash": "5 3",
            }
        ],
        xlim=(0.6, 2.4),
        ylim=(0.0, 1.0),
        xlabel="t / t̄_∞",
        ylabel="F（累積割合）",
        xticks=[0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
        yticks=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        vlines=[(0.75, "t_min/t̄ = 3/4")],
        caption="縮約 RTD の重ね描き。H を浅くするほど ykep（色）が文献（黒）に寄り、背圧 r = −0.3（破線）は r = 0 と重なる（普遍性）。t/t̄_∞ > 2.4 の裾は省略。",
        legend_pos=(400, 200),
    )
    overlay_tail = line_chart(
        [pt_series]
        + [
            {
                "label": f"ykep H = {c['H'] * 1e3:g} mm",
                "x": c["curve_t"],
                "y": c["curve_F"],
                "color": SERIES_COLORS[c["H"]],
                "width": 1.5,
            }
            for c in series
        ],
        xlim=(0.7, 5.0),
        ylim=(0.8, 1.0),
        xlabel="t / t̄_∞",
        ylabel="F",
        xticks=[1, 2, 3, 4, 5],
        yticks=[0.8, 0.85, 0.9, 0.95, 1.0],
        caption="裾（F > 0.8）の拡大。側壁近傍の遅い流線が長時間側に足され、H/W が大きいほど裾が重い（Bigg & Middleman 1974 の機構）。",
        height=300,
        legend_pos=(400, 200),
    )
    # 図: 周回数 vs 短時間側
    loops_fig = line_chart(
        [
            {
                "label": "t_min / t̄_∞",
                "x": [c["n_loops"] for c in length],
                "y": [c["t_min_over_ref"] for c in length],
                "color": ACCENT,
                "markers": True,
                "width": 2,
            },
            {
                "label": "p10 / t̄_∞",
                "x": [c["n_loops"] for c in length],
                "y": [c["p10_over_ref"] for c in length],
                "color": "#4D6FA8",
                "markers": True,
                "width": 2,
            },
            {
                "label": "p50 / t̄_∞",
                "x": [c["n_loops"] for c in length],
                "y": [c["p50_over_ref"] for c in length],
                "markers": True,
                "width": 1.5,
                "dash": "5 3",
            },
        ],
        xlim=(0.0, 12.0),
        ylim=(0.6, 0.9),
        xlabel="周回数 ≈ t̄_∞ / T_loop（流量中央値の流線対）",
        ylabel="t / t̄_∞",
        xticks=[0, 2, 4, 6, 8, 10, 12],
        yticks=[0.6, 0.65, 2 / 3, 0.7, 0.75, 0.8, 0.85, 0.9],
        hlines=[
            (2 / 3, "2/3: ξ = 1/2 の粒子が周回せず抜ける"),
            (0.75, "3/4: 文献の t_min（停留線 ξ = 2/3）"),
            (pt["t_p10_ratio"], "文献 p10"),
            (pt["t_p50_ratio"], "文献 p50"),
        ],
        caption="区間長（周回数）と短時間側。周回 1〜2 回では t_min が 2/3 に落ち、周回 5 回以上で文献の 3/4 に寄る。p50 は周回 2 回で既に 2% 内。",
        legend_pos=(480, 300),
    )
    # 図: 隙間・時間刻み
    gap_fig = line_chart(
        [
            pt_series,
            {
                "label": "閉チャネル δ = 0（ゲート）",
                "x": shallow["curve_t"],
                "y": shallow["curve_F"],
                "color": ACCENT,
                "width": 1.5,
            },
            {
                "label": f"隙間 δ/H = {gap['delta_over_H']:.3f}",
                "x": gap["curve_t"],
                "y": gap["curve_F"],
                "color": "#4D6FA8",
                "width": 1.5,
            },
            {
                "label": "閉チャネル、cfl = 1.0（既定の刻み）",
                "x": cfl1["curve_t"],
                "y": cfl1["curve_F"],
                "color": "#4D6FA8",
                "dash": "5 3",
            },
        ],
        xlim=(0.6, 3.0),
        ylim=(0.0, 1.0),
        xlabel="t / t̄_∞",
        ylabel="F",
        xticks=[0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
        yticks=[0, 0.25, 0.5, 0.75, 0.9, 1.0],
        caption="ゲートの条件を 1 つずつ外す。隙間（実線・青）は裾を大きく伸ばし、既定の時間刻み cfl = 1.0（破線）も裾と中央値を押し上げる。",
        legend_pos=(380, 200),
    )

    bs = meta["base_spec"]
    gap_label = f"隙間 δ/H = {gap['delta_over_H']:.3f}、cfl = 0.1"
    md = f"""# ゲート G5 — 文献 RTD（Pinto–Tadmor 1970）との照合

[<- docs](../../README.md) | [<- 設計文書](../../design/single-screw-extruder.md) | [<- G5 設計](../../design/extruder-g5-literature-rtd.md) | [<- 図解レポート](README.md)

**実行**: {today} / branch `{meta["branch"]}` @ `{meta["commit"]}` / 生成 {meta["generated"]}
**コマンド**: `PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/g5_literature.py --out /tmp/of-g5` → `g5_report.py`
**判定**: G5 {"合格" if passed else "不合格"}（全て比 < 1.00 が合格）

## 1. 何を何と比べるか

実機データが無いので、Phase 2 に進む前提を「実機との突き合わせ」から**文献の解析
RTD との照合**に差し替えた。真値は Pinto & Tadmor (1970) の計量部 RTD。等温ニュートン・
無限幅・漏れ無しの 1D 速度場を、フライトで折り返す流線対の周回平均で積分した
閉じた曲線で、Wolf & White (1976) の放射性トレーサ実験が計量部で確認している。
ykep 側は同じ 40 mm 機（D = {bs["D"] * 1e3:g}, lead = {bs["lead"] * 1e3:g}, e = {bs["e"] * 1e3:g} mm, φ = {bs["phi_deg"]:.1f}°,
N = {bs["N"] * 60:.0f} rpm, μ = {meta["mu"]:g} Pa·s）で H を 4 → 2 → 1 mm と浅くし、無限幅の極限に
**近づく向き**と**最浅での一致**を見る。

{OVERVIEW_SVG}

### 判定表（H = 1 mm、H/W = {shallow["H_over_W"]:.3f}、格子 {meta["grid"][0]}×{meta["grid"][1]}、z = {meta["z_axial"]} m、cfl = {meta["cfl"]}）

| 検査 | 比（閾値規格化） | 偏差 / 閾値 | 備考 |
|---|---|---|---|
{gate_rows}

文献値（t̄ 規格化）: t_min = 3/4、p10 = {pt["t_p10_ratio"]:.4f}、p50 = {pt["t_p50_ratio"]:.4f}、p90 = {pt["t_p90_ratio"]:.4f}。

### H 系列（すべて t̄_∞ = t̄_theory·F_d 規格化。括弧は文献比 − 1）

| H [mm] | H/W | F_d | 周回数 | ⟨t⟩/t̄_theory | t_min/t̄_∞ | p10 | p50 | p90 | 曲線 L1 | 曲線 max | 追跡 [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
{series_rows}

## 2. 縮約 RTD の重ね描き

{overlay}

{overlay_tail}

## 3. なぜそうなるか（メカニズム）

### 3.1 文献曲線が背圧 r によらない — 恒等式 3ξ(1−ξ) = ξ − (3ξ² − 2ξ)

{profile_figure()}

下流速度は ŵ = ξ + 3rξ(1−ξ)、横断速度は û = 3ξ² − 2ξ（正味流量 0）。圧力流れの
分布 3ξ(1−ξ) は「引きずり分布 ξ − 横断分布 û」に恒等的に等しい。横断速度は閉じた
流線 1 周で変位ゼロなので周平均が消え、どの流線でも周回平均の下流速度は
(1+r)·w̄_drag、流線対の流量も (1+r) 倍。t も t̄ も同じ因子で割られて F(t/t̄) は r に
依らない。ykep で r = −0.3 相当の背圧（G = {press["G"]:.2e} Pa/m）を掛けた曲線が
r = 0 と重なる（p50 の文献比 {_pct(press["p50_over_ref"] / pt["t_p50_ratio"] - 1)}、曲線 L1 {press["curve_l1"]:.4f}）のは、
有限幅でもこの恒等式が溝中央で効いているから。側壁があると F_d ≠ F_p なので
r の定義に (F_d + r F_p)/(1 + r) の補正を入れて規格化している。

同じ 3ξ(1−ξ) は軸方向速度 u cosφ + w sinφ = 3ξ(1−ξ)·V sinφ cosφ の形でもある。
バレル面（ξ = 1）と根元（ξ = 0）で軸方向速度がゼロ、ξ = 1/2 で最大、というのが
§3.3 の「周回数」の話の土台になる。

### 3.2 側壁 — F_d 規格化と裾の重さ

側壁（フライト）は二つの仕方で効く。**流量**: 側壁近傍の遅い w が流量を F_d 倍に
減らし、t̄_theory = V/Q を 1/F_d 倍に延ばす。分位点を担う溝中央の流線は側壁を知らない
ので、絶対時間は文献の t̄_∞ = HWL/Q_∞ に対して決まる。だから ykep の t/t̄_theory に
F_d を掛けてから比べる（H/W → 0 で F_d → 1 なので極限の主張はどちらでも同じ）。
**分布の形**: 側壁近傍の遅い流線そのものが長時間側に足される。これは
Bigg & Middleman (1974) が有限幅の数値解で示した機構で、上の裾の拡大図がそれ。
H/W = 0.117 → 0.029 で p90 の文献比が {_pct(series[0]["p90_over_ref"] / pt["t_p90_ratio"] - 1)} → {_pct(series[-1]["p90_over_ref"] / pt["t_p90_ratio"] - 1)}、
p50 が {_pct(series[0]["p50_over_ref"] / pt["t_p50_ratio"] - 1)} → {_pct(series[-1]["p50_over_ref"] / pt["t_p50_ratio"] - 1)} と単調に縮む。

### 3.3 周回数 — 短時間側は「何回回ったか」で決まる

{loops_fig}

| z [m] | 長さ/D | 周回数 | t_min/t̄_∞ | p10 | p50 | p90 | 曲線 L1 |
|---|---|---|---|---|---|---|---|
{length_rows}

文献モデルは粒子が流線対を無限回周回する極限で、滞留時間は流線ごとの
**周回平均速度**で決まる。周回が 1〜2 回しかないと、滞留時間は「どの高さで入ったか」で
決まる。軸方向速度 3ξ(1−ξ) は ξ = 1/2 で最大なので、そこに入った粒子が周回する前に
抜けて t_min/t̄ = 2/3 になる（文献は停留線 ξ = 2/3 の粒子の 3/4）。周回時間 ≈ W/V_x は
H に依らないので、H 系列は z を固定すれば周回数が揃う。

これは実機にも言えることで、この 40 mm 機の計量部 5D（z = 0.2 m）は周回 2 回程度。
**実機の計量部の短時間側は文献曲線より広い**（t_min が 3/4 でなく 2/3 側）のが
モデルの帰結で、文献照合はあくまで「長い極限で一致する」ことの確認になる。

### 3.4 隙間と時間刻み — ゲートの条件を 1 つずつ外す

{gap_fig}

{cmp_head}
{cmp_row("閉チャネル δ = 0、cfl = 0.1（ゲート）", shallow)}
{cmp_row(gap_label, gap)}
{cmp_row("閉チャネル、cfl = 1.0（既定）", cfl1)}
{cmp_row("閉チャネル、r = −0.3（背圧）", press)}

**隙間。** 隙間を越える材料はバレル直下から出発するので軸方向速度がほぼゼロ
（バレル面では u cosφ + w sinφ = 0）。隙間を通る間は軸方向に進まず、その分だけ
長時間側に足される。隙間を少なくとも一度通る流量の割合は ≈ 2 tanφ (δ/H) L/W で、
δ/H を固定すると H/W → 0 でも消えない。だからゲートは閉チャネルで取り、隙間は
G4b の知見（短時間側の広がり、デッドゾーン解消）と合わせて別に観察する。

**時間刻み。** ψ を双一次補間した速度場は発散ゼロだが、セル境界で速度勾配が
不連続になる。既定の cfl = 1.0 はセル 1 個分を 1 ステップで跨ぐので、RK4 が
流線を横切る誤差を残し、周回を重ねると粒子が壁際の遅い領域へ流れ込む。
閉チャネル H = 1 mm、周回 ≈ 5 で |Δψ|/ψ_max の中央値が cfl 1.0 → 0.25 → 0.1 で
2.8% → 0.5% → 収束。p90 の文献比は
{_pct(cfl1["p90_over_ref"] / pt["t_p90_ratio"] - 1)} → {_pct(shallow["p90_over_ref"] / pt["t_p90_ratio"] - 1)}。cfl = 1.0 では壁際に流れ込んだ粒子の
{100 * cfl1["extrapolated_weight_fraction"]:.1f}%（流束重み）が最大ステップ数で止まり、残りの距離を外挿した滞留時間が
平均を ⟨t⟩/t̄_theory = {cfl1["t_mean_over_theory"]:.2f} まで押し上げる（p10・p50 は溝中央の流線なのでほぼ無傷）。
cfl = 0.1 では止まる粒子が無く、平均も {shallow["t_mean_over_theory"]:.3f} に戻る。G4b で「閉チャネルの平均は当てにならない」と
した観察の大半はこの数値効果だった（`ParticleTrackInput.cfl` の docstring と
`TestStreamlineDrift` に記録）。短い区間（周回 1〜2 回）では見えないので G4b の
判定は変わらない。

### 3.5 実測との鎖

Pinto & Tadmor (1970) の曲線は Wolf & White (1976) が放射性トレーサで計量部の
RTD を測って確認しており、Bigg & Middleman (1974) は有限幅での裾の重さを数値解で
示した。本ゲートは ykep の追跡が「無限幅・長い区間」の極限で Pinto–Tadmor に
収束し、有限幅で Bigg–Middleman の向きに外れることを確かめたもので、実測値の
転記はしていない（設計 §6）。

## 4. 再現

```bash
cd ~/work/ykep-cae
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/g5_literature.py --out /tmp/of-g5
PYTHONPATH=. .venv/bin/python experiments/extruder/g5_report.py --work /tmp/of-g5 --out docs/reports/extruder/g5-literature-rtd.md
```

実験 6〜8 分（2 スレッド、格子 {meta["grid"][0]}×{meta["grid"][1]}、粒子 {shallow["n_particles"]} 個/ケース）。
ゲートのテストは `tests/test_extruder_literature_rtd.py`（細格子系列は `slow`、
粗格子の最浅 1 本は既定の回帰）。
"""
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {args.out}  ({'PASS' if passed else 'FAIL'})")


if __name__ == "__main__":
    main()
