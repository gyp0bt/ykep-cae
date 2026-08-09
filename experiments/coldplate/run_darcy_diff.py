"""run_darcy.py の結果 npz から差分可視化を再生成する.

全開ケースを外すとカラースケールが 4 ケースだけに最適化され、ケース間の
差分が見えるようになる:

  行 1: log10 K/K_open (各パネル自動スケール — 浅いグレーディングも可視化)
  行 2: |u| (対数カラー、4 ケース共通スケール)
  行 3: T_s (線形、4 ケース共通のタイトスケール ~0-9 K)
  行 4: ΔT_s = T_s(case) - T_s(γ=10) (発散カラー、γ=10 基準)

列は γ 降順 (10 → 0.01)、左から右へ構造が濃くなる方向。再最適化はせず、
output/coldplate_darcy_result.npz の設計場から前進解のみ再計算する。

実行: python run_darcy_diff.py 2>&1 | tee logs/log-darcy-diff-$(date +%s).log
"""

from __future__ import annotations

from pathlib import Path

import darcy as dc
import numpy as np
import torch

OUT = Path(__file__).parent / "output"


def main() -> None:
    cfg = dc.DarcyConfig()
    geo = dc.make_geo(cfg)
    d = np.load(OUT / "coldplate_darcy_result.npz")
    gammas = sorted(d["gammas"].tolist(), reverse=True)  # 10 → 0.01
    tags = [f"g={g}" for g in gammas]

    fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
    for tag in tags:
        s_b = torch.tensor(d[f"s_{tag}"])
        with torch.no_grad():
            s = dc.expand(geo, s_b)
            flow = dc.solve_flow(cfg, geo, s)
            ht = dc.solve_heat(cfg, geo, s, flow)
        r = dc.evaluate(cfg, geo, s_b)
        fields[tag] = (s.numpy(), dc.cell_speed(cfg, geo, flow), ht["t_s"].numpy(), r)
        print(
            f"{tag}: dp={r['dp']:.2f}Pa T_peak={r['T_peak']:.3f}K "
            f"T_std={r['T_block_std']:.3f}K solidity={r['solidity_mean']:.3f}"
        )

    ref_tag = tags[0]  # γ=10 (最小介入) を差分の基準にする
    ts_ref = fields[ref_tag][2]
    t_hi = max(f[2].max() for f in fields.values())
    sp_hi = max(f[1].max() for f in fields.values()) * 1e3
    dt_amp = max(np.abs(f[2] - ts_ref).max() for f in fields.values())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(tags)
    fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 16.5), facecolor="#0e1015")
    for c, tag in enumerate(tags):
        s_np, sp_np, ts_np, r = fields[tag]
        logk = -5.0 * s_np
        dc.panel(
            axes[0, c],
            geo,
            logk,
            f"({tag}) log10 K/K_open  min={logk.min():.2f}",
            "viridis",
            fig=fig,  # 各パネル自動スケール
        )
        dc.panel(
            axes[1, c],
            geo,
            np.maximum(sp_np * 1e3, sp_hi * 1e-3),
            f"({tag}) |u| [mm/s] (log)",
            "magma",
            fig=fig,
            norm=LogNorm(vmin=sp_hi * 1e-3, vmax=sp_hi),
        )
        dc.panel(
            axes[2, c],
            geo,
            ts_np,
            f"({tag}) T_s [K]  peak={r['T_peak']:.2f}",
            "inferno",
            0.0,
            t_hi,
            fig=fig,
        )
        dt = ts_np - ts_ref
        sub = f"dT vs {ref_tag}" if tag != ref_tag else "reference (dT=0)"
        dc.panel(
            axes[3, c],
            geo,
            dt,
            f"({tag}) {sub}  [{dt.min():+.2f}, {dt.max():+.2f}] K",
            "coolwarm",
            -dt_amp,
            dt_amp,
            fig=fig,
        )

    fig.suptitle(
        "Case-to-case differences (all-open excluded) — pseudo-3D Darcy designs\n"
        "row1: permeability (per-panel scale) | row2: speed (log, shared) | "
        f"row3: T_s (shared 0-{t_hi:.1f} K) | row4: T_s difference vs {ref_tag}",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_darcy_diff.png", dpi=110, bbox_inches="tight")
    print(f"saved: {OUT / 'coldplate_darcy_diff.png'}")


if __name__ == "__main__":
    main()
