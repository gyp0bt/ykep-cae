"""設計ブロック粒度の感度確認 (HANDOVER Task 4).

ユーザー仮説「セル単位だと枝的最適解が出る」は未検証のまま 5mm ブロックを
採用していた。セルグリッドを固定 (48×64, h=1.25mm) したまま設計グリッドの
粒度だけを 10 / 5 / 2.5 / 1.25mm と振り、最適化モチーフと指標の変化を見る。

- 10mm で割り切れるよう margin_right=4 (bx=12)。基準形状 (bx=11) と 5mm
  1 列だけ違うので、この比較は本スクリプト内で閉じる (他 run と直接比較しない)
- 物理はピンフィン・ダルシー (Forchheimer なし)。粒度→モチーフの関係は
  設計空間のパラメータ化の問題で、慣性補正は ΔP のスケールを変えるだけ
- 枝的度合いの定量化: TV = 面平均 |Δφ| (粗い設計場ほど小さい) と
  細長度 = 開水路 (φ<0.1) の周長/面積比

実行: python run_granularity.py 2>&1 | tee logs/log-granularity-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import darcy as dc
import numpy as np
import torch

torch.set_num_threads(4)

OUT = Path(__file__).parent / "output"
GAMMA = 1.0
ITERS = 600
SIZES = ((8, 6, "10mm"), (16, 12, "5mm"), (32, 24, "2.5mm"), (64, 48, "1.25mm(cell)"))

KEYS = ("dp", "T_peak", "T_block_mean", "T_block_std", "T_fluid_out", "solidity_mean")


def tv_and_perimeter(geo: dc.Geo, s: torch.Tensor, phi_max: float) -> tuple[float, float]:
    """面平均 |Δφ| (TV) と開水路 (φ<0.1) の周長/面積比 [1/セル]."""
    phi = (phi_max * s).reshape(geo.ncy, geo.ncx).numpy()
    tv_x = np.abs(np.diff(phi, axis=1)).mean()
    tv_y = np.abs(np.diff(phi, axis=0)).mean()
    open_mask = phi < 0.1
    per = (
        np.abs(np.diff(open_mask.astype(float), axis=1)).sum()
        + np.abs(np.diff(open_mask.astype(float), axis=0)).sum()
    )
    area = max(open_mask.sum(), 1)
    return 0.5 * (tv_x + tv_y), per / area


def main() -> None:
    cfg = dc.DarcyConfig(pin_fin=True, margin_right=4)  # bx=12 → ncx=48 (10mm で割り切れる)
    geo = dc.make_geo(cfg)
    dp_ref = dc.dp_reference(cfg, geo)
    print(
        f"granularity study: cells {geo.ncx}x{geo.ncy} (h={geo.h * 1e3:.2f}mm), "
        f"gamma={GAMMA}, iters={ITERS}, seed=0\n"
        f"dp_ref(all-open)={dp_ref:.2f}Pa"
    )

    results: dict[str, dict] = {}
    for dy, dx, label in SIZES:
        print(f"\n=== design grid {dx}x{dy} ({label}, {dx * dy} vars) ===")
        t0 = time.perf_counter()
        xi = dc.optimize(
            cfg, geo, gamma_p=GAMMA, iters=ITERS, seed=0, verbose=True, design_shape=(dy, dx)
        )
        s_b = torch.sigmoid(xi)
        r = dc.evaluate(cfg, geo, s_b)
        s = dc.expand(geo, s_b)
        tv, per = tv_and_perimeter(geo, s, cfg.phi_max)
        dp_ref_j = dc.dp_reference(cfg, geo)
        j, _ = dc.objective(cfg, geo, xi, GAMMA, dp_ref_j)
        print(f"  optimize {time.perf_counter() - t0:.0f}s")
        print(
            f"  J={float(j):.4f}  TV={tv:.4f}  perim/area={per:.3f}  "
            + "  ".join(f"{k}={r[k]:.4g}" for k in KEYS)
        )
        assert r["heat_balance_rel"] < 1e-8
        results[label] = {"xi": xi, "r": r, "j": float(j), "tv": tv, "per": per}

    print("\n--- まとめ (γ=1, seed=0, 同一セルグリッド) ---")
    print(
        f"{'granularity':14s} {'vars':>5s} {'J':>8s} {'dp[Pa]':>7s} {'T_peak':>7s} {'TV':>7s} {'per/area':>8s}"
    )
    for (dy, dx, label), (_, d) in zip(SIZES, results.items(), strict=True):
        print(
            f"{label:14s} {dx * dy:5d} {d['j']:8.4f} {d['r']['dp']:7.1f} "
            f"{d['r']['T_peak']:7.2f} {d['tv']:7.4f} {d['per']:8.3f}"
        )

    # ------------------------------------------------------------------
    # 可視化: 行 = (φ, |u|, T_s), 列 = 粒度
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncase = len(results)
    fig = plt.figure(figsize=(4.0 * ncase + 1.0, 13.0), facecolor="#0e1015")
    gs = fig.add_gridspec(3, ncase)
    t_hi = 0.0
    fields = {}
    for label, d in results.items():
        with torch.no_grad():
            s = dc.expand(geo, torch.sigmoid(d["xi"]))
            flow = dc.solve_flow(cfg, geo, s)
            ht = dc.solve_heat(cfg, geo, s, flow)
        fields[label] = (cfg.phi_max * s.numpy(), dc.cell_speed(cfg, geo, flow), ht["t_s"].numpy())
        t_hi = max(t_hi, float(ht["t_s"].max()))

    for c, (label, d) in enumerate(results.items()):
        phi_np, sp_np, ts_np = fields[label]
        panels = (
            (phi_np, "pin fill φ", "viridis", 0.0, cfg.phi_max),
            (sp_np * 1e3, "|u| [mm/s]", "magma", None, None),
            (ts_np, f"T_s [K] peak={d['r']['T_peak']:.1f}", "inferno", 0.0, t_hi),
        )
        for row, (fld, ttl, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[row, c])
            dc.panel(ax, geo, fld, f"({label}) {ttl}", cmap, vmin, vmax, fig=fig)

    fig.suptitle(
        "Design-grid granularity sensitivity — same 48x64 cell grid, design vars "
        "10/5/2.5/1.25mm blocks\n"
        f"pin-fin Darcy, gamma={GAMMA}, seed=0, {ITERS} iters | "
        "hypothesis under test: cell-level design yields branch-like optima",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_granularity.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_granularity.png'}")

    np.savez(
        OUT / "coldplate_granularity_result.npz",
        labels=np.array([label for _, _, label in SIZES]),
        **{f"xi_{label}": results[label]["xi"].numpy() for _, _, label in SIZES},
        metrics=np.array([[results[label]["r"][k] for k in KEYS] for _, _, label in SIZES]),
        js=np.array([results[label]["j"] for _, _, label in SIZES]),
        tv=np.array([results[label]["tv"] for _, _, label in SIZES]),
        metric_keys=np.array(KEYS),
    )
    print(f"saved: {OUT / 'coldplate_granularity_result.npz'}")


if __name__ == "__main__":
    main()
