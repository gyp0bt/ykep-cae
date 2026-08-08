"""擬3D ダルシー則モデルの 2 目的最適化: γ スイープでパレートフロント.

2 列 × 4 行ヒーターアレイ (coldplate 同一形状)、左辺中央 in / 右辺中央 out。
設計変数は 5mm ブロックごとの連続透過率 K ∈ [K_open·1e-5, K_open]。

  J = J_T + γ·ΔP/ΔP_ref を γ ∈ GAMMAS で最小化し、
  (ΔP, T_peak) 平面のパレートフロントと設計場を可視化する。

実行: python run_darcy.py 2>&1 | tee logs/log-darcy-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import darcy as dc
import numpy as np
import torch

OUT = Path(__file__).parent / "output"
GAMMAS = (0.01, 0.1, 1.0, 10.0)
ITERS = 600

KEYS = (
    "dp",
    "T_peak",
    "T_block_mean",
    "T_block_std",
    "T_block_min",
    "T_fluid_out",
    "heat_balance_rel",
    "solidity_mean",
)


def main() -> None:
    cfg = dc.DarcyConfig()
    geo = dc.make_geo(cfg)
    dp_ref = dc.dp_reference(cfg, geo)
    t_ref = dc.t_ref_scale(cfg, geo)
    print(
        f"blocks {geo.bx}x{geo.by}  cells {geo.ncx}x{geo.ncy}  h={geo.h * 1e3:.2f}mm  "
        f"dp_ref(all-open)={dp_ref:.2f}Pa  T_ref={t_ref:.3f}K"
    )

    cases: dict[str, tuple[torch.Tensor, dict[str, float]]] = {}

    # 基準: 全開 (s=0)
    s_open = torch.zeros(geo.by, geo.bx)
    r_open = dc.evaluate(cfg, geo, s_open)
    cases["all-open"] = (s_open, r_open)
    print("\n=== all-open ===\n  " + "  ".join(f"{k}={r_open[k]:.4g}" for k in KEYS))

    for gamma in GAMMAS:
        print(f"\n=== gamma={gamma} ===")
        t0 = time.perf_counter()
        xi = dc.optimize(cfg, geo, gamma_p=gamma, iters=ITERS, seed=0, verbose=True)
        s_b = torch.sigmoid(xi)
        r = dc.evaluate(cfg, geo, s_b)
        print(f"  optimize {time.perf_counter() - t0:.0f}s")
        print("  " + "  ".join(f"{k}={r[k]:.4g}" for k in KEYS))
        assert r["heat_balance_rel"] < 1e-8
        cases[f"g={gamma}"] = (s_b, r)

    print("\n--- 比較 (2x4, Al t=3mm + 流路2mm, 水 5 g/s, 10 W/block) ---")
    for k in KEYS:
        print(f"  {k:18s} " + "  ".join(f"{tag}={r[k]:.4g}" for tag, (_, r) in cases.items()))

    # ------------------------------------------------------------------
    # 可視化: 行 = (K場, 流速, 板温度), 列 = ケース, 右端 = パレート
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncase = len(cases)
    fields = {}
    t_hi = 0.0
    for tag, (s_b, _) in cases.items():
        with torch.no_grad():
            s = dc.expand(geo, s_b)
            flow = dc.solve_flow(cfg, geo, s)
            ht = dc.solve_heat(cfg, geo, s, flow)
        fields[tag] = (s.numpy(), dc.cell_speed(cfg, geo, flow), ht["t_s"].numpy())
        t_hi = max(t_hi, float(ht["t_s"].max()))

    fig = plt.figure(figsize=(4.0 * ncase + 5.0, 13.0), facecolor="#0e1015")
    gs = fig.add_gridspec(3, ncase + 1, width_ratios=[1.0] * ncase + [1.2])
    for c, (tag, (_s_b, r)) in enumerate(cases.items()):
        s_np, sp_np, ts_np = fields[tag]
        ax = fig.add_subplot(gs[0, c])
        dc.panel(ax, geo, -5.0 * s_np, f"({tag}) log10 K/K_open", "viridis", -5.0, 0.0, fig=fig)
        ax = fig.add_subplot(gs[1, c])
        dc.panel(ax, geo, sp_np * 1e3, f"({tag}) |u| [mm/s]", "magma", fig=fig)
        ax = fig.add_subplot(gs[2, c])
        dc.panel(
            ax,
            geo,
            ts_np,
            f"({tag}) T_s [K]  peak={r['T_peak']:.1f}",
            "inferno",
            0.0,
            t_hi,
            fig=fig,
        )

    axp = fig.add_subplot(gs[:, ncase])
    axp.set_facecolor("#171a21")
    for tag, (_, r) in cases.items():
        axp.scatter(r["dp"], r["T_peak"], s=60, zorder=3)
        axp.annotate(
            tag, (r["dp"], r["T_peak"]), textcoords="offset points", xytext=(8, 4), color="w"
        )
    pts = sorted((r["dp"], r["T_peak"]) for _, r in cases.values())
    axp.plot([p[0] for p in pts], [p[1] for p in pts], "--", color="0.6", lw=1)
    axp.set_xlabel("ΔP [Pa]", color="w")
    axp.set_ylabel("T_peak [K over inlet]", color="w")
    axp.set_title("Pareto: pressure drop vs peak temperature", color="w", fontsize=10)
    axp.set_xscale("log")
    axp.tick_params(colors="w")
    for s_ in axp.spines.values():
        s_.set_color("w")
    axp.grid(alpha=0.2)

    fig.suptitle(
        "Pseudo-3D Darcy 2-objective optimization — design var = per-5mm-block "
        "permeability (continuous field)\n"
        f"2x4 array, Al t_base=3mm + channel 2mm, water 5 g/s, 10 W/block | "
        f"dP_ref(all-open)={dp_ref:.1f} Pa | J = J_T + gamma*dP/dP_ref",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_darcy.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_darcy.png'}")

    np.savez(
        OUT / "coldplate_darcy_result.npz",
        gammas=np.array(GAMMAS),
        **{f"s_{tag}": s_b.numpy() for tag, (s_b, _) in cases.items()},
        metrics=np.array([[r[k] for k in KEYS] for _, r in cases.values()]),
        metric_keys=np.array(KEYS),
    )
    print(f"saved: {OUT / 'coldplate_darcy_result.npz'}")


if __name__ == "__main__":
    main()
