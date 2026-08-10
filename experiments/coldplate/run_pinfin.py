"""ピンフィン物理モデル (pin_fin=True) の 2 目的最適化: γ スイープ.

HANDOVER Task 1: s=1「流れを止める固体 = 良伝導体」の非物理を解消し、
s をピン充填率 φ = phi_max·s と解釈して

  K(φ)  = 平行平板 + Gebart 円柱列の直列 (流れをある程度通す)
  U''(φ) = プライム面 + ピンフィン増倍 (z 方向実効 h が φ で上がる)

を同時にモデル化した。旧 logK モデル (coldplate_darcy_result.npz) と
同じ γ で最適化し、モチーフと指標がどう変わるかを比較する。

実行: python run_pinfin.py 2>&1 | tee logs/log-pinfin-$(date +%s).log
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
    cfg = dc.DarcyConfig(pin_fin=True)
    geo = dc.make_geo(cfg)
    dp_ref = dc.dp_reference(cfg, geo)
    t_ref = dc.t_ref_scale(cfg, geo)
    print(
        f"pin-fin model: d={cfg.d_pin * 1e3:.1f}mm phi_max={cfg.phi_max} Nu_pin={cfg.nu_pin}\n"
        f"blocks {geo.bx}x{geo.by}  cells {geo.ncx}x{geo.ncy}  h={geo.h * 1e3:.2f}mm  "
        f"dp_ref(all-open)={dp_ref:.2f}Pa  T_ref={t_ref:.3f}K"
    )

    cases: dict[str, tuple[torch.Tensor, dict[str, float]]] = {}

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

    print("\n--- 比較 (pin-fin, 2x4, Al t=3mm + 流路2mm, 水 5 g/s, 10 W/block) ---")
    for k in KEYS:
        print(f"  {k:18s} " + "  ".join(f"{tag}={r[k]:.4g}" for tag, (_, r) in cases.items()))

    # 旧 logK モデルとの比較 (同 γ)
    legacy_path = OUT / "coldplate_darcy_result.npz"
    if legacy_path.exists():
        lg = np.load(legacy_path)
        lk = list(lg["metric_keys"])
        idx_dp, idx_tp = lk.index("dp"), lk.index("T_peak")
        print("\n--- vs 旧 logK モデル (dp[Pa] / T_peak[K]) ---")
        tags = ["all-open"] + [f"g={g}" for g in lg["gammas"]]
        for i, tag in enumerate(tags):
            if tag in cases:
                r = cases[tag][1]
                print(
                    f"  {tag:10s} legacy {lg['metrics'][i, idx_dp]:7.1f} / "
                    f"{lg['metrics'][i, idx_tp]:5.1f}   pin-fin {r['dp']:7.1f} / "
                    f"{r['T_peak']:5.1f}"
                )

    # ------------------------------------------------------------------
    # 可視化: 行 = (φ場, 流速, 板温度), 列 = ケース, 右端 = パレート
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
        fields[tag] = (
            cfg.phi_max * s.numpy(),
            dc.cell_speed(cfg, geo, flow),
            ht["t_s"].numpy(),
        )
        t_hi = max(t_hi, float(ht["t_s"].max()))

    fig = plt.figure(figsize=(4.0 * ncase + 5.0, 13.0), facecolor="#0e1015")
    gs = fig.add_gridspec(3, ncase + 1, width_ratios=[1.0] * ncase + [1.2])
    for c, (tag, (_s_b, r)) in enumerate(cases.items()):
        phi_np, sp_np, ts_np = fields[tag]
        ax = fig.add_subplot(gs[0, c])
        dc.panel(ax, geo, phi_np, f"({tag}) pin fill φ", "viridis", 0.0, cfg.phi_max, fig=fig)
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
    axp.set_title("Pareto: pin-fin model", color="w", fontsize=10)
    axp.set_xscale("log")
    axp.tick_params(colors="w")
    for s_ in axp.spines.values():
        s_.set_color("w")
    axp.grid(alpha=0.2)

    fig.suptitle(
        "Pin-fin graded coldplate — K(φ): plates+Gebart series, U''(φ): prime+fin "
        f"augmentation (d={cfg.d_pin * 1e3:.0f}mm, φ_max={cfg.phi_max})\n"
        f"2x4 array, Al t_base=3mm + channel 2mm, water 5 g/s, 10 W/block | "
        f"dP_ref(all-open)={dp_ref:.1f} Pa | J = J_T + gamma*dP/dP_ref",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_pinfin.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_pinfin.png'}")

    np.savez(
        OUT / "coldplate_pinfin_result.npz",
        gammas=np.array(GAMMAS),
        **{f"s_{tag}": s_b.numpy() for tag, (s_b, _) in cases.items()},
        metrics=np.array([[r[k] for k in KEYS] for _, r in cases.values()]),
        metric_keys=np.array(KEYS),
    )
    print(f"saved: {OUT / 'coldplate_pinfin_result.npz'}")


if __name__ == "__main__":
    main()
