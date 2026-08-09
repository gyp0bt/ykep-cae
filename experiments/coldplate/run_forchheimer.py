"""Forchheimer 慣性補正の効果検証 (HANDOVER Task 2).

Part A: ピンフィン最適化済み設計 (coldplate_pinfin_result.npz) を
        ダルシー / ダルシー+Forchheimer の両物理で評価し、ΔP 補正量と
        間隙 Re を定量化する (慣性損失は ΔP の絶対値にどれだけ効くか)。
Part B: Forchheimer 物理のまま γ を再最適化し、設計がどれだけ動くかを見る
        (ダルシー設計を D-F 物理で評価 vs D-F 物理で直接最適化)。

実行: python run_forchheimer.py 2>&1 | tee logs/log-forchheimer-$(date +%s).log
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


def re_gap(cfg: dc.DarcyConfig, geo: dc.Geo, s_b: torch.Tensor) -> float:
    """設計場の最大セル流速から間隙 Reynolds 数 Re = ρ·u·t/μ を出す."""
    with torch.no_grad():
        s = dc.expand(geo, s_b)
        flow = dc.solve_flow(cfg, geo, s)
    u_max = float(dc.cell_speed(cfg, geo, flow).max())
    return cfg.rho_f * u_max * cfg.t_chan / cfg.mu


def main() -> None:
    cfg_d = dc.DarcyConfig(pin_fin=True)
    cfg_f = dc.DarcyConfig(pin_fin=True, forchheimer=True)
    geo = dc.make_geo(cfg_d)

    pin = np.load(OUT / "coldplate_pinfin_result.npz")
    tags = ["all-open"] + [f"g={g}" for g in pin["gammas"]]

    # ------------------------------------------------------------------
    # Part A: 同一設計での ΔP 比較 (ダルシー vs D-F)
    # ------------------------------------------------------------------
    print("=== Part A: 同一設計の ΔP — ダルシー vs ダルシー+Forchheimer ===")
    print(
        f"{'case':10s} {'dp_D[Pa]':>9s} {'dp_DF[Pa]':>10s} {'ratio':>6s} {'Re_gap':>7s} {'T_peak_D':>9s} {'T_peak_DF':>10s}"
    )
    for tag in tags:
        s_b = torch.tensor(pin[f"s_{tag}"])
        r_d = dc.evaluate(cfg_d, geo, s_b)
        r_f = dc.evaluate(cfg_f, geo, s_b)
        re = re_gap(cfg_d, geo, s_b)
        print(
            f"{tag:10s} {r_d['dp']:9.1f} {r_f['dp']:10.1f} {r_f['dp'] / r_d['dp']:6.2f} "
            f"{re:7.0f} {r_d['T_peak']:9.2f} {r_f['T_peak']:10.2f}"
        )

    # ------------------------------------------------------------------
    # Part B: D-F 物理での再最適化
    # ------------------------------------------------------------------
    print("\n=== Part B: Forchheimer 物理での再最適化 ===")
    dp_ref = dc.dp_reference(cfg_f, geo)
    t_ref = dc.t_ref_scale(cfg_f, geo)
    print(f"dp_ref(all-open)={dp_ref:.2f}Pa  T_ref={t_ref:.3f}K")

    cases: dict[str, tuple[torch.Tensor, dict[str, float]]] = {}
    s_open = torch.zeros(geo.by, geo.bx)
    cases["all-open"] = (s_open, dc.evaluate(cfg_f, geo, s_open))

    for gamma in GAMMAS:
        print(f"\n--- gamma={gamma} ---")
        t0 = time.perf_counter()
        xi = dc.optimize(cfg_f, geo, gamma_p=gamma, iters=ITERS, seed=0, verbose=True)
        s_b = torch.sigmoid(xi)
        r = dc.evaluate(cfg_f, geo, s_b)
        print(f"  optimize {time.perf_counter() - t0:.0f}s")
        print("  " + "  ".join(f"{k}={r[k]:.4g}" for k in KEYS))
        assert r["heat_balance_rel"] < 1e-8
        cases[f"g={gamma}"] = (s_b, r)

        # ダルシー設計を D-F 物理で評価した場合との差 (設計シフトの利得)
        s_darcy = torch.tensor(pin[f"s_g={gamma}"])
        r_cross = dc.evaluate(cfg_f, geo, s_darcy)
        print(
            f"  [設計シフト] ダルシー設計を DF 評価: dp={r_cross['dp']:.1f} "
            f"T_peak={r_cross['T_peak']:.2f} | DF 直接最適化: dp={r['dp']:.1f} "
            f"T_peak={r['T_peak']:.2f}"
        )

    print("\n--- 比較 (pin-fin + Forchheimer) ---")
    for k in KEYS:
        print(f"  {k:18s} " + "  ".join(f"{tag}={r[k]:.4g}" for tag, (_, r) in cases.items()))

    # ------------------------------------------------------------------
    # 可視化 (run_pinfin と同レイアウト) + npz
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
            flow = dc.solve_flow(cfg_f, geo, s)
            ht = dc.solve_heat(cfg_f, geo, s, flow)
        fields[tag] = (
            cfg_f.phi_max * s.numpy(),
            dc.cell_speed(cfg_f, geo, flow),
            ht["t_s"].numpy(),
        )
        t_hi = max(t_hi, float(ht["t_s"].max()))

    fig = plt.figure(figsize=(4.0 * ncase + 5.0, 13.0), facecolor="#0e1015")
    gs = fig.add_gridspec(3, ncase + 1, width_ratios=[1.0] * ncase + [1.2])
    for c, (tag, (_s_b, r)) in enumerate(cases.items()):
        phi_np, sp_np, ts_np = fields[tag]
        ax = fig.add_subplot(gs[0, c])
        dc.panel(ax, geo, phi_np, f"({tag}) pin fill φ", "viridis", 0.0, cfg_f.phi_max, fig=fig)
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
    # ダルシー最適化設計の DF 評価も重ねる (設計シフトの視覚化)
    for gamma in GAMMAS:
        s_darcy = torch.tensor(pin[f"s_g={gamma}"])
        r_cross = dc.evaluate(cfg_f, geo, s_darcy)
        axp.scatter(r_cross["dp"], r_cross["T_peak"], s=30, marker="x", color="0.6", zorder=2)
    axp.set_xlabel("ΔP [Pa]", color="w")
    axp.set_ylabel("T_peak [K over inlet]", color="w")
    axp.set_title("Pareto: D-F optimized (o) vs Darcy designs under D-F (x)", color="w", fontsize=9)
    axp.set_xscale("log")
    axp.tick_params(colors="w")
    for s_ in axp.spines.values():
        s_.set_color("w")
    axp.grid(alpha=0.2)

    fig.suptitle(
        "Pin-fin + Forchheimer (Ergun c_E=1.75) — inertia-corrected 2-objective "
        f"optimization\n2x4 array, water 5 g/s, 10 W/block | dP_ref={dp_ref:.1f} Pa | "
        "J = J_T + gamma*dP/dP_ref",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_forchheimer.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_forchheimer.png'}")

    np.savez(
        OUT / "coldplate_forchheimer_result.npz",
        gammas=np.array(GAMMAS),
        **{f"s_{tag}": s_b.numpy() for tag, (s_b, _) in cases.items()},
        metrics=np.array([[r[k] for k in KEYS] for _, r in cases.values()]),
        metric_keys=np.array(KEYS),
    )
    print(f"saved: {OUT / 'coldplate_forchheimer_result.npz'}")


if __name__ == "__main__":
    main()
