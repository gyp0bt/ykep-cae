"""γ 連続スイープによるパレート前線の密化 (HANDOVER Task 5).

γ を対数等間隔 15 点 (30 → 0.01, 降順) で振り、直前の解からウォームスタート
して重み付き和を連続に辿る。物理はピンフィン・ダルシーで最適化し、得られた
各設計を Forchheimer 込みでも再評価して両方の ΔP でパレートを描く
(慣性補正は評価コストが安いので全点で正直に載せる)。

- 最初の点 (γ=30) はコールドスタート 600 反復、以降は 300 反復
- 既存 4 点 (run_pinfin, コールドスタート) と重なる γ では J を比較し、
  ウォームスタートが劣化していないことを確認する

実行: python run_sweep.py 2>&1 | tee logs/log-sweep-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import darcy as dc
import numpy as np
import torch

torch.set_num_threads(4)

OUT = Path(__file__).parent / "output"
GAMMAS = np.geomspace(30.0, 0.01, 15)
ITERS_COLD = 600
ITERS_WARM = 300

KEYS = ("dp", "T_peak", "T_block_mean", "T_block_std", "T_fluid_out", "solidity_mean")


def main() -> None:
    cfg = dc.DarcyConfig(pin_fin=True)
    cfg_f = dc.DarcyConfig(pin_fin=True, forchheimer=True)
    geo = dc.make_geo(cfg)
    dp_ref = dc.dp_reference(cfg, geo)
    print(
        f"gamma sweep: {len(GAMMAS)} points geomspace({GAMMAS[0]:.3g} -> {GAMMAS[-1]:.3g}), "
        f"warm-start descending | pin-fin Darcy optimize + D-F re-evaluation\n"
        f"dp_ref(all-open)={dp_ref:.2f}Pa"
    )

    rows = []
    s_fields = {}
    xi = None
    for i, gamma in enumerate(GAMMAS):
        iters = ITERS_COLD if i == 0 else ITERS_WARM
        t0 = time.perf_counter()
        xi = dc.optimize(cfg, geo, gamma_p=float(gamma), iters=iters, seed=0, xi0=xi)
        s_b = torch.sigmoid(xi)
        r_d = dc.evaluate(cfg, geo, s_b)
        r_f = dc.evaluate(cfg_f, geo, s_b)
        j, _ = dc.objective(cfg, geo, xi, float(gamma), dp_ref)
        assert r_d["heat_balance_rel"] < 1e-8
        rows.append(
            {
                "gamma": float(gamma),
                "j": float(j),
                "dp_d": r_d["dp"],
                "dp_f": r_f["dp"],
                **{k: r_d[k] for k in KEYS if k != "dp"},
            }
        )
        s_fields[f"s_g{gamma:.4g}"] = s_b.numpy()
        print(
            f"  g={gamma:8.4g} ({iters}it {time.perf_counter() - t0:3.0f}s)  J={float(j):7.4f}  "
            f"dp_D={r_d['dp']:7.1f}  dp_DF={r_f['dp']:7.1f}  T_peak={r_d['T_peak']:6.2f}  "
            f"solidity={r_d['solidity_mean']:.3f}"
        )

    # 既存コールドスタート 4 点との照合 (ウォームスタートの劣化チェック)
    pin = np.load(OUT / "coldplate_pinfin_result.npz")
    lk = list(pin["metric_keys"])
    print("\n--- コールドスタート (run_pinfin) との J 比較 ---")
    for g_ref in pin["gammas"]:
        i_near = int(np.argmin(np.abs(GAMMAS - g_ref)))
        s_cold = torch.tensor(pin[f"s_g={g_ref}"])
        xi_cold = torch.logit(s_cold.clamp(1e-9, 1 - 1e-9))
        j_cold, _ = dc.objective(cfg, geo, xi_cold, float(GAMMAS[i_near]), dp_ref)
        print(
            f"  γ={GAMMAS[i_near]:.4g} (cold設計をγ同値で評価 J={float(j_cold):.4f}) "
            f"vs warm J={rows[i_near]['j']:.4f}"
        )

    # 全開基準
    r_open_d = dc.evaluate(cfg, geo, torch.zeros(geo.by, geo.bx))
    r_open_f = dc.evaluate(cfg_f, geo, torch.zeros(geo.by, geo.bx))

    # ------------------------------------------------------------------
    # 可視化: パレート前線 (密) + 代表 5 設計の φ 場
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19, 10), facecolor="#0e1015")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.6, 1.0])

    axp = fig.add_subplot(gs[0, :])
    axp.set_facecolor("#171a21")
    dps_d = [r["dp_d"] for r in rows]
    dps_f = [r["dp_f"] for r in rows]
    tps = [r["T_peak"] for r in rows]
    axp.plot(dps_d, tps, "o-", color="#6fbf73", lw=1.5, label="Darcy ΔP (optimized physics)")
    axp.plot(dps_f, tps, "s--", color="#e0a458", lw=1.2, label="ΔP re-evaluated with Forchheimer")
    for r in rows:
        axp.annotate(
            f"{r['gamma']:.3g}",
            (r["dp_d"], r["T_peak"]),
            textcoords="offset points",
            xytext=(6, 5),
            color="0.8",
            fontsize=7,
        )
    # 旧 4 点 (cold) を重ねる
    idx_dp, idx_tp = lk.index("dp"), lk.index("T_peak")
    axp.scatter(
        pin["metrics"][1:, idx_dp],
        pin["metrics"][1:, idx_tp],
        s=70,
        marker="x",
        color="#7f9fe0",
        label="cold-start 4 pts (run_pinfin)",
        zorder=4,
    )
    axp.scatter([r_open_d["dp"]], [r_open_d["T_peak"]], s=70, color="w", zorder=4)
    axp.annotate(
        f"all-open ({r_open_d['dp']:.0f}→{r_open_f['dp']:.0f} Pa w/ DF)",
        (r_open_d["dp"], r_open_d["T_peak"]),
        textcoords="offset points",
        xytext=(8, -4),
        color="w",
        fontsize=8,
    )
    axp.set_xscale("log")
    axp.set_xlabel("ΔP [Pa]", color="w")
    axp.set_ylabel("T_peak [K over inlet]", color="w")
    axp.set_title(
        f"Dense Pareto front — {len(GAMMAS)} γ points, warm-started descending", color="w"
    )
    axp.legend(facecolor="#171a21", labelcolor="w", edgecolor="0.4", fontsize=8)
    axp.tick_params(colors="w")
    for s_ in axp.spines.values():
        s_.set_color("w")
    axp.grid(alpha=0.2, which="both")

    show_idx = [0, 3, 7, 11, 14]
    for c, i in enumerate(show_idx):
        ax = fig.add_subplot(gs[1, c])
        g = rows[i]["gamma"]
        dc.panel(
            ax,
            geo,
            cfg.phi_max * dc.expand(geo, torch.tensor(s_fields[f"s_g{g:.4g}"])).numpy(),
            f"γ={g:.3g}  φ field  T_peak={rows[i]['T_peak']:.1f}K",
            "viridis",
            0.0,
            cfg.phi_max,
            fig=fig,
        )

    fig.suptitle(
        "Gamma continuation sweep (Task 5) — pin-fin model, seed=0, warm start | "
        f"dP_ref={dp_ref:.1f} Pa | J = J_T + gamma*dP/dP_ref",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_sweep.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_sweep.png'}")

    np.savez(
        OUT / "coldplate_sweep_result.npz",
        gammas=GAMMAS,
        metrics=np.array(
            [
                [r["gamma"], r["j"], r["dp_d"], r["dp_f"], r["T_peak"], r["T_block_std"]]
                for r in rows
            ]
        ),
        metric_keys=np.array(["gamma", "j", "dp_darcy", "dp_forchheimer", "T_peak", "T_block_std"]),
        **s_fields,
    )
    print(f"saved: {OUT / 'coldplate_sweep_result.npz'}")


if __name__ == "__main__":
    main()
