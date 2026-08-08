"""固体伝熱レイヤー (mode="thermal") の実行: 2 列 × 4 行、境界任意 single-port.

3 ケースを同一条件 (cap=4.0, Al 6061 t=3mm, 水 5 g/s, 10 W/block) で比較:
  fresh   — 従来の鮮度スカラー目的で位相を解く → 温度は事後評価のみ
  thermal — ブロック温度 smooth-max (ピーク温度最小化) を目的に位相から解く
  t-unif  — thermal + ブロック温度分散罰則 (温度均一化)

実行: python run_thermal.py 2>&1 | tee logs/log-thermal-$(date +%s).log
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import coldplate as cp
import numpy as np
import torch

OUT = Path(__file__).parent / "output"
CAP = 4.0

KEYS = (
    "T_peak",
    "T_block_mean",
    "T_block_std",
    "T_block_min",
    "T_fluid_out",
    "heat_balance_rel",
    "blocks_covered",
    "log_diss",
    "vel_cv",
)


def make() -> cp.Problem:
    return cp.make_problem(n_cols=2, n_rows=4, ntu_coef=1e-2, port_side="boundary")


def describe(prob: cp.Problem, cfg: cp.Config, x: np.ndarray, mask: cp.DesignMask) -> None:
    with torch.no_grad():
        st = cp.state(prob, cfg, torch.tensor(x), mask)
        th = cp.thermal(prob, cfg, st["rho"], st["w"], st["q"], st["b"])
        t_b = cp.block_temps(prob, th["T_s"]).numpy()
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    xy = prob.graph.coords()
    pn = prob.ports.numpy()
    k_in, k_out = int(np.argmax(yi)), int(np.argmax(yo))
    print(
        f"  inlet ({int(xy[pn[k_in], 0])},{int(xy[pn[k_in], 1])})  outlet "
        f"({int(xy[pn[k_out], 0])},{int(xy[pn[k_out], 1])})"
    )
    print("  per-block T [K over inlet]:", np.array2string(t_b, precision=2))


def run(tag: str, mode: str, mu_tvar: float):
    print(f"\n================ {tag} (mode={mode}, mu_tvar={mu_tvar}) ================")
    prob = make()
    cfg = cp.Config(vol_frac=0.25, mode=mode, mu_tvar=mu_tvar)
    print(
        f"T_ref = {cp.t_ref_scale(prob, cfg):.3f} K  "
        f"(Q={cfg.q_block_w * len(prob.block_nodes):.0f} W, m_dot={cfg.m_dot * 1e3:.1f} g/s)"
    )
    t0 = time.perf_counter()
    _, x_fin, mask = cp.solve_pipeline(prob, cfg, log_p_max=CAP, single_ports=True, verbose=True)
    print(f"pipeline {time.perf_counter() - t0:.0f}s")
    # 温度指標は mode="thermal" の評価 cfg で統一 (fresh 解も同じ物理で事後評価)
    cfg_eval = replace(cfg, mode="thermal")
    r = cp.report(prob, cfg_eval, x_fin, mask)
    print(
        "  "
        + "  ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in KEYS)
    )
    mc = cp.mass_check(prob, cfg, x_fin, mask)
    cc = cp.connectivity_check(prob, cfg, x_fin, mask=mask)
    print(f"  mass_check {mc}")
    print(f"  connectivity {cc}")
    describe(prob, cfg_eval, x_fin, mask)
    assert cc["connected"] and mc["leak_inactive"] == 0.0
    assert r["heat_balance_rel"] < 1e-6
    return prob, cfg_eval, x_fin, mask, r


if __name__ == "__main__":
    cases = {
        "fresh": run("fresh (phi objective)", "fresh", 0.0),
        "thermal": run("thermal (peak-T objective)", "thermal", 0.0),
        "t-unif": run("t-unif (peak-T + variance)", "thermal", 10.0),
    }

    print("\n--- 比較 (2x4, cap=4.0, 全ケース温度は同一物理で評価) ---")
    for k in KEYS:
        row = "  ".join(
            f"{tag}={cases[tag][4][k]:.3f}"
            if isinstance(cases[tag][4][k], float)
            else f"{tag}={cases[tag][4][k]}"
            for tag in cases
        )
        print(f"  {k:18s} {row}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 温度スケールは 3 ケース共通
    t_lo, t_hi = np.inf, -np.inf
    for prob, cfg_eval, x_fin, mask, _ in cases.values():
        with torch.no_grad():
            st = cp.state(prob, cfg_eval, torch.tensor(x_fin), mask)
            th = cp.thermal(prob, cfg_eval, st["rho"], st["w"], st["q"], st["b"])
        t_lo = min(t_lo, float(th["T_s"].min()))
        t_hi = max(t_hi, float(th["T_s"].max()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 12), facecolor="#0e1015")
    for col, (tag, res) in enumerate(cases.items()):
        prob, cfg_eval, x_fin, mask, _ = res
        cp.plot(
            prob,
            cfg_eval,
            x_fin,
            fig,
            axes[0, col],
            title=f"({'abc'[col]}) {tag} - solid T",
            mask=mask,
            color_by="T",
            t_range=(t_lo, t_hi),
        )
        cp.plot(
            prob,
            cfg_eval,
            x_fin,
            fig,
            axes[1, col],
            title=f"({'def'[col]}) {tag} - velocity |q|/w^2",
            mask=mask,
            color_by="v",
        )
    fig.suptitle(
        f"2x4 array, solid conduction layer (Al t=3mm, water {5.0:.0f} g/s, 10 W/block, "
        f"cap={CAP}) | top: T_s [{t_lo:.1f}, {t_hi:.1f}] K | bottom: velocity | TRUE SCALE",
        color="w",
        fontsize=11,
    )
    fig.savefig(OUT / "coldplate_thermal.png", dpi=110, bbox_inches="tight")
    print(f"saved: {OUT / 'coldplate_thermal.png'}")

    prob, cfg_eval, x_fin, mask, r = cases["t-unif"]
    np.savez(
        OUT / "coldplate_thermal_result.npz",
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_in_on=mask.port_in_on,
        port_out_on=mask.port_out_on,
        block_on=mask.block_on,
        cap=CAP,
    )
    print(f"saved: {OUT / 'coldplate_thermal_result.npz'}")
