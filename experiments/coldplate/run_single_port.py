"""single-port モード (物理ポート in/out 各 1 箇所) のフル問題実行.

分散ポート解 (output/coldplate_v2_result.npz) と比較し、
「分配を流路パターン (ヘッダ木) で実現する」設計のコストを定量化する。

実行: python run_single_port.py 2>&1 | tee logs/log-single-port-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import coldplate as cp
import numpy as np
import torch

OUT = Path(__file__).parent / "output"


def describe_ports(prob: cp.Problem, cfg: cp.Config, x: np.ndarray, mask: cp.DesignMask) -> None:
    with torch.no_grad():
        st = cp.state(prob, cfg, torch.tensor(x), mask)
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    for k in np.nonzero(yi > 0.01)[0]:
        print(f"  inlet : j={k:2d}  share={yi[k]:.3f}")
    for k in np.nonzero(yo > 0.01)[0]:
        print(f"  outlet: j={k:2d}  share={yo[k]:.3f}")


def run(cap: float) -> tuple[np.ndarray, cp.DesignMask, dict, cp.Problem, cp.Config]:
    prob = cp.make_problem(ntu_coef=1e-2)
    cfg = cp.Config(vol_frac=0.25)
    t0 = time.perf_counter()
    _, x_fin, mask = cp.solve_pipeline(prob, cfg, log_p_max=cap, single_ports=True, verbose=True)
    print(f"pipeline {time.perf_counter() - t0:.0f}s")
    r = cp.report(prob, cfg, x_fin, mask)
    return x_fin, mask, r, prob, cfg


if __name__ == "__main__":
    # 散逸上限: 分散ポートの正規値 3.5 で試し、罰則が飽和するなら緩める
    results = {}
    for cap in (3.5, 4.5, 5.5):
        print(f"\n================ single-port, log_p_max = {cap} ================")
        x_fin, mask, r, prob, cfg = run(cap)
        keys = (
            "cooling",
            "blocks_covered",
            "block_cv",
            "block_min_over_mean",
            "log_diss",
            "eff_ports_in",
            "eff_ports_out",
        )
        print(
            "  "
            + "  ".join(
                f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys
            )
        )
        mc = cp.mass_check(prob, cfg, x_fin, mask)
        cc = cp.connectivity_check(prob, cfg, x_fin, mask=mask)
        print(f"  mass_check {mc}")
        print(f"  connectivity {cc}")
        describe_ports(prob, cfg, x_fin, mask)
        assert cc["connected"] and mc["leak_inactive"] == 0.0
        results[cap] = (x_fin, mask, r, prob, cfg)

    # 散逸上限内 (log_diss <= cap + 0.05) の解のうち cooling 最良を採用
    feasible = {c: v for c, v in results.items() if v[2]["log_diss"] <= c + 0.05}
    pick = max(feasible or results, key=lambda c: (feasible or results)[c][2]["cooling"])
    x_fin, mask, r, prob, cfg = results[pick]
    print(f"\nselected cap = {pick} (log_diss = {r['log_diss']:.3f})")

    # 分散ポート解との比較
    d = np.load(OUT / "coldplate_v2_result.npz")
    dist_mask = cp.DesignMask(
        d["edge_on"],
        d.get("port_in_on", d.get("port_on")),
        d.get("port_out_on", d.get("port_on")),
        d["block_on"],
    )
    x_dist = d["x_final"]
    r_dist = cp.report(prob, cfg, x_dist, dist_mask)
    print("\n--- distributed vs single-port ---")
    for k in ("cooling", "blocks_covered", "block_cv", "block_min_over_mean", "log_diss"):
        print(
            f"  {k:20s} dist={r_dist[k]:.3f}  single={r[k]:.3f}"
            if isinstance(r[k], float)
            else f"  {k:20s} dist={r_dist[k]}  single={r[k]}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 18), facecolor="#0e1015")
    cp.plot(prob, cfg, x_dist, fig, axes[0, 0], title="(a) distributed ports - phi", mask=dist_mask)
    cp.plot(prob, cfg, x_fin, fig, axes[0, 1], title="(b) SINGLE ports - phi", mask=mask)
    cp.plot(
        prob,
        cfg,
        x_dist,
        fig,
        axes[1, 0],
        title="(c) distributed - velocity |q|/w^2",
        color_by="v",
        mask=dist_mask,
    )
    cp.plot(
        prob,
        cfg,
        x_fin,
        fig,
        axes[1, 1],
        title="(d) SINGLE - velocity |q|/w^2",
        color_by="v",
        mask=mask,
    )
    fig.suptitle(
        "distributed vs single physical ports | top: freshness phi | bottom: velocity | TRUE SCALE",
        color="w",
        fontsize=11,
    )
    fig.savefig(OUT / "coldplate_single_port.png", dpi=110, bbox_inches="tight")
    print(f"saved: {OUT / 'coldplate_single_port.png'}")

    np.savez(
        OUT / "coldplate_single_port_result.npz",
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_in_on=mask.port_in_on,
        port_out_on=mask.port_out_on,
        block_on=mask.block_on,
        cap=pick,
    )
    print(f"saved: {OUT / 'coldplate_single_port_result.npz'}")
