"""log_p_max (散逸上限) を変えてフル問題を実行し、被覆と均一性を比較する.

実行: python run_compare.py 2>&1 | tee logs/log-compare-$(date +%s).log
"""

from __future__ import annotations

import time

import coldplate as cp
import numpy as np


def run(cap: float) -> None:
    print(f"\n================ log_p_max = {cap} ================")
    prob = cp.make_problem(ntu_coef=1e-2)
    cfg = cp.Config(vol_frac=0.25)
    t0 = time.perf_counter()
    x_raw, x_fin, mask = cp.solve_pipeline(prob, cfg, log_p_max=cap, verbose=True)
    print(f"pipeline {time.perf_counter() - t0:.0f}s")
    for tag, xx, mm in (("raw", x_raw, None), ("pruned", x_fin, mask)):
        r = cp.report(prob, cfg, xx, mm)
        mc = cp.mass_check(prob, cfg, xx, mm)
        keys = (
            "cooling",
            "blocks_covered",
            "block_cv",
            "block_min_over_mean",
            "log_diss",
            "grey",
            "eff_ports_in",
            "eff_ports_out",
            "leak_flow_frac",
        )
        vals = "  ".join(
            f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys
        )
        print(f"  [{tag:6s}] {vals}")
        print(f"  [{tag:6s}] mass_check {mc}")
    cc = cp.connectivity_check(prob, cfg, x_fin, mask=mask)
    print(f"  connectivity {cc}")
    np.savez(
        f"output/compare_cap{cap:.1f}.npz",
        x_raw=x_raw,
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_on=mask.port_on,
        block_on=mask.block_on,
    )


if __name__ == "__main__":
    for cap in (5.0, 3.5):
        run(cap)
