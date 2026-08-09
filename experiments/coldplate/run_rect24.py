"""発熱源 2 列 × 4 行 (長方形配置) での境界任意 single-port 実行.

散逸上限を掃引して被覆優先で baseline を選び、同条件で均一化あり
(mu_vu=2, mu_phiu=25) を比較する。

実行: python run_rect24.py 2>&1 | tee logs/log-rect24-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import coldplate as cp
import numpy as np
import torch

OUT = Path(__file__).parent / "output"

KEYS = (
    "cooling",
    "blocks_covered",
    "block_min_over_mean",
    "log_diss",
    "vel_cv",
    "phi_block_mean",
    "phi_block_std",
    "phi_block_min",
    "phi_block_max",
)


def make() -> cp.Problem:
    return cp.make_problem(n_cols=2, n_rows=4, ntu_coef=1e-2, port_side="boundary")


def describe_ports(prob: cp.Problem, cfg: cp.Config, x: np.ndarray, mask: cp.DesignMask) -> None:
    with torch.no_grad():
        st = cp.state(prob, cfg, torch.tensor(x), mask)
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    xy = prob.graph.coords()
    pn = prob.ports.numpy()
    for k in np.nonzero(yi > 0.01)[0]:
        i, j = xy[pn[k]]
        print(f"  inlet : node ({int(i)},{int(j)})  share={yi[k]:.3f}")
    for k in np.nonzero(yo > 0.01)[0]:
        i, j = xy[pn[k]]
        print(f"  outlet: node ({int(i)},{int(j)})  share={yo[k]:.3f}")


def run(tag: str, cap: float, mu_vu: float, mu_phiu: float):
    print(
        f"\n================ {tag} (cap={cap}, mu_vu={mu_vu}, mu_phiu={mu_phiu}) ================"
    )
    prob = make()
    gr = prob.graph
    print(
        f"grid {gr.nx}x{gr.ny}  nodes={gr.n_nodes}  edges={gr.n_edges}  "
        f"blocks={len(prob.block_edges)}  ports={len(prob.ports)}"
    )
    cfg = cp.Config(vol_frac=0.25, mu_vu=mu_vu, mu_phiu=mu_phiu)
    t0 = time.perf_counter()
    _, x_fin, mask = cp.solve_pipeline(prob, cfg, log_p_max=cap, single_ports=True, verbose=True)
    print(f"pipeline {time.perf_counter() - t0:.0f}s")
    r = cp.report(prob, cfg, x_fin, mask)
    print(
        "  "
        + "  ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in KEYS)
    )
    mc = cp.mass_check(prob, cfg, x_fin, mask)
    cc = cp.connectivity_check(prob, cfg, x_fin, mask=mask)
    print(f"  mass_check {mc}")
    print(f"  connectivity {cc}")
    describe_ports(prob, cfg, x_fin, mask)
    with torch.no_grad():
        st = cp.state(prob, cfg, torch.tensor(x_fin), mask)
    pb = np.array([float(st["q_heat"][e].sum()) for e in prob.block_edges])
    print("  per-block heat:", np.array2string(pb, precision=4))
    assert cc["connected"] and mc["leak_inactive"] == 0.0
    return prob, cfg, x_fin, mask, r


if __name__ == "__main__":
    # baseline: cap 掃引 -> 被覆優先で採用
    base = {}
    for cap in (3.0, 4.0, 5.0):
        base[cap] = run(f"baseline cap={cap}", cap, 0.0, 0.0)
    pick = max(base, key=lambda c: (base[c][4]["blocks_covered"], base[c][4]["cooling"]))
    print(f"\nselected baseline cap = {pick}")

    uni = run("uniform", pick, 2.0, 25.0)

    rb, ru = base[pick][4], uni[4]
    print("\n--- baseline vs uniform (2x4) ---")
    for k in KEYS:
        print(
            f"  {k:20s} base={rb[k]:.3f}  unif={ru[k]:.3f}"
            if isinstance(rb[k], float)
            else f"  {k:20s} base={rb[k]}  unif={ru[k]}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 15), facecolor="#0e1015")
    for col, (tag, res) in enumerate((("baseline", base[pick]), ("uniform", uni))):
        prob, cfg, x_fin, mask, _ = res
        cp.plot(prob, cfg, x_fin, fig, axes[0, col], title=f"({'ab'[col]}) {tag} - phi", mask=mask)
        cp.plot(
            prob,
            cfg,
            x_fin,
            fig,
            axes[1, col],
            title=f"({'cd'[col]}) {tag} - velocity |q|/w^2",
            color_by="v",
            mask=mask,
        )
    fig.suptitle(
        f"2x4 heater array, single port on boundary (cap={pick}) | top: phi | bottom: velocity | TRUE SCALE",
        color="w",
        fontsize=11,
    )
    fig.savefig(OUT / "coldplate_rect24.png", dpi=110, bbox_inches="tight")
    print(f"saved: {OUT / 'coldplate_rect24.png'}")

    prob, cfg, x_fin, mask, r = uni
    np.savez(
        OUT / "coldplate_rect24_result.npz",
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_in_on=mask.port_in_on,
        port_out_on=mask.port_out_on,
        block_on=mask.block_on,
        cap=pick,
    )
    print(f"saved: {OUT / 'coldplate_rect24_result.npz'}")
