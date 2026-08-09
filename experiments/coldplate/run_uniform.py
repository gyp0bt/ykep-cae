"""境界任意の単一ポート + 流速/鮮度の空間均一化の検証.

設定: inlet / outlet は全境界節点から各 1 箇所 (port_side="boundary",
single_ports=True)。比較:
  (a) baseline — 均一化罰則なし (worst-block 最大化のみ)
  (b) uniform  — mu_vu (流速 CV²) + mu_phiu (ブロック位置 φ 分散) を追加

物理ノート: 単一ポートでは総鮮度収支 Σq_heat = Q·(1-φ_out) による
上流→下流の φ 勾配は消せない。均一化できるのは「各発熱体が見る φ」の
空間分布で、下流ブロックへ新鮮な水を優先経路で届ける引き回しが解になる。

実行: python run_uniform.py 2>&1 | tee logs/log-uniform-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import coldplate as cp
import numpy as np
import torch

OUT = Path(__file__).parent / "output"
CAP = 5.5  # 単一ポートで全被覆に必要な散逸上限 (run_single_port.py の結果)

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


def run(tag: str, mu_vu: float, mu_phiu: float):
    print(f"\n================ {tag} (mu_vu={mu_vu}, mu_phiu={mu_phiu}) ================")
    prob = cp.make_problem(ntu_coef=1e-2, port_side="boundary")
    cfg = cp.Config(vol_frac=0.25, mu_vu=mu_vu, mu_phiu=mu_phiu)
    t0 = time.perf_counter()
    _, x_fin, mask = cp.solve_pipeline(prob, cfg, log_p_max=CAP, single_ports=True, verbose=True)
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
    assert cc["connected"] and mc["leak_inactive"] == 0.0
    return prob, cfg, x_fin, mask, r


if __name__ == "__main__":
    cases = [
        ("baseline", 0.0, 0.0),
        ("uniform", 2.0, 25.0),
    ]
    results = {tag: run(tag, mv, mp) for tag, mv, mp in cases}

    print("\n--- baseline vs uniform ---")
    rb, ru = results["baseline"][4], results["uniform"][4]
    for k in KEYS:
        print(
            f"  {k:20s} base={rb[k]:.3f}  unif={ru[k]:.3f}"
            if isinstance(rb[k], float)
            else f"  {k:20s} base={rb[k]}  unif={ru[k]}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 18), facecolor="#0e1015")
    for col, tag in enumerate(("baseline", "uniform")):
        prob, cfg, x_fin, mask, _ = results[tag]
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
        "single port anywhere on boundary | left: baseline, right: velocity+phi uniformised | TRUE SCALE",
        color="w",
        fontsize=11,
    )
    fig.savefig(OUT / "coldplate_uniform.png", dpi=110, bbox_inches="tight")
    print(f"saved: {OUT / 'coldplate_uniform.png'}")

    prob, cfg, x_fin, mask, r = results["uniform"]
    np.savez(
        OUT / "coldplate_uniform_result.npz",
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_in_on=mask.port_in_on,
        port_out_on=mask.port_out_on,
        block_on=mask.block_on,
        mu_vu=2.0,
        mu_phiu=25.0,
        cap=CAP,
    )
    print(f"saved: {OUT / 'coldplate_uniform_result.npz'}")
