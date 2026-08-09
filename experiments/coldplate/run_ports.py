"""ポート位置の設計変数化 (HANDOVER Task 3).

左右エッジ上のガウシアン窓 (σ=2.5mm 固定) の中心 y_in / y_out を
設計場と同時に Adam で最適化する。物理はピンフィン + Forchheimer (Task 1+2)。

ネットワーク版では対角カウンターフローが選好された — 擬3D ダルシーでも
同じ選好が出るかを検証する。y 対称形状なので (下in, 上out) と (上in, 下out)
は等価。初期値を変えた 3 ケース × γ 2 水準:

  fixed-center: ポート中央固定 (ガウシアン窓での公平な基準)
  free-center:  中央から自由最適化 (対称点は勾配ゼロ → 逃げられるか)
  free-diag:    対角初期値 (in 下 12% / out 上 88%) から自由最適化

実行: python run_ports.py 2>&1 | tee logs/log-ports-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import darcy as dc
import numpy as np
import torch

OUT = Path(__file__).parent / "output"
GAMMAS = (1.0, 0.1)
ITERS = 600
CASES = (
    ("fixed-center", (0.0, 0.0), False),
    ("free-center", (0.0, 0.0), True),
    ("free-diag", (-2.0, 2.0), True),
)

KEYS = ("dp", "T_peak", "T_block_mean", "T_block_std", "T_fluid_out", "solidity_mean")


def main() -> None:
    cfg = dc.DarcyConfig(pin_fin=True, forchheimer=True)
    geo = dc.make_geo(cfg)
    dp_ref = dc.dp_reference(cfg, geo)
    h_mm = geo.ncy * geo.h * 1e3
    print(
        f"pin-fin+Forchheimer, port window σ={cfg.port_sigma * 1e3:.1f}mm, H={h_mm:.0f}mm\n"
        f"dp_ref(all-open, 固定中央 5mm)={dp_ref:.2f}Pa"
    )

    results: dict[str, dict] = {}
    for gamma in GAMMAS:
        for tag, eta0, free in CASES:
            name = f"{tag}@g={gamma}"
            print(f"\n=== {name} (eta0={eta0}, free={free}) ===")
            t0 = time.perf_counter()
            xi, eta_in, eta_out = dc.optimize_ports(
                cfg,
                geo,
                gamma_p=gamma,
                iters=ITERS,
                seed=0,
                eta0=eta0,
                free_ports=free,
                verbose=True,
            )
            ports = dc.make_ports(cfg, geo, eta_in, eta_out)
            s_b = torch.sigmoid(xi)
            r = dc.evaluate(cfg, geo, s_b, ports=ports)
            j, _ = dc.objective(cfg, geo, xi, gamma, dp_ref, ports=ports)
            y_in = h_mm * float(torch.sigmoid(eta_in))
            y_out = h_mm * float(torch.sigmoid(eta_out))
            print(f"  optimize {time.perf_counter() - t0:.0f}s")
            print(
                f"  J={float(j):.4f}  y_in={y_in:.1f}mm  y_out={y_out:.1f}mm  "
                + "  ".join(f"{k}={r[k]:.4g}" for k in KEYS)
            )
            assert r["heat_balance_rel"] < 1e-8
            results[name] = {
                "xi": xi,
                "eta": (float(eta_in), float(eta_out)),
                "y": (y_in, y_out),
                "j": float(j),
                "r": r,
                "gamma": gamma,
            }

    print("\n--- まとめ (J は同一 dp_ref 基準) ---")
    print(f"{'case':22s} {'J':>8s} {'dp[Pa]':>8s} {'T_peak':>7s} {'y_in':>6s} {'y_out':>6s}")
    for name, d in results.items():
        print(
            f"{name:22s} {d['j']:8.4f} {d['r']['dp']:8.1f} {d['r']['T_peak']:7.2f} "
            f"{d['y'][0]:6.1f} {d['y'][1]:6.1f}"
        )

    # ------------------------------------------------------------------
    # 可視化: 列 = ケース (γ ごとに 3), 行 = (φ, |u|, T_s)。ポート位置を線で表示
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    ncase = len(names)
    fig = plt.figure(figsize=(4.0 * ncase + 1.0, 13.0), facecolor="#0e1015")
    gs = fig.add_gridspec(3, ncase)
    w_mm = geo.ncx * geo.h * 1e3
    t_hi = 0.0
    fields = {}
    for name in names:
        d = results[name]
        ports = dc.make_ports(cfg, geo, torch.tensor(d["eta"][0]), torch.tensor(d["eta"][1]))
        with torch.no_grad():
            s = dc.expand(geo, torch.sigmoid(d["xi"]))
            flow = dc.solve_flow(cfg, geo, s, ports=ports)
            ht = dc.solve_heat(cfg, geo, s, flow)
        fields[name] = (cfg.phi_max * s.numpy(), dc.cell_speed(cfg, geo, flow), ht["t_s"].numpy())
        t_hi = max(t_hi, float(ht["t_s"].max()))

    for c, name in enumerate(names):
        d = results[name]
        phi_np, sp_np, ts_np = fields[name]
        panels = (
            (phi_np, "pin fill φ", "viridis", 0.0, cfg.phi_max),
            (sp_np * 1e3, "|u| [mm/s]", "magma", None, None),
            (ts_np, f"T_s [K] peak={d['r']['T_peak']:.1f}", "inferno", 0.0, t_hi),
        )
        for row, (fld, ttl, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[row, c])
            dc.panel(ax, geo, fld, f"({name}) {ttl}", cmap, vmin, vmax, fig=fig)
            y_in, y_out = d["y"]
            ax.plot([0, 0], [y_in - 4, y_in + 4], color="cyan", lw=3)
            ax.plot([w_mm, w_mm], [y_out - 4, y_out + 4], color="orange", lw=3)

    fig.suptitle(
        "Port position as design variable — Gaussian window ports (σ=2.5mm), "
        "pin-fin + Forchheimer physics\n"
        "cyan = inlet (left edge), orange = outlet (right edge) | "
        f"dP_ref={dp_ref:.1f} Pa | J = J_T + gamma*dP/dP_ref",
        color="w",
        fontsize=12,
    )
    fig.savefig(OUT / "coldplate_ports.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: {OUT / 'coldplate_ports.png'}")

    np.savez(
        OUT / "coldplate_ports_result.npz",
        names=np.array(names),
        **{f"xi_{n}": results[n]["xi"].numpy() for n in names},
        **{f"eta_{n}": np.array(results[n]["eta"]) for n in names},
        metrics=np.array([[results[n]["r"][k] for k in KEYS] for n in names]),
        js=np.array([results[n]["j"] for n in names]),
        metric_keys=np.array(KEYS),
    )
    print(f"saved: {OUT / 'coldplate_ports_result.npz'}")


if __name__ == "__main__":
    main()
