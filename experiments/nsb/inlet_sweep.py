"""質量流量を固定して inlet の位置・幅・壁を変える探索デモ（flat 72×48）.

冷却流路設計の前段: 流量 ṁ を固定し、inlet の位置とサイズで圧力損失がどう変わるかを見る。
使用例::

    python experiments/nsb/inlet_sweep.py 2>&1 | tee experiments/nsb/logs/inlet-sweep-$(date +%s).log
"""

from __future__ import annotations

from pathlib import Path

import yaml

from nsb import BC, NSBSettings, make_case, solve_steady
from nsb.geo import LX, LY
from nsb.utils import inlet_velocity, summary
from xkep_cae_fluid.brinkman_flow import east_span, north_span, west_span

HERE = Path(__file__).resolve().parent
MDOT = 0.1  # kg/s（h=1e-3, 幅 0.1 で U=1 m/s 相当）


def settings_for(inp) -> NSBSettings:
    u = inlet_velocity(inp)
    return NSBSettings(
        cfl_init=0.5,
        velocity_floor=0.1 * u,
        pseudo_time_in_residual=False,
        alpha_u=1.0,
        init_field="stokes",
        newton_max_iter=150,
    )


def main() -> None:
    outlet = BC.pressure_outlet(west_span(0.05, 0.15))
    cases: dict[str, BC] = {}
    for y0, y1 in [(0.25, 0.35), (0.30, 0.35), (0.20, 0.35), (0.15, 0.35), (0.30, 0.40)]:
        cases[f"west_y{y0:g}-{y1:g}"] = BC(
            patches=(BC.mass_flow_inlet(west_span(y0, y1), MDOT), outlet)
        )
    for x0, x1 in [(0.30, 0.40), (0.55, 0.65)]:
        cases[f"north_x{x0:g}-{x1:g}"] = BC(
            patches=(BC.mass_flow_inlet(north_span(x0, x1, LY), MDOT), outlet)
        )
    cases["east_y0.25-0.35"] = BC(
        patches=(BC.mass_flow_inlet(east_span(0.25, 0.35, LX), MDOT), outlet)
    )

    out: dict[str, dict] = {}
    for name, bc in cases.items():
        print(f"===== {name} =====", flush=True)
        inp = make_case("flat", 1, bc=bc)
        inp = make_case("flat", 1, bc=bc, settings=settings_for(inp))
        res = solve_steady(inp, log=lambda m: print(m, flush=True))
        out[name] = summary(inp, res)
        out[name]["u_inlet"] = float(inlet_velocity(inp))
        print(f"--> {name}: {out[name]}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "inlet_sweep_flat_r1.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"\n==== summary (mdot fixed = {MDOT:.3g} kg/s) ====")
    for k, v in out.items():
        print(
            f"{k:20s} conv={v['converged']!s:5s} it={v['n_iter']:3d} u_in={v['u_inlet']:.3f} "
            f"p_in={v['p_inlet_mean']:9.1f} Pa  speed_max={v['speed_max']:.3f}  m_out/m_in={v['mass_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
