"""領域内マニホールドのデモ: 紙面垂直方向の注入/吸出マニホールドを平板の内部に置く（flat 72×48）.

  A. 注入マニホールド（円板、ṁ=0.1 kg/s）→ 左壁 outlet
  B. 注入マニホールド → 圧力指定マニホールド（境界 outlet なし、圧力基準はマニホールド側）
  C. 注入 2 か所（合計 ṁ）→ 圧力指定マニホールド 1 か所（冷却プレートのヘッダ配置例）

使用例::

    python experiments/nsb/manifold_demo.py 2>&1 | tee experiments/nsb/logs/manifold-demo-$(date +%s).log
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from nsb import BC, NSBSettings, make_case, solve_steady
from nsb.utils import save_fields, summary
from xkep_cae_fluid.brinkman_flow import disk_mask, west_span
from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization

HERE = Path(__file__).resolve().parent
MDOT = 0.1  # kg/s
COND = 1e-4  # kg/(s·Pa): 圧力指定マニホールドのコンダクタンス（Δp=1 kPa で 0.1 kg/s）


def settings_for(inp) -> NSBSettings:
    u = BrinkmanDiscretization(inp.to_flow_input()).u_scale
    return NSBSettings(
        cfl_init=0.5,
        velocity_floor=0.1 * u,
        pseudo_time_in_residual=False,
        alpha_u=1.0,
        init_field="stokes",
        newton_max_iter=150,
    )


def main() -> None:
    def two_in(x, y):
        return ((x - 0.15) ** 2 + (y - 0.1) ** 2 < 0.04**2) | (
            (x - 0.15) ** 2 + (y - 0.3) ** 2 < 0.04**2
        )

    cases: dict[str, BC] = {
        "A_src_to_west_outlet": BC(
            patches=(
                BC.interior_source(disk_mask(0.45, 0.2, 0.05), MDOT),
                BC.pressure_outlet(west_span(0.05, 0.15)),
            )
        ),
        "B_src_to_pressure_manifold": BC(
            patches=(
                BC.interior_source(disk_mask(0.15, 0.2, 0.05), MDOT),
                BC.interior_pressure_sink(disk_mask(0.55, 0.2, 0.05), COND, p=0.0),
            )
        ),
        "C_two_src_one_manifold": BC(
            patches=(
                BC.interior_source(two_in, MDOT),
                BC.interior_pressure_sink(disk_mask(0.55, 0.2, 0.05), COND, p=0.0),
            )
        ),
    }
    out: dict[str, dict] = {}
    for name, bc in cases.items():
        print(f"===== {name} =====", flush=True)
        inp = make_case("flat", 1, bc=bc)
        inp = make_case("flat", 1, bc=bc, settings=settings_for(inp))
        res = solve_steady(inp, log=lambda m: print(m, flush=True))
        out[name] = summary(inp, res)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        src = disc.q_src > 0
        out[name]["p_source_mean"] = float(res.p[src].mean())
        out[name]["u_scale"] = float(disc.u_scale)
        print(f"--> {name}: {out[name]}", flush=True)
        if res.converged:
            save_fields(HERE / "results" / f"manifold_{name}_fields.npz", inp, res)

    (HERE / "results" / "manifold_demo_flat_r1.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False)
    )
    print("\n==== summary ====")
    for k, v in out.items():
        print(
            f"{k:28s} conv={v['converged']!s:5s} it={v['n_iter']:3d} p_src={v['p_source_mean']:9.1f} Pa "
            f"speed_max={v['speed_max']:.3f} m_out/m_in={v['mass_ratio']:.4f}"
        )
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
