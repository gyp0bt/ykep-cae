"""陰的緩和 α_u の寄与を、Δτ 管理（速度下限）の有無それぞれで確認する（72×48, U=2）.

使用例::

    python experiments/nsb/run_alpha.py uturn 2>&1 | tee experiments/nsb/logs/alpha-uturn-r1-U2.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from nsb import NSBSettings, make_case, solve_steady
from nsb.utils import summary

HERE = Path(__file__).resolve().parent


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "uturn"
    u_in = 2.0
    out: dict[str, dict] = {}
    for floor_name, floor in [("nofloor", 0.0), ("floor", 0.1 * u_in)]:
        for alpha in [1.0, 0.7, 0.5]:
            name = f"{model}_r1_U2_{floor_name}_alpha{alpha:g}"
            st = NSBSettings(
                cfl_init=0.5,
                velocity_floor=floor,
                pseudo_time_in_residual=False,
                alpha_u=alpha,
                newton_max_iter=150,
            )
            print(f"===== {name} =====", flush=True)
            inp = make_case(model, 1, u_in, st)
            res = solve_steady(inp, log=lambda m: print(m, flush=True))
            out[name] = summary(inp, res)
            print(f"--> {name}: {out[name]}", flush=True)
    (HERE / "results" / f"alpha_{model}_r1_U2.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False)
    )
    print("\n==== summary ====")
    for k, v in out.items():
        print(
            f"{k:34s} conv={v['converged']!s:5s} it={v['n_iter']:3d} step1={v['first_step_ratio']:.2e} "
            f"rel={v['rel_final']:.2e} rel_min={v['rel_min']:.2e}"
        )


if __name__ == "__main__":
    main()
