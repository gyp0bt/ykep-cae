"""速度下限 0.1 U_in の構成で反復上限を伸ばし、下限だけで収束に届くかを確認する.

使用例::

    python experiments/nsb/run_floor_long.py uturn 1 2.0 200 2>&1 | tee experiments/nsb/logs/floor-long-uturn-r1-U2.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from nsb import NSBSettings, make_case, solve_steady
from nsb.utils import save_fields, summary

HERE = Path(__file__).resolve().parent


def main() -> None:
    model, refine, u_in, max_iter = (
        sys.argv[1],
        int(sys.argv[2]),
        float(sys.argv[3]),
        int(sys.argv[4]),
    )
    name = f"{model}_r{refine}_U{u_in:g}_floor_it{max_iter}"
    st = NSBSettings(
        cfl_init=0.5,
        local_dtau=True,
        velocity_floor=0.1 * u_in,
        pseudo_time_in_residual=False,
        newton_max_iter=max_iter,
    )
    print(f"===== {name} =====", flush=True)
    inp = make_case(model, refine, u_in, st)
    res = solve_steady(inp, log=lambda m: print(m, flush=True))
    out = summary(inp, res)
    print(f"--> {name}: {out}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{name}.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    if res.converged:
        save_fields(HERE / "results" / f"{name}_fields.npz", inp, res)


if __name__ == "__main__":
    main()
