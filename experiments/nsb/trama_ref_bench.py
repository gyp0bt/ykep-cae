"""参照ケース（flat / uturn r1, U=0.1 と uturn U=1）で新オプションの Newton 反復数を比べる."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nsb import NSBSettings, make_case, solve_steady  # noqa: E402

CONFIGS = {
    "default": {},
    "beta1": {"pseudo_compressibility": 1.0},
    "fd": {"jacobian": "fd"},
    "ls4": {"line_search_halvings": 4},
    "beta1+fd+ls4": {"pseudo_compressibility": 1.0, "jacobian": "fd", "line_search_halvings": 4},
    "sser": {"pseudo_time_in_residual": False},
}
CASES = [("flat", 1, 0.1), ("uturn", 1, 0.1), ("uturn", 1, 1.0), ("uturn", 2, 1.0)]

for model, refine, u_in in CASES:
    for name, kw in CONFIGS.items():
        inp = make_case(model, refine, u_in, settings=NSBSettings(**kw))
        res = solve_steady(inp, log=None)
        print(
            f"[ref] {model} r{refine} U={u_in:g} {name:14s}: converged={res.converged} it={res.n_iter} "
            f"gmres={res.n_gmres_total} fact={res.n_factorizations} elapsed={res.elapsed:.1f}s rel={res.rel_steady_residual:.2e}",
            flush=True,
        )
