"""nsb solve_steady のホットスポット計測（monkeypatch で各段の累積時間を取る）.

    python experiments/nsb/profile_stages.py 2>&1 | tee experiments/nsb/logs/profile-stages-$(date +%s).log

flat 72×48 / 144×96 / 288×192（推奨構成）と 72×48（手元構成）の内訳を出す。
status-32 の結果: experiments/nsb/logs/profile-stages-flat-r124.log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.sparse import linalg as spla

from nsb import NSBSettings, make_case, solve_steady
from nsb import assembly as asm
from nsb import solver as nsolver

T: dict[str, float] = {}
C: dict[str, int] = {}


def wrap(obj, name, key):
    f = getattr(obj, name)

    def g(*a, **k):
        t = time.perf_counter()
        r = f(*a, **k)
        T[key] = T.get(key, 0.0) + time.perf_counter() - t
        C[key] = C.get(key, 0) + 1
        return r

    setattr(obj, name, g)


wrap(asm.BrinkmanDiscretization, "compute_state", "compute_state")
wrap(asm.BrinkmanDiscretization, "residual_from_state", "residual_from_state")
wrap(asm.BrinkmanDiscretization, "jacobian_first_order", "jacobian")
wrap(asm.BrinkmanDiscretization, "__init__", "disc_init")
wrap(spla, "splu", "splu")
wrap(spla, "gmres", "gmres(total)")
nsolver.spla = spla


def run(refine: int, u_in: float, cfg: str) -> None:
    T.clear()
    C.clear()
    if cfg == "fixed":
        s = NSBSettings(
            velocity_floor=0.1 * u_in, init_field="stokes", alpha_u=1.0, newton_max_iter=80
        )
    else:
        s = NSBSettings(newton_max_iter=80)
    inp = make_case("flat", refine, u_in, settings=s)
    t0 = time.perf_counter()
    res = solve_steady(inp, log=None)
    tot = time.perf_counter() - t0
    n3 = 3 * inp.nx * inp.ny
    print(
        f"\n== flat refine={refine} ({inp.nx}x{inp.ny}, n3={n3}) U={u_in} cfg={cfg}: "
        f"converged={res.converged} it={res.n_iter} total={tot:.2f}s"
    )
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(
            f"  {k:22s} {v:7.2f}s  {100 * v / tot:5.1f}%  calls={C[k]:4d}  per={1e3 * v / C[k]:.1f}ms"
        )


for refine in (1, 2, 4):
    run(refine, 1.0, "fixed")
run(1, 1.0, "mine")
