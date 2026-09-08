"""SER 制御パラメータの掃引（status-41）: 6 ケースで Newton 反復・GMRES 反復・時間・収束を並べる.

ケース: flat 288×192 を 144×96 の解（tol 1e-4）から双一次 / 注入で発進、flat 144×96 Stokes 発進、
uturn 144×96 U=1、uturn 72×48 U=2、uturn 144×96 U=2（いずれも Stokes 発進）。

使用例::

    python experiments/nsb/ser_sweep.py 2>&1 | tee experiments/nsb/logs/ser-sweep-$(date +%s).log
    python experiments/nsb/ser_sweep.py base cflinit0.5 norej
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

from nsb import NSBSettings, make_case, prolong_bilinear, prolong_inject, solve_steady

CONFIGS: dict[str, dict] = {
    "base": {},
    "cflinit0.1": dict(cfl_init=0.1),
    "cflinit0.5": dict(cfl_init=0.5),
    "cflinit1": dict(cfl_init=1.0),
    "norej": dict(reject_lin_ratio=0.0),
    "rej0.1": dict(reject_lin_ratio=0.1),
    "growth1.5": dict(ser_growth=1.5),
    "growth3": dict(ser_growth=3.0),
    "shrink0.3": dict(ser_shrink=0.3),
    "cflmax100": dict(cfl_max=100.0),
    "pcratio4": dict(precond_cfl_ratio=4.0),
    "pcratio0": dict(precond_cfl_ratio=0.0),
}
CASES = (
    "flat4-bil",
    "flat4-inj",
    "flat2-stokes",
    "uturn2-stokes",
    "uturn1U2-stokes",
    "uturn2U2-stokes",
)


def main() -> None:
    names = sys.argv[1:] or list(CONFIGS)
    base = NSBSettings(newton_max_iter=60)
    solve_steady(make_case("flat", 1, 1.0, settings=base), log=None)  # ウォームアップ
    coarse = solve_steady(
        make_case("flat", 2, 1.0, settings=replace(base, newton_tol=1e-4)), log=None
    )
    prolong = {"flat4-bil": prolong_bilinear, "flat4-inj": prolong_inject}

    def cases(s: NSBSettings):
        for k, p in prolong.items():
            yield (
                k,
                replace(
                    make_case("flat", 4, 1.0, settings=s),
                    u0=p(coarse.u),
                    v0=p(coarse.v),
                    p0=p(coarse.p),
                ),
            )
        yield "flat2-stokes", make_case("flat", 2, 1.0, settings=s)
        yield "uturn2-stokes", make_case("uturn", 2, 1.0, settings=s)
        yield "uturn1U2-stokes", make_case("uturn", 1, 2.0, settings=s)
        yield "uturn2U2-stokes", make_case("uturn", 2, 2.0, settings=s)

    table = []
    for name in names:
        s = replace(base, **CONFIGS[name])
        row: dict[str, tuple] = {"name": name}
        for cname, inp in cases(s):
            t0 = time.perf_counter()
            res = solve_steady(inp, log=None)
            el = time.perf_counter() - t0
            row[cname] = (res.n_iter, res.n_gmres_total, res.n_factorizations, el, res.converged)
            print(
                f"[{name:12s}] {cname:16s} it={res.n_iter:3d} gmres={res.n_gmres_total:5d} "
                f"fact={res.n_factorizations:3d} {el:6.1f}s conv={res.converged} {res.failure_reason}",
                flush=True,
            )
            print(f"      cfl: {' '.join(f'{c:.3g}' for c in res.cfl_history)}", flush=True)
            print(
                f"      rel: {' '.join(f'{x / res.residual_ref:.1e}' for x in res.residual_history)}",
                flush=True,
            )
        table.append(row)
    print("\n==== summary ====")
    print(f"{'config':12s}" + "".join(f" | {c:>26s}" for c in CASES) + " | total_s")
    for row in table:
        tot = sum(row[c][3] for c in CASES)
        cells = "".join(
            f" | it={r[0]:3d} gm={r[1]:5d} {r[3]:6.1f}s{'' if r[4] else '!'}"
            for r in (row[c] for c in CASES)
        )
        print(f"{row['name']:12s}{cells} | {tot:6.1f}")


if __name__ == "__main__":
    main()
