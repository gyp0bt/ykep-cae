"""入れ子反復の粒度比較: 段数 × 粗格子の解き込み深さ × 補間法 の総時間（flat U=1、status-41）.

使用例::

    python experiments/nsb/nested_schedules.py 4 2>&1 | tee experiments/nsb/logs/nested-schedules-r4-$(date +%s).log
    python experiments/nsb/nested_schedules.py 8 stokes bil:4:1e-4 bil:2,4:1e-4 bil:1,2,4:1e-4

スケジュール表記: ``<prolong>:<coarse refines>:<coarse tol>``（例 ``bil:1,2:1e-4`` は 72×48 → 144×96 を
tol 1e-4 で解いてから目的格子を双一次補間で発進）。``stokes`` は Stokes 解発進（入れ子なし）。
"""

from __future__ import annotations

import sys
import time

import numpy as np

from nsb import NSBSettings, make_case, solve_nested, solve_steady

PROLONG = {"inj": "inject", "bil": "bilinear"}
DEFAULT_SCHEDULES = (
    "stokes",
    "inj:2:1e-6",
    "bil:2:1e-6",
    "bil:2:1e-4",
    "bil:2:1e-3",
    "bil:2:1e-2",
    "bil:1,2:1e-6",
    "bil:1,2:1e-4",
    "bil:1,2:1e-3",
    "bil:1,2:1e-2",
    "inj:1,2:1e-4",
)


def _init_line(lines: list[str]) -> tuple[float, float]:
    it0 = next(ln for ln in lines if " it=0 " in ln)
    return float(it0.split("rel=")[1].split()[0]), float(it0.split("cfl=")[1].split()[0])


def main() -> None:
    fine = int(sys.argv[1])
    scheds = sys.argv[2:] or list(DEFAULT_SCHEDULES)
    s = NSBSettings(newton_max_iter=150)
    solve_steady(make_case("flat", 1, 1.0, settings=s), log=None)  # numba / pyamg のウォームアップ
    ref = None
    rows = []
    for sc in scheds:
        print(f"=== fine r={fine} schedule {sc} ===", flush=True)
        if sc == "stokes":
            refines, tol, pro = [fine], s.newton_tol, "bilinear"
        else:
            pro_key, lv, tol_s = sc.split(":")
            refines = [int(x) for x in lv.split(",") if int(x) < fine] + [fine]
            tol, pro = float(tol_s), PROLONG[pro_key]
        lines: list[str] = []
        t0 = time.perf_counter()
        nested = solve_nested(
            lambda r: make_case("flat", r, 1.0),
            refines,
            coarse_tol=tol,
            prolongation=pro,
            settings=s,
            log=lines.append,
        )
        wall = time.perf_counter() - t0
        for lv_ in nested.levels:
            r_ = lv_.result
            print(
                f"  {lv_.nx}x{lv_.ny} tol={lv_.newton_tol:.0e} it={r_.n_iter} gmres={r_.n_gmres_total} "
                f"fact={r_.n_factorizations} {lv_.elapsed:.1f}s conv={r_.converged}",
                flush=True,
            )
        res = nested.result
        # 最終段の it=0 行（初期残差比と出発 CFL）
        last_start = max(i for i, ln in enumerate(lines) if " it=0 " in ln)
        rel0, cfl0 = _init_line(lines[last_start:])
        if ref is None:
            ref = res
        du = float(np.abs(res.u - ref.u).max() / np.abs(ref.u).max())
        fine_t = nested.levels[-1].elapsed
        rows.append(
            (
                sc,
                wall,
                wall - fine_t,
                fine_t,
                res.n_iter,
                res.n_gmres_total,
                rel0,
                cfl0,
                du,
                res.converged,
            )
        )
        print(
            f"    TOTAL {wall:.1f}s (coarse {wall - fine_t:.1f}s + fine {fine_t:.1f}s) fine it={res.n_iter} "
            f"gmres={res.n_gmres_total} init rel={rel0:.2e} cfl={cfl0:.3g} |du|/|u| vs first={du:.1e}",
            flush=True,
        )
        print(f"    fine cfl: {' '.join(f'{c:.3g}' for c in res.cfl_history[:16])}", flush=True)
    print(f"\n==== summary fine r={fine} ====")
    print(
        f"{'schedule':16s} {'total':>7s} {'coarse':>7s} {'fine':>7s} {'fine_it':>7s} {'gmres':>6s} "
        f"{'init_rel':>9s} {'init_cfl':>8s} {'du':>8s} conv"
    )
    for sc, tot, co, fi, it, gm, rel0, cfl0, du, conv in rows:
        print(
            f"{sc:16s} {tot:7.1f} {co:7.1f} {fi:7.1f} {it:7d} {gm:6d} {rel0:9.2e} {cfl0:8.3g} {du:8.1e} {conv}"
        )


if __name__ == "__main__":
    main()
