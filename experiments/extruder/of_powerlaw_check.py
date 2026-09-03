"""G3b の前提: OpenFOAM powerLaw の (k, n) と ykep-cae の (K, n) の対応づけを 1D 厳密解で確定する.

平行平板間のべき乗則 Poiseuille 流れを simpleFoam で解き、中心速度と流量を
厳密解と比較する。0.5% 以内で一致すれば「k = K（ρ=1）、γ̇ の定義も同じ」が確定。

    OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/of_powerlaw_check.py --out /tmp/of-pl1d
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from experiments.extruder.foam_io import latest_time, read_cell_centres, read_internal_field, run_of
from experiments.extruder.of_case import powerlaw_poiseuille_exact, write_poiseuille_case

H, K, N_PL, G, NY = 0.004, 2.0e4, 0.4, 5.0e7, 100
GAMMA_MIN = 1.0e-2
"""ykep PowerLawViscosity の既定クランプ。壁せん断速度は (G·h/K)^(1/n) = 56 s⁻¹."""
THRESHOLD = 5.0e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ny", type=int, default=NY)
    args = ap.parse_args()

    case = args.out
    write_poiseuille_case(case, H=H, ny=args.ny, K=K, n=N_PL, G=G, gamma_min=GAMMA_MIN)
    run_of(case, "blockMesh", log=os.path.join(case, "log.blockMesh"))
    run_of(case, "simpleFoam", log=os.path.join(case, "log.simpleFoam"))
    run_of(
        case,
        "postProcess",
        "-func",
        "writeCellCentres",
        "-latestTime",
        log=os.path.join(case, "log.post"),
    )

    t = latest_time(case)
    C = read_cell_centres(case, t)
    U = read_internal_field(os.path.join(case, t, "U"))
    y = C[:, 1]
    order = np.argsort(y)
    y, u = y[order], U[order, 0]
    dy = H / args.ny

    u_exact = powerlaw_poiseuille_exact(y, H=H, K=K, n=N_PL, G=G, gamma_min=GAMMA_MIN)
    u_max_exact = float(
        powerlaw_poiseuille_exact(np.array([0.5 * H]), H=H, K=K, n=N_PL, G=G, gamma_min=GAMMA_MIN)[
            0
        ]
    )
    # 厳密流量（単位奥行き）: クランプ込み厳密解を細かい格子で数値積分
    yy = np.linspace(0.0, H, 200001)
    q_exact = float(
        np.trapezoid(powerlaw_poiseuille_exact(yy, H=H, K=K, n=N_PL, G=G, gamma_min=GAMMA_MIN), yy)
    )
    q_of = float(np.sum(u) * dy)
    u_max_of = float(u.max())
    l2 = float(np.sqrt(np.sum((u - u_exact) ** 2) / np.sum(u_exact**2)))

    out = {
        "ny": args.ny,
        "u_max_exact": u_max_exact,
        "u_max_of": u_max_of,
        "u_max_rel_err": abs(u_max_of / u_max_exact - 1.0),
        "q_exact": q_exact,
        "q_of": q_of,
        "q_rel_err": abs(q_of / q_exact - 1.0),
        "u_profile_l2": l2,
        "threshold": THRESHOLD,
    }
    out["ratios"] = {
        "u_max": out["u_max_rel_err"] / THRESHOLD,
        "q": out["q_rel_err"] / THRESHOLD,
        "u_l2": l2 / THRESHOLD,
    }
    out["passed"] = all(r < 1.0 for r in out["ratios"].values())
    with open(os.path.join(case, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
