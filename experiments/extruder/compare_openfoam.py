"""ゲート G3: OpenFOAM の断面解と ykep-cae `ExtruderFlowProcess` の突き合わせ.

同じ格子（compare 時にセル中心座標で検査）で、以下を**閾値で規格化した比**で報告する。
比 < 1.00 が合格。

| 量 | 定義 | 閾値 |
|---|---|---|
| Q       | ∫∫w dA の相対差 | 1% |
| Q_leak  | x=0 面を通る横断流束の相対差 | 1% |
| w(y)    | チャネル中央 (x≈0) の鉛直分布の L2 相対誤差 | 1% |
| u(y)    | 同上 | 1% |
| Q_axial | Q + L_turn·Q_leak の相対差（参考） | 1% |

    PYTHONPATH=. .venv/bin/python experiments/extruder/compare_openfoam.py --case /tmp/of-g3a --model newtonian
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from experiments.extruder.foam_io import (
    continuity_converged,
    latest_time,
    read_cell_centres,
    read_internal_field,
    read_patch_field,
)
from experiments.extruder.of_case import DEFAULT_G, DEFAULT_SPEC
from xkep_cae_fluid.extruder import (
    ExtruderFlowProcess,
    NewtonianViscosity,
    PowerLawViscosity,
)
from xkep_cae_fluid.extruder.data import ExtruderFlowInput

THRESHOLD = 1.0e-2


def map_cells(C: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray, float]:
    """OpenFOAM セル中心 → (i, j) 添字。格子一致の検査値（最大ずれ / 最小セル幅）も返す."""
    xe = np.concatenate([[0.0], np.cumsum(grid.dx)])
    ye = np.concatenate([[0.0], np.cumsum(grid.dy)])
    i = np.clip(np.searchsorted(xe, C[:, 0]) - 1, 0, grid.nx - 1)
    j = np.clip(np.searchsorted(ye, C[:, 1]) - 1, 0, grid.ny - 1)
    dev = np.maximum(
        np.abs(C[:, 0] - grid.xc[i]) / grid.dx[i], np.abs(C[:, 1] - grid.yc[j]) / grid.dy[j]
    )
    return i, j, float(dev.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", choices=["newtonian", "powerlaw"], required=True)
    ap.add_argument("--mu", type=float, default=1000.0)
    ap.add_argument("--K", type=float, default=2.0e4)
    ap.add_argument("--n", type=float, default=0.4)
    ap.add_argument("--G", type=float, default=DEFAULT_G)
    ap.add_argument("--out", default=None, help="結果 JSON の出力先（既定: <case>/compare.json）")
    args = ap.parse_args()

    spec = DEFAULT_SPEC
    model = (
        NewtonianViscosity(mu=args.mu)
        if args.model == "newtonian"
        else PowerLawViscosity(K=args.K, n=args.n)
    )
    t0 = time.perf_counter()
    proc = ExtruderFlowProcess()
    proc.viscosity = model
    flow = proc.process(ExtruderFlowInput(spec=spec, G=args.G))
    t_ykep = time.perf_counter() - t0
    grid = flow.grid

    case = args.case
    t = latest_time(case)
    C = read_cell_centres(case, t)
    U = read_internal_field(os.path.join(case, t, "U"))
    i, j, grid_dev = map_cells(C, grid)
    if grid_dev > 1e-3:
        msg = f"OpenFOAM の格子が ykep-cae と一致しない（セル中心のずれ {grid_dev:.2e} セル幅）"
        raise RuntimeError(msg)
    fluid = ~grid.solid
    if fluid.sum() != C.shape[0]:
        msg = f"流体セル数不一致: ykep {int(fluid.sum())}, OpenFOAM {C.shape[0]}"
        raise RuntimeError(msg)

    # OpenFOAM 場を (nx, ny) に並べ直す（固体は NaN）
    u_of = np.full((grid.nx, grid.ny), np.nan)
    v_of = np.full_like(u_of, np.nan)
    w_of = np.full_like(u_of, np.nan)
    u_of[i, j], v_of[i, j], w_of[i, j] = U[:, 0], U[:, 1], U[:, 2]

    dA = grid.dx[:, None] * grid.dy[None, :]
    Q_of = float(np.nansum(w_of * dA))
    # x=0 (left cyclic) 面の流束。外向き法線は −x なので符号を反転して +x 向きの流量にする
    phi_left = read_patch_field(os.path.join(case, t, "phi"), "left")
    dz = spec.H  # write_channel_case の z 厚み。面積 = dy·dz
    Q_leak_of = float(-phi_left.sum() / dz)
    Q_axial_of = Q_of + spec.L_turn * Q_leak_of

    # プロファイル比較: x ≈ 0 の列（チャネル中央、周期境界の直隣）
    i0 = 0
    w_prof_err = float(np.sqrt(np.nansum((w_of[i0] - flow.w[i0]) ** 2) / np.sum(flow.w[i0] ** 2)))
    u_prof_err = float(np.sqrt(np.nansum((u_of[i0] - flow.u[i0]) ** 2) / np.sum(flow.u[i0] ** 2)))
    # 全域 L2（参考）
    m = fluid
    w_all = float(np.sqrt(np.sum((w_of[m] - flow.w[m]) ** 2) / np.sum(flow.w[m] ** 2)))
    u_all = float(np.sqrt(np.sum((u_of[m] - flow.u[m]) ** 2) / np.sum(flow.u[m] ** 2)))
    v_all = float(np.sqrt(np.sum((v_of[m] - flow.v[m]) ** 2) / np.sum(flow.v[m] ** 2)))

    n_iter, converged = continuity_converged(os.path.join(case, "log.simpleFoam"))

    rel = {
        "Q": abs(Q_of / flow.Q - 1.0),
        "Q_leak": abs(Q_leak_of / flow.Q_leak - 1.0),
        "Q_axial": abs(Q_axial_of / flow.Q_axial - 1.0),
        "w_profile": w_prof_err,
        "u_profile": u_prof_err,
    }
    out = {
        "model": args.model,
        "G": args.G,
        "case": case,
        "time": t,
        "openfoam_iterations": n_iter,
        "openfoam_converged": converged,
        "grid_deviation_cells": grid_dev,
        "n_fluid_cells": int(fluid.sum()),
        "ykep": {
            "Q": flow.Q,
            "Q_leak": flow.Q_leak,
            "Q_axial": flow.Q_axial,
            "n_iter": flow.n_iter,
            "elapsed_s": t_ykep,
        },
        "openfoam": {"Q": Q_of, "Q_leak": Q_leak_of, "Q_axial": Q_axial_of},
        "rel_err": rel,
        "l2_all": {"u": u_all, "v": v_all, "w": w_all},
        "threshold": THRESHOLD,
        "ratios": {k: v / THRESHOLD for k, v in rel.items()},
    }
    gate = ["Q", "Q_leak", "w_profile", "u_profile"]
    out["passed"] = all(out["ratios"][k] < 1.0 for k in gate)
    out["gate_keys"] = gate
    path = args.out or os.path.join(case, "compare.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"phi = {math.degrees(spec.phi):.4f} deg")


if __name__ == "__main__":
    main()
