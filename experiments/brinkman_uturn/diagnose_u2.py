"""U=2 m/s（発散ケース）の失敗機構切り分け（ベースメッシュ 72×48）.

同じ問題を以下の変種で解き、どの要因が収束を左右するかを比較する:
  A. 基準（SOU+Venkat, JFNK, cfl_init=5, alpha_u=0.7, ラインサーチ無し）
  B. cfl_init=0.5（擬似時間を強める）
  C. Armijo ラインサーチ
  D. 1 次風上（リミター/2 次精度の影響を除く）
  E. defect correction（GMRES の影響を除く）
  F. 継続法: U=1 の収束解を初期値に U=2
  G. 擬似時間の速度スケール下限を U_in に（静止初期場での Δτ を小さく）+ cfl_init=1
  H. G + defect correction

使用例::

    python experiments/brinkman_uturn/diagnose_u2.py uturn 2>&1 | tee experiments/brinkman_uturn/logs/diag-uturn-$(date +%s).log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

from xkep_cae_fluid.brinkman_flow import (
    BrinkmanFlowFVMProcess,
    BrinkmanFlowInput,
    BrinkmanGeometry,
    BrinkmanSolverSettings,
    ConvectionSchemeType,
    JacobianMode,
    ThicknessInput,
    ThicknessModel,
    ThicknessSpec,
    UTurnThicknessProcess,
)

HERE = Path(__file__).resolve().parent


def solve(model: str, nx: int, ny: int, u_in: float, settings: BrinkmanSolverSettings, init=None):
    geo = BrinkmanGeometry()
    th = (
        UTurnThicknessProcess()
        .execute(ThicknessInput(nx, ny, ThicknessSpec(model=ThicknessModel(model)), geo))
        .thickness
    )
    kw = {} if init is None else {"u0": init.u, "v0": init.v, "p0": init.p}
    inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, u_inlet=u_in, settings=settings, **kw)
    return BrinkmanFlowFVMProcess(log=lambda m: print(m, flush=True)).execute(inp)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "uturn"
    refine = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nx, ny = 72 * refine, 48 * refine
    u2 = 2.0
    base = dict(newton_max_iter=80)
    variants: dict[str, BrinkmanSolverSettings] = {
        "A_baseline": BrinkmanSolverSettings(**base),
        "B_cfl0.5": BrinkmanSolverSettings(cfl_init=0.5, **base),
        "C_linesearch": BrinkmanSolverSettings(line_search=True, **base),
        "D_first_order": BrinkmanSolverSettings(
            convection_scheme=ConvectionSchemeType.FIRST_ORDER_UPWIND, **base
        ),
        "E_defect_correction": BrinkmanSolverSettings(
            jacobian_mode=JacobianMode.DEFECT_CORRECTION, **base
        ),
        "G_floor1_cfl1": BrinkmanSolverSettings(velocity_floor_ratio=1.0, cfl_init=1.0, **base),
        "H_floor1_cfl1_dc": BrinkmanSolverSettings(
            velocity_floor_ratio=1.0,
            cfl_init=1.0,
            jacobian_mode=JacobianMode.DEFECT_CORRECTION,
            **base,
        ),
    }
    out: dict[str, dict] = {}
    for name, st in variants.items():
        print(f"===== {model} r{refine} U={u2} variant {name} =====", flush=True)
        t0 = time.perf_counter()
        res = solve(model, nx, ny, u2, st)
        out[name] = {
            "converged": bool(res.converged),
            "reason": res.failure_reason,
            "n_newton": res.n_newton,
            "rel_final": float(res.residual_history[-1] / res.residual_history[0]),
            "rel_min": float(min(res.residual_history) / res.residual_history[0]),
            "elapsed": time.perf_counter() - t0,
        }
        print(f"--> {name}: {out[name]}", flush=True)

    print(f"===== {model} r{refine} variant F_continuation (U=1 -> U=2) =====", flush=True)
    t0 = time.perf_counter()
    res1 = solve(model, nx, ny, 1.0, BrinkmanSolverSettings(**base))
    print(f"    U=1 stage: converged={res1.converged} it={res1.n_newton}", flush=True)
    res2 = solve(model, nx, ny, u2, BrinkmanSolverSettings(**base), init=res1)
    out["F_continuation"] = {
        "converged": bool(res2.converged),
        "reason": res2.failure_reason,
        "n_newton": res2.n_newton,
        "rel_final": float(res2.residual_history[-1] / res2.residual_history[0]),
        "rel_min": float(min(res2.residual_history) / res2.residual_history[0]),
        "elapsed": time.perf_counter() - t0,
        "stage1_converged": bool(res1.converged),
    }
    print(f"--> F_continuation: {out['F_continuation']}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"diagnose_u2_{model}_r{refine}.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False)
    )
    print("\n==== summary ====")
    for k, v in out.items():
        print(
            f"{k:22s} conv={v['converged']!s:5s} reason={v['reason']:12s} it={v['n_newton']:3d} rel_final={v['rel_final']:.2e} rel_min={v['rel_min']:.2e}"
        )
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
