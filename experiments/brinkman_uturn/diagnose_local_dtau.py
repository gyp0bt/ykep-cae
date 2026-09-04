"""局所 Δτ vs 大域 Δτ、および RC 係数への擬似時間項混入の影響切り分け.

「局所時間増分にすると発散する」現象の再現を狙い、以下の変種を比較する:
  L_floor0.1        局所 Δτ、速度下限 0.1 U_in（本リポジトリの基準。cfl_init=0.5）
  L_nofloor         局所 Δτ、速度下限ほぼ無し（静止セルで Δτ→∞、擬似時間項が消える）
  L_rc_floor0.1     局所 Δτ + RC 係数 d_f = V/(a_P + ρV/Δτ)
  L_rc_nofloor      同上 + 速度下限ほぼ無し
  G_floor0.1        大域 Δτ（局所 Δτ の全セル最小値）
  G_rc_floor0.1     大域 Δτ + RC 係数に擬似時間項
  L_floor0.1_cfl5   局所 Δτ、cfl_init=5（status-28 の基準 A 相当）
  G_floor0.1_cfl5   大域 Δτ、cfl_init=5

使用例::

    python experiments/brinkman_uturn/diagnose_local_dtau.py uturn 1 2.0 2>&1 | tee experiments/brinkman_uturn/logs/diag-ldt-uturn-r1-$(date +%s).log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

from xkep_cae_fluid.brinkman_flow import (
    BrinkmanFlowFVMProcess,
    BrinkmanFlowInput,
    BrinkmanGeometry,
    BrinkmanSolverSettings,
    PseudoTimeMode,
    ThicknessInput,
    ThicknessModel,
    ThicknessSpec,
    UTurnThicknessProcess,
)

HERE = Path(__file__).resolve().parent


def solve(
    model: str, nx: int, ny: int, u_in: float, settings: BrinkmanSolverSettings
) -> tuple[float, dict]:
    geo = BrinkmanGeometry()
    th = (
        UTurnThicknessProcess()
        .execute(ThicknessInput(nx, ny, ThicknessSpec(model=ThicknessModel(model)), geo))
        .thickness
    )
    inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, u_inlet=u_in, settings=settings)
    t0 = time.perf_counter()
    res = BrinkmanFlowFVMProcess(log=lambda m: print(m, flush=True)).execute(inp)
    h = res.residual_history
    return time.perf_counter() - t0, {
        "converged": bool(res.converged),
        "reason": res.failure_reason,
        "n_newton": res.n_newton,
        "rel_final": float(h[-1] / h[0]),
        "rel_min": float(min(h) / h[0]),
        "first_step_ratio": float(h[1] / h[0]) if len(h) > 1 else float("nan"),
        "steady_residual_ratio": float(res.steady_residual_ratio),
        "mass_in": float(res.mass_in),
        "mass_out": float(res.mass_out),
    }


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "uturn"
    refine = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    u_in = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    nx, ny = 72 * refine, 48 * refine
    base = dict(newton_max_iter=80)
    L, G = PseudoTimeMode.LOCAL, PseudoTimeMode.GLOBAL
    variants: dict[str, BrinkmanSolverSettings] = {
        "L_floor0.1": BrinkmanSolverSettings(cfl_init=0.5, pseudo_time_mode=L, **base),
        "L_nofloor": BrinkmanSolverSettings(
            cfl_init=0.5, pseudo_time_mode=L, velocity_floor_ratio=1e-6, **base
        ),
        "L_rc_floor0.1": BrinkmanSolverSettings(
            cfl_init=0.5, pseudo_time_mode=L, rhie_chow_pseudo_time=True, **base
        ),
        "L_rc_nofloor": BrinkmanSolverSettings(
            cfl_init=0.5,
            pseudo_time_mode=L,
            rhie_chow_pseudo_time=True,
            velocity_floor_ratio=1e-6,
            **base,
        ),
        "G_floor0.1": BrinkmanSolverSettings(cfl_init=0.5, pseudo_time_mode=G, **base),
        "G_rc_floor0.1": BrinkmanSolverSettings(
            cfl_init=0.5, pseudo_time_mode=G, rhie_chow_pseudo_time=True, **base
        ),
        "L_floor0.1_cfl5": BrinkmanSolverSettings(cfl_init=5.0, pseudo_time_mode=L, **base),
        "G_floor0.1_cfl5": BrinkmanSolverSettings(cfl_init=5.0, pseudo_time_mode=G, **base),
    }
    out: dict[str, dict] = {}
    for name, st in variants.items():
        print(f"===== {model} r{refine} U={u_in} variant {name} =====", flush=True)
        elapsed, out[name] = solve(model, nx, ny, u_in, st)
        out[name]["elapsed"] = elapsed
        print(f"--> {name}: {out[name]}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"diagnose_ldt_{model}_r{refine}_U{u_in:g}.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False)
    )
    print("\n==== summary ====")
    for k, v in out.items():
        print(
            f"{k:18s} conv={v['converged']!s:5s} reason={v['reason']:10s} it={v['n_newton']:3d} "
            f"step1={v['first_step_ratio']:.2e} rel_final={v['rel_final']:.2e} "
            f"rel_min={v['rel_min']:.2e} steady={v['steady_residual_ratio']:.2e} "
            f"m_out/m_in={v['mass_out'] / v['mass_in'] if v['mass_in'] else float('nan'):.4f}"
        )


if __name__ == "__main__":
    main()
