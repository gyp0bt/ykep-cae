"""Brinkman U ターン / 平板モデルの収束性スイープ（再現実験）.

メッシュ（72×48 の 1×, 2×, 4×）× inlet 流速（0.1, 1, 2 m/s）× モデル（flat, uturn）を走らせ、
収束/失敗と残差履歴を YAML に記録する。

使用例::

    python experiments/brinkman_uturn/sweep.py --models uturn flat --refine 1 2 4 \
        --velocities 0.1 1 2 2>&1 | tee experiments/brinkman_uturn/logs/sweep-$(date +%s).log
"""

from __future__ import annotations

import argparse
import subprocess
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

BASE_NX, BASE_NY = 72, 48
HERE = Path(__file__).resolve().parent


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_case(
    model: str,
    refine: int,
    u_inlet: float,
    scheme: str,
    mode: str,
    settings_kw: dict,
    out_dir: Path,
    save_fields: bool,
) -> dict:
    nx, ny = BASE_NX * refine, BASE_NY * refine
    geo = BrinkmanGeometry()
    th = (
        UTurnThicknessProcess()
        .execute(ThicknessInput(nx, ny, ThicknessSpec(model=ThicknessModel(model)), geo))
        .thickness
    )
    settings = BrinkmanSolverSettings(
        convection_scheme=ConvectionSchemeType(scheme),
        jacobian_mode=JacobianMode(mode),
        **settings_kw,
    )
    inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, u_inlet=u_inlet, settings=settings)
    tag = f"{model}_r{refine}_U{u_inlet:g}_{scheme}_{mode}"
    print(f"===== case {tag}  (nx={nx}, ny={ny}, N={3 * nx * ny} unknowns) =====", flush=True)
    res = BrinkmanFlowFVMProcess(log=lambda m: print(m, flush=True)).execute(inp)

    re_inlet = inp.rho * u_inlet * (geo.inlet_y1 - geo.inlet_y0) / inp.mu
    record = {
        "tag": tag,
        "model": model,
        "refine": refine,
        "nx": nx,
        "ny": ny,
        "u_inlet": u_inlet,
        "re_inlet": float(re_inlet),
        "scheme": scheme,
        "jacobian_mode": mode,
        "settings": {k: (v.value if hasattr(v, "value") else v) for k, v in vars(settings).items()},
        "converged": bool(res.converged),
        "failure_reason": res.failure_reason,
        "n_newton": res.n_newton,
        "elapsed_seconds": float(res.elapsed_seconds),
        "residual_history": [float(r) for r in res.residual_history],
        "residual_components": [[float(c) for c in cc] for cc in res.residual_components],
        "cfl_history": [float(c) for c in res.cfl_history],
        "gmres_iterations": list(res.gmres_iterations),
        "final_relative_residual": float(res.residual_history[-1] / res.residual_history[0]),
        "min_relative_residual": float(min(res.residual_history) / res.residual_history[0]),
        "mass_in": res.mass_in,
        "mass_out": res.mass_out,
        "u_abs_max": float(np.nanmax(np.abs(res.u))),
        "v_abs_max": float(np.nanmax(np.abs(res.v))),
        "p_max": float(np.nanmax(res.p)),
        "p_min": float(np.nanmin(res.p)),
    }
    (out_dir / f"{tag}.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True)
    )
    if save_fields:
        np.savez_compressed(out_dir / f"{tag}_fields.npz", u=res.u, v=res.v, p=res.p, h=th)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["uturn", "flat"])
    ap.add_argument("--refine", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--velocities", nargs="+", type=float, default=[0.1, 1.0, 2.0])
    ap.add_argument("--scheme", default="second_order_upwind")
    ap.add_argument("--mode", default="jfnk")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--cfl-init", type=float, default=5.0)
    ap.add_argument("--alpha-u", type=float, default=0.7)
    ap.add_argument("--line-search", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--save-fields", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"branch={_git_branch()} commit={_git_hash()} argv={' '.join(sys.argv)}", flush=True)
    settings_kw = {
        "newton_max_iter": args.max_iter,
        "cfl_init": args.cfl_init,
        "alpha_u": args.alpha_u,
        "line_search": args.line_search,
    }

    records: list[dict] = []
    t0 = time.perf_counter()
    for model in args.models:
        for refine in args.refine:
            for u_in in args.velocities:
                rec = run_case(
                    model,
                    refine,
                    u_in,
                    args.scheme,
                    args.mode,
                    settings_kw,
                    out_dir,
                    args.save_fields,
                )
                records.append(rec)
                print(
                    f"--> {rec['tag']}: converged={rec['converged']} reason='{rec['failure_reason']}' "
                    f"it={rec['n_newton']} rel_final={rec['final_relative_residual']:.2e} "
                    f"rel_min={rec['min_relative_residual']:.2e} t={rec['elapsed_seconds']:.0f}s",
                    flush=True,
                )

    summary = {
        "branch": _git_branch(),
        "commit": _git_hash(),
        "argv": sys.argv,
        "total_seconds": time.perf_counter() - t0,
        "cases": [
            {
                k: r[k]
                for k in (
                    "tag",
                    "model",
                    "refine",
                    "u_inlet",
                    "re_inlet",
                    "converged",
                    "failure_reason",
                    "n_newton",
                    "final_relative_residual",
                    "min_relative_residual",
                    "elapsed_seconds",
                    "mass_in",
                    "mass_out",
                )
            }
            for r in records
        ],
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"summary-{stamp}.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print("\n==== summary ====")
    print(
        f"{'case':38s} {'Re':>8s} {'conv':>5s} {'reason':>16s} {'it':>4s} {'rel_final':>10s} {'rel_min':>10s} {'t[s]':>6s}"
    )
    for r in summary["cases"]:
        print(
            f"{r['tag']:38s} {r['re_inlet']:8.0f} {str(r['converged']):>5s} {r['failure_reason']:>16s} "
            f"{r['n_newton']:4d} {r['final_relative_residual']:10.2e} {r['min_relative_residual']:10.2e} {r['elapsed_seconds']:6.0f}"
        )


if __name__ == "__main__":
    main()
