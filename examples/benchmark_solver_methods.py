"""ソルバー手法別ベンチマーク: Python GS / Vectorized Jacobi / Numba GS 比較.

使い方:
    python examples/benchmark_solver_methods.py 2>&1 | tee /tmp/log-$(date +%s).log

結果は YAML 形式で output/benchmark_results.yaml に出力される。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
)
from xkep_cae_fluid.heat_transfer.solver import HeatTransferFDMProcess

# ベンチマークケース定義
CASES = [
    {"name": "small_10x10x10", "n": 10},
    {"name": "medium_20x20x20", "n": 20},
    {"name": "large_30x30x30", "n": 30},
]

METHODS = [
    {"label": "Python GS", "kwargs": {"vectorized": False, "method": "jacobi"}},
    {"label": "Vectorized Jacobi", "kwargs": {"vectorized": True, "method": "jacobi"}},
    {"label": "Numba GS", "kwargs": {"method": "numba"}},
]

MAX_ITER = 500
TOL = 1e-6


def _make_input(n: int) -> HeatTransferInput:
    """定常1D熱伝導(3D格子)のベンチマーク入力を作成."""
    L = 1.0
    nx = ny = nz = n
    k = np.ones((nx, ny, nz)) * 50.0
    C = np.ones((nx, ny, nz)) * 1.0
    q = np.zeros((nx, ny, nz))
    T0 = np.full((nx, ny, nz), 50.0)

    return HeatTransferInput(
        Lx=L,
        Ly=L,
        Lz=L,
        k=k,
        C=C,
        q=q,
        T0=T0,
        bc_xm=BoundarySpec(condition=BoundaryCondition.DIRICHLET, value=100.0),
        bc_xp=BoundarySpec(condition=BoundaryCondition.DIRICHLET, value=0.0),
        bc_ym=BoundarySpec(condition=BoundaryCondition.ADIABATIC),
        bc_yp=BoundarySpec(condition=BoundaryCondition.ADIABATIC),
        bc_zm=BoundarySpec(condition=BoundaryCondition.ADIABATIC),
        bc_zp=BoundarySpec(condition=BoundaryCondition.ADIABATIC),
        dt=0.0,
        t_end=0.0,
        max_iter=MAX_ITER,
        tol=TOL,
    )


def main() -> None:
    """ベンチマーク実行."""
    print("=" * 60)
    print("ソルバー手法別ベンチマーク")
    print("=" * 60)

    # Numba ウォームアップ（初回JITコンパイルを計測外にする）
    print("\nNumba JIT ウォームアップ中...")
    warmup_input = _make_input(5)
    solver_warmup = HeatTransferFDMProcess(method="numba")
    solver_warmup.process(warmup_input)
    print("ウォームアップ完了\n")

    results: list[dict] = []

    for case in CASES:
        n = case["n"]
        name = case["name"]
        inp = _make_input(n)
        n_cells = n * n * n
        print(f"--- {name} ({n_cells} cells) ---")

        for method_info in METHODS:
            label = method_info["label"]
            kwargs = method_info["kwargs"]

            solver = HeatTransferFDMProcess(**kwargs)

            t0 = time.perf_counter()
            result = solver.process(inp)
            elapsed = time.perf_counter() - t0

            n_iters = result.iteration_counts[0] if result.iteration_counts else 0
            final_res = (
                result.residual_history[0][-1]
                if result.residual_history and result.residual_history[0]
                else float("nan")
            )

            record = {
                "case": name,
                "method": label,
                "n_cells": n_cells,
                "elapsed_sec": round(elapsed, 4),
                "iterations": n_iters,
                "final_residual": float(f"{final_res:.2e}"),
                "converged": result.converged,
            }
            results.append(record)

            status = "OK" if result.converged else "NOT CONVERGED"
            print(
                f"  {label:20s}: {elapsed:8.4f}s, {n_iters:4d} iters, res={final_res:.2e}, {status}"
            )

    # YAML 出力
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "benchmark_results.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            {"benchmark": results, "settings": {"max_iter": MAX_ITER, "tol": TOL}},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )
    print(f"\n結果出力: {output_path}")

    # 比較テーブル
    print("\n=== 速度比較テーブル ===")
    print(f"{'Case':<20s} {'Method':<22s} {'Time(s)':>10s} {'Speedup':>10s}")
    print("-" * 65)
    for case in CASES:
        case_results = [r for r in results if r["case"] == case["name"]]
        base_time = next(
            (r["elapsed_sec"] for r in case_results if r["method"] == "Python GS"),
            None,
        )
        for r in case_results:
            speedup = base_time / r["elapsed_sec"] if base_time and r["elapsed_sec"] > 0 else 0
            print(f"{r['case']:<20s} {r['method']:<22s} {r['elapsed_sec']:>10.4f} {speedup:>9.1f}x")

    return None


if __name__ == "__main__":
    sys.exit(main() or 0)
