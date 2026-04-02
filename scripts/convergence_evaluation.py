"""空気実物性 自然対流ソルバー収束評価.

PISO / SIMPLEC / SIMPLE × upwind / van_leer の組み合わせで
粘性を段階的に下げ、mass残差と収束性を比較する。

STA2防止: tee でログ保存、YAML 出力と照合可能にする。
"""

import time
import numpy as np
import yaml

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess


def run_case(label, **overrides):
    """1ケースを実行し結果を返す."""
    # 空気実物性ベース (T=300K, 1atm)
    L = 0.02  # 2cm キャビティ
    nx, ny, nz = 10, 10, 3
    T_hot, T_cold = 310.0, 300.0

    defaults = dict(
        Lx=L, Ly=L, Lz=L,
        nx=nx, ny=ny, nz=nz,
        rho=1.19,
        mu=1.85e-5,
        Cp=1007.0,
        k_fluid=0.026,
        beta=3.3e-3,
        T_ref=300.0,
        gravity=(0.0, -9.81, 0.0),
        bc_xm=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_hot,
        ),
        bc_xp=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_cold,
        ),
        dt=0.001,
        t_end=0.01,  # 10 ステップ
        max_simple_iter=50,
        tol_simple=1e-4,
        alpha_u=0.3,
        alpha_p=0.1,
        alpha_T=0.5,
    )
    defaults.update(overrides)

    inp = NaturalConvectionInput(**defaults)

    # Ra 数の計算
    nu = inp.mu / inp.rho
    alpha_th = inp.k_fluid / (inp.rho * inp.Cp)
    delta_T = T_hot - T_cold
    Ra = 9.81 * inp.beta * delta_T * L**3 / (nu * alpha_th)

    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  mu={inp.mu:.2e}, Ra={Ra:.1f}")
    print(f"  coupling={inp.coupling_method}, convection={inp.convection_scheme}")
    print(f"  alpha_u={inp.alpha_u}, alpha_p={inp.alpha_p}")
    print(f"  dt={inp.dt}, t_end={inp.t_end}, max_simple={inp.max_simple_iter}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    solver = NaturalConvectionFDMProcess()
    result = solver.process(inp)
    elapsed = time.perf_counter() - t0

    # 結果サマリー
    has_nan = bool(np.any(np.isnan(result.u)) or np.any(np.isnan(result.T)))
    mass_final = result.residual_history["mass"][-1] if result.residual_history["mass"] else float("nan")
    u_max = float(np.max(np.abs(result.u)))
    v_max = float(np.max(np.abs(result.v)))
    T_min = float(np.min(result.T))
    T_max = float(np.max(result.T))

    # v > 0 の割合（上向き流れが正常）
    v_positive_ratio = float(np.mean(result.v > 0)) if not has_nan else 0.0

    summary = {
        "label": label,
        "mu": float(inp.mu),
        "Ra": float(Ra),
        "coupling": inp.coupling_method,
        "convection": inp.convection_scheme,
        "converged": bool(result.converged),
        "has_nan": has_nan,
        "n_timesteps": result.n_timesteps,
        "n_outer_iterations": result.n_outer_iterations,
        "mass_residual_final": float(mass_final),
        "u_max": u_max,
        "v_max": v_max,
        "T_min": T_min,
        "T_max": T_max,
        "v_positive_ratio": v_positive_ratio,
        "elapsed_s": round(elapsed, 2),
    }

    print(f"  converged={result.converged}, nan={has_nan}")
    print(f"  mass_final={mass_final:.4e}")
    print(f"  u_max={u_max:.4e}, v_max={v_max:.4e}")
    print(f"  T: [{T_min:.1f}, {T_max:.1f}], v>0 ratio={v_positive_ratio:.2%}")
    print(f"  timesteps={result.n_timesteps}, outer_iter={result.n_outer_iterations}")
    print(f"  elapsed={elapsed:.2f}s")

    return summary


def main():
    results = []

    # --- 1) 適応dt + 小さいdt でSIMPLE/PISO比較 (空気実物性) ---
    print("\n" + "#" * 70)
    print("# Phase 1: dt=1e-3 でSIMPLE/SIMPLEC/PISO比較 (mu=1.85e-5)")
    print("#" * 70)
    for method in ["simple", "simplec", "piso"]:
        r = run_case(
            f"{method}_upw_air_dt1e-3",
            coupling_method=method,
            convection_scheme="upwind",
            alpha_u=0.3,
            alpha_p=0.1,
        )
        results.append(r)

    # --- 2) 粘性スイープ (SIMPLE, 適応dt) ---
    print("\n" + "#" * 70)
    print("# Phase 2: 粘性スイープ (SIMPLE, dt=1e-3)")
    print("#" * 70)
    for mu in [1e-2, 1e-3, 1e-4, 1.85e-5]:
        r = run_case(
            f"simple_mu{mu:.0e}",
            mu=mu,
            coupling_method="simple",
            alpha_u=0.5,
            alpha_p=0.2,
        )
        results.append(r)

    # --- 3) メッシュ細分化テスト (20x20, 空気実物性) ---
    print("\n" + "#" * 70)
    print("# Phase 3: メッシュ細分化 (20x20x3, mu=1.85e-5)")
    print("#" * 70)
    r = run_case(
        "simple_20x20_air",
        nx=20, ny=20, nz=3,
        coupling_method="simple",
        alpha_u=0.3,
        alpha_p=0.1,
        dt=0.0002,
        t_end=0.002,
    )
    results.append(r)
    r = run_case(
        "piso_20x20_air",
        nx=20, ny=20, nz=3,
        coupling_method="piso",
        alpha_u=0.5,
        alpha_p=0.2,
        dt=0.0002,
        t_end=0.002,
    )
    results.append(r)

    # --- 4) 超保守的 (alpha_u=0.05, alpha_p=0.01, max_iter=200) ---
    print("\n" + "#" * 70)
    print("# Phase 4: 超保守的緩和 (空気実物性)")
    print("#" * 70)
    r = run_case(
        "ultraconserv_air",
        coupling_method="simple",
        alpha_u=0.05,
        alpha_p=0.01,
        max_simple_iter=200,
        dt=0.0001,
        t_end=0.0005,
    )
    results.append(r)

    # YAML 出力
    yaml_path = "/tmp/convergence_evaluation.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    print(f"\n結果を {yaml_path} に保存しました")

    # サマリーテーブル
    print("\n" + "=" * 100)
    print(f"{'Label':<25} {'mu':>10} {'Method':<8} {'Conv':>5} {'NaN':>4} "
          f"{'mass_res':>10} {'v>0%':>6} {'T_range':>15} {'time':>6}")
    print("-" * 100)
    for r in results:
        t_range = f"[{r['T_min']:.1f},{r['T_max']:.1f}]"
        print(f"{r['label']:<25} {r['mu']:>10.2e} {r['coupling']:<8} "
              f"{'Y' if r['converged'] else 'N':>5} {'Y' if r['has_nan'] else 'N':>4} "
              f"{r['mass_residual_final']:>10.2e} {r['v_positive_ratio']:>5.1%} "
              f"{t_range:>15} {r['elapsed_s']:>5.1f}s")


if __name__ == "__main__":
    main()
"""
"""
