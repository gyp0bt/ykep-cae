"""十分発達した自然対流の計算時間見積もり.

物理的時間スケールと計算コストを見積もり、
実際の短時間ベンチマークで校正する。
"""

import time
import numpy as np

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess


def estimate_physics():
    """物理的時間スケールの見積もり."""
    L = 0.02  # 2cm cavity
    g = 9.81
    beta = 3.3e-3
    dT = 10.0  # T_hot - T_cold
    rho = 1.19
    mu = 1.85e-5
    nu = mu / rho  # 1.55e-5 m²/s
    Cp = 1007.0
    k = 0.026
    alpha_th = k / (rho * Cp)  # 2.17e-5 m²/s

    # Ra number
    Ra = g * beta * dT * L**3 / (nu * alpha_th)

    # Characteristic buoyancy velocity
    U_buoy = np.sqrt(g * beta * dT * L)

    # Time scales
    t_conv = L / U_buoy  # convective
    t_diff = L**2 / alpha_th  # thermal diffusion
    t_visc = L**2 / nu  # viscous diffusion

    # Fully developed: ~5-10 convective times or ~1 diffusion time
    t_develop_min = 5 * t_conv
    t_develop_max = t_diff

    # Re_cell constraint: Re_cell = U*dx/nu < 2
    dx_max = 2 * nu / U_buoy
    nx_min = int(np.ceil(L / dx_max))

    print("=" * 60)
    print("物理的時間スケール見積もり")
    print("=" * 60)
    print(f"  L = {L} m, ΔT = {dT} K")
    print(f"  ν = {nu:.4e} m²/s, α = {alpha_th:.4e} m²/s")
    print(f"  Ra = {Ra:.0f}")
    print(f"  U_buoy = {U_buoy:.4f} m/s")
    print()
    print(f"  t_convective = {t_conv:.3f} s")
    print(f"  t_thermal_diff = {t_diff:.1f} s")
    print(f"  t_viscous_diff = {t_visc:.1f} s")
    print()
    print(f"  十分発達まで: {t_develop_min:.1f} 〜 {t_develop_max:.1f} s")
    print()
    print(f"  Re_cell < 2 のための dx_max = {dx_max*1000:.3f} mm")
    print(f"  → 最低 nx = {nx_min}")
    print()

    return {
        "L": L, "U_buoy": U_buoy, "nu": nu, "alpha_th": alpha_th,
        "Ra": Ra, "t_conv": t_conv, "t_diff": t_diff,
        "t_develop_min": t_develop_min, "t_develop_max": t_develop_max,
        "dx_max": dx_max, "nx_min": nx_min,
    }


def benchmark(nx, n_steps, dt):
    """短時間ベンチマーク."""
    L = 0.02
    inp = NaturalConvectionInput(
        Lx=L, Ly=L, Lz=L,
        nx=nx, ny=nx, nz=3,
        rho=1.19, mu=1.85e-5, Cp=1007.0, k_fluid=0.026,
        beta=3.3e-3, T_ref=300.0, gravity=(0.0, -9.81, 0.0),
        bc_xm=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET, temperature=310.0,
        ),
        bc_xp=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET, temperature=300.0,
        ),
        dt=dt, t_end=dt * n_steps,
        max_simple_iter=50, tol_simple=1e-4,
        alpha_u=0.3, alpha_p=0.1,
        coupling_method="simple",
    )

    t0 = time.perf_counter()
    result = NaturalConvectionFDMProcess().process(inp)
    elapsed = time.perf_counter() - t0

    has_nan = bool(np.any(np.isnan(result.u)))
    mass_final = result.residual_history["mass"][-1] if result.residual_history["mass"] else float("nan")
    cost_per_step = elapsed / max(result.n_timesteps, 1)

    return {
        "nx": nx,
        "n_steps": result.n_timesteps,
        "n_outer": result.n_outer_iterations,
        "elapsed": elapsed,
        "cost_per_step": cost_per_step,
        "mass_final": mass_final,
        "has_nan": has_nan,
        "T_max": float(np.max(result.T)) if not has_nan else float("nan"),
    }


def main():
    phys = estimate_physics()

    print("=" * 60)
    print("計算コスト ベンチマーク (5ステップ)")
    print("=" * 60)

    configs = [
        (10, 5, 0.001),
        (15, 5, 0.0005),
        (20, 5, 0.0003),
        (30, 5, 0.0001),
    ]

    bench_results = []
    for nx, n_steps, dt in configs:
        dx = 0.02 / nx
        Re_cell = phys["U_buoy"] * dx / phys["nu"]
        CFL = phys["U_buoy"] * dt / dx

        r = benchmark(nx, n_steps, dt)
        bench_results.append(r)

        print(f"\n  {nx}×{nx}×3: dx={dx*1000:.2f}mm, Re_cell={Re_cell:.2f}, CFL={CFL:.3f}")
        print(f"    {r['n_steps']}steps in {r['elapsed']:.2f}s "
              f"({r['cost_per_step']:.3f}s/step, {r['n_outer']/r['n_steps']:.0f} outer/step)")
        print(f"    mass={r['mass_final']:.2e}, T_max={r['T_max']:.1f}, nan={r['has_nan']}")

    # 計算時間の外挿
    print("\n" + "=" * 60)
    print("十分発達までの計算時間 外挿")
    print("=" * 60)

    t_target = 5.0  # 5秒（十分発達の目安）

    for r in bench_results:
        nx = r["nx"]
        dx = 0.02 / nx
        # 適応dtを考慮: CFL=0.5 → dt_adaptive = 0.5 * dx / U
        dt_adaptive = 0.5 * dx / phys["U_buoy"]
        n_steps_needed = int(np.ceil(t_target / dt_adaptive))
        wall_time = n_steps_needed * r["cost_per_step"]

        print(f"\n  {nx}×{nx}×3:")
        print(f"    dt_adaptive ≈ {dt_adaptive:.4f} s (CFL=0.5)")
        print(f"    t=5s まで: {n_steps_needed} steps")
        print(f"    推定計算時間: {wall_time:.0f} s ({wall_time/60:.1f} min)")


if __name__ == "__main__":
    main()
