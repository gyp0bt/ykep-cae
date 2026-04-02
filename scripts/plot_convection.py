"""2cm 空気実物性キャビティの自然対流コンター図.

流速ベクトル + 温度・速度コンター図を生成して物理的妥当性を確認する。
"""

import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess

logging.basicConfig(level=logging.WARNING)


def run_simulation():
    """t=5s まで計算."""
    L = 0.02
    T_hot, T_cold = 310.0, 300.0
    inp = NaturalConvectionInput(
        Lx=L,
        Ly=L,
        Lz=L * 3 / 20,
        nx=20,
        ny=20,
        nz=3,
        rho=1.19,
        mu=1.85e-5,
        Cp=1007.0,
        k_fluid=0.026,
        beta=3.3e-3,
        T_ref=0.5 * (T_hot + T_cold),
        gravity=(0.0, -9.81, 0.0),
        bc_xm=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_hot,
        ),
        bc_xp=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_cold,
        ),
        bc_ym=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_yp=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_zm=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_zp=FluidBoundarySpec(
            condition=FluidBoundaryCondition.NO_SLIP,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        dt=0.01,
        t_end=5.0,
        max_simple_iter=50,
        tol_simple=1e-4,
        alpha_u=0.7,
        alpha_p=0.3,
        coupling_method="piso",
        pressure_solver="amg",
        max_pressure_iter=100,
        output_interval=50,
    )

    print("計算開始: 20x20x3 空気実物性 2cm cavity, t_end=5.0s")
    t0 = time.perf_counter()
    result = NaturalConvectionFDMProcess().process(inp)
    elapsed = time.perf_counter() - t0
    print(f"計算完了: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  収束={result.converged}, NaN={np.any(np.isnan(result.u))}")
    print(f"  |v|_max={np.abs(result.v).max():.4e} m/s")
    print(f"  T=[{result.T.min():.2f}, {result.T.max():.2f}] K")

    return inp, result


def plot_contours(inp, result, outpath="/tmp/natural_convection_contours.png"):
    """中央断面 (z=nz//2) のコンター図を4パネルで描画."""
    nz_mid = inp.nz // 2
    L = inp.Lx

    # 中央断面の2Dスライス
    T_2d = result.T[:, :, nz_mid]
    u_2d = result.u[:, :, nz_mid]
    v_2d = result.v[:, :, nz_mid]
    p_2d = result.p[:, :, nz_mid]
    speed = np.sqrt(u_2d**2 + v_2d**2)

    # セル中心座標
    x = np.linspace(inp.dx / 2, L - inp.dx / 2, inp.nx)
    y = np.linspace(inp.dy / 2, L - inp.dy / 2, inp.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"2cm Differentially Heated Cavity (Air, Ra~7700)\n"
        f"20x20x3, PISO+AMG, t=5.0s",
        fontsize=14,
        fontweight="bold",
    )

    # --- Panel 1: Temperature ---
    ax = axes[0, 0]
    cf = ax.contourf(X * 1000, Y * 1000, T_2d, levels=30, cmap="RdBu_r")
    cs = ax.contour(X * 1000, Y * 1000, T_2d, levels=10, colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    plt.colorbar(cf, ax=ax, label="T [K]")
    ax.set_title("Temperature")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    # --- Panel 2: Velocity magnitude + vectors ---
    ax = axes[0, 1]
    cf = ax.contourf(X * 1000, Y * 1000, speed * 1000, levels=30, cmap="hot_r")
    plt.colorbar(cf, ax=ax, label="|V| [mm/s]")
    # Quiver (subsample for clarity)
    skip = 2
    ax.quiver(
        X[::skip, ::skip] * 1000,
        Y[::skip, ::skip] * 1000,
        u_2d[::skip, ::skip],
        v_2d[::skip, ::skip],
        color="blue",
        alpha=0.7,
        scale=0.05,
    )
    ax.set_title("Velocity Magnitude + Vectors")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    # --- Panel 3: u-velocity (horizontal) ---
    ax = axes[1, 0]
    u_mm = u_2d * 1000
    vmax_u = max(abs(u_mm.min()), abs(u_mm.max()))
    cf = ax.contourf(
        X * 1000, Y * 1000, u_mm, levels=30, cmap="coolwarm", vmin=-vmax_u, vmax=vmax_u
    )
    plt.colorbar(cf, ax=ax, label="u [mm/s]")
    ax.set_title("u-velocity (horizontal)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    # --- Panel 4: v-velocity (vertical) ---
    ax = axes[1, 1]
    v_mm = v_2d * 1000
    vmax_v = max(abs(v_mm.min()), abs(v_mm.max()))
    cf = ax.contourf(
        X * 1000, Y * 1000, v_mm, levels=30, cmap="coolwarm", vmin=-vmax_v, vmax=vmax_v
    )
    plt.colorbar(cf, ax=ax, label="v [mm/s]")
    ax.set_title("v-velocity (vertical)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n図を保存: {outpath}")

    # --- 追加: 中心線プロファイル ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Midline Profiles", fontsize=13, fontweight="bold")

    # v along horizontal midline (y=L/2)
    mid_y = inp.ny // 2
    ax2 = axes2[0]
    ax2.plot(x * 1000, v_2d[:, mid_y] * 1000, "b-o", markersize=3)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("v [mm/s]")
    ax2.set_title("v-velocity at y=L/2")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # T along horizontal midline
    ax2 = axes2[1]
    ax2.plot(x * 1000, T_2d[:, mid_y], "r-o", markersize=3)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("T [K]")
    ax2.set_title("Temperature at y=L/2")
    ax2.grid(True, alpha=0.3)

    outpath2 = outpath.replace(".png", "_profiles.png")
    fig2.tight_layout()
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    print(f"図を保存: {outpath2}")

    return outpath, outpath2


if __name__ == "__main__":
    inp, result = run_simulation()
    p1, p2 = plot_contours(inp, result)
