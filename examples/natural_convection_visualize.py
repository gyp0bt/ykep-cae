"""自然対流シミュレーション結果の可視化.

温度場（カラーマップ）+ 速度ベクトル（矢印）を各ケースごとに画像保存する。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

T_REF = 300.0


# ---------------------------------------------------------------------------
# 境界条件ビルダー（investigation.py と同一）
# ---------------------------------------------------------------------------
def _build_bc_closed(T_ref: float) -> dict[str, FluidBoundarySpec]:
    wall = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {"bc_xm": wall, "bc_xp": wall, "bc_ym": wall, "bc_yp": wall}


def _build_bc_semi_open(T_ref: float, v_inlet: float = 0.001) -> dict[str, FluidBoundarySpec]:
    wall = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    inlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.INLET_VELOCITY,
        velocity=(0.0, v_inlet, 0.0),
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    outlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.OUTLET_PRESSURE,
        pressure=0.0,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {"bc_xm": wall, "bc_xp": wall, "bc_ym": inlet, "bc_yp": outlet}


def _build_bc_three_open(T_ref: float, v_inlet: float = 0.001) -> dict[str, FluidBoundarySpec]:
    outlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.OUTLET_PRESSURE,
        pressure=0.0,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    inlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.INLET_VELOCITY,
        velocity=(0.0, v_inlet, 0.0),
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {"bc_xm": outlet, "bc_xp": outlet, "bc_ym": inlet, "bc_yp": outlet}


BC_BUILDERS = {
    "A_closed": _build_bc_closed,
    "B_semi_open": _build_bc_semi_open,
    "C_three_open": _build_bc_three_open,
}

BC_LABELS = {
    "A_closed": "A: 密閉 (全辺 NO_SLIP)",
    "B_semi_open": "B: 半開放 (左右壁, 上 OUTLET)",
    "C_three_open": "C: 3辺開放 (左右+上 OUTLET)",
}


def run_and_visualize(
    bc_pattern: str,
    L: float,
    q_value: float,
    rho: float,
    mu: float,
    Cp: float,
    k_fluid: float,
    beta: float,
    dt: float,
    n_steps: int,
    nx: int,
    ny: int,
    output_dir: Path,
    label: str,
):
    """1ケースを実行し可視化."""
    nz = 1
    Lz = L / nx

    heater_cells = max(1, nx // 5)
    i_s = nx // 2 - heater_cells // 2
    j_s = ny // 2 - heater_cells // 2
    q_vol = np.zeros((nx, ny, nz))
    q_vol[i_s : i_s + heater_cells, j_s : j_s + heater_cells, :] = q_value

    bcs = BC_BUILDERS[bc_pattern](T_REF)
    z_bc = FluidBoundarySpec(
        condition=FluidBoundaryCondition.SYMMETRY,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )

    nu = mu / rho
    alpha_th = k_fluid / (rho * Cp)
    ra_star = 9.81 * beta * q_value * L**5 / (k_fluid * nu * alpha_th)

    logger.info("Run: %s, bc=%s, L=%.3f, q=%.0f, Ra*=%.2e", label, bc_pattern, L, q_value, ra_star)

    inp = NaturalConvectionInput(
        Lx=L,
        Ly=L,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=rho,
        mu=mu,
        Cp=Cp,
        k_fluid=k_fluid,
        beta=beta,
        T_ref=T_REF,
        gravity=(0.0, -9.81, 0.0),
        q_vol=q_vol,
        bc_xm=bcs["bc_xm"],
        bc_xp=bcs["bc_xp"],
        bc_ym=bcs["bc_ym"],
        bc_yp=bcs["bc_yp"],
        bc_zm=z_bc,
        bc_zp=z_bc,
        dt=dt,
        t_end=dt * n_steps,
        max_simple_iter=30,
        tol_simple=1e-3,
        alpha_u=0.5,
        alpha_p=0.2,
        alpha_T=0.8,
        output_interval=max(1, n_steps // 5),
    )

    solver = NaturalConvectionFDMProcess()
    result = solver.process(inp)

    # 2Dスライス (z=0)
    T_2d = result.T[:, :, 0]
    u_2d = result.u[:, :, 0]
    v_2d = result.v[:, :, 0]

    dx = L / nx
    x_centers = np.linspace(dx / 2, L - dx / 2, nx)
    y_centers = np.linspace(dx / 2, L - dx / 2, ny)
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    v_max = max(np.abs(u_2d).max(), np.abs(v_2d).max())
    speed = np.sqrt(u_2d**2 + v_2d**2)

    return T_2d, u_2d, v_2d, X, Y, speed, v_max, ra_star, result


def plot_comparison(cases_data: list[dict], output_path: Path, suptitle: str):
    """3つのBC条件を横並びで比較プロット."""
    n = len(cases_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, cases_data, strict=True):
        T_2d = case["T_2d"]
        u_2d = case["u_2d"]
        v_2d = case["v_2d"]
        X = case["X"]
        Y = case["Y"]
        bc = case["bc_pattern"]
        v_max = case["v_max"]
        ra_star = case["ra_star"]
        result = case["result"]

        # 温度カラーマップ
        T_min = T_REF - 5
        T_max_plot = max(T_REF + 10, float(T_2d.max()) * 1.05)
        # NaN以外の妥当な温度範囲にクリップ
        T_plot = np.clip(T_2d, T_min, min(T_max_plot, T_REF + 500))

        im = ax.pcolormesh(
            X * 1000,
            Y * 1000,
            T_plot,
            cmap="hot",
            shading="auto",
        )
        plt.colorbar(im, ax=ax, label="T [K]", shrink=0.85)

        # 速度ベクトル（間引き表示）
        nx, ny = T_2d.shape
        skip = max(1, nx // 10)
        if v_max > 1e-10:
            ax.quiver(
                X[::skip, ::skip] * 1000,
                Y[::skip, ::skip] * 1000,
                u_2d[::skip, ::skip],
                v_2d[::skip, ::skip],
                color="cyan",
                alpha=0.7,
                scale_units="xy",
                angles="xy",
            )

        # 発熱体領域表示
        heater_cells = max(1, nx // 5)
        i_s = nx // 2 - heater_cells // 2
        j_s = ny // 2 - heater_cells // 2
        L = case["L"]
        dx = L / nx
        hx0 = i_s * dx * 1000
        hy0 = j_s * dx * 1000
        hw = heater_cells * dx * 1000
        rect = plt.Rectangle(
            (hx0, hy0),
            hw,
            hw,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        conv_str = "conv" if result.converged else "NOT conv"
        ax.set_title(
            f"{BC_LABELS[bc]}\n"
            f"Ra*={ra_star:.1e}, v_max={v_max:.3e} m/s\n"
            f"dT_max={float(T_2d.max()) - T_REF:.1f} K, {conv_str}",
            fontsize=9,
        )

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def main():
    """可視化メイン."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # CJKフォント設定
    try:
        import matplotlib.font_manager as fm

        jp_fonts = [
            f.name for f in fm.fontManager.ttflist if "Gothic" in f.name or "gothic" in f.name
        ]
        if jp_fonts:
            plt.rcParams["font.family"] = jp_fonts[0]
    except Exception:
        pass

    # ===== 空気実物性ケース =====
    AIR = {"rho": 1.18, "mu": 1.85e-5, "Cp": 1007.0, "k_fluid": 0.026, "beta": 1.0 / 300.0}
    NX = 30

    configs = [
        # (L, q, dt, n_steps, filename_prefix, suptitle)
        (
            0.05,
            100.0,
            0.05,
            100,
            "nc_air_L005_q100",
            "Air (300K), L=50mm, q=100 W/m3, t=5.0s",
        ),
        (
            0.05,
            500.0,
            0.05,
            100,
            "nc_air_L005_q500",
            "Air (300K), L=50mm, q=500 W/m3, t=5.0s",
        ),
        (
            0.1,
            100.0,
            0.1,
            100,
            "nc_air_L010_q100",
            "Air (300K), L=100mm, q=100 W/m3, t=10.0s",
        ),
        (
            0.1,
            500.0,
            0.1,
            100,
            "nc_air_L010_q500",
            "Air (300K), L=100mm, q=500 W/m3, t=10.0s",
        ),
    ]

    for L, q, dt, n_steps, prefix, suptitle in configs:
        cases_data = []
        for bc in ["A_closed", "B_semi_open", "C_three_open"]:
            T_2d, u_2d, v_2d, X, Y, speed, v_max, ra_star, result = run_and_visualize(
                bc,
                L,
                q,
                dt=dt,
                n_steps=n_steps,
                nx=NX,
                ny=NX,
                output_dir=output_dir,
                label=prefix,
                **AIR,
            )
            cases_data.append(
                {
                    "bc_pattern": bc,
                    "T_2d": T_2d,
                    "u_2d": u_2d,
                    "v_2d": v_2d,
                    "X": X,
                    "Y": Y,
                    "speed": speed,
                    "v_max": v_max,
                    "ra_star": ra_star,
                    "result": result,
                    "L": L,
                }
            )

        plot_comparison(cases_data, output_dir / f"{prefix}.png", suptitle)

    logger.info("Done. All images saved to %s/", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
