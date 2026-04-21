"""水槽ヒーター自然対流デモ (Phase 6.2b).

`AquariumGeometryProcess` でメッシュ + 底床マスクを生成し、`HeaterProcess` で
ヒーター体積熱源 `q_vol` を作成、`NaturalConvectionFDMProcess` で水槽内の
温度分布と自然対流循環を解く最小デモ。Process 連携の動作確認が目的。

出力:
    output/aquarium_heater_natural_convection.png
        温度分布（正面図: x-z 平面, y=中央）+ 速度ベクトル + ヒーター位置

注意: 現状の NaturalConvectionFDMProcess は実水の物性（mu=1e-3, 高 Ra）では
SIMPLE 連成が各ステップ内で十分収束しないため（status-11 参照）、本デモでは
小型水槽 (30×10×15 cm) と既存ベンチマークで安定動作が確認されている人工
物性（mu=0.01, rho=1.0, Cp=1000, k=1.0, beta=1e-3）を採用する。実物性対応は
Phase 6 後半の SIMPLEC/PISO 拡張 (status-12 残存課題) と並行して進める。

STA2 防止: tee でログ保存、YAML 出力と照合可能にする。
    python examples/aquarium_heater_natural_convection.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xkep_cae_fluid.aquarium import (
    AquariumGeometryInput,
    AquariumGeometryProcess,
    AquariumGeometryResult,
    HeaterGeometry,
    HeaterInput,
    HeaterMode,
    HeaterProcess,
)
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


# ---------------------------------------------------------------------------
# 物性（人工安定化、既存 NC ベンチマーク Phase 1 と同一）
# ---------------------------------------------------------------------------
FLUID_RHO = 1.0  # [kg/m³] 人工
FLUID_MU = 0.01  # [Pa·s] 人工（実水 1e-3 の 10 倍）
FLUID_CP = 1000.0  # [J/(kg·K)] 人工
FLUID_K = 1.0  # [W/(m·K)] 人工
FLUID_BETA = 1.0e-3  # [1/K] 人工
SAND_K = 0.3  # [W/(m·K)] 砂層（大まかな値）
T_REF = 298.15  # 25°C 基準


def build_demo_inputs(
    nx: int = 12,
    ny: int = 4,
    nz: int = 8,
    heater_power_watts: float = 2.0,
    dt: float = 0.1,
    t_end: float = 10.0,
) -> tuple[NaturalConvectionInput, HeaterInput, AquariumGeometryResult]:
    """デモ用入力を組み立てる（30×10×15 cm 小型水槽を粗い格子で表現）."""
    geom_inp = AquariumGeometryInput(
        Lx=0.3,
        Ly=0.1,
        Lz=0.15,
        nx=nx,
        ny=ny,
        nz=nz,
        substrate_depth=0.02,  # 2 cm 砂層
        substrate_refinement_ratio=1.5,  # 底床側を少し細かく
    )
    geom_res = AquariumGeometryProcess().process(geom_inp)

    # ヒーター: 右奥、z=5〜12 cm の棒状ヒーター
    heater_inp = HeaterInput(
        x_centers=geom_res.x_centers,
        y_centers=geom_res.y_centers,
        z_centers=geom_res.z_centers,
        dx=geom_res.dx,
        dy=geom_res.dy,
        dz=geom_res.dz,
        geometry=HeaterGeometry(
            x_range=(0.23, 0.28),
            y_range=(0.03, 0.07),
            z_range=(0.05, 0.12),
        ),
        mode=HeaterMode.CONSTANT_FLUX,
        power_watts=heater_power_watts,
    )
    heater_res = HeaterProcess().process(heater_inp)

    # 境界条件: 全面 NO_SLIP + ADIABATIC（水面も rigid-lid 断熱近似）
    wall_adiabatic = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )

    T0 = np.full((nx, ny, nz), T_REF)
    k_solid = np.full((nx, ny, nz), SAND_K)

    nc_inp = NaturalConvectionInput(
        Lx=geom_inp.Lx,
        Ly=geom_inp.Ly,
        Lz=geom_inp.Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=FLUID_RHO,
        mu=FLUID_MU,
        Cp=FLUID_CP,
        k_fluid=FLUID_K,
        beta=FLUID_BETA,
        T_ref=T_REF,
        gravity=geom_res.gravity,  # (0, 0, -9.81)
        solid_mask=geom_res.solid_mask,
        k_solid=k_solid,
        q_vol=heater_res.q_vol,
        T0=T0,
        bc_xm=wall_adiabatic,
        bc_xp=wall_adiabatic,
        bc_ym=wall_adiabatic,
        bc_yp=wall_adiabatic,
        bc_zm=wall_adiabatic,
        bc_zp=wall_adiabatic,
        dt=dt,
        t_end=t_end,
        max_simple_iter=30,
        max_inner_iter=40,
        tol_simple=5e-3,
        tol_inner=1e-8,
        alpha_u=0.5,
        alpha_p=0.3,
        alpha_T=0.8,
        coupling_method="simple",
        pressure_solver="bicgstab",
        convection_scheme="upwind",
        time_scheme="euler",
    )
    return nc_inp, heater_inp, geom_res


def plot_xz_slice(
    nc_res,
    geom_res,
    heater_inp: HeaterInput,
    output_path: Path,
    title: str,
):
    """y 方向中央断面の温度 + 速度ベクトル + ヒーター位置を描画."""
    j_mid = nc_res.T.shape[1] // 2

    T_slice = nc_res.T[:, j_mid, :]
    u_slice = nc_res.u[:, j_mid, :]
    w_slice = nc_res.w[:, j_mid, :]

    X, Z = np.meshgrid(geom_res.x_centers * 100, geom_res.z_centers * 100, indexing="ij")

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    T_plot = np.ma.array(T_slice - 273.15, mask=geom_res.solid_mask[:, j_mid, :])
    im = ax.pcolormesh(X, Z, T_plot, cmap="hot", shading="auto")
    cbar = plt.colorbar(im, ax=ax, label="T [°C]", shrink=0.9)
    cbar.ax.tick_params(labelsize=9)

    speed = np.sqrt(u_slice**2 + w_slice**2)
    v_max = float(speed.max())

    skip_x = max(1, nc_res.u.shape[0] // 14)
    skip_z = max(1, nc_res.u.shape[2] // 12)
    if v_max > 1e-8:
        ax.quiver(
            X[::skip_x, ::skip_z],
            Z[::skip_x, ::skip_z],
            u_slice[::skip_x, ::skip_z],
            w_slice[::skip_x, ::skip_z],
            color="cyan",
            alpha=0.85,
            scale_units="xy",
            angles="xy",
        )

    # ヒーター位置を緑破線で
    hx0, hx1 = heater_inp.geometry.x_range
    hz0, hz1 = heater_inp.geometry.z_range
    rect = plt.Rectangle(
        (hx0 * 100, hz0 * 100),
        (hx1 - hx0) * 100,
        (hz1 - hz0) * 100,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        linestyle="--",
        label="Heater",
    )
    ax.add_patch(rect)

    # 底床グレー
    sd_cm = geom_res.dz[0] * 100 * np.sum(geom_res.substrate_mask[0, 0, :])
    if sd_cm > 0:
        ax.axhspan(0, sd_cm, color="sandybrown", alpha=0.3, label="Substrate")

    ax.set_xlabel("x [cm] (width)")
    ax.set_ylabel("z [cm] (height)")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def main() -> int:
    """水槽ヒーター自然対流デモ実行."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    nc_inp, heater_inp, geom_res = build_demo_inputs(
        nx=12,
        ny=4,
        nz=8,
        heater_power_watts=2.0,
        dt=0.1,
        t_end=10.0,
    )

    logger.info(
        "Domain: %.2f×%.2f×%.2f m, grid=(%d,%d,%d), substrate=%d cells",
        nc_inp.Lx,
        nc_inp.Ly,
        nc_inp.Lz,
        nc_inp.nx,
        nc_inp.ny,
        nc_inp.nz,
        int(geom_res.substrate_mask.sum()),
    )
    logger.info("Heater: power=%.1f W, mode=%s", heater_inp.power_watts, heater_inp.mode.value)

    solver = NaturalConvectionFDMProcess()
    result = solver.process(nc_inp)
    logger.info(
        "Done: converged=%s, n_steps=%d, t_total=%.1fs, elapsed=%.2fs",
        result.converged,
        result.n_timesteps,
        nc_inp.t_end,
        result.elapsed_seconds,
    )
    T_water = result.T[~geom_res.solid_mask]
    logger.info(
        "T (water): min=%.2f°C, max=%.2f°C, mean=%.2f°C",
        T_water.min() - 273.15,
        T_water.max() - 273.15,
        T_water.mean() - 273.15,
    )

    v_max = float(np.sqrt(result.u**2 + result.v**2 + result.w**2).max())
    logger.info("|v|_max = %.3e m/s", v_max)

    plot_xz_slice(
        result,
        geom_res,
        heater_inp,
        output_dir / "aquarium_heater_natural_convection.png",
        title=(
            f"Aquarium heater natural convection (y=mid, t={nc_inp.t_end:.0f}s)\n"
            f"30×10×15 cm, artificial fluid, {heater_inp.power_watts:.1f} W heater, "
            f"dT={T_water.max() - T_REF:.2f} K, |v|_max={v_max:.2e} m/s"
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
