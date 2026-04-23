"""水槽フィルター循環デモ (Phase 6.3b).

`AquariumGeometryProcess` + `HeaterProcess` + `AquariumFilterProcess` +
`NaturalConvectionFDMProcess` の 4 段 Process 連携。小型水槽 (30×10×15 cm) で
外部フィルター循環（INLET 上部から +x 吐出、OUTLET 右下から吸入）と
ヒーター自然対流（左中央）を同時に解く。

出力:
    output/aquarium_filter_circulation.png
        左 : 温度分布 + 速度ベクトル（y 中央 xz 断面、ヒーター/INLET/OUTLET 位置）
        右 : 流入/流出体積流量バーと残差履歴（質量保存チェック）

注意: status-25 の既知課題で SIMPLE 連成の mass 残差は有限反復では完全には
消えない（xfail 扱い）。本デモは「オーダーが一致するか」の定性検証が目的。
SIMPLEC/PISO 実装後に厳密化する。物性は既存 NC ベンチマークで安定動作が
確認されている人工値（mu=0.01, rho=1.0）を採用（実物性対応は Phase 6.0 後）。

STA2 防止: tee でログ保存。
    python examples/aquarium_filter_circulation.py 2>&1 | tee /tmp/log-$(date +%s).log
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
    AquariumFilterInput,
    AquariumFilterProcess,
    AquariumGeometryInput,
    AquariumGeometryProcess,
    HeaterGeometry,
    HeaterInput,
    HeaterMode,
    HeaterProcess,
    NozzleGeometry,
)
from xkep_cae_fluid.natural_convection import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionFDMProcess,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- 物性（人工安定化。既存 NC ベンチマークと同一） -----------------------------
FLUID_RHO = 1.0
FLUID_MU = 0.01
FLUID_CP = 1000.0
FLUID_K = 1.0
FLUID_BETA = 1.0e-3
T_REF = 298.15

# --- 水槽形状（30×10×15 cm 小型）と機材配置 ------------------------------------
LX, LY, LZ = 0.30, 0.10, 0.15
NX, NY, NZ = 12, 4, 8

HEATER_BOX = HeaterGeometry(
    x_range=(0.13, 0.18),
    y_range=(0.03, 0.07),
    z_range=(0.04, 0.11),
)
INFLOW_BOX = NozzleGeometry(
    x_range=(0.025, 0.055),
    y_range=(0.03, 0.07),
    z_range=(0.12, 0.15),
)
OUTFLOW_BOX = NozzleGeometry(
    x_range=(0.245, 0.275),
    y_range=(0.03, 0.07),
    z_range=(0.0, 0.03),
)


def build_inputs(
    heater_power_watts: float = 2.0,
    flow_rate_lph: float = 120.0,
    inflow_temperature_K: float | None = None,
    dt: float = 0.2,
    t_end: float = 12.0,
):
    geom = AquariumGeometryProcess().process(
        AquariumGeometryInput(Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ, substrate_depth=0.0)
    )
    heater = HeaterProcess().process(
        HeaterInput(
            x_centers=geom.x_centers,
            y_centers=geom.y_centers,
            z_centers=geom.z_centers,
            dx=geom.dx,
            dy=geom.dy,
            dz=geom.dz,
            geometry=HEATER_BOX,
            mode=HeaterMode.CONSTANT_FLUX,
            power_watts=heater_power_watts,
        )
    )
    filt = AquariumFilterProcess().process(
        AquariumFilterInput(
            x_centers=geom.x_centers,
            y_centers=geom.y_centers,
            z_centers=geom.z_centers,
            dx=geom.dx,
            dy=geom.dy,
            dz=geom.dz,
            inflow_geometry=INFLOW_BOX,
            outflow_geometry=OUTFLOW_BOX,
            flow_rate_lph=flow_rate_lph,
            inflow_direction=(1.0, 0.0, 0.0),
            inflow_temperature_K=inflow_temperature_K,
            label="eheim_like",
        )
    )

    wall = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )
    T0 = np.full((NX, NY, NZ), T_REF)
    nc = NaturalConvectionInput(
        Lx=LX,
        Ly=LY,
        Lz=LZ,
        nx=NX,
        ny=NY,
        nz=NZ,
        rho=FLUID_RHO,
        mu=FLUID_MU,
        Cp=FLUID_CP,
        k_fluid=FLUID_K,
        beta=FLUID_BETA,
        T_ref=T_REF,
        gravity=geom.gravity,
        solid_mask=geom.solid_mask,
        q_vol=heater.q_vol,
        T0=T0,
        bc_xm=wall,
        bc_xp=wall,
        bc_ym=wall,
        bc_yp=wall,
        bc_zm=wall,
        bc_zp=wall,
        dt=dt,
        t_end=t_end,
        max_simple_iter=80,
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
        internal_face_bcs=(filt.inflow_bc, filt.outflow_bc),
    )
    return nc, geom, heater, filt


def compute_flux(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    dx: np.ndarray,
    direction: tuple[float, float, float],
) -> float:
    """マスクセルの方向投影面積で体積流量 [m^3/s] を積算."""
    ax = np.einsum("j,k->jk", dy, dz)
    ay = np.einsum("i,k->ik", dx, dz)
    az = np.einsum("i,j->ij", dx, dy)
    flux = 0.0
    idx = np.where(mask)
    for i, j, k in zip(*idx, strict=True):
        flux += direction[0] * u[i, j, k] * ax[j, k]
        flux += direction[1] * v[i, j, k] * ay[i, k]
        flux += direction[2] * w[i, j, k] * az[i, j]
    return float(flux)


def plot_results(res, geom, heater_inp, filt, nc_inp, mass_balance, out_path: Path):
    """温度+速度断面 + 質量保存バー + 残差履歴を 2x2 で描画."""
    j_mid = NY // 2
    T_slice = res.T[:, j_mid, :]
    u_slice = res.u[:, j_mid, :]
    w_slice = res.w[:, j_mid, :]
    X, Z = np.meshgrid(geom.x_centers * 100, geom.z_centers * 100, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # [0,0] 温度 + 速度ベクトル + 機材位置
    ax = axes[0, 0]
    T_plot = np.ma.array(T_slice - 273.15, mask=geom.solid_mask[:, j_mid, :])
    im = ax.pcolormesh(X, Z, T_plot, cmap="hot", shading="auto")
    plt.colorbar(im, ax=ax, label="T [°C]", shrink=0.85)
    skip = 1
    speed = np.sqrt(u_slice**2 + w_slice**2)
    if speed.max() > 1e-8:
        ax.quiver(
            X[::skip, ::skip],
            Z[::skip, ::skip],
            u_slice[::skip, ::skip],
            w_slice[::skip, ::skip],
            color="cyan",
            alpha=0.85,
            scale_units="xy",
            angles="xy",
        )
    for box, col, label in (
        (HEATER_BOX, "lime", "Heater"),
        (INFLOW_BOX, "deepskyblue", "Inflow (INLET)"),
        (OUTFLOW_BOX, "magenta", "Outflow (OUTLET)"),
    ):
        hx0, hx1 = box.x_range
        hz0, hz1 = box.z_range
        ax.add_patch(
            plt.Rectangle(
                (hx0 * 100, hz0 * 100),
                (hx1 - hx0) * 100,
                (hz1 - hz0) * 100,
                linewidth=2,
                edgecolor=col,
                facecolor="none",
                linestyle="--",
                label=label,
            )
        )
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_aspect("equal")
    ax.set_title(f"T + velocity (y=mid, t={nc_inp.t_end:.0f}s)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # [0,1] ヒーター通過温度プロファイル（z 方向平均を x で）
    ax = axes[0, 1]
    T_xavg = (res.T - 273.15).mean(axis=(1, 2))
    ax.plot(geom.x_centers * 100, T_xavg, "r-", lw=2, label="mean over (y, z)")
    inflow_x = np.array(INFLOW_BOX.x_range) * 100
    outflow_x = np.array(OUTFLOW_BOX.x_range) * 100
    ax.axvspan(inflow_x[0], inflow_x[1], color="deepskyblue", alpha=0.2, label="Inflow")
    ax.axvspan(outflow_x[0], outflow_x[1], color="magenta", alpha=0.2, label="Outflow")
    ax.axvspan(
        HEATER_BOX.x_range[0] * 100,
        HEATER_BOX.x_range[1] * 100,
        color="lime",
        alpha=0.2,
        label="Heater",
    )
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("mean T [°C]")
    ax.set_title("Heater / inflow interaction (T along x)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # [1,0] 質量保存バー
    ax = axes[1, 0]
    target_Q = filt.flow_rate_m3s
    q_in, q_out = mass_balance
    bars = ax.bar(
        ["target Q", "|inflow|", "|outflow|"],
        [target_Q, abs(q_in), abs(q_out)],
        color=["gray", "deepskyblue", "magenta"],
    )
    for b, val in zip(bars, [target_Q, abs(q_in), abs(q_out)], strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("volumetric flux [m³/s]")
    ratio = abs(q_out) / abs(q_in) if abs(q_in) > 0 else float("nan")
    ax.set_title(f"Mass balance (outflow/inflow = {ratio:.3f})")
    ax.grid(axis="y", alpha=0.3)

    # [1,1] 残差履歴
    ax = axes[1, 1]
    for key in ("u", "v", "w", "p", "T", "mass"):
        hist = res.residual_history.get(key, [])
        if hist:
            ax.semilogy(hist, label=key, alpha=0.8)
    ax.set_xlabel("SIMPLE iteration (accumulated)")
    ax.set_ylabel("residual")
    ax.set_title("Residual history (log scale)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Aquarium filter circulation demo "
        f"({LX * 100:.0f}×{LY * 100:.0f}×{LZ * 100:.0f} cm, "
        f"Q={filt.flow_rate_m3s * 1e3 * 3600:.0f} L/h, "
        f"heater={nc_inp.q_vol[nc_inp.q_vol > 0].sum() * nc_inp.dx * nc_inp.dy * nc_inp.dz:.1f} W)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def main() -> int:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 注: status-25 の既知 SIMPLE mass 残差増幅により、buoyancy 駆動下では
    # t>2-3s 付近で累積誤差により速度場が崩壊するケースがある。本デモでは
    # 強制循環の立ち上がりが観測できる短時間 (t_end=2.0s) で止め、
    # 残る質量不整合を可視化する。SIMPLEC/PISO 後に長時間安定化予定。
    nc_inp, geom, heater_inp, filt = build_inputs(
        heater_power_watts=2.0,
        flow_rate_lph=60.0,
        inflow_temperature_K=None,
        dt=0.2,
        t_end=2.0,
    )
    logger.info(
        "Domain: %.2f×%.2f×%.2f m grid=(%d,%d,%d) | Q=%.3e m³/s, |v|_in=%.3e m/s",
        LX,
        LY,
        LZ,
        NX,
        NY,
        NZ,
        filt.flow_rate_m3s,
        float(np.linalg.norm(filt.inflow_velocity)),
    )

    solver = NaturalConvectionFDMProcess()
    res = solver.process(nc_inp)
    logger.info(
        "Done: converged=%s, n_steps=%d, elapsed=%.2fs",
        res.converged,
        res.n_timesteps,
        res.elapsed_seconds,
    )

    q_in = compute_flux(
        res.u, res.v, res.w, filt.inflow_mask, geom.dy, geom.dz, geom.dx, (1.0, 0.0, 0.0)
    )
    # OUTLET 側は速度符号が自由なので絶対値で流出量を定義
    q_out_raw = compute_flux(
        res.u, res.v, res.w, filt.outflow_mask, geom.dy, geom.dz, geom.dx, (1.0, 0.0, 0.0)
    )
    logger.info(
        "Mass balance: inflow Q=%.3e (target %.3e), outflow signed Q=%.3e, ratio=%.3f",
        q_in,
        filt.flow_rate_m3s,
        q_out_raw,
        abs(q_out_raw) / abs(q_in) if abs(q_in) > 0 else float("nan"),
    )

    plot_results(
        res,
        geom,
        heater_inp,
        filt,
        nc_inp,
        mass_balance=(q_in, q_out_raw),
        out_path=output_dir / "aquarium_filter_circulation.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
