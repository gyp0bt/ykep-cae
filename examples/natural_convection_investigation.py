"""自然対流シミュレーション調査スクリプト.

2次元直交格子の非定常熱流体モデルを使い、中心発熱源からの自然対流の
発達度合いを、ドメインサイズ・発熱量・境界条件の観点から調査する。

全ケース非定常time-marchingで実行（定常SIMPLEではq_volソース付き
自然対流問題が発散するため）。

Phase 1 (制御解析):
  人工物性（高粘性・高熱伝導率）で安定したBC比較を実施。

Phase 2 (実物性解析):
  空気の実物性を用いた非定常解析で対流発達を確認。

境界条件パターン:
  A: 密閉キャビティ（全辺 NO_SLIP 壁 + DIRICHLET温度）
  B: 半開放（左右壁、下 INLET 微小速度、上 OUTLET_PRESSURE）
  C: 3辺開放（左右+上 OUTLET_PRESSURE、下 INLET 微小速度）
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

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


@dataclass
class CaseResult:
    """1ケースの結果."""

    phase: str
    bc_pattern: str
    domain_size: float
    q_vol: float
    Ra_or_Ra_star: float
    converged: bool
    n_iterations: int
    n_timesteps: int
    elapsed_s: float
    v_max: float
    delta_T_max: float
    mass_residual: float
    T_center: float
    T_max: float
    has_nan: bool = False


# ---------------------------------------------------------------------------
# 境界条件ビルダー
# ---------------------------------------------------------------------------


def _build_bc_closed(T_ref: float) -> dict[str, FluidBoundarySpec]:
    """密閉キャビティ: 全辺 NO_SLIP + DIRICHLET温度."""
    wall = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {"bc_xm": wall, "bc_xp": wall, "bc_ym": wall, "bc_yp": wall}


def _build_bc_semi_open(T_ref: float, v_inlet: float = 0.001) -> dict[str, FluidBoundarySpec]:
    """半開放: 左右壁、下 INLET、上 OUTLET_PRESSURE."""
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
    """3辺開放: 左右+上 OUTLET_PRESSURE、下 INLET."""
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


def _last_residual(result, key: str) -> float:
    hist = result.residual_history.get(key, [])
    return float(hist[-1]) if hist else float("nan")


def _build_q_vol(nx: int, ny: int, nz: int, q_value: float) -> np.ndarray:
    """中心 20% 領域に発熱源を配置."""
    heater_cells = max(1, nx // 5)
    i_s = nx // 2 - heater_cells // 2
    j_s = ny // 2 - heater_cells // 2
    q_vol = np.zeros((nx, ny, nz))
    q_vol[i_s : i_s + heater_cells, j_s : j_s + heater_cells, :] = q_value
    return q_vol


def run_transient_case(
    phase: str,
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
    nx: int = 20,
    ny: int = 20,
    alpha_u: float = 0.5,
    alpha_p: float = 0.2,
    alpha_T: float = 0.8,
    max_simple: int = 20,
) -> CaseResult:
    """非定常ケースを実行."""
    nz = 1
    Lz = L / nx

    q_vol = _build_q_vol(nx, ny, nz, q_value)
    bcs = BC_BUILDERS[bc_pattern](T_REF)
    z_bc = FluidBoundarySpec(
        condition=FluidBoundaryCondition.SYMMETRY,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )

    nu = mu / rho
    alpha_th = k_fluid / (rho * Cp)
    ra_star = 9.81 * beta * q_value * L**5 / (k_fluid * nu * alpha_th)
    t_end = dt * n_steps

    logger.info(
        "%s: bc=%s, L=%.3f, q=%.0f, Ra*=%.2e, dt=%.4f, steps=%d",
        phase,
        bc_pattern,
        L,
        q_value,
        ra_star,
        dt,
        n_steps,
    )

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
        t_end=t_end,
        max_simple_iter=max_simple,
        tol_simple=1e-3,
        alpha_u=alpha_u,
        alpha_p=alpha_p,
        alpha_T=alpha_T,
        output_interval=max(1, n_steps // 5),
    )

    solver = NaturalConvectionFDMProcess()
    result = solver.process(inp)

    has_nan = bool(np.any(np.isnan(result.T)) or np.any(np.isnan(result.u)))
    if has_nan:
        v_max = delta_T_max = T_center = T_max = mass_res = float("nan")
    else:
        v_max = float(max(np.abs(result.u).max(), np.abs(result.v).max()))
        delta_T_max = float(result.T.max() - T_REF)
        T_center = float(result.T[nx // 2, ny // 2, 0])
        T_max = float(result.T.max())
        mass_res = _last_residual(result, "mass")

    logger.info(
        "  -> conv=%s, steps=%d, v_max=%.4e, dT_max=%.2f, T_ctr=%.2f, %.1fs%s",
        result.converged,
        result.n_timesteps,
        v_max if not has_nan else 0.0,
        delta_T_max if not has_nan else 0.0,
        T_center if not has_nan else 0.0,
        result.elapsed_seconds,
        " [NaN!]" if has_nan else "",
    )

    return CaseResult(
        phase=phase,
        bc_pattern=bc_pattern,
        domain_size=L,
        q_vol=q_value,
        Ra_or_Ra_star=round(ra_star, 2),
        converged=result.converged,
        n_iterations=result.n_outer_iterations,
        n_timesteps=result.n_timesteps,
        elapsed_s=round(result.elapsed_seconds, 2),
        v_max=round(v_max, 8) if not has_nan else float("nan"),
        delta_T_max=round(delta_T_max, 4) if not has_nan else float("nan"),
        mass_residual=round(mass_res, 8) if not has_nan else float("nan"),
        T_center=round(T_center, 4) if not has_nan else float("nan"),
        T_max=round(T_max if not has_nan else float("nan"), 4),
        has_nan=has_nan,
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main():
    """2段階パラメトリックスタディを実行."""
    results: list[dict] = []
    t_start = time.perf_counter()

    # ===== Phase 1: 制御解析（高粘性人工物性、非定常） =====
    logger.info("=" * 70)
    logger.info("Phase 1: 制御解析（人工物性、非定常time-marching）")
    logger.info("=" * 70)

    # 人工物性: mu=0.01, k=1.0 → nu=0.01, alpha=0.001
    # dx=L/nx=0.005 (L=0.1, nx=20)
    # dt制約: dt < dx²/(2α) = 0.005²/(2*0.001) = 0.0125
    P1_RHO, P1_MU, P1_CP, P1_K = 1.0, 0.01, 1000.0, 1.0

    phase1_cases = [
        # (L, q_vol, beta, dt, n_steps, label)
        (0.1, 5000.0, 0.001, 0.005, 200, "low_beta"),
        (0.1, 5000.0, 0.01, 0.005, 200, "mid_beta"),
        (0.1, 20000.0, 0.001, 0.005, 200, "high_q"),
        (0.2, 5000.0, 0.001, 0.01, 200, "large_domain"),
    ]

    for L, q, beta, dt, n_steps, _label in phase1_cases:
        for bc in ["A_closed", "B_semi_open", "C_three_open"]:
            logger.info("-" * 50)
            try:
                cr = run_transient_case(
                    "phase1",
                    bc,
                    L,
                    q,
                    P1_RHO,
                    P1_MU,
                    P1_CP,
                    P1_K,
                    beta,
                    dt,
                    n_steps,
                )
                results.append(asdict(cr))
            except Exception as e:
                logger.error("Phase1 failed: %s", e)
                results.append(
                    {
                        "phase": "phase1",
                        "bc_pattern": bc,
                        "domain_size": L,
                        "q_vol": q,
                        "converged": False,
                        "error": str(e),
                    }
                )

    # ===== Phase 2: 実物性解析（空気 300K、非定常） =====
    logger.info("=" * 70)
    logger.info("Phase 2: 実物性解析（空気 300K、非定常）")
    logger.info("=" * 70)

    # 空気物性 (300K)
    A_RHO, A_MU = 1.18, 1.85e-5
    A_CP, A_K = 1007.0, 0.026
    A_BETA = 1.0 / 300.0
    A_ALPHA = A_K / (A_RHO * A_CP)

    # L=0.05, nx=20: dx=0.0025
    # dt < dx²/(2α) = 0.0025²/(2*2.19e-5) = 0.143
    phase2_cases = [
        # (L, q_vol, dt, n_steps)
        (0.05, 100.0, 0.05, 100),
        (0.05, 500.0, 0.05, 100),
        (0.1, 100.0, 0.1, 100),
        (0.1, 500.0, 0.1, 100),
    ]

    for L, q, dt, n_steps in phase2_cases:
        for bc in ["A_closed", "B_semi_open", "C_three_open"]:
            logger.info("-" * 50)
            try:
                cr = run_transient_case(
                    "phase2",
                    bc,
                    L,
                    q,
                    A_RHO,
                    A_MU,
                    A_CP,
                    A_K,
                    A_BETA,
                    dt,
                    n_steps,
                    alpha_u=0.5,
                    alpha_p=0.2,
                    alpha_T=0.8,
                    max_simple=30,
                )
                results.append(asdict(cr))
            except Exception as e:
                logger.error("Phase2 failed: %s", e)
                results.append(
                    {
                        "phase": "phase2",
                        "bc_pattern": bc,
                        "domain_size": L,
                        "q_vol": q,
                        "converged": False,
                        "error": str(e),
                    }
                )

    total_time = time.perf_counter() - t_start

    # 結果出力
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "natural_convection_investigation.yaml"

    output_data = {
        "investigation": "natural_convection_parametric_study",
        "date": "2026-04-01",
        "method": "transient SIMPLE + FDM (Rhie-Chow) + Boussinesq + q_vol",
        "mesh": "nx=ny=20, nz=1 (2D), heater=center 20%",
        "phases": {
            "phase1": {
                "description": "制御解析（人工物性・非定常）",
                "rho": P1_RHO,
                "mu": P1_MU,
                "Cp": P1_CP,
                "k": P1_K,
            },
            "phase2": {
                "description": "実物性解析（空気 300K・非定常）",
                "rho": A_RHO,
                "mu": A_MU,
                "Cp": A_CP,
                "k": A_K,
                "beta": round(A_BETA, 6),
                "Pr": round((A_MU / A_RHO) / A_ALPHA, 4),
            },
        },
        "total_elapsed_s": round(total_time, 1),
        "results": results,
    }

    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)

    _print_summary(results, total_time)
    logger.info("Results: %s", output_path)
    return 0


def _print_summary(results: list[dict], total_time: float):
    """結果サマリー表を表示."""
    print("\n" + "=" * 120)
    print(
        f"{'Phase':>7} {'BC':>15} {'L(m)':>6} {'q':>8} {'Ra*':>12} "
        f"{'conv':>5} {'steps':>6} {'v_max':>10} {'dT_max(K)':>10} "
        f"{'T_center':>10} {'T_max':>10} {'NaN':>4}"
    )
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(
                f"{'?':>7} {r.get('bc_pattern', '?'):>15} "
                f"{r.get('domain_size', 0):>6.3f} {r.get('q_vol', 0):>8.0f} ERROR: {r['error'][:40]}"
            )
            continue
        print(
            f"{r['phase']:>7} {r['bc_pattern']:>15} {r['domain_size']:>6.3f} "
            f"{r['q_vol']:>8.0f} {r['Ra_or_Ra_star']:>12.2e} "
            f"{str(r['converged']):>5} {r['n_timesteps']:>6} "
            f"{r['v_max']:>10.4e} {r['delta_T_max']:>10.4f} "
            f"{r['T_center']:>10.4f} {r.get('T_max', float('nan')):>10.4f} "
            f"{str(r.get('has_nan', False)):>4}"
        )
    print("=" * 120)
    print(f"Total elapsed: {total_time:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
