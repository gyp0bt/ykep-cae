"""ポスト処理・面値⇄セル値の変換などのユーティリティ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nsb.core import NSBInput, NSBResult


def cell_to_face_x(phi: np.ndarray, phi_west: np.ndarray, phi_east: np.ndarray) -> np.ndarray:
    """セル値 (nx, ny) → x 面値 (nx+1, ny)。内部面は線形補間、境界面は与えた境界値."""
    f = np.empty((phi.shape[0] + 1, phi.shape[1]))
    f[1:-1] = 0.5 * (phi[:-1] + phi[1:])
    f[0], f[-1] = phi_west, phi_east
    return f


def cell_to_face_y(phi: np.ndarray, phi_south: np.ndarray, phi_north: np.ndarray) -> np.ndarray:
    """セル値 (nx, ny) → y 面値 (nx, ny+1)."""
    f = np.empty((phi.shape[0], phi.shape[1] + 1))
    f[:, 1:-1] = 0.5 * (phi[:, :-1] + phi[:, 1:])
    f[:, 0], f[:, -1] = phi_south, phi_north
    return f


def face_to_cell_x(fx: np.ndarray) -> np.ndarray:
    """x 面値 (nx+1, ny) → セル値 (nx, ny)（東西面の平均）."""
    return 0.5 * (fx[:-1] + fx[1:])


def face_to_cell_y(fy: np.ndarray) -> np.ndarray:
    """y 面値 (nx, ny+1) → セル値 (nx, ny)（南北面の平均）."""
    return 0.5 * (fy[:, :-1] + fy[:, 1:])


def cell_centers(inp: NSBInput) -> tuple[np.ndarray, np.ndarray]:
    """セル中心座標 (x (nx,), y (ny,))."""
    x = (np.arange(inp.nx) + 0.5) * inp.dx
    y = (np.arange(inp.ny) + 0.5) * inp.dy
    return x, y


def speed(res: NSBResult) -> np.ndarray:
    return np.hypot(res.u, res.v)


def mass_balance(res: NSBResult) -> float:
    """m_out / m_in（1 なら質量保存）."""
    return res.mass_out / res.mass_in if res.mass_in != 0.0 else float("nan")


def inlet_cells(inp: NSBInput) -> np.ndarray:
    """inlet 面に接するセルの bool マスク (nx, ny)."""
    from nsb.assembly import BrinkmanDiscretization

    sides = BrinkmanDiscretization(inp.to_flow_input()).sides
    m = np.zeros((inp.nx, inp.ny), dtype=bool)
    m[0, :] |= sides["W"].is_inlet
    m[-1, :] |= sides["E"].is_inlet
    m[:, 0] |= sides["S"].is_inlet
    m[:, -1] |= sides["N"].is_inlet
    return m


def inlet_mean_pressure(inp: NSBInput, res: NSBResult) -> float:
    """inlet に接するセルの平均圧力 [Pa]（outlet が p=0 なら圧力損失に相当）."""
    return float(res.p[inlet_cells(inp)].mean())


def inlet_velocity(inp: NSBInput) -> float:
    """inlet の最大流入速度 [m/s]（質量流入境界は換算後）."""
    from nsb.assembly import BrinkmanDiscretization

    return BrinkmanDiscretization(inp.to_flow_input()).u_scale


def hele_shaw_pressure_drop(inp: NSBInput, path_length: float) -> float:
    """Hele-Shaw 平行平板の理論圧損 Δp = 12 μ_b U L / h² [Pa]（h は流路部の最大厚さ）."""
    h = float(inp.h.max())
    return 12.0 * inp.mu_b * inlet_velocity(inp) * path_length / h**2


def summary(inp: NSBInput, res: NSBResult) -> dict[str, float | bool | int | str]:
    """YAML 保存用の要約."""
    return {
        "converged": bool(res.converged),
        "reason": res.failure_reason,
        "n_iter": int(res.n_iter),
        "n_rejected": int(res.n_rejected),
        "rel_final": float(res.rel_residual),
        "rel_steady_final": float(res.rel_steady_residual),
        "rel_min": float(min(res.residual_history) / res.residual_history[0]),
        "first_step_ratio": float(res.residual_history[1] / res.residual_history[0])
        if len(res.residual_history) > 1
        else float("nan"),
        "mass_ratio": float(mass_balance(res)),
        "p_inlet_mean": float(inlet_mean_pressure(inp, res)),
        "speed_max": float(speed(res).max()),
        "elapsed": float(res.elapsed),
    }


def save_fields(path: Path, inp: NSBInput, res: NSBResult) -> Path:
    """u, v, p, h を npz に保存（experiments/brinkman_uturn/plot_fields.py で描画可能）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, u=res.u, v=res.v, p=res.p, h=inp.h)
    return path
