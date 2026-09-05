"""Darcy 流れソルバーの入出力契約."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from xkep_cae_fluid.core.data import MeshData


class DarcyBCKind(Enum):
    """Darcy 流れの境界条件種別."""

    PRESSURE = "pressure"  # 圧力固定 [Pa]
    VELOCITY = "velocity"  # 法線流入速度指定 [m/s]（正 = 領域へ流入）
    WALL = "wall"  # 不透過（法線速度ゼロ）


@dataclass(frozen=True)
class DarcyPatchBC:
    """1 パッチの Darcy 境界条件.

    Parameters
    ----------
    kind : DarcyBCKind
    pressure : float | np.ndarray
        PRESSURE の値（パッチ面ごとの配列も可）
    velocity : float | np.ndarray
        VELOCITY の法線流入速度 u_n [m/s]（正 = 流入、負 = 流出）
    """

    kind: DarcyBCKind = DarcyBCKind.WALL
    pressure: float | np.ndarray = 0.0
    velocity: float | np.ndarray = 0.0

    @staticmethod
    def pressure_bc(value: float | np.ndarray) -> DarcyPatchBC:
        return DarcyPatchBC(DarcyBCKind.PRESSURE, pressure=value)

    @staticmethod
    def velocity_bc(u_n: float | np.ndarray) -> DarcyPatchBC:
        return DarcyPatchBC(DarcyBCKind.VELOCITY, velocity=u_n)

    @staticmethod
    def wall() -> DarcyPatchBC:
        return DarcyPatchBC(DarcyBCKind.WALL)


@dataclass(frozen=True)
class DarcyFlowInput:
    """Darcy 流れの入力.

    支配方程式: ∇·u = S, u = −(K/μ) ∇p（K: 透過率 [m²]、μ: 粘度 [Pa·s]）

    Parameters
    ----------
    mesh : MeshData
        面情報と ``boundary_patches`` を持つメッシュ
    permeability : float | np.ndarray
        透過率 K [m²]（スカラーかセル配列）
    viscosity : float
        粘度 μ [Pa·s]
    density : float
        密度 ρ [kg/m³]（質量流束の換算用）
    bcs : Mapping[str, DarcyPatchBC]
        パッチ名 → 境界条件。未指定のパッチは不透過壁
    source : np.ndarray | None
        体積流量ソース S [1/s]（セル配列）。None ならゼロ
    p0 : np.ndarray | None
        圧力の初期値（反復解法の初期推定）
    linear_solver : str
        ``direct`` / ``bicgstab`` / ``amg``
    tol, max_iter : 反復解法の設定
    max_nonorthogonal_iter : int
        非直交メッシュでの遅延補正の最大反復回数（直交メッシュでは 1 回で終わる）
    """

    mesh: MeshData
    permeability: float | np.ndarray
    viscosity: float
    density: float = 1000.0
    bcs: Mapping[str, DarcyPatchBC] = field(default_factory=dict)
    source: np.ndarray | None = None
    p0: np.ndarray | None = None
    linear_solver: str = "direct"
    tol: float = 1e-10
    max_iter: int = 1000
    max_nonorthogonal_iter: int = 20


@dataclass(frozen=True)
class DarcyFlowResult:
    """Darcy 流れの出力.

    Parameters
    ----------
    p : np.ndarray
        圧力 (n_cells,) [Pa]
    velocity : np.ndarray
        Darcy 速度（見かけ速度）(n_cells, 3) [m/s]
    face_flux : np.ndarray
        面の体積流量 (n_faces,) [m³/s]（内部面は owner → neighbour、境界面は外向きが正）
    mass_residual : np.ndarray
        各セルの体積流量の不整合 Σ_f q_f − S V (n_cells,) [m³/s]
    converged : bool
    residual : float
        線形系の相対残差
    elapsed_seconds : float
    inflow, outflow : float
        境界からの流入・流出体積流量 [m³/s]（流入は正、流出は正で報告）
    """

    p: np.ndarray
    velocity: np.ndarray
    face_flux: np.ndarray
    mass_residual: np.ndarray
    converged: bool
    residual: float
    elapsed_seconds: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0
    n_nonorthogonal_iter: int = 1  # 非直交補正の反復回数（直交メッシュは 1）
