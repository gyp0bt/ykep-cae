"""伝熱解析用データスキーマ.

3次元等間隔直交格子（ボクセル）上の伝熱解析に必要な
入出力データ契約を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BoundaryCondition(Enum):
    """境界条件の種別."""

    DIRICHLET = "dirichlet"  # 温度固定
    NEUMANN = "neumann"  # 熱流束指定
    ADIABATIC = "adiabatic"  # 断熱


@dataclass(frozen=True)
class BoundarySpec:
    """1面の境界条件仕様.

    Parameters
    ----------
    condition : BoundaryCondition
        境界条件種別
    value : float
        DIRICHLET: 壁面温度 [K]
        NEUMANN: 熱流束 [W/m²] (正=流入)
        ADIABATIC: 無視される
    """

    condition: BoundaryCondition
    value: float = 0.0


@dataclass(frozen=True)
class HeatTransferInput:
    """伝熱解析ソルバー入力.

    等間隔直交格子上の3次元非定常伝熱解析の入力データ。

    Parameters
    ----------
    Lx, Ly, Lz : float
        領域サイズ [m]
    k : np.ndarray
        熱伝導率 (nx, ny, nz) [W/(m·K)]
    C : np.ndarray
        体積熱容量 ρCp (nx, ny, nz) [J/(m³·K)]
    q : np.ndarray
        体積発熱量 (nx, ny, nz) [W/m³]
    T0 : np.ndarray
        初期温度 (nx, ny, nz) [K]
    bc_xm, bc_xp, bc_ym, bc_yp, bc_zm, bc_zp : BoundarySpec
        各面の境界条件 (x-, x+, y-, y+, z-, z+)
    dt : float
        時間刻み [s] (0 = 定常解析)
    t_end : float
        終了時刻 [s]
    max_iter : int
        ガウスザイデル反復の最大反復数
    tol : float
        収束判定閾値（残差のL2ノルム）
    output_interval : int
        結果出力間隔（タイムステップ数）
    """

    Lx: float
    Ly: float
    Lz: float
    k: np.ndarray
    C: np.ndarray
    q: np.ndarray
    T0: np.ndarray
    bc_xm: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    bc_xp: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    bc_ym: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    bc_yp: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    bc_zm: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    bc_zp: BoundarySpec = field(default_factory=lambda: BoundarySpec(BoundaryCondition.ADIABATIC))
    dt: float = 0.0
    t_end: float = 0.0
    max_iter: int = 10000
    tol: float = 1e-6
    output_interval: int = 1

    @property
    def nx(self) -> int:
        return self.k.shape[0]

    @property
    def ny(self) -> int:
        return self.k.shape[1]

    @property
    def nz(self) -> int:
        return self.k.shape[2]

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def dz(self) -> float:
        return self.Lz / self.nz

    @property
    def is_transient(self) -> bool:
        return self.dt > 0.0


@dataclass(frozen=True)
class HeatTransferResult:
    """伝熱解析ソルバー出力.

    Parameters
    ----------
    T : np.ndarray
        最終温度場 (nx, ny, nz) [K]
    converged : bool
        最終タイムステップが収束したか
    n_timesteps : int
        実行タイムステップ数
    iteration_counts : tuple[int, ...]
        各タイムステップの反復回数
    residual_history : tuple[tuple[float, ...], ...]
        各タイムステップの残差履歴
    time_history : tuple[float, ...]
        出力時刻
    T_history : tuple[np.ndarray, ...]
        出力温度場のスナップショット
    elapsed_seconds : float
        計算時間 [s]
    """

    T: np.ndarray
    converged: bool
    n_timesteps: int = 0
    iteration_counts: tuple[int, ...] = ()
    residual_history: tuple[tuple[float, ...], ...] = ()
    time_history: tuple[float, ...] = ()
    T_history: tuple[np.ndarray, ...] = ()
    elapsed_seconds: float = 0.0
