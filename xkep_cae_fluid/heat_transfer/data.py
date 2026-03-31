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
    ROBIN = "robin"  # 対流熱伝達 h(T_inf - T)


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
        ROBIN: 無視される（h_conv, T_inf を使用）
    h_conv : float
        対流熱伝達係数 [W/(m²·K)]（ROBIN 用）
    T_inf : float
        外部流体温度 [K]（ROBIN 用）
    """

    condition: BoundaryCondition
    value: float = 0.0
    h_conv: float = 0.0
    T_inf: float = 0.0


@dataclass(frozen=True)
class HeatTransferInput:
    """伝熱解析ソルバー入力.

    等間隔直交格子上の3次元非定常伝熱解析の入力データ。
    dx_array/dy_array/dz_array が指定された場合、不等間隔格子に対応する。

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
    dx_array : np.ndarray | None
        x方向の各セル幅 (nx,)。不等間隔格子用。None の場合は等間隔。
    dy_array : np.ndarray | None
        y方向の各セル幅 (ny,)。不等間隔格子用。None の場合は等間隔。
    dz_array : np.ndarray | None
        z方向の各セル幅 (nz,)。不等間隔格子用。None の場合は等間隔。
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
    dx_array: np.ndarray | None = None
    dy_array: np.ndarray | None = None
    dz_array: np.ndarray | None = None

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

    @property
    def is_nonuniform(self) -> bool:
        """不等間隔格子かどうか."""
        return self.dx_array is not None

    @classmethod
    def from_mesh(
        cls,
        mesh_result: object,
        k: np.ndarray,
        C: np.ndarray,
        q: np.ndarray,
        T0: np.ndarray,
        **kwargs: object,
    ) -> HeatTransferInput:
        """StructuredMeshResult からインスタンスを生成.

        Parameters
        ----------
        mesh_result : StructuredMeshResult
            メッシュ生成結果（dx, dy, dz 配列を持つ）
        k, C, q, T0 : np.ndarray
            物性値と初期条件（(nx, ny, nz) 配列）
        **kwargs
            その他のパラメータ（bc_xm, dt, tol 等）
        """
        dx_arr = mesh_result.dx  # type: ignore[attr-defined]
        dy_arr = mesh_result.dy  # type: ignore[attr-defined]
        dz_arr = mesh_result.dz  # type: ignore[attr-defined]
        Lx = float(np.sum(dx_arr))
        Ly = float(np.sum(dy_arr))
        Lz = float(np.sum(dz_arr))
        return cls(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            k=k,
            C=C,
            q=q,
            T0=T0,
            dx_array=dx_arr,
            dy_array=dy_arr,
            dz_array=dz_arr,
            **kwargs,  # type: ignore[arg-type]
        )


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
