"""汎用スカラー輸送用データスキーマ.

対流-拡散-ソース方程式をもつ任意のスカラー（CO2, O2, トレーサー等）を
3次元等間隔直交格子上で輸送するソルバーの入出力契約を定義する。

速度場は外部ソルバー（NaturalConvectionProcess 等）から与えられることを前提とし、
水槽設計 CAE（Phase 6）での生体反応・ガス交換・気泡プルームとの接続点を
`ScalarFieldSpec.source`（体積ソース項）で受け取る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ScalarBoundaryCondition(Enum):
    """スカラー輸送境界条件の種別."""

    DIRICHLET = "dirichlet"  # φ 固定
    NEUMANN = "neumann"  # Γ∂φ/∂n 指定（正=流入）
    ADIABATIC = "adiabatic"  # ゼロ勾配
    ROBIN = "robin"  # J = h_mass·(φ_inf - φ_surface)


@dataclass(frozen=True)
class ScalarBoundarySpec:
    """1面のスカラー境界条件仕様.

    Parameters
    ----------
    condition : ScalarBoundaryCondition
        境界条件種別
    value : float
        DIRICHLET: 面上の φ 値
    flux : float
        NEUMANN: Γ∂φ/∂n [単位·kg/(m²·s)]（正=流入）
    h_mass : float
        ROBIN: 物質伝達係数 [m/s]
    phi_inf : float
        ROBIN: 外部側 φ 値（例: ヘンリー則平衡濃度）
    """

    condition: ScalarBoundaryCondition = ScalarBoundaryCondition.ADIABATIC
    value: float = 0.0
    flux: float = 0.0
    h_mass: float = 0.0
    phi_inf: float = 0.0


@dataclass(frozen=True)
class ScalarFieldSpec:
    """スカラー場の物性・初期値・ソース項.

    Parameters
    ----------
    name : str
        スカラー名（"CO2", "O2", "tracer" 等）
    diffusivity : float
        拡散係数 Γ [kg/(m·s)] または等価な単位
    phi0 : np.ndarray
        初期スカラー場 (nx, ny, nz)
    source : np.ndarray | None
        体積ソース項 S_φ (nx, ny, nz) [単位/(m³·s)]。
        None の場合はソースなし。
    """

    name: str
    diffusivity: float
    phi0: np.ndarray
    source: np.ndarray | None = None


@dataclass(frozen=True)
class ScalarTransportInput:
    """スカラー輸送ソルバー入力.

    3次元等間隔直交格子上で対流-拡散-ソースを解く。速度場は外部から与える。

    Parameters
    ----------
    Lx, Ly, Lz : float
        領域サイズ [m]
    nx, ny, nz : int
        各方向のセル数
    rho : float
        密度 [kg/m³]
    u, v, w : np.ndarray
        外部速度場 (nx, ny, nz) [m/s]
    field : ScalarFieldSpec
        輸送されるスカラー仕様
    solid_mask : np.ndarray | None
        固体マスク (nx, ny, nz)。True=固体、False=流体。
        固体セルは対流が無効化される（拡散のみ）。
    bc_xm, bc_xp, bc_ym, bc_yp, bc_zm, bc_zp : ScalarBoundarySpec
        各面の境界条件
    dt : float
        時間刻み [s]（0 = 定常解析）
    t_end : float
        終了時刻 [s]（定常時は無視）
    max_iter : int
        内部反復の最大回数（BiCGSTAB のイテレーション上限）
    tol : float
        内部反復の収束判定閾値
    """

    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    rho: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    field: ScalarFieldSpec
    solid_mask: np.ndarray | None = None
    bc_xm: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_xp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_ym: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_yp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_zm: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_zp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    dt: float = 0.0
    t_end: float = 0.0
    max_iter: int = 200
    tol: float = 1e-8

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
class ExtraScalarSpec:
    """NaturalConvection 統合用の追加スカラー仕様.

    `NaturalConvectionInput.extra_scalars` に複数件渡すことで、
    SIMPLE 外部反復内で温度と同じ Rhie-Chow 面速度を用いて輸送される。

    Parameters
    ----------
    field : ScalarFieldSpec
        スカラー場仕様（名前・拡散係数・初期値・ソース項）
    bc_xm, bc_xp, bc_ym, bc_yp, bc_zm, bc_zp : ScalarBoundarySpec
        各面の境界条件（Dirichlet / Neumann / Adiabatic / Robin）
    alpha : float
        スカラー更新の緩和係数（0 < alpha ≤ 1、デフォルト 1.0 = 更新 100%）
    """

    field: ScalarFieldSpec
    bc_xm: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_xp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_ym: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_yp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_zm: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    bc_zp: ScalarBoundarySpec = field(default_factory=ScalarBoundarySpec)
    alpha: float = 1.0


@dataclass(frozen=True)
class ScalarTransportResult:
    """スカラー輸送ソルバー出力.

    Parameters
    ----------
    phi : np.ndarray
        輸送後スカラー場 (nx, ny, nz)
    converged : bool
        内部反復が収束したか（全タイムステップで成功した場合 True）
    n_timesteps : int
        実行タイムステップ数（非定常時。定常時は 0）
    residual_history : tuple[float, ...]
        最終ステップの内部反復残差履歴（デバッグ用）
    elapsed_seconds : float
        計算時間 [s]
    """

    phi: np.ndarray
    converged: bool
    n_timesteps: int = 0
    residual_history: tuple[float, ...] = ()
    elapsed_seconds: float = 0.0
