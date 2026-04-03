"""自然対流解析用データスキーマ.

3次元等間隔直交格子上の自然対流解析（SIMPLE法 + Boussinesq近似）に
必要な入出力データ契約を定義する。固体-流体練成伝熱に対応。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FluidBoundaryCondition(Enum):
    """流体境界条件の種別."""

    NO_SLIP = "no_slip"  # 壁面（速度=0、圧力勾配=0）
    SLIP = "slip"  # すべり壁（法線速度=0、接線応力=0）
    INLET_VELOCITY = "inlet_velocity"  # 速度流入
    OUTLET_PRESSURE = "outlet_pressure"  # 圧力流出（ゼロ勾配）
    OUTLET_CONVECTIVE = "outlet_convective"  # 対流流出（非反射）
    SYMMETRY = "symmetry"  # 対称面


class ThermalBoundaryCondition(Enum):
    """温度境界条件の種別."""

    DIRICHLET = "dirichlet"  # 温度固定
    NEUMANN = "neumann"  # 熱流束指定
    ADIABATIC = "adiabatic"  # 断熱


@dataclass(frozen=True)
class FluidBoundarySpec:
    """1面の流体境界条件仕様.

    Parameters
    ----------
    condition : FluidBoundaryCondition
        流体境界条件種別
    velocity : tuple[float, float, float]
        INLET_VELOCITY: 流入速度 (u, v, w) [m/s]
    pressure : float
        OUTLET_PRESSURE: 出口圧力 [Pa]
    thermal : ThermalBoundaryCondition
        温度境界条件種別
    temperature : float
        DIRICHLET: 壁面温度 [K]
    heat_flux : float
        NEUMANN: 熱流束 [W/m²]（正=流入）
    """

    condition: FluidBoundaryCondition = FluidBoundaryCondition.NO_SLIP
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pressure: float = 0.0
    thermal: ThermalBoundaryCondition = ThermalBoundaryCondition.ADIABATIC
    temperature: float = 300.0
    heat_flux: float = 0.0


@dataclass(frozen=True)
class NaturalConvectionInput:
    """自然対流解析ソルバー入力.

    等間隔直交格子上の3次元非圧縮性自然対流解析の入力データ。
    SIMPLE法 + Boussinesq近似を使用する。

    Parameters
    ----------
    Lx, Ly, Lz : float
        領域サイズ [m]
    nx, ny, nz : int
        各方向のセル数
    rho : float
        基準密度 [kg/m³]
    mu : float
        動粘性係数 [Pa·s]
    Cp : float
        比熱 [J/(kg·K)]
    k_fluid : float
        流体の熱伝導率 [W/(m·K)]
    beta : float
        体膨張係数 [1/K]
    T_ref : float
        Boussinesq近似の基準温度 [K]
    gravity : tuple[float, float, float]
        重力加速度ベクトル [m/s²]
    solid_mask : np.ndarray | None
        固体領域マスク (nx, ny, nz)。True=固体、False=流体。
        None の場合は全流体。
    k_solid : np.ndarray | None
        固体領域の熱伝導率 (nx, ny, nz) [W/(m·K)]。
        solid_mask=None の場合は無視。
    q_vol : np.ndarray | None
        体積熱生成 (nx, ny, nz) [W/m³]。
        None の場合は熱生成なし。
    T0 : np.ndarray
        初期温度場 (nx, ny, nz) [K]
    bc_xm, bc_xp, bc_ym, bc_yp, bc_zm, bc_zp : FluidBoundarySpec
        各面の境界条件
    dt : float
        時間刻み [s]（0 = 定常解析）
    t_end : float
        終了時刻 [s]（定常解析時は無視）
    max_simple_iter : int
        SIMPLEの最大外部反復数
    max_inner_iter : int
        各方程式の最大内部反復数
    tol_simple : float
        SIMPLE外部反復の収束判定閾値
    tol_inner : float
        内部反復の収束判定閾値
    alpha_u : float
        速度の緩和係数
    alpha_p : float
        圧力の緩和係数
    alpha_T : float
        温度の緩和係数
    output_interval : int
        結果出力間隔
    coupling_method : str
        圧力-速度連成手法。"simple", "simplec", "piso" のいずれか。
        SIMPLEC は圧力補正のd係数を行列行和で計算し alpha_p=1.0 が使用可能。
        PISO は複数回の圧力補正で質量保存を大幅改善（非定常向け）。
    n_piso_correctors : int
        PISO の圧力補正回数（デフォルト2）。coupling_method="piso" 時のみ有効。
    convection_scheme : str
        対流スキーム。"upwind"（1次風上）, "van_leer", "superbee" のいずれか。
        TVDスキームは遅延補正法で実装（行列は1次風上、補正はRHSソース）。
    time_scheme : str
        時間積分スキーム。"euler"（1次後退Euler）または "bdf2"（2次BDF）。
        BDF2 は2次精度で、最初のステップは自動的にEulerで実行される。
    pressure_solver : str
        圧力方程式の線形ソルバー。"bicgstab"（BiCGSTAB+ILU）または "amg"（PyAMG前処理+CG）。
        圧力補正方程式は対称正定値ラプラシアンなので、AMG+CGが最適。
    adaptive_relaxation : bool
        適応的緩和係数の有効化。残差の減少率に応じて alpha_u, alpha_p を
        自動調整し、収束を加速する。
    max_pressure_iter : int
        圧力方程式の最大内部反復数。圧力補正は収束が遅いため、
        運動量方程式より多めの反復数が有効。0の場合はmax_inner_iterを使用。
    """

    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    rho: float
    mu: float
    Cp: float
    k_fluid: float
    beta: float
    T_ref: float
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    solid_mask: np.ndarray | None = None
    k_solid: np.ndarray | None = None
    q_vol: np.ndarray | None = None
    T0: np.ndarray | None = None
    bc_xm: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    bc_xp: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    bc_ym: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    bc_yp: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    bc_zm: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    bc_zp: FluidBoundarySpec = field(default_factory=FluidBoundarySpec)
    dt: float = 0.0
    t_end: float = 0.0
    max_simple_iter: int = 500
    max_inner_iter: int = 50
    tol_simple: float = 1e-5
    tol_inner: float = 1e-6
    alpha_u: float = 0.7
    alpha_p: float = 0.3
    alpha_T: float = 0.9
    output_interval: int = 1
    coupling_method: str = "simple"
    n_piso_correctors: int = 2
    convection_scheme: str = "upwind"
    time_scheme: str = "euler"
    pressure_solver: str = "bicgstab"
    adaptive_relaxation: bool = False
    max_pressure_iter: int = 0

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
    def nu(self) -> float:
        """動粘度 [m²/s]."""
        return self.mu / self.rho

    @property
    def alpha_thermal(self) -> float:
        """温度拡散率 [m²/s]."""
        return self.k_fluid / (self.rho * self.Cp)

    @property
    def Pr(self) -> float:
        """プラントル数."""
        return self.nu / self.alpha_thermal


@dataclass(frozen=True)
class NaturalConvectionResult:
    """自然対流解析ソルバー出力.

    Parameters
    ----------
    u, v, w : np.ndarray
        速度場 (nx, ny, nz) [m/s]
    p : np.ndarray
        圧力場 (nx, ny, nz) [Pa]
    T : np.ndarray
        温度場 (nx, ny, nz) [K]
    converged : bool
        SIMPLEが収束したか
    n_outer_iterations : int
        SIMPLE外部反復回数
    residual_history : dict[str, list[float]]
        各変数の残差履歴 {u, v, w, p, T, mass}
    elapsed_seconds : float
        計算時間 [s]
    n_timesteps : int
        実行タイムステップ数（非定常時）
    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    T: np.ndarray
    converged: bool
    n_outer_iterations: int = 0
    residual_history: dict[str, list[float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    n_timesteps: int = 0
