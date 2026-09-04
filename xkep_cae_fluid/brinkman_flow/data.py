"""2D Brinkman 補正 Navier-Stokes (FVM) データスキーマ.

薄流路を深さ平均した 2 次元場で、Brinkman 貫通項 -(12 mu_b / h^2) u を持つ
非圧縮 Navier-Stokes を定常解として解くソルバーの入出力契約を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ConvectionSchemeType(Enum):
    """対流面値の補間スキーム."""

    FIRST_ORDER_UPWIND = "first_order_upwind"
    SECOND_ORDER_UPWIND = "second_order_upwind"  # Green-Gauss 勾配 + Venkatakrishnan リミター


class JacobianMode(Enum):
    """Newton 系の線形化方式."""

    JFNK = "jfnk"  # 有限差分 J2·v を GMRES、LU(J1) を前処理
    DEFECT_CORRECTION = "defect_correction"  # J1 δ = -R2 を LU で直接求解


class ThicknessModel(Enum):
    """厚さ場のモデル種別."""

    FLAT = "flat"  # 全域 h_channel
    UTURN = "uturn"  # U ターン経路のみ h_channel、残りは h_blocked


@dataclass(frozen=True)
class BrinkmanGeometry:
    """解析領域と inlet/outlet 位置（いずれも左壁 x=0 上）.

    Parameters
    ----------
    lx, ly : float
        領域サイズ [m]
    inlet_y0, inlet_y1 : float
        速度 inlet の y 範囲 [m]（左壁）
    outlet_y0, outlet_y1 : float
        圧力 outlet の y 範囲 [m]（左壁）
    """

    lx: float = 0.7
    ly: float = 0.4
    inlet_y0: float = 0.25
    inlet_y1: float = 0.35
    outlet_y0: float = 0.05
    outlet_y1: float = 0.15


@dataclass(frozen=True)
class ThicknessSpec:
    """厚さ場の生成仕様（UTurnThicknessProcess の入力）.

    Parameters
    ----------
    model : ThicknessModel
        FLAT / UTURN
    h_channel : float
        流路部の厚さ [m]
    h_blocked : float
        閉塞部の厚さ [m]（UTURN のみ使用）
    channel_width : float
        U ターン経路の流路幅 [m]。既定は inlet 高さと同じ 0.1
    turn_x0 : float
        往路と復路をつなぐ折返し区間の開始 x [m]。None なら lx - channel_width
    """

    model: ThicknessModel = ThicknessModel.FLAT
    h_channel: float = 1.0e-3
    h_blocked: float = 1.0e-5
    channel_width: float = 0.1
    turn_x0: float | None = None


@dataclass(frozen=True)
class BrinkmanSolverSettings:
    """Newton–Krylov + 擬似時間 + 陰的緩和の設定.

    Parameters
    ----------
    convection_scheme : ConvectionSchemeType
        残差評価に使う対流スキーム（ヤコビアンは常に 1 次風上）
    jacobian_mode : JacobianMode
        JFNK（GMRES + LU 前処理）か defect correction（LU 直接）
    venkat_k : float
        Venkatakrishnan リミター定数 K（ε² = (K Δx)³）
    newton_tol : float
        相対残差 ||R||/||R0|| の収束判定
    newton_abs_tol : float
        絶対残差の収束判定（どちらか満たせば収束）
    newton_max_iter : int
        Newton 反復上限
    cfl_init, cfl_max, ser_growth : float
        擬似時間 CFL の初期値・上限・SER 成長率（残差比の逆数を成長率上限として乗じる）
    alpha_u : float
        陰的緩和係数（運動量対角を a_P/α に置換）。1.0 で緩和なし
    gmres_tol, gmres_restart, gmres_maxiter : float, int, int
        GMRES 設定
    divergence_ratio : float
        ||R||/||R0|| がこの値を超えたら発散として停止
    line_search : bool
        Armijo 型の簡易ラインサーチ（既定 False: 再現実験のため）
    """

    convection_scheme: ConvectionSchemeType = ConvectionSchemeType.SECOND_ORDER_UPWIND
    jacobian_mode: JacobianMode = JacobianMode.JFNK
    venkat_k: float = 5.0
    newton_tol: float = 1.0e-6
    newton_abs_tol: float = 1.0e-10
    newton_max_iter: int = 60
    cfl_init: float = 5.0
    cfl_max: float = 1.0e6
    ser_growth: float = 2.0
    alpha_u: float = 0.7
    gmres_tol: float = 1.0e-3
    gmres_restart: int = 40
    gmres_maxiter: int = 5
    divergence_ratio: float = 1.0e6
    line_search: bool = False


@dataclass(frozen=True)
class BrinkmanFlowInput:
    """Brinkman 流れソルバー入力.

    Parameters
    ----------
    nx, ny : int
        分割数
    geometry : BrinkmanGeometry
        領域と inlet/outlet 位置
    thickness : np.ndarray
        厚さ場 h(x,y) (nx, ny) [m]
    rho, mu : float
        密度 [kg/m³]・粘度 [Pa·s]
    mu_brinkman : float
        Brinkman 粘度 [Pa·s]
    brinkman_factor : float
        貫通係数 = brinkman_factor · mu_brinkman / h²（Hele-Shaw なら 12）
    u_inlet : float
        inlet 流速 [m/s]（x 正方向）
    settings : BrinkmanSolverSettings
        ソルバー設定
    u0, v0, p0 : np.ndarray | None
        初期場（None ならゼロ）
    """

    nx: int
    ny: int
    thickness: np.ndarray
    geometry: BrinkmanGeometry = field(default_factory=BrinkmanGeometry)
    rho: float = 1000.0
    mu: float = 1.0e-3
    mu_brinkman: float = 1.0e-3
    brinkman_factor: float = 12.0
    u_inlet: float = 0.1
    settings: BrinkmanSolverSettings = field(default_factory=BrinkmanSolverSettings)
    u0: np.ndarray | None = None
    v0: np.ndarray | None = None
    p0: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.thickness.shape != (self.nx, self.ny):
            raise ValueError(
                f"thickness の形状 {self.thickness.shape} が (nx, ny)=({self.nx}, {self.ny}) と一致しません"
            )
        if np.any(self.thickness <= 0.0):
            raise ValueError("thickness は正である必要があります")

    @property
    def dx(self) -> float:
        return self.geometry.lx / self.nx

    @property
    def dy(self) -> float:
        return self.geometry.ly / self.ny


@dataclass(frozen=True)
class BrinkmanFlowResult:
    """Brinkman 流れソルバー結果.

    Parameters
    ----------
    u, v, p : np.ndarray
        速度・圧力場 (nx, ny)
    converged : bool
        Newton 収束フラグ
    failure_reason : str
        未収束時の理由（"nan", "diverged", "max_iter", "gmres_breakdown" 等）。収束時は ""
    n_newton : int
        実行した Newton 反復数
    residual_history : tuple[float, ...]
        各反復の残差 2 ノルム（反復 0 = 初期残差）
    residual_components : tuple[tuple[float, float, float], ...]
        各反復の (u, v, p) 残差ノルム
    cfl_history : tuple[float, ...]
        各反復の擬似時間 CFL
    gmres_iterations : tuple[int, ...]
        各反復の GMRES 反復数（defect correction では 0）
    mass_in, mass_out : float
        inlet / outlet の質量流量 [kg/s]（面速度ベース）
    elapsed_seconds : float
        計算時間
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    converged: bool
    failure_reason: str
    n_newton: int
    residual_history: tuple[float, ...]
    residual_components: tuple[tuple[float, float, float], ...]
    cfl_history: tuple[float, ...]
    gmres_iterations: tuple[int, ...]
    mass_in: float
    mass_out: float
    elapsed_seconds: float
