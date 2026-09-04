"""型宣言: 境界条件・入力・設定・結果."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from xkep_cae_fluid.brinkman_flow.data import (
    BrinkmanFlowInput,
    BrinkmanGeometry,
    ConvectionSchemeType,
)


class FaceType(Enum):
    """境界面の種別."""

    WALL = "wall"  # no-slip
    VELOCITY_INLET = "velocity_inlet"  # u = u_in（x 正方向）, v = 0
    PRESSURE_OUTLET = "pressure_outlet"  # p = 0, 速度ゼロ勾配


@dataclass(frozen=True)
class BC:
    """境界条件.

    west に (ny,) の FaceType 配列を与える。north / south / east は WALL 固定
    （共有している離散化が左壁 inlet/outlet のみ対応のため）。
    inlet / outlet はそれぞれ連続した 1 区間である必要がある。

    Parameters
    ----------
    west : np.ndarray
        左壁の各セル面の FaceType (ny,)
    u_inlet : float
        inlet 流速 [m/s]
    """

    west: np.ndarray
    u_inlet: float

    def _span(self, kind: FaceType, ly: float) -> tuple[float, float]:
        idx = np.flatnonzero(np.array([f is kind for f in self.west]))
        if idx.size == 0:
            raise ValueError(f"west に {kind.value} がありません")
        if np.any(np.diff(idx) != 1):
            raise ValueError(f"{kind.value} は連続した 1 区間である必要があります")
        dy = ly / self.west.size
        return float(idx[0] * dy), float((idx[-1] + 1) * dy)

    def to_geometry(self, lx: float, ly: float) -> BrinkmanGeometry:
        """FaceType 配列から共有離散化用の inlet/outlet 区間へ変換."""
        in0, in1 = self._span(FaceType.VELOCITY_INLET, ly)
        out0, out1 = self._span(FaceType.PRESSURE_OUTLET, ly)
        return BrinkmanGeometry(
            lx=lx, ly=ly, inlet_y0=in0, inlet_y1=in1, outlet_y0=out0, outlet_y1=out1
        )


@dataclass(frozen=True)
class NSBSettings:
    """Newton + 擬似時間の制御則（既定値は手元構成に合わせた「踏んでいる線」込み）.

    Parameters
    ----------
    convection : str
        残差の対流スキーム "sou"（2 次風上 + Venkatakrishnan）/ "fou"（1 次風上）。
        ヤコビアンは常に 1 次風上
    venkat_k : float
        Venkatakrishnan 定数 K
    linear_solver : str
        "jfnk"（有限差分 J v を GMRES、LU(J1) 前処理）/ "lu"（J1 δ = -R を LU 直接）
    cfl_init, cfl_max, ser_growth : float
        擬似時間 CFL の初期値・上限・SER 成長率上限
    local_dtau : bool
        True: セル局所 Δτ、False: 局所 Δτ の全セル最小値を一律に使う
    velocity_floor : float
        Δτ の速度スケール下限 [m/s]（絶対値）。0 なら下限なし（静止セルで Δτ→∞）
    pseudo_time_in_residual : bool
        True: 残差にも ρV(u - u_prev)/Δτ を加える（dual-time 型）。収束判定・SER も
        その残差で行う。False: 対角補強のみ（残差は Δτ 非依存）
    sub_iters : int
        1 擬似時間ステップあたりの Newton 反復数（u_prev を凍結）。1 で通常の擬似時間 Newton
    rc_with_pseudo_time : bool
        Rhie–Chow 係数を d_f = V/(a_P + ρV/Δτ) にする
    alpha_u : float
        陰的緩和（運動量対角を a_P/α_u）。1.0 で無し
    newton_tol, newton_max_iter : float, int
        相対残差の収束判定と反復上限（擬似時間ステップ数 × sub_iters が上限）
    gmres_tol, gmres_restart, gmres_maxiter : float, int, int
        GMRES 設定
    divergence_ratio : float
        ||R||/||R0|| がこれを超えたら発散停止
    init_field : str
        "zero": 静止場から開始 / "stokes": 対流を無視した Stokes–Brinkman 解
        （ゼロ場からの擬似時間なし 1 次風上 Newton 1 ステップ）を初期場にし、
        その残差を収束判定の基準 R0 にする
    reject_growth : float
        0 より大なら、更新後の残差が reject_growth × 更新前残差を超えたステップを棄却し、
        CFL を半分にして再試行する（backtracking on CFL）。0 で無効
    max_rejects : int
        1 擬似時間ステップあたりの棄却回数上限（超えたら受け入れる）
    cfl_min : float
        棄却で CFL を下げる下限。Δτ→0 では圧力が連続式を満たすために発散するので必要
    """

    convection: str = "sou"
    venkat_k: float = 5.0
    linear_solver: str = "jfnk"
    cfl_init: float = 0.5
    cfl_max: float = 1.0e6
    ser_growth: float = 2.0
    local_dtau: bool = True
    velocity_floor: float = 0.0
    pseudo_time_in_residual: bool = True
    sub_iters: int = 1
    rc_with_pseudo_time: bool = False
    alpha_u: float = 0.7
    newton_tol: float = 1.0e-6
    newton_max_iter: int = 80
    gmres_tol: float = 1.0e-3
    gmres_restart: int = 40
    gmres_maxiter: int = 5
    divergence_ratio: float = 1.0e6
    init_field: str = "zero"
    reject_growth: float = 0.0
    max_rejects: int = 6
    cfl_min: float = 1.0e-2

    @property
    def scheme(self) -> ConvectionSchemeType:
        return {
            "sou": ConvectionSchemeType.SECOND_ORDER_UPWIND,
            "fou": ConvectionSchemeType.FIRST_ORDER_UPWIND,
        }[self.convection]


@dataclass(frozen=True)
class NSBInput:
    """ソルバー入力.

    Parameters
    ----------
    nx, ny : int
        分割数
    lx, ly : float
        領域サイズ [m]
    h : np.ndarray
        厚さ場 (nx, ny) [m]
    bc : BC
        境界条件
    rho, mu, mu_b : float
        密度・粘度・Brinkman 粘度
    settings : NSBSettings
        ソルバー設定
    u0, v0, p0 : np.ndarray | None
        初期場（None ならゼロ）
    """

    nx: int
    ny: int
    lx: float
    ly: float
    h: np.ndarray
    bc: BC
    rho: float = 1000.0
    mu: float = 1.0e-3
    mu_b: float = 1.0e-3
    settings: NSBSettings = field(default_factory=NSBSettings)
    u0: np.ndarray | None = None
    v0: np.ndarray | None = None
    p0: np.ndarray | None = None

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    def to_flow_input(self) -> BrinkmanFlowInput:
        """共有離散化（BrinkmanDiscretization）用の入力へ変換."""
        return BrinkmanFlowInput(
            nx=self.nx,
            ny=self.ny,
            thickness=self.h,
            geometry=self.bc.to_geometry(self.lx, self.ly),
            rho=self.rho,
            mu=self.mu,
            mu_brinkman=self.mu_b,
            brinkman_factor=12.0,
            u_inlet=self.bc.u_inlet,
        )


@dataclass(frozen=True)
class NSBResult:
    """ソルバー結果.

    Parameters
    ----------
    u, v, p : np.ndarray
        セル中心の速度・圧力 (nx, ny)
    converged : bool
        収束フラグ
    failure_reason : str
        未収束理由（"" なら収束）
    n_iter : int
        実行した Newton 反復数（sub_iters 込み）
    residual_history : tuple[float, ...]
        反復ごとの残差ノルム（pseudo_time_in_residual=True なら擬似時間項込み）
    steady_residual_history : tuple[float, ...]
        反復ごとの Δτ 非依存の定常残差ノルム
    cfl_history : tuple[float, ...]
        反復ごとの CFL
    mass_in, mass_out : float
        inlet / outlet 質量流量 [kg/s]
    elapsed : float
        計算時間 [s]
    n_rejected : int
        棄却した更新の回数（線形解の追加コスト）
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    converged: bool
    failure_reason: str
    n_iter: int
    residual_history: tuple[float, ...]
    steady_residual_history: tuple[float, ...]
    cfl_history: tuple[float, ...]
    mass_in: float
    mass_out: float
    elapsed: float
    n_rejected: int = 0

    @property
    def rel_residual(self) -> float:
        return self.residual_history[-1] / self.residual_history[0]

    @property
    def rel_steady_residual(self) -> float:
        return self.steady_residual_history[-1] / self.steady_residual_history[0]
