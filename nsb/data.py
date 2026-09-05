"""2D Brinkman 補正 Navier-Stokes (FVM) データスキーマ.

【スナップショット】`xkep_cae_fluid/brinkman_flow/data.py` のコミット 1647839 時点の複製（import 行のみ書き換え）。
2026-09-05 に本体側と切り離した。本体側は面ベース FVM 層へ移行するため、以後は同期しない。

薄流路を深さ平均した 2 次元場で、Brinkman 貫通項 -(12 mu_b / h^2) u を持つ
非圧縮 Navier-Stokes を定常解として解くソルバーの入出力契約を定義する。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

MaskFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""境界面中心座標 (x, y)（同形状の配列）を受け取り bool 配列を返す座標マスク関数."""

WeightFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""セル中心座標 (x, y) を受け取り [0, 1] の重み配列を返す滑らかな窓関数（領域内パッチ用）."""


class ConvectionSchemeType(Enum):
    """対流面値の補間スキーム."""

    FIRST_ORDER_UPWIND = "first_order_upwind"
    SECOND_ORDER_UPWIND = "second_order_upwind"  # Green-Gauss 勾配 + Venkatakrishnan リミター


class JacobianMode(Enum):
    """Newton 系の線形化方式."""

    JFNK = "jfnk"  # 有限差分 J2·v を GMRES、LU(J1) を前処理
    DEFECT_CORRECTION = "defect_correction"  # J1 δ = -R2 を LU で直接求解


class PseudoTimeMode(Enum):
    """擬似時間増分の取り方."""

    LOCAL = "local"  # セルごとに Δτ = CFL·Δx / max(|u|+|v|, r·U_in)
    GLOBAL = "global"  # 全セル一律に Δτ = min_cells(局所 Δτ)


class BoundaryKind(Enum):
    """境界面の種別."""

    WALL = "wall"  # no-slip
    VELOCITY_INLET = "velocity_inlet"  # 法線方向に一様流入速度
    MASS_FLOW_INLET = "mass_flow_inlet"  # 質量流量指定（厚さ込み、面の h と長さで一様速度に換算）
    PRESSURE_OUTLET = "pressure_outlet"  # 圧力指定、速度ゼロ勾配
    # --- 領域内（紙面垂直方向のマニホールド）: マスクはセル中心で評価 ---
    INTERIOR_MASS_SOURCE = "interior_mass_source"  # 質量流量指定の流入（面内運動量ゼロで注入）
    INTERIOR_MASS_SINK = "interior_mass_sink"  # 質量流量指定の流出（局所運動量を持ち出す）
    INTERIOR_PRESSURE_SINK = "interior_pressure_sink"  # 圧力指定マニホールド: q = C (p - p_out)


INTERIOR_KINDS = frozenset(
    {
        BoundaryKind.INTERIOR_MASS_SOURCE,
        BoundaryKind.INTERIOR_MASS_SINK,
        BoundaryKind.INTERIOR_PRESSURE_SINK,
    }
)


@dataclass(frozen=True)
class BoundaryPatch:
    """座標マスクで指定する境界パッチ.

    境界種別（WALL / *_INLET / PRESSURE_OUTLET）は領域の 4 辺（x=0, x=lx, y=0, y=ly）上の
    境界面中心に mask(x, y) を評価し、True の面に kind を割り当てる。
    領域内種別（INTERIOR_*）は**セル中心**に mask を評価し、True のセルに紙面垂直方向の
    マニホールド（面内速度ゼロで注入 / 局所速度で吸出）を割り当てる。
    境界パッチが重なった場合は後のものが優先、領域内パッチは重ね合わせ（加算）。どのパッチにも属さない面は WALL。

    Parameters
    ----------
    kind : BoundaryKind
        境界種別
    mask : MaskFn
        (x, y) -> bool 配列。例: ``lambda x, y: (x < 1e-9) & (y > 0.25) & (y < 0.35)``
    velocity : float
        VELOCITY_INLET の流入速度 [m/s]（内向き法線方向、正で流入）
    mass_flow : float
        MASS_FLOW_INLET の質量流量 [kg/s]。深さ方向の厚さ h を含む 3 次元値で、
        u_n = mass_flow / (ρ Σ_f h_f A_f) の一様流入速度に換算する。
        INTERIOR_MASS_SOURCE / SINK では、セルの h_c V_c で按分した単位深さソース
        q_c = mass_flow · V_c / Σ_c h_c V_c [kg/s] になる（正で注入 / 吸出）
    pressure : float
        PRESSURE_OUTLET / INTERIOR_PRESSURE_SINK の圧力 [Pa]
    conductance : float
        INTERIOR_PRESSURE_SINK のマニホールドコンダクタンス [kg/(s·Pa)]（3 次元値）。
        単位深さでは q_c = conductance · V_c / Σ_c h_c V_c · (p_c - pressure)。
        p_c < pressure なら逆流（面内運動量ゼロで注入）
    weight : WeightFn | None
        領域内パッチ用の滑らかな重み w(x, y) ∈ [0, 1]。与えると mask の代わりに使い、
        ソースを w_c V_c / Σ_c w_c h_c V_c で按分する。位置・径を連続設計変数にするために
        使う（`smooth_disk` 参照）。境界種別では無視
    name : str
        識別用ラベル
    """

    kind: BoundaryKind
    mask: MaskFn | None = None
    velocity: float = 0.0
    mass_flow: float = 0.0
    pressure: float = 0.0
    conductance: float = 0.0
    weight: WeightFn | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mask is None and (not self.is_interior or self.weight is None):
            raise ValueError("mask が必要です（領域内パッチは weight でも可）")

    @property
    def is_interior(self) -> bool:
        return self.kind in INTERIOR_KINDS

    def weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """セル中心での重み配列（weight があればそれ、無ければ mask を 0/1 に）."""
        if self.weight is not None:
            return np.clip(np.asarray(self.weight(x, y), dtype=float), 0.0, 1.0)
        return np.asarray(self.mask(x, y), dtype=float)


def smooth_disk(cx: float, cy: float, r: float, eps: float) -> WeightFn:
    """中心 (cx, cy)、半径 r の円板の滑らかな窓関数 w = ½(1 + tanh((r − d)/eps)).

    eps は遷移幅 [m]（セル幅程度にすると格子で解像される）。cx, cy, r に対して滑らかなので
    位置・径を連続設計変数にできる。
    """

    def w(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return 0.5 * (1.0 + np.tanh((r - d) / eps))

    return w


def rect_mask(x0: float, x1: float, y0: float, y1: float) -> MaskFn:
    """矩形 (x0, x1)×(y0, y1) を選ぶマスク（領域内パッチ用）."""
    return lambda x, y: (x > x0) & (x < x1) & (y > y0) & (y < y1)


def disk_mask(cx: float, cy: float, r: float) -> MaskFn:
    """中心 (cx, cy)、半径 r の円板を選ぶマスク（領域内パッチ用）."""
    return lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < r**2


def west_span(y0: float, y1: float, lx: float = 0.0) -> MaskFn:
    """左壁 x=0 の y∈(y0, y1) を選ぶマスク（lx は未使用、シグネチャ統一のため）."""
    return lambda x, y: (x <= 1e-12) & (y > y0) & (y < y1)


def east_span(y0: float, y1: float, lx: float) -> MaskFn:
    """右壁 x=lx の y∈(y0, y1)."""
    return lambda x, y: (np.abs(x - lx) <= 1e-12) & (y > y0) & (y < y1)


def south_span(x0: float, x1: float) -> MaskFn:
    """下壁 y=0 の x∈(x0, x1)."""
    return lambda x, y: (y <= 1e-12) & (x > x0) & (x < x1)


def north_span(x0: float, x1: float, ly: float) -> MaskFn:
    """上壁 y=ly の x∈(x0, x1)."""
    return lambda x, y: (np.abs(y - ly) <= 1e-12) & (x > x0) & (x < x1)


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
    velocity_floor_ratio : float
        擬似時間の速度スケール下限 = velocity_floor_ratio × 最大流入速度。
        Δτ = CFL·Δx / max(|u|+|v|, 下限)。静止初期場では下限が Δτ を決める
    pseudo_time_mode : PseudoTimeMode
        LOCAL: セル局所 Δτ、GLOBAL: 局所 Δτ の全セル最小値を一律に使う
    rhie_chow_pseudo_time : bool
        True なら Rhie–Chow 係数を d_f = V/(a_P + ρV/Δτ) とする（擬似時間項を
        運動量対角に含めたまま RC を組む実装の再現用）。残差が Δτ に依存するようになる。
        既定 False（d_f = V/a_P、残差は Δτ に依存しない）
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
    velocity_floor_ratio: float = 0.1
    pseudo_time_mode: PseudoTimeMode = PseudoTimeMode.LOCAL
    rhie_chow_pseudo_time: bool = False


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
        inlet 流速 [m/s]（boundaries=None のとき geometry の inlet に使う）
    boundaries : tuple[BoundaryPatch, ...] | None
        座標マスクによる境界パッチ。None なら geometry + u_inlet から
        「左壁 inlet（速度）/ 左壁 outlet（p=0）」を生成する（従来互換）
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
    boundaries: tuple[BoundaryPatch, ...] | None = None
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

    def effective_boundaries(self) -> tuple[BoundaryPatch, ...]:
        """boundaries が None なら geometry + u_inlet から従来の左壁 inlet/outlet を組む."""
        if self.boundaries is not None:
            return tuple(self.boundaries)
        g = self.geometry
        return (
            BoundaryPatch(
                BoundaryKind.VELOCITY_INLET,
                west_span(g.inlet_y0, g.inlet_y1),
                velocity=self.u_inlet,
                name="inlet",
            ),
            BoundaryPatch(
                BoundaryKind.PRESSURE_OUTLET,
                west_span(g.outlet_y0, g.outlet_y1),
                pressure=0.0,
                name="outlet",
            ),
        )

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
    steady_residual_ratio : float
        最終場で評価した Δτ 非依存の定常残差 ||R(x)||/||R0||。
        `rhie_chow_pseudo_time=False` では residual_history[-1]/residual_history[0] と一致する
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
    steady_residual_ratio: float = float("nan")
