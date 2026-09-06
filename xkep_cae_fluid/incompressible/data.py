"""非圧縮 Navier–Stokes（面ベース FVM、``MeshData`` 上）の入出力スキーマ.

:class:`FlowPatchBC` は 1 パッチの速度境界（:class:`~xkep_cae_fluid.fvm.momentum.VelocityPatchBC`）と
温度境界（:class:`~xkep_cae_fluid.fvm.PatchBC`、None は断熱）の組。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.boundary import PatchBC
from xkep_cae_fluid.fvm.momentum import VelocityPatchBC
from xkep_cae_fluid.fvm.viscosity import ViscosityModelStrategy


@dataclass(frozen=True)
class FlowPatchBC:
    """1 パッチの流体境界条件（速度 + 温度）.

    Parameters
    ----------
    velocity : VelocityPatchBC
        WALL / INLET / OUTLET / SLIP（対称面）
    thermal : PatchBC | None
        温度境界（Dirichlet / Neumann（熱流束 W/m²、正 = 流入）/ Robin）。None は断熱
    """

    velocity: VelocityPatchBC = field(default_factory=VelocityPatchBC.wall)
    thermal: PatchBC | None = None

    @staticmethod
    def wall(
        temperature: float | None = None,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        heat_flux: float | None = None,
        film: tuple[float, float] | None = None,
    ) -> FlowPatchBC:
        """壁。``temperature`` で温度固定、``heat_flux`` で熱流束、``film=(h, T_inf)`` で対流熱伝達."""
        return FlowPatchBC(VelocityPatchBC.wall(velocity), _thermal(temperature, heat_flux, film))

    @staticmethod
    def rotating_wall(
        angular_velocity: tuple[float, float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        temperature: float | None = None,
        heat_flux: float | None = None,
        film: tuple[float, float] | None = None,
    ) -> FlowPatchBC:
        """剛体回転する壁 u(x) = velocity + ω × (x − center)（回転するバレル・インペラ）."""
        return FlowPatchBC(
            VelocityPatchBC.rotating_wall(angular_velocity, center, velocity),
            _thermal(temperature, heat_flux, film),
        )

    @staticmethod
    def inlet(
        velocity: tuple[float, float, float], temperature: float | None = None
    ) -> FlowPatchBC:
        return FlowPatchBC(VelocityPatchBC.inlet(velocity), _thermal(temperature, None, None))

    @staticmethod
    def outlet(pressure: float = 0.0, temperature: float | None = None) -> FlowPatchBC:
        return FlowPatchBC(VelocityPatchBC.outlet(pressure), _thermal(temperature, None, None))

    @staticmethod
    def symmetry() -> FlowPatchBC:
        return FlowPatchBC(VelocityPatchBC.slip(), None)

    @staticmethod
    def outflow(temperature: float | None = None) -> FlowPatchBC:
        """対流流出（速度・圧力ゼロ勾配、流束を他の境界の流入と釣り合わせる。圧力の基準は持たない）."""
        return FlowPatchBC(VelocityPatchBC.outflow(), _thermal(temperature, None, None))


def _thermal(
    temperature: float | None, heat_flux: float | None, film: tuple[float, float] | None
) -> PatchBC | None:
    if temperature is not None:
        return PatchBC.dirichlet(temperature)
    if heat_flux is not None:
        return PatchBC.neumann(heat_flux)
    if film is not None:
        return PatchBC.robin(film[0], film[1])
    return None


class InternalCellBCKind(Enum):
    """内部セル境界条件の種別（構造格子版 ``InternalFaceBCKind`` と同じ意味）."""

    INLET = "inlet"  # 吐出: 速度（任意で温度）を固定、p' = 0 ピン留め
    OUTLET = "outlet"  # 吸入: p' = 0 ピン留め（圧力基準、速度はそのまま）


@dataclass(frozen=True)
class InternalCellBC:
    """領域内部のセル集合に課す吐出（INLET）/ 吸入（OUTLET）.

    外部フィルターの吐出口・吸込口のように、境界面ではなく領域内部のセルに流れの
    湧き出し・吸い込みを置く。INLET セルは運動量行を速度固定に、エネルギー行を
    ``temperature`` 固定に（None なら拘束しない）置き換え、圧力補正は p' = 0 に固定する
    （質量の湧き出しを許す）。OUTLET セルは圧力補正だけ p' = 0 に固定する。

    Parameters
    ----------
    kind : InternalCellBCKind
    mask : np.ndarray
        (n_cells,) bool。True のセルに適用
    velocity : tuple[float, float, float]
        INLET の速度 [m/s]
    temperature : float | None
        INLET の温度 [K]
    label : str
        識別子（ログ用）
    """

    kind: InternalCellBCKind
    mask: np.ndarray
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temperature: float | None = None
    label: str = ""

    @staticmethod
    def inlet(
        mask: np.ndarray,
        velocity: tuple[float, float, float],
        temperature: float | None = None,
        label: str = "",
    ) -> InternalCellBC:
        return InternalCellBC(InternalCellBCKind.INLET, mask, velocity, temperature, label)

    @staticmethod
    def outlet(mask: np.ndarray, label: str = "") -> InternalCellBC:
        return InternalCellBC(InternalCellBCKind.OUTLET, mask, label=label)


@dataclass(frozen=True)
class ScalarSpec:
    """流れと同じ面質量流束で輸送する追加スカラー（トレーサ、溶存 CO₂ など）.

    ∂φ/∂t + ∇·(u φ) = ∇·(Γ∇φ) + S（時間項の係数 1。密度で重み付けしたければ Γ・S を換算する）

    Parameters
    ----------
    name : str
    diffusivity : float | np.ndarray
        拡散係数 Γ（スカラーかセル配列）
    phi0 : float | np.ndarray
        初期値
    source : np.ndarray | None
        体積ソース (n_cells,)
    bcs : Mapping[str, PatchBC]
        パッチ名 → 境界条件（未指定はゼロ勾配。流入面は Dirichlet を与えると流入値になる）
    alpha : float
        陰的緩和係数
    """

    name: str
    diffusivity: float | np.ndarray
    phi0: float | np.ndarray = 0.0
    source: np.ndarray | None = None
    bcs: Mapping[str, PatchBC] = field(default_factory=dict)
    alpha: float = 1.0


@dataclass(frozen=True)
class NavierStokesFVMInput:
    """非圧縮 NS（SIMPLE + Rhie–Chow、Boussinesq、Brinkman 抵抗、エネルギー連成）の入力.

    Parameters
    ----------
    mesh : MeshData
        面情報と ``boundary_patches`` を持つメッシュ
    rho, mu : float
        密度 [kg/m³]、粘性係数 [Pa·s]（``viscosity_model`` を与えたときは参照値。ログにだけ使う）
    viscosity_model : ViscosityModelStrategy | None
        非ニュートン粘度 μ(γ̇)（:mod:`xkep_cae_fluid.fvm.viscosity` の POWER LAW / CARREAU など）。
        外部反復ごとに最小二乗の速度勾配から γ̇ = sqrt(2 D:D) を評価して μ を更新する Picard 結合。
        変粘度の応力 ∇·(μ∇uᵀ) の余剰項 Σ_j ∂_i u_j ∂_j μ は陽的ソースに入れる
    alpha_mu : float
        粘度更新の緩和係数 μ ← (1−α) μ + α μ(γ̇)（既定 0.5。押出の専用ソルバーと同じ）
    bcs : Mapping[str, FlowPatchBC]
        パッチ名 → 境界条件（未指定は静止壁・断熱）
    solve_energy : bool
        エネルギー方程式を解く（False なら T は ``T0`` のまま、浮力なし）
    Cp, k_fluid : float
        比熱 [J/(kg·K)]、流体の熱伝導率 [W/(m·K)]
    beta, T_ref, gravity :
        Boussinesq 浮力 −ρ β (T − T_ref) g（β = 0 で浮力なし）
    T0, u0, p0 :
        初期温度 (n_cells,)、初期速度 (n_cells, 3)、初期圧力 (n_cells,)
    solid_mask : np.ndarray | None
        固体セル (n_cells,) bool。速度 0、熱伝導は ``k_solid``
    k_solid : np.ndarray | None
        セルごとの熱伝導率 (n_cells,)（固体セルで使う。None なら k_fluid）
    heat_source : np.ndarray | None
        体積発熱 (n_cells,) [W/m³]
    permeability : np.ndarray | None
        Brinkman 抵抗の透過率 K (n_cells,) [m²]（inf で抵抗なし）。抵抗係数 μ/K
    dt, t_end : float
        非定常なら dt > 0
    max_outer_iter : int
        SIMPLE 外部反復の上限（非定常では各ステップの上限）
    tol : float
        収束判定（運動量の相対初期残差と質量不整合の最大値）
    alpha_u, alpha_p, alpha_T : float
        緩和係数
    adaptive_relaxation : bool
        残差の推移で α_u / α_p を動かす（:func:`~xkep_cae_fluid.fvm.relaxation.adapt_relaxation_factors`、
        構造格子版と同じ規則。非定常では各ステップ内で調整し、ステップを跨いで引き継ぐ）
    coupling : str
        ``simple`` / ``simplec`` / ``piso``（PISO は α_p = 1 で ``n_piso_correctors`` 回の圧力補正）
    n_piso_correctors : int
        PISO の圧力補正回数（既定 2）
    n_nonorthogonal_correctors : int
        圧力補正の非直交補正の反復回数（既定 2。1 = 補正なし。2 以上で前回の p' の勾配から
        陽的な T_f 流束を右辺に足して p' を解き直す。直交メッシュでは常に 1 回。
        非直交角 45° 付近では遅延補正の反復自体が縮小しないので 3 以上にしない）
    convection, limiter : str
        対流スキーム ``upwind``（既定）/ ``tvd`` と TVD リミッタ ``van_leer`` / ``superbee``
        （運動量・エネルギー・追加スカラーに共通、遅延補正）。``none`` は運動量の対流項を落とす
        Stokes 流れ（エネルギー・追加スカラーは風上のまま輸送する）
    body_force : np.ndarray | tuple[float, float, float] | None
        一様または セルごとの体積力 [N/m³]（(3,) または (n_cells, 3)）。運動量のソースに加える。
        周期境界の圧力跳び Δp を P = βx + p̃ に分解したときの −β（押出の G）はここに入れる
    time_scheme : str
        ``euler``（陰的 1 次）/ ``bdf2``（2 次。最初のステップは Euler）
    scalars : tuple[ScalarSpec, ...]
        追加スカラー（収束判定には含めない）
    internal_bcs : tuple[InternalCellBC, ...]
        領域内部の吐出・吸入セル
    linear_solver, pressure_solver : str
        運動量 / 圧力補正の線形ソルバー（direct / bicgstab / amg）
    tol_inner, max_inner_iter :
        反復線形ソルバーの設定
    """

    mesh: MeshData
    rho: float
    mu: float
    bcs: Mapping[str, FlowPatchBC] = field(default_factory=dict)
    viscosity_model: ViscosityModelStrategy | None = None
    alpha_mu: float = 0.5
    solve_energy: bool = False
    Cp: float = 1000.0
    k_fluid: float = 1.0
    beta: float = 0.0
    T_ref: float = 300.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    T0: np.ndarray | None = None
    u0: np.ndarray | None = None
    p0: np.ndarray | None = None
    solid_mask: np.ndarray | None = None
    k_solid: np.ndarray | None = None
    heat_source: np.ndarray | None = None
    permeability: np.ndarray | None = None
    dt: float = 0.0
    t_end: float = 0.0
    max_outer_iter: int = 500
    tol: float = 1e-5
    alpha_u: float = 0.7
    alpha_p: float = 0.3
    alpha_T: float = 0.9
    coupling: str = "simple"
    n_piso_correctors: int = 2
    n_nonorthogonal_correctors: int = 2
    adaptive_relaxation: bool = False
    convection: str = "upwind"
    limiter: str = "van_leer"
    time_scheme: str = "euler"
    body_force: np.ndarray | tuple[float, float, float] | None = None
    scalars: tuple[ScalarSpec, ...] = ()
    internal_bcs: tuple[InternalCellBC, ...] = ()
    linear_solver: str = "bicgstab"
    pressure_solver: str = "bicgstab"
    tol_inner: float = 1e-8
    max_inner_iter: int = 200
    output_interval: int = 1

    @property
    def is_transient(self) -> bool:
        return self.dt > 0.0


@dataclass(frozen=True)
class NavierStokesFVMResult:
    """非圧縮 NS の出力.

    Parameters
    ----------
    velocity : np.ndarray
        (n_cells, 3)
    p, T : np.ndarray
        (n_cells,)
    mass_flux : np.ndarray
        面質量流束 (n_faces,)
    converged : bool
    n_outer_iterations : int
        実行した SIMPLE 反復の総数
    n_timesteps : int
    residual_history : dict[str, list[float]]
        u / v / w / T / mass（と追加スカラー名）の履歴
    residual_fields : dict[str, np.ndarray]
        最終反復のセル別残差 res_u / res_v / res_w / res_T / res_mass（/ res_<スカラー名>）
    scalars : dict[str, np.ndarray]
        追加スカラーの最終場（``ScalarSpec.name`` → (n_cells,)）
    alpha_history : dict[str, list[float]]
        ``adaptive_relaxation`` のときの外部反復ごとの ``alpha_u`` / ``alpha_p``（それ以外は空）
    viscosity : np.ndarray | None
        ``viscosity_model`` のときのセル粘度 μ [Pa·s] (n_cells,)（それ以外は None）
    strain_rate, mixing_index : np.ndarray | None
        収束後の速度勾配から作るせん断速度 γ̇ = sqrt(2 D:D) [1/s] と
        混合指数 λ = |D|/(|D|+|Ω|)（0: 純回転、0.5: 単純せん断、1: 純伸長）(n_cells,)。
        粘度モデルの有無に関わらず出す（混練性・滞留時間分布の評価に使う）
    elapsed_seconds : float
    """

    velocity: np.ndarray
    p: np.ndarray
    T: np.ndarray
    mass_flux: np.ndarray
    converged: bool
    n_outer_iterations: int = 0
    n_timesteps: int = 0
    residual_history: dict[str, list[float]] = field(default_factory=dict)
    residual_fields: dict[str, np.ndarray] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    time_history: tuple[float, ...] = ()
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    alpha_history: dict[str, list[float]] = field(default_factory=dict)
    viscosity: np.ndarray | None = None
    strain_rate: np.ndarray | None = None
    mixing_index: np.ndarray | None = None
