"""非圧縮 Navier–Stokes（面ベース FVM、``MeshData`` 上）の入出力スキーマ.

:class:`FlowPatchBC` は 1 パッチの速度境界（:class:`~xkep_cae_fluid.fvm.momentum.VelocityPatchBC`）と
温度境界（:class:`~xkep_cae_fluid.fvm.PatchBC`、None は断熱）の組。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.boundary import PatchBC
from xkep_cae_fluid.fvm.momentum import VelocityPatchBC


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


@dataclass(frozen=True)
class NavierStokesFVMInput:
    """非圧縮 NS（SIMPLE + Rhie–Chow、Boussinesq、Brinkman 抵抗、エネルギー連成）の入力.

    Parameters
    ----------
    mesh : MeshData
        面情報と ``boundary_patches`` を持つメッシュ
    rho, mu : float
        密度 [kg/m³]、粘性係数 [Pa·s]
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
    coupling : str
        ``simple`` / ``simplec``
    linear_solver, pressure_solver : str
        運動量 / 圧力補正の線形ソルバー（direct / bicgstab / amg）
    tol_inner, max_inner_iter :
        反復線形ソルバーの設定
    """

    mesh: MeshData
    rho: float
    mu: float
    bcs: Mapping[str, FlowPatchBC] = field(default_factory=dict)
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
        u / v / w / T / mass の履歴
    residual_fields : dict[str, np.ndarray]
        最終反復のセル別残差 res_u / res_v / res_w / res_T / res_mass
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
