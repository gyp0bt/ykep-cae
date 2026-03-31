"""プロセス間 Input/Output データ契約（流体解析向け）.

dataclass(frozen=True) で不変性を保証する。
FDM（差分法）・FVM（有限体積法）の共通データ型を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# メッシュデータ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshData:
    """メッシュ生成結果（構造化/非構造化共通）.

    構造化格子: node_coords + dimensions で表現
    非構造化格子: node_coords + connectivity + cell_types で表現
    """

    node_coords: np.ndarray  # (n_nodes, ndim)  ndim=2 or 3
    connectivity: np.ndarray  # (n_cells, max_nodes_per_cell)
    cell_volumes: np.ndarray  # (n_cells,)
    face_areas: np.ndarray | None = None  # (n_faces,)
    face_normals: np.ndarray | None = None  # (n_faces, ndim)
    face_centers: np.ndarray | None = None  # (n_faces, ndim)
    cell_centers: np.ndarray | None = None  # (n_cells, ndim)
    cell_types: np.ndarray | None = None  # (n_cells,) セルタイプID
    # 構造化格子用
    dimensions: tuple[int, ...] | None = None  # (nx, ny) or (nx, ny, nz)
    # フェイス-セル接続
    face_owner: np.ndarray | None = None  # (n_faces,) 各面のオーナーセル
    face_neighbour: np.ndarray | None = None  # (n_internal_faces,) 各面の隣接セル

    @property
    def n_nodes(self) -> int:
        return self.node_coords.shape[0]

    @property
    def n_cells(self) -> int:
        return self.connectivity.shape[0]

    @property
    def ndim(self) -> int:
        return self.node_coords.shape[1]

    @property
    def is_structured(self) -> bool:
        return self.dimensions is not None


@dataclass(frozen=True)
class BoundaryData:
    """境界条件.

    FDM/FVM 共通の境界条件表現。
    各境界パッチごとに種別（Dirichlet/Neumann/Robin等）と値を保持する。
    """

    # パッチ名 -> 境界面インデックス配列
    patch_faces: dict[str, np.ndarray] | None = None
    # パッチ名 -> 境界条件種別 ("dirichlet", "neumann", "symmetry", "inlet", "outlet", "wall")
    patch_types: dict[str, str] | None = None
    # パッチ名 -> 境界値（スカラー場やベクトル場）
    patch_values: dict[str, np.ndarray | float] | None = None
    # 固定セル（Dirichlet条件のセルインデックス、FDM用）
    fixed_cells: np.ndarray | None = None
    fixed_values: np.ndarray | None = None


@dataclass(frozen=True)
class FluidProperties:
    """流体物性値."""

    density: float  # kg/m^3
    viscosity: float  # Pa*s (動粘度ではなく粘度)
    specific_heat: float = 0.0  # J/(kg*K)
    thermal_conductivity: float = 0.0  # W/(m*K)
    # 非ニュートン流体用
    power_law_n: float = 1.0  # べき乗則指数（1.0 = ニュートン流体）
    power_law_k: float = 0.0  # べき乗則定数

    @property
    def kinematic_viscosity(self) -> float:
        """動粘度 nu = mu / rho."""
        return self.viscosity / self.density if self.density > 0 else 0.0


# ---------------------------------------------------------------------------
# ソルバー入出力
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowFieldData:
    """流れ場データ（ソルバー入出力）."""

    velocity: np.ndarray  # (n_cells, ndim) 速度場
    pressure: np.ndarray  # (n_cells,) 圧力場
    temperature: np.ndarray | None = None  # (n_cells,) 温度場
    turbulent_ke: np.ndarray | None = None  # (n_cells,) 乱流エネルギー k
    turbulent_epsilon: np.ndarray | None = None  # (n_cells,) 散逸率 epsilon
    scalar_fields: dict[str, np.ndarray] | None = None  # 追加スカラー場


@dataclass(frozen=True)
class SolverInputData:
    """ソルバー入力（定常/非定常共通）.

    FDM/FVM ソルバーへの統一入力インタフェース。
    """

    mesh: MeshData
    boundary: BoundaryData
    fluid: FluidProperties
    initial_field: FlowFieldData | None = None
    # 時間進行パラメータ
    dt: float = 0.0  # 0.0 = 定常解析
    t_end: float = 0.0
    # NR / SIMPLE ソルバーパラメータ
    max_iterations: int = 1000
    tol_residual: float = 1e-6
    tol_velocity: float = 1e-6
    tol_pressure: float = 1e-6
    # 圧力-速度連成
    coupling_method: str = "SIMPLE"  # "SIMPLE", "SIMPLEC", "PISO", "coupled"
    # 緩和係数
    relax_velocity: float = 0.7
    relax_pressure: float = 0.3
    # 体積力
    gravity: np.ndarray | None = None  # (ndim,)
    source_terms: np.ndarray | None = None  # (n_cells, ndim) 外力項

    @property
    def is_transient(self) -> bool:
        """非定常解析かどうか."""
        return self.dt > 0.0


@dataclass(frozen=True)
class SolverResultData:
    """ソルバー結果."""

    field: FlowFieldData
    converged: bool
    n_iterations: int
    residual_history: tuple = ()
    elapsed_seconds: float = 0.0
    # 非定常解析用
    n_timesteps: int = 0
    time_history: tuple = ()
    field_history: tuple = ()  # FlowFieldData のタプル（スナップショット）
    diagnostics: object | None = None


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyInput:
    """検証プロセスへの入力."""

    solver_result: SolverResultData
    mesh: MeshData
    expected: dict[str, float]  # {"max_velocity": 1.23, ...}
    tolerance: float = 0.05


@dataclass(frozen=True)
class VerifyResult:
    """検証結果."""

    passed: bool
    checks: dict[str, tuple[float, float, bool]]  # {name: (actual, expected, ok)}
    report_markdown: str = ""
    snapshot_paths: tuple[str, ...] = ()
