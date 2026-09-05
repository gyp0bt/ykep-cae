"""プロセス間 Input/Output データ契約（流体解析向け）.

dataclass(frozen=True) で不変性を保証する。
FDM（差分法）・FVM（有限体積法）の共通データ型を定義する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# メッシュデータ
# ---------------------------------------------------------------------------


# VTK のセル種別 ID（``MeshData.cell_types``）
CELL_TYPE_TET = 10
CELL_TYPE_HEX = 12
CELL_TYPE_WEDGE = 13
CELL_TYPE_PYRAMID = 14
CELL_TYPE_POLYHEDRON = 42


@dataclass(frozen=True)
class MeshData:
    """メッシュ生成結果（構造化/非構造化共通）.

    構造化格子: node_coords + dimensions で表現
    非構造化格子: node_coords + connectivity + cell_types で表現

    面の並び（OpenFOAM 流）:

    - 先頭 ``n_internal_faces`` 本が内部面。法線は owner → neighbour 向き
    - 続く ``n_boundary_faces`` 本が境界面。owner のみで法線は領域外向き
    - ``face_owner`` は全面、``face_neighbour`` は内部面だけの長さ
    - ``boundary_patches`` はパッチ名 → 境界面インデックス配列（構造格子は
      ``XM/XP/YM/YP/ZM/ZP`` の 6 面、polyMesh は boundary ファイルのパッチ名、
      .inp は ``*SURFACE`` 名）

    ``connectivity`` は要素の節点順序を保った表（六面体は Abaqus C3D8 と同じ
    底面 4 節点 + 上面 4 節点の右手系）。多面体など決まった順序が無いセルは
    ``-1`` 詰めで節点集合だけを持ち、``cell_types`` に
    ``CELL_TYPE_POLYHEDRON`` を入れる。
    """

    node_coords: np.ndarray  # (n_nodes, ndim)  ndim=2 or 3
    connectivity: np.ndarray  # (n_cells, max_nodes_per_cell)
    cell_volumes: np.ndarray  # (n_cells,)
    face_areas: np.ndarray | None = None  # (n_faces,)
    face_normals: np.ndarray | None = None  # (n_faces, ndim)
    face_centers: np.ndarray | None = None  # (n_faces, ndim)
    cell_centers: np.ndarray | None = None  # (n_cells, ndim)
    cell_types: np.ndarray | None = None  # (n_cells,) セルタイプID（VTK の ID）
    # 構造化格子用
    dimensions: tuple[int, ...] | None = None  # (nx, ny) or (nx, ny, nz)
    # フェイス-セル接続
    face_owner: np.ndarray | None = None  # (n_faces,) 各面のオーナーセル
    face_neighbour: np.ndarray | None = None  # (n_internal_faces,) 各面の隣接セル
    # 境界パッチ: パッチ名 → 境界面インデックス（n_internal_faces 以上）
    boundary_patches: Mapping[str, np.ndarray] | None = None

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

    @property
    def n_faces(self) -> int:
        return 0 if self.face_owner is None else int(self.face_owner.shape[0])

    @property
    def n_internal_faces(self) -> int:
        return 0 if self.face_neighbour is None else int(self.face_neighbour.shape[0])

    @property
    def n_boundary_faces(self) -> int:
        return self.n_faces - self.n_internal_faces

    @property
    def boundary_faces(self) -> np.ndarray:
        """境界面のインデックス配列 (n_boundary_faces,)."""
        return np.arange(self.n_internal_faces, self.n_faces, dtype=np.int64)

    def patch_faces(self, name: str) -> np.ndarray:
        """パッチ名から境界面インデックスを返す（無ければ KeyError）."""
        if not self.boundary_patches or name not in self.boundary_patches:
            known = sorted(self.boundary_patches or ())
            raise KeyError(f"境界パッチ {name!r} がありません（定義済み: {known}）")
        return np.asarray(self.boundary_patches[name], dtype=np.int64)


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
