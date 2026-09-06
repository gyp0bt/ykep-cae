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
