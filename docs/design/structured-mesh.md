# StructuredMeshProcess 設計

[← README](../../README.md)

## 概要

不等間隔直交格子（ストレッチング対応）を生成する PreProcess。
MeshData を返し、後続のソルバーで利用される。

## 入力: StructuredMeshInput

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| Lx, Ly, Lz | float | 各方向の領域サイズ [m] |
| nx, ny, nz | int | 各方向のセル数 |
| stretch_x/y/z | tuple[float, ...] | ストレッチング指定 |
| origin | tuple[float, float, float] | 原点座標 |

### ストレッチング指定

- **長さ1** `(1.0,)`: 等間隔
- **長さ2** `(ratio, grading)`: 幾何級数ストレッチング
  - ratio: 最大幅/最小幅の比率
  - grading > 0: 一方向（先端細→末端粗）
  - grading < 0: 逆方向
  - grading ≈ 0: 両端集中
- **長さn** `(r1, r2, ..., rn)`: 各セルの幅比率を直接指定

## 出力: StructuredMeshResult

- `mesh: MeshData` — ノード座標、セル接続、体積、面情報、隣接関係
- `dx, dy, dz: np.ndarray` — 各方向のセル幅配列

## MeshData の構成

- `node_coords`: (n_nodes, 3) — 全ノード座標
- `connectivity`: (n_cells, 8) — 六面体セルの8頂点（C3D8 順序: 底面 4 + 上面 4）
- `cell_types`: (n_cells,) — 全て `CELL_TYPE_HEX`（VTK の 12）
- `cell_volumes`: (n_cells,) — セル体積
- `cell_centers`: (n_cells, 3) — セル中心座標
- `face_areas`: (n_faces,) — 面面積
- `face_normals`: (n_faces, 3) — 面法線（内部面は owner → neighbour、境界面は領域外向き）
- `face_centers`: (n_faces, 3) — 面中心（原点オフセット込み）
- `face_owner`: (n_faces,) — 面のオーナーセル
- `face_neighbour`: (n_internal_faces,) — 内部面の隣接セル
- `boundary_patches`: `{"XM": ..., "XP": ..., "YM": ..., "YP": ..., "ZM": ..., "ZP": ...}`
  — 境界面インデックス（.inp の予約面名と同じ）
- `dimensions`: (nx, ny, nz) — 構造化格子の次元

面の並びは OpenFOAM 流で、先頭 `n_internal_faces` 本が内部面（x, y, z 方向の順）、
その後ろに境界面が XM, XP, YM, YP, ZM, ZP の順に続く。`MeshData.n_faces` /
`n_internal_faces` / `n_boundary_faces` / `boundary_faces` / `patch_faces(name)` で参照する。
セル添字は `i * (ny * nz) + j * nz + k`（i 最遅・k 最速）で、各ソルバーの `ravel()` と同じ。

境界面まで持つので、面ベース FVM 層（[fvm-layer.md](fvm-layer.md)）の境界条件をそのまま載せられる。

## 使用例

```python
from xkep_cae_fluid.core import StructuredMeshProcess, StructuredMeshInput

inp = StructuredMeshInput(
    Lx=1.0, Ly=0.5, Lz=0.1,
    nx=20, ny=10, nz=5,
    stretch_x=(3.0, 1.0),  # x方向: 先端で細かく
)
result = StructuredMeshProcess().process(inp)
mesh = result.mesh
```
