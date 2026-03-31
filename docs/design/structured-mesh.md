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
- `connectivity`: (n_cells, 8) — 六面体セルの8頂点
- `cell_volumes`: (n_cells,) — セル体積
- `cell_centers`: (n_cells, 3) — セル中心座標
- `face_areas`: (n_internal_faces,) — 内部面面積
- `face_normals`: (n_internal_faces, 3) — 内部面法線
- `face_centers`: (n_internal_faces, 3) — 内部面中心
- `face_owner`: (n_internal_faces,) — 面のオーナーセル
- `face_neighbour`: (n_internal_faces,) — 面の隣接セル
- `dimensions`: (nx, ny, nz) — 構造化格子の次元

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
