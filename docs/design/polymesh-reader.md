# PolyMeshReaderProcess 設計文書

[← README](../../README.md)

## 概要

OpenFOAM の `constant/polyMesh/` ディレクトリ形式の非構造化メッシュを
読み込み、MeshData として返す PreProcess。

## 入出力

### PolyMeshInput
- `mesh_dir`: polyMesh ディレクトリのパス

### PolyMeshResult
- `mesh`: MeshData（ノード座標、セル接続、体積、面情報）
- `boundary_patches`: 境界パッチ情報（パッチ名 → type, nFaces, startFace）

## 対応ファイル

| ファイル | 内容 |
|---------|------|
| `points` | ノード座標 (n_points, 3) |
| `faces` | 面のノードリスト |
| `owner` | 各面の owner セル |
| `neighbour` | 内部面の neighbour セル |
| `boundary` | 境界パッチ定義 |

## アルゴリズム

1. 各ファイルをパース（`FoamFile` ヘッダの `format` で ASCII / binary を自動判定）
2. 面の幾何情報（面積、法線、中心）を三角形分割で計算
3. セルの幾何情報（体積、中心）を発散定理で計算（中心は面中心の平均。非凸セルでは近似）
4. `connectivity` を面リストから**節点順序付き**で復元する（`build_ordered_connectivity`）
   - 六面体（4 節点面 × 6）・楔（三角 2 + 四角 3）: 外向き法線 z が最小の面を底面に取り、
     内向き（右手系で上向き）に並べ替えて、底面各節点から側面の辺をたどって上面節点を決める
     （Abaqus C3D8 / C3D6 と同じ順序）
   - 四面体・角錐: 底面 + 頂点
   - それ以外の多面体: 節点集合をソートして `-1` 詰め、`cell_types = CELL_TYPE_POLYHEDRON`
5. `boundary` のパッチ（`startFace`, `nFaces`）を `MeshData.boundary_patches` の
   面インデックス配列にする（`PolyMeshResult.boundary_patches` には型などの生情報も残す）

## 制限事項

- 圧縮形式は非対応
- セル中心・体積は一次近似（面中心の平均）。高精度が要る場合は四面体分割が必要
- 節点順序を復元できない多面体は接続表としては使えない（面ベースの FVM 層では面リストだけ使うので問題ない）
