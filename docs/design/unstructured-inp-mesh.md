# .inp → 面ベース非構造メッシュ（`InpMeshProcess`）設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← .inp フォーマット](inp-format.md) | [← FVM 層](fvm-layer.md)

## 目的

`*NODE` / `*ELEMENT` で与えられた六面体（C3D8 系）/ 四辺形（CPS4 系）メッシュを、
箱格子かどうかに関係なく owner/neighbour の面リスト（`MeshData`）に変換する。
`StructuredGridRecoveryProcess`（(i, j, k) 復元、箱格子限定）の代替で、
面ベース FVM 層の方程式ファミリー（`ScalarTransportFVMProcess`、`DarcyFlowProcess`）が使う。

`CaseDefinition` はそのまま。`.inp` フォーマットも変えない。

## 入出力

- `InpMeshInput(case, depth_2d=1.0, reserved_patches=True)`
- `InpMeshResult(mesh, element_ids, node_ids, cell_sets, node_sets, surface_faces, ndim)`
  - `cell_index_of(element_ids)` / `mask_for_elements(element_ids)` / `node_values_to_cells(node_ids, values)`
    は `StructuredGridMap` の同名メソッドの非構造版（結果は `(n_cells,)`）

## アルゴリズム

1. 要素の節点数は 4（2D）か 8（3D）のどちらか一種。2D は厚さ `depth_2d` で z 方向に押し出して六面体にする
   （四辺形は xy の符号付き面積で反時計回りに揃える）
2. 六面体は `(n1−n0)×(n3−n0)·(n4−n0)` の符号で右手系（底面 → 上面）に正規化する
3. 各要素の 6 面（Abaqus の S1..S6）を、owner から見て外向きに並べた上で節点集合をキーに照合する。
   2 回現れた面が内部面（owner = 先に現れた要素）、1 回だけの面が境界面。3 回以上は不正メッシュ
4. 面の幾何（面積・法線・中心）と要素の幾何（体積・中心）は polyMesh リーダと同じ
   `compute_face_geometry` / `compute_cell_geometry`（三角形分割 + 発散定理）
5. `*SURFACE` は (要素, 面ラベル) → 面 index に解決し `mesh.boundary_patches` に載せる。
   内部面を含む面は `UnsupportedMeshError`（内部面境界条件は今後）。2D の辺 S1..S4 は押し出し六面体の側面 S3..S6
6. 予約面名 `XM/XP/YM/YP/ZM/ZP` は境界面を外向き法線の主軸（成分最大の軸と符号）で分類して自動生成。
   曲面境界では意味が薄いので、その場合は `*SURFACE` を使う

## 構造格子との整合

同じ箱格子を `StructuredMeshProcess` と `InpMeshProcess` で作ると、セル体積・面積・owner/neighbour の
隣接グラフ・6 パッチの面集合が一致する（`tests/test_inp_mesh.py`）。セルの並びは要素の記述順で、
構造格子の `i*(ny*nz)+j*nz+k` とは一般に異なる（`element_ids` で対応を取る）。

## 制限

- 六面体・四辺形のみ（四面体・楔・2 次要素は未対応）
- セル中心は面中心の平均（一次近似）。強く歪んだ要素では非直交補正が要る（`CorrectedDiffusionScheme` は未接続）
- 内部面の `*SURFACE`（`InternalFaceBC` 相当）は未対応
