# .inp → 面ベース非構造メッシュ（`InpMeshProcess`）設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← .inp フォーマット](inp-format.md) | [← FVM 層](fvm-layer.md)

## 目的

`*NODE` / `*ELEMENT` で与えられた六面体（C3D8 系）/ 楔（C3D6 系）/ 四面体（C3D4 系）、
2D なら四辺形（CPS4 系）/ 三角形（CPS3 系）のメッシュ（種別の混在可）を、
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

1. 要素種別は `TYPE=` に `3D` を含むかで 2D / 3D を決め（混在不可）、節点数で四面体 4 / 楔 6 / 六面体 8
   （2D は三角形 3 / 四辺形 4）を判定する（種別の混在可、2 次要素は不可）。2D は厚さ `depth_2d` で
   z 方向に押し出して楔 / 六面体にする
2. 出力用の接続（`MeshData.connectivity`、要素の最大節点数幅で -1 詰め）は右手系（底面 → 上面、四面体は
   正の体積）に正規化する。面ラベル S1.. は **元の節点順**で定義するので、反転した要素（時計回りの
   四辺形、底面と上面が逆の六面体）でも `*SURFACE` は同じ幾何面を指す。`cell_types` は VTK 種別
   （四面体 10 / 楔 13 / 六面体 12）
3. 各要素の面（Abaqus の S1..: 四面体 4 面、楔 5 面、六面体 6 面）を、owner から見て外向きに並べた上で
   節点集合をキーに照合する。2 回現れた面が内部面（owner = 先に現れた要素）、1 回だけの面が境界面。
   3 回以上は不正メッシュ
4. 面の幾何（面積・法線・中心）と要素の幾何（体積・中心）は polyMesh リーダと同じ
   `compute_face_geometry` / `compute_cell_geometry`（三角形分割 + 発散定理）
5. `*SURFACE` は (要素, 面ラベル) → 面 index に解決し `mesh.boundary_patches` に載せる。
   内部面を含む面は `UnsupportedMeshError`（内部面境界条件は今後）。2D の辺 S1..S4（三角形は S1..S3）は
   押し出し六面体（楔）の側面 S3..S6（S3..S5）
6. 予約面名 `XM/XP/YM/YP/ZM/ZP` は境界面を外向き法線の主軸（成分最大の軸と符号）で分類して自動生成。
   曲面境界では意味が薄いので、その場合は `*SURFACE` を使う

## 構造格子との整合

同じ箱格子を `StructuredMeshProcess` と `InpMeshProcess` で作ると、セル体積・面積・owner/neighbour の
隣接グラフ・6 パッチの面集合が一致する（`tests/test_inp_mesh.py`）。セルの並びは要素の記述順で、
構造格子の `i*(ny*nz)+j*nz+k` とは一般に異なる（`element_ids` で対応を取る）。

## 四面体・楔での精度

四面体は面中心がセル中心を結ぶ直線から外れる（スキュー）ので、fvm 層の Green–Gauss 勾配は
P–N 直線と面平面の交点で補間した値にスキュー補正 ∇φ_f·(x_f − x'_f) を反復して足す
（`geometry.face_skewness` / `cell_gradient`。Kuhn 分割の四面体で 1 反復あたり誤差 0.15 倍）。
非直交補正（over-relaxed 分解）と合わせ、全面 Dirichlet の線形場は四面体でも 1e-7、両端 Dirichlet・
側面断熱の 1D 熱伝導は 1e-5 で再現する（`tests/test_inp_mesh.py::TestInpMeshPhysics`）。
最大非直交角は Kuhn 四面体で 35°、三角形押し出しの楔で 34°。

## 制限

- 2 次要素（C3D10 / C3D20 / CPS6 / CPS8）、角錐（C3D5）は未対応
- セル中心は面中心の平均（一次近似。四面体では重心と一致）
- 内部面の `*SURFACE`（`InternalFaceBC` 相当）は未対応。3D 描画（mirador）は四面体 / 楔も C3D4 / C3D6 として描く
