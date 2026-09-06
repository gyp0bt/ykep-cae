# .inp → 面ベース非構造メッシュ（`InpMeshProcess`）設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← .inp フォーマット](inp-format.md) | [← FVM 層](fvm-layer.md)

## 目的

`*NODE` / `*ELEMENT` で与えられた六面体（C3D8 系）/ 楔（C3D6 系）/ 四面体（C3D4 系）/ 角錐（C3D5）、
2D なら四辺形（CPS4 系）/ 三角形（CPS3 系）のメッシュ（種別の混在可、2 次要素 C3D10 / C3D15 / C3D20 /
CPS6 / CPS8 は頂点だけを使う）を、箱格子かどうかに関係なく owner/neighbour の面リスト（`MeshData`）に変換する。
`StructuredGridRecoveryProcess`（(i, j, k) 復元、箱格子限定）の代替で、
面ベース FVM 層の方程式ファミリー（`ScalarTransportFVMProcess`、`DarcyFlowProcess`）が使う。

`CaseDefinition` はそのまま。`.inp` フォーマットも変えない。

## 入出力

- `InpMeshInput(case, depth_2d=1.0, reserved_patches=True, baffle_surfaces=())`
- `InpMeshResult(mesh, element_ids, node_ids, cell_sets, node_sets, surface_faces, ndim, baffle_surfaces, baffle_faces, periodic_faces, periodic_surfaces)`
  - `cell_index_of(element_ids)` / `mask_for_elements(element_ids)` / `node_values_to_cells(node_ids, values)`
    は `StructuredGridMap` の同名メソッドの非構造版（結果は `(n_cells,)`）
  - `baffle_surfaces` は実際に内部面を分割した `*SURFACE` 名、`baffle_faces` はその境界面 index（両側）
  - `periodic_surfaces` は `*BOUNDARY, TYPE=PERIODIC` に使った面名（境界条件は置けない）、
    `periodic_faces` は併合してできた内部面 index

### 周期境界（`*BOUNDARY, TYPE=PERIODIC`）

対の 2 面の**面中心を並進 t で照合**し、master 面を内部面に昇格（neighbour = slave 側のセル）して
slave 面を消す。並進を省くと両面の面中心の平均差から決める。併合した内部面には
`MeshData.face_offset = −t` を持たせ、fvm 層はそれで neighbour セル中心を owner 側に戻す
（[fvm-layer.md](fvm-layer.md) の `neighbour_centers`）。幾何（面積・法線・セル体積・セル中心）は
**併合前**に計算するので、周期面があってもセルの幾何は変わらない。

照合できないとき（面数が違う、並進が合わない、法線が反平行でない、内部面を含む、面が重複する）は
ずれの最大値を添えてエラーにする。並進周期のみ（回転・螺旋は未対応）。

## アルゴリズム

1. 要素種別は `TYPE=` に `3D` を含むかで 2D / 3D を決め（混在不可）、節点数で四面体 4 / 角錐 5 / 楔 6 /
   六面体 8（2D は三角形 3 / 四辺形 4）を判定する（種別の混在可）。2 次要素（3D 10 / 15 / 20 節点、
   2D 6 / 8 節点）は Abaqus の並び（頂点が先頭）から**頂点だけ**を取り出して 1 次要素として扱う
   （中間節点は座標表に残るが面の照合・幾何・接続には使わない。1 次の面ベース FVM なので中間節点の
   幾何情報は捨てる）。2D は厚さ `depth_2d` で z 方向に押し出して楔 / 六面体にする
2. 出力用の接続（`MeshData.connectivity`、要素の最大節点数幅で -1 詰め）は右手系（底面 → 上面、四面体・
   角錐は正の体積）に正規化する。面ラベル S1.. は **元の節点順**で定義するので、反転した要素（時計回りの
   四辺形、底面と上面が逆の六面体）でも `*SURFACE` は同じ幾何面を指す。`cell_types` は VTK 種別
   （四面体 10 / 楔 13 / 六面体 12 / 角錐 14）
3. 各要素の面（Abaqus の S1..: 四面体 4 面、角錐 5 面（S1 が底面）、楔 5 面、六面体 6 面）を、owner から
   見て外向きに並べた上で節点集合をキーに照合する。2 回現れた面が内部面（owner = 先に現れた要素）、
   1 回だけの面が境界面。3 回以上は不正メッシュ
4. **バッフル**（`baffle_surfaces`）: 指定した `*SURFACE` が内部面を含めば、その内部面を owner 側（元の
   節点列、owner から外向き）と neighbour 側（逆順）の 2 枚の境界面に分割し、内部面リストから外して
   境界面リストの末尾に付ける。どちらの要素側から `*SURFACE` を書いても両側の面が入る。
   厚さゼロの薄板・仕切りに使う（両側が同じ境界条件を受ける。片側ごとの条件や熱伝達の連成は無い）
5. 面の幾何（面積・法線・中心）と要素の幾何（体積・中心）は polyMesh リーダと同じ
   `compute_face_geometry` / `compute_cell_geometry`（三角形分割 + 発散定理）
6. `*SURFACE` は (要素, 面ラベル) → 面 index に解決し、境界面だけの面を `mesh.boundary_patches` に載せる。
   内部面を含む面（バッフルにしなかったもの）はパッチにならず `surface_faces` に内部面 index のまま残る
   （境界条件の target にすると `.inp` マッピングが「バッフルにするには…」のエラーを出す）。
   2D の辺 S1..S4（三角形は S1..S3）は押し出し六面体（楔）の側面 S3..S6（S3..S5）
7. 予約面名 `XM/XP/YM/YP/ZM/ZP` は境界面を外向き法線の主軸（成分最大の軸と符号）で分類して自動生成
   （バッフル面は外皮ではないので分類しない）。曲面境界では意味が薄いので、その場合は `*SURFACE` を使う

`.inp` ランナー（`InpCaseRunnerProcess`）は `*BOUNDARY` / `*DFLUX, S` / `*SFILM` の target になった
`*SURFACE` を `baffle_surfaces` に渡す。箱格子で `StructuredGridRecoveryProcess` が通っても、その
`*SURFACE` が外皮に無い面を含めば非構造経路に切り替える（`--mesh=structured` ではエラー）。
例題 [channel-baffle-1](../../examples/inp/channel-baffle-1.inp)（`*GRID` の流路に下半分を塞ぐ薄板）。

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
最大非直交角は Kuhn 四面体で 35°、三角形押し出しの楔で 34°。角錐（立方体を中心頂点で 6 分割）は
体積・閉包・面数・向きの正規化を `TestInpMeshAPI::test_pyramids_fill_cube` で確認している（伝熱の精度検証は未）。

## 制限

- 2 次要素は頂点だけを使う（曲面境界の中間節点は無視。2 次精度の幾何は持ち込まない）
- セル中心は面中心の平均（一次近似。四面体では重心と一致）
- バッフルは両側同条件（片側ごとの条件、薄板の熱伝導・熱容量、内部の流入・流出面は無い。内部の吐出・吸入は
  要素集合を target にした `*BOUNDARY` → `InternalCellBC`）
- 周期境界は並進のみ。予約面名（`XM..ZP`）は `*SURFACE` のパッチと**同じ面を含みうる**ので、
  両方に境界条件を書くと後勝ちになる（明示した `*SURFACE` を使うのが安全）
- 3D 描画（mirador）は四面体 / 楔を C3D4 / C3D6 として描くが、角錐（C3D5）は messi 側に要素タイプが無いので
  `MiradorExportProcess` が `ValueError`（VTK 出力は可）
