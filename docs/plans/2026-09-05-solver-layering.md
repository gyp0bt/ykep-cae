# ソルバー層の棚卸しと分離計画（2026-09-05）

[← README](../../README.md) | [← roadmap](../roadmap.md) | [FVM 層の設計](../design/fvm-layer.md)

実験で作ってきたソルバー群を低レイヤーから棚卸しし、実験側（`nsb/`、`experiments/`）は残したまま
本体（`xkep_cae_fluid/`）を **Geo 層 / FVM 低レイヤー / 方程式ファミリー** の 3 層に分ける。
非構造格子対応を前提にする。status ファイルは書かず、気づいたことは末尾の findings メモに残す。

## 1. 棚卸し（着手時点、コミット 91e2705）

### 本体ソルバー群

| パッケージ | 格子表現 | 境界条件 | 線形ソルバー | 離散化の書き方 | 非構造へ流用 |
|---|---|---|---|---|---|
| `heat_transfer` | 物性配列の shape から `(nx,ny,nz)` を推定、`dx=Lx/nx`（不等間隔は `dx_array`） | 6 面固定 `bc_xm..bc_zp`、Dirichlet/Neumann/Robin | spsolve / spilu+bicgstab / pyamg+cg / Jacobi / numba GS（独自） | `[1:-1]` スライスと i,j,k 三重ループ、4 実装並存 | 不可 |
| `natural_convection` | `nx,ny,nz`、等間隔のみ | 6 面固定 + `InternalFaceBC`（セルマスク罰則） | spilu+bicgstab、pyamg（AMG キャッシュは heat_transfer の複製） | 構造格子スライス。`di,dj,dk` 近傍ブロックが 4 回複製、TVD/RC/勾配もスライス | 不可 |
| `scalar_transport` | `nx,ny,nz` | 6 面固定（heat_transfer の改名版） | spilu+bicgstab（natural_convection の複製） | 1 本の融合ループ、外部から面速度を受けられる | 部分（最小） |
| `brinkman_flow` | 2D `(nx,ny)`、面 id 明示 | 座標マスク関数 `BoundaryPatch` | 解析ヤコビアン + splu / JFNK GMRES | 疎な cell↔face 演算子の合成（`Dx@diag(f)@W`）。残差評価はスライス、境界は W/E/S/N 固定 | 部分（演算子合成は流用可） |
| `core/strategies` | `MeshData` の owner/neighbour | なし | CSR を返すだけ | 面ループ、非直交補正付き | 可（利用者ゼロだった） |

**繋がっていなかったもの**: `MeshData.connectivity` を読むコードはゼロ、`core/strategies` はテストからしか呼ばれず、
`StrategySlot` の実利用は `extruder` の粘度モデルだけ。`PolyMeshReaderProcess` の本番呼び出しはゼロで、
`connectivity` は `sorted()` した節点集合（頂点順序が壊れている）。`StructuredMeshProcess` は docstring に反して
境界面を作っていなかった。`core/data.BoundaryData` / `SolverInputData` は re-export のみ。

**複製**: BC 係数（Dirichlet `2k/d²` / Neumann / Robin）5 か所、面調和平均 5 か所、`_flat_index` 系 3 か所、
AMG キャッシュ 2 か所、spilu+bicgstab ラッパ 2 か所、van Leer/Superbee が strategies と natural_convection で二重。

### 幾何・境界条件・入力層

- `.inp` 層は `StructuredGridRecoveryProcess`（`inp/grid.py`）が唯一のチョークポイントで、箱格子以外を 7 段の検査で拒否。
  `CaseDefinition` 自体は順序付き接続を持ち位相中立
- `*SURFACE` は 6 面のどれかに潰され、部分面・内部面は拒否
- 幾何系 PreProcess は 8 個あるが出力型がバラバラ（`MeshData`、dx/dy/dz、3D マスク、厚さ場、`ChannelGrid`）
- 出力（NPZ の `x_lines`、VTK `RECTILINEAR_GRID`、mirador の line 配列）は非構造を表現できなかった

### 実験ソルバー群（実験側に残す）

| モジュール | 物理 / 手法 | 本体依存 | 備考 |
|---|---|---|---|
| `nsb/` | Brinkman-NS、Newton+擬似時間、PARDISO | なし（スナップショット） | `data.py`/`assembly.py` は `brinkman_flow` のコミット 1647839 時点の複製（切り離し済み）。`solver.py` は遅延前処理・棄却・Stokes 初期場で分岐済み |
| `experiments/coldplate/darcy.py` | Darcy + Forchheimer + 2 層エネルギー、FVM、離散随伴 | なし（torch） | `(fi, fj)` 面リストで既に非構造。`DarcyFlowProcess` の原型 |
| `experiments/coldplate/coldplate.py` | 配管ネットワーク、SIMP | なし（torch） | グラフ生来、密行列 |
| `experiments/brinkman_uturn`, `scripts/{convergence_evaluation,...}` | 本体ソルバーの診断スクリプト | あり | 本体の入力型が変わったら追随が必要 |
| `scripts/fdm_gs_1d_v3.py` 等 | 1D ジュール加熱 | なし | 面抵抗ループ、Process 化されていない |

Process クラスを定義している実験モジュールはゼロ。

## 2. 分離の形

| 層 | 役割 | 実装 |
|---|---|---|
| Geo 層（PreProcess） | 幾何・領域・境界パッチを `MeshData`（内部面 → 境界面、`boundary_patches`）に載せる | `StructuredMeshProcess`（6 パッチ）、`PolyMeshReaderProcess`（節点順序付き接続 + パッチ）、`InpMeshProcess`（`.inp` → 面リスト、`*SURFACE` → パッチ） |
| FVM 低レイヤー | 面リストの上の境界条件・面演算・係数組み立て・線形ソルバー Strategy | `xkep_cae_fluid.fvm`（[設計](../design/fvm-layer.md)） |
| 方程式ファミリー（SolverProcess） | 低レイヤーを組み合わせる薄い層 | `ScalarTransportFVMProcess`（パイロット）、`DarcyFlowProcess`（`*DARCY`） |

実験側は据え置き。`nsb/{data,assembly}.py` は **コミット 1647839 時点のスナップショット**として本体側から切り離した
（2026-09-05 決定。同期スクリプト `scripts/sync_nsb_from_xkep.py` と乖離テストは削除、`nsb/README.md` に明記）。
本体側の `brinkman_flow` はここから非構造格子（面ベース FVM 層）へ移行する。

## 3. 進め方と現在地

1. [x] 低レイヤーの穴埋め: 境界面、polyMesh 接続、`InpMeshProcess`（同じ箱格子を構造経路と面経路で作って
   体積・面積・隣接・パッチが一致する回帰）
2. [x] `ScalarTransportFVMProcess`（構造格子 FDM と定常・非定常で 1e-8 一致）
3. [x] `*DARCY`（新ファミリー、最初の非構造ケース。例題 `examples/inp/darcy-1.inp`）
4. [ ] `HeatTransfer` → `Brinkman` → `NaturalConvection` の順に載せ替え
   - [x] `HeatTransferFVMProcess`（FDM と 1e-8 一致、`*HEAT TRANSFER` の非箱格子 / `--mesh=unstructured` 経路、例題 plate-ht-2）
   - [x] 非直交補正（over-relaxed 分解 + 境界接線補正 + `solve_corrected`）を fvm 層に追加、Darcy / スカラー輸送 / 伝熱に接続
   - [ ] 非構造 NS（SIMPLE + Rhie–Chow を面リストで、Brinkman 抵抗・Boussinesq・エネルギーを 1 ファミリーに）
5. [x] 出力層: 非構造 NPZ / VTK `UNSTRUCTURED_GRID` / mirador の `mesh=` 入力
6. [x] `internal_face_bcs` 欠落バグの修正（分離とは独立）

## 4. findings メモ（status の代わり）

- `StructuredMeshProcess._build_faces` の内部面中心は原点オフセットを無視していた（`np.cumsum(dx)[ii]`）。
  境界面を足して発散定理の体積検査を入れたら露見。修正済み（`origin=(1,2,3)` でも体積誤差 1e-16）
- `PolyMeshReaderProcess` の `connectivity` は `sorted()` で頂点順序が壊れていた。面リストから
  「最も下の面を底面 → 内向きに並べ替え → 側面の辺で上面を決める」で C3D8 順序を復元する
  `build_ordered_connectivity` に置換（六面体・楔・四面体・角錐、他は多面体として -1 詰め）
- `NaturalConvectionFDMProcess` の過渡 dt 差し替え（`solver.py`）は入力 dataclass を全フィールド手書きで
  再構築しており `internal_face_bcs` が 2 ステップ目以降で空になっていた。`dataclasses.replace` に統一。
  当初書いた「大きな dt で回して inlet 速度を確認する」回帰テストは、解が NaN でも inlet セルだけは
  強制値のままなので pre-fix でも通ってしまった（テストの罠）。`_simple_iteration` が受け取る入力を
  捕捉する形に書き直して、pre-fix で失敗 / post-fix で成功を確認
- Darcy のセル速度を Green–Gauss 勾配 × Γ_P で出すと、透過率が不連続な界面セルで両側の混合になり
  一様流にならない（2 層テストで検出）。面流束からの再構成 `u_P = (1/V) Σ q_f (x_f − x_P)` に変更
- `resolve_boundary` は当初「全パッチを順に塗る」実装で、`*SURFACE` 名（INLET）と予約面名（XM）が同じ面を
  指すと後の XM（既定ゼロ勾配）が INLET の Dirichlet を上書きし「圧力の基準がありません」になった
  （darcy-1 例題で検出）。既定を先に塗ってから指定パッチだけ上書きする形に修正、回帰テスト追加
- `fvm.boundary` の Neumann `flux` の意味は「拡散流束 J = −Γ∇φ の流入量（正 = 流入）」。既存 FDM の
  docstring「Γ∂φ/∂n = flux（正=流入）」と同じ物理だが、式の書き方が紛らわしいので設計文書にはこう書いた
- CI の `test` ジョブには pyamg / numba が無く `TestAMGSolverPhysics` / `TestNumbaSolverPhysics` の 9 件が
  master でも赤（status-34 の TODO、PR #32 にコメント済み）。`AMGSolver` のテストは `importorskip` にした
- 既存 `ScalarTransportProcess`（FDM）は境界面の対流項を持たない。FVM 版はセル速度入力なら境界流出入も
  入るので、回帰は「内部面はセル平均・境界面ゼロ」の面質量流束を明示して取った
- `nsb/{data,assembly}.py` はコミット 1647839 時点のスナップショットとして切り離した（2026-09-05）。同期スクリプトと
  乖離テストを削除。`tests/test_nsb_standalone.py` の subprocess テストは pypardiso が無いと skip
- 非直交補正: 当初「内部面の over-relaxed 分解」だけ入れたら、せん断メッシュで全面 Dirichlet の線形場が
  0.1% ずれた。原因は傾いた境界面で法線勾配を (φ_b − φ_P)/d_b と評価していたこと（評価点が法線の足から
  接線方向にずれる）。境界にも接線補正 −c_b ∇φ_P·t_b を足して厳密になった（`tests/test_fvm_layer.py`）
- 一様せん断メッシュ + 全面 Dirichlet では補正なしでも線形場が厳密に出る（内部面と境界面の誤差が
  行内で打ち消す偶然）。「補正の効果」のテストは Dirichlet 面が傾き、側面が断熱のケースで取ること
- 遅延補正なので、収束解から `diffusive_face_flux` で作る面流量の発散（Darcy の質量不整合）は
  tol × 流量のオーダー（darcy-1: 2e-17 vs 流量 2.5e-6、tol 1e-10）。機械精度ではない
- darcy-1 例題の流量は補正で 2.564e-6 → 2.518e-6 m³/s に変わった（せん断 14°）。結果ファイルを再生成
