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
   - [x] 非構造 NS `NavierStokesFVMProcess`（SIMPLE/SIMPLEC + Rhie–Chow を面リストで、Brinkman 抵抗・Boussinesq・
     エネルギー・固体マスクを 1 ファミリーに。`*NAVIER STOKES` の非箱格子 / `--mesh=unstructured` 経路、例題 cavity-nc-2）
   - [x] 構造格子版に残る機能の移植（TVD 遅延補正、BDF2、PISO、`InternalFaceBC` → `InternalCellBC`、追加スカラー、
     対流流出 OUTFLOW、`.inp` の `CONVECTION` / `LIMITER` / `TIME=BDF2` / `PRESSURE_VELOCITY=PISO` / `TYPE=OUTLET`）— 2026-09-06
5. [x] 出力層: 非構造 NPZ / VTK `UNSTRUCTURED_GRID` / mirador の `mesh=` 入力
6. [x] `internal_face_bcs` 欠落バグの修正（分離とは独立）
7. [x] 残件の消化（2026-09-06）: 四面体・楔・混在の `InpMeshProcess`（スキュー補正付き勾配）、
   `CorrectedDiffusionScheme` を fvm 層の包みに、Darcy の Forchheimer / 非定常、要素集合 target の内部吐出・吸入、
   死んだ汎用スキーマの削除

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
- 非構造 NS の圧力勾配を Green–Gauss（非 Dirichlet 境界は φ_b = φ_P + 接線外挿）で取ると、流入面や壁に接する
  セルで法線方向の勾配が半分になり、Brinkman 流路の入口セル速度が U/2 になった。圧力・圧力補正の勾配は
  最小二乗 `cell_gradient_lsq`（内部隣接 + Dirichlet 境界面、擬似逆行列）に変更して線形場で厳密に
- 運動量残差を ‖b − A u‖/‖b‖ で正規化すると、恒等的にゼロの成分（2D 流路の v）で b が丸め誤差だけになり
  残差が 1 に張り付いて「未収束」になった。max(‖b‖, ‖A u‖, ‖a_P‖ U_ref) で正規化
- SIMPLE の 1 反復目は初期場（静止・一様温度）の残差がゼロになり得るので、収束判定は 2 反復目から。
  エネルギー連成では T の残差も判定に含める（構造格子版は除外していたが、正規化が正しければ問題なかった）
- せん断メッシュの Poiseuille 流路で出口面に一様圧力を課すと出口近傍の流れが横方向に歪む。境界条件の
  帰結（傾いた面に等圧を強制している）でソルバーの誤差ではないので、検証は流路中央で行う
- cavity-nc-1（箱格子）を `--mesh=unstructured` で解くと 333 反復・max|U| 0.0379（FDM 版 226 反復・0.0357）。
  壁セルの圧力勾配や境界フラックスの評価が違うため 6% ほど差がある。粗い 12×12 での差で、
  Nu はどちらも de Vahl Davis の 20% 以内
- PISO の第 2 補正を「修正済み速度・圧力で Rhie–Chow 流束を作り直して p'' を解く」だけにすると、修正済みの面流束は
  既に保存的なので p'' ≈ 0 で何も変わらない（構造格子版はこの形）。Issa の通り、修正済み速度で隣接項
  H(u) = b − A_off u を再評価し、新しい圧力勾配で u** = H/a_P を作ってから p'' を解く必要がある。運動量行列を
  成分ごとに保持して実装した（`_State.iterate`）
- 上を検証しようとして「外部反復 1 回の非定常ステップ」を連成解と比べたら、補正回数によらず速度が 27–48% ずれた。
  原因は SLIP（対称面）の遅延評価: 接線成分の Dirichlet 値を owner 速度から作るので、初期の静止場では
  z 対称面が no-slip 壁として効いていた。軸に平行な面では法線成分 Dirichlet 0・接線成分ゼロ勾配を陰的に組む
  ように変えた（傾いた対称面だけ遅延評価が残る）。副作用で cavity-nc-2 が 165 → 75 反復、cavity-nc-1 の
  非構造経路が 333 → 274 反復で収束するようになった。PISO の分離誤差は補正 1 → 2 → 3 回で 6% → 1.5% → 0.8%
- 内部吸入セル（`InternalCellBC.outlet`、p' = 0 固定で質量を吸い込む）では発散形 ∇·(ṁT) の対角が流出分だけに
  なり、流入 > 流出のセルで T が流入値を超えた（350 K の吐出で 377 K）。対流を有界形 ∇·(ṁφ) − φ∇·ṁ
  （`assemble_convection(bounded=True)`）にして解消。運動量・エネルギー・追加スカラーとも有界形にした
  （収束した保存的な流束では発散形と同じ）。質量残差からは吐出・吸入セルを除く（湧き出しは設計どおり）
- TVD（van Leer）の蓋駆動キャビティ Re=100（24×24）は中心線 u の極小値が −0.2113（Ghia −0.2109）。1 次風上は
  −0.185。1D 対流拡散（Pe=10、20 セル）では最大誤差 0.069 → 0.022
- 24×24 の 1 次風上キャビティは 322 反復・7 s、TVD は 378 反復・10 s（直接法）。テストの物理ケースは合計 15 s ほど
- 四面体では面中心が P–N 直線から外れる（スキュー）ので、距離重みの Green–Gauss 勾配は線形場でも 2 割以上外れ、
  熱伝導の線形分布が 4% ずれた。P–N 直線と面平面の交点で補間し ∇φ_f·(x_f − x'_f) を反復して足す
  スキュー補正（`face_skewness` / `cell_gradient`、Kuhn 四面体で 1 反復あたり 0.15〜0.3 倍）で線形場が
  1e-8 になった。反復回数は変化が 1e-10 を切るまで（最大 30）
- `InpMeshProcess` の要素向き正規化は出力の接続だけに使い、面ラベル S1.. は元の節点順で解決する。
  以前は時計回りの四辺形を反転していたので、`*SURFACE` の辺ラベルがずれる可能性があった（テスト追加）
- `.inp` の `*MATERIAL` サブキーワード追加で、`_handle_material_property` が dataclass を全フィールド手書きで
  再構築していて新フィールドが落ちた（`internal_face_bcs` のときと同じ罠）。`dataclasses.replace` に統一
- PISO 第 2 補正を検証するテストは、非定常ステップの外部反復 1 回と連成解を比べる形が分かりやすい
  （質量残差の履歴は「修正前の流束の不整合」なので PISO でも減らない）
