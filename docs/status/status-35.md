# status-35: ソルバー体験の現在地整理 + Phase 11 残 TODO の消化

[<- README](../../README.md) | [<- status-index](status-index.md) | [ロードマップ](../roadmap.md) | [前: status-34](status-34.md)

**日付**: 2026-09-06
**ブランチ**: `claude/phase11-roadmap-todo-n3s0lj`
**起点コミット**: 40c8e8f（PR #32 マージ後の master）
**テスト数**: 775（収集。615 + 本 status で追加した 17 件 + Phase 11 残件で status に未記録だった分）
**契約違反**: 0 件（登録プロセス 35、起点時点）

## 目的

roadmap Phase 11 の残 TODO（圧力補正への非直交補正、`ADAPTIVE` 緩和の非構造版、2 次要素・角錐の
`InpMeshProcess`、内部面 `*SURFACE` の境界条件）を消化する前に、**ユーザーがソルバーをどう使うか
（ソルバー体験）の現在地**を 1 か所に整理する。status-34 以降の Phase 11 の作業（PR #32 に含まれる
a585b90 / 8179eb9 / 75abcf2 等）は status ファイルを作らず
[plans/2026-09-05-solver-layering.md](../plans/2026-09-05-solver-layering.md) の findings メモに
書いてきたので、status-index との不整合もここで埋める。

## 1. ソルバー体験の現在地（2026-09-06、コミット 40c8e8f 時点）

### 1.1 入口: 何を書いて、何を打つか

| 入口 | 使い方 | 位置付け |
|---|---|---|
| **`.inp` + `ykep` CLI**（主経路） | `ykep -j=<job>.inp int [-o=dir] [-p name=value] [--mesh=auto\|structured\|unstructured] [--check]` | Abaqus 風キーワードで問題を 1 ファイルに完全記述。`*PARAMETER` の式、`*INCLUDE` のメッシュ分離、`*CONTROLS` で離散化・緩和・線形解法を選ぶ。終了コード 0 = 収束 / 3 = 未収束 / 1 = エラー |
| **Python API（Process）** | `XxxProcess().execute(XxxInput(...))` | `.inp` に無い機能（追加スカラー `ScalarSpec`、`InternalCellBC` の細かい指定、水槽 CAE の Geometry/Heater/Filter 連携、押出解析）はこちら。全機能は Process クラス、契約検証 `contracts/validate_process_contracts.py` が関所 |
| **`ykep view`** | `ykep -j=<job> view [--slice=x=0.05] [--cut=z=0.5] [--collapse-panel]` | 解析せず既存の NPZ から messi mirador の 3D HTML を作る |

### 1.2 方程式ファミリーと 2 本の経路

`.inp` の手続きキーワードごとに、**構造格子（箱格子）経路**と**非構造（面ベース FVM）経路**がある。
`--mesh=auto`（既定）は箱格子なら構造格子、そうでなければ非構造へ落ちる。

| 手続き | 構造格子（箱格子のみ） | 非構造（面ベース FVM 層 `xkep_cae_fluid.fvm`） | 非構造だけの制限 |
|---|---|---|---|
| `*NAVIER STOKES` | `NaturalConvectionFDMProcess`（SIMPLE/SIMPLEC/PISO、TVD、BDF2、`ADAPTIVE` 緩和、CFL 適応 dt、`InternalFaceBC`、追加スカラー） | `NavierStokesFVMProcess`（SIMPLE/SIMPLEC/PISO、TVD、BDF2、Boussinesq、Brinkman 抵抗、固体マスク、`InternalCellBC`、追加スカラー、OUTFLOW） | `ADAPTIVE` 緩和なし、圧力補正に非直交補正なし、CFL 適応 dt なし |
| `*HEAT TRANSFER` | `HeatTransferFDMProcess`（Jacobi / GS / 直接 / BiCGSTAB / AMG / Numba、Robin、過渡） | `HeatTransferFVMProcess`（直接 / BiCGSTAB / AMG、非直交補正、Robin、過渡） | 定常のみ `res_T`（構造格子版と同じ） |
| `*DARCY` | なし | `DarcyFlowProcess`（Forchheimer の Picard、比貯留の非定常） | — |

メッシュ入力（Geo 層）: `StructuredMeshProcess`（箱格子 6 パッチ）/ `PolyMeshReaderProcess`（OpenFOAM
polyMesh）/ `InpMeshProcess`（`.inp` の六面体 / 楔 / 四面体、2D 四辺形 / 三角形。`*SURFACE` → パッチ、
`*ELSET` → セル集合）。3 者とも同じ `MeshData`（内部面 → 境界面の順、`boundary_patches`）に落ちるので
方程式ファミリーは格子種別を知らない。

### 1.3 境界条件の書き方（`.inp`）

| 書式 | 構造格子 | 非構造 |
|---|---|---|
| `*BOUNDARY, TYPE=WALL / VELOCITY / PRESSURE / OUTLET / SYMMETRY / TEMPERATURE` + 面名 | 予約面名 XM..ZP か、6 面のどれか 1 面全体に一致する `*SURFACE` | `*SURFACE` は部分面でも可（境界面のみ）。予約面名は外向き法線から自動生成 |
| `*BOUNDARY, TYPE=VELOCITY / PRESSURE / TEMPERATURE` + **要素集合** | 不可（Python API の `InternalFaceBC`） | 領域内部の吐出セル / 吸入セル（`InternalCellBC`） |
| `*DFLUX` S / BF、`*SFILM`、`*DLOAD GRAV` | 可 | 可 |
| 内部面の `*SURFACE`（バッフル・薄板） | 不可 | 不可（`UnsupportedMeshError`）→ **本 status で対応** |

### 1.4 出力と見方

- `<job>.npz`（場、常に）/ `<job>.yaml`（収束・反復数・最終残差・経過時間・`*PARAMETER`・コミットハッシュ）/
  `<job>.log`（反復ログ、`int` で端末にも）/ `<job>.vtk`（`FORMAT=VTK`、ParaView）/ `<job>.html`
  （messi mirador。`FORMAT=` 未指定なら messi のある環境で自動）
- 残差マップ `res_u / res_v / res_w / res_T / res_mass` が場として出る（`*ELEMENT OUTPUT` に `RES`）
- mirador: 外皮 + 断面スラブ + 速度矢印 + 任意平面の view cut（`--cut`）。対数スケール・時系列・複数平面は未対応

### 1.5 収束の現在地（例題の実測、`examples/inp/results/*.yaml`）

| 例題 | 経路 | 反復 | 時間 | 備考 |
|---|---|---|---|---|
| cavity-nc-1（Ra=1000、12×12×1、箱格子） | 構造格子 FDM | 226 | 数秒 | Nu=1.169（de Vahl Davis 1.118 の 5%） |
| cavity-nc-1 `--mesh=unstructured` | 非構造 FVM | 274 | — | max\|U\| が FDM 版と 6% 差（粗い格子の壁セル評価差） |
| cavity-nc-2（平行四辺形、非直交 14°） | 非構造 FVM | 75 | 1.1 s | 対称面を陰的にして 165 → 75 |
| plate-ht-1 / plate-ht-2 | FDM / FVM | 1 | — | 定常伝熱、せん断メッシュで線形場厳密 |
| darcy-1（せん断 + 低透過率ブロック） | FVM | 非直交補正 反復 | — | 流入 = 流出 |

未解決の物理問題（CLAUDE.md の焦点ガード）: 構造格子版の**空気実物性（mu=1.85e-5）+ q_vol** で
mass 残差 O(1–100)、長時間で流速方向の不安定化。status-13〜17 で SIMPLEC / PISO / AMG / 適応緩和を入れて
改善したが完全解消ではない。非構造 NS ではこの構成（低粘性 + 体積発熱の自然対流）を**まだ評価していない**。

### 1.6 体験上のギャップ（優先順）

1. **構造格子 / 非構造の機能差**: `ADAPTIVE` 緩和、CFL 適応 dt は構造格子版だけ。同じ `.inp` が
   `--mesh` によって受理されたりエラーになったりする（本 status で `ADAPTIVE` を埋める）
2. **歪んだメッシュでの圧力補正**: 運動量・エネルギーは非直交補正付き、圧力補正は無し。
   歪みが強いと外部反復で吸収するしかなく反復数が増える（本 status で対応）
3. **メッシュ入力の受理範囲**: 2 次要素（C3D10 / C3D20 / CPS6 / CPS8）と角錐（C3D5）は `UnsupportedMeshError`。
   汎用メッシャの出力をそのまま食えない（本 status で対応）
4. **内部面の `*SURFACE`**: バッフル・薄板・仕切りが置けない。水槽 CAE（仕切り板）と冷却流路（フィン）で必要
   （本 status で「2 面境界に分割するバッフル」として対応）
5. **`HEAT TRANSFER=NONE` の無駄**: 構造格子版はエネルギー方程式を温度一様で解き続ける（status-33 TODO）
6. **CI の赤**: `test` ジョブ（pyamg / numba 無し）で `TestAMGSolverPhysics` / `TestNumbaSolverPhysics` が
   ImportError（status-32 以降）。`ykep` を触らない人にも「master が赤」に見える
7. **ドキュメント**: Phase 11 の作業が status に無く、plans の findings メモに散っている（本 status で回収）
8. **Python API と `.inp` の差**: 追加スカラー、`InternalCellBC` の温度、水槽 CAE 連携は API のみ
9. **後処理**: 対数スケール残差、時系列フレーム、複数 view cut、矢印間引き（status-34 TODO、messi 側）

## 2. 本 status での作業

（各項目は実装後に追記）

### 2.1 非構造 NS: 圧力補正の非直交補正（`n_nonorthogonal_correctors` / `NONORTHOGONAL_CORRECTORS`）

`fvm/momentum.py` に `pressure_correction_nonorthogonal`（c_f = ρ D_f (∇p')_f·T_f、前回の p' の最小二乗勾配で
陽的に評価）を追加し、`assemble_pressure_correction(explicit_flux=)` / `correct_mass_flux(explicit_flux=)` で
同じ c_f を右辺と流束修正の両方に使う（修正後の流束の発散が解いた線形系と厳密に整合）。
`NavierStokesFVMProcess` は各圧力補正で `n_nonorthogonal_correctors` 回（既定 2）p' を解き直す。
直交メッシュ（`is_orthogonal`）では 1 回。`.inp` は `*CONTROLS, PARAMETERS=DISCRETIZATION` の
`NONORTHOGONAL_CORRECTORS=`（構造格子版では明示エラー）。

**実測**（scratchpad `shear_ref.py` / `shear_sweep.py`、tee ログ `shear_ref.log` / `shear_sweep.log`。
16×16 の Stokes 的キャビティ μ=1、dt=0.002 の 1 ステップを SIMPLE で収束させる。tol 1e-9）:

| せん断（非直交角） | α_u, α_p | 補正 1 回 | 2 回 | 3 回 |
|---|---|---|---|---|
| 0（0°） | 0.8, 0.5 | 27 | 27 | 27 |
| 0.3（17°） | 0.8, 0.5 | 150 | 28 | 28 |
| 0.6（31°） | 0.8, 0.5 | **発散** | 28 | 42 |
| 1.0（45°） | 0.8, 0.5 | **発散** | 30 | **発散** |
| 0.6（31°） | 0.7, 0.3 | 61 | 52 | 52 |
| 1.0（45°） | 0.7, 0.3 | 600 で未収束 | 55 | 53 |

- 定常 SIMPLE では p' → 0 なので**収束解は補正回数によらない**（差 1e-10）。効くのは緩和が強いときの安定性
- 3 回目が 45° で発散するのは、遅延補正の不動点反復の縮小率が |T_f|/|E_f| = tan θ ≈ 1 で縮小しないため。
  既定は 2 回にし、45° を超えるメッシュは limited 版（T_f を ψ 倍）が要る（TODO）
- cavity-nc-2（14°、α = 0.5/0.2）は補正回数によらず 75 反復（scratchpad `nonorth_sweep_cavity2_final.log`）。
  差分加熱の自然対流では p' が小さく効かない

テスト: `test_navier_stokes_fvm.py::test_pressure_nonorthogonal_correction_stabilizes_sheared_cavity`
（31°、α=(0.8,0.5): 1 回は 60 反復で未収束、2 回は 40 反復以内で収束し保守的緩和の収束解と 1e-5 で一致、
直交メッシュでは補正回数によらず同じ反復数）。

### 2.2 Rhie–Chow の D_f を緩和前の a_P で（Majumdar 1988）— 発見した不整合の修正

上の検証中に、**収束解が α_u に依存する**ことが分かった（同じせん断キャビティで α_u = 0.7 の収束解に対し
0.8 で 0.8%、0.5 で 1.9% の差。tol を 1e-12 にしても残る）。原因は Rhie–Chow の D_f = V/a_P に**緩和後**の
a_P（= a⁰_P/α_u）を使っていたこと。RC の補正項が α_u 倍にスケールされ、収束解の面流束が α_u に依存する
（Majumdar 1988 で知られた問題）。D⁰_f = V/(α_u a_P) に変えた（圧力補正方程式と速度修正の D_f は従来どおり）。
修正後は α_u = 0.5 / 0.7 / 0.8 の収束解の差が 3e-8 以下（scratchpad `piso_nonorth.py` 系の再測定、本文 2.1 の表と
同じケース）。既存の物理テスト（Poiseuille、Ghia、de Vahl Davis、PISO、内部セル、スカラー）は全て通過。
構造格子版 `NaturalConvectionFDMProcess` にも同じ形（`a_P_u_eff` は緩和後）が残っているので TODO
（空気実物性の不安定化調査と合わせて確認する価値がある）。

### 2.3 適応緩和 `ADAPTIVE` の非構造版 + 規則の共有と改良

`fvm/relaxation.py` を新設し、status-16 の規則（前回比 0.8 未満で 1.1 倍、1.2 超で 0.8 倍、上限 0.9/0.5、
下限 0.1/0.05）を `adapt_relaxation_factors` に切り出した。構造格子版 `_adapt_relaxation` はこれを呼ぶ。
`NavierStokesFVMProcess` は `adaptive_relaxation=True`（`.inp` の `ADAPTIVE=YES`）で毎反復呼び、
`alpha_history` に記録する。

そのまま移植すると cavity-nc-2 で **75 → 474 反復に悪化**した。`alpha_history` を追うと、α が 0.9/0.46 まで
上がった後に残差が 1 反復あたり 1.05 倍ずつじわじわ増え（規則の不感帯 0.8〜1.2 の中）、最小値 1.7e-3 から
7.97 まで発散してから戻っていた。2 つ足した:

- **停滞検出**: 最小残差の `stall_ratio`（5）倍を超えたら前回比によらず保守化（保守化後は最小値を置き直す）
- **SIMPLE の目安** α_p ≤ 1 − α_u（`simple_cap`、新しい α_u で評価）

| ケース | 固定 α | 適応（旧規則） | 適応（改良） |
|---|---|---|---|
| cavity-nc-2（非構造、14°） | 75 | 474 | **62**（補正 2 回では 68） |
| cavity-nc-1 `--mesh=unstructured` | 275 | 219 | 219 |
| cavity-nc-1 構造格子 FDM | 226 | 226 | 226（残差の推移が不感帯内で一度も動かない） |

（scratchpad `adaptive_cmp_final.log` / `nonorth_sweep_cavity2_final.log`）。構造格子版のテスト
（`test_adaptive_relaxation`、slow、pyamg）は規則変更後も通過。

### 2.4 `InpMeshProcess`: 2 次要素・角錐・バッフル

- **2 次要素**（C3D10 / C3D15 / C3D20、CPS6 / CPS8）: Abaqus の並びの頂点だけを使う（`_CORNERS_3D/_2D`）。
  `builder.py` の要素タイプ検査も緩和（構造格子復元は 8 節点のみなので、2 次要素は自動的に非構造経路）。
  C3D20 の 2 要素メッシュが C3D8 と同じ面リスト・体積・面ラベルになること、CPS8 シートの体積・閉包をテスト
- **角錐 C3D5**: 面表 `_PYRAMID_FACES`（S1 底面）、右手系への正規化、`CELL_TYPE_PYRAMID`。立方体を中心頂点で
  6 分割したメッシュで体積 1/6 ずつ・閉包・内部面 12 / 境界面 6・向きの正規化を確認。mirador は messi に
  C3D5 が無いので `ValueError`（VTK 出力は可）
- **バッフル**（`InpMeshInput.baffle_surfaces`）: 指定 `*SURFACE` の内部面を owner 側 / neighbour 側の
  2 枚の境界面に分割して同名パッチに。どちら側の要素から書いても両側が入る。予約面名の自動分類から除外。
  内部面を含む `*SURFACE` は（バッフルにしなければ）パッチにならず `surface_faces` に残るだけ（旧: エラー）
- ランナー: `*BOUNDARY` / `*DFLUX, S` / `*SFILM` の target の `*SURFACE` を `baffle_surfaces` に渡す。
  箱格子で構造格子に復元できても、その `*SURFACE` が外皮に無い面を含めば非構造経路へ切り替える
  （`--mesh=structured` ではエラー）
- マッピング: バッフルに `VELOCITY` / `PRESSURE` / `OUTLET` は明示エラー。`*HEAT TRANSFER` で `TYPE=WALL` を
  断熱として受理（バッフルを WALL で置くため）
- 例題 `examples/inp/channel-baffle-1.inp`: `*GRID` 32×8 の流路（Re 0.4）の中央に下半分を塞ぐ薄板。26 反復で収束、
  隙間の平均流速 0.0184（入口の 1.8 倍）、流入 = 流出（1e-5）、板の面の流束 0（tee ログ `examples/inp/results/channel-baffle-1.log`）
- テスト: `test_inp_mesh.py` +4、`test_inp_mapping.py::TestInpMeshBaffleMapping`（4）、`test_inp_runner.py` +2
  （CPS4 流路と例題）、`test_heat_transfer_fvm.py::test_adiabatic_baffle_decouples_halves`
  （4×1 板の中央を断熱バッフルにすると左 2 セルが 1、右 2 セルが 0）

### 2.5 その他

- `NaturalConvectionInput.solve_energy`（既定 True）。`HEAT TRANSFER=NONE` で構造格子版もエネルギー方程式を
  組まない（T は初期場のまま、`res_T` は 0）。status-33 TODO
- `NavierStokesFVMProcess` に反復ログ（1〜5 反復と 10 反復ごと、`ykep ... int` で端末に出る。従来は最終行だけ）、
  発散検出（NaN / 1e20 超で打ち切り）、非定常ステップの未収束警告
- CI: `tests/test_heat_transfer_fdm.py` の `TestAMGSolverPhysics` / `TestNumbaSolverPhysics` を
  `importlib.util.find_spec` の `skipif` に（`test` ジョブが status-32 以降ずっと赤だった件）
- 既存テストの更新: 内部面 `*SURFACE` が例外でなくなった件、HT の WALL 受理、NS FVM の ADAPTIVE 受理、
  C3D10 / CPS6 が構文エラーでなくなった件

## 3. 検証（STA2 防止）

- ブランチ `claude/phase11-roadmap-todo-n3s0lj`、起点 40c8e8f。scratchpad のスクリプトと tee ログは
  リポジトリ外（`/tmp/.../scratchpad/`: `shear_ref.log`, `shear_sweep.log`, `piso_nonorth.log`,
  `nonorth_sweep_cavity2*.log`, `adaptive_cmp*.log`, `tests-*.log`）。数値は本文の表に転記した
- 例題は全件を `python -m xkep_cae_fluid.inp -j=<job> int -o=examples/inp/results` で再実行し、
  `examples/inp/results/*.yaml` / `*.log` を更新（cavity-nc-1: 226 反復・max|U| 0.0357、cavity-nc-2: 75 反復・
  max|U| 0.03791（RC の変更で 0.03787 から）、channel-baffle-1: 26 反復、plate-ht-1/2・darcy-1: 変化なし）

```
ruff check xkep_cae_fluid/ tests/ → All checks passed / ruff format --check → 全ファイル整形済み
python contracts/validate_process_contracts.py → 契約違反なし（登録プロセス 35）
python -m pytest tests/ --collect-only -q → 775 tests collected
python -m pytest tests/ -q -m "not slow and not external" -p no:cacheprovider
→ 740 passed / 1 failed / 15 skipped / 18 deselected / 1 xfailed（742.84 s、コミット 0d30756、pyamg・pypardiso 導入済み、numba・messi 未導入）
  1 failed は本 status で追加した test_solve_energy_false_keeps_temperature_field の enum 名の誤り
  （FluidBoundaryCondition.INLET → INLET_VELOCITY）。修正して単独再実行 → 1 passed（本体コードの変更なし）
  skip 15 = messi 未導入の mirador 系 + numba 系 + pypardiso の subprocess テスト等
```

## 4. 次にやること

- [ ] 非直交補正の limited 版（45° 超）。角錐の mirador 描画（messi に C3D5 を足すか 2 四面体に分割して描く）
- [ ] バッフルの片側ごとの条件、薄板の熱伝導・熱容量（水槽の仕切り板・冷却流路のフィンに要る）
- [ ] 構造格子版 `NaturalConvectionFDMProcess` の Rhie–Chow を緩和前の a_P に（2.2）。空気実物性 + q_vol の
  不安定化（CLAUDE.md の焦点ガード）をこの観点で再調査、非構造 NS でも同条件を評価
- [ ] 適応緩和の規則を SIMPLEC / PISO 向けに再検討（上限 0.5 は SIMPLEC には保守的）
- [ ] `.inp` に追加スカラー（`ScalarSpec`）と CFL 適応 dt（非構造版は未実装）
- [ ] status-34 の表示まわり TODO（対数スケール、時系列、複数 view cut、矢印間引き）は据え置き

## ファイル

- 追加: `xkep_cae_fluid/fvm/relaxation.py`、`examples/inp/channel-baffle-1.inp`、`examples/inp/results/channel-baffle-1.{yaml,log}`、`docs/status/status-35.md`
- 変更: `xkep_cae_fluid/fvm/{__init__,momentum}.py`、`xkep_cae_fluid/incompressible/{data,solver}.py`、
  `xkep_cae_fluid/natural_convection/{data,solver}.py`、`xkep_cae_fluid/inp/{builder,mapping,mesh,runner}.py`、
  `tests/test_{fvm_layer,navier_stokes_fvm,inp_mesh,inp_mapping,inp_runner,inp_parser,heat_transfer_fvm,heat_transfer_fdm,natural_convection}.py`、
  `docs/design/{navier-stokes-fvm,fvm-layer,inp-format,unstructured-inp-mesh}.md`、`docs/roadmap.md`、`docs/status/status-index.md`、
  `README.md`、`examples/inp/results/*.{yaml,log}`
