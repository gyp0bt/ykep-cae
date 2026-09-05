# ykep .inp 入力フォーマット設計仕様（Abaqus 風キーワード構文）

[← README](../../README.md) | [← 設計文書索引](README.md) | [← ロードマップ](../roadmap.md) | [status-33](../status/status-33.md)

## 目的

解くべき問題を **1 つのテキストファイルで完全に記述**し、`ykep -j=<job>.inp int` で
ykep（xkep-cae-fluid）の各ソルバーを実行できるようにする。構文は Abaqus の
キーワード形式（`*KEYWORD, PARAM=VALUE` + データ行）に揃え、将来 OpenFOAM / Fluent
向けの書き出しを同じ中立表現（`CaseDefinition`）から派生できるようにする。

- 単位系は **SI 固定**（m, kg, s, K, Pa, W）
- キーワード名・パラメータ名・集合名は大文字小文字を区別しない
- ykep 独自の拡張キーワード（`*GRID` 等）は他ソルバーへの書き出し時に標準キーワードへ展開する前提で、
  中立表現には残さない（`*GRID` は節点・要素に展開されて保持される）

## 処理パイプライン（すべて Process）

```
InpKeywordParseProcess   .inp テキスト → KeywordBlock 列
   ├ *INCLUDE 展開、** コメント除去、継続行（行末カンマ）連結
   └ *PARAMETER 評価 + <expr> 置換（ast ホワイトリスト評価。eval 不使用）
InpCaseBuildProcess      KeywordBlock 列 → CaseDefinition（中立表現）
StructuredGridRecoveryProcess
                         *NODE/*ELEMENT（または *GRID）→ StructuredGridMap
                         （軸平行の完全な箱格子であることを検証、要素→(i,j,k)、*SURFACE→領域面）
                         *NAVIER STOKES / *HEAT TRANSFER のステップがあるときだけ
InpMeshProcess           *NODE/*ELEMENT → 面ベースの非構造 MeshData（任意の六面体 / 四辺形、
                         *SURFACE → 境界パッチ、*ELSET → セル集合）。*DARCY のステップがあるときだけ
                         （設計: unstructured-inp-mesh.md）
InpToNaturalConvectionProcess   *NAVIER STOKES → NaturalConvectionInput
InpToHeatTransferProcess        *HEAT TRANSFER → HeatTransferInput（+ 線形解法名）
InpToDarcyProcess               *DARCY → DarcyFlowInput（非構造メッシュ経由）
NaturalConvectionFDMProcess / HeatTransferFDMProcess / DarcyFlowProcess
InpOutputWriterProcess   *OUTPUT, FIELD → <job>.npz / <job>.yaml / <job>.vtk / <job>.html（MiradorExportProcess）
                         非構造（*DARCY）では NPZ に node_coords / connectivity、VTK は UNSTRUCTURED_GRID
InpCaseRunnerProcess     上記を束ねる BatchProcess（ykep コマンドの本体）
```

`CaseDefinition` を挟むことで、ソルバー固有の Input（`NaturalConvectionInput` 等）に
依存しない形でケースを保持できる。OpenFOAM / Fluent 対応はこの層から書き出し Process を
追加すれば良い（未実装）。

## 実行コマンド

```bash
ykep -j=examples/inp/cavity-nc-1.inp int                # Abaqus 風（-j= / job=、int / interactive）
ykep job=cavity-nc-1 interactive -o=results             # 拡張子 .inp は省略可
ykep -j=cavity-nc-1 view -o=results --slice=x=0.05     # 解析せず NPZ → HTML（messi mirador 3D ビューア）
ykep -j=case.inp int -p n=24 -p L=0.2                   # *PARAMETER の初期値（.inp 内定義が優先）
ykep -j=case.inp --check                                # 解析せず読込・格子復元・マッピングのみ検証
python -m xkep_cae_fluid.inp -j=case.inp int            # エントリポイント無しでも同じ
```

- `int` / `interactive`: 反復ログを端末にも表示。無指定でもログは常に `<out>/<job>.log` に残る
  （CLAUDE.md のログ出力ルール。`tee` 相当をコマンド側で担保）
- 終了コード: 0 = 収束、3 = 未収束（ファイルは出力済み）、1 = エラー、2 = 引数エラー
- 出力先は既定で .inp と同じディレクトリ（Abaqus と同じ）

## キーワード一覧

### モデルデータ

| キーワード | パラメータ | データ行 | ykep での扱い |
|---|---|---|---|
| `*HEADING` | – | 自由文 | サマリに記録 |
| `*PARAMETER` | – | `name = 式`（Python 式。`sqrt`, `pi` 等の数学関数のみ許可） | 以降の行の `<name>` / `<式>` を置換 |
| `*INCLUDE` | `INPUT=file` | – | 呼び出し元からの相対パスで展開（循環・深さ 16 超はエラー） |
| `*NODE` | `NSET=` | `id, x, y[, z]` | 2D は z=0 を補う |
| `*ELEMENT` | `TYPE=, ELSET=` | `id, n1, …` | `C3D8` 系（8 節点）と `CPS4/CPE4` 系（4 節点）のみ。四面体・二次要素はエラー |
| `*GRID`（ykep 拡張） | `NX=, NY=, NZ=, LX=, LY=, LZ=, ORIGIN="x y z", ELSET=` | – | 等間隔直交格子の節点・C3D8 要素を生成。要素集合 `ALL`、節点集合 `ALL`、面 `XM..ZP` を自動定義 |
| `*NSET` / `*ELSET` | `NSET=` / `ELSET=`, `GENERATE` | ID、既存集合名、または `first, last, inc` | 名前は大文字正規化 |
| `*SURFACE` | `NAME=, TYPE=ELEMENT` | `elset_or_id, S#` | 面ラベルは要素の幾何から外向き方向を判定（節点順序の回転に依存しない） |
| `*MATERIAL` | `NAME=` | – | 以下のサブキーワードを束ねる |
| `*DENSITY` `*VISCOSITY` `*CONDUCTIVITY` `*SPECIFIC HEAT` | – | 値 1 つ | ρ, μ, k, Cp |
| `*EXPANSION` | `ZERO=T_ref` | β | Boussinesq 体膨張係数と基準温度 |
| `*PERMEABILITY` | – | K [m²] | `*DARCY` の透過率（セクションごと。固体セクションにも指定可） |
| `*FLUID SECTION` / `*SOLID SECTION` | `ELSET=, MATERIAL=` | – | 流体 / 固体（`solid_mask` + `k_solid`）。全セルを重複なく覆う必要あり |
| `*INITIAL CONDITIONS` | `TYPE=TEMPERATURE` | `target, 値` | 節点ベース（nset / 節点 ID / `ALL`）は要素節点平均でセル値に。elset / 要素 ID はセル直接指定（ykep 拡張）。後勝ち |
| `*BOUNDARY` `*SFILM` `*DFLUX` `*DLOAD` | – | – | `*STEP` の外に書くと全ステップに適用 |

### ステップ

```
*STEP, NAME=..., INC=<最大外部反復（MAX_OUTER の既定）>
*<手続き>
[*CONTROLS ...]
[*BOUNDARY / *SFILM / *DFLUX / *DLOAD ...]
[*OUTPUT, FIELD ...]
*END STEP
```

#### 手続き（方程式ファミリー）

| キーワード | パラメータ | データ行（非定常） | ykep ソルバー |
|---|---|---|---|
| `*NAVIER STOKES` | `TURBULENCE=LAMINAR`, `STEADY STATE`, `HEAT TRANSFER=NONE\|COUPLED` | `dt, time_period` | `NaturalConvectionFDMProcess`（SIMPLE 系）。`NONE` は β=0・温度一様・温度境界無視 |
| `*HEAT TRANSFER` | `STEADY STATE` | `dt, time_period` | `HeatTransferFDMProcess` |
| `*DARCY` | `STEADY STATE` | – | `DarcyFlowProcess`（面ベース FVM、非構造六面体メッシュ可）。境界は `TYPE=PRESSURE` / `VELOCITY`（1 成分なら法線流入速度、3 成分なら内向き法線成分）/ `WALL` / `SYMMETRY`（不透過）。`*SFILM` / `*DFLUX` / `*DLOAD` は不可 |

`TURBULENCE` は `LAMINAR` 以外を明示エラーにする（乱流モデルは Phase 5）。

#### `*CONTROLS, PARAMETERS=…`（データ行は `KEY=VALUE, KEY=VALUE`）

| PARAMETERS | KEY | 値 | 対応先 |
|---|---|---|---|
| `DISCRETIZATION` | `CONVECTION` | `UPWIND`（既定）, `VAN LEER`, `SUPERBEE` | `convection_scheme` |
| | `TIME` | `EULER`（既定）, `BDF2` | `time_scheme` |
| | `PRESSURE_VELOCITY` | `SIMPLE`（既定）, `SIMPLEC`, `PISO` | `coupling_method` |
| | `PISO_CORRECTORS` | 整数（既定 2） | `n_piso_correctors` |
| | `LIMITER` | （予約。NaturalConvection ではエラー。Brinkman の Venkatakrishnan 用） | – |
| `RELAXATION` | `VELOCITY` `PRESSURE` `TEMPERATURE` | 0〜1（既定 0.7 / 0.3 / 0.9） | `alpha_u`, `alpha_p`, `alpha_T` |
| | `ADAPTIVE` | `YES` / `NO` | `adaptive_relaxation` |
| `SOLVER` | `PRESSURE` | `BICGSTAB`（既定）, `AMG` | `pressure_solver` |
| `SOLVER`（`*DARCY`） | `METHOD` `TOL` `MAX_ITER` | `DIRECT`（既定）, `BICGSTAB`, `AMG` | `DarcyFlowInput.linear_solver` 等 |
| | `MAX_OUTER` `MAX_INNER` `MAX_PRESSURE_ITER` | 整数 | `max_simple_iter` 等 |
| | `TOL` `TOL_INNER` | 実数 | `tol_simple`, `tol_inner` |
| | `METHOD`（`*HEAT TRANSFER`） | `JACOBI`, `DIRECT`, `BICGSTAB`（既定）, `AMG`, `NUMBA` | `HeatTransferFDMProcess(method=)` |
| | `MAX_ITER` `TOL`（`*HEAT TRANSFER`） | | `max_iter`, `tol` |
| `TIME INCREMENTATION` | `OUTPUT_INTERVAL` | 整数 | `output_interval`（既定は `*OUTPUT, FREQUENCY=`） |

未知のキーは **エラー**（綴り間違いを黙って無視しない）。

#### 境界条件

`*BOUNDARY` の target は `*SURFACE` 名か予約面名 `XM, XP, YM, YP, ZM, ZP`
（別名 `WEST/EAST/SOUTH/NORTH/BOTTOM/TOP`, `X-/X+/...`）。`*SURFACE` は領域 6 面のいずれか
**1 面全体**に一致する必要がある（部分面・内部面は `UnsupportedMeshError`）。

| 書式 | 意味 |
|---|---|
| `*BOUNDARY, TYPE=WALL` + `target[, SLIP]` | no-slip（`SLIP` ですべり壁） |
| `*BOUNDARY, TYPE=VELOCITY` + `target, ux, uy, uz` | 速度流入（全成分 0 なら WALL） |
| `*BOUNDARY, TYPE=PRESSURE` + `target, p` | 圧力流出（ゼロ勾配） |
| `*BOUNDARY, TYPE=OUTLET` + `target` | 対流流出（非反射） |
| `*BOUNDARY, TYPE=SYMMETRY` + `target` | 対称面 |
| `*BOUNDARY, TYPE=TEMPERATURE` + `target, T` | 温度固定 |
| `*BOUNDARY` + `target, dof1, dof2, 値`（Abaqus 自由度番号） | 1-3: 速度成分、8: 圧力、11: 温度。`1, 3, 0.` は WALL |
| `*DFLUX` + `surface, S, q` | 面熱流束 [W/m²]（正 = 流入） |
| `*DFLUX` + `elset, BF, q` | 体積発熱 [W/m³]（`q_vol` / `q`） |
| `*SFILM` + `surface, F, T_inf, h` | 対流熱伝達（`*HEAT TRANSFER` のみ。Robin BC） |
| `*DLOAD` + `elset, GRAV, g, nx, ny, nz` | 重力（大きさ × 方向余弦）。**無指定なら無重力** |

既定: 流体面は no-slip + 断熱。2D 要素（4 節点）のケースでは z の 2 面が **対称面** になる
（nz=1 の準 2D）。

#### 出力

```
*OUTPUT, FIELD [, FORMAT=NPZ|VTK|HTML] [, FREQUENCY=n] [, VARIABLE=U,P,T]
*ELEMENT OUTPUT            ← 変数リスト（*NODE OUTPUT も同じ扱い）
 U, P, T
```

- `<job>.npz`: `x_lines, y_lines, z_lines` と選択した場（`U` は (nx,ny,nz,3)、`P`, `T` は (nx,ny,nz)）。常に出力
- `<job>.yaml`: 収束・反復数・最終残差・経過時間・格子・`*PARAMETER` 値・コミットハッシュ・出力ファイル名
  （STA2 防止ルール: ログと照合可能）
- `<job>.vtk`: `FORMAT=VTK` 指定時。legacy ASCII `RECTILINEAR_GRID` + `CELL_DATA`（ParaView で開ける、依存なし）
- `<job>.html`: `FORMAT=HTML` 指定時（`FORMAT=VTK+HTML` のように併記可）。messi mirador（three.js）の 3D ビューア
  （[設計文書](mirador-export.md)。messi 未導入なら警告して他の出力は続行）。`ykep -j=<job> view` で NPZ から後追い生成もできる。
  **`FORMAT=` を書かなければ（`*OUTPUT` 自体が無くても）messi が import できる環境では自動で HTML も出す**（明示した FORMAT はそのまま）
- 残差マップ: 最終反復のセル別残差 `res_u / res_v / res_w / res_T / res_mass`（+ `res_phi_<name>`、伝熱は定常のみ `res_T`）が場として出る。
  `*ELEMENT OUTPUT` の変数に `RES`（別名 `RESIDUAL`）を書くと全部を選択。変数リストを書かなければ全変数（残差マップ含む）
- 変数の別名: `NT11`/`NT`/`TEMP` → `T`、`V`/`VELOCITY` → `U`、`PRESSURE` → `P`、`RES`/`RESIDUAL` → `res_*` 全部

## 構造格子の復元規則

1. 全要素の節点数が同じ（8 → 3D、4 → 2D）
2. 節点座標を軸ごとにソートし、相対許容 `rel_tol`（既定 1e-8 × 領域寸法）で格子線を抽出
3. 節点数が (nx+1)(ny+1)(nz+1)、要素数が nx·ny·nz に一致（欠損・非直交はエラー）
4. 各要素の 8（4）節点が 1 セルの隅に一致することを検証し、要素 → (i, j, k) を確定
5. 不等間隔格子は `HeatTransferFDMProcess` のみ対応（`NaturalConvectionFDM` は等間隔必須）
6. 2D 要素は z 方向に `depth_2d`（既定 1 m）の 1 セルを補う

## 例題

| ファイル | 内容 | 結果（`examples/inp/results/`） |
|---|---|---|
| [`cavity-nc-1.inp`](../../examples/inp/cavity-nc-1.inp) | `*GRID` + `*PARAMETER` で Ra=1000 差分加熱キャビティ（12×12×3、z 対称面、α_u/α_p/α_T = 0.2/0.05/0.5） | 226 反復で収束、Nu = 1.169（de Vahl Davis 1.118、+4.6%） |
| [`plate-ht-1.inp`](../../examples/inp/plate-ht-1.inp) + [`plate-mesh.inp`](../../examples/inp/plate-mesh.inp) | `*NODE/*ELEMENT` を `*INCLUDE`、`*SURFACE`（S2/S4/S6）、`*SFILM`、`*DFLUX` S/BF、自由度番号形式の `*BOUNDARY` | 直接法で 1 回、T ∈ [355.6, 373.3] K |
| [`darcy-1.inp`](../../examples/inp/darcy-1.inp) + [`darcy-mesh.inp`](../../examples/inp/darcy-mesh.inp) | `*DARCY`: せん断で歪んだ 12×6×2 六面体メッシュ（箱格子ではないので `InpMeshProcess` 経由）、低透過率ブロック `CLAY`（`*SOLID SECTION` + `*PERMEABILITY`）、`*SURFACE` の INLET/OUTLET に圧力 1 kPa / 0 | 直接法で 1 回、流入 = 流出 2.564e-6 m³/s（相対差 2e-13）、p ∈ [18.7, 981.3] Pa、質量不整合 9e-21 |

## 制限（現状）と次の段階

- `*NAVIER STOKES` / `*HEAT TRANSFER` は構造格子のみ（`StructuredGridRecoveryProcess` が非箱格子を拒否）。
  `*DARCY` は `InpMeshProcess` の面ベース非構造メッシュで解く（任意の六面体 / 四辺形、部分面の `*SURFACE` 可）。
  内部面の `*SURFACE` は未対応
- `*DARCY` は定常のみ（Forchheimer / Brinkman 項なし）
- `*NAVIER STOKES, HEAT TRANSFER=NONE` はエネルギー方程式を **スキップせず** 温度一様で解いている
  （`NaturalConvectionInput` に `solve_energy` フラグを足せば計算量を減らせる）
- `*INITIAL CONDITIONS, TYPE=VELOCITY|PRESSURE`: 解析されるがソルバーに初期速度入力が無いため無視
- `*NODE OUTPUT` は `*ELEMENT OUTPUT` と同じ扱い（節点補間はしない）
- 複数 `*STEP` は独立に実行される（前ステップの場を引き継がない）。出力名は `<job>_<k>`
- 格子の次段は「非構造格子（面ベース FVM）」に決めた。`CaseDefinition` はそのままで、
  `InpMeshProcess`（`StructuredGridRecoveryProcess` の代替）と `InpToDarcyProcess` を足した。
  既存の NS / 伝熱ソルバーも同じ層に載せ替えていく（[fvm-layer.md](fvm-layer.md) の移行順）

## テスト

| テストクラス | 対応プロセス / 内容 |
|---|---|
| `tests/test_inp_parser.py::TestInpKeywordParseAPI` | 正規化、フラグ、コメント、継続行、`*PARAMETER`（式・外部上書き・エラー位置）、安全評価、`*INCLUDE`（循環） |
| `tests/test_inp_parser.py::TestInpCaseBuildAPI` | 節点/要素/集合/面/材料/セクション/ステップ、自由度番号形式、エラー 10 種 |
| `tests/test_inp_grid.py::TestStructuredGridRecoveryAPI` | `*GRID`、ID シャッフル + 節点順回転、2D、不等間隔、面解決（全面/部分/内部/複数面）、非箱格子の拒否 |
| `tests/test_inp_mapping.py::TestInpToNaturalConvectionAPI` | 全フィールドの対応、非定常、等温、2D 既定対称面、未対応 8 種、不等間隔拒否、セクション未割当 |
| `tests/test_inp_mapping.py::TestInpToHeatTransferAPI` | k/C/q/T0（節点平均）、Dirichlet/Neumann/Robin、不等間隔、非定常の必須物性、未対応 4 種 |
| `tests/test_inp_mesh.py::TestInpMeshAPI` / `TestInpMeshPhysics` | `InpMeshProcess`: 箱格子で `StructuredMeshProcess` と体積・面積・隣接・パッチが一致、せん断メッシュ、2D 押し出し、`*SURFACE`（plate-mesh.inp）、内部面拒否 |
| `tests/test_inp_mapping.py::TestInpToDarcyAPI` | 透過率のセクション割当、PRESSURE / VELOCITY（ベクトル → 内向き法線成分）/ SYMMETRY、初期圧力、SOLVER、未対応 6 種 |
| `tests/test_inp_runner.py::TestInpOutputWriterAPI` | NPZ/YAML/VTK、変数選択・別名、YAML 往復 |
| `tests/test_inp_runner.py::TestInpCaseRunnerAPI` | 伝熱例題の収束、NS パイプライン、`--check`、`*DARCY` パイプライン（非構造 NPZ / VTK、`view --cut`）、パラメータ上書き |
| `tests/test_inp_runner.py::TestYkepCli` / `TestInpPhysics` | 引数解釈・終了コード・ログファイル、1D 熱伝導の線形分布、darcy-1.inp の質量保存と低透過率ブロック |
