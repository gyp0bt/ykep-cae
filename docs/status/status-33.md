# status-33: ykep .inp 入力フォーマット（Abaqus 風キーワード構文）+ `ykep -j=<job>.inp int` コマンド

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/inp-format.md) | [前: status-32](status-32.md)

**日付**: 2026-09-05
**ブランチ**: `claude/abaqus-inp-format-ykep-liaxdu`
**テスト数**: 598（`tests/test_inp_parser.py` +22、`tests/test_inp_grid.py` +9、`tests/test_inp_mapping.py` +17、`tests/test_inp_runner.py` +11。本セッション環境（pyamg / numba 未導入）で `pytest tests/ -m "not slow and not external"` の結果は本文末尾）
**契約違反**: 0 件（登録プロセス 26。+7: InpKeywordParse / InpCaseBuild / StructuredGridRecovery / InpToNaturalConvection / InpToHeatTransfer / InpOutputWriter / InpCaseRunner）

## 目的

Abaqus の `.inp` と同じキーワード構文で **解くべき問題を 1 ファイルに完全記述**し、
`ykep -j=<job>.inp int` で ykep の既存ソルバーを実行できるようにする。
究極的には同じ .inp から OpenFOAM / Fluent も回したいので、ソルバー非依存の中立表現
`CaseDefinition` を挟む。今回は ykep 対応のみ（`*NAVIER STOKES` → NaturalConvectionFDM、
`*HEAT TRANSFER` → HeatTransferFDM。`*DARCY` は書式のみ）。

セッション冒頭で設計上の質問 9 点を提示し、すべて推奨案で確定（加えて `*PARAMETER` 対応と
`ykep -j=nsb-1.inp int` の CLI 形式を指定）。

## 実装（`xkep_cae_fluid/inp/`、すべて Process）

| モジュール | Process | 内容 |
|---|---|---|
| `parameters.py` | – | `*PARAMETER` 用の `ast` ホワイトリスト評価（`eval` 不使用）と `<expr>` 置換 |
| `parser.py` | `InpKeywordParseProcess` | `*INCLUDE` 展開、`**` コメント、継続行、`*PARAMETER`、`KeywordBlock` 列化 |
| `case.py` | – | 中立表現 `CaseDefinition`（節点・要素・集合・面・材料・セクション・初期条件・ステップ） |
| `builder.py` | `InpCaseBuildProcess` | キーワードの意味付け。`*GRID`（ykep 拡張、等間隔箱格子生成）も含む |
| `grid.py` | `StructuredGridRecoveryProcess` | `*NODE/*ELEMENT` を軸平行の箱格子として検証・復元、`*SURFACE` → 領域 6 面 |
| `mapping.py` | `InpToNaturalConvectionProcess` / `InpToHeatTransferProcess` | 境界・荷重・物性・`*CONTROLS`（離散化/緩和/ソルバー）の対応付け。未対応指定は `UnsupportedFeatureError` |
| `output.py` | `InpOutputWriterProcess` | `<job>.npz` / `<job>.yaml`（依存なし簡易 YAML）/ `<job>.vtk`（RECTILINEAR_GRID） |
| `runner.py` | `InpCaseRunnerProcess` | ステップごとに方程式ファミリーで振り分けて実行・出力 |
| `cli.py` | – | `ykep -j=<job>[.inp] [int] [-o=dir] [-p name=value] [--check]`。`pyproject.toml` の `[project.scripts]` に登録 |

キーワード一覧・`*CONTROLS` の語彙・境界条件の書式・格子復元規則は
[設計文書](../design/inp-format.md) を参照。

### 設計上の決定

- **メッシュ**: `*NODE/*ELEMENT`（C3D8 系 / CPS4 系）を読み、直交構造格子 (i,j,k) に復元する。
  非構造・非直交・部分面は明示エラー。`*GRID` は短縮記法
- **方程式ファミリー**は `*STEP` 内の手続きキーワード（`*NAVIER STOKES, TURBULENCE=LAMINAR, STEADY STATE, HEAT TRANSFER=COUPLED`）
- **離散化・緩和・ソルバー**は `*CONTROLS, PARAMETERS=DISCRETIZATION|RELAXATION|SOLVER|TIME INCREMENTATION`。未知キーはエラー
- `*BOUNDARY` は `TYPE=` 名前形式と Abaqus 自由度番号形式（1-3 速度、8 圧力、11 温度）の両方
- 重力は `*DLOAD, GRAV` で明示（無指定は無重力）。`*EXPANSION, ZERO=` が Boussinesq の β と T_ref
- 出力は NPZ + YAML サマリ（STA2: コミットハッシュ・`*PARAMETER` 値・最終残差を記録）+ 任意で VTK

## 例題と結果（ログ・YAML は `examples/inp/results/`）

| 例題 | コマンド | 結果 |
|---|---|---|
| `examples/inp/cavity-nc-1.inp`（Ra=1000 差分加熱キャビティ、12×12×3、z 対称面、α = 0.2/0.05/0.5） | `ykep -j=examples/inp/cavity-nc-1.inp int -o=examples/inp/results` | **226 反復で収束**（max_residual 9.78e-5）、Nu = 1.169（de Vahl Davis 1.118、+4.6%。既存 slow テストと同じ許容 20% 内） |
| `examples/inp/plate-ht-1.inp`（`*INCLUDE` した 4×2×2 六面体、`*SURFACE` 3 面、`*SFILM`、`*DFLUX` S/BF） | `ykep -j=examples/inp/plate-ht-1 int -o=examples/inp/results` | 直接法 1 回で収束、T ∈ [355.6, 373.3] K |

最初の試行（10×10×4、α = 0.7/0.3/0.9）は 155 反復で発散した（`converged: false` を YAML に記録）。
既存ベンチマーク `test_differentially_heated_cavity_nusselt` と同じ緩和係数に揃えて収束。

## 調査で分かったこと（未解決、次セッションへ）

- `NaturalConvectionFDMProcess` の **SYMMETRY / SLIP 面は既定緩和（α_u=0.7, α_p=0.3）で発散**する
  （8×8×1 のリッド駆動でも、浮力なしでも 20 反復程度で発散。no-slip なら収束）。
  強い緩和（0.2/0.05）では収束するので BC そのものの誤りではなく安定性の問題と思われるが、
  境界セルの接線成分のゼロ勾配処理と圧力補正の整合を要確認。2D ケースは z 面が既定で対称面になるため影響が大きい

## テスト実行（本セッション環境: pyamg / numba 未導入）

```
python -m pytest tests/test_inp_parser.py tests/test_inp_grid.py tests/test_inp_mapping.py tests/test_inp_runner.py -q
→ 69 passed
python contracts/validate_process_contracts.py → 契約違反なし（登録プロセス 26）
ruff check xkep_cae_fluid/ tests/ → All checks passed / ruff format --check → 全ファイル整形済み
python -m pytest tests/ -q -m "not slow and not external" → 本文末尾の「全体テスト」参照
```

## 次にやること

- [ ] `*DARCY` の実行対応（圧力ポアソン型 `DarcyFlowProcess` 新設、または 2D Brinkman への割当）
- [ ] `NaturalConvectionInput.solve_energy` を追加し `HEAT TRANSFER=NONE` でエネルギー方程式をスキップ
- [ ] SYMMETRY / SLIP 面の既定緩和での発散を切り分け（上記）
- [ ] 部分面境界（Brinkman の座標マスク相当）と `InternalFaceBC`（内部セル集合）の .inp 表現
- [ ] 格子の次段: 直交格子 + 幾何解像用の局所格子（FloEFD 方式、界面は近似方程式）か非構造格子か。
  どちらでも `CaseDefinition` は据え置き、`StructuredGridRecoveryProcess` の代替とマッピングを足す
- [ ] OpenFOAM（`blockMesh`/`0`/`constant`/`system` 書き出し）・Fluent 向けの書き出し Process

## ファイル

- 追加: `xkep_cae_fluid/inp/{__init__,__main__,parameters,parser,case,builder,grid,mapping,output,runner,cli}.py`
- 追加: `tests/test_inp_{parser,grid,mapping,runner}.py`
- 追加: `examples/inp/{cavity-nc-1,plate-ht-1,plate-mesh}.inp`、`examples/inp/results/*.{yaml,log}`
- 追加: `docs/design/inp-format.md`、`docs/status/status-33.md`
- 変更: `pyproject.toml`（`[project.scripts] ykep`）、`.gitignore`、`README.md`、`docs/README.md`、`docs/design/README.md`、`docs/roadmap.md`、`docs/status/status-index.md`

## 全体テスト

（下記に実行結果を追記）
