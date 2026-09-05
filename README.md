# xkep-cae-fluid

FDM（差分法）・FVM（有限体積法）による流体ソルバー基盤。
[xkep-cae](https://github.com/gyp0bt/xkep-cae) と共通の Process Architecture を基盤とし、
流体解析特有の離散化・乱流モデル・圧力-速度連成をモジュール化する。

## xkep-cae との関係

| 項目 | xkep-cae | xkep-cae-fluid |
|------|----------|----------------|
| 手法 | FEM（有限要素法） | FDM / FVM |
| 対象 | 構造解析（撚線曲げ揺動等） | 流体解析（非圧縮性NS等） |
| 共通基盤 | Process Architecture | Process Architecture |
| Strategy | Penalty, Friction, ContactForce | Convection, Turbulence, PV-Coupling |

## 現在の状態

**612 テスト** -- 2026-09-05 3D レンダリング: [messi mirador 連携](docs/design/mirador-export.md)（`MiradorExportProcess`、構造格子 → C3D8 + 断面スラブ elset + セル場、速度矢印、`*OUTPUT, FIELD, FORMAT=HTML` / `ykep -j=<job> view --slice=x=0.05`。messi 側は v0.10.0 で要素場カラーマップ・矢印・`.vtk` リーダを追加 / [status-34](docs/status/status-34.md)）。前: ykep .inp 入力フォーマット（[Abaqus 風キーワード構文](docs/design/inp-format.md)、`*PARAMETER`/`*CONTROLS`/`*GRID`、中立表現 `CaseDefinition`、`ykep -j=<job>.inp int` コマンド、`*NAVIER STOKES` → NaturalConvectionFDM / `*HEAT TRANSFER` → HeatTransferFDM、NPZ/YAML/VTK 出力。例題 Ra=1000 キャビティは 226 反復で収束し Nu=1.169 / [status-33](docs/status/status-33.md)）。前: nsb を xkep_cae_fluid から切り離し（[`data`/`assembly` のコピー方式 + 同期スクリプト](nsb/README.md)、numpy/scipy/pypardiso だけで単体持ち出し可）+ 高速化見積り実測（LU 分解が 70〜81%、for ループ削減は 0%、JAX は autodiff 目的のみ）+ 疎 LU を PARDISO 前提に + 前処理 LU の遅延更新（144×96 で 40 s → 17 s、MKL スレッド分割が鍵 / [status-32](docs/status/status-32.md)）。前: Brinkman 流路の[座標マスク境界条件 + 質量流入 + 領域内マニホールド + 随伴設計感度](docs/design/brinkman-flow-fvm.md)（4 辺任意配置、流量固定で inlet 探索、紙面垂直方向のヘッダ、位置・径の勾配を陰関数定理で、冷却流路設計の前段 / [status-31](docs/status/status-31.md)）。前: 収束破綻の再現と機構切り分け（[status-30](docs/status/status-30.md)、[nsb/](nsb/README.md)） | 契約違反 **0件**（27プロセス） | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

前: [単軸押出解析 Phase 1/1.5 + G5 文献照合](docs/design/single-screw-extruder.md)（展開チャネル 2.5D、ゲート G1〜G5 全通過、OpenFOAM 検算・Pinto–Tadmor RTD 照合済み / [status-29](docs/status/status-29.md) / [図解レポート](docs/reports/extruder/README.md)）

## パッケージ構成

```
xkep_cae_fluid/
+-- core/              # プロセスアーキテクチャ基盤（xkep-cae共通設計）
|   +-- base.py        # AbstractProcess + ProcessMeta + ProcessMetaclass
|   +-- registry.py    # ProcessRegistry
|   +-- slots.py       # StrategySlot
|   +-- categories.py  # PreProcess / SolverProcess / PostProcess / VerifyProcess / BatchProcess
|   +-- data.py        # MeshData / FlowFieldData / SolverInputData / SolverResultData
|   +-- mesh.py        # StructuredMeshProcess（不等間隔直交格子生成）
|   +-- mesh_reader.py # PolyMeshReaderProcess（OpenFOAM polyMesh 読込）
|   +-- runner.py      # ProcessRunner
|   +-- diagnostics.py # 実行診断
|   +-- benchmark.py   # BenchmarkRunnerProcess
|   +-- tree.py        # ProcessTree（依存グラフ）
|   +-- testing.py     # binds_to（テスト紐付け）
|   +-- strategies/    # Strategy Protocol 定義 + 具象スキーム（拡散/対流/TVD/非直交補正）
|   +-- docs/          # コアモジュール設計文書
+-- natural_convection/ # 3次元自然対流解析 (FDM + SIMPLE法)
|   +-- data.py        # NaturalConvectionInput / Result / FluidBoundarySpec
|   +-- assembly.py    # 疎行列アセンブリ（運動量・圧力補正・エネルギー）
|   +-- solver.py      # NaturalConvectionFDMProcess (SIMPLE/SIMPLEC/PISO + TVD + BDF2)
+-- scalar_transport/  # 汎用スカラー輸送 (Phase 6.1a 水槽CAE基盤)
|   +-- data.py        # ScalarFieldSpec / ScalarBoundarySpec / Input / Result
|   +-- assembly.py    # 疎行列アセンブリ（対流-拡散-ソース、Dirichlet/Neumann/Robin BC）
|   +-- solver.py      # ScalarTransportProcess (陰的Euler + BiCGSTAB+ILU)
+-- brinkman_flow/     # 2D Brinkman 補正 Navier-Stokes (FVM, Newton–Krylov) — 収束破綻の再現実験
|   +-- data.py        # BrinkmanFlowInput / Result / SolverSettings / ThicknessSpec / BoundaryPatch（座標マスク・質量流入）
|   +-- geometry.py    # UTurnThicknessProcess（flat / uturn 厚さ場）
|   +-- assembly.py    # 同位置 FVM 残差（1次/2次風上+Venkatakrishnan）+ 1次風上ヤコビアン + Rhie-Chow + 4 辺の座標マスク境界
|   +-- solver.py      # BrinkmanFlowFVMProcess（Newton + GMRES/LU(J1) + 擬似時間 + 陰的緩和）
+-- aquarium/          # 水槽設計 CAE ドメイン（Phase 6.2 / 6.3）
|   +-- geometry.py    # AquariumGeometryProcess（90×30×45 cm + 底床/ガラス/水マスク + z-refinement）
|   +-- heater.py      # HeaterProcess（定熱流束 + 定温ヒステリシス）
|   +-- filter.py      # AquariumFilterProcess + InternalFaceBC（外部フィルター循環, Q[L/h]）
+-- extruder/          # 単軸押出 2.5D 断面解析（Phase 7、status-28/29）
|   +-- geometry.py    # ScrewGeometryProcess（展開チャネル + 隙間の等比格子）
|   +-- shape_factors.py  # 形状係数 Fd/Fp の級数解（ゲート G1/G2 の真値）
|   +-- down_channel.py   # DownChannelFlowProcess（w: 可変係数 Poisson）
|   +-- cross_channel.py  # CrossChannelStokesProcess（u,v,p: MAC Stokes 鞍点系）
|   +-- viscosity.py   # Newtonian / PowerLaw / Carreau Strategy + Green-Gauss γ̇
|   +-- solver.py      # ExtruderFlowProcess（Picard 結合、Q_axial = Q + L_turn·Q_leak）
|   +-- tracker.py     # ParticleTrackerProcess（ψ 双一次補間、RK4、ζ 座標）
|   +-- rtd.py         # RTDProcess（流束重み付き RTD、パーセンタイル、累積せん断）
+-- post/              # 後処理: mirador.py = messi mirador 3D レンダリング（MiradorExportProcess、status-34）
+-- inp/               # ykep .inp 入力フォーマット（Abaqus 風キーワード構文、status-33）
|   +-- parameters.py  # *PARAMETER の安全な式評価 + <expr> 置換
|   +-- parser.py      # InpKeywordParseProcess（*INCLUDE / コメント / 継続行 / KeywordBlock 列）
|   +-- case.py        # CaseDefinition（ソルバー非依存の中立表現）
|   +-- builder.py     # InpCaseBuildProcess（意味付け、*GRID 拡張）
|   +-- grid.py        # StructuredGridRecoveryProcess（*NODE/*ELEMENT → 直交構造格子、*SURFACE → 領域面）
|   +-- mapping.py     # InpToNaturalConvectionProcess / InpToHeatTransferProcess（*CONTROLS 含む）
|   +-- output.py      # InpOutputWriterProcess（NPZ / YAML サマリ / VTK）
|   +-- runner.py      # InpCaseRunnerProcess（方程式ファミリーで振り分け）
|   +-- cli.py         # ykep コマンド（ykep -j=<job>.inp int）
+-- heat_transfer/     # 3次元非定常伝熱解析 (FDM)
|   +-- data.py        # HeatTransferInput / HeatTransferResult / BoundarySpec (Robin対応)
|   +-- solver.py      # HeatTransferFDMProcess (ヤコビ/GS/疎行列/AMG/Numba)
|   +-- solver_vectorized.py  # NumPy ベクトル化ヤコビ法
|   +-- solver_sparse.py      # SciPy 疎行列ソルバー (直接解法/BiCGSTAB/AMG)
|   +-- solver_numba.py       # Numba JIT 高速化ガウスザイデル法
|   +-- multilayer.py  # MultilayerBuilderProcess (多層シート物性値ビルダー)
|   +-- visualize.py   # TemperatureMapProcess (温度マップ/CJK/ミラーリング)
+-- examples/          # 実行例
|   +-- multilayer_sheet_temperature.py  # 4層多層シート温度マップ
|   +-- multilayer_robin_analysis.py     # MultilayerBuilder+FDM+Robin BC 連携例
|   +-- benchmark_solver_methods.py      # ソルバー手法別ベンチマーク
|   +-- aquarium_heater_natural_convection.py  # Geometry+Heater+NC 3 段（Phase 6.2b）
|   +-- aquarium_filter_circulation.py         # Geometry+Heater+Filter+NC 4 段（Phase 6.3b）
|   +-- inp/           # .inp 例題（cavity-nc-1: Ra=1000 キャビティ、plate-ht-1: *INCLUDE メッシュの平板伝熱）+ results/
+-- experiments/brinkman_uturn/  # Brinkman U ターン収束性スイープ（sweep.py / diagnose_u2.py / diagnose_local_dtau.py / results / logs）
+-- nsb/               # 手元構成ミラー（core / solver / utils / geo / adjoint + data / assembly のコピー。xkep_cae_fluid 非依存で単体持ち出し可）+ theory.md（数理ノート）+ ルート main.py
+-- experiments/nsb/   # nsb パラメータスタディの results / logs
+-- tests/             # テスト
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ドキュメント総覧](docs/README.md) | ドキュメント一覧 + xkep-cae との関係 |
| [Process Architecture](docs/process-architecture.md) | 共通アーキテクチャ設計仕様 |
| [データスキーマ](docs/data-schemas.md) | MeshData / FlowFieldData 等の仕様 |
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [水槽設計ロードマップ](docs/roadmap-aquarium.md) | Phase 6 持続的水槽設計 CAE 詳細計画 |
| [設計文書一覧](docs/design/README.md) | 設計仕様書リンク集（コロケーション方式） |
| [.inp 入力フォーマット](docs/design/inp-format.md) | Abaqus 風キーワード構文と `ykep -j=<job>.inp int` コマンド |
| [3D レンダリング（messi mirador）](docs/design/mirador-export.md) | 解析結果を messi の three.js ビューアで表示（断面スラブ・速度矢印、`FORMAT=HTML` / `ykep view`） |
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |

## インストール

```bash
pip install -e ".[dev]"
```

## .inp で実行（ykep コマンド）

```bash
ykep -j=examples/inp/cavity-nc-1.inp int            # Abaqus 風: -j=<job>[.inp] と int（対話ログ）
ykep -j=examples/inp/plate-ht-1 int -o=out          # 出力先指定（<job>.npz / .yaml / .vtk / .log）
ykep -j=case.inp --check                            # 解析せず読込・格子復元・マッピングのみ検証
ykep -j=examples/inp/cavity-nc-1 view -o=out --slice=x=0.05   # 解析せず NPZ → <job>.html（messi mirador 3D ビューア）
```

## 3D レンダリング（messi mirador 連携）

[messi](https://github.com/gyp0bt/messi)（v0.10.0 以降）を入れると、構造格子の結果を three.js の
自己完結 HTML に書き出してブラウザで回せる（`*OUTPUT, FIELD, FORMAT=VTK+HTML` か `ykep ... view`、
Python からは `MiradorExportProcess`）。外皮 + 断面スラブ（elset 切替）+ 速度矢印、場ごとのカラーマップ、
probe で値表示。詳細は [設計文書](docs/design/mirador-export.md)。

```bash
pip install -e ../messi     # 任意依存（未導入なら FORMAT=HTML は警告してスキップ）
```

キーワード一覧は [設計文書](docs/design/inp-format.md) を参照。

## テスト実行

```bash
pytest tests/ -v -m "not slow and not external"
```

## Lint / Format

```bash
ruff check xkep_cae_fluid/ tests/
ruff format xkep_cae_fluid/ tests/
```

## ライセンス

[MIT License](LICENSE)

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は [docs/status/](docs/status/status-index.md) を参照。
