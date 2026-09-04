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

**316 テスト** -- 2026-09-04 Brinkman 流路の[座標マスク境界条件 + 質量流入境界](docs/design/brinkman-flow-fvm.md)（4 辺任意配置、流量固定で inlet 位置・サイズ探索、冷却流路設計の前段 / [status-29](docs/status/status-29.md)）。前: 収束破綻の再現と機構切り分け（[status-28](docs/status/status-28.md)、[nsb/](nsb/README.md)） | 契約違反 **0件**（13プロセス） | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

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
+-- experiments/brinkman_uturn/  # Brinkman U ターン収束性スイープ（sweep.py / diagnose_u2.py / diagnose_local_dtau.py / results / logs）
+-- nsb/               # 手元構成ミラー（core / solver / utils / geo、離散化は brinkman_flow を共有）+ theory.md（数理ノート）+ ルート main.py
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
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |

## インストール

```bash
pip install -e ".[dev]"
```

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
