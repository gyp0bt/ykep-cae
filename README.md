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

**221 テスト** -- 2026-04-02 AMG圧力ソルバー + 面ベース質量残差修正 + 適応的緩和 | 契約違反 **0件**（6プロセス） | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

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
+-- tests/             # テスト
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ドキュメント総覧](docs/README.md) | ドキュメント一覧 + xkep-cae との関係 |
| [Process Architecture](docs/process-architecture.md) | 共通アーキテクチャ設計仕様 |
| [データスキーマ](docs/data-schemas.md) | MeshData / FlowFieldData 等の仕様 |
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
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
