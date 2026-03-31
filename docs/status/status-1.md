# status-1: 初期移植 -- xkep-cae から Process Architecture 移植

[<- status-index](status-index.md) | [<- README](../../README.md)

## 日付

2026-03-31

## 概要

xkep-cae（FEMソルバー基盤）から共通の Process Architecture を xkep-cae-fluid（FDM/FVMソルバー基盤）に移植した。

## 移植内容

### そのまま移植（パッケージ名のみ変更）
- `core/base.py`: AbstractProcess, ProcessMeta, ProcessMetaclass
- `core/registry.py`: ProcessRegistry, RegistryProxy
- `core/slots.py`: StrategySlot
- `core/diagnostics.py`: ProcessExecutionLog, DeprecatedProcessError
- `core/tree.py`: ProcessTree, ProcessNode
- `core/runner.py`: ProcessRunner, ExecutionContext
- `core/testing.py`: binds_to
- `core/categories.py`: PreProcess, SolverProcess, PostProcess, VerifyProcess, BatchProcess
- `core/benchmark.py`: BenchmarkRunnerProcess, RunManifest

### FDM/FVM 向けに適応
- `core/data.py`: MeshData（構造化/非構造化対応）, BoundaryData（パッチ型境界条件）, FluidProperties, FlowFieldData, SolverInputData, SolverResultData
- `core/strategies/protocols.py`: ConvectionSchemeStrategy, DiffusionSchemeStrategy, TimeIntegrationStrategy, TurbulenceModelStrategy, PressureVelocityCouplingStrategy, LinearSolverStrategy

### 新規作成
- `CLAUDE.md`: コーディング規約（xkep-cae準拠 + 流体向け適応）
- `README.md`: プロジェクト概要
- `pyproject.toml`: パッケージ設定
- `docs/roadmap.md`: 開発ロードマップ
- `docs/design/README.md`: 設計文書索引
- `contracts/validate_process_contracts.py`: 契約検証スクリプト

## テスト結果

- テスト数: 0（初期移植段階）
- 契約違反: 0件

## 次のステップ

- Phase 2: メッシュ生成・離散化スキームの具象 Process 実装
- 初期ベンチマーク: Lid-driven cavity
