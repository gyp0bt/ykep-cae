# xkep-cae-fluid ドキュメント

[<- README](../README.md)

## ドキュメント一覧

| ドキュメント | 内容 |
|------------|------|
| [ロードマップ](roadmap.md) | 全体計画・マイルストーン・TODO |
| [水槽設計ロードマップ](roadmap-aquarium.md) | Phase 6 持続的水槽設計 CAE 詳細計画 |
| [設計文書一覧](design/README.md) | 設計仕様書リンク集（コロケーション方式） |
| [ステータス一覧](status/status-index.md) | 全statusファイル + テスト数推移 |

## xkep-cae との共通設計

本リポジトリは [xkep-cae](https://github.com/gyp0bt/xkep-cae)（FEMソルバー基盤）と
同一の **Process Architecture** を採用している。

### 共通部分（二重管理）

| モジュール | 概要 |
|-----------|------|
| `core/base.py` | AbstractProcess + ProcessMeta + ProcessMetaclass |
| `core/registry.py` | ProcessRegistry（プロセス登録・検索・依存グラフ） |
| `core/slots.py` | StrategySlot（動的 Strategy 注入） |
| `core/diagnostics.py` | 実行ログ + 呼び出し元追跡 + レポート生成 |
| `core/tree.py` | ProcessTree（依存関係グラフ + Mermaid出力） |
| `core/runner.py` | ProcessRunner（依存チェック + checksum + プロファイル） |
| `core/benchmark.py` | BenchmarkRunnerProcess（STA2防止マニフェスト記録） |
| `core/testing.py` | binds_to（プロセス-テスト 1:1 紐付け） |
| `core/categories.py` | PreProcess / SolverProcess / PostProcess / VerifyProcess / BatchProcess |

### ドメイン固有部分

| モジュール | xkep-cae（FEM） | xkep-cae-fluid（FDM/FVM） |
|-----------|-----------------|--------------------------|
| `core/data.py` | ContactSetupData, AssembleCallbacks 等 | FlowFieldData, FluidProperties 等 |
| `core/strategies/` | PenaltyStrategy, FrictionStrategy 等 | ConvectionSchemeStrategy, TurbulenceModelStrategy 等 |

## コロケーション方式

設計文書は実装コードのそばに配置する。

```
xkep_cae_fluid/
+-- some_module/
    +-- strategy.py
    +-- process.py
    +-- docs/
        +-- design.md      <- この strategy/process の設計文書
```

`docs/design/README.md` が全設計文書へのリンク集として機能する。
