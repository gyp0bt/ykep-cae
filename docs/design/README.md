# 設計文書索引

[<- README](../../README.md) | [<- docs](../README.md) | [<- roadmap](../roadmap.md)

> 設計仕様書は実装コードのそばに配置（コロケーション方式）。
> 本ファイルは全設計文書へのリンク集。

## プロセスアーキテクチャ基盤（xkep-cae 共通設計）

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [benchmark_runner.md](../../xkep_cae_fluid/core/docs/benchmark_runner.md) | `xkep_cae_fluid/core/docs/` | BenchmarkRunner マニフェスト自動記録 | 完了 |
| [process_diagnostics.md](../../xkep_cae_fluid/core/docs/process_diagnostics.md) | `xkep_cae_fluid/core/docs/` | Process 実行診断 | 完了 |

## 伝熱モジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [heat-transfer-fdm.md](heat-transfer-fdm.md) | `docs/design/` | 3D FDM 伝熱解析ソルバー（Robin BC対応） | 完了 |
| [temperature-map.md](temperature-map.md) | `docs/design/` | 温度マップ可視化 PostProcess | 完了 |
| [multilayer-builder.md](multilayer-builder.md) | `docs/design/` | 多層シート物性値ビルダー PreProcess | 完了 |

## メッシュモジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [structured-mesh.md](structured-mesh.md) | `docs/design/` | StructuredMeshProcess（不等間隔直交格子） | 完了 |
| [polymesh-reader.md](polymesh-reader.md) | `docs/design/` | PolyMeshReaderProcess（OpenFOAM互換） | 完了 |

## 流体モジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [natural-convection-fdm.md](natural-convection-fdm.md) | `docs/design/` | 3D自然対流ソルバー (SIMPLE法+Boussinesq+練成) | 完了 |
| [scalar-transport-fdm.md](scalar-transport-fdm.md) | `docs/design/` | 汎用スカラー輸送ソルバー (Phase 6.1a 水槽 CAE 基盤) | 完了 |
| (未作成) | - | 乱流モデル Strategy 設計 | 予定 |

---
