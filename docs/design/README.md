# 設計文書索引

[<- README](../../README.md) | [<- docs](../README.md) | [<- roadmap](../roadmap.md)

> 設計仕様書は実装コードのそばに配置（コロケーション方式）。
> 本ファイルは全設計文書へのリンク集。

## プロセスアーキテクチャ基盤（xkep-cae 共通設計）

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [benchmark_runner.md](../../xkep_cae_fluid/core/docs/benchmark_runner.md) | `xkep_cae_fluid/core/docs/` | BenchmarkRunner マニフェスト自動記録 | 完了 |
| [process_diagnostics.md](../../xkep_cae_fluid/core/docs/process_diagnostics.md) | `xkep_cae_fluid/core/docs/` | Process 実行診断 | 完了 |

## 入力フォーマット設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [inp-format.md](inp-format.md) | `docs/design/` | ykep .inp 入力フォーマット（Abaqus 風キーワード構文、`ykep -j=<job>.inp int`、`*PARAMETER`、`*CONTROLS`） | 完了（NS / 伝熱 / Darcy） |
| [unstructured-inp-mesh.md](unstructured-inp-mesh.md) | `docs/design/` | InpMeshProcess（`*NODE/*ELEMENT` → 面ベース非構造 MeshData、六面体 / 楔 / 四面体・2D 四辺形 / 三角形の混在可、`*SURFACE` → 境界パッチ） | 完了（experimental） |

## 伝熱モジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [heat-transfer-fdm.md](heat-transfer-fdm.md) | `docs/design/` | 3D FDM 伝熱解析ソルバー（Robin BC対応） | 完了 |
| [temperature-map.md](temperature-map.md) | `docs/design/` | 温度マップ可視化 PostProcess | 完了 |
| [mirador-export.md](mirador-export.md) | `docs/design/` | 3D レンダリング PostProcess（messi mirador 連携、断面スラブ + 速度矢印、`FORMAT=HTML` / `ykep view`） | 完了（experimental） |
| [multilayer-builder.md](multilayer-builder.md) | `docs/design/` | 多層シート物性値ビルダー PreProcess | 完了 |

## メッシュモジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [structured-mesh.md](structured-mesh.md) | `docs/design/` | StructuredMeshProcess（不等間隔直交格子） | 完了 |
| [polymesh-reader.md](polymesh-reader.md) | `docs/design/` | PolyMeshReaderProcess（OpenFOAM互換） | 完了 |
| [fvm-layer.md](fvm-layer.md) | `docs/design/` | 面ベース FVM 共通低レイヤー `xkep_cae_fluid.fvm`（境界パッチ条件・面演算・非直交 / スキュー補正・風上 + TVD・Euler / BDF2・線形ソルバー Strategy）と 3 層分離の方針 | 完了（experimental） |

## 流体モジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [natural-convection-fdm.md](natural-convection-fdm.md) | `docs/design/` | 3D自然対流ソルバー (SIMPLE法+Boussinesq+練成) | 完了 |
| [scalar-transport-fdm.md](scalar-transport-fdm.md) | `docs/design/` | 汎用スカラー輸送ソルバー (Phase 6.1a 水槽 CAE 基盤) | 完了 |
| [darcy-flow-fvm.md](darcy-flow-fvm.md) | `docs/design/` | DarcyFlowProcess（`*DARCY`、面ベース FVM、非構造メッシュ可、非直交補正、Forchheimer、非定常） | 完了（experimental） |
| [navier-stokes-fvm.md](navier-stokes-fvm.md) | `docs/design/` | NavierStokesFVMProcess（面ベース FVM の非圧縮 NS: SIMPLE/SIMPLEC/PISO/COUPLED + Rhie–Chow、TVD、BDF2、Stokes、体積力、非ニュートン粘度、回転壁、周期境界、Boussinesq、Brinkman 抵抗、エネルギー、追加スカラー、内部吐出・吸入、対流流出、非構造メッシュ可） | 完了（experimental） |
| [heat-transfer-fvm.md](heat-transfer-fvm.md) | `docs/design/` | HeatTransferFVMProcess（面ベース FVM 版の伝熱、構造格子 FDM と一致、`*HEAT TRANSFER` の非構造経路） | 完了（experimental） |
| [brinkman-flow-fvm.md](brinkman-flow-fvm.md) | `docs/design/` | 2D Brinkman 補正 NS (FVM, Newton–Krylov) と U ターン収束性再現実験 | 実験中 |
| (未作成) | - | 乱流モデル Strategy 設計 | 予定 |

## 押出モジュール設計文書

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [inp-generic-extrusion.md](inp-generic-extrusion.md) | `docs/design/` | 汎用記法（.inp）で押出級の流れを書く（周期境界・体積力・非ニュートン粘度・回転壁・Stokes / COUPLED、`ExtruderChannelInpProcess`） | 完了（Phase 12） |
| [single-screw-extruder.md](single-screw-extruder.md) | `docs/design/` | 単軸押出 展開チャネル 2.5D（混練性・RTD） | **設計完了・未実装** |

## 水槽モジュール設計文書（Phase 6）

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [aquarium-geometry.md](aquarium-geometry.md) | `docs/design/` | AquariumGeometryProcess（90×30×45 cm + 底床/ガラス/水マスク） | 完了 |
| [aquarium-heater.md](aquarium-heater.md) | `docs/design/` | HeaterProcess（定熱流束 + 定温ヒステリシス） | 完了 |
| [aquarium-filter.md](aquarium-filter.md) | `docs/design/` | AquariumFilterProcess + InternalFaceBC（外部フィルター循環） | 完了 |

---
