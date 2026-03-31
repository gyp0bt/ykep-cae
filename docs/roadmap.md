# xkep-cae-fluid ロードマップ

[<- README](../README.md)

## 目標

FDM/FVM による非圧縮性 Navier-Stokes ソルバーを Process Architecture 上に構築する。

## Phase 1: 基盤移植（完了）

- [x] Process Architecture 移植（AbstractProcess, Registry, StrategySlot, Diagnostics）
- [x] 流体向け Strategy Protocol 定義（Convection, Diffusion, Turbulence, PV-Coupling）
- [x] 流体向けデータスキーマ（MeshData, FlowFieldData, SolverInputData）
- [x] 契約検証スクリプト（validate_process_contracts.py）
- [x] CLAUDE.md / README.md / pyproject.toml
- [x] 初期テストスイート

## Phase 1.5: 伝熱解析FDM（完了）

- [x] 3次元非定常伝熱解析 HeatTransferFDMProcess（ガウスザイデル法）
- [x] 伝熱データスキーマ（HeatTransferInput / HeatTransferResult）
- [x] 境界条件（Dirichlet / Neumann / Adiabatic）
- [x] 陰的オイラー法による時間積分
- [x] 面間熱伝導率の調和平均
- [x] 不均一材料分布対応
- [x] APIテスト + 物理テスト（解析解比較5ケース）
- [x] NumPy ベクトル化ヤコビ法ソルバー（高速化）
- [x] TemperatureMapProcess（温度マップ可視化 PostProcess）
- [x] 4層多層シート温度マップ例題
- [x] Robin境界条件（対流熱伝達 h(T∞-T)）
- [x] MultilayerBuilderProcess（多層シート物性値ビルダー）
- [x] CJK日本語フォント自動検出・設定
- [x] 対称ミラーリング表示（1/8対称→全体展開）
- [x] 非定常Robin BC物理テスト（冷却漸近 + エネルギー収支）
- [x] 冷却フィンベンチマーク（解析解比較、温度分布+底端熱流束）
- [x] MultilayerBuilder + HeatTransferFDM + Robin BC 連携例
- [x] SciPy 疎行列ソルバー（直接解法 SuperLU / ILU前処理付き BiCGSTAB）
- [x] 冷却フィンアレイ 2D/3D 拡張テスト（断面メッシュ・メッシュ収束性）
- [x] GitHub Actions CI ワークフロー（lint/test/契約検証、Python 3.10-3.12）

## Phase 2: メッシュ・離散化（予定）

### 設計方針

Phase 1.5 の等間隔直交格子を一般化し、不等間隔格子および非構造化メッシュへ拡張する。
既存の `HeatTransferInput.dx/dy/dz` に依存する離散化を、メッシュオブジェクト経由で
セル体積・面面積・面法線を取得する形に抽象化する。

- **StructuredMeshProcess**: `nx, ny, nz` + 各方向の分割比率から不等間隔直交格子を生成
- **UnstructuredMeshReaderProcess**: OpenFOAM の `polyMesh/` ディレクトリを読み込み
- **MeshData**: セル中心座標、面積ベクトル、セル体積、隣接関係を保持する共通データ構造
- 離散化スキームは **Strategy Pattern** で実装（ConvectionSchemeStrategy, DiffusionSchemeStrategy）
- 既存の伝熱ソルバーは Phase 2 完了後にメッシュ依存部分をリファクタリング

### タスク

- [ ] MeshData スキーマ設計（セル中心、面面積、体積、隣接行列）
- [ ] StructuredMeshProcess 実装（不等間隔直交格子）
- [ ] 非構造化メッシュ読み込み Process（OpenFOAM互換）
- [ ] 中心差分拡散スキーム実装
- [ ] 1次風上対流スキーム実装
- [ ] TVD対流スキーム実装（van Leer, Superbee）

## Phase 3: SIMPLE ソルバー（予定）

- [ ] 運動量方程式アセンブリ Process
- [ ] SIMPLE 圧力-速度連成 Process
- [ ] 線形ソルバー Strategy（直接法 + 反復法）
- [ ] Lid-driven cavity ベンチマーク
- [ ] Poiseuille 流れ検証

## Phase 4: 時間進行（予定）

- [ ] 陰的オイラー時間積分
- [ ] BDF2 時間積分
- [ ] 非定常キャビティ流れ検証
- [ ] カルマン渦列（円柱まわり流れ）

## Phase 5: 乱流モデル（予定）

- [ ] 標準 k-epsilon モデル
- [ ] k-omega SST モデル
- [ ] 壁関数
- [ ] 乱流チャネル流れ検証

## 将来構想

- LES / DES
- 多相流（VOF）
- 伝熱ソルバー高速化（Numba JIT / PyAMG マルチグリッド）
- 適応格子細分化（AMR）

---

## 推奨ソルバー構成（初期）

- **対流スキーム**: TVD (van Leer) — 安定性と精度のバランス
- **拡散スキーム**: 中心差分 + 非直交補正
- **圧力-速度連成**: SIMPLE
- **線形ソルバー**: 圧力=AMG, 速度=BiCGSTAB+ILU
- **時間積分**: 1次陰的オイラー（初期）→ BDF2
