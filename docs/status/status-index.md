# ステータスインデックス

[<- README](../../README.md)

| # | 日付 | テスト数 | 概要 |
|---|------|---------|------|
| 1 | 2026-03-31 | 16 | 初期移植: xkep-cae から Process Architecture 移植 |
| 2 | 2026-03-31 | 25 | 3次元非定常伝熱解析 (FDM) HeatTransferFDMProcess 実装 |
| 3 | 2026-03-31 | 31 | ソルバー高速化 + 可視化PostProcess + 多層シート温度マップ |
| 4 | 2026-03-31 | 39 | Robin BC + 多層ビルダー + CJK対応 + ミラーリング表示 |
| 5 | 2026-03-31 | 49 | status-4 TODO消化: 非定常Robin BC・冷却フィン・連携例・scipy復旧 |
| 6 | 2026-03-31 | 59 | status-5 TODO消化: 疎行列ソルバー・フィンアレイ2D/3D・CI整備・Phase 2設計 |
| 7 | 2026-03-31 | 88 | status-6 TODO消化: StructuredMeshProcess + PyAMG + Numba JIT |
| 8 | 2026-03-31 | 124 | status-7 TODO全消化: 離散化スキーム + MeshData対応 + polyMesh読込 |
| 9 | 2026-04-01 | 138 | 3D自然対流ソルバー (SIMPLE法 + Boussinesq) + 固体-流体練成 |
| 10 | 2026-04-01 | 176 | status-9 TODO全消化: TVD/Rhie-Chow/非直交補正/AMGキャッシュ/バイナリpolyMesh |
| 11 | 2026-04-01 | 180 | 自然対流調査: q_vol追加 + パラメトリックスタディ(24ケース) + 設計指針 |
| 12 | 2026-04-01 | 180 | ソ��バー安定性改善: adaptive dt + RC energy + 収束判定修正 |
| 13 | 2026-04-02 | 197 | SIMPLEC連成 + BDF2���間積分 + Poiseuille検証 + 保守的緩和テスト |
| 14 | 2026-04-02 | 214 | PISO連成 + TVD対流スキーム統合 + 対流流出BC |
| 15 | 2026-04-02 | 214 | 空気実物性 収束評価 + PISO速度緩和修正 |
| 16 | 2026-04-02 | 221 | AMG圧力ソルバー + 面ベース質量残差修正 + 適応的緩和 |
| 17 | 2026-04-03 | 224 | CG+AMG圧力ソルバー + Ra=1e4ベンチマーク修正 + 長時間安定性検証 |
| 18 | 2026-04-07 | 224 | 1D過渡ジュール電熱 Gauss-Seidel 検算スクリプト |
| 19 | 2026-04-08 | 224 | 1D FDMソルバー 輻射実装 + 断熱BC修正 + LineAreaヘルパー |
| 20 | 2026-04-20 | 224 | 持続的水槽設計CAE Phase 6 ロードマップ策定（90×30×45 水草水槽） |
| 21 | 2026-04-20 | 232 | 汎用スカラー輸送 `ScalarTransportProcess` 新設（Phase 6.1a） |
| 22 | 2026-04-20 | 240 | NaturalConvection に `extra_scalars` を統合（Phase 6.1b、温度+トレーサー同時輸送） |
| 23 | 2026-04-21 | 254 | `AquariumGeometryProcess` 新設（Phase 6.2a、90×30×45 cm 水槽 + 底床/ガラス/水マスク） |
| 24 | 2026-04-21 | 267 | `HeaterProcess` + 水槽ヒーター自然対流デモ（Phase 6.2b、Geometry+Heater+NC 3 段連携） |
| 25 | 2026-04-21 | 286 | `AquariumFilterProcess` + `InternalFaceBC`（Phase 6.3a、外部フィルター循環 INLET/OUTLET BC） |
| 26 | 2026-04-23 | 286 | `examples/aquarium_filter_circulation.py`（Phase 6.3b、Geometry+Heater+Filter+NC 4 段連携デモ） |
| 27 | 2026-09-02 | 286 | 単軸押出解析 設計策定（展開チャネル 2.5D、RTD 目的、実装未着手） |
| 28 | 2026-09-03 | 438 | 単軸押出解析 Phase 1/1.5 実装（`extruder/` 6 プロセス、ゲート G1〜G4 全通過、OpenFOAM G3 検算） |
| 29 | 2026-09-04 | 460 | ゲート G5 文献 RTD 照合（Pinto–Tadmor 1970 再導出、浅溝極限で収束。Phase 2 前提を実機データから差し替え） |
| 30 | 2026-09-04 | 502 | 2D Brinkman 補正 NS (FVM, Newton–Krylov) 新設 + U ターン/平板 収束破綻の再現実験と機構切り分け（局所/大域 Δτ 比較、手元構成ミラー `nsb/` を含む） |
| 31 | 2026-09-04 | 516 | Brinkman 流路の座標マスク境界条件 + 質量流入 + 領域内マニホールド + 位置・径の随伴設計感度（冷却流路設計の前段） |
| 32 | 2026-09-04 | 531 | nsb を xkep_cae_fluid から切り離し（`data`/`assembly` のコピー方式 + 同期スクリプト）+ 高速化見積り実測（LU 分解が 70〜81%）+ PARDISO 化（後方互換なし、分解/三角解のスレッド分割、`KMP_BLOCKTIME=0`）+ 前処理 LU の遅延更新（144×96: 40 s → 17 s） |
| 33 | 2026-09-05 | 598 | ykep .inp 入力フォーマット（Abaqus 風キーワード構文、`*PARAMETER`/`*CONTROLS`/`*GRID`、`CaseDefinition` 中立表現、`ykep -j=<job>.inp int` CLI）+ NS/伝熱マッピング + 例題 2 本（Ra=1000 キャビティ Nu=1.169） |
| 34 | 2026-09-05 | 612 | 3D レンダリング: messi mirador 連携 `MiradorExportProcess`（断面スラブ + 速度矢印、`FORMAT=HTML` / `ykep view`）+ messi v0.10.0（要素場カラーマップ・矢印・`.vtk` リーダ） |
