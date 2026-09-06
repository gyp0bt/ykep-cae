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
| 34 | 2026-09-05 | 615 | 3D レンダリング: messi mirador 連携 `MiradorExportProcess`（断面スラブ + 速度矢印 + 残差マップ + 任意平面の view cut `--cut`、`FORMAT=HTML` 自動出力 / `ykep view`）+ messi v0.10.0（要素場カラーマップ Abaqus レインボー・矢印・`.vtk` リーダ・操作パネル畳み込み・断面 view cut） |
| 35 | 2026-09-06 | 775 | ソルバー体験の現在地整理 + Phase 11 残 TODO 消化（圧力補正の非直交補正・適応緩和の共有規則・Rhie–Chow の緩和非依存化・2 次要素/角錐/バッフルの `InpMeshProcess`・`solve_energy`・CI skip） |
| 36 | 2026-09-06 | 826 | 汎用記法（.inp）で押出級の流れを書く（Phase 12）: 周期境界 `*BOUNDARY, TYPE=PERIODIC`（`MeshData.face_offset` + fvm 層の `neighbour_centers`）+ 一様体積力 `*DLOAD, BX/BY/BZ/BF` + 非ニュートン粘度 `*VISCOSITY, TYPE=POWER LAW｜CARREAU`（`fvm/viscosity.py`、非構造 γ̇ の Picard）+ 回転壁 `*ORIENTATION`/`*MPC`/自由度 4-6 + Stokes `CONVECTION=NONE` と連成 `PRESSURE_VELOCITY=COUPLED` + `ExtruderChannelInpProcess` と例題 extruder-channel-1（専用 2.5D ソルバーと Q が機械精度一致、Q_axial 1.4e-3） |
| 37 | 2026-09-06 | 838 | 非構造メッシュの粒子追跡と滞留時間分布（Phase 12 完了）: 面流束から再構成したセル内アフィン場（直交六面体で Pollock、四面体で RT0）を辿る `ParticleTrackFVMProcess` + `ResidenceTimeProcess`、NS 結果に γ̇ と混合指数 λ を常時追加、厳密関係 ⟨t⟩ = length·V/Σw が周期 Poiseuille で 1e-12 一致、構造格子トラッカーと t_p10/p50/p90 が 1e-3 台で一致。**後方互換を全撤去**（RegistryProxy / deprecated 機構 / 再輸出シム）+ **全件テスト 14 分 26 秒 → 2 分 28 秒**（流れ場の共有・6 件を slow へ・pytest-xdist） |
