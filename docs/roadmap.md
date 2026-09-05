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

- [x] MeshData スキーマ設計（セル中心、面面積、体積、隣接行列）— core/data.py 既存 + 面情報充実
- [x] StructuredMeshProcess 実装（不等間隔直交格子）— core/mesh.py
- [x] 非構造化メッシュ読み込み Process（OpenFOAM互換）— core/mesh_reader.py
- [x] 中心差分拡散スキーム実装 — strategies/diffusion.py
- [x] 1次風上対流スキーム実装 — strategies/convection.py
- [x] 既存伝熱ソルバーの MeshData 対応リファクタリング（不等間隔格子対応）
- [x] TVD対流スキーム実装（van Leer, Superbee） — strategies/tvd_convection.py
- [x] 非直交補正付き拡散スキーム実装 — strategies/corrected_diffusion.py
- [x] PolyMeshReader バイナリ形式対応 — core/mesh_reader.py
- [x] PyAMG AMG 構築キャッシュ化 — heat_transfer/solver_sparse.py

## Phase 3: SIMPLE ソルバー（完了）

- [x] 運動量方程式アセンブリ — natural_convection/assembly.py
- [x] SIMPLE 圧力-速度連成 — natural_convection/solver.py (NaturalConvectionFDMProcess)
- [x] Boussinesq 近似による浮力項
- [x] 固体-流体練成伝熱（Conjugate Heat Transfer）
- [x] エネルギー方程式（対流+拡散、固体/流体統一）
- [x] 境界条件（NO_SLIP/SLIP/INLET/OUTLET/SYMMETRY + DIRICHLET/NEUMANN/ADIABATIC）
- [x] 線形ソルバー（BiCGSTAB + ILU前処理 / AMG前処理）
- [x] Rhie-Chow 補間（チェッカーボード圧力抑制） — assembly.py (RC付き圧力補正)
- [x] 差分加熱キャビティベンチマーク（Ra=10³〜10⁴ 定量検証、de Vahl Davis比較）
- [x] 体積熱生成項（q_vol）追加 — data.py / assembly.py
- [x] 自然対流パラメトリックスタディ（密閉/半開放/3辺開放 BC比較）
- [x] Poiseuille 流れ検証（チャネル流プロファイル・横方向速度・収束テスト）
- [x] SIMPLEC 圧力-速度連成（Van Doormaal-Raithby, alpha_p=1.0 自動適用）
- [x] PISO 圧力-速度連成（Issa 1986, 複数回圧力補正, 速度緩和不要）
- [x] TVD 対流スキーム統合（van Leer / Superbee, 遅延補正法）
- [x] 対流流出境界条件 OUTLET_CONVECTIVE（非反射出口BC）
- [x] AMG 圧力ソルバー（PyAMG Ruge-Stuben + BiCGSTAB、キャッシュ付き）
- [x] 面ベース質量残差（Rhie-Chow 面速度と整合的な収束判定）
- [x] 適応的緩和係数（残差減少率に応じた alpha_u/alpha_p 自動調整）
- [x] 初期残差方式の収束判定（OpenFOAM 方式）

## Phase 1.6: 1D電熱FDMスクリプト（進行中）

- [x] 1D過渡ジュール電熱 Gauss-Seidel ソルバー（scripts/fdm_gs_1d_v3.py）
- [x] 制御体積法 + 陰的Euler + 直列抵抗モデル面コンダクタンス
- [x] 温度依存電気抵抗率 + 対流放熱
- [x] 断熱境界条件バグ修正（a_P_bnd 誤加算）
- [x] 輻射ソース項実装（ε·σ·(T⁴-T_env⁴) 線形化）
- [x] LineArea dataclass + solve_from_line_areas ヘルパー
- [ ] 輻射物理テスト（解析解比較）
- [ ] HeatTransferFDMProcess (3D) への輻射境界条件統合

## Phase 4: 時間進行（進行中）

- [x] 陰的オイラー時間積分（natural_convection 非定常モード）
- [x] 非定常自然対流テスト（温度発展・浮力onset・残差履歴）
- [x] BDF2 時間積分（2次精度、初回自動Eulerフォールバック）
- [x] 長時間安定性検証（空気実物性 t=5.0s、NaN なし）
- [x] Ra=10⁴ベンチマーク修正（偽時間進行法で定常解を取得）
- [x] CG+AMG圧力ソルバー（対称正定値ラプラシアンに最適）
- [ ] 非定常キャビティ流れ検証
- [ ] カルマン渦列（円柱まわり流れ）
- [ ] 偽時間進行の定常ソルバー内蔵化
- [ ] 陰的対流スキーム（CFL制約の緩和）

## Phase 5: 乱流モデル（予定）

- [ ] 標準 k-epsilon モデル
- [ ] k-omega SST モデル
- [ ] 壁関数
- [ ] 乱流チャネル流れ検証

## Phase 6: 持続的水槽設計 CAE システム（策定済、着手予定）

90×30×45 cm 水草水槽を題材に、スカラー輸送（CO2/O2）・多孔質媒体（底床/ろ材）・
機器モデル（ヒーター/ライト/外部フィルター/CO2 添加）・生体反応（光合成/呼吸）・
ガス交換界面を統合する。詳細は [水槽設計ロードマップ](roadmap-aquarium.md) を参照。

- [ ] Phase 6.0: SIMPLEC/PISO mass 残差改善（並行別 PR）
- [x] Phase 6.1a: 汎用スカラー輸送 `ScalarTransportProcess` 新設（status-21）
- [ ] Phase 6.1b: NaturalConvection 統合 `extra_scalars`（1 PR）
- [ ] Phase 6.2: 水槽ジオメトリ + ヒーター最小デモ（2 PR）← 初デモ
- [ ] Phase 6.3: 外部フィルター `InternalFaceBC`（2 PR）
- [ ] Phase 6.4: 多孔質媒体 Darcy-Forchheimer（2 PR）
- [ ] Phase 6.5: 植物ライト Beer-Lambert（1 PR）
- [ ] Phase 6.6: 生体反応（光合成/呼吸）（2 PR）
- [ ] Phase 6.7: ガス交換界面 + CO2 気泡プルーム（2 PR）
- [ ] Phase 6.8: 水槽システム統合デモ（1〜2 PR）
- [ ] Phase 6.9: 日周期拡張 + 設計最適化の足場（将来）

## Phase 7: 単軸押出解析（Phase 1 / 1.5 完了）

螺旋対称性で 2.5D に落とした計量部の断面解析と RTD。設計は
[docs/design/single-screw-extruder.md](design/single-screw-extruder.md)、
実装計画は [docs/plans/2026-09-02-single-screw-extruder-impl.md](plans/2026-09-02-single-screw-extruder-impl.md)。

- [x] 設計策定（status-27）
- [x] Phase 1: 幾何 → 形状係数 Fd/Fp → 下流 Poisson → 粘度 Strategy → 断面内 Stokes → Picard 結合（G1/G2/G2b、status-28）
- [x] G3: OpenFOAM 同一格子検算（ニュートン・べき乗則、[レポート](reports/extruder/g3-openfoam.md)）
- [x] Phase 1.5: 粒子追跡 + RTD（G4a/G4b、status-28）
- [x] 図解レポート 2 本を Artifact 公開（[reports/extruder](reports/extruder/README.md)）
- [x] 文献 RTD 照合 G5（Phase 2 の前提。実機データが無いので Pinto–Tadmor 1970 との照合に差し替え、2026-09-04 通過）
- [ ] Phase 2: 粘性発熱 `Φ = μγ̇²` + 温度依存粘度
- [ ] Phase 3: 混練エレメント（3D、messi + OpenFOAM）

## Phase 8: 2D Brinkman 補正 NS (FVM, Newton–Krylov) 収束性研究 → 冷却流路設計（進行中）

薄流路の深さ平均 2D 流れ（Brinkman 貫通項）を同位置 FVM + Newton–Krylov で定常解析し、
メッシュ細分化・流速増加で収束が破綻する現象を再現・分析する（学習目的の実験）。
詳細は [設計文書](design/brinkman-flow-fvm.md) と [status-30](status/status-30.md)。

- [x] `BrinkmanFlowFVMProcess`（Newton + GMRES/LU(J1)、JFNK / defect correction、擬似時間 SER、陰的緩和）
- [x] `UTurnThicknessProcess`（flat / uturn 厚さ場）
- [x] 1 次風上ヤコビアンの FD 検証テスト、質量保存・Hele-Shaw 圧損の物理テスト
- [x] 再現スイープ（72×48 の 1×/2×/4× × U=0.1/1/2 × flat/uturn）— status-30
- [x] U=2 失敗機構切り分け（CFL 初期値 / ラインサーチ / 1 次風上 / defect correction / 継続法）
- [x] 局所 Δτ vs 大域 Δτ、速度下限の有無、RC 係数への擬似時間項混入の切り分け（`pseudo_time_mode`, `rhie_chow_pseudo_time`）— status-30
- [x] 手元構成ミラー `nsb/`（core/solver/utils/geo + main.py）で「速度下限なし・擬似時間項を残差に含む」構成の停滞を再現 — status-30
- [x] 発散対策の本質的検討（速度下限 / Stokes 初期場 / α_u=1、backtracking は効かず）— status-30
- [x] 座標マスク境界条件 + 質量流入境界（4 辺任意配置、流量固定で inlet 探索）— status-31
- [x] 領域内マニホールド（注入 / 流量指定吸出 / 圧力指定ヘッダ、連続式ソース）— status-31
- [x] 領域内マニホールドの位置・径を連続設計変数に（滑らかな窓 + 随伴 VJP `nsb/adjoint.py`）— status-31
- [x] `nsb/` を xkep_cae_fluid 非依存に（`data`/`assembly` コピー + `scripts/sync_nsb_from_xkep.py` + 乖離テスト）— status-32
- [x] nsb 高速化の効果見積り（実測: LU 分解 70〜81% → pypardiso 2.5〜4×、for ループ削減 0%、JAX は autodiff 目的のみ）— status-32
- [x] nsb の疎 LU を PARDISO（pypardiso）前提に + 前処理 LU の遅延更新 `precond_lag`（144×96: 40 s → 17 s、MKL スレッド分割）— status-32
- [ ] 18 コア実機で再計測して `precond_lag` 既定を確定、GMRES 反復数の削減 — status-32 TODO
- [ ] 境界 inlet の位置・幅の連続化、冷却設計向け目的関数 — status-31 TODO
- [ ] 熱ソルバー連携（流量場 → 熱伝達コンダクタンス → 上下プレート温度）
- [ ] 非定常（時間精度）モードで定常解の存在を確認

## Phase 9: ykep .inp 入力フォーマット（Abaqus 風キーワード構文）（着手、status-33）

解くべき問題を 1 ファイルで完全記述し `ykep -j=<job>.inp int` で実行する。中立表現 `CaseDefinition` を挟み、
将来 OpenFOAM / Fluent へ同じ .inp から書き出す。詳細は [設計文書](design/inp-format.md)。

- [x] キーワードトークナイザ（`*INCLUDE`、`*PARAMETER` の安全評価と `<expr>` 置換、継続行）— status-33
- [x] `CaseDefinition`（節点・要素・集合・面・材料・セクション・初期条件・ステップ・`*CONTROLS`・`*OUTPUT`）— status-33
- [x] `*NODE/*ELEMENT`（C3D8 / CPS4 系）と `*GRID` からの直交構造格子復元、`*SURFACE` → 領域 6 面 — status-33
- [x] `*NAVIER STOKES`（層流、定常/非定常、等温/伝熱連成）→ NaturalConvectionFDM、`*HEAT TRANSFER` → HeatTransferFDM — status-33
- [x] `ykep` CLI（`-j=`, `int`, `-o=`, `-p name=value`, `--check`）+ NPZ/YAML/VTK 出力 — status-33
- [x] 例題: Ra=1000 キャビティ（Nu=1.169）、`*INCLUDE` メッシュの平板伝熱 — status-33
- [x] `*DARCY` の実行対応（`DarcyFlowProcess` 新設、面ベース FVM、`InpMeshProcess` の非構造メッシュ経由）— Phase 11
- [ ] `HEAT TRANSFER=NONE` でエネルギー方程式をスキップ（`solve_energy`）— status-33 TODO
- [ ] SYMMETRY / SLIP 面の既定緩和での発散の切り分け — status-33 TODO
- [ ] 部分面境界・`InternalFaceBC` の .inp 表現 — status-33 TODO
- [x] 格子の次段の方針決定 → 非構造格子（面ベース FVM）。`InpMeshProcess` を追加 — Phase 11
- [ ] OpenFOAM / Fluent 書き出し Process

## Phase 10: 3D レンダリング（messi mirador 連携）（status-34）

構造格子の解析結果（U / P / T / 追加スカラー）を [messi](https://github.com/gyp0bt/messi) の three.js
ビューア mirador で回して眺める。詳細は [設計文書](design/mirador-export.md)。

- [x] `MiradorExportProcess`（格子 → C3D8 + 断面スラブ elset + セル場、`hidden_groups` で外皮を初期非表示）— status-34
- [x] messi 側: `export_html` に要素場カラーマップ・矢印・`init_mode`・`hidden_groups`、`.vtk` legacy リーダ（messi v0.10.0）— status-34
- [x] `*OUTPUT, FIELD, FORMAT=HTML` と `ykep -j=<job> view [--slice=x=0.05]`（NPZ から後追い生成）— status-34
- [x] 残差マップ `res_*`、`FORMAT=` 未指定時の HTML 自動出力、Abaqus レインボー、操作パネルの畳み込み — status-34 追記
- [ ] 残差マップの対数スケール表示、`ykep view --colormap/--init-mode`、過渡伝熱の `res_T`（status-34「次にやること」）
- [x] 任意平面の断面（view cut）: messi mirador の「断面」（クリップ + 切り口をセル値で着色）、`cut_plane` / `ykep view --cut` — status-34 追記 6
- [ ] view cut の節点補間・複数平面、時系列（`T_history`）のフレーム切替
- [x] 非構造格子を `MeshData.connectivity` から同じ経路で載せる（`MiradorExportInput.mesh`、`*DARCY` の出力と `ykep view`）— Phase 11
- [ ] 水槽 CAE（Phase 6）の `AquariumGeometryProcess` マスクと連携した実例（水・ガラス・底床の elset 分け）

## Phase 11: ソルバー層の分離（GeoProcess / FVM 低レイヤー / 方程式ファミリー）と非構造格子（着手）

実験ソルバー群（`nsb/`、`experiments/coldplate`）は実験側に残し、本体を「幾何・境界パッチ（Geo 層）」
「面ベース FVM 共通低レイヤー」「薄い方程式ファミリー」の 3 層に分ける。棚卸しと計画は
[plans/2026-09-05-solver-layering.md](plans/2026-09-05-solver-layering.md)、層の設計は
[design/fvm-layer.md](design/fvm-layer.md)。

- [x] 棚卸し: 本体 4 ソルバー・実験群・幾何/BC 層の格子表現・境界条件・線形解法・複製箇所の一覧
- [x] `MeshData` に境界面・パッチ・セル種別（構造格子 6 パッチ、polyMesh の節点順序付き接続）
- [x] `xkep_cae_fluid.fvm`（PatchBC / resolve_boundary、面演算、拡散・風上・時間項・ソース、Direct / BiCGSTAB / AMG Strategy）
- [x] `ScalarTransportFVMProcess`（パイロット。構造格子 FDM と 1e-8 で一致）
- [x] `InpMeshProcess`（`.inp` → 面ベース非構造メッシュ、`*SURFACE` → パッチ）
- [x] `DarcyFlowProcess` + `*DARCY`（`InpToDarcyProcess`、非構造 NPZ / VTK / HTML 出力、例題 darcy-1）
- [x] `NaturalConvectionFDMProcess` の過渡 dt 差し替えで `internal_face_bcs` が落ちる回帰を修正
- [x] `HeatTransferFVMProcess`（面カーネル版、FDM と 1e-8 一致）+ `*HEAT TRANSFER` の非構造経路（`--mesh=auto|structured|unstructured`、例題 plate-ht-2）— Phase 11
- [x] 非直交補正（over-relaxed 分解 + 傾いた境界面の接線補正 + `solve_corrected`）を fvm 層に追加し Darcy / スカラー輸送 / 伝熱に接続 — Phase 11
- [x] `nsb/{data,assembly}.py` をコミット 1647839 時点のスナップショットとして切り離し（同期スクリプト・乖離テスト削除）— Phase 11
- [ ] `BrinkmanFlowFVMProcess` の演算子合成を owner/neighbour で組み直す（非構造 NS ファミリーへ統合）
- [ ] `NaturalConvectionFDMProcess`（SIMPLE、Rhie–Chow）を面リストで
- [ ] 四面体・楔の `InpMeshProcess` 対応、`core/strategies/CorrectedDiffusionScheme` の整理（fvm 層に統合済みの機能と重複）
- [ ] 内部面の `*SURFACE`（`InternalFaceBC` 相当）、Darcy の Forchheimer / Brinkman 項、非定常
- [ ] `core/data.BoundaryData` / `SolverInputData` 等の死んだスキーマの整理

## 将来構想

- LES / DES
- 多相流（VOF）
- ~~伝熱ソルバー高速化（Numba JIT / PyAMG マルチグリッド）~~ → Phase 1.5 で実装済み
- ~~Numba JIT 性能ベンチマーク~~ → status-8 で実施（Python GS比 176〜256倍）
- ~~CI に pyamg/numba オプション依存テスト追加~~ → status-8 で実施
- 適応格子細分化（AMR）

---

## 推奨ソルバー構成（初期）

- **対流スキーム**: TVD (van Leer) — 安定性と精度のバランス
- **拡散スキーム**: 中心差分 + 非直交補正
- **圧力-速度連成**: PISO（非定常推奨）/ SIMPLEC（定常推奨）/ SIMPLE
- **出口境界**: OUTLET_CONVECTIVE（対流流出）
- **線形ソルバー**: 圧力=AMG, 速度=BiCGSTAB+ILU
- **時間積分**: BDF2（推奨）/ 1次陰的オイラー
