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
- [ ] Phase 6.1: 汎用スカラー輸送基盤 `ScalarTransportProcess`（2 PR）
- [ ] Phase 6.2: 水槽ジオメトリ + ヒーター最小デモ（2 PR）← 初デモ
- [ ] Phase 6.3: 外部フィルター `InternalFaceBC`（2 PR）
- [ ] Phase 6.4: 多孔質媒体 Darcy-Forchheimer（2 PR）
- [ ] Phase 6.5: 植物ライト Beer-Lambert（1 PR）
- [ ] Phase 6.6: 生体反応（光合成/呼吸）（2 PR）
- [ ] Phase 6.7: ガス交換界面 + CO2 気泡プルーム（2 PR）
- [ ] Phase 6.8: 水槽システム統合デモ（1〜2 PR）
- [ ] Phase 6.9: 日周期拡張 + 設計最適化の足場（将来）

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
