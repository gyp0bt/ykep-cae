# status-9: 3D自然対流ソルバー (FDM + SIMPLE法) — 固体-流体練成伝熱

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-04-01

## 概要

3次元自然対流ソルバー（NaturalConvectionFDMProcess）を新規実装。
SIMPLE法による圧力-速度連成 + Boussinesq近似による浮力項を使い、
等間隔直交格子上の自然対流問題を解く。固体-流体練成伝熱（Conjugate Heat Transfer）に対応。

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae_fluid/natural_convection/__init__.py` | 自然対流モジュール新規作成 |
| `xkep_cae_fluid/natural_convection/data.py` | データスキーマ（NaturalConvectionInput/Result, FluidBoundarySpec） |
| `xkep_cae_fluid/natural_convection/assembly.py` | 疎行列アセンブリ（運動量・圧力補正・エネルギー方程式） |
| `xkep_cae_fluid/natural_convection/solver.py` | NaturalConvectionFDMProcess（SIMPLE法ソルバー） |
| `tests/test_natural_convection.py` | APIテスト6件 + 物理テスト8件 + Nusseltベンチマーク1件(slow) |
| `docs/design/natural-convection-fdm.md` | 設計文書新規作成 |
| `docs/status/status-9.md` | 本ステータスファイル |
| `docs/status/status-index.md` | インデックス更新 |
| `docs/roadmap.md` | Phase 3 タスク更新 |
| `README.md` | テスト数・プロセス数・パッケージ構成更新 |

### 機能詳細

#### 1. NaturalConvectionFDMProcess

- **支配方程式**: 非圧縮性Navier-Stokes + Boussinesq近似 + エネルギー方程式
- **圧力-速度連成**: SIMPLE法（Semi-Implicit Method for Pressure-Linked Equations）
- **対流項**: 1次風上差分
- **拡散項**: 中心差分
- **時間積分**: 陰的オイラー法（非定常時）
- **線形ソルバー**: BiCGSTAB + ILU前処理（SciPy）
- **緩和**: 速度・圧力・温度の独立緩和係数
- **発散検出**: NaN/大残差で早期停止

#### 2. 固体-流体練成伝熱

- `solid_mask` (ndarray, bool) で固体/流体領域を指定
- 固体領域: 速度=0（大係数ペナルティ法）、圧力方程式除外
- エネルギー方程式: 固体=拡散のみ、流体=対流+拡散
- 界面: 調和平均熱伝導率で温度・熱流束連続条件を自動適用

#### 3. 境界条件

| 種別 | 速度 | 温度 |
|------|------|------|
| NO_SLIP | 壁面速度=0 | — |
| SLIP | 法線=0、接線応力=0 | — |
| INLET_VELOCITY | 指定速度 | — |
| OUTLET_PRESSURE | 外挿 | — |
| SYMMETRY | 法線=0 | — |
| DIRICHLET | — | 温度固定 |
| NEUMANN | — | 熱流束指定 |
| ADIABATIC | — | 断熱 |

#### 4. データスキーマ

- **NaturalConvectionInput**: 領域・物性・BC・ソルバー設定（frozen dataclass）
- **NaturalConvectionResult**: 速度(u,v,w)・圧力・温度・収束情報・残差履歴
- **FluidBoundarySpec**: 流体BC + 温度BC の複合仕様
- プロパティ: dx/dy/dz, nu, Pr, alpha_thermal, is_transient

## テスト結果

- 新規テスト数: **14**（API 6 + 物理 8）+ slow 1（Nusseltベンチマーク）
- 既存テスト: 変更なし
- 契約違反: **0件**（6プロセス登録）

### テスト一覧

| テスト | 種別 | 内容 |
|-------|------|------|
| test_meta_exists | API | ProcessMeta定義確認 |
| test_process_returns_result | API | 戻り値型確認 |
| test_data_schema_properties | API | プロパティ計算検証 |
| test_boundary_spec_defaults | API | デフォルト値確認 |
| test_result_frozen | API | frozen不変性 |
| test_execute_input_immutability | API | C9入力不変性 |
| test_no_gravity_no_flow | 物理 | 重力なし→速度≈0 |
| test_pure_conduction_temperature | 物理 | 浮力なし→線形温度分布 |
| test_buoyancy_drives_vertical_flow | 物理 | 浮力→垂直流発生 |
| test_solid_region_zero_velocity | 物理 | 固体領域速度=0 |
| test_temperature_bounded | 物理 | 温度が境界値範囲内 |
| test_residual_history_populated | 物理 | 残差履歴記録 |
| test_transient_timestep | 物理 | 非定常タイムステップ数 |
| test_symmetry_boundary | 物理 | 対称BC動作確認 |
| test_differentially_heated_cavity_nusselt | 物理(slow) | de Vahl Davis Nu比較 |

## status-8 TODO 消化状況

- [ ] TVD 対流スキーム実装（van Leer, Superbee）→ 未着手
- [ ] 非直交補正付き拡散スキーム実装 → 未着手
- [ ] PolyMeshReader のバイナリ形式対応 → 未着手
- [x] Phase 3: SIMPLE ソルバー着手（運動量方程式アセンブリ）→ **完了**
- [ ] PyAMG の AMG 構築キャッシュ化 → 未着手

## TODO

- [ ] Rhie-Chow 補間の実装（コロケーション格子のチェッカーボード圧力抑制）
- [ ] 差分加熱キャビティベンチマーク（Ra=10³〜10⁵）の定量検証
- [ ] 非定常自然対流テスト（カルマン渦列は Phase 4 へ）
- [ ] TVD 対流スキーム実装（van Leer, Superbee）
- [ ] 非直交補正付き拡散スキーム実装
- [ ] PolyMeshReader のバイナリ形式対応
- [ ] PyAMG の AMG 構築キャッシュ化

## 設計上の懸念

- **Rhie-Chow未実装**: コロケーション配置でRhie-Chow補間なしのため、大きな温度差・低粘度の場合にチェッカーボード圧力振動が発生し発散しやすい。緩和係数を低めに設定する必要がある（推奨: alpha_u≤0.3, alpha_p≤0.1）
- **圧力基準点**: 最初の流体セルの圧力補正をゼロ固定。閉じた系では問題ないが、開放境界が複数ある場合は注意
- **FDM限定**: 現在は等間隔直交格子のみ。不等間隔格子は未対応（既存の heat_transfer は対応済み）

## 開発運用メモ

- 効果的: 既存の HeatTransferFDMProcess のパターン（ProcessMeta, frozen dataclass, binds_to）を踏襲し、新規プロセスの骨格作成が迅速に完了
- 効果的: 物理テストでパラメータ調整を丁寧に行い、安定性の限界を把握できた
- 注意: SIMPLE法の安定性はRhie-Chow補間に大きく依存。次回優先的に実装すべき
- 注意: コロケーション配置では圧力勾配の中心差分が不安定要因。スタッガード格子への移行も検討
