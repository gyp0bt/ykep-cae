# status-21: 汎用スカラー輸送 ScalarTransportProcess 新設（Phase 6.1a）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-20
**ブランチ**: `claude/check-status-todos-HLE58`
**テスト数**: 232（+8 from 224）
**契約違反**: 0 件（登録プロセス 7 → `ScalarTransportProcess` 追加）

## 概要

水槽設計 CAE ロードマップ Phase 6.1a として汎用スカラー輸送プロセス
`ScalarTransportProcess` を新設。任意のスカラー（CO2/O2 濃度、トレーサー、塩分）を
既存流体場の上で陰的 Euler + BiCGSTAB で輸送する。Phase 6.2（水槽ジオメトリ +
ヒーターデモ）以降での CO2/O2 分布・生体反応ソースの基盤となる。

## 実装内容

### 新規ディレクトリ: `xkep_cae_fluid/scalar_transport/`

| ファイル | 役割 |
|---------|------|
| `__init__.py` | パブリック API のエクスポート |
| `data.py` | `ScalarFieldSpec` / `ScalarBoundarySpec` / `ScalarBoundaryCondition` / `ScalarTransportInput` / `ScalarTransportResult` |
| `assembly.py` | 対流-拡散-ソース疎行列アセンブリ（Dirichlet/Neumann/Adiabatic/Robin） |
| `solver.py` | `ScalarTransportProcess`（SolverProcess 派生、BiCGSTAB+ILU） |

### 設計文書: `docs/design/scalar-transport-fdm.md`

- 支配方程式: ∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(Γ∇φ) + S
- 空間: 1次風上（対流）+ 中心差分（拡散）
- 時間: 陰的 Euler（BDF2 は Phase 6.1b 以降）
- BC: Dirichlet / Neumann / Adiabatic / Robin（ヘンリー則で使用）
- Phase 6.1b での `NaturalConvectionInput.extra_scalars` 統合計画を明記

### テスト: `tests/test_scalar_transport.py`

8 件のテストを追加（全て合格）:

**TestScalarTransportAPI**（4 件）
- `test_meta_exists`: ProcessMeta 契約
- `test_process_returns_result`: 戻り値の型と形状
- `test_zero_velocity_adiabatic_no_source_holds_initial`: 非定常ゼロソース+初期一様で値保持
- `test_transient_returns_n_steps`: 非定常ステップ数

**TestScalarTransportConvergence**（2 件、解析解検証）
- `test_1d_steady_pure_diffusion_linear`: 両端 Dirichlet → 線形プロファイル（atol 1e-6）
- `test_neumann_flux_adds_expected_increment`: Dirichlet + Neumann → dφ/dx=flux/Γ

**TestScalarTransportPhysics**（2 件）
- `test_robin_bc_equilibrium`: 全面 Robin + ソース無し → 平衡濃度 φ_inf（atol 1e-4）
- `test_1d_pure_convection_conserves_integral`: 矩形波の合計濃度が単調非増加（湧き出しなし）

## 設計判断と妥協点

### 判断: natural_convection/assembly.py の refactor は見送り

status-20 計画では「エネルギー方程式アセンブリを共通ヘルパー化」とあったが、
`build_energy_system`（1158 行のうち 225 行）の引数は Rhie-Chow 面速度や BDF2
`T_old_old_time` 等で NaturalConvection 固有のため、共通化より**新規実装**を選択した。
Phase 6.1b（`NaturalConvectionInput.extra_scalars` 統合）時に、共通部分の
抽出リファクタを判断する。

### 妥協点

- 対流スキームは 1 次風上のみ（TVD の移植は Phase 6.1b）
- 時間積分は陰的 Euler のみ（BDF2 は Phase 6.1b）
- 非直交補正未対応（等間隔直交格子のみ）
- 拡散係数はスカラー定数（空間変化は Phase 6.4 多孔質実装で拡張）

## 動作確認

- `python -m pytest tests/test_scalar_transport.py` → 8 passed
- `python -m pytest tests/` → 230 passed, 4 failed（事前存在の Numba 未インストール問題のみ）
- `ruff check xkep_cae_fluid/ tests/` → All checks passed
- `ruff format --check xkep_cae_fluid/ tests/` → pass
- `python contracts/validate_process_contracts.py` → 契約違反なし（7 プロセス登録）

## 次に着手すべきタスク

### Phase 6.1b（status-22 予定）
- [ ] `NaturalConvectionInput.extra_scalars: list[ScalarFieldSpec]` 追加
- [ ] SIMPLE 外部反復内で Rhie-Chow 面速度を共有し、温度と他スカラーを同時輸送
- [ ] 物理テスト: 閉じた系で質量保存、温度と共にトレーサーが運搬される

### Phase 6.2a（status-23 予定）
- [ ] `AquariumGeometryProcess` 新設（90×30×45 cm 領域 + 底床/ガラスマスク）

### 並行 Phase 6.0
- [ ] natural_convection/assembly.py d 係数計算の見直し（mu=1.85e-5 mass 残差課題）

## 設計上の懸念・運用メモ

### 運用上の気づき

- **PR 粒度**: 設計文書→データ→アセンブリ→ソルバー→テストの順に 1 コミット
  単位にせず、まとめて 1 コミットで実装した方が差分の整合性レビューがしやすい
  と判断。次回以降も「Phase 6.1a 全体を 1 コミット + ドキュメント 1 コミット」の
  2 コミット構成を標準とする。
- **不良設定問題の扱い**: 定常 adiabatic + ソース無しは数学的に解が定数倍の
  自由度を持ち、BiCGSTAB では初期値保持が保証されない。テストでは非定常化で
  対応。将来的には BC 検証で「全面 adiabatic + 定常 + ソース無し」を警告する
  バリデーション層を検討。

### 設計上の懸念

- **Phase 6.1b 統合時のインターフェース**: 現在の `ScalarTransportInput.u, v, w`
  は NaturalConvection の cell 速度を想定している。Rhie-Chow 面速度も受け取れる
  ように拡張する必要がある（`rc_face_velocities` オプション引数）。
- **Robin BC のパラメータ命名**: `h_mass` と `phi_inf` は水槽のヘンリー則用だが、
  熱伝達の `h_conv` / `T_inf` と用語が分離している。ドキュメントで対応関係を
  明記した（scalar-transport-fdm.md）。

## 変更ファイル

- **新規**: `xkep_cae_fluid/scalar_transport/__init__.py`
- **新規**: `xkep_cae_fluid/scalar_transport/data.py`
- **新規**: `xkep_cae_fluid/scalar_transport/assembly.py`
- **新規**: `xkep_cae_fluid/scalar_transport/solver.py`
- **新規**: `tests/test_scalar_transport.py`
- **新規**: `docs/design/scalar-transport-fdm.md`
- **新規**: `docs/status/status-21.md`（本ファイル）
- **編集**: `README.md`（テスト数 224 → 232、プロセス数 6 → 7、ディレクトリ一覧）
- **編集**: `docs/design/README.md`（設計文書索引に scalar-transport-fdm.md 追加）
- **編集**: `docs/status/status-index.md`（本 status 行追加）
- **編集**: `docs/roadmap-aquarium.md`（Phase 6.1a 完了マーク）
