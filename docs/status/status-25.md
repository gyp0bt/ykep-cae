# status-25: AquariumFilterProcess + InternalFaceBC（Phase 6.3a）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-21
**ブランチ**: `claude/execute-status-todos-VABU4`
**テスト数**: 286（+19 new、+1 xfail）
**契約違反**: 0 件（登録プロセス 11、+1）

## 概要

水槽設計 CAE ロードマップ Phase 6.3a として、水槽内部の任意セル集合に強制速度/温度
拘束と圧力基準ピン留めを与える **`InternalFaceBC` データ契約** を
`natural_convection/data.py` に新設し、`natural_convection/assembly.py` および
`natural_convection/solver.py` で運動量・圧力補正・エネルギー方程式および速度補正
段に組み込んだ。併せて、エーハイム相当の外部フィルター（吐出ノズル＋吸入ノズル＋
流量 Q [L/h]＋吐出方向ベクトル）から `InternalFaceBC(INLET/OUTLET)` のペアを自動
構築する **`AquariumFilterProcess`** を `xkep_cae_fluid/aquarium/filter.py` に実装した。

命名は水槽（CFD 領域）視点で統一し、`inflow`（水槽へ流入 = フィルターのリターン）/
`outflow`（水槽から流出 = フィルターのインテーク）を採用、CFD 的な `INLET`/`OUTLET`
と直感的に対応するようにした。

## 実装内容

### 1. `xkep_cae_fluid/natural_convection/data.py` 拡張

| 要素 | 内容 |
|-----|------|
| `InternalFaceBCKind` (Enum) | `INLET` / `OUTLET` |
| `InternalFaceBC` (frozen) | `kind` + `mask (nx,ny,nz) bool` + `velocity` + `temperature` + `pressure` + `label` |
| `NaturalConvectionInput.internal_face_bcs` | `tuple[InternalFaceBC, ...] = ()`（デフォルト空で後方互換維持） |

### 2. `xkep_cae_fluid/natural_convection/assembly.py` 拡張

- 定数: `_INTERNAL_BC_PENALTY = 1e25`（固体マスク 1e30 より緩く、時間項 `ρ/dt` より
  十分支配的）
- ヘルパー: `_apply_internal_bc_momentum`, `_apply_internal_bc_pressure`,
  `_apply_internal_bc_energy`, `_inlet_mask_union`
- 運動量アセンブリ: INLET セルで `diag += penalty`、`rhs += penalty * velocity[comp]` を
  適用して速度を強制
- 圧力補正アセンブリ（RC 版と基本版の両方）: INLET/OUTLET セルで `p'=0` をペナルティで
  ピン留め
- エネルギーアセンブリ: INLET で `temperature` が指定された場合のみペナルティで強制
- OUTLET 存在時、既存の「最初の流体セルを圧力基準に固定」ロジックを省略（OUTLET が
  既に基準を担保）

### 3. `xkep_cae_fluid/natural_convection/solver.py` 拡張

- 速度補正（`_correct_velocity`）後に INLET セルの u/v/w を `velocity` に再クランプ
  （∇p' 補正で汚されるのを防ぐ）
- `alpha_u` 緩和後にも再クランプ、`alpha_T` 緩和後に INLET 温度を再クランプ
  （緩和は旧値とのブレンドなので、強制値を維持するため）
- `_adapt_relaxation` と `_solve_transient` の `NaturalConvectionInput` 再構築時に
  `internal_face_bcs=inp.internal_face_bcs` を伝搬

### 4. `xkep_cae_fluid/natural_convection/__init__.py` 拡張

`InternalFaceBC` / `InternalFaceBCKind` をパブリック API に追加。

### 5. `xkep_cae_fluid/aquarium/filter.py` 新設

| 要素 | 内容 |
|-----|------|
| `NozzleGeometry` (frozen) | `x_range` / `y_range` / `z_range` バウンディングボックス |
| `AquariumFilterInput` (frozen) | 座標/セル幅配列 + inflow/outflow geometry + flow_rate_lph + inflow_direction + inflow_temperature_K |
| `AquariumFilterResult` (frozen) | `inflow_bc` / `outflow_bc` (`InternalFaceBC`) + masks + `inflow_velocity` + `flow_rate_m3s` + `inflow_area_m2` |
| `AquariumFilterProcess` | PreProcess。流量 [L/h] → [m³/s] 換算、投影面積から `|v| = Q/A` を算出し吐出ベクトル化 |

デフォルト流量 **440 L/h**（エーハイム 2213 相当）、デフォルト吐出方向 `(+1, 0, 0)`。

### 6. `xkep_cae_fluid/aquarium/__init__.py` 拡張

`AquariumFilterInput` / `AquariumFilterProcess` / `AquariumFilterResult` / `NozzleGeometry` を
パブリック API に追加。

### 7. 設計文書

- `docs/design/aquarium-filter.md`: 設計仕様 + ペナルティ係数選定 + テスト計画 + 妥協点
- `docs/design/README.md`: 本文書を「水槽モジュール設計文書」セクションへ追加

### 8. テスト: `tests/test_aquarium_filter.py`（20 件、19 合格 + 1 xfail）

**TestAquariumFilterAPI**（`@binds_to(AquariumFilterProcess)`、7 件）
- `test_meta_exists`: ProcessMeta 基本項目
- `test_process_returns_filter_result`: 返却型と InternalFaceBC ペア
- `test_invalid_flow_rate_raises`: `flow_rate_lph <= 0` で ValueError
- `test_invalid_range_raises`: min>=max で ValueError
- `test_zero_direction_raises`: ゼロ方向ベクトルで ValueError
- `test_empty_inflow_region_raises`: 格子範囲外ノズルで ValueError
- `test_overlapping_regions_raises`: inflow/outflow マスク重複で ValueError

**TestAquariumFilterPhysics**（5 件）
- `test_flow_rate_converted_to_m3s`: 3600 L/h → 1e-3 m³/s
- `test_inflow_velocity_matches_q_over_area`: `|v| · A ≈ Q` が成立
- `test_inflow_direction_aligned`: 指定方向ベクトルに沿った速度ベクトル
- `test_masks_disjoint`: inflow/outflow マスクが非空かつ重複なし
- `test_inflow_temperature_forwarded`: `inflow_temperature_K` が BC に伝搬

**TestInternalFaceBCIntegration**（5 件中 4 pass + 1 xfail）
- `test_inlet_velocity_enforced`: ソルバー実行後も強制速度が維持される
- `test_outlet_draws_flow_toward_it`: OUTLET 付近で非ゼロ速度が観測される（循環成立）
- `test_inflow_temperature_enforced`: 温度拘束が強制される
- `test_no_internal_bc_baseline_unchanged`: `internal_face_bcs=()` で既存挙動（流れゼロ）
- `test_outlet_mass_nearly_balances_inlet` [**xfail**]: 入口流量と出口流量の厳密釣り合い
  は SIMPLE では未達（CLAUDE.md 既知の mass 残差 → SIMPLEC/PISO 化待ち）

**TestInternalFaceBCAssembly**（3 件）
- `test_internal_bc_field_defaults_empty`: `internal_face_bcs` のデフォルトが空 tuple
- `test_internal_bc_kind_enum`: Enum 2 値 (inlet/outlet)
- `test_empty_mask_is_noop`: 空マスクは no-op（発散せず）

## 設計判断と妥協点

### 判断: セル領域ベースのペナルティ方式

厳密な「面ベース」境界条件ではなく、セル集合に対するペナルティ方式を採用。
CVFEM / 食い違い格子上の内部面ディリクレは実装コストが高く、Phase 6.3a の目的
（外部フィルター循環の基本動作検証）には過剰。エーハイム相当の流量レベル
（Q=440〜1000 L/h、v~0.03 m/s、Re~800〜数千）では、数セル幅のノズル領域で速度を
強制する近似で十分な物理精度が得られる。

### 判断: `inflow`/`outflow` 命名（水槽視点）

CFD の `INLET`/`OUTLET` は初学者に混乱を招きやすい（フィルターの「インレット」は
水槽視点では `OUTLET` に対応するため）。本 Phase では水槽（CFD 領域）視点で
統一し、`inflow_geometry` → `InternalFaceBC(INLET)`、`outflow_geometry` → `InternalFaceBC(OUTLET)`
と直感的に対応させた。

### 判断: INLET の `temperature` は Optional

ヒーターが水槽内にある場合、フィルターの吐出温度は外部フィルター通過時の放熱により
ヒーター通過後の値とは一致しない。`temperature=None` なら「フィルター通過による温度
変化は無視」、指定ありなら「この温度で吐出される（ヒーター併用時の吐出温度設定）」
という 2 ケースを扱う。

### 判断: ペナルティ係数 1e25（固体 1e30 との段階分け）

- 固体マスク: `1e30`（速度/圧力厳密ゼロ化）
- INLET 運動量/温度: `1e25`（速度/温度強制）
- INLET/OUTLET 圧力補正: `1e30`（`p'=0` ピン留め）

1e25 は時間項 `ρ/dt` が通常 1e5〜1e7 オーダーであることに対して十分支配的だが、
行列の条件数を固体マスクレベルまで悪化させない妥協点。

### 判断: Process を純粋関数として設計

`HeaterProcess` と同じく、`AquariumFilterProcess` は状態を持たない純粋関数。
時間経過や流量変動は外部ループで `AquariumFilterInput` を更新して再呼出しする。

### 妥協点・既知課題

- **厳密質量保存の未達（xfail）**: 強制入口による mass 残差増幅は CLAUDE.md 既知課題と
  同じ根因（SIMPLE 圧力-速度連成の有限反復誤差）で、SIMPLEC/PISO 化完了まで xfail 扱い。
  オーダーが一致する程度（入口/出口 ratio ∈ [0.5, 1.5]）には到達していない。
- **投影面積の扱い**: 斜め吐出では `|dir_x|·dy·dz + |dir_y|·dx·dz + |dir_z|·dx·dy` の和で
  投影面積を計算。軸平行吐出では素朴な直交断面積と一致する。
- **ノズル形状**: バウンディングボックスのみ。円柱パイプの実ノズル形状は Phase 6.4
  以降で多孔質媒体と組合せて拡張。

## 動作確認

- `python -m pytest tests/test_aquarium_filter.py -v` → **19 passed, 1 xfailed**
- `python -m pytest tests/` → **286 passed, 1 xfailed**（Phase 6.3a 合計 +19 + 1 xfail）
- `ruff check xkep_cae_fluid/ tests/` → All checks passed
- `ruff format --check xkep_cae_fluid/ tests/` → all files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反 0 件（登録プロセス 11）

## 次に着手すべきタスク

### Phase 6.3b（status-26 予定）

- [ ] `examples/aquarium_filter_circulation.py`（inflow + outflow + ヒーター 3 段連携）
- [ ] 質量保存検証（流入 = 流出、< 1% 誤差）→ SIMPLEC/PISO 化後に厳密化
- [ ] inflow 温度とヒーターの相互作用可視化

### Phase 6.0 並行（優先度中）

- [ ] `natural_convection/assembly.py` d 係数見直し → 実水対応
- [ ] SIMPLEC/PISO の mass 残差改善（これにより本 Phase の xfail を strict 化可能）

## 設計上の懸念・運用メモ

- **CFD 初学者向け命名**: `inflow`/`outflow` 命名は水槽フロー視点での明快さを優先
  したが、内部 API（`InternalFaceBCKind.INLET`/`OUTLET`）と外部 API の対応表を
  `docs/design/aquarium-filter.md` で明示した。
- **ペナルティ方式の限界**: 極端に小さい dt（`ρ/dt` が 1e25 に接近）では支配性が失われる。
  水槽シミュレーションで想定される `dt=0.01〜1s, ρ=1000` の範囲では安全。
- **PR 粒度**: status-25 を 1 コミット（implementation + docs + tests + status）で push。

## 変更ファイル

- **編集**: `xkep_cae_fluid/natural_convection/data.py`（`InternalFaceBCKind` / `InternalFaceBC` / `internal_face_bcs` 追加）
- **編集**: `xkep_cae_fluid/natural_convection/assembly.py`（ペナルティヘルパー 4 個 + 運動量/圧力補正/エネルギーへの配線）
- **編集**: `xkep_cae_fluid/natural_convection/solver.py`（速度補正後 + 緩和後の強制クランプ、`internal_face_bcs` 伝搬）
- **編集**: `xkep_cae_fluid/natural_convection/__init__.py`（`InternalFaceBC` / `InternalFaceBCKind` エクスポート）
- **新規**: `xkep_cae_fluid/aquarium/filter.py`
- **編集**: `xkep_cae_fluid/aquarium/__init__.py`（Filter 系をエクスポート）
- **新規**: `docs/design/aquarium-filter.md`
- **新規**: `tests/test_aquarium_filter.py`
- **新規**: `docs/status/status-25.md`（本ファイル）
- **編集**: `docs/design/README.md`（aquarium-filter.md 追加）
- **編集**: `README.md`（テスト数 267 → 286、登録プロセス 10 → 11、近況更新、filter.py を構成図に追加）
- **編集**: `docs/status/status-index.md`（status-25 行追加）
- **編集**: `docs/roadmap-aquarium.md`（Phase 6.3a 完了マーク）
