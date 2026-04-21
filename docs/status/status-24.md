# status-24: HeaterProcess + 水槽ヒーター自然対流デモ（Phase 6.2b）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-21
**ブランチ**: `claude/execute-status-todos-waU5V`
**テスト数**: 267（+13 new）
**契約違反**: 0 件（登録プロセス 10、+1）

## 概要

水槽設計 CAE ロードマップ Phase 6.2b として、水槽用ヒーター相当の体積熱源 `q_vol` を
計算する `HeaterProcess` を新設し、`AquariumGeometryProcess`（status-23）+ `HeaterProcess`
+ `NaturalConvectionFDMProcess` の 3 段連携デモを追加した。ヒーターは
**CONSTANT_FLUX（定熱流束）** と **CONSTANT_TEMPERATURE（定温ヒステリシス）** の
2 モードを持ち、定温モードは Process を純粋関数に保つため `prev_on` を呼出し側で
保持して渡す設計とした。

## 実装内容

### 1. `xkep_cae_fluid/aquarium/heater.py` 新設

| 要素 | 内容 |
|-----|------|
| `HeaterMode` (Enum) | `CONSTANT_FLUX` / `CONSTANT_TEMPERATURE` |
| `HeaterGeometry` (frozen) | `x_range` / `y_range` / `z_range` バウンディングボックス |
| `HeaterInput` (frozen) | 座標/セル幅配列 + geometry + モード + power/setpoint/band/measured_T/prev_on |
| `HeaterResult` (frozen) | `q_vol: ndarray (nx,ny,nz)` + `mask` + `on` + `volume_m3` |
| `HeaterProcess` | PreProcess。バウンディングボックスからマスク作成 → `q_vol` 分配 |

### 2. `xkep_cae_fluid/aquarium/__init__.py` 拡張

`HeaterGeometry` / `HeaterInput` / `HeaterMode` / `HeaterProcess` / `HeaterResult` を
パブリック API に追加。

### 3. 設計文書

- `docs/design/aquarium-heater.md`: 設計仕様 + ヒステリシス状態遷移表 + テスト計画
- `docs/design/README.md`: 本文書を「水槽モジュール設計文書」セクションへ追加

### 4. テスト: `tests/test_aquarium_heater.py`（13 件、全合格）

**TestHeaterAPI**（`@binds_to(HeaterProcess)`、6 件）
- `test_meta_exists`: ProcessMeta 基本項目
- `test_process_returns_heater_result`: 返却型と shape
- `test_invalid_power_raises`: `power_watts < 0` で ValueError
- `test_invalid_band_raises`: `hysteresis_band_K < 0` で ValueError
- `test_invalid_range_raises`: `x_range` min>=max で ValueError
- `test_empty_heater_region_raises`: 格子範囲外の heater 範囲で ValueError

**TestHeaterPhysics**（6 件）
- `test_constant_flux_always_on`: 高温でも CONSTANT_FLUX は常時 ON
- `test_constant_flux_power_conserved`: `∫q_vol dV ≈ power_watts`（相対誤差 < 1e-10）
- `test_constant_flux_q_vol_zero_outside_mask`: マスク外で q_vol=0
- `test_hysteresis_on_below_setpoint`: 設定-band/2 以下で ON
- `test_hysteresis_off_above_setpoint`: 設定+band/2 以上で OFF（q_vol 全ゼロ）
- `test_hysteresis_midband_keeps_prev_state`: 中間帯では `prev_on` 維持

**TestHeaterIntegration**（1 件）
- `test_q_vol_shape_matches_nc_input`: dtype float64 / shape (nx, ny, nz)

### 5. デモ: `examples/aquarium_heater_natural_convection.py`

- 30×10×15 cm 小型水槽を粗格子 (12×4×8) で構成
- `AquariumGeometryProcess`（底床 2cm + 1.5 倍 refinement） → `HeaterProcess`（2W CONSTANT_FLUX、右奥 z=5-12cm） → `NaturalConvectionFDMProcess` の 3 段連携
- 全面 `NO_SLIP` + `ADIABATIC`（水面も rigid-lid 断熱近似）
- 出力: `output/aquarium_heater_natural_convection.png`（y 中央断面の温度 + 速度ベクトル + ヒーター位置 + 底床）
- 10 秒シミュレーション結果: T 25.0°C → 25.96〜37.15°C（ΔT≈12 K）、|v|_max ≈ 1.3 mm/s

## 設計判断と妥協点

### 判断: Process を純粋関数に保ち、状態 (prev_on) は呼出し側管理

`HeaterProcess` が直接ヒステリシス状態を保持すると frozen dataclass 契約と衝突し、
過渡ループを明示的に書くという本リポジトリのスタイルと乖離する。`HeaterInput.prev_on` を
呼出し側で保持して渡す設計は `extra_scalars` の方針と整合する。

### 判断: `uses=[]`（AquariumGeometry への依存を宣言しない）

`HeaterProcess` は `AquariumGeometryResult` の配列データを入力として受け取るが、
`AquariumGeometryProcess` 自体を呼ばないため C5（未宣言依存）に該当しない。
純粋に座標/セル幅配列から q_vol を計算する薄い Process として扱う。

### 判断: デモは小型水槽 + 人工物性を採用

現行 `NaturalConvectionFDMProcess` は status-11/12 で既知の通り実水物性
（mu=1e-3、高 Ra）では SIMPLE 連成が十分収束しない。Phase 6.2b の目的は **Process 連携の
検証** であり、ソルバー改良（Phase 6.0 並行 PR）ではないため、既存 NC ベンチマーク
Phase 1 と同一の人工物性（rho=1.0, mu=0.01, Cp=1000, k=1.0, beta=1e-3）を採用した。
実水対応は Phase 6 後半の SIMPLEC/PISO 拡張とセットで進める。

### 判断: デモの水槽サイズを 30×10×15 cm に縮小

90×30×45 cm + 粗格子（18×6×12）は空間スケールが大きく、SIMPLE の圧力-速度連成の
反復数不足によって温度場が非物理的に暴れる（T_min が -110°C など）。小型水槽では
ΔT≈12K・|v|≈1.3 mm/s の物理的に妥当な場が得られるため、まず連携動作の可視化と
リグレッション検出を優先した。実スケールは Phase 6.0 改善後に `configs/` にパラメトリック
ケースとして追加予定。

### 妥協点

- CONSTANT_TEMPERATURE モードは OFF 時 `q_vol=0.0` を返すため、大きなバンドで
  離散的に ON/OFF が切替わる。スムーズな制御が必要な場合は Phase 6.3+ で PID 拡張。
- ヒーター形状はバウンディングボックスのみ。棒状ヒーターの実際の円柱形状は
  将来拡張（`HeaterGeometry` をプロトコル化）で対応。

## 動作確認

- `python -m pytest tests/test_aquarium_heater.py -v` → 13 passed
- `python -m pytest tests/test_aquarium_geometry.py tests/test_aquarium_heater.py` → 27 passed
- `python -m pytest tests/` → 267 passed（Phase 6.2a+6.2b 合計 +27）
- `ruff check xkep_cae_fluid/ tests/ examples/aquarium_heater_natural_convection.py` → All checks passed
- `ruff format --check xkep_cae_fluid/ tests/ examples/aquarium_heater_natural_convection.py` → 57 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反 0 件（登録プロセス 10）
- `python examples/aquarium_heater_natural_convection.py` → PNG 出力成功、ΔT≈12 K、|v|_max ≈ 1.3 mm/s

## 次に着手すべきタスク

### Phase 6.3a（status-25 予定）

- [ ] `InternalFaceBC` 実装（外部フィルター吐出用、`natural_convection/assembly.py` 拡張）
- [ ] Eheim 相当のインレット/アウトレット + 流量 Q [L/h] パラメータ化

### Phase 6.3b（status-26 予定）

- [ ] `examples/aquarium_filter_circulation.py`（インレット＋アウトレット＋ヒーター）
- [ ] 質量保存検証（流入=流出、<1% 誤差）

### Phase 6.0 並行（優先度中）

- [ ] `natural_convection/assembly.py` d 係数見直し → 実水対応
  - 水槽デモの実スケール化（90×30×45 cm + 水実物性）はこの PR に依存

## 設計上の懸念・運用メモ

- **デモの物性と現実の乖離**: ユーザーが「水槽ヒーター」のデモを見ると実水を期待するが、
  現状は人工物性で動作させている。PNG のタイトルと docstring 冒頭で「artificial fluid」と
  明記して誤解を防いでいる。Phase 6.0 完了時にデモを実水物性で再実行し PNG を差し替える。
- **PR 粒度**: status-23（Geometry）/ status-24（Heater + Demo）をそれぞれ 1 コミット、
  ドキュメントまとめを 1 コミット（計 3 コミット）で push する運用。

## 変更ファイル

- **新規**: `xkep_cae_fluid/aquarium/heater.py`
- **編集**: `xkep_cae_fluid/aquarium/__init__.py`（Heater 系をエクスポート）
- **新規**: `docs/design/aquarium-heater.md`
- **新規**: `tests/test_aquarium_heater.py`
- **新規**: `examples/aquarium_heater_natural_convection.py`
- **新規**: `docs/status/status-24.md`（本ファイル）
- **編集**: `docs/design/README.md`（aquarium-heater.md 追加）
- **編集**: `README.md`（テスト数 240 → 267、登録プロセス 7 → 10、近況更新）
- **編集**: `docs/status/status-index.md`（status-23 / status-24 行追加）
- **編集**: `docs/roadmap-aquarium.md`（Phase 6.2a / 6.2b 完了マーク）
