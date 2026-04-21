# status-22: NaturalConvection に extra_scalars を統合（Phase 6.1b）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-20
**ブランチ**: `claude/check-status-todos-J3tPr`
**テスト数**: 240（+6 new + 2 pre-existing numba 復活）
**契約違反**: 0 件（登録プロセス 7、追加なし）

## 概要

水槽設計 CAE ロードマップ Phase 6.1b として、`NaturalConvectionFDMProcess` に
追加スカラー（CO2/O2/トレーサー等）の同時輸送機能を統合した。SIMPLE 外部反復内で
エネルギー方程式と同じ Rhie-Chow 面速度を共有してスカラー輸送を解くため、
チェッカーボード抑制と質量保存をワンパスで達成する。Phase 6.2 以降で水槽内部の
生体反応・ガス交換ソース項を流し込む基盤となる。

## 実装内容

### 1. `xkep_cae_fluid/scalar_transport/` 拡張

| ファイル | 変更 |
|---------|------|
| `data.py` | `ExtraScalarSpec`（field + 6 面 BC + alpha 緩和）を追加 |
| `assembly.py` | `build_scalar_system` に `rc_face_velocities` 引数を追加し、NaturalConvection の面速度で対流フラックスを組めるように拡張 |
| `__init__.py` | `ExtraScalarSpec` をパブリック API に追加 |

### 2. `xkep_cae_fluid/natural_convection/` 拡張

| ファイル | 変更 |
|---------|------|
| `data.py` | `NaturalConvectionInput.extra_scalars: tuple[ExtraScalarSpec, ...]` 追加 / `NaturalConvectionResult.extra_scalars: dict[str, np.ndarray]` 追加 |
| `solver.py` | `_make_scalar_transport_input` / `_solve_extra_scalars` ヘルパー新設 / `_simple_iteration` を `phi_state` を引数・返値に含める形に拡張 / `_solve_steady` `_solve_transient` で `phi_state` を初期化して引き回し、結果に格納 / 残差履歴に `phi_<name>` キーを追加 |

### 3. テスト: `tests/test_natural_convection_scalar.py`

6 件のテストを追加（全て合格）:

**TestNaturalConvectionScalarAPI**（3 件）
- `test_extra_scalars_default_is_empty`: extra_scalars 未指定時は結果 dict も空
- `test_extra_scalar_appears_in_result`: 指定名が結果に入り形状が一致
- `test_residual_history_contains_scalar_key`: 残差履歴に `phi_<name>` が出現

**TestNaturalConvectionScalarPhysics**（2 件）
- `test_closed_domain_tracer_mass_conservation`: 全面 Adiabatic + ソースなしで合計濃度の相対誤差 < 1%
- `test_tracer_gets_redistributed_by_flow`: 非一様な初期分布のトレーサーが自然対流で運搬される

**TestNaturalConvectionScalarMultiple**（1 件）
- `test_two_scalars_transported`: 2 つの独立スカラーが両方結果に入る

## 設計判断と妥協点

### 判断: scalar_transport を再利用、共通化は先送り

status-21 の設計上の懸念で記載した「natural_convection と scalar_transport の
共通アセンブリへの切り出し」は見送り。現状は natural convection 側に軽量な
アダプタ (`_make_scalar_transport_input`) を置き、`build_scalar_system` を
そのまま呼ぶ。理由:

- `build_scalar_system` は既に `rc_face_velocities` を受け取れるため、
  新設エネルギーヘルパーを作るより直接呼ぶほうがコード重複が少ない
- 共通化は Phase 6.4（多孔質）や Phase 6.6（生体反応）でソース項の扱いが
  確定した後のほうが設計リスクが低い

### 判断: スカラーは SIMPLE 外部反復ごとに解く（セグリゲート）

温度と同じく外部反復の末尾で解く。定常時は外部反復の中で速度・圧力と共に
収束へ向かい、非定常時は各タイムステップ開始時に `phi_old_time` を凍結して
BDF2 非対応（陰的 Euler のみ）で進める。BDF2 対応は Phase 6.1c 以降で判断。

### 判断: スカラーは収束判定に含めない

`_simple_convergence_residual` は速度・圧力・質量のみで SIMPLE 収束を評価する。
スカラー残差 `phi_<name>` は履歴と診断のみに使う（温度と同じ扱い）。
理由: スカラーの RHS が時間項で支配され、相対残差が偽の未収束を生むため。

### 妥協点

- 対流スキームは 1 次風上のみ（TVD 遅延補正の scalar 実装は Phase 6.1c 以降）
- スカラー方程式の内部反復は共通の `max_inner_iter` / `tol_inner` を流用
- 固体セルの扱い: 対流のみ無効化（拡散は有効）。生体反応の固体ソース禁止は
  Phase 6.6 のソース項実装時に検討

## 動作確認

- `python -m pytest tests/test_natural_convection_scalar.py -v` → 6 passed
- `python -m pytest tests/test_scalar_transport.py` → 8 passed（既存不変）
- `python -m pytest tests/test_natural_convection.py` → 63 passed（既存不変、286 秒）
- `python -m pytest tests/` → 240 passed（279 秒、numba 依存のテスト含む）
- `ruff check xkep_cae_fluid/ tests/` → All checks passed
- `ruff format --check xkep_cae_fluid/ tests/` → 51 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反 0 件

## 次に着手すべきタスク

### Phase 6.2a（status-23 予定）
- [ ] `AquariumGeometryProcess` 新設（90×30×45 cm 領域 + 底床マスク + ガラスマスク）
- [ ] `StructuredMeshProcess` の不等間隔機能を底床/水面近傍の refinement に応用

### Phase 6.2b（status-24 予定）
- [ ] `HeaterProcess` 新設（定温ヒステリシス + 定熱流束の 2 モード）
- [ ] `examples/aquarium_heater_natural_convection.py` でヒーター単独デモ

### 並行 Phase 6.0
- [ ] `natural_convection/assembly.py` d 係数計算の見直し（mu=1.85e-5 mass 残差課題）
  — 水が主対象のため Phase 6.2 のブロッカーではない

### Phase 6.1c（優先度低、必要になれば着手）
- [ ] extra_scalars の TVD 遅延補正対応
- [ ] extra_scalars の BDF2 時間積分対応（`phi_old_old_time` 拡張）

## 設計上の懸念・運用メモ

### 運用上の気づき

- **PR 粒度**: status-21 の反省通り「Phase 6.1b 全体を 1 コミット + ドキュメント 1 コミット」の
  2 コミット構成で進めた。レビューしやすく、差分の整合性も取りやすい。
- **テストの設計**: 物理テストで「質量保存」を見るときは、解析解が存在する
  純拡散や純対流ではなく、実際に自然対流と同時運搬させるケースにした方が
  運用側のリグレッション検出力が高い。

### 設計上の懸念

- **スカラーの緩和係数**: `ExtraScalarSpec.alpha` のデフォルトは 1.0（更新 100%）。
  温度の `alpha_T=0.9` と揃えず独立にした。水槽 CAE の強反応（光合成・CO2 添加）で
  数値振動が出た場合は PR ごとに調整する方針で妥当。
- **複数スカラー間の結合**: 現状、各スカラーは独立に解かれる。Phase 6.6 の光合成・
  呼吸では CO2 と O2 が化学量論比で結合するため、ソース項レベルで両方を参照する
  モデルが必要。`BiologicalReactionProcess` で一括して `source` を生成する設計に
  すれば、スカラー方程式自体は独立のままで済む。
- **境界条件の記述量**: 6 面 × N スカラーの BC を指定すると冗長になる。
  ヘンリー則で上面のみ Robin、他は Adiabatic のようなプリセットを Phase 6.7a
  `AirWaterInterfaceBC` で導入予定。

## 変更ファイル

- **新規**: `tests/test_natural_convection_scalar.py`
- **新規**: `docs/status/status-22.md`（本ファイル）
- **編集**: `xkep_cae_fluid/scalar_transport/data.py`（`ExtraScalarSpec` 追加）
- **編集**: `xkep_cae_fluid/scalar_transport/assembly.py`（`rc_face_velocities` 追加）
- **編集**: `xkep_cae_fluid/scalar_transport/__init__.py`（`ExtraScalarSpec` エクスポート）
- **編集**: `xkep_cae_fluid/natural_convection/data.py`（`extra_scalars` 追加）
- **編集**: `xkep_cae_fluid/natural_convection/solver.py`（SIMPLE 統合）
- **編集**: `README.md`（テスト数 232 → 240、状態更新）
- **編集**: `docs/status/status-index.md`（本 status 行追加）
- **編集**: `docs/roadmap-aquarium.md`（Phase 6.1b 完了マーク）
