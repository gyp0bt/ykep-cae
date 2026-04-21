# status-23: AquariumGeometryProcess（Phase 6.2a）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-21
**ブランチ**: `claude/execute-status-todos-waU5V`
**テスト数**: 254（+14 new）
**契約違反**: 0 件（登録プロセス 9、+1）

## 概要

水槽設計 CAE ロードマップ Phase 6.2a として、90×30×45 cm 水槽の構造化メッシュと
底床（砂層）マスク・ガラス壁マスク・水領域マスクを一括生成する
`AquariumGeometryProcess` を新設した。既存 `StructuredMeshProcess` を `uses` 宣言で
再利用し、底床側の refinement にストレッチング機能を応用する。`solid_mask` と
推奨重力ベクトル `(0, 0, -9.81)` を `NaturalConvectionInput` にそのまま渡せる形で返却する。

## 実装内容

### 1. `xkep_cae_fluid/aquarium/` 新設

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `AquariumGeometryInput` / `AquariumGeometryProcess` / `AquariumGeometryResult` エクスポート |
| `geometry.py` | 本 Process 実装 + 底床・ガラスマスク構築ヘルパー |

### 2. 入出力データ

| フィールド | 型 | 説明 |
|-----------|----|------|
| `AquariumGeometryInput.Lx/Ly/Lz` | float | 水槽内部領域 [m]（デフォルト 0.9/0.3/0.45） |
| `AquariumGeometryInput.nx/ny/nz` | int | 各方向セル数（デフォルト 36/12/30） |
| `AquariumGeometryInput.substrate_depth` | float | 底床厚さ [m]（0 で無し） |
| `AquariumGeometryInput.substrate_refinement_ratio` | float | z 方向ストレッチ比（>1 で底床細かく） |
| `AquariumGeometryInput.glass_thickness` | float | ガラス厚 [m]（0 で無し） |
| `AquariumGeometryResult.substrate_mask` / `glass_mask` / `water_mask` / `solid_mask` | ndarray bool (nx,ny,nz) | 各領域マスク |
| `AquariumGeometryResult.gravity` | tuple | 推奨重力 (0, 0, -9.81) |
| `AquariumGeometryResult.mesh` / `dx/dy/dz` / `x_centers/y_centers/z_centers` | — | `StructuredMeshProcess` 由来 |

### 3. 設計文書

- `docs/design/aquarium-geometry.md`: 設計仕様 + 入出力表 + テスト計画 + 拡張方針
- `docs/design/README.md`: 本文書を「水槽モジュール設計文書」セクションへ追加

### 4. テスト: `tests/test_aquarium_geometry.py`（14 件、全合格）

**TestAquariumGeometryAPI**（`@binds_to(AquariumGeometryProcess)`、6 件）
- `test_meta_exists`: ProcessMeta 基本項目
- `test_default_input_returns_result`: デフォルト入力で期待 shape を返す
- `test_solid_mask_is_union_of_substrate_and_glass`: `solid_mask == substrate | glass`
- `test_gravity_vector_is_vertical_z`: 推奨 gravity == (0, 0, -9.81)
- `test_negative_substrate_depth_raises`: `substrate_depth < 0` で ValueError
- `test_substrate_filling_domain_raises`: `substrate_depth >= Lz` で ValueError

**TestAquariumGeometryPhysics**（7 件）
- `test_no_substrate_no_glass`: `substrate_depth=0 / glass_thickness=0` で全セル水
- `test_substrate_confined_to_lower_z`: 底床セルの z 中心 ≤ substrate_depth
- `test_total_volume_conserved`: `sum(cell_volumes) == Lx·Ly·Lz`（相対誤差 < 1e-10）
- `test_refinement_bottom_finer_than_top`: ratio=3.0 で `dz[-1]/dz[0] ≈ 3.0`
- `test_no_refinement_is_uniform`: ratio=1.0 で dz 等間隔
- `test_glass_on_xy_edges`: glass > セル幅のとき x/y 両端がガラス、z 端は水
- `test_x_centers_within_domain`: セル中心が (0, L) 内に収まる

**TestAquariumGeometryIntegration**（1 件）
- `test_solid_mask_shape_matches_natconvection`: `solid_mask.dtype == bool` / `ndim == 3`

## 設計判断と妥協点

### 判断: 既存 StructuredMeshProcess を `uses` でラップ

底床 refinement のロジック重複を避けるため、不等間隔メッシュは `StructuredMeshProcess`
（Phase 3 で完成済み）に完全委譲する。`stretch_z=(ratio, 1.0)` の片側幾何級数を採用し、
z=0（底床側）が最細、z=Lz（水面側）が最粗になる。

### 判断: ガラス形状はまだ完全な境界条件ではない

Phase 6.2a では `glass_mask` は "ガラス相当の固体セル" として `solid_mask` に合流する。
Phase 6.3b（外部フィルター統合例）以降で、外部 Robin BC を受け取れるよう拡張する予定。

### 判断: `gravity` は `(0, 0, -9.81)` を返却（NC の既定 y-down と異なる）

`docs/roadmap-aquarium.md` の座標系に合わせ、z 軸を鉛直方向とした。
`NaturalConvectionInput.gravity=(0, 0, -9.81)` として渡す。`NaturalConvection` 側は
`gx, gy, gz = inp.gravity` で全 3 成分を扱うため追加修正は不要。

### 妥協点

- 底床の不均一厚（傾斜・局所深め）は未対応。Phase 6.2+ でユーザー指定マスクを
  受け入れる拡張が必要。
- ガラスは x/y 4 面のみ。z 面（底面・水面）はガラスとしない（底面=底床、水面=自由表面）。

## 動作確認

- `python -m pytest tests/test_aquarium_geometry.py -v` → 14 passed
- `ruff check xkep_cae_fluid/ tests/` → All checks passed
- `ruff format --check xkep_cae_fluid/ tests/` → 57 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反 0 件（登録プロセス 9）

## 次に着手すべきタスク

### Phase 6.2b（status-24 予定）

- [ ] `HeaterProcess` 新設（定熱流束 + 定温ヒステリシスの 2 モード）
- [ ] `examples/aquarium_heater_natural_convection.py` で Geometry + Heater + NC デモ

### Phase 6.3 以降（既存）

- Phase 6.3a: `InternalFaceBC` 実装（外部フィルター吐出）
- Phase 6.0 並行: `natural_convection/assembly.py` d 係数見直し（mu=1.85e-5 課題）

## 設計上の懸念・運用メモ

- **不等間隔と NC ソルバーの整合**: `StructuredMeshProcess` の `dx/dy/dz` 配列は
  `NaturalConvectionInput` と同じ記法で互換性があることを `NaturalConvectionInput`
  `@dataclass(frozen=True)` の `__post_init__` のない現仕様で確認済。
- **PR 粒度**: Phase 6.2a（Geometry）と Phase 6.2b（Heater + Demo）を 2 コミットで分割する
  予定。README/status/roadmap は別コミット（ドキュメントのみ）で最後にまとめる。

## 変更ファイル

- **新規**: `xkep_cae_fluid/aquarium/__init__.py`
- **新規**: `xkep_cae_fluid/aquarium/geometry.py`
- **新規**: `docs/design/aquarium-geometry.md`
- **新規**: `tests/test_aquarium_geometry.py`
- **新規**: `docs/status/status-23.md`（本ファイル）
- **編集**: `docs/design/README.md`（aquarium-geometry.md 追加）
