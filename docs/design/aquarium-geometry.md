# AquariumGeometryProcess 設計仕様（Phase 6.2a）

[← README](../../README.md) | [← 設計文書索引](README.md) | [← 水槽ロードマップ](../roadmap-aquarium.md)

## 概要

持続的水槽設計 CAE（Phase 6）の最初のジオメトリ生成プロセス。
90×30×45 cm（W×D×H）の水槽を想定し、底床（砂層）マスク・ガラス壁マスク・水領域マスクを
構造化メッシュ上に構築する `PreProcess`。

既存の `StructuredMeshProcess` を `uses` で宣言して再利用する。底床近傍や水面近傍の
refinement をストレッチング比率で指定できるようにし、Phase 6.2b（ヒーター）以降で
利用する `solid_mask` / `q_vol` 形状と完全互換な numpy 配列を返却する。

## 座標系

- x: 水槽の幅（Width, デフォルト 0.9 m）
- y: 水槽の奥行き（Depth, デフォルト 0.3 m）
- z: 水槽の高さ（Height, デフォルト 0.45 m）、鉛直方向（重力は -z）

ロードマップ `docs/roadmap-aquarium.md` の座標系に合わせ、推奨重力ベクトルは
`(0, 0, -9.81)` m/s² を返却する。`NaturalConvectionInput.gravity` にそのまま渡す。

## 入力: AquariumGeometryInput

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `Lx`, `Ly`, `Lz` | `float` | 0.9, 0.3, 0.45 | 水槽内部領域 [m] |
| `nx`, `ny`, `nz` | `int` | 36, 12, 30 | 各方向のセル数 |
| `substrate_depth` | `float` | 0.05 | 底床（砂層）厚さ [m]（0 で底床なし） |
| `substrate_refinement_ratio` | `float` | 1.0 | z 方向ストレッチング比（>1 で底床側を細かく、=1 で等間隔） |
| `glass_thickness` | `float` | 0.0 | ガラス壁厚さ [m]（0 で厚みなし＝純水領域のみ） |
| `origin` | `tuple[float, float, float]` | (0, 0, 0) | 原点座標 |

### ストレッチング規則

- `substrate_refinement_ratio == 1.0`: z 方向等間隔。
- `substrate_refinement_ratio > 1.0`: z 方向に幾何級数ストレッチ
  `stretch_z=(ratio, 1.0)` を採用（底床が最細、水面側が最粗）。

x, y 方向は常に等間隔。

### 底床マスク

底床は z 方向下端（`z = origin_z` 〜 `z = origin_z + substrate_depth`）に配置される。
`substrate_depth == 0.0` の場合は空マスク。

### ガラスマスク

`glass_thickness > 0` のとき、x/y の両端 1 〜 N 層（厚さが届くまで）をガラスセルとする。
`glass_thickness == 0.0` の場合は空マスク。

### 水領域マスク

`water_mask = ~(substrate_mask | glass_mask)` で計算。

## 出力: AquariumGeometryResult

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `mesh` | `MeshData` | 構造化メッシュデータ |
| `dx`, `dy`, `dz` | `np.ndarray` | 各方向セル幅配列 |
| `x_centers`, `y_centers`, `z_centers` | `np.ndarray` | 各方向セル中心座標 |
| `substrate_mask` | `np.ndarray (nx, ny, nz) bool` | 底床セル |
| `glass_mask` | `np.ndarray (nx, ny, nz) bool` | ガラス壁セル |
| `water_mask` | `np.ndarray (nx, ny, nz) bool` | 水セル |
| `solid_mask` | `np.ndarray (nx, ny, nz) bool` | `substrate_mask \| glass_mask`（`NaturalConvectionInput.solid_mask` 互換） |
| `gravity` | `tuple[float, float, float]` | 推奨重力ベクトル (0, 0, -9.81) |

## 依存プロセス

- `StructuredMeshProcess`（`uses` 宣言）— 構造化メッシュ生成

## 使用例

```python
from xkep_cae_fluid.aquarium import AquariumGeometryProcess, AquariumGeometryInput

inp = AquariumGeometryInput(
    Lx=0.9, Ly=0.3, Lz=0.45,
    nx=30, ny=10, nz=25,
    substrate_depth=0.05,
    substrate_refinement_ratio=2.0,
)
res = AquariumGeometryProcess().process(inp)

# NaturalConvectionInput に直接渡せる
from xkep_cae_fluid.natural_convection import NaturalConvectionInput

nc_input = NaturalConvectionInput(
    Lx=inp.Lx, Ly=inp.Ly, Lz=inp.Lz,
    nx=inp.nx, ny=inp.ny, nz=inp.nz,
    # ...
    solid_mask=res.solid_mask,
    gravity=res.gravity,
    # ...
)
```

## テスト計画

### API テスト（`TestAquariumGeometryAPI`）

- ProcessMeta / document_path 契約
- process() が AquariumGeometryResult を返し、形状が入力通り
- `solid_mask == substrate_mask | glass_mask` が常に成立

### 物理/整合性テスト（`TestAquariumGeometryPhysics`）

- `substrate_depth=0`: substrate_mask が全 False、水領域が全セル
- `substrate_depth=0.05` かつ `Lz=0.45`: 底床セルの z 座標が全て ≤ 0.05
- `substrate_refinement_ratio > 1`: 底床セルの dz が水面側より小さい
- `glass_thickness > 0`: x/y 端セルがガラス、内部セルが水
- セル体積の合計 = Lx·Ly·Lz（保存）

## 将来拡張（Phase 6.2b 以降）

- 底床の不均一厚（傾斜、局所深め）対応 — ユーザーの指定マスクを受け入れる
- ガラスの熱伝導率・外部 Robin BC 受け渡し — Phase 6.3b 以降
- 給排水ポート位置の組み込み — Phase 6.3a `InternalFaceBC`
