# HeaterProcess 設計仕様（Phase 6.2b）

[← README](../../README.md) | [← 設計文書索引](README.md) | [← 水槽ロードマップ](../roadmap-aquarium.md)

## 概要

水槽用ヒーター（棒状ヒーター相当）の体積熱源を計算する `PreProcess`。
`NaturalConvectionInput.q_vol` 経路にそのまま流し込める 3 次元配列を返却する。

2 つの運転モードに対応:

- **CONSTANT_FLUX**: 定熱流束。電力 `power_watts` を常時投入。
- **CONSTANT_TEMPERATURE**: 定温制御。`measured_T_K` と `setpoint_K` を比較し、
  ヒステリシスバンド `hysteresis_band_K` に従って ON/OFF を切替える。

## ヒステリシス制御の状態遷移

現在温度を $T$、設定温度を $T_\text{set}$、バンド幅を $\Delta T$ とすると:

| 条件 | 出力状態 |
|------|---------|
| $T \le T_\text{set} - \Delta T/2$ | **ON** |
| $T \ge T_\text{set} + \Delta T/2$ | **OFF** |
| それ以外（中間帯） | `prev_on` を維持 |

Process は状態を持たないため、中間帯のロジックは呼出し側で `prev_on` を保持して渡す。
このデザインは `extra_scalars` と同じく Process を純粋関数として保つ方針。

## ヒーター形状

バウンディングボックス指定:

```python
HeaterGeometry(
    x_range=(0.10, 0.12),   # x[m] の範囲
    y_range=(0.14, 0.16),   # y[m] の範囲
    z_range=(0.10, 0.30),   # z[m] の範囲（棒状ヒーター）
)
```

`AquariumGeometryResult` の `x_centers`, `y_centers`, `z_centers` を受け取って
範囲内のセルを `heater_mask` とする。

## 入力: HeaterInput

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `x_centers` | `np.ndarray` | — | x 方向セル中心座標 [m] |
| `y_centers` | `np.ndarray` | — | y 方向セル中心座標 [m] |
| `z_centers` | `np.ndarray` | — | z 方向セル中心座標 [m] |
| `dx`, `dy`, `dz` | `np.ndarray` | — | 各方向セル幅（ヒーター体積計算に使用） |
| `geometry` | `HeaterGeometry` | — | ヒーター形状 |
| `mode` | `HeaterMode` | `CONSTANT_FLUX` | 運転モード |
| `power_watts` | `float` | 200.0 | 定格電力 [W] |
| `setpoint_K` | `float` | 299.15 | 設定温度 [K]（26°C） |
| `hysteresis_band_K` | `float` | 1.0 | ヒステリシス幅 [K] |
| `measured_T_K` | `float` | 298.15 | 温度センサ位置の測定温度 [K] |
| `prev_on` | `bool` | `True` | 直前ステップの ON/OFF 状態 |

## 出力: HeaterResult

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `q_vol` | `np.ndarray (nx, ny, nz)` | 体積熱源 [W/m³]。ヒーター領域のみ非ゼロ、OFF 時は全ゼロ |
| `mask` | `np.ndarray (nx, ny, nz) bool` | ヒーターセルマスク |
| `on` | `bool` | 現在の ON/OFF 状態（次ステップに渡す） |
| `volume_m3` | `float` | ヒーター領域の体積合計 |

OFF 時（`on=False`）は `q_vol` は全ゼロになる。
ON 時の出力熱量は `q_vol[mask].sum() × volume_cell ≈ power_watts` となるように分配する。

## 依存プロセス

なし（`AquariumGeometryResult` からの配列入力を受け取る薄い Process）。

## 使用例

```python
from xkep_cae_fluid.aquarium import (
    AquariumGeometryInput,
    AquariumGeometryProcess,
    HeaterGeometry,
    HeaterInput,
    HeaterMode,
    HeaterProcess,
)

geom_res = AquariumGeometryProcess().process(AquariumGeometryInput())

heater_inp = HeaterInput(
    x_centers=geom_res.x_centers,
    y_centers=geom_res.y_centers,
    z_centers=geom_res.z_centers,
    dx=geom_res.dx,
    dy=geom_res.dy,
    dz=geom_res.dz,
    geometry=HeaterGeometry(
        x_range=(0.80, 0.85),
        y_range=(0.14, 0.16),
        z_range=(0.10, 0.30),
    ),
    mode=HeaterMode.CONSTANT_FLUX,
    power_watts=200.0,
)
heater_res = HeaterProcess().process(heater_inp)

# NaturalConvection に q_vol として渡す
nc_input = NaturalConvectionInput(
    ...,
    q_vol=heater_res.q_vol,
)
```

## テスト計画

### API テスト（`TestHeaterAPI`）

- ProcessMeta / document_path 契約
- `q_vol.shape == (nx, ny, nz)` と `mask.shape` が一致
- 範囲外ヒーター形状で ValueError

### 物理テスト（`TestHeaterPhysics`）

- CONSTANT_FLUX: `∫q_vol dV ≈ power_watts` が成立（相対誤差 < 1e-10）
- CONSTANT_FLUX: OFF 状態は観測されない（常に `on=True`）
- CONSTANT_TEMPERATURE: `measured_T < setpoint - band/2` → `on=True`
- CONSTANT_TEMPERATURE: `measured_T > setpoint + band/2` → `on=False`
- CONSTANT_TEMPERATURE: 中間帯では `prev_on` を維持

## 将来拡張（Phase 6.3 以降）

- 吸熱体としてのヒーター表面熱伝達 — 現在は体積熱源として近似
- 複数ヒーターの並列制御（ヒーター群） — 現在は 1 台
- PID 制御モード — 現在はヒステリシス on/off のみ
