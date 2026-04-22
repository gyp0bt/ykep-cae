# AquariumFilterProcess + InternalFaceBC 設計仕様（Phase 6.3a）

[← README](../../README.md) | [← 設計文書索引](README.md) | [← 水槽ロードマップ](../roadmap-aquarium.md)

## 概要

水槽の外部フィルター（エーハイム相当）による強制対流を、水槽内部の任意セル集合に
`InternalFaceBC` として流し込むための 2 層設計:

1. **`InternalFaceBC` データ契約**（`natural_convection/data.py`）
   任意セル集合（マスク指定）に強制速度または圧力基準を適用する BC。
2. **`AquariumFilterProcess`**（`aquarium/filter.py`）
   バウンディングボックス + 流量 Q [L/h] + 吐出方向から
   `InternalFaceBC` のペア（INLET/OUTLET）を自動構築する PreProcess。

既存の `FluidBoundarySpec` は領域外面 6 面のみを扱うが、本 Phase で水槽内部の
任意セル（吐出ノズル等）に BC を適用できるようになる。

## `InternalFaceBC`

### 種別（`InternalFaceBCKind`）

| 種別 | 意味 | 運動量 | 圧力補正 | エネルギー |
|------|------|--------|---------|-----------|
| `INLET` | 水槽への吐出（領域へ流入） | ペナルティで `velocity` 強制 | `p'=0` ピン留め | `temperature` 指定時ペナルティ |
| `OUTLET` | 水槽からの吸入（領域から流出） | 変更なし（ゼロ勾配相当） | `p'=0` ピン留め（基準） | 変更なし |

### フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `InternalFaceBCKind` | — | INLET または OUTLET |
| `mask` | `np.ndarray (nx, ny, nz) bool` | — | BC 適用セル |
| `velocity` | `tuple[float, float, float]` | `(0, 0, 0)` | INLET: 強制速度ベクトル [m/s] |
| `temperature` | `float | None` | `None` | INLET: 温度拘束 [K]（None で自由） |
| `pressure` | `float` | `0.0` | OUTLET: 基準圧力値（参考情報） |
| `label` | `str` | `""` | 識別子（ログ・デバッグ用） |

### ペナルティ係数の選定

- 固体マスク: `1e30`（速度/圧力厳密ゼロ化）
- INLET 運動量: `1e25`（速度強制）
- INLET 温度: `1e25`（温度強制）
- INLET/OUTLET 圧力補正: `1e30`（`p'=0` ピン留め）

運動量・温度の INLET ペナルティを `1e25` に抑えたのは、時間項 `ρ/dt`・対流・拡散に
対して十分支配的でありつつ、行列の条件数を極端に悪化させないためである。

### 複数 OUTLET と圧力基準

内部 OUTLET が存在する場合、既存の「最初の流体セルを圧力基準に固定」ロジックは
省略される（OUTLET が既に `p'=0` ピン留めしているため）。

## `AquariumFilterProcess`

### 用語

| 用語 | 意味 |
|------|------|
| **inflow** | 水槽（CFD 領域）への流入 = フィルターのリターン（水槽へ吐出する側） |
| **outflow** | 水槽（CFD 領域）からの流出 = フィルターのインテーク（水槽から吸入する側） |

この命名を使うことで、内部的な CFD の `INLET`/`OUTLET` と直感的に対応する
（inflow → `InternalFaceBC(INLET)`、outflow → `InternalFaceBC(OUTLET)`）。

### 入力: `AquariumFilterInput`

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `x_centers`, `y_centers`, `z_centers` | `np.ndarray` | — | セル中心座標 [m] |
| `dx`, `dy`, `dz` | `np.ndarray` | — | セル幅 |
| `inflow_geometry` | `NozzleGeometry` | — | 吐出ノズルのバウンディングボックス |
| `outflow_geometry` | `NozzleGeometry` | — | 吸入ノズルのバウンディングボックス |
| `flow_rate_lph` | `float` | 440.0 | 流量 Q [L/h]（エーハイム 2213 相当） |
| `inflow_direction` | `tuple[float, float, float]` | `(1, 0, 0)` | 吐出方向（単位化される） |
| `inflow_temperature_K` | `float \| None` | `None` | 吐出水温 [K]（None で温度拘束なし） |
| `label` | `str` | `"filter"` | 識別子 |

### 出力: `AquariumFilterResult`

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `inflow_bc` | `InternalFaceBC` | INLET BC |
| `outflow_bc` | `InternalFaceBC` | OUTLET BC |
| `inflow_mask`, `outflow_mask` | `np.ndarray` | ノズル占有セル |
| `inflow_velocity` | `tuple[float, float, float]` | 計算された吐出速度ベクトル [m/s] |
| `flow_rate_m3s` | `float` | 流量 [m³/s]（L/h 換算後） |
| `inflow_area_m2` | `float` | 吐出方向への投影面積 [m²] |

### 速度計算

```
Q [m³/s]  = flow_rate_lph * 1e-3 / 3600
A [m²]    = Σ (マスクセルの方向投影面積) = Σ (|dir_x|·dy·dz + |dir_y|·dx·dz + |dir_z|·dx·dy)
|v_in|    = Q / A
v_in      = |v_in| · (inflow_direction / ‖inflow_direction‖)
```

軸に沿った吐出（`direction = (±1, 0, 0)` など）では、素朴な直交断面積に一致する。
斜め方向でもマスクセルの軸投影面積の和で連続に扱える。

## 使用例

```python
from xkep_cae_fluid.aquarium import (
    AquariumFilterInput,
    AquariumFilterProcess,
    AquariumGeometryInput,
    AquariumGeometryProcess,
    NozzleGeometry,
)
from xkep_cae_fluid.natural_convection import NaturalConvectionInput

geom = AquariumGeometryProcess().process(AquariumGeometryInput())

filt_res = AquariumFilterProcess().process(
    AquariumFilterInput(
        x_centers=geom.x_centers,
        y_centers=geom.y_centers,
        z_centers=geom.z_centers,
        dx=geom.dx, dy=geom.dy, dz=geom.dz,
        inflow_geometry=NozzleGeometry(
            x_range=(0.85, 0.90), y_range=(0.14, 0.16), z_range=(0.38, 0.44)
        ),
        outflow_geometry=NozzleGeometry(
            x_range=(0.00, 0.05), y_range=(0.14, 0.16), z_range=(0.02, 0.06)
        ),
        flow_rate_lph=440.0,
        inflow_direction=(-1.0, 0.0, 0.0),  # 右上から左（x-）に向けて吐出
        inflow_temperature_K=300.15,
    )
)

nc_inp = NaturalConvectionInput(
    ...,
    internal_face_bcs=(filt_res.inflow_bc, filt_res.outflow_bc),
)
```

## テスト計画

### API テスト（`TestAquariumFilterAPI`）

- ProcessMeta / document_path 契約
- 非正の流量で `ValueError`
- 範囲不正（min >= max）で `ValueError`
- ゼロ方向ベクトルで `ValueError`
- 格子範囲外ノズルで `ValueError`
- inflow/outflow マスク重複で `ValueError`

### 物理テスト（`TestAquariumFilterPhysics`）

- Q[L/h] → Q[m³/s] 換算（3600 L/h = 1e-3 m³/s）
- `|v_in| · A ≈ Q` が成立
- 指定方向ベクトルに沿った速度ベクトル
- inflow/outflow マスクが非空かつ重複なし
- `inflow_temperature_K` が BC に伝搬

### 統合テスト（`TestInternalFaceBCIntegration`）

- INLET: ソルバー実行後も強制速度が維持される（`u[mask] ≈ velocity[0]`）
- OUTLET 付近で非ゼロ速度が観測される（循環成立）
- 温度拘束が強制される
- `internal_face_bcs=()` で既存挙動（流れゼロ）
- 入口流量と出口流量が大まかに釣り合う（ratio ∈ [0.5, 1.5]）

### アセンブリテスト（`TestInternalFaceBCAssembly`）

- `NaturalConvectionInput.internal_face_bcs` のデフォルトが空 tuple
- `InternalFaceBCKind` の 2 値 (inlet/outlet)
- 空マスクは no-op（発散せず）

## 設計判断と妥協点

### 判断: セル領域ベースのペナルティ方式

厳密な「面ベース」境界条件ではなく、セル集合に対するペナルティ方式を採用した。
CVFEM / 食い違い格子上の内部面ディリクレは実装コストが高く、Phase 6.3a の目的
（外部フィルター循環の基本動作検証）に対して過剰である。エーハイム相当の
流量レベル（Q=440〜1000 L/h、v~0.05 m/s、Re~5000 程度）では、数セル幅のノズル
領域で速度を強制する近似で十分な物理精度が得られる。

### 判断: INLET の `temperature` は Optional

ヒーターが水槽内にある場合、フィルターの吐出温度は常にヒーター通過後の値と
一致しない（外部フィルター通過時の放熱があるため）。`temperature=None` なら
「フィルター通過による温度変化は無視、水槽の温度場に任せる」、指定ありなら
「この温度で吐出される（ヒーター併用時の吐出温度設定）」という 2 ケースを扱う。

### 判断: Process を純粋関数として設計

`HeaterProcess` と同じく、`AquariumFilterProcess` は状態を持たない純粋関数。
時間経過や流量変動は外部ループで `AquariumFilterInput` を更新して再呼出しする。

### 妥協点

- **ペナルティ係数 1e25**: 時間項 `ρ/dt` が `1e25` オーダーに達するような
  極端な `dt → 0` では支配しきれない。通常の水槽シミュレーション
  （dt=0.01〜1s, ρ=1000）では 1e5〜1e7 オーダーなので十分支配的。
- **質量保存**: 入口/出口のセル以外では連続の式が満たされる設計だが、
  入口/出口自体は陽的なソース/シンクなので、ドメイン全体 L2 残差は
  入口流量分だけ常に残る。これは物理的に正当であり、バグではない。
- **面ノズル**: バウンディングボックス指定のみ対応（円柱パイプの実ノズル形状は
  Phase 6.4 以降の多孔質媒体と組合せて拡張）。

## 依存プロセス

`AquariumFilterProcess.uses = []`（`AquariumGeometryResult` の配列を受け取る薄い
Process なので、プロセス呼出しはしない）。

## 将来拡張（Phase 6.3b 以降）

- `examples/aquarium_filter_circulation.py`: 吐出 + 吸入 + ヒーター
- 質量保存の定量検証（流入 = 流出、誤差 < 1%）
- 円柱ノズル形状の追加（`NozzleGeometry` をプロトコル化）
- CO2 拡散器の吐出としての流用（Phase 6.7b）
