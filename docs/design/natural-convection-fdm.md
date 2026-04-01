# NaturalConvectionFDMProcess 設計文書

[← README](../../README.md) | [← 設計文書一覧](README.md)

## 概要

3次元等間隔直交格子上の自然対流解析ソルバー。
SIMPLE法（Semi-Implicit Method for Pressure-Linked Equations）による
圧力-速度連成と Boussinesq 近似による浮力項を使用する。
固体-流体練成伝熱（Conjugate Heat Transfer）に対応。

## 離散化手法

- **空間離散化**: FDM（有限差分法）、等間隔直交格子、コロケーション配置
- **対流項**: 1次風上差分
- **拡散項**: 中心差分
- **時間積分**: 陰的オイラー法（非定常時）
- **圧力-速度連成**: SIMPLE法
- **浮力**: Boussinesq近似 `ρ ≈ ρ₀(1 - β(T - T_ref))`

## 支配方程式

### 非圧縮性 Navier-Stokes 方程式

連続の式:
```
∂u/∂x + ∂v/∂y + ∂w/∂z = 0
```

運動量方程式（Boussinesq近似）:
```
ρ₀(∂u/∂t + u·∇u) = -∂p/∂x + μ∇²u + ρ₀gₓβ(T - T_ref)
ρ₀(∂v/∂t + u·∇v) = -∂p/∂y + μ∇²v + ρ₀g_yβ(T - T_ref)
ρ₀(∂w/∂t + u·∇w) = -∂p/∂z + μ∇²w + ρ₀g_zβ(T - T_ref)
```

エネルギー方程式:
```
ρ₀Cp(∂T/∂t + u·∇T) = k∇²T
```

## SIMPLE法アルゴリズム

1. 速度場 u*, v*, w* を推定（前回の圧力場 p* を使用）
2. 圧力補正方程式を解く → p'
3. 速度を補正: u = u* + d_u * (∂p'/∂x), 同様に v, w
4. 圧力を更新: p = p* + α_p * p'
5. エネルギー方程式を解く → T
6. 収束判定（質量残差 + 各変数の残差）
7. 収束しなければ 1 へ戻る

## 固体-流体練成

- `solid_mask` で固体/流体領域を指定
- 固体領域: 速度=0、圧力方程式から除外、エネルギー方程式のみ（拡散のみ）
- 界面: 温度と熱流束の連続条件を自動適用（調和平均熱伝導率）

## 境界条件

### 速度
- **NO_SLIP**: 壁面速度=0（デフォルト）
- **SLIP**: 法線速度=0、接線応力=0
- **INLET_VELOCITY**: 指定速度
- **OUTLET_PRESSURE**: 圧力指定、速度は外挿
- **SYMMETRY**: 対称面

### 温度
- **DIRICHLET**: 温度固定
- **NEUMANN**: 熱流束指定
- **ADIABATIC**: 断熱（デフォルト）

## データスキーマ

### 入力: NaturalConvectionInput

| フィールド | 型 | 説明 |
|-----------|-----|------|
| Lx, Ly, Lz | float | 領域サイズ [m] |
| nx, ny, nz | int | セル数 |
| rho | float | 基準密度 [kg/m³] |
| mu | float | 動粘度 [Pa·s] |
| Cp | float | 比熱 [J/(kg·K)] |
| k_fluid | float | 流体熱伝導率 [W/(m·K)] |
| beta | float | 体膨張係数 [1/K] |
| T_ref | float | 基準温度 [K] |
| gravity | tuple | 重力ベクトル [m/s²] |
| solid_mask | ndarray\|None | 固体マスク |
| k_solid | ndarray\|None | 固体熱伝導率 |
| bc_* | FluidBoundarySpec | 各面の境界条件 |

### 出力: NaturalConvectionResult

| フィールド | 型 | 説明 |
|-----------|-----|------|
| u, v, w | ndarray | 速度場 [m/s] |
| p | ndarray | 圧力場 [Pa] |
| T | ndarray | 温度場 [K] |
| converged | bool | 収束判定 |
| residual_history | dict | 残差履歴 |

## 検証ベンチマーク

- 差分加熱キャビティ流れ（de Vahl Davis, 1983）: Ra = 10³, 10⁴, 10⁵
- Nusselt数、最大速度の比較

## Process契約

- **分類**: SolverProcess
- **入力型**: NaturalConvectionInput
- **出力型**: NaturalConvectionResult
- **依存**: なし（self-contained）
- **テスト紐付け**: tests/test_natural_convection.py::TestNaturalConvectionAPI
