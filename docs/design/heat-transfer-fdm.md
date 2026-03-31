# 3次元非定常伝熱解析 (FDM) 設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md)

## 概要

等間隔直交格子（ボクセル）上での3次元非定常伝熱解析をFDM（有限差分法）で実装する。
ガウスザイデル法による反復解法を用いる。

## 支配方程式

$$
\rho C \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + q
$$

- $\rho C$ → 熱容量 `C [J/(m³·K)]`（体積あたり）
- $k$ → 熱伝導率 `k [W/(m·K)]`
- $q$ → 体積発熱量 `q [W/m³]`
- $T$ → 温度 `T [K]`

## 離散化

### 空間離散化

等間隔直交格子（dx, dy, dz）上で中心差分:

$$
\nabla \cdot (k \nabla T) \approx \sum_{d \in \{x,y,z\}} \frac{1}{\Delta d^2} \left[ k_{i+\frac{1}{2}} (T_{i+1} - T_i) - k_{i-\frac{1}{2}} (T_i - T_{i-1}) \right]
$$

面間熱伝導率は調和平均:

$$
k_{i+\frac{1}{2}} = \frac{2 k_i k_{i+1}}{k_i + k_{i+1}}
$$

### 時間離散化

陰的オイラー法（後退オイラー）:

$$
\frac{C_{ijk}}{\Delta t} (T_{ijk}^{n+1} - T_{ijk}^{n}) = [\nabla \cdot (k \nabla T)]^{n+1} + q_{ijk}
$$

### ガウスザイデル反復

各セル (i,j,k) について:

$$
T_{ijk}^{n+1} = \frac{
  \frac{C_{ijk}}{\Delta t} T_{ijk}^{n} + q_{ijk} + \sum_{\text{nb}} \frac{k_{\text{face}}}{\Delta d^2} T_{\text{nb}}^{n+1}
}{
  \frac{C_{ijk}}{\Delta t} + \sum_{\text{nb}} \frac{k_{\text{face}}}{\Delta d^2}
}
$$

## 領域定義

- `Lx, Ly, Lz`: 領域サイズ [m]
- `nx, ny, nz`: 各方向のセル数
- `dx = Lx/nx`, `dy = Ly/ny`, `dz = Lz/nz`

## 材料分布（セルごと配列）

| 変数 | 形状 | 単位 | 説明 |
|------|------|------|------|
| `k` | (nx, ny, nz) | W/(m·K) | 面内熱伝導率 |
| `C` | (nx, ny, nz) | J/(m³·K) | 体積熱容量 (ρCp) |
| `q` | (nx, ny, nz) | W/m³ | 体積発熱量 |
| `T0` | (nx, ny, nz) | K | 初期温度 |

## 境界条件

各面（x-, x+, y-, y+, z-, z+）に対して以下の3種:

| 種別 | パラメータ | 説明 |
|------|-----------|------|
| `dirichlet` | `T_wall` | 温度固定境界 |
| `neumann` | `q_flux` | 熱流束指定境界 (q_flux > 0 = 流入) |
| `adiabatic` | なし | 断熱境界 (q_flux = 0) |
| `robin` | `h_conv`, `T_inf` | 対流熱伝達境界 q = h(T_inf - T_surface) |

### Robin境界条件の離散化

壁面からセル中心までの熱抵抗 $d/(2k)$ と対流抵抗 $1/h$ を合成:

$$
U_{\text{eff}} = \frac{2kh}{2k + hd}
$$

体積あたりの寄与:

$$
a_{\text{bc}} = \frac{U_{\text{eff}}}{d}, \quad \text{flux}_{\text{bc}} = a_{\text{bc}} \cdot T_\infty
$$

## プロセス設計

### HeatTransferFDMProcess

- **カテゴリ**: `SolverProcess[HeatTransferInput, HeatTransferResult]`
- **入力**: `HeatTransferInput` — 領域・材料・境界・時間パラメータ
- **出力**: `HeatTransferResult` — 温度場・収束履歴・時間履歴

## テスト計画

### API テスト (TestHeatTransferFDMAPI)
- 入力バリデーション
- Process契約準拠（ProcessMeta, document_path）

### 物理テスト (TestHeatTransferFDMPhysics)
- 1D定常伝熱（解析解との比較）
- 1D非定常伝熱（半無限固体の解析解）
- エネルギー保存
- 断熱境界の検証
