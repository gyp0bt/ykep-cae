# 汎用スカラー輸送解析 (FDM) 設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← 水槽ロードマップ](../roadmap-aquarium.md)

## 概要

既存流体場の上で任意のスカラー量（CO2/O2 濃度、トレーサー、塩分、汚染物質等）を
輸送する汎用ソルバーを定義する。3次元等間隔直交格子上で対流-拡散-ソース方程式を
陰的に解き、`ScalarTransportProcess` として提供する。

Phase 6.1a の位置付けは水槽設計 CAE（Phase 6）での CO2/O2 輸送の基盤。
`NaturalConvectionInput` のエネルギー方程式と同じ離散化原理を用い、
水槽固有の機能（生体反応、ガス交換、気泡プルーム）からスカラーを供給する経路を
あらかじめ抽象化しておく。

## 支配方程式

$$
\frac{\partial (\rho \phi)}{\partial t}
  + \nabla \cdot (\rho \mathbf{u} \phi)
  = \nabla \cdot (\Gamma \nabla \phi)
  + S_\phi
$$

- $\phi$: 輸送スカラー（質量分率、濃度 [mol/m³] 等）
- $\rho$: 密度 [kg/m³]（`NaturalConvectionInput.rho` と共通）
- $\Gamma$: スカラー拡散係数 [kg/(m·s)]（`diffusivity`）
- $S_\phi$: 体積ソース項 [単位/(m³·s)]（光合成/CO2 添加等）

流速場は外部（既存 `NaturalConvectionProcess` 等）から与えられる前提。
Phase 6.1b で `NaturalConvection` への統合（`extra_scalars`）を実装する。

## 離散化

### 空間離散化

- 拡散項: 中心差分（既存 `heat_transfer` と同一）
- 対流項: 1次風上差分（TVD 遅延補正は Phase 6.1b 以降で対応予定）
- 面間拡散係数: 調和平均

### 時間離散化

- 陰的 Euler（BDF2 は Phase 6.1b 以降で対応予定）
- 定常: `dt=0.0`

### 境界条件

| 種別 | パラメータ | 説明 |
|------|-----------|------|
| `dirichlet` | `value` | φ 固定 |
| `neumann` | `flux` | Γ∂φ/∂n 指定（正=流入） |
| `adiabatic` | なし | ゼロ勾配（`flux=0` と等価） |
| `robin` | `h_mass`, `phi_inf` | J = h_mass·(φ_inf - φ_surface)（ガス交換/ヘンリー則） |

Robin BC の離散化は既存 `heat_transfer.solver` の `_bc_coefficients` と同形:

$$
U_\text{eff} = \frac{2\Gamma h_\text{mass}}{2\Gamma + h_\text{mass} \cdot d}
$$

## データ契約

### ScalarFieldSpec

個々のスカラー場の物性・初期値をまとめたもの。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `name` | `str` | スカラー名（"CO2", "O2", "tracer" 等） |
| `diffusivity` | `float` | 拡散係数 Γ [kg/(m·s)] |
| `phi0` | `np.ndarray (nx,ny,nz)` | 初期スカラー場 |
| `source` | `np.ndarray \| None` | 体積ソース項 S_φ |

### ScalarBoundaryCondition / ScalarBoundarySpec

enum + frozen dataclass（`FluidBoundarySpec` と同じ構造）。

### ScalarTransportInput

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `Lx, Ly, Lz` | `float` | 領域サイズ |
| `nx, ny, nz` | `int` | セル数 |
| `rho` | `float` | 密度（非圧縮性仮定） |
| `u, v, w` | `np.ndarray (nx,ny,nz)` | 外部速度場 |
| `field` | `ScalarFieldSpec` | 輸送スカラー仕様 |
| `solid_mask` | `np.ndarray \| None` | 固体マスク（速度強制 0） |
| `bc_xm..bc_zp` | `ScalarBoundarySpec` | 各面の BC |
| `dt`, `t_end` | `float` | 時間パラメータ（`dt=0.0` で定常） |
| `max_iter`, `tol` | `int, float` | 内部反復 |

### ScalarTransportResult

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `phi` | `np.ndarray (nx,ny,nz)` | 輸送後スカラー場 |
| `converged` | `bool` | 内部反復が収束したか |
| `n_timesteps` | `int` | 実行ステップ数（非定常時） |
| `residual_history` | `tuple[float, ...]` | 内部反復残差 |
| `elapsed_seconds` | `float` | 計算時間 |

## プロセス設計

### ScalarTransportProcess

- **カテゴリ**: `SolverProcess[ScalarTransportInput, ScalarTransportResult]`
- **依存**: なし（既存 `core/strategies` の再利用は Phase 6.1b で検討）
- **stability**: experimental

使い方の流れ:

1. 外部ソルバー（`NaturalConvectionProcess` 等）が `(u, v, w)` を返す
2. 水槽固有ソース（生体反応・気泡プルーム）が `source` を提供
3. 本プロセスが陰的 Euler で 1 ステップ（もしくは `dt=0` で定常）進める

## テスト計画

### API テスト（`TestScalarTransportAPI`）
- `ProcessMeta` / `document_path` 契約
- `process()` が `ScalarTransportResult` を返す
- ゼロ速度・ゼロソース・全面断熱で初期値維持

### 収束テスト（`TestScalarTransportConvergence`）
- 1D 純拡散 定常解析解（Dirichlet 両端、線形温度分布に対応する φ 分布）

### 物理テスト（`TestScalarTransportPhysics`）
- 1D 純対流: 一定速度で移流される矩形波の質量保存（合計 Σφ が不変、< 1e-10）
- 2D 対流-拡散: 体積ソース注入から下流で検出される濃度の正当性

## Phase 6.1b 以降での拡張

- `extra_scalars: list[ScalarFieldSpec]` を `NaturalConvectionInput` に追加し、
  SIMPLE 外部反復内で RC 面速度を共有
- TVD 遅延補正（van Leer / Superbee）の移植
- BDF2 時間積分
- 非直交補正（`CorrectedDiffusionStrategy` の再利用）
- Robin BC のガス交換プリセット（ヘンリー則）
