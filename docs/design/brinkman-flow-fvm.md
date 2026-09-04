# 2D Brinkman 補正 Navier-Stokes (FVM) 設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← ロードマップ](../roadmap.md)

## 目的

薄い流路（厚さ h）を深さ平均した 2 次元場で扱う **Brinkman 補正付き非圧縮 Navier-Stokes**
を、同位置格子 FVM + Newton–Krylov で定常解として解く。
U ターン流路（厚さ場で流路を表現）と平板（全域同一厚さ）を題材に、
**メッシュ細分化・流速増加に対する収束破綻を再現・分析する**ことが第一目的。

## 支配方程式

$$
\rho (\mathbf{u}\cdot\nabla)\mathbf{u}
 = -\nabla p + \mu \nabla^2 \mathbf{u} - \frac{12\,\mu_b}{h^2}\mathbf{u}
,\qquad \nabla\cdot\mathbf{u} = 0
$$

- $h(x,y)$: 厚さ場。流路部 $h=10^{-3}$、閉塞部 $h=10^{-5}$（Brinkman 貫通項が $10^4$ 倍になり実質的な壁）
- $12\mu_b/h^2$: Hele-Shaw 平行平板の深さ平均抵抗（`brinkman_factor=12.0` で変更可）
- 平板モデルは全域 $h=10^{-3}$、U ターンモデルは経路のみ $h=10^{-3}$

## 離散化（`assembly.py`）

- 等間隔直交格子、同位置配置（u, v, p をセル中心）
- 未知数ベクトル $x=[u, v, p]$（各 $n_x n_y$）
- 対流面値: `first_order_upwind` / `second_order_upwind`（勾配は Green–Gauss、Venkatakrishnan リミター）
- 拡散: 中心差分
- 連続式: 面質量流束に **Rhie–Chow 補間**（$d_f = V/a_P$、$a_P$ は運動量対角）を適用しチェッカーボードを抑制
- 境界: `BoundaryPatch(kind, mask, ...)` の列で指定。領域 4 辺の境界面中心で座標マスク `mask(x, y) -> bool`
  を評価し、True の面に種別を割り当てる（後のパッチ優先、未指定は no-slip 壁）。`boundaries=None` なら
  `geometry` + `u_inlet` から従来の「左壁上部 速度 inlet / 左壁下部 圧力 outlet」を生成

| 種別 | 速度 | 圧力 | 備考 |
|---|---|---|---|
| `WALL` | Dirichlet 0 | ゼロ勾配 | 既定 |
| `VELOCITY_INLET` | 内向き法線方向に一様 `velocity` | ゼロ勾配 | |
| `MASS_FLOW_INLET` | 一様 $u_n = \dot m / (\rho \sum_f h_f A_f)$ | ゼロ勾配 | `mass_flow` [kg/s] は厚さ $h$ 込みの 3 次元値。$h_f$ は隣接セルの厚さ |
| `PRESSURE_OUTLET` | ゼロ勾配 | Dirichlet `pressure` | |

  マスク補助: `west_span(y0, y1)`, `east_span(y0, y1, lx)`, `south_span(x0, x1)`, `north_span(x0, x1, ly)`。
  質量流量を固定したまま inlet の位置・サイズ・壁を変える探索（冷却流路設計）に使う。
  連続式は $\partial_i u_i = 0$（$h$ を含まない）なので、`mass_in/mass_out` は単位深さの値 [kg/s]（= $\dot m / h_\mathrm{in}$）で報告する

## 非線形反復（`solver.py`）

| 項目 | 設定 |
|---|---|
| 外側 | Newton（残差は選択スキーム、ヤコビアンは常に **1 次風上**で解析的に組む） |
| 内側 | GMRES（`jacobian="jfnk"`: 有限差分 $J_2 v$、`"defect_correction"`: $J_1 \delta=-R_2$） |
| 前処理 | `scipy.sparse.linalg.splu` による $J_1$ の完全 LU |
| 擬似時間 | 対角に $\rho V/\Delta\tau$ を加算。$\Delta\tau = \mathrm{CFL}\,\Delta x/\max(|u|+|v|,\ r\,U_\mathrm{in})$（$r$=`velocity_floor_ratio`）、SER で CFL を成長 |
| 局所/大域 Δτ | `pseudo_time_mode`: `LOCAL`（セルごとの Δτ、既定）/ `GLOBAL`（局所 Δτ の全セル最小値を一律に使用） |
| RC と擬似時間 | 既定は $d_f = V/a_P$（残差は Δτ に依存しない）。`rhie_chow_pseudo_time=True` で $d_f = V/(a_P + \rho V/\Delta\tau)$ とする再現用変種（残差が Δτ に依存し、収束解も Δτ 場に依存する）。結果の `steady_residual_ratio` に Δτ 非依存の定常残差を常に報告 |
| 陰的緩和 | 運動量対角を $a_P/\alpha_u$ に置換（残差は変えない） |

収束判定は $\|R\|_2/\|R_0\|_2 <$ `newton_tol`。発散（NaN / 残差爆発 / GMRES 破綻 / 反復上限）は
`converged=False` と `failure_reason` で報告する（数値の捏造禁止）。

## 入出力

- `BrinkmanFlowInput`: 形状・格子・物性・厚さ場・境界パッチ（`boundaries`）・スキーム・ソルバー設定
- `BrinkmanFlowResult`: u, v, p, 残差履歴, CFL 履歴, 収束フラグ, 失敗理由, 質量収支
- `UTurnThicknessProcess`（PreProcess）: `"flat"` / `"uturn"` の厚さ場を生成

## 関連

- 実験スクリプト: `experiments/brinkman_uturn/sweep.py`
- テスト: `tests/test_brinkman_flow.py`
