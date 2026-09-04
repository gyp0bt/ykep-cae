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
- 境界: 左壁上部に速度 inlet、左壁下部に圧力 outlet（p=0, 速度ゼロ勾配）、他は no-slip

## 非線形反復（`solver.py`）

| 項目 | 設定 |
|---|---|
| 外側 | Newton（残差は選択スキーム、ヤコビアンは常に **1 次風上**で解析的に組む） |
| 内側 | GMRES（`jacobian="jfnk"`: 有限差分 $J_2 v$、`"defect_correction"`: $J_1 \delta=-R_2$） |
| 前処理 | `scipy.sparse.linalg.splu` による $J_1$ の完全 LU |
| 擬似時間 | 対角に $\rho V/\Delta\tau$ を加算。$\Delta\tau = \mathrm{CFL}\,\Delta x/\max(|u|+|v|,\ r\,U_\mathrm{in})$（$r$=`velocity_floor_ratio`）、SER で CFL を成長 |
| 陰的緩和 | 運動量対角を $a_P/\alpha_u$ に置換（残差は変えない） |

収束判定は $\|R\|_2/\|R_0\|_2 <$ `newton_tol`。発散（NaN / 残差爆発 / GMRES 破綻 / 反復上限）は
`converged=False` と `failure_reason` で報告する（数値の捏造禁止）。

## 入出力

- `BrinkmanFlowInput`: 形状・格子・物性・厚さ場・BC・スキーム・ソルバー設定
- `BrinkmanFlowResult`: u, v, p, 残差履歴, CFL 履歴, 収束フラグ, 失敗理由, 質量収支
- `UTurnThicknessProcess`（PreProcess）: `"flat"` / `"uturn"` の厚さ場を生成

## 関連

- 実験スクリプト: `experiments/brinkman_uturn/sweep.py`
- テスト: `tests/test_brinkman_flow.py`
