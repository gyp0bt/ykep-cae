# status-30: Brinkman 流路の座標マスク境界条件・質量流入・領域内マニホールド・随伴設計感度（冷却流路設計の前段）

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/brinkman-flow-fvm.md) | [nsb/README](../../nsb/README.md) | [前: status-28](status-28.md)（status-29 は押出解析、別ブランチ）

**日付**: 2026-09-04
**ブランチ**: `claude/2dfvm-brinkman-convergence-tn39cz`
**テスト数**: 322（status-28 の 308 + 14: `tests/test_brinkman_flow.py` +10、`tests/test_nsb.py` +2、`tests/test_nsb_adjoint.py` +2。`pytest tests/` で 321 passed / 1 xfailed）
**契約違反**: 0 件（登録プロセス 13）

## 目的

用途は上下アルミプレートを熱伝達コンダクタンスで結ぶ冷却流路の設計（熱側は既存、流量場を与えれば解ける。
外側から autodiff でラップする予定）。そのために

1. 境界条件を**座標マスク関数**で指定できるようにする（左壁固定をやめ、4 辺のどこにでも inlet/outlet を置ける）
2. **質量流入境界**を追加し、流量を固定したまま inlet の位置・サイズを変える探索を可能にする

## 実装

### `xkep_cae_fluid/brinkman_flow/data.py`

- `BoundaryKind`（WALL / VELOCITY_INLET / MASS_FLOW_INLET / PRESSURE_OUTLET）
- `BoundaryPatch(kind, mask, velocity, mass_flow, pressure, name)`。`mask(x, y) -> bool` を 4 辺の境界面中心で評価。後のパッチ優先、未指定は WALL
- マスク補助 `west_span / east_span / south_span / north_span`
- `BrinkmanFlowInput.boundaries`（None なら従来の geometry + u_inlet から生成。既存テスト・実験は無変更で通る）

### `xkep_cae_fluid/brinkman_flow/assembly.py`

- `BoundarySide`（辺ごとの種別・流入速度・outlet 圧力）と `_resolve_boundaries`
- 質量流入: $u_n = \dot m / (\rho \sum_f h_f A_f)$（$h_f$ は隣接セル厚さ、$\dot m$ は厚さ込みの 3 次元値）でパッチ内一様
- 線形面値・拡散の Dirichlet/ゼロ勾配・圧力面値・風上セレクタ（outlet 面）・`mass_flow` を 4 辺に一般化
- 擬似時間の速度スケール `u_scale` = 最大流入速度（solver の速度下限に使用。`u_inlet` 依存を除去）
- `boundary_report()`（辺ごとの面数・流入速度・範囲）

### `nsb/`

- `BC(patches=...)` に置換（`BC.velocity_inlet / mass_flow_inlet / pressure_outlet`）。`FaceType` は `BoundaryKind` のエイリアス
- `make_case(model, refine, u_in=None, mass_flow=None, inlet_y=..., outlet_y=..., bc=None)`。uturn の厚さ場は inlet/outlet 位置に追従
- `utils.inlet_cells / inlet_velocity / inlet_mean_pressure` を任意壁対応に

## テスト

- 既定境界が従来の左壁配置と一致、質量流入の換算速度、inlet 欠落で ValueError
- **4 辺すべてにパッチ**（上壁 速度 inlet、右壁 質量流入、下壁・左壁 outlet）で 1 次風上ヤコビアンが FD と一致（列相対 1e-3）
- 上壁 質量流入 → 右壁 outlet、左壁 2 か所 質量流入（1 マスク）で収束・質量保存（`mass_in = ṁ/h`）
- nsb: 流量固定で inlet 幅を 2 倍にすると換算速度が 1/2、任意壁の質量流入で収束

## 探索デモ（`experiments/nsb/inlet_sweep.py`、flat 72×48、ṁ=0.1 kg/s 固定、outlet 左壁 y∈(0.05,0.15)）

設定: 局所 Δτ、速度下限 0.1 u_n、Stokes 初期場、α_u=1.0、cfl 0.5（status-28 の推奨構成）。
ログ `experiments/nsb/logs/inlet-sweep-flat-r1.log`、YAML `experiments/nsb/results/inlet_sweep_flat_r1.yaml`。

| inlet | u_n [m/s] | 反復 | inlet 平均圧力 [Pa] | 最大流速 [m/s] |
|---|---|---|---|---|
| 左壁 y∈(0.25,0.35)（基準） | 1.000 | 11 | 2396 | 1.29 |
| 左壁 y∈(0.30,0.35)（幅半分） | 2.000 | 13 | 2272 | 1.96 |
| 左壁 y∈(0.20,0.35)（幅 1.5 倍） | 0.667 | 10 | 2208 | 1.27 |
| 左壁 y∈(0.15,0.35)（幅 2 倍） | 0.500 | 10 | 2182 | 1.36 |
| 左壁 y∈(0.30,0.40)（上端） | 1.000 | 11 | 2785 | 1.32 |
| 上壁 x∈(0.30,0.40) | 1.029 | 11 | 2858 | 1.29 |
| 上壁 x∈(0.55,0.65) | 1.029 | 11 | 3644 | 1.28 |
| 右壁 y∈(0.25,0.35) | 1.000 | 12 | 3685 | 1.28 |

8 ケースすべて 10〜13 反復で収束（status-28 の推奨構成の効果）。流量固定では inlet 幅を変えても圧損は 1 割程度しか変わらず、
inlet を outlet から遠ざけるほど（上壁右寄り・右壁）Hele-Shaw 圧損 $\kappa U L$ の経路長 $L$ 分だけ増える。

## 領域内マニホールド（紙面垂直方向の注入・吸出）

境界を端部だけでなく領域内に置けるよう、`INTERIOR_*` 種別を追加した（マスクは**セル中心**で評価）。
深さ平均の連続式にソース $\partial_i u_i = s$ を置き、3 次元流量を $q_c = \dot m\, V_c/\sum_c h_c V_c$ でパッチ内のセルに按分する
（数理は `nsb/theory.md` §2.5）。

| 種別 | 連続式 | 運動量 | 用途 |
|---|---|---|---|
| `INTERIOR_MASS_SOURCE` | $-q_c$ | なし（面内運動量ゼロで注入。保存形の対流項が希釈を表す） | 流量指定の注入ヘッダ |
| `INTERIOR_MASS_SINK` | $+q_c$ | $+q_c u_{i,c}$（局所運動量を持ち出す） | 流量指定の吸出ヘッダ（圧力基準が別に必要） |
| `INTERIOR_PRESSURE_SINK` | $+C_c(p_c - p_\mathrm{m})$ | $+\max(C_c(p_c-p_\mathrm{m}),0)\,u_{i,c}$ | 圧力指定ヘッダ（Robin 型で圧力基準を与える。逆流は注入扱い） |

- 圧力基準（`PRESSURE_OUTLET` か `INTERIOR_PRESSURE_SINK`）が無い構成は ValueError（圧力が不定になるため）
- ヤコビアンに吸出の対角 $q^{\mathrm{out}}_c$、圧力結合 $\partial(q^{\mathrm{out}} u)/\partial p = C_c u_{i,c}$（吸出側のみ）、$\partial R^p/\partial p = C_c$ を追加。
  注入 + 流量指定吸出 + 圧力指定（吸出/逆流の両側を含む乱数場）で FD ヤコビアンと一致することをテスト
- 擬似時間の速度スケールは、境界 inlet が無い場合は注入総流量 / (ρ · 周長 4√A) で見積もる
- `mass_in / mass_out` にマニホールド分を含める（`mass_flow(st, x)`）
- nsb: `BC.interior_source / interior_sink / interior_pressure_sink`、マスク補助 `rect_mask / disk_mask`

### デモ（`experiments/nsb/manifold_demo.py`、flat 72×48、ṁ=0.1 kg/s、C=1e-4 kg/(s·Pa)、推奨構成）

ログ `experiments/nsb/logs/manifold-demo-flat-r1.log`、YAML `results/manifold_demo_flat_r1.yaml`、図 `experiments/nsb/output/manifold_*.png`。

| ケース | 反復 | 注入部平均圧力 [Pa] | 最大流速 [m/s] | m_out/m_in |
|---|---|---|---|---|
| A. 注入円板（中央）→ 左壁 outlet | 10 | 2823 | 1.28 | 1.0000 |
| B. 注入円板 → 圧力指定マニホールド（境界 outlet なし） | 9 | 2465 | 0.44 | 1.0000 |
| C. 注入 2 か所（合計 ṁ）→ 圧力指定マニホールド 1 か所 | 8 | 2354 | 0.43 | 1.0000 |

B の注入部圧力 2465 Pa は「マニホールド間の Hele-Shaw 圧損 + 吸出側の $\dot m/C = 1000$ Pa」で構成される。
境界 outlet を全く持たない構成（B, C）でも圧力基準が効いて 10 反復以内で収束した。

## マニホールドの位置・径を連続設計変数に（滑らかな窓 + 随伴感度）

- `BoundaryPatch.weight`: 領域内パッチを bool マスクの代わりに滑らかな窓 $w(x,y)\in[0,1]$ で指定。
  `smooth_disk(cx, cy, r, eps)` は $w = \tfrac12(1+\tanh((r-d)/\varepsilon))$（$\varepsilon \approx \Delta x$）。
  ソースは $w_c V_c/\sum w_c h_c V_c$ で按分。領域内パッチの重なりは加算に変更（窓の裾が重なるため）。
- `nsb/adjoint.py`:
  - `colored_fd_jacobian`: 選択スキーム（SOU + リミター + RC + マニホールド）の残差から彩色中心差分で厳密な疎ヤコビアン
    （構造格子、ステンシル半径 2、残差評価 150 回）。12×8 で密 FD と 1e-8 で一致
  - `ImplicitSolve(build_input)`: `forward(θ)`, `jacobian`, `adjoint`（$J^\top\lambda = \bar x$）, `vjp`（$\bar\theta = -(\partial R/\partial\theta)^\top\lambda$）, `gradient(θ, x, Objective)`
  - `source_mean_pressure_objective`: 注入部平均圧力（圧損）とその ∂f/∂x
- 数理は `nsb/theory.md` §8b

### 検証と最適化デモ（`experiments/nsb/manifold_optimize.py`、flat 72×48、ṁ=0.1 kg/s、C=1e-4）

θ = 吸出マニホールドの (cx, cy, r)、注入は (0.15, 0.2) r=0.05 固定、目的関数 = 注入部平均圧力。
ログ `experiments/nsb/logs/manifold-opt-flat-r1.log`、YAML `results/manifold_optimize.yaml`。

| | ∂f/∂cx | ∂f/∂cy | ∂f/∂r |
|---|---|---|---|
| 随伴 | 3331.0 | 1908.5 | −5870.0 |
| 全体解き直しの中心差分 | 3326.2 | 1897.7 | −5876.8 |
| 相対誤差 | 1.4e-3 | 5.7e-3 | 1.2e-3 |

（36×24・newton_tol=1e-10 のテストでは cx 5e-5、r 1e-5。72×48 の差は newton_tol=1e-9 と FD 刻みによる）
forward + 勾配で 3.8 s（forward 10 反復 + ヤコビアン 150 残差評価 + LU 2 回）。

射影勾配法 15 反復（r ≤ 0.08 の射影）: θ0=(0.55, 0.30, 0.04), f=2585 Pa → θ=(0.262, 0.205, 0.08), f=1419 Pa。
吸出は注入に向かって直進し、径は上限に張り付く（圧損最小化としては当然の解。冷却設計では流量分布の均一性など別の目的関数が要る）。

## autodiff でラップする際の注意（設計メモ）

- bool マスクの境界 inlet 位置は微分不能。領域内マニホールドは `weight`（滑らかな窓）で連続化済み。
  境界 inlet も面ごとの重み $w_f$ で同様に連続化できる（未実装）
- 随伴は `ImplicitSolve.vjp` として実装済み（彩色 FD の厳密ヤコビアン $J_2$ を LU）。外側 autodiff の custom VJP から
  `forward` / `vjp` を呼ぶだけでよい。$\partial R/\partial\theta$ は中心差分なので θ が多い（数百以上）場合は
  ソース配列 $q_c(\theta)$ に対する解析微分（$\partial R/\partial q$ は定数）に置き換えるとよい
- 連続式が $h$ を含まないため、$h$ で微分すると質量流入の換算 $u_n(h)$ 経由の依存だけが出る点に注意

## 次にやること

- [x] 領域内マニホールドの位置・径を `weight` で連続設計変数にし、随伴（`nsb/adjoint.py`）で勾配 → 本 status
- [ ] 境界 inlet の位置・幅も面重みで連続化する
- [ ] 冷却設計向けの目的関数（流量分布の均一性、最小流速）と ∂f/∂x を追加
- [ ] 熱ソルバーとの連携（流量場 → 熱伝達コンダクタンス → 上下プレート温度）の I/O 契約を決める
- [ ] Process ソルバー側にも Stokes 初期場・α_u=1 推奨を反映（現状は `nsb` のみ）

## ファイル

- `xkep_cae_fluid/brinkman_flow/{data,assembly,solver,__init__}.py`
- `nsb/{core,geo,utils,README}.py|md`、`nsb/theory.md`（§2.1 を任意壁・質量流入に更新）
- `nsb/adjoint.py`、`tests/test_nsb_adjoint.py`（+2）
- `tests/test_brinkman_flow.py`（+10）、`tests/test_nsb.py`（+2）
- `experiments/nsb/inlet_sweep.py` + `logs/inlet-sweep-flat-r1.log` + `results/inlet_sweep_flat_r1.yaml`
- `experiments/nsb/manifold_demo.py` + `logs/manifold-demo-flat-r1.log` + `results/manifold_*` + `output/manifold_*.png`
- `experiments/nsb/manifold_optimize.py` + `logs/manifold-opt-flat-r1.log` + `results/manifold_optimize.yaml`
- `experiments/brinkman_uturn/plot_fields.py`（タグに U が無い npz にも対応）
- `docs/design/brinkman-flow-fvm.md`（境界条件表 + 領域内マニホールド表）、`nsb/theory.md` §2.5
