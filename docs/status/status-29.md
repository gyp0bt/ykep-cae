# status-29: Brinkman 流路の座標マスク境界条件と質量流入境界（冷却流路設計の前段）

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/brinkman-flow-fvm.md) | [nsb/README](../../nsb/README.md) | [前: status-28](status-28.md)

**日付**: 2026-09-04
**ブランチ**: `claude/2dfvm-brinkman-convergence-tn39cz`
**テスト数**: 316（status-28 の 308 + 8: `tests/test_brinkman_flow.py` +6、`tests/test_nsb.py` +2。`pytest tests/` で 315 passed / 1 xfailed）
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

## autodiff でラップする際の注意（設計メモ）

- マスク関数は bool を返すので inlet 位置に対しては微分不能。位置・幅を設計変数にするなら、
  (a) 離散候補の列挙、(b) 質量流入速度を面ごとの重み $w_f \in [0,1]$（滑らかな窓関数）で按分する連続緩和、のどちらかが要る。
  (b) は `BoundaryPatch` に `weight(x, y) -> float` を足せば実装できる（未実装）
- 流量 $\dot m$、厚さ場 $h$、物性に対しては残差 $R(x; \theta)$ が滑らか（リミターは Venkatakrishnan で微分可能）なので、
  収束解での随伴 $\partial x/\partial\theta = -J^{-1} \partial R/\partial\theta$ が使える。$J_1$（1 次風上）は既にあるので、
  随伴には SOU の厳密ヤコビアン $J_2$ か、JFNK と同様に $J_1$ を前処理にした反復が要る
- 連続式が $h$ を含まないため、$h$ で微分すると質量流入の換算 $u_n(h)$ 経由の依存だけが出る点に注意

## 次にやること

- [ ] 連続緩和した inlet 重み（`weight(x, y)`）で位置・幅を連続設計変数にする
- [ ] 収束解での随伴（$J^\top \lambda = \partial f/\partial x$）を `nsb` に追加し、autodiff ラップの土台にする
- [ ] 熱ソルバーとの連携（流量場 → 熱伝達コンダクタンス → 上下プレート温度）の I/O 契約を決める
- [ ] Process ソルバー側にも Stokes 初期場・α_u=1 推奨を反映（現状は `nsb` のみ）

## ファイル

- `xkep_cae_fluid/brinkman_flow/{data,assembly,solver,__init__}.py`
- `nsb/{core,geo,utils,README}.py|md`、`nsb/theory.md`（§2.1 を任意壁・質量流入に更新）
- `tests/test_brinkman_flow.py`（+6）、`tests/test_nsb.py`（+2）
- `experiments/nsb/inlet_sweep.py` + `logs/inlet-sweep-flat-r1.log` + `results/inlet_sweep_flat_r1.yaml`
- `docs/design/brinkman-flow-fvm.md`（境界条件表）
