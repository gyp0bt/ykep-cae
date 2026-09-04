# nsb: 手元構成ミラーの Brinkman-NS 実験パッケージ

[<- README](../README.md) | [数理ノート（総和規約）](theory.md) | [設計文書（共有離散化）](../docs/design/brinkman-flow-fvm.md) | [status-28](../docs/status/status-28.md)

手元の 2D FVM Brinkman 補正 Navier-Stokes コードと**同じファイル構成・同じ制御則**で比較するための薄いレイヤ。
離散化（残差、1 次風上ヤコビアン、Rhie–Chow、境界条件）は
`xkep_cae_fluid.brinkman_flow.assembly.BrinkmanDiscretization` を共有し、
Newton + 擬似時間の制御則だけを `solver.solve_steady` に関数として書き下している。

| ファイル | 役割 |
|---|---|
| `core.py` | 型宣言: `BC`（座標マスクの境界パッチ列。`BC.velocity_inlet / mass_flow_inlet / pressure_outlet`）, `NSBSettings`, `NSBInput`, `NSBResult` |
| `solver.py` | メイン: `solve_steady`, `compute_dtau`, `solve_linear` |
| `utils.py` | ポスト処理、面値⇄セル値変換、要約、npz 保存 |
| `geo.py` | uturn / flat の厚さ場（inlet/outlet 位置に追従）、BC プリセット（速度 or 質量流量）、`run_uturn`, `run_flat`, `make_case` |
| `../main.py` | パラメータスタディ（構成 × モデル × 細分化 × 流速） |
| `adjoint.py` | 設計感度: 彩色 FD ヤコビアン `colored_fd_jacobian`、陰関数定理の VJP `ImplicitSolve`（forward / jacobian / vjp / gradient）、`Objective` |
| `theory.md` | 数理ノート: 支配方程式〜離散化〜Newton/擬似時間〜発散機構〜随伴感度を総和規約で記述 |

## `NSBSettings` の「踏んではいけない線」スイッチ

| 設定 | 既定（手元構成の推定） | 修正構成 | 影響（status-28 の実験結果） |
|---|---|---|---|
| `local_dtau` | True（局所 Δτ） | True / False | 大域 Δτ は同じ CFL で減衰が約 10 倍強く、高 CFL に寛容 |
| `velocity_floor` [m/s] | 0（下限なし） | 0.1·U_in | 下限なしだと静止・低速セルで Δτ→∞ となり Newton が素になる。停滞の主因 |
| `pseudo_time_in_residual` | True（残差に ρV(u−u_prev)/Δτ） | False | 収束判定・SER が擬似時間項込みの残差で動く。u_prev の更新が正しければ定常解は同じ |
| `sub_iters` | 1 | 1 | 1 擬似時間ステップあたりの Newton 反復数（u_prev 凍結） |
| `rc_with_pseudo_time` | False | False | RC 係数 d_f に ρV/Δτ を含める。本実装では致命的ではない |
| `cfl_init` | 0.5 | 0.5 | 局所 Δτ では 5 で発散、0.5 なら可 |
| `alpha_u` | 0.7 | 1.0 | 速度下限ありなら緩和なしが最速（uturn 36 反復 vs 102）。下限なしでは効きがモデルごとに逆転 |
| `init_field` | "zero" | "stokes" | Stokes–Brinkman 初期場で反復 25〜35% 減。対流項込み残差で作ると max\|u\| が U_in の 14 倍になるので注意 |
| `reject_growth` / `cfl_min` | 0（無効） | 0 | CFL backtracking は効かず。Δτ→0 で圧力が発散するので cfl_min が要る |

## 境界条件（座標マスク + 質量流入）

```python
from nsb import BC, NSBSettings, make_case, solve_steady
from xkep_cae_fluid.brinkman_flow import north_span, west_span

# 流量 0.1 kg/s を固定し、inlet を上壁 x∈(0.3, 0.4) に置く（outlet は左壁下部）
bc = BC(patches=(
    BC.mass_flow_inlet(north_span(0.3, 0.4, 0.4), 0.1),
    BC.pressure_outlet(west_span(0.05, 0.15)),
))
inp = make_case("flat", 1, bc=bc, settings=NSBSettings(velocity_floor=0.1, init_field="stokes"))
res = solve_steady(inp)

# 左壁 inlet の位置・幅だけ変える場合（uturn では厚さ場も追従）
inp = make_case("uturn", 1, mass_flow=0.1, inlet_y=(0.20, 0.35))
```

```python
# 領域内マニホールド（紙面垂直方向のヘッダ）: マスクはセル中心で評価
from xkep_cae_fluid.brinkman_flow import disk_mask
bc = BC(patches=(
    BC.interior_source(disk_mask(0.15, 0.2, 0.05), 0.1),                 # 注入 0.1 kg/s
    BC.interior_pressure_sink(disk_mask(0.55, 0.2, 0.05), 1e-4, p=0.0),  # 吸出 q = C (p - 0)
))
```

```python
# 位置・径を連続設計変数に: 滑らかな窓 smooth_disk(cx, cy, r, eps) を weight に渡し、随伴で dθ を得る
from nsb import ImplicitSolve, source_mean_pressure_objective
from xkep_cae_fluid.brinkman_flow import smooth_disk

def build(theta):                       # θ = (cx, cy, r) -> NSBInput
    cx, cy, r = theta
    bc = BC(patches=(
        BC.interior_source(disk_mask(0.15, 0.2, 0.05), 0.1),
        BC.interior_pressure_sink(None, 1e-4, weight=smooth_disk(cx, cy, r, eps=0.7 / 72)),
    ))
    return make_case("flat", 1, bc=bc, settings=NSBSettings(velocity_floor=0.05, init_field="stokes", alpha_u=1.0))

prob = ImplicitSolve(build)
res, x = prob.forward(theta)
f, dtheta = prob.gradient(theta, x, source_mean_pressure_objective())   # 圧損とその θ 勾配
theta_bar = prob.vjp(theta, x, x_bar)                                    # 外側 autodiff 用の VJP
```

任意の `mask(x, y) -> bool`（境界種別は 4 辺の境界面中心、領域内種別はセル中心で評価）を渡せる。領域内パッチは `weight(x, y) ∈ [0,1]` の滑らかな窓でも指定でき、重なりは加算になる。飛び飛びの複数 inlet も 1 マスクで指定でき、
その場合は合計流量を面の $h_f A_f$ で按分した一様速度になる。探索デモ: `experiments/nsb/inlet_sweep.py`、マニホールドデモ: `experiments/nsb/manifold_demo.py`、位置・径の最適化デモ: `experiments/nsb/manifold_optimize.py`。

## 使い方

```bash
python main.py --models uturn flat --refine 1 --u 0.1 1 2 --configs mine fixed \
    2>&1 | tee experiments/nsb/logs/main-$(date +%s).log
```

```python
from nsb import NSBSettings, run_uturn

inp, res = run_uturn(refine=1, u_in=2.0, settings=NSBSettings(velocity_floor=0.2))
print(res.converged, res.rel_residual, res.rel_steady_residual)
```

- テスト: `tests/test_nsb.py`
- 結果: `experiments/nsb/results/*.yaml`、ログ: `experiments/nsb/logs/`
