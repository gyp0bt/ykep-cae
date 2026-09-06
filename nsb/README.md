# nsb: 手元構成ミラーの Brinkman-NS 実験パッケージ

[<- README](../README.md) | [数理ノート（総和規約）](theory.md) | [設計文書（共有離散化）](../docs/design/brinkman-flow-fvm.md) | [status-30](../docs/status/status-30.md) | [status-32（切り離し・高速化見積）](../docs/status/status-32.md)

手元の 2D FVM Brinkman 補正 Navier-Stokes コードと**同じファイル構成・同じ制御則**で比較するための薄いレイヤ。
離散化（残差、1 次風上ヤコビアン、Rhie–Chow、境界条件）は `nsb/assembly.py` の `BrinkmanDiscretization`、
境界条件・入力型は `nsb/data.py` に持ち、Newton + 擬似時間の制御則だけを `solver.solve_steady` に関数として書き下している。

## 単体で持ち出せる（xkep_cae_fluid 非依存、スナップショット）

`nsb/` ディレクトリは **numpy / scipy / pypardiso だけ**で動き、`xkep_cae_fluid` を import しない（status-32）。
`nsb/{data,assembly}.py` は `xkep_cae_fluid/brinkman_flow/{data,assembly}.py` の**スナップショット**
（コミット 1647839 時点、import 行のみ `from nsb.` に書き換え）で、2026-09-05 に本体側と**切り離した**。
本体側は面ベース FVM 共通低レイヤー（[`xkep_cae_fluid.fvm`](../docs/design/fvm-layer.md)）へ移行して非構造格子対応を進め、
nsb 側は構造格子の旧離散化をそのまま保つ。以後は同期しない（同期スクリプト `scripts/sync_nsb_from_xkep.py` は削除済み）。

```bash
cp -r nsb /path/to/elsewhere/        # そのまま持ち出せる（pip install numpy scipy pypardiso）
pytest tests/test_nsb_standalone.py  # xkep_cae_fluid を import せずに読み込めることの検査
```

注意: `nsb.data.BoundaryKind` と `xkep_cae_fluid.brinkman_flow.BoundaryKind` は別クラスなので、
nsb の入力を Process ソルバー（`BrinkmanFlowFVMProcess`）へ渡すときは名前で詰め替える
（`tests/test_nsb.py::to_xkep_flow_input`）。

## 線形ソルバー: PARDISO 必須 + 前処理 LU の遅延更新（status-32）

疎 LU は **pypardiso（Intel MKL PARDISO）前提**で、scipy `splu` へのフォールバックは無い
（`nsb/linalg.py::PardisoLU`。`pip install pypardiso`、libmkl_rt が見つからなければ `PYPARDISO_MKL_RT=/path/to/libmkl_rt.so`）。
実測（flat、4 コア、`experiments/nsb/logs/bench-linear-solver-flat-r124.log`）では分解が splu の 4〜5 倍速い。

- **スレッド分割**: 分解は全スレッド、三角解（GMRES 前処理）は 1 スレッド（`PardisoLU(factor_threads, solve_threads)`）。
  三角解は数十回/反復呼ばれる小さな処理で、スレッド同期の方が高い（4 スレッド 15 ms vs 1 スレッド 5.7 ms、72×48）
- **`KMP_BLOCKTIME=0`** を import 時に既定設定する。MKL スレッドの spin 待ち（既定 200 ms）が GMRES 内の numpy 処理と
  CPU を奪い合い、三角解が 57 ms まで劣化した（実測）
- **遅延更新** `NSBSettings.precond_lag`（既定 4）: 1 回の LU を最大 4 Newton 反復で使い回す。再分解の条件は
  age ≥ lag / 直前 GMRES 反復数 > `precond_refresh_gmres`（30）/ 分解時から CFL が `precond_cfl_ratio`（2）倍以上変化 /
  棄却後。GMRES が収束しなければ即再分解して解き直す。結果は `NSBResult.n_factorizations`、`n_gmres_total` に記録
- 効果は格子・コア数依存（status-32 の表）。SER で CFL が毎反復 2 倍伸びる局面では擬似時間対角が前処理と食い違い
  GMRES 反復が増えるので、`precond_cfl_ratio` で抑えている。`precond_lag=1` で従来の毎反復分解に戻る

| ファイル | 役割 |
|---|---|
| `linalg.py` | `PardisoLU`（分解と三角解を分離、スレッド分割、MKL パス探索）、`pardiso_solve` |
| `data.py` | （スナップショット）`BoundaryKind` / `BoundaryPatch` / `BrinkmanFlowInput`、マスク補助 `west_span` 等、`disk_mask` / `smooth_disk` |
| `assembly.py` | （スナップショット）`BrinkmanDiscretization`: 残差、1 次風上ヤコビアン、Rhie–Chow、境界条件、領域内マニホールド |
| `core.py` | 型宣言: `BC`（座標マスクの境界パッチ列。`BC.velocity_inlet / mass_flow_inlet / pressure_outlet`）, `NSBSettings`, `NSBInput`, `NSBResult` |
| `solver.py` | メイン: `solve_steady`, `compute_dtau`, `solve_linear`, `LaggedPreconditioner`（前処理 LU の遅延更新） |
| `utils.py` | ポスト処理、面値⇄セル値変換、要約、npz 保存 |
| `geo.py` | uturn / flat の厚さ場（inlet/outlet 位置に追従）、BC プリセット（速度 or 質量流量）、`run_uturn`, `run_flat`, `make_case` |
| `../main.py` | パラメータスタディ（構成 × モデル × 細分化 × 流速） |
| `adjoint.py` | 設計感度: 彩色 FD ヤコビアン `colored_fd_jacobian`、陰関数定理の VJP `ImplicitSolve`（forward / jacobian / vjp / gradient、転置系は PARDISO）、`Objective` |
| `theory.md` | 数理ノート: 支配方程式〜離散化〜Newton/擬似時間〜発散機構〜随伴感度を総和規約で記述 |

## `NSBSettings` の「踏んではいけない線」スイッチ

| 設定 | 既定（手元構成の推定） | 修正構成 | 影響（status-30 の実験結果） |
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
from nsb import north_span, west_span

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
from nsb import disk_mask
bc = BC(patches=(
    BC.interior_source(disk_mask(0.15, 0.2, 0.05), 0.1),                 # 注入 0.1 kg/s
    BC.interior_pressure_sink(disk_mask(0.55, 0.2, 0.05), 1e-4, p=0.0),  # 吸出 q = C (p - 0)
))
```

```python
# 位置・径を連続設計変数に: 滑らかな窓 smooth_disk(cx, cy, r, eps) を weight に渡し、随伴で dθ を得る
from nsb import ImplicitSolve, source_mean_pressure_objective
from nsb import smooth_disk

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
