# nsb: 手元構成ミラーの Brinkman-NS 実験パッケージ

[<- README](../README.md) | [数理ノート（総和規約）](theory.md) | [設計文書（共有離散化）](../docs/design/brinkman-flow-fvm.md) | [status-30](../docs/status/status-30.md) | [status-32（切り離し・高速化見積）](../docs/status/status-32.md) | [status-38（SIMPLE 型前処理）](../docs/status/status-38.md)

手元の 2D FVM Brinkman 補正 Navier-Stokes コードと**同じファイル構成・同じ制御則**で比較するための薄いレイヤ。
離散化（残差、1 次風上ヤコビアン、Rhie–Chow、境界条件）は `nsb/assembly.py` の `BrinkmanDiscretization`、
境界条件・入力型は `nsb/data.py` に持ち、Newton + 擬似時間の制御則だけを `solver.solve_steady` に関数として書き下している。

## 単体で持ち出せる（xkep_cae_fluid 非依存、スナップショット）

`nsb/` ディレクトリは **numpy / scipy / pypardiso / pyamg（+ 任意で numba）だけ**で動き、`xkep_cae_fluid` を import しない（status-32）。
`nsb/{data,assembly}.py` は `xkep_cae_fluid/brinkman_flow/{data,assembly}.py` の**スナップショット**
（コミット 1647839 時点、import 行のみ `from nsb.` に書き換え）で、2026-09-05 に本体側と**切り離した**。
本体側は面ベース FVM 共通低レイヤー（[`xkep_cae_fluid.fvm`](../docs/design/fvm-layer.md)）へ移行して非構造格子対応を進め、
nsb 側は構造格子の旧離散化をそのまま保つ。以後は同期しない（同期スクリプト `scripts/sync_nsb_from_xkep.py` は削除済み）。

```bash
cp -r nsb /path/to/elsewhere/        # そのまま持ち出せる（pip install numpy scipy pypardiso pyamg numba）
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

## 線形ソルバー: SIMPLE 型ブロック前処理（ILU + Schur 補元 AMG、status-38）

`NSBSettings.linear_solver="jfnk_simple"`（既定）で、疎 LU の代わりに **SIMPLE 型ブロック前処理**
（`nsb/precond.py::SimpleBlockPreconditioner`、pyamg 必須）を GMRES に使う。3N×3N の 1 次風上ヤコビアン
J = [[A, B], [C, D]]（A: 速度、B: 圧力勾配、C: 発散、D: Rhie–Chow 圧力項）に対し

1. A u* = r_u を **ILU**（scipy `spilu`）で近似解
2. Ŝ δp = r_p − C u*、Ŝ = D − C diag(A)⁻¹ B を **smoothed aggregation AMG**（非対称、V サイクル 1 回）で近似解
3. u = u* − diag(A)⁻¹ B δp

を 1 回の前処理適用とする。組立・適用とも O(N) なので、LU の fill-in（288×192 で元の nnz の 40 倍）と
三角解（155 ms、1 スレッド）が支配的だった大格子で効く。`jfnk_simple` は有限差分の J v（JFNK）、`dc_simple` は
J1 v（defect correction: 残差評価を伴わないので 1 反復が軽いが Newton は線形収束で反復数が 2 倍）。

- **部品の選定**（status-38 の切り分け、288×192、J1 に対する GMRES 反復数）: 運動量は ILU が最良で、
  Gauss–Seidel と運動量 AMG は高 CFL（対流優勢）で発散。Schur 補元は compact 5 点 Poisson が支配的だが RC と Newton 項の
  遠方項（±2〜3 セル、符号混在）で Ruge–Stüben の収束率が 0.76/サイクルに落ち GMRES 199 反復、遠方項の lumping は
  逆効果、**smoothed aggregation では 43 反復**（運動量・Schur とも厳密解の SIMPLE で 64 反復）
- 設定: `simple_schur_cycles`、`simple_ilu_drop_tol` / `simple_ilu_fill_factor`（運動量 Jacobi と `dc_simple` は status-40 で廃止）。
  `precond_lag` は共通（組立が軽いので lag の利得は小さい）
- Stokes 参照場も同じ前処理付き GMRES（rtol 1e-10）で解く。収束しなければ PARDISO 1 回に落ちる
  （ログの `stokes ref (gmres=N)` で確認できる）
- 実測は下の表（`experiments/nsb/bench_precond.py`、4 コア、ログ `experiments/nsb/logs/bench-precond-flat-r124.log`）

| 格子 | 構成 | 収束 | Newton | 前処理組立 | GMRES 総反復 | 全体 | 1 Newton | PARDISO 比 | 解の差 max\|Δu\|/max\|u\| |
|---|---|---|---|---|---|---|---|---|---|
| 72x48 | jfnk (pardiso, lag=4) | True | 13 | 10 | 111 | 2.9 s | 0.22 s | 1.00× | 0.0e+00 |
| 72x48 | jfnk_simple lag=1 | True | 13 | 14 | 308 | 2.6 s | 0.20 s | 1.09× | 8.7e-09 |
| 72x48 | jfnk_simple lag=4 | True | 13 | 10 | 301 | 2.1 s | 0.16 s | 1.40× | 5.0e-09 |
| 72x48 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 13 | 10 | 202 | 1.7 s | 0.13 s | 1.73× | 1.9e-08 |
| 72x48 | dc_simple lag=4 | True | 29 | 14 | 467 | 2.6 s | 0.09 s | 1.12× | 2.0e-06 |
| 72x48 | jfnk_simple lag=4 ilu=1e-2/1.5 | True | 13 | 10 | 307 | 1.8 s | 0.14 s | 1.57× | 4.5e-09 |
| 144x96 | jfnk (pardiso, lag=4) | True | 18 | 11 | 280 | 18.6 s | 1.03 s | 1.00× | 0.0e+00 |
| 144x96 | jfnk_simple lag=1 | True | 18 | 19 | 645 | 16.0 s | 0.89 s | 1.16× | 1.9e-08 |
| 144x96 | jfnk_simple lag=4 | True | 18 | 13 | 669 | 13.9 s | 0.77 s | 1.33× | 1.9e-08 |
| 144x96 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 19 | 13 | 460 | 11.3 s | 0.59 s | 1.65× | 1.2e-06 |
| 144x96 | dc_simple lag=4 | True | 36 | 17 | 919 | 17.5 s | 0.49 s | 1.06× | 1.7e-05 |
| 144x96 | jfnk_simple lag=4 ilu=1e-2/1.5 | True | 18 | 13 | 627 | 13.1 s | 0.73 s | 1.41× | 2.0e-08 |
| 288x192 | jfnk (pardiso, lag=4) | True | 36 | 22 | 850 | 229.6 s | 6.38 s | 1.00× | 0.0e+00 |
| 288x192 | jfnk_simple lag=1 | True | 24 | 25 | 1641 | 149.0 s | 6.21 s | 1.54× | 5.7e-05 |
| 288x192 | jfnk_simple lag=4 | True | 33 | 29 | 2573 | 212.8 s | 6.45 s | 1.08× | 5.9e-05 |
| 288x192 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 26 | 18 | 872 | 86.3 s | 3.32 s | 2.66× | 5.7e-05 |
| 288x192 | dc_simple lag=4 | True | 60 | 52 | 2560 | 224.0 s | 3.73 s | 1.02× | 5.4e-05 |
| 288x192 | jfnk_simple lag=4 ilu=1e-2/1.5 | True | 27 | 22 | 1443 | 120.1 s | 4.45 s | 1.91× | 5.7e-05 |

- 288×192 では `jfnk_simple lag=4 gmres_tol=1e-2` が **229.6 s → 86.3 s（2.66×）**、1 Newton 反復 6.4 s → 3.3 s。解の差 5.7e-5 は Newton の経路（反復数 36 vs 26）が変わったことによる収束判定 1e-6 相当の差で、72×48 / 144×96 では経路が同じで 1e-8
- 4 コアでの数字。18 コア機では PARDISO の分解が縮む一方 SIMPLE 側（ILU 三角解・AMG V サイクル・残差評価）は 1 スレッドのままなので比は縮む見込み。実機で `python experiments/nsb/bench_precond.py 4 2>&1 | tee ...` を回して確定する

## 線形ソルバー: FGMRES・SA 階層の再利用・残差の numba 化（status-39）

status-38 のコードを 20 コア機で取り直すと PARDISO 91.5 s / SIMPLE 73.8 s（288×192）で、PARDISO の分解が縮む分だけ比が潰れた。
1 Newton 反復 = 組立 + n_GMRES × (前処理適用 + 残差評価 + 直交化) の各項を削った（[status-39](../docs/status/status-39.md)）:

- **GMRES を自作の右前処理 FGMRES に**（`nsb/krylov.py::fgmres`、CGS2 を gemv で + Givens）。scipy `gmres` は再出発ごとに内側の許容を締めるので
  `rtol=1e-2` 指定でも 2e-3 まで 37 反復回っていた（FGMRES は 16 反復）。JFNK の FD matvec は残差の折れ点（風上切替・リミター）のため
  線形写像から 1e-4〜1e-3 ずれ、Givens 推定と真の残差が高 CFL で食い違う。真の残差で再出発しても雑音の床を割れないので、JFNK では
  `check_true_residual=False` で止める選択肢を `fgmres` に用意した（ソルバーでは真の残差確認を残す。Givens 推定だけだと PARDISO 側の Newton が収束しない走行があった）
- **SA 階層の再利用**: 集約 P, R は最初の 1 回だけ作り、以後は Ŝ を差し替えて
  Galerkin 積 R Ŝ P で粗格子だけ組み直す（667 → 10 ms）。CFL 1.4 → 27 で使い回しても GMRES 反復数は毎回構築と同等以下
- **V サイクルの直接呼び出し**（`MultilevelSolver.solve` の残差ノルム評価を省く、6.8 → 4.6 ms）
- **残差評価の numba 化**（`nsb/fastres.py::residual_kernel`、`prange` 7 パス、5.8 → 0.3〜0.5 ms、numpy 経路と 1e-17 一致）。
  numba が無ければ自動で numpy 経路
- **決定性**: pyamg のスペクトル半径推定がグローバル乱数を使い、前処理の微差で Newton 反復数が 22〜37 と振れていた。SA 構築の間だけ
  固定シードにして同じ行列から同じ階層が出るようにした（経路の敏感さ自体は SER の CFL 倍化則に由来し、残る）
- 採用しなかったもの: `gmres_tol=1e-2`（正直に止めると Newton が増える）、ILU 1e-2/1.5（FGMRES 化後に 288×192 で発散）、
  Schur 2 サイクル・ILU 1e-4/5・GS 前進/後退・SIMPLEC 対角・運動量 ILU の Richardson（いずれも基準より遅い）

実測（`experiments/nsb/bench_precond.py`、20 コア、ログ `experiments/nsb/logs/bench-precond-flat-r124-status39.log`）:

| 格子 | 構成 | 収束 | Newton | 前処理組立 | GMRES 総反復 | GMRES/Newton | 全体 | 1 Newton | PARDISO 比 | 解の差 max\|Δu\|/max\|u\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 72x48 | jfnk (pardiso, lag=4) | True | 13 | 10 | 97 | 7 | 1.2 s | 0.10 s | 1.00× | 0.0e+00 |
| 72x48 | jfnk_simple lag=1 | True | 13 | 14 | 213 | 16 | 0.5 s | 0.04 s | 2.55× | 2.7e-09 |
| 72x48 | jfnk_simple lag=4 | True | 13 | 10 | 210 | 16 | 0.4 s | 0.03 s | 3.05× | 2.7e-09 |
| 72x48 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 13 | 10 | 144 | 11 | 0.3 s | 0.03 s | 3.68× | 5.4e-08 |
| 72x48 | jfnk_simple lag=4 fast_residual=False | True | 13 | 10 | 210 | 16 | 0.5 s | 0.04 s | 2.45× | 2.7e-09 |
| 72x48 | jfnk_simple lag=4 schur_cycles=2 | True | 13 | 10 | 205 | 16 | 0.5 s | 0.04 s | 2.72× | 2.4e-09 |
| 72x48 | dc_simple lag=4 | True | 29 | 14 | 402 | 14 | 0.8 s | 0.03 s | 1.52× | 2.1e-06 |
| 144x96 | jfnk (pardiso, lag=4) | True | 19 | 12 | 221 | 12 | 5.7 s | 0.30 s | 1.00× | 0.0e+00 |
| 144x96 | jfnk_simple lag=1 | True | 19 | 20 | 464 | 24 | 3.9 s | 0.20 s | 1.46× | 4.5e-07 |
| 144x96 | jfnk_simple lag=4 | True | 19 | 13 | 468 | 25 | 3.5 s | 0.19 s | 1.60× | 2.3e-07 |
| 144x96 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 18 | 11 | 267 | 15 | 2.3 s | 0.13 s | 2.43× | 9.9e-07 |
| 144x96 | jfnk_simple lag=4 fast_residual=False | True | 19 | 13 | 469 | 25 | 4.5 s | 0.24 s | 1.25× | 2.2e-07 |
| 144x96 | jfnk_simple lag=4 schur_cycles=2 | True | 17 | 11 | 407 | 24 | 4.4 s | 0.26 s | 1.28× | 1.6e-06 |
| 144x96 | dc_simple lag=4 | True | 36 | 16 | 697 | 19 | 5.2 s | 0.15 s | 1.08× | 1.7e-05 |
| 288x192 | jfnk (pardiso, lag=4) | True | 37 | 20 | 668 | 18 | 55.0 s | 1.49 s | 1.00× | 0.0e+00 |
| 288x192 | jfnk_simple lag=1 | True | 45 | 46 | 2555 | 57 | 53.9 s | 1.20 s | 1.02× | 3.2e-05 |
| 288x192 | jfnk_simple lag=4 | True | 36 | 31 | 1985 | 55 | 43.1 s | 1.20 s | 1.28× | 3.1e-05 |
| 288x192 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 40 | 31 | 1468 | 37 | 35.3 s | 0.88 s | 1.56× | 3.2e-05 |
| 288x192 | jfnk_simple lag=4 fast_residual=False | True | 22 | 18 | 933 | 42 | 25.6 s | 1.16 s | 2.15× | 3.2e-05 |
| 288x192 | jfnk_simple lag=4 schur_cycles=2 | True | 25 | 20 | 1081 | 43 | 28.8 s | 1.15 s | 1.91× | 3.3e-05 |
| 288x192 | dc_simple lag=4 | True | 58 | 49 | 2363 | 41 | 49.9 s | 0.86 s | 1.10× | 4.0e-05 |

単位コスト（288×192）は GMRES 1 反復 26.6 → 16.7 ms、前処理組立 863 → 265 ms。総時間は Newton 反復数（同じ設定でも経路で 22〜36 回。決定化後は再現する）に比例して振れる。次の律速は前処理適用（ILU 三角解 3.9 + V サイクル 4.6 ms）× GMRES 反復数（高 CFL で 42〜56）。

| ファイル | 役割 |
|---|---|
| `linalg.py` | `PardisoLU`（分解と三角解を分離、スレッド分割、MKL パス探索）、`pardiso_solve` |
| `precond.py` | `SimpleBlockPreconditioner`（SIMPLE 型ブロック前処理: 運動量 ILU + Schur 補元 SA-AMG、`PardisoLU` 互換の factorize / solve / free。SA 集約の再利用・V サイクル直呼び） |
| `krylov.py` | `fgmres`（右前処理 flexible GMRES。CGS2 + Givens、指定 rtol で止まる。scipy `gmres` の置き換え） |
| `fastres.py` | `residual_kernel`（残差評価の numba `prange` カーネル。`BrinkmanDiscretization.residual_fast` から呼ぶ） |
| `data.py` | （スナップショット）`BoundaryKind` / `BoundaryPatch` / `BrinkmanFlowInput`、マスク補助 `west_span` 等、`disk_mask` / `smooth_disk` |
| `assembly.py` | （スナップショット）`BrinkmanDiscretization`: 残差、1 次風上ヤコビアン、Rhie–Chow、境界条件、領域内マニホールド |
| `core.py` | 型宣言: `BC`（座標マスクの境界パッチ列。`BC.velocity_inlet / mass_flow_inlet / pressure_outlet`）, `NSBSettings`, `NSBInput`, `NSBResult` |
| `solver.py` | メイン: `solve_steady`, `compute_dtau`, `solve_linear`, `LaggedPreconditioner`（前処理 LU / SIMPLE 型の遅延更新） |
| `utils.py` | ポスト処理、面値⇄セル値変換、要約、npz 保存 |
| `geo.py` | uturn / flat の厚さ場（inlet/outlet 位置に追従）、BC プリセット（速度 or 質量流量）、`run_uturn`, `run_flat`, `make_case` |
| `../main.py` | パラメータスタディ（構成 × モデル × 細分化 × 流速） |
| `adjoint.py` | 設計感度: 彩色 FD ヤコビアン `colored_fd_jacobian`、陰関数定理の VJP `ImplicitSolve`（forward / jacobian / vjp / gradient、転置系は PARDISO）、`Objective` |
| `theory.md` | 数理ノート: 支配方程式〜離散化〜Newton/擬似時間〜発散機構〜随伴感度を総和規約で記述 |

## `NSBSettings`（status-40 で一長一短の切替と数値パラメータだけに絞った）

実験で一方が常に劣ると分かった切替は落とした: 静止場発進（`init_field`。参照場・初期場とも Stokes 解、
初期場は `NSBInput.u0/v0/p0` で差し替え可）、LU 直接 / defect correction（`"lu"` / `"dc_simple"`）、
運動量 Jacobi（`simple_momentum`）、CFL backtracking（`reject_growth` / `max_rejects` / `cfl_min`）、
速度下限なし（`velocity_floor` [m/s] → `velocity_floor_ratio` 既定 0.1）、numpy 残差（`fast_residual`）、
SA 階層の毎回構築（`reuse_hierarchy`）。既定は `linear_solver="jfnk_simple"`、`alpha_u=1.0`、`precond_cfl_ratio=2.0`。
収束判定の基準 r_ref は常に「Stokes 場で評価した完全 NS の定常残差」で、初期場の良し悪しに依らない。
SER は古典形 `CFL = cfl_init·|R_ref|/|R|` で出発するので、粗格子解を `u0/v0/p0` に入れれば初期 CFL が自動で大きく出る。

| 設定 | 既定 | 一長一短 |
|---|---|---|
| `linear_solver` | "jfnk_simple" | SIMPLE 型前処理は大格子で速い（288×192 で PARDISO 比 1.3〜2.2×）が GMRES 反復 42〜56。"jfnk"（PARDISO LU）は反復 18 で頑健、三角解 1 スレッドで大格子に弱い |
| `local_dtau` | True | 大域 Δτ は同じ CFL で減衰が約 10 倍強く高 CFL に寛容、収束は遅い |
| `pseudo_time_in_residual` | True | dual-time 型（収束判定・SER が擬似時間項込み）か対角補強のみか。定常解は同じ |
| `velocity_floor_ratio` | 0.1 | 小さいほど Newton に近く速いが、静止・低速セルで Δτ→∞ となり停滞する（0 は不可） |
| `cfl_init` / `ser_growth` / `cfl_max` | 0.5 / 2 / 1e6 | 出発は `cfl_init·|R_ref|/|R_init|`。成長率を上げると速いが Newton 経路が敏感になる |
| `alpha_u` | 1.0 | 陰的緩和。速度下限ありなら 1.0 が最速、頑健側に振るなら 0.7 |
| `precond_lag` / `precond_cfl_ratio` / `precond_refresh_gmres` | 4 / 2 / 30 | 前処理の使い回し。組立回数と GMRES 反復数のトレードオフ |
| `simple_schur_cycles` / `simple_ilu_*` | 1 / 1e-3, 3.0 | 前処理 1 適用の重さと反復数のトレードオフ。ILU 1e-2/1.5 は零ピボット・発散 |
| `gmres_tol` / `gmres_restart` / `gmres_maxiter` | 1e-3 / 40 / 5 | inexact Newton の許容。1e-2 は 1 Newton が軽いが Newton が増える |
| `convection` / `venkat_k` | "sou" / 5 | 2 次風上 + リミター（精度）か 1 次風上（頑健）か |
| `sub_iters` / `rc_with_pseudo_time` | 1 / False | 1 擬似時間ステップの Newton 反復数、RC 係数に ρV/Δτ を含めるか |

## 境界条件（座標マスク + 質量流入）

```python
from nsb import BC, NSBSettings, make_case, solve_steady
from nsb import north_span, west_span

# 流量 0.1 kg/s を固定し、inlet を上壁 x∈(0.3, 0.4) に置く（outlet は左壁下部）
bc = BC(patches=(
    BC.mass_flow_inlet(north_span(0.3, 0.4, 0.4), 0.1),
    BC.pressure_outlet(west_span(0.05, 0.15)),
))
inp = make_case("flat", 1, bc=bc)
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
    return make_case("flat", 1, bc=bc)

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

inp, res = run_uturn(refine=1, u_in=2.0)
print(res.converged, res.rel_residual, res.rel_steady_residual)
```

- テスト: `tests/test_nsb.py`
- 結果: `experiments/nsb/results/*.yaml`、ログ: `experiments/nsb/logs/`
