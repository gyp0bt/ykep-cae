# status-39: nsb 線形ソルバーの高速化 第 2 段 — FGMRES・SA 階層の再利用・残差の numba 化

[<- README](../../README.md) | [<- docs](../README.md) | [<- ステータス一覧](status-index.md) | [<- roadmap](../roadmap.md)

- 日付: 2026-09-07
- ブランチ: `claude/nsb-fgmres-hierarchy-numba`
- 前: [status-38](status-38.md)（SIMPLE 型ブロック前処理の導入、4 コアで PARDISO 比 2.66×）
- 環境: gyp さんの手元機（Core Ultra 7 265K、20 コア、30 GB）。numpy 2.4.5 / scipy 1.17.1 / pypardiso 0.4.7 / pyamg 5.3.0 / numba 0.67.0

---

## 1. 何を求められたか

リモート（PR #34、status-38）を master に取り込んだうえで、NSB（[`nsb/`](../../nsb/README.md)）の高速化を続ける。
status-38 は 4 コアのコンテナでの実測だったので、まず 20 コア機で同じベンチ（`experiments/nsb/bench_precond.py`、flat 288×192）を取り直した。

| 構成（status-38 のコード） | 20 コア | 4 コア（status-38） |
|---|---|---|
| jfnk（PARDISO、lag=4） | 91.5 s（36 Newton、三角解 66 ms × 1016 回 = 73%） | 229.6 s |
| jfnk_simple lag=4 gmres_tol=1e-2 | 73.8 s（36 Newton、GMRES 1756 回） | 86.3 s |

20 コアでは PARDISO の分解が 4.6 s → 0.35 s に縮む一方、三角解（1 スレッド）と SIMPLE 側の各部品は縮まないので、
比が 2.66× → 1.24× に潰れた。status-38 の見込みどおり。ここから SIMPLE 側の 1 Newton 反復 2.05 s をどう削るか。

## 2. 全体像: 1 Newton 反復のコスト構造と、どこを削ったか

```
1 Newton 反復 = 前処理組立（lag で償却） + n_GMRES × ( 前処理適用 + matvec（残差評価） + Krylov の直交化 )
```

<svg viewBox="0 0 900 330" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">
  <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
  <text x="20" y="24" font-size="15" font-weight="bold">288×192、1 Newton 反復あたり [ms]（20 コア）</text>
  <text x="20" y="46" fill="#555">上: status-38 のコード（scipy gmres、毎回 SA 構築、numpy 残差）　下: 本 status（FGMRES、SA 再利用、numba 残差）</text>
  <!-- before -->
  <text x="20" y="90" font-weight="bold">前</text>
  <rect x="60" y="72" width="222" height="26" fill="#c96"/><text x="66" y="90" fill="#fff">組立 695（SA 667 + ILU 228, 29/36 回）</text>
  <rect x="282" y="72" width="205" height="26" fill="#69c"/><text x="288" y="90" fill="#fff">適用 12.7 × 49 = 620</text>
  <rect x="487" y="72" width="94" height="26" fill="#6a6"/><text x="492" y="90" fill="#fff">残差 5.8×49=285</text>
  <rect x="581" y="72" width="60" height="26" fill="#999"/><text x="586" y="90" fill="#fff">直交化 140</text>
  <rect x="641" y="72" width="60" height="26" fill="#ccc"/><text x="646" y="90">その他</text>
  <text x="720" y="90" font-weight="bold">計 2050</text>
  <!-- after -->
  <text x="20" y="150" font-weight="bold">後</text>
  <rect x="60" y="132" width="78" height="26" fill="#c96"/><text x="64" y="150" fill="#fff">組立 230</text>
  <rect x="138" y="132" width="196" height="26" fill="#69c"/><text x="143" y="150" fill="#fff">適用 10.8 × 55 = 594</text>
  <rect x="334" y="132" width="33" height="26" fill="#6a6"/><text x="337" y="150" fill="#fff" font-size="11">残差 100</text>
  <rect x="367" y="132" width="25" height="26" fill="#999"/>
  <rect x="392" y="132" width="40" height="26" fill="#ccc"/>
  <text x="440" y="150" font-weight="bold">計 ≈ 1200（GMRES 42〜55 反復）</text>
  <!-- explanations -->
  <line x1="170" y1="100" x2="100" y2="130" stroke="#555" marker-end="url(#a)"/>
  <text x="20" y="200" fill="#c96" font-weight="bold">組立 695 → 230（1 回 863 → 265 ms）</text>
  <text x="20" y="218">SA の集約 P, R を最初の 1 回だけ作り、以後は Galerkin 積 R Ŝ P で粗格子だけ組み直す（667 → 10 ms）。ILU 228 ms は残る</text>
  <text x="20" y="246" fill="#69c" font-weight="bold">適用 12.7 → 10.8 ms（反復数は経路で 42〜55）</text>
  <text x="20" y="264">V サイクルを MultilevelSolver.solve を通さず直接辿る（残差ノルム評価の spmv 1 回分、6.8 → 4.6 ms）。反復数は停止則（§3.1）と経路で変わる</text>
  <text x="20" y="292" fill="#6a6" font-weight="bold">残差 5.8 → 1.8 ms、直交化 2.8 → 1.4 ms（1 反復あたり）</text>
  <text x="20" y="310">残差評価を numba prange 7 パス（カーネル 0.5 ms）、Gram–Schmidt を基底行列との gemv（BLAS）に。GMRES 1 反復 26.6 → 16.7 ms</text>
</svg>

各部品の単体計測（288×192、`scratchpad/profile_parts.py`）:

| 部品 | 前 | 後 | 手段 |
|---|---|---|---|
| SA 階層構築 | 100〜1900 ms（乱数依存で振れる） | 10 ms（Galerkin 再構築） | 集約の再利用（§3.2） |
| ILU（spilu 1e-3/3.0） | 237 ms | 237 ms | 変更なし |
| V サイクル 1 回 | 6.8 ms | 4.6 ms | 直接呼び出し（§3.3） |
| ILU 三角解 | 3.9 ms | 3.9 ms | 変更なし |
| 残差評価（JFNK の matvec） | 5.8 ms（compute_state 4.6 + residual 1.2） | 0.3〜0.5 ms | numba（§3.4） |
| Krylov 直交化 1 反復 | 2.8 ms（scipy） | 1.4 ms | CGS2 を gemv で（§3.1） |
| GMRES 反復数（同じ J1、rtol 1e-2） | 37（scipy） | 16（FGMRES） | 停止則（§3.1） |

## 3. 機構ごとの論点

### 3.1 GMRES の停止則: scipy は指定より深く解き、JFNK の matvec は線形でない

**現象**: 同じ J1 + SIMPLE 前処理に `rtol=1e-2` を指定すると、scipy `gmres` は 37 反復（真の残差 2e-3）、自作 FGMRES は 16 反復（9.8e-3）。

**原因**: scipy は再出発ごとに真の残差を確認し、届いていなければ内側の許容を `ptol_max_factor` で締めていく。
inexact Newton では「指定した許容で止まる」ことが要で、深く解いても Newton の経路はほとんど変わらない（線形系の解の質は FD ヤコビアンの近似度で頭打ち）。

**対策**: 右前処理 flexible GMRES を自作（[`nsb/krylov.py::fgmres`](../../nsb/krylov.py)）。Gram–Schmidt は基底行列 V（restart × n）との gemv 2 回（CGS2）で BLAS に任せ、Givens 回転で最小二乗残差を更新する。前処理ベクトル Z = M⁻¹V を保持する flexible 版なので、前処理が反復ごとに変わっても正しい（status-38 残件の FGMRES）。

**副作用として見つかったこと**: JFNK の有限差分 matvec は線形写像から **1e-4〜1e-3** ずれる（`scratchpad/exp4.py`: `‖A(v+w) − Av − Aw‖/‖Av‖`）。風上切替 `fx ≥ 0`、`max(fx, 0)` の a_P、Venkatakrishnan の `min` など残差の折れ点を摂動が跨ぐためで、刻み幅を 0.1〜100 倍にしても中心差分にしても 1e-4 を割らない（折れ点の密度で決まる）。この非線形性で Arnoldi 関係が崩れ、Givens 推定 8.5e-4 に対して真の残差 5.3e-3 という食い違いが高 CFL で出る。真の残差で再出発しても FD 雑音の床を割れないので、`maxiter` 回の再出発（例: 40 + 6 + 6 + 5 + 5 = 62 反復）を空回りして「gmres not converged」→ 前処理を組み直して解き直す二重コストになっていた。

**対策**: JFNK では `check_true_residual=False`（Givens 推定を信じて止める。matvec 1 回分も節約）。`dc_simple` と Stokes 初期場（J1 の厳密 matvec）は従来どおり真の残差で確認する。

| FD の取り方（288×192、cap#21 CFL 39） | 線形性誤差 | GMRES 反復（rtol 1e-3） | Givens 推定 | 真の残差 |
|---|---|---|---|---|
| 前進差分 ε × 0.1 | 3.0e-5 | 39（not converged） | 8.4e-4 | 5.1e-3 |
| **前進差分 ε × 1（既定）** | 6.8e-4 | 32 | 8.4e-4 | 9.1e-4 |
| 前進差分 ε × 10 | 1.8e-3 | 30 | 9.3e-4 | 9.7e-4 |
| 中心差分 ε × 1 | 3.0e-4 | 32 | 8.3e-4 | 8.9e-4 |
| J1 の厳密 matvec | 0 | 44 | 9.0e-4 | 9.0e-4 |

（ε = √ε_mach·√(1+‖x‖)/‖v‖。J1 の matvec の方が反復が多いのは、前処理が J1 用なのに真のヤコビアン（SOU 残差）とは別物だから。刻みを小さくすると丸め雑音が勝って悪化する）

### 3.2 SA 階層の再利用: 集約は固定、粗格子行列だけ Galerkin 積で組み直す

**現象**: `pyamg.smoothed_aggregation_solver` の構築が 100〜1900 ms と走行ごとに振れる（スペクトル半径推定の Arnoldi が乱数初期ベクトルで反復数を変える）。ILU の 237 ms と合わせて組立が 1 Newton 反復の 34%。

**機構**: SA の強度・集約・平滑化プロロンゲータ P は Ŝ の**形**（compact 5 点 + RC の遠方項）で決まり、擬似時間対角 ρV/Δτ が CFL で変わっても集約が変わる理由はない。R Ŝ P の Galerkin 積は疎行列積 3 レベル分で 10 ms。

**実測**（`scratchpad/exp2.py`、cap#2 CFL 1.35 で作った階層を CFL 27 まで使い回し）:

| Newton 反復 (CFL) | 毎回構築 | 再利用 |
|---|---|---|
| #2 (1.35) | 6 | 6 |
| #5 (3.5) | 15 | 16 |
| #11 (8.8) | 46 | 45 |
| #14 (16) | **66** | **44** |
| #17 (27) | 39 | 36 |
| #23 (19) | 37 | 30 |

再利用の方が反復数が同等以下。毎回構築だと集約がわずかに変わるたびに前処理の質が揺れる。`SimpleBlockPreconditioner(reuse_hierarchy=True)` が既定で、`free()` で階層を捨てる（`solve_steady` の終了時）。

### 3.3 V サイクルの直接呼び出し

`MultilevelSolver.solve(maxiter=1)` は各サイクル後に真の残差ノルム（spmv 1 回 + norm）を評価して `tol` と比べる。前処理としては固定 1 サイクルなので不要。階層の `presmoother` / `R` / `P` / `postsmoother` / `coarse_solver` を直接辿る `_vcycle` に置き換え（6.8 → 4.6 ms、`test_vcycle_matches_pyamg_solve` で同一写像を確認）。

### 3.4 残差評価の numba 化

`compute_state` + `residual_from_state` は numpy の配列演算で 288×192 が 5.8 ms。配列演算は 1 演算ごとに 55k 要素の中間配列を作るので、演算数（≈ 300）× メモリ往復で決まり、コア数では縮まない。
[`nsb/fastres.py::residual_kernel`](../../nsb/fastres.py) は面ごと・セルごとの `prange` ループ 7 パス（線形面値 → a_P と d_cell → RC 流束 → Venkatakrishnan の外挿量 → 対流面値 → 残差）で中間配列を最小限にし、0.3〜0.5 ms。numpy 経路と 1e-17 で一致（`tests/test_nsb_fastres.py`: flat / uturn / マニホールド × SOU / FOU × 擬似時間対角 × 対流の有無）。
`NSBSettings.fast_residual=True` が既定で、numba が無ければ自動で numpy 経路。JFNK の matvec と収束判定に使い、ヤコビアン組立には従来の `compute_state` を使う。

### 3.5 決定性: 前処理の微差で Newton の経路が変わる

**現象**: 同じ問題・同じ設定で Newton 反復数が 22 / 26 / 34 / 37、`lag=1` では 35 / 115 と走行ごとに変わる。

**原因**: pyamg のスペクトル半径推定が numpy のグローバル乱数を使うので、前処理がわずかに違う → GMRES の解（rtol 1e-3 の範囲内で違う）→ Newton 更新が違う → SER で CFL が毎反復 2 倍伸びる局面では 1 歩の違いが CFL 経路の違いに増幅される。線形ソルバーの問題ではなく、非線形反復（SER + 擬似時間）が経路に敏感なこと自体の現れ。

**対策**: SA 構築の間だけ乱数を固定シードにして（`_build_hierarchy`）、同じ行列から同じ階層が出るようにした。2 回の走行が反復数・GMRES 総数・残差履歴まで一致することを確認。**経路の敏感さそのものは残っている**（§5）。

### 3.6 試して採用しなかったもの

- **`gmres_tol=1e-2`**（status-38 の最良構成）: scipy が実質 2e-3 まで解いていたから効いていた。正直に 1e-2 で止めると Newton が 34 → 43 回に増えて総時間は同等以下。既定 1e-3 のままにする
- **ILU 1e-2 / 1.5**: FGMRES に替えた走行で 288×192 が**発散**（85 反復、`du=7.5e+01`）。status-38 の零ピボットと同根（前処理として不安定）。ベンチ構成から外した
- **前処理の変種**（`scratchpad/exp5.py`、cap#12 CFL 10、J1 に対する反復数 × 1 適用）: Schur 2 サイクル 447 ms、ILU 1e-4/5 366 ms、GS 前進/後退 315 ms、対称 GS 2 回 394 ms、SIMPLEC 対角（行和）547 ms、運動量 ILU の Richardson 2 回 523 ms に対し**基準 297 ms が最良**。SIMPLE 型の質はここが限界で、高 CFL で反復数 45〜56（PARDISO LU は 18）は前処理の構造（Ŝ の近似 + 1 次風上 J1）由来
- FD の刻み幅・中心差分（§3.1 の表）: 線形性誤差は折れ点の密度で決まり改善しない

## 4. 実測（`experiments/nsb/bench_precond.py`、flat、U=1、推奨構成、20 コア、[ログ](../../experiments/nsb/logs/bench-precond-flat-r124-status39.log)）

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

内訳（288×192）:

| 構成 | 前処理組立 [s] (回数, 1 回) | 前処理適用 [s] (回数, 1 回) | 残差評価 [s] (回数, 1 回) | GMRES 全体 [s] |
|---|---|---|---|---|
| jfnk (pardiso, lag=4) | 7.0 (20, 347.4 ms) | 43.5 (669, 65.0 ms) | 1.6 (843, 1.9 ms) | 46.6 |
| jfnk_simple lag=1 | 12.0 (46, 260.4 ms) | 27.7 (2555, 10.8 ms) | 4.9 (2780, 1.8 ms) | 39.5 |
| jfnk_simple lag=4 | 8.2 (31, 266.0 ms) | 21.7 (1985, 10.9 ms) | 3.8 (2165, 1.7 ms) | 33.1 |
| jfnk_simple lag=4 gmres_tol=1e-2 | 8.2 (31, 266.1 ms) | 15.9 (1468, 10.8 ms) | 3.0 (1625, 1.9 ms) | 25.3 |
| jfnk_simple lag=4 fast_residual=False | 4.7 (18, 262.5 ms) | 10.2 (933, 10.9 ms) | 6.1 (1058, 5.8 ms) | 19.5 |
| jfnk_simple lag=4 schur_cycles=2 | 5.2 (20, 262.0 ms) | 16.3 (1081, 15.1 ms) | 2.2 (1203, 1.8 ms) | 22.4 |
| dc_simple lag=4 | 13.0 (49, 265.3 ms) | 25.6 (2363, 10.8 ms) | 0.3 (175, 1.9 ms) | 33.8 |

単位コスト（288×192、経路に依らない指標。status-38 のコードは本セッション冒頭の同機実測 `scratchpad/bench-base-r4.log`）:

| 指標 | status-38 のコード | 本 status | 比 |
|---|---|---|---|
| GMRES 1 反復（前処理適用 + 残差評価 + 直交化） | 26.6 ms（46.7 s / 1756） | **16.7 ms**（33.1 s / 1985） | 0.63 |
| 前処理組立 1 回（SA + ILU + ブロック分割） | 863 ms | **265 ms** | 0.31 |
| 残差評価 1 回 | 5.8 ms | 1.8 ms（カーネル 0.5 ms + 擬似時間項・呼び出し） | 0.31 |
| 前処理適用 1 回 | 12.7 ms | 10.8 ms | 0.85 |
| PARDISO 三角解 1 回（参考） | 65.8 ms | 65.0 ms | 1.00 |

総時間は Newton 反復数（同じ設定で 22〜36 回、§3.5）に比例して振れる。`jfnk_simple lag=4` は本走行で 36 回（43.1 s）、
`fast_residual=False` は 22 回（25.6 s）だが、これは numba 化の損得ではなく経路差（1 GMRES 反復あたりでは 20.9 ms → 16.7 ms）。

- 比較の基準となる **status-38 のコードでの同機実測**は PARDISO 91.5 s / SIMPLE 73.8 s（§1）。PARDISO 側も FGMRES で GMRES 反復が 886 → 668 に減って 55 s になった
- 288×192 の `jfnk_simple`（lag=4）は **26〜43 s**（PARDISO 55 s 比 1.3〜2.2×）。1 Newton 反復 2.05 s → 1.2 s、残る内訳は前処理適用（ILU 三角解 3.9 + V サイクル 4.6 + spmv）× 42〜55 反復と ILU 組立 260 ms
- `gmres_tol=1e-2` は Newton 40 回・GMRES 37 回/Newton で 35.3 s。正直に止めると 1 Newton は軽いが Newton が増える
- 72×48 / 144×96 では Newton 経路が PARDISO と一致し（解の差 1e-9〜1e-7）、1.6〜3×

## 5. 分かったこと・限界

- 20 コアでの現在地: 288×192 が **status-38 のコード 73.8 s → 26〜43 s**（Newton 経路で変動）、単位コストは GMRES 1 反復 0.63 倍・組立 0.31 倍。Fluent 級（600 反復 1 分）にはまだ 1 桁足りない
- **次の律速は前処理適用 × GMRES 反復数**（1 Newton の 60%）。SIMPLE 型の変種では縮まなかった（§3.6）ので、(a) 適用の C 化（ILU 三角解と V サイクルを numba で。合わせて 10 → 5 ms 程度が上限）、(b) 反復数を減らすには前処理の構造を変える（J1 でなく SOU の線形化を前処理に使う、または LSC / augmented Lagrangian）
- **Newton 経路の敏感さ**が総時間の分散を支配する（22〜37 反復）。SER が毎反復 CFL を 2 倍にする規則のため、1 歩の違いが経路に増幅される。CFL 成長率の上限を下げる／残差比で連続的に制御するなど非線形側の設計が要る。これは手元構成の制御則そのものなので、gyp さんの判断で変える
- JFNK の FD matvec の非線形性（1e-4〜1e-3）は残差関数の折れ点由来で、これが inexact Newton の実効的な許容の床になる。`gmres_tol` を 1e-4 に締めても意味がない

## 6. 残件

- [ ] 前処理適用の numba 化（ILU(0) または spilu の L/U を CSR で取り出して三角解、V サイクルの GS を prange の色分け）
- [ ] SER の CFL 成長則の見直し（経路の敏感さ）— 手元構成と相談
- [ ] `dc_simple` で `gmres_maxiter` を絞った固定サイクル外部反復（Fluent 型）— status-38 からの持ち越し
- [ ] Ŝ の 5 点化 + SA 階層（status-38 残件。階層再利用で組立が消えたので優先度は下がった）
- master 取り込み時点で `python contracts/validate_process_contracts.py` が **C3: BenchmarkRunnerProcess にテストが紐付けられていない** を 1 件報告する（`tests/test_benchmark_runner.py` に `@binds_to` はある。検出スクリプト側の取りこぼし。本ブランチの変更とは無関係、未修正）
- status-37 からの持ち越し: `.inp` の `*RTD` キーワード、空気実物性の SIMPLE 連成（CLAUDE.md の最優先事項）

## テスト実行

```
python -m pytest tests/test_nsb_krylov.py tests/test_nsb_fastres.py tests/test_nsb_precond.py tests/test_nsb_linalg.py tests/test_nsb_standalone.py tests/test_nsb.py tests/test_nsb_adjoint.py -q -n 4
→ 75 passed（67.6 s。krylov 9 + fastres 25 + precond 10 を含む）
python contracts/validate_process_contracts.py → 契約違反 1 件（C3 BenchmarkRunnerProcess、master 由来。上記残件）
ruff check nsb/ tests/ experiments/nsb/ && ruff format --check nsb/ tests/ experiments/nsb/ → All checks passed
```

## ファイル

- 追加: `nsb/krylov.py`、`nsb/fastres.py`、`tests/test_nsb_krylov.py`、`tests/test_nsb_fastres.py`、
  `experiments/nsb/logs/bench-precond-flat-r124-status39.log`、`docs/status/status-39.md`
- 変更: `nsb/{precond,solver,assembly,core}.py`、`nsb/README.md`、`experiments/nsb/bench_precond.py`、`tests/test_nsb_precond.py`、
  `pyproject.toml`（`nsb` / `dev` extra に numba）、`README.md`、`docs/roadmap.md`、`docs/status/status-index.md`
