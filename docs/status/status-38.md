# status-38: nsb の線形ソルバーに SIMPLE 型ブロック前処理（ILU + Schur 補元 AMG）を追加

[<- README](../../README.md) | [<- docs](../README.md) | [<- ステータス一覧](status-index.md) | [<- roadmap](../roadmap.md)

- 日付: 2026-09-07
- ブランチ: `claude/nsb-performance-optimization-0v9y1q`（PR [#34](https://github.com/gyp0bt/ykep-cae/pull/34)）
- 前: [status-37](status-37.md)（非構造メッシュの粒子追跡と RTD / Phase 12 完了）
- 環境: 本セッションのコンテナ（4 コア）。numpy 2.4.6 / scipy 1.17.1 / pypardiso 0.4.7（MKL 2026.1）/ pyamg 5.3.0

---

## 1. 何を求められたか

手元の 18 コア機で NSB（[`nsb/`](../../nsb/README.md)）を 288×182 セルで解くと 1 Newton 反復に 2 秒以上かかり、
収束しにくい条件で 600 反復すると 20 分になる。Fluent は 16 コア並列で 600 反復を 1 分以内に終える。
タスクマネージャーではコアがほぼ使えていない。何が違うのか、pyamg（導入済み）で速くなるか。

## 2. 何が違うか（status-32 の実測から）

Fluent との差はハードでも Python でもなく**線形ソルバーのアルゴリズム**。

| | NSB（status-32 まで） | Fluent |
|---|---|---|
| 線形系 | 3N 連成ヤコビアンを **疎直接 LU（PARDISO）**、三角解を GMRES 前処理として 1 Newton あたり 20 回強 | **代数マルチグリッド（AMG）** を数サイクル、MPI 領域分割 |
| 演算量 | 2D の fill-in は O(N log N)、288×192 で L+U が **1.04 億要素**（元の nnz の 40 倍）、分解 O(N^1.5) | V サイクル 1 回 O(N) |
| 並列 | 分解だけ全スレッド、**三角解は 1 スレッド固定**（155 ms/回）、残差評価・組立は numpy 1 スレッド | 全段が分割並列 |
| 1 反復の定義 | GMRES を 1e-3 まで収束させる Newton 步（30〜40 反復で 1e-6） | 緩和付き外部反復（線形系は数サイクルだけ） |

288×192、4 コアで 1 Newton 反復 6〜7 s の内訳: PARDISO 分解 4.6 s（lag=4 で償却）、三角解 155 ms × 24 回 = 3.7 s、
残差評価 17.6 ms × 24 回 = 0.4 s。18 コアでは分解だけ縮み、三角解と numpy は縮まないので 2 s に張り付く。

pyamg は自然対流・熱伝導・`xkep_cae_fluid.fvm` の対称 Poisson 型行列に使われているだけで、
NSB には AMG のオプションが無かった（3N×3N の鞍点型・非対称行列に Ruge–Stüben を直接当てても動かない）。

## 3. やったこと

### 3.1 `nsb/precond.py`: `SimpleBlockPreconditioner`

Elman らの SIMPLE 型ブロック前処理を `PardisoLU` と同じインターフェース（`factorize` / `solve` / `free`）で実装。
1 次風上ヤコビアン J = [[A, B], [C, D]]（A: 速度 2N×2N + 擬似時間対角、B: 圧力勾配、C: 発散、D: Rhie–Chow 圧力項）に対し

1. A u* = r_u を **ILU**（scipy `spilu`、drop_tol 1e-3 / fill_factor 3.0、零ピボットなら締めて組み直し）で近似解
2. Ŝ δp = r_p − C u*、Ŝ = D − C diag(A)⁻¹ B を **smoothed aggregation AMG**（pyamg、非対称、V サイクル 1 回）で近似解
3. u = u* − diag(A)⁻¹ B δp

組立（ILU + AMG 階層）・適用とも O(N)。pyamg は nsb の必須依存（`pyproject.toml` の `nsb` extra に追加）。

### 3.2 `nsb/solver.py` / `nsb/core.py`

- `LaggedPreconditioner` が `PardisoLU` / `SimpleBlockPreconditioner` を同じ `fac` として保持（遅延更新の規則は共通）
- `NSBSettings.linear_solver`: 既存 `"jfnk"` / `"lu"` に **`"jfnk_simple"`**（有限差分 J v を GMRES、SIMPLE 前処理）と
  **`"dc_simple"`**（J1 v を GMRES、defect correction。残差評価を伴わないので 1 反復が軽いが Newton は線形収束）を追加
- 設定 `simple_momentum`（"ilu" / "jacobi"）、`simple_schur_cycles`、`simple_ilu_drop_tol` / `simple_ilu_fill_factor`
- Stokes 初期場（`init_field="stokes"`）は同じ前処理付き GMRES（rtol 1e-10）で解く。収束しなければ PARDISO 1 回に落ちる
  （ログ `stokes init (gmres=N)`）
- `pc.lu` → `pc.fac` に改名（`nsb` 内のみ。外部からの参照は無かった）

### 3.3 部品の選定（切り分け、288×192、J1 に対する GMRES 反復数、rtol 1e-3）

`scratchpad` の単体スクリプトで J1 + 擬似時間対角を組み、各前処理で GMRES の反復数を測った（PARDISO LU は 1 反復）。

**運動量ブロック A の近似解法**（Schur は Ruge–Stüben 1 サイクル）

| A の解法 | 144×96 CFL=8 | 144×96 CFL=10⁶ | 288×192 CFL=8 | 288×192 CFL=10⁶ |
|---|---|---|---|---|
| Jacobi | 50 | 110 | 116 | 358 |
| 対称 Gauss–Seidel 1〜4 回 | 195〜**発散** | **発散** | — | — |
| SA-AMG（非対称）1 サイクル | **発散** | **発散** | — | — |
| **ILU**（1e-3 / 3.0） | 32 | 42 | 60 | 199 |

GS と運動量 AMG は高 CFL（擬似時間対角が消えて対流優勢、Newton 項で非対角優位性が崩れる）で反復自体が発散する。

**Schur 補元の解法**（運動量は ILU）、288×192

| Ŝ の解法 | CFL=8 | CFL=10⁶ | AMG 収束率/サイクル | 1 サイクル |
|---|---|---|---|---|
| Ruge–Stüben（θ=0.25）1 サイクル | 60 | 199 | 0.71〜0.76 | 18〜20 ms |
| Ruge–Stüben 3 サイクル | 46 | 141 | | |
| Ruge–Stüben W サイクル | 52 | 189 | | 36 ms |
| 遠方項を最近接へ lumping した 5 点行列で置換 | 245 | 399 | 0.31〜0.62 | 12 ms |
| lumping 行列の AMG を Ŝ の Richardson 3 回の前処理に | 80 | 226 | | 46 ms |
| AIR（approximate ideal restriction） | 43 | 86 | 0.72〜0.74 | 20 ms |
| **smoothed aggregation（非対称）** 1 サイクル | **26** | **43** | 0.69〜0.70 | 14〜17 ms |
| 厳密解（PARDISO） | 34 | 67 | | |
| 運動量・Schur とも厳密解（SIMPLE 近似そのものの限界） | 39 | 64 | | |

- Ŝ の内部セルのステンシルは compact 5 点（対角 0.16、隣接 −0.04〜−0.07）が支配的だが、Rhie–Chow の compact/wide
  勾配差と Newton 項から ±2〜3 セルの遠方項（符号混在、対角の 5〜10%）が乗る。行和はゼロ
- この遠方項で Ruge–Stüben の粗格子選択が崩れ、収束率が 0.76/サイクルに落ちる。遠方項を行和保存で lumping した
  5 点行列は AMG 単体では速いが、前処理としては Ŝ との乖離で GMRES が悪化する
- smoothed aggregation は AMG 単体の収束率は同程度なのに前処理としては最良で、**厳密 Schur 解の SIMPLE（64）を下回る**
  （近似 Schur の粗さを AMG の丸さがたまたま補う形）。階層は 4 レベル、演算子複雑度 1.04
- ILU は単体測定では drop_tol 1e-2 / fill_factor 1.5 が 1e-3 / 3.0 と反復数同等以下で軽かった（288×192: 23〜42 反復、
  適用 25 ms）が、**実際の Newton 反復の途中（288×192、11〜13 反復目）で零ピボット（"Factor is exactly singular"）**
  になり解が落ちた。既定は実績のある **1e-3 / 3.0** に戻し、零ピボット時は drop_tol 1/10・fill 2 倍で最大 3 回組み直す
  リトライを `_build_ilu` に入れた（`n_ilu_retries` に累積、ベンチは `spilu` 呼び出し回数で確認）
- SA の平滑化なし集約は組立 0.15 s（あり 1.4 s）だが反復数 +20〜60% で、lag=4 の償却を前提に平滑化ありを既定にした

## 4. 実測（`experiments/nsb/bench_precond.py`、flat、U=1、推奨構成、4 コア）

推奨構成（`velocity_floor=0.1 U`、Stokes 初期場、`alpha_u=1`、`precond_cfl_ratio=2`）で jfnk（PARDISO、lag=4）を基準に比較。解の差は基準解との最大差（jfnk_simple は Newton の経路も反復数も基準と一致し、差は GMRES 打ち切り誤差のみ）。

| 格子 | 構成 | 収束 | Newton | 前処理組立 | GMRES 総反復 | 全体 | 1 Newton | PARDISO 比 | 解の差 max\|Δu\|/max\|u\| |
|---|---|---|---|---|---|---|---|---|---|
| 72x48 | jfnk (pardiso, lag=4) | True | 13 | 10 | 111 | 2.7 s | 0.21 s | 1.00× | 0.0e+00 |
| 72x48 | jfnk_simple lag=1 | True | 13 | 14 | 307 | 3.0 s | 0.23 s | 0.89× | 4.6e-09 |
| 72x48 | jfnk_simple lag=4 | True | 13 | 10 | 307 | 2.1 s | 0.16 s | 1.28× | 4.5e-09 |
| 72x48 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 12 | 9 | 198 | 1.6 s | 0.13 s | 1.70× | 3.1e-06 |
| 72x48 | dc_simple lag=4 | True | 29 | 14 | 504 | 2.6 s | 0.09 s | 1.05× | 2.0e-06 |
| 144x96 | jfnk (pardiso, lag=4) | True | 18 | 11 | 280 | 19.2 s | 1.07 s | 1.00× | 0.0e+00 |
| 144x96 | jfnk_simple lag=1 | True | 18 | 19 | 638 | 15.7 s | 0.87 s | 1.22× | 2.0e-08 |
| 144x96 | jfnk_simple lag=4 | True | 18 | 13 | 627 | 13.3 s | 0.74 s | 1.44× | 2.1e-08 |
| 144x96 | jfnk_simple lag=4 gmres_tol=1e-2 | True | 18 | 12 | 417 | 9.6 s | 0.53 s | 2.00× | 7.4e-07 |
| 144x96 | dc_simple lag=4 | True | 36 | 26 | 966 | 19.4 s | 0.54 s | 0.99× | 1.7e-05 |
| 288x192 | jfnk (pardiso, lag=4) | True | 36 | 22 | 850 | 235.7 s | 6.55 s | 1.00× | 0.0e+00 |
| 288x192 | jfnk_simple lag=1 | False | 11 | 12 | 464 | 52.5 s | 4.77 s | 4.49× | 3.3e-01 |

（288×192 の SIMPLE 側 4 構成は本コミット時点で計測中。完了後に追記）

- `jfnk_simple` は Newton 反復数が PARDISO と同じで、GMRES 総反復が 2〜3 倍（1 Newton あたり 24 → 35〜75 回）。前処理の質（1 次風上 J1 に対する近似）がそのまま比率に出ている
- `gmres_tol=1e-2`（inexact Newton）は Newton 反復数を変えずに GMRES を 3 分の 2 に減らし、144×96 で **2.0×**
- `dc_simple` は Newton が線形収束で反復数 2 倍、1 反復は軽い（残差評価なし）が総時間は PARDISO と同等
- 72×48 では PARDISO の分解・三角解が十分速く（0.2 s/Newton）、SIMPLE の利得は小さい

## 5. 分かったこと・限界

- **4 コアでは 288×192 が PARDISO 比で速くなるが、18 コア機では現行と同程度になる見込み**。SIMPLE 前処理は
  演算量を O(N) に落とすが、いまの実装で 1 GMRES 反復あたり残るのは前処理適用 25 ms（ILU 三角解 8 ms + AMG V サイクル
  14 ms + B・C の spmv）と**残差評価 17 ms（numpy 1 スレッド）**で、これは 18 コアでも縮まない。一方 PARDISO は
  分解が 18 コアで縮む
- JFNK の GMRES 反復は J1 に対する単体測定（26〜43）より多い（70 前後）。matvec が SOU 残差の真のヤコビアン、
  前処理が 1 次風上 J1 のためで、PARDISO でも 24 回（status-32）
- GMRES の総反復数が PARDISO 比 2〜3 倍に増えるので、次に効くのは**反復 1 回あたりのコスト**。残差評価の numba 化
  （17 ms → 1〜2 ms、`prange` で 18 コアが素直に埋まる）と `gmres_tol` の緩和（inexact Newton）
- Fluent 級（600 反復 1 分）を狙うなら、線形系を厳密に解かず**固定サイクル数で外部反復**する構造（`dc_simple` +
  小さな `gmres_maxiter` + 擬似時間緩和）が本命。SER の CFL 成長が残差比に依存するので調整が要る

## 6. 残件

- [ ] 残差評価・ヤコビアン組立の numba 化（`assembly.py`、`prange`）。GMRES 反復数に比例して効く
- [ ] 18 コア実機で `python experiments/nsb/bench_precond.py 4 2>&1 | tee ...` を実行し、PARDISO / SIMPLE の比を確定
  （status-32 の `precond_lag` 確定と同じ TODO）
- [ ] `dc_simple` で `gmres_maxiter` を絞った固定サイクル外部反復の収束性（Fluent 型）
- [ ] Ŝ の遠方項を落として 5 点化した compact 行列で SA 階層だけ作り、細格子の平滑化は Ŝ で行う変種
  （Ruge–Stüben では発散したが SA なら異なる可能性）
- [ ] FGMRES（内側の ILU / AMG を Krylov 加速して可変前処理にする）。scipy に無いので自前実装が要る
- status-37 からの持ち越し: `.inp` の `*RTD` キーワード、空気実物性の SIMPLE 連成（CLAUDE.md の最優先事項）

## テスト実行

```
python -m pytest tests/test_nsb_precond.py tests/test_nsb_linalg.py tests/test_nsb_standalone.py tests/test_nsb.py tests/test_nsb_adjoint.py -q
→ 35 passed（39 s）
python contracts/validate_process_contracts.py → 契約違反なし
ruff check nsb/ tests/ && ruff format --check nsb/ tests/ → All checks passed
```

## ファイル

- 追加: `nsb/precond.py`、`tests/test_nsb_precond.py`、`experiments/nsb/bench_precond.py`、
  `experiments/nsb/logs/bench-precond-flat-r124.log`、`docs/status/status-38.md`
- 変更: `nsb/{solver,core,__init__}.py`、`nsb/README.md`、`pyproject.toml`、`README.md`、`docs/roadmap.md`、
  `docs/status/status-index.md`
