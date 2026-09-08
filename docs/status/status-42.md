# status-42: nsbp — nsb の離散化を PETSc（petsc4py）で解く限界高速化

[<- README](../../README.md) | [status-index](status-index.md) | [status-41](status-41.md) | [nsbp/README](../../nsbp/README.md) | [roadmap](../roadmap.md)

日付: 2026-09-08 / ブランチ: `claude/nsbp-petsc` / gyp さんの依頼「nsb とは別プロジェクトで petsc4py で NSB の限界高速化トライ。推奨設計で自動承認」

## 1. 結論

- **離散化は nsb と同一のまま**、ソルバー部品を PETSc に置き換えた `nsbp/` を作った。解は nsb と一致
  （flat 144×96: max|Δu|/max|u| = 1.7e-5、max|Δp|/max|p| = 7.8e-7。両者とも定常残差 1e-7 まで収束）。
- 速さ（表は §4）: flat 288×192 が nsb 最良 18.5 s（入れ子、20 コア）に対し nsbp は **逐次 13.8 s、8 ランク 5.2 s**（Stokes 発進）。
  入れ子（72→144→288、各段 tol 1e-4、双一次）は 8 ランクで **4.3 s**。576×384 は nsb 195 s（入れ子）に対し
  入れ子 8 ランク **39.1 s**、Stokes 発進 8 ランク 53.2 s。uturn 144×96 U=1 は nsb 13.2 s に対し逐次 5.6 s、8 ランク **1.2 s**。
- 効いた部品は 3 つ。(a) **FD カラーリングの厳密ヤコビアン**（残差 77 回で組む。JFNK の差分雑音が消え、
  1 反復あたりの KSP が 5〜70 反復）、(b) **13 点疎パターン + ASM(重なり 2)/ILU(2)**（並列で KSP 反復を保つ。
  bjacobi は 4 ランクで反復 6 倍、Schur/hypre は uturn で発散、Euclid は 17 倍遅い）、(c) **リミター凍結**
  （厳密 Newton は Venkatakrishnan の分岐で 1e-4〜1e-5 に停滞する。定常残差 1e-3 で ψ を固定して 2 次収束へ）。
- 律速は KSP（ILU 適用 + matvec）。残差評価とヤコビアン組立は合計 10% 以下。

## 2. 全体像

```
   nsb（既存、参照実装）                          nsbp（本ステータス）
 ┌────────────────────────────┐   係数・境界   ┌─────────────────────────────────────────────┐
 │ NSBInput / BrinkmanDiscretization │ ───────────▶ │ problem.py  ゴースト幅 2 のパッチへ切り出し        │
 │ 厚さ場 h・抗力 12μ/h²・境界・流量源 │              │ kernels.py  パッチ残差（numba、fastres と同値）    │
 ├────────────────────────────┤              ├─────────────────────────────────────────────┤
 │ JFNK 差分 matvec + 自作 FGMRES     │   ≠          │ DMDA(dof=3, box 幅 2)  MPI 分割・ゴースト交換      │
 │ SIMPLE 型前処理 ILU + SA-AMG       │   ≠          │ SNES FD カラーリング 75 色 → 13 点パターンの厳密 J │
 │ SER 乗法形・線形失敗の棄却         │   =          │ KSP FGMRES + ASM(重なり 2)/ILU(2)                 │
 │ 収束判定 |R_τ|                     │   ≠          │ SER 同じ制御則、ただし定常残差で駆動 + リミター凍結  │
 └────────────────────────────┘              └─────────────────────────────────────────────┘
        解の一致 max|Δu|/max|u| 1e-5 ◀──────────────────────▶ 同じ格子・同じ境界・同じ収束判定 1e-6
```

1 反復の流れ（nsbp）: Δτ を決める → R_τ = R + ρV(u−u_prev)/Δτ → J_steady（FD カラーリング、残差 77 回）
→ J_τ = J_steady + diag → FGMRES(ASM/ILU(2)) で δ → 真の残差比 > 0.3 なら棄却・CFL×0.1 → x += δ
→ 定常残差で SER（×2 まで成長、増えたら比で縮小）。

## 3. メカニズム（なぜこの形になったか）

### 3.1 厳密ヤコビアンが SER と収束判定を変える

nsb は擬似時間込み残差 |R_τ| で収束判定と SER をしていた。JFNK（gmres_tol 1e-3）では擬似ステップが解き残るので
|R_τ| は定常残差と同程度で、CFL 成長とともに一致していく。厳密ヤコビアンではステップ内 Newton がほぼ厳密に解けて
|R_τ| が毎ステップ 1e-2〜1e-3 倍に落ち、**定常残差が 1e-4 でも |R_τ| は 1e-6 を割る**（偽収束、flat 72×48 で発生）。
nsbp は判定・SER とも Δτ 非依存の定常残差で行う。

### 3.2 リミターの分岐で Newton が止まる

厳密 Newton は flat 288×192 で定常残差 4e-5〜1e-4、uturn 144×96 で 5e-6 に停滞し、CFL 150〜300 で残差が上下する。
1 次風上や ψ≡1（`venkat_k=1e6`）なら 13〜15 反復で 1e-6。差分幅（列差分 1e-5 / 1e-3、Walker–Pernice）は無関係。
→ Venkatakrishnan の分岐（上流側の切替、面ごとの min）で残差が非滑らかになり、Newton 方向が分岐を跨いで振動する。
対処: 定常残差 1e-3 で ψ をセルごとに固定（`limiter_freeze_tol`）。以後は滑らかな 2 次風上で 2 次収束（3 反復で 1e-8）。
凍結時点以降の ψ の変化ぶんだけ真の制限解とずれるが、nsb の収束解との差は 1e-5 台。nsb 自身も |R_τ| 判定なので
「定常残差 1e-6 の制限解」を持っているわけではない（flat 144×96 では nsb も定常 1e-7 に達している）。

### 3.3 前処理: Schur は uturn で壊れ、全系 ILU は擬似時間対角があれば足りる

| 候補 | uturn 72×48 U=1（Stokes 参照場 → NS） | 判断 |
|---|---|---|
| fieldsplit Schur selfp + hypre（運動量 ILU） | Stokes から KSP 674 反復・真の残差比 7.9、NS は棄却の連続 | 抗力 1e4 倍のコントラストで BoomerAMG の V サイクルが不安定（真の残差比 > 1 = 前処理が発散） |
| 同、運動量を LU | 同様に発散 | 運動量側ではない |
| 同、Ŝ を LU / gamg | 収束するが KSP 624 / 1410 | Ŝ = D − C diag(A)⁻¹B の近似が弱い |
| PETSc 直接 LU | 16 Newton / KSP 21 / 4.0 s | Newton 経路の基準（ヤコビアン・SER は正しい） |
| **全系 ILU(2)**（bjacobi / asm） | 16 / 242〜324 / 1.1〜1.3 s | 擬似時間対角 ρV/Δτ が入るので鞍点性が弱く、ILU で十分 |

並列では bjacobi の分割で圧力の大域結合が切れ、flat 288×192 で KSP 反復が 1 → 4 ランクで 563 → 3279 と 6 倍に
なり速くならない。重なり 2 の ASM は 8 ランクで 851（1.5 倍）に留まり 5.2 s。並列 ILU（hypre Euclid）は反復を保つが
適用が遅く 87.9 s。

### 3.4 疎パターン

DMDA の `createMatrix()` は box 幅 2 の 25 セル（1 行 75 非零）。残差の真の依存は 13 セル（39 非零）で、
乱数場で要素ごとに一致することを確認して 13 点パターンを自前で確保した。ILU のレベルはパターン相対なので、
25 点の ILU(1) と同等の強さには 13 点で ILU(2) が要る（ILU(1) のままだと KSP が 200 に張り付いて発散）。

### 3.5 Stokes 参照場

FD ヤコビアンは Rhie–Chow 係数 d = V/a_P の速度依存まで微分に入るので、cs=0 の残差も 1 ステップでは 2 割残る。
参照場は |R_ref| を決めるだけなので Newton を 1e-2 で止める（288×192: 1e-8 まで解くと 16 s、1e-2 で 2.3 s。
擬似時間対角のない鞍点系は ILU に重い）。

## 4. 計測

全て `experiments/nsbp/bench.py`（既定: asm/ILU(2)、ksp_rtol 1e-3、cfl_init 0.25、limiter_freeze_tol 1e-3、Stokes 発進、
newton_tol 1e-6 は**定常残差**に対して）。20 コア機、ランクごとの numba スレッドは 20/ランク数。
ログ: `experiments/nsbp/logs/scaling-{flat-r4,flat-r8,uturn}.log`、`parallel-pc-flat-r4.log`、`uturn-r1-{pc,schur}-sweep.log`、
`flat-r4-pc.log`、`uturn-r2-pc.log`、`mg-trial-flat-r4.log`、`nested-uturn-U2.log`。nsb の値は status-41（20 コア）。

### 4.1 flat 288×192（U=0.1）— ランク数スケーリング

| ランク | Newton | KSP 総反復 | Stokes 参照場 [s] | KSP [s] | 残差+J [s] | 合計 [s] | nsb 比 |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 563 | 2.5 | 11.9 | 2.9 | **13.8** | Stokes 発進 44.5 s の 3.2× |
| 2 | 15 | 578 | 1.7 | 7.6 | 1.9 | 8.8 | |
| 4 | 15 | 795 | 1.2 | 5.9 | 1.4 | 6.8 | |
| 8 | 15 | 851 | 0.9 | 4.5 | 1.1 | **5.2** | 8.6×、入れ子最良 18.5 s の 3.6× |
| 16 | 15 | 909 | 1.0 | 5.0 | 0.8 | 5.6 | 8 ランクで頭打ち（KSP がメモリ帯域律速） |
| 1、入れ子 72→144→288 | 8+9+10 | 103+391+565 | | | | 0.5+2.5+12.5 = 15.5 | 細格子の KSP が減らない |
| 8、入れ子 72→144→288 | 8+9+10 | 154+276+592 | | | | 0.2+0.5+3.6 = **4.3** | nsb 入れ子 18.5 s の 4.3× |

入れ子で Newton は 15 → 10 に減るが、細格子の KSP 総反復（565）は Stokes 発進（563）と変わらない。
後半（CFL 1e2〜1e3、凍結後）の 1 反復あたり KSP が 60〜70 で、Newton 数より 1 反復の線形コストが支配的。

### 4.2 flat 576×384（U=0.1）

| 構成 | Newton | KSP 総反復 | Stokes [s] | KSP [s] | 合計 [s] | nsb 比 |
|---|---|---|---|---|---|---|
| 8 ランク、Stokes 発進 | 18 | 2515 | 8.4 | 49.9 | **53.2** | Stokes 発進 309.9 s の 5.8× |
| 16 ランク、Stokes 発進 | 18 | 2697 | 8.8 | 55.9 | 59.3 | 16 ランクは遅い（帯域） |
| 8 ランク、入れ子 72→…→576 | 8+9+8+10 | 154+276+500+1684 | 8.3 | 33.4 | 0.2+0.5+3.1+35.3 = **39.1** | 入れ子最良 194.6 s の 5.0× |
| 16 ランク、入れ子 | 8+9+8+10 | 177+288+592+1695 | 8.7 | 35.4 | 41.5 | |

1 Newton あたり KSP が 288×192 の 57 → 576×384 の 140 に伸びる（ILU は格子とともに劣化、粗格子補正がない）。
細格子の Stokes 参照場 8.4 s も入れ子では純粋な overhead（§7）。

### 4.3 uturn（厚さ 1e-3 / 1e-5、抗力 1e4 倍）

| ケース | ランク | Newton | KSP 総反復 | 合計 [s] | nsb |
|---|---|---|---|---|---|
| 144×96 U=1 | 1 | 27 | 811 | 5.6 | 50 Newton、13.2 s（本セッションの比較走行） |
| 144×96 U=1 | 4 | 26 | 633 | 1.9 | |
| 144×96 U=1 | 8 | 26 | 564 | **1.2** | 11× |
| 144×96 U=2 | 1 | 40 | 1080 | 7.7 | status-41 の掃引で収束（時間は個別記録なし） |
| 144×96 U=2 | 4 | 47 | 1885 | 4.3 | Newton 数がランク数で動く（ASM の分割で経路が変わる） |
| 144×96 U=2 | 8 | 61 | 2778 | 3.8 | |
| 72×48 U=2 | 4 | 20 | 319 | 0.5 | |
| 144×96 U=2、入れ子 72→144 | 8 | 18 + 61 | 347 + 2778 | 0.3 + 3.6 | 双一次補間の初期場は参照の 162 倍（注入でも 2.4 倍）→ Stokes 発進に戻す |

uturn の入れ子は効かない: 厚さの境界（幅 0.1 m は 72×48 で 10.3 セル、格子に乗らない）をまたぐ補間が
閉塞セルへ流速を持ち込み、抗力残差 12μ/h²·u が参照残差の 100 倍超になる。古典形出発 CFL が 1e-3 まで落ちた
状態で ILU が壊れて棄却が連鎖するので、「初期場が参照場より悪ければ Stokes 発進」の規則を入れた。

### 4.4 nsb との解の一致

| ケース | nsbp Newton / nsb Newton | max\|Δu\|/max\|u\| | max\|Δp\|/max\|p\| | 定常残差（nsbp / nsb） |
|---|---|---|---|---|
| flat 144×96 U=0.1 | 12 / 13 | 1.7e-5 | 7.8e-7 | 1.2e-7 / 9.8e-8 |
| uturn 144×96 U=1 | 27 / 50 | 2.0e-04 | 5.2e-06 | 3.7e-07 / 4.4e-07 |

3 ランクと逐次の解の差は 2.5e-6（ASM の分割で Newton 経路と凍結時点が動く。`tests/test_nsbp_solver.py`）。

### 4.5 やって駄目だったこと（設計上の論点として）

| 試行 | 結果 | 機構 |
|---|---|---|
| 擬似時間込み残差 \|R_τ\| で判定・SER（nsb と同じ） | flat 72×48 で定常 1e-4 のまま「収束」 | 厳密 Newton でステップ内が解け切り \|R_τ\| が情報を失う（§3.1） |
| Schur fieldsplit + hypre / gamg / 内側 GMRES | uturn で発散、flat 288×192 で 139 s | Ŝ の AMG V サイクルが抗力コントラストで不安定（§3.3） |
| bjacobi/ILU を並列に | 4 ランクで KSP 6 倍、速くならない | 圧力の大域結合が切れる。ASM 重なり 2 で解決 |
| hypre Euclid（並列 ILU(2)） | 8 ランク 87.9 s | 反復は保つが適用が 17 倍遅い |
| DMDA 幾何 MG（Galerkin、Q0 補間、ASM/ILU 平滑化、粗格子 LU） | 真の残差比 24 で発散 | 鞍点系 + 上流差分に単純な MG 平滑化が効かない。KSP に DM を渡すだけで ASM の分割が変わり KSP 倍増 |
| FD 差分幅（1e-5、1e-3、Walker–Pernice） | 停滞は変わらず | 停滞はリミターの分岐（§3.2） |
| 13 点パターンで ILU(1) | KSP 200 に張り付き発散 | ILU レベルはパターン相対（§3.4） |
| pip で `--download-mpich` / システム OpenMPI | ビルド失敗 / シングルトン起動ハング | §5 |


## 5. 環境構築で嵌まったところ

- `pip install petsc` はシステム OpenMPI（Ubuntu 25.10 の 5.0.10）に対して組めるが、**mpiexec 無しのシングルトン
  起動で MPI_Init が prte 待ちでハングする**（`mpiexec -n 1` なら動く。pmix/prrte のヘルプファイル欠損と同根と思われる）。
  pytest や対話利用が成り立たないので PETSc に MPICH を同梱ビルドさせた。
- `pip install petsc` に `--download-mpich` を付けると pip の隔離ビルド環境で MPICH の make が失敗する（原因は
  一時ディレクトリごと消えて追えない）。sdist を展開して手で configure/make/install する手順に切り替えた（nsbp/README）。
- `pip install petsc4py` は隔離環境で PETSc をもう一度ビルドしようとする（10 分の無駄）。`--no-build-isolation`。
- mpi4py はリンク不良（`MPI_UNWEIGHTED` 未定義）で使わないことにした。大域 min は PETSc の Vec で取る。
- `snes.computeJacobian` を直接呼ぶと SEGV: SNESComputeJacobianDefaultColor が解ベクトルとの比較で NULL を触る。
  `snes.setSolution(X)` を先に呼ぶ。

## 6. 変更ファイル

- 新規: `nsbp/{__init__,problem,kernels,solver,launch}.py`、`nsbp/README.md`、`experiments/nsbp/bench.py`、
  `tests/test_nsbp_kernels.py`（14）、`tests/test_nsbp_solver.py`（5、うち MPI 1 は slow）、
  `experiments/nsbp/logs/*.log`
- 変更: `pyproject.toml`（extras `nsbp`）、`README.md`、`docs/status/status-index.md`、`docs/roadmap.md`

### テスト

- 新規 19 件（`tests/test_nsbp_kernels.py` 14、`tests/test_nsbp_solver.py` 5）。MPI の 1 件は `mpiexec -n 3` を subprocess で
  回す slow。全件を `-n 8` で実行: 960（941 passed / 18 failed / 1 xfailed。失敗 18 は全て既存: `test_inp_runner` 9・`test_post_mirador` 8（いずれも `VizMixin.export_html()` の `vector_field` 引数）・`test_natural_convection::TestAMGPressureSolver::test_adaptive_relaxation` 1。nsbp は全通過）。
- **既存の失敗 18 件**（本セッションの変更と無関係。代表 2 件は単独実行でも再現。status-37 以降全件が回っていなかった）:
  `tests/test_inp_runner.py` 9 件と `tests/test_post_mirador.py` 8 件は `VizMixin.export_html()` が `vector_field` 引数を
  受けない同一原因、`tests/test_natural_convection.py::TestAMGPressureSolver::test_adaptive_relaxation` は `result.converged` が False。
  roadmap に TODO として残す。

## 7. 残課題（roadmap へ）

- KSP が律速（1 反復 20 ms、逐次）。次は前処理の使い回し（`pc_lag`）、修正 Newton（`jacobian_lag`）、ILU レベルの掃引、
  KSP の `ksp_rtol` 緩和の効果を測る。
- 入れ子反復で細格子の Stokes 参照場を省く（粗格子の |R_ref| をスケーリングで推定できるか）。
- Schur 型を活かすなら Ŝ に pyamg 相当の smoothed aggregation（PETSc の gamg は反復 1410）か、抗力を対角スケーリングで
  均してから hypre に渡す。
- リミター凍結の閾値 1e-3 の妥当性（凍結解と厳密制限解のずれの定量。uturn で nsb との差 2e-4）。
- 既存テストの失敗 18 件（mirador `export_html` の `vector_field` 17 件、自然対流 AMG の適応緩和 1 件）の修正。
