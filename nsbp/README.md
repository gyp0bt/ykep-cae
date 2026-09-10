# nsbp: nsb の離散化を PETSc（petsc4py）で解く

[<- README](../README.md) | [nsb](../nsb/README.md) | [status-42](../docs/status/status-42.md)

`nsb/`（2D 深さ平均 Brinkman–NS、Newton + 擬似時間 SER）の**離散化を一切変えずに**、ソルバー部品だけを
PETSc に置き換えて限界まで速くする実験パッケージ。nsb との解の一致（max|Δu|/max|u| ≈ 1e-6、収束判定の差）で
「離散化は同じ、違うのはソルバーだけ」を担保する。

```
                nsb（既存）                                nsbp（本パッケージ）
  ┌──────────────────────────────┐          ┌──────────────────────────────────────────┐
  │ NSBInput / BrinkmanDiscretization │ ──係数──▶ │ problem.py   ゴースト付きパッチへ切り出し   │
  │   （厚さ場・境界・抗力・流量源）    │          │ kernels.py   パッチ残差 numba（fastres と同値）│
  ├──────────────────────────────┤          ├──────────────────────────────────────────┤
  │ JFNK 差分 matvec + 自作 FGMRES     │          │ DMDA(dof=3, box 幅 2)  MPI 分割・ゴースト交換  │
  │ SIMPLE 型前処理（ILU + SA-AMG）    │          │ SNES FD カラーリング  厳密 J を残差 77 回で組立 │
  │ 擬似時間 SER（乗法形・棄却）        │          │ KSP FGMRES + PC（bjacobi/asm ILU、fieldsplit） │
  │                                  │          │ 擬似時間 SER（同じ制御則、定常残差で駆動）      │
  └──────────────────────────────┘          └──────────────────────────────────────────┘
```

## ファイル

| ファイル | 役割 |
|---|---|
| `problem.py` | `Patch`（所有範囲とゴースト範囲）、`PatchCoefficients`（nsb の係数配列をパッチに切り出す） |
| `kernels.py` | `residual_patch`（ゴースト付きパッチの残差、nsb.fastres と同じ離散化）、`dtau_patch` |
| `solver.py` | `NSBPSettings` / `NSBPSolver` / `NSBPResult` / `solve_steady`。DMDA・SNES・KSP を持つ Newton + 擬似時間ループ |
| `launch.py` | PETSc 同梱 MPICH の `mpiexec` を探す |
| `../experiments/nsbp/bench.py` | 1 ケース実行 → `[bench-json]` 1 行。`--nested 1,2,4` で入れ子反復、`--compare-nsb` で nsb と解を比較 |
| `../tests/test_nsbp_kernels.py` | 残差カーネルが nsb と一致（全格子・2×2 分割のゴースト幅 2）、petsc4py 不要 |
| `../tests/test_nsbp_solver.py` | FD ヤコビアンの Newton、flat の nsb との一致、uturn 収束、3 ランク MPI が逐次解を再現（slow） |

## 使い方

```bash
python experiments/nsbp/bench.py --model flat --refine 4 --u 0.1              # 逐次
$(python -c "from nsbp.launch import mpiexec_path; print(mpiexec_path())") -n 4 \
    python experiments/nsbp/bench.py --model flat --refine 4 --u 0.1          # 4 ランク
python experiments/nsbp/bench.py --model uturn --refine 2 --u 1 --nested 1,2   # 入れ子反復（72×48 → 144×96）
```

```python
from nsb.geo import make_case
from nsbp.solver import NSBPSettings, solve_steady
res = solve_steady(make_case("uturn", 2, u_in=1.0), NSBPSettings())   # res.u/v/p は全ランクに同じ (nx, ny)
```

`nsbp.solver` は import 時に PETSc（MPI_Init）を初期化する。`nsbp` パッケージ自体は petsc4py 無しでも読み込める
（`nsbp.HAVE_PETSC`）。

## 実測（status-42、20 コア機、定常残差 1e-6）

| ケース | nsbp 逐次 | nsbp 8 ランク | nsb（status-41、20 コア） |
|---|---|---|---|
| flat 288×192、Stokes 発進 | 13.8 s（15 Newton） | **5.2 s** | 44.5 s（43 Newton） |
| flat 288×192、入れ子 72→144→288 | 15.5 s | **4.3 s** | 18.5 s（13 Newton） |
| flat 576×384、Stokes 発進 | — | 53.2 s（18 Newton） | 309.9 s |
| flat 576×384、入れ子 72→…→576 | — | **39.1 s** | 194.6 s |
| uturn 144×96 U=1 | 5.6 s（27 Newton） | **1.2 s** | 13.2 s（50 Newton） |
| uturn 144×96 U=2 | 7.7 s（40） | 3.8 s（61） | 収束（時間の個別記録なし） |

nsb との解の差は max|Δu|/max|u| で 1e-5 台。律速は KSP（ILU 適用 + matvec）で、8 ランク超は伸びない（メモリ帯域）。
詳細・機構・駄目だった試行は [status-42](../docs/status/status-42.md)。

## 壁セル（status-48）

`NSBInput.h_solid` を与えると h ≤ h_solid のセルを固体として未知数から外す（実装は
`nsb.assembly` の面マスク `wall_x` / `wall_y` をパッチに切り出すだけ。詳細は `nsb/README.md`）。
FD カラーリングは固体セルの行を全ゼロにするので、`jacobian_steady` で対角に 1 を足して単位行にする。

これは nsbp にとって単なる高速化ではない。**ILU(2) を壊していたのは行列サイズではなく
抗力コントラスト**（流路 2493 : 閉塞 3.6e8 = 1.4×10⁵ 倍）で、固体セルを外すとそれが 1 になる。
uturn 144×96 U=1 で KSP 811 → 395 反復、12.2 → 4.2 s（4 ランクでも 1 ランクと 5 桁一致）。

## 設計の論点（なぜこの形か）

### ステンシル幅 2 とパッチ端の扱い

残差 R(i) は u(i−2 … i+2)（対角含む box）に依存する: 面流束 f(i+1/2) は Rhie–Chow 係数 d(i), d(i+1) を使い、
d は a_P、a_P は面速度 u_f(i+3/2) = (u(i+1)+u(i+2))/2 を使う。2 次風上の外挿量 ex(i+1) も勾配（面値 i+1/2, i+3/2）と
Venkatakrishnan の隣接極値 u(i+2) を使う。したがって DMDA の box ステンシル幅 2 で足り、ゴーストセル 2 層の外側
（分割の切れ目）を outlet と同じ零勾配コピーで埋めても、その影響はゴーストセル自身の残差にしか届かない
（`tests/test_nsbp_kernels.py::test_split_patches_with_ghost_width_2_reproduce_full_residual`）。

### FD カラーリングの厳密ヤコビアン（JFNK をやめた理由）

nsb の JFNK は差分 matvec の非線形雑音（1e-4〜1e-3）が GMRES の Givens 推定と食い違い、真の残差確認で空回りする
（status-39）。nsbp は DMDA の色分け（幅 2 box、dof 3 → 75 色 + 基準 2 回 = 77 回の残差評価）で
**リミター込みの厳密ヤコビアン**を組む。288×192 で残差 1 回 0.5 ms 級なので組立は 40〜60 ms、
matvec は疎行列積になり雑音がない。副作用: 擬似時間ステップ内の Newton がほぼ厳密に解けるので
**擬似時間込み残差 |R_τ| は毎ステップ 1e-2〜1e-3 倍に落ち、収束判定にも SER にも使えない**。
nsbp は Δτ 非依存の定常残差で判定・SER する（nsb は JFNK の解き残し（gmres_tol 1e-3）と CFL 成長で |R_τ| と定常残差が
合流するので |R_τ| で足りていた）。

Stokes 参照場も同じ FD ヤコビアンで Newton する（Rhie–Chow 係数 d = V/a_P の速度依存まで微分に入るので
1 ステップでは残差が 2 割残る。4〜6 回で 1e-8）。nsb の「1 次風上ヤコビアンで 1 ステップ」とは参照場が僅かに違うが、
参照残差 |R_ref| の差は数 % で収束判定の閾値にしか効かない。

### ヤコビアンの疎パターン（13 点）

`da.createMatrix()` は box 幅 2 の 25 セル分（1 行 75 非零）を零で埋めた構造を返す。残差の真の依存は
(0,0)、(±1,0)、(0,±1)、(±2,0)、(0,±2)、(±1,±1) の 13 セル（1 行 39 非零）なので、`NSBPSolver._create_jacobian_13pt`
が DMDA の番号付けでその構造だけを確保する（乱数場で box 版と要素ごとに一致することを確認済み）。
matvec と ILU の nnz が半分になる。**ILU のレベルはパターン相対**なので、box パターンの ILU(1) と同じ強さを出すには
13 点パターンでは ILU(2) が要る（ILU(1) のままだと 288×192 で KSP が 200 反復に張り付いて発散した）。

### 前処理の選定（全系 ASM + ILU(2) が既定、Schur は uturn で壊れる）

| 前処理 | uturn 72×48 U=1 | flat 288×192 | 備考 |
|---|---|---|---|
| **asm(重なり 2) + ILU(2)**（既定） | 16 Newton / KSP 242 / 1.3 s | 15 / 563 / 13.3 s（1 ランク）、15 / 851 / **5.2 s**（8 ランク） | 逐次では ILU(2) そのもの。重なりで圧力の大域結合を保つ |
| bjacobi + ILU(2) | 16 / 324 / 1.1 s | 1 ランク同上、2 / 4 / 8 / 16 ランクで KSP 1024 / 3279 / 3068 / 3125、8 ランク 13.6 s | 分割で結合が切れ、並列で速くならない |
| hypre Euclid ILU(2)（並列 ILU） | — | 8 ランク: KSP 1078 / 87.9 s | 反復は保つが適用が桁で遅い |
| lu（PETSc 直接法） | 16 / 21 / 4.0 s | — | Newton 経路の検証用（ヤコビアン・SER が正しいことの基準） |
| fieldsplit Schur selfp + hypre（運動量 ILU） | **発散**（Stokes から KSP 674 反復・残差比 7.9） | 収束せず 60 反復 139 s | Ŝ = D − C diag(A)⁻¹B に BoomerAMG 1 V サイクル: 抗力 1e4 倍のコントラストで不安定 |
| 同上、運動量を LU | 発散 | — | 運動量側の問題ではない |
| 同上、Ŝ を LU | 16 / 624 | — | Ŝ 自体は使えるが近似が弱く反復が倍 |
| 同上、Ŝ を gamg | 16 / 1410 | — | |

nsb の SIMPLE 型（ILU + Schur SA-AMG）は同じ構造だが pyamg の smoothed aggregation が uturn でも保った。PETSc では
Schur の内側に hypre/gamg を preonly で置くと FGMRES が壊れる（真の残差比 > 1 = 前処理が発散している）。
**擬似時間対角があれば全系 ILU で足りる**（CFL 1e2〜1e3 の後半でも KSP 5〜10 反復）ので Schur を使わない。

### リミター凍結（厳密ヤコビアンの副作用への対処）

2 次風上 + Venkatakrishnan の残差は分岐（上流側の切替、ψ の min）で非滑らかで、厳密ヤコビアンの Newton は
定常残差 1e-4〜1e-5 で振動して止まる（288×192: CFL 150〜300 で残差が上下し SER が揺れる）。
差分幅を変えても（列差分 1e-5 / 1e-3、Walker–Pernice）変わらない。1 次風上か ψ≡1（`venkat_k=1e6`）なら 13〜15 反復で
1e-6 に落ちるので、リミターの分岐そのものが原因。`limiter_freeze_tol`（既定 1e-3）で定常残差がその比を割った時点の場で
ψ をセルごとに固定し（`kernels.limiter_psi` → 残差・ヤコビアンとも固定 ψ）、以後は滑らかな 2 次風上として
2 次収束させる。凍結後 3 反復で 1e-8 級。凍結時点以降の ψ の変化ぶんだけ真の制限解とずれるが、
nsb の収束解との差は max|Δu|/max|u| で 1e-5 台（flat 144×96）。

### 参照場と収束判定

Stokes 参照場（cs=0）は FD ヤコビアンで Newton し `stokes_tol=1e-2` で止める。参照場は |R_ref| = |R_NS(x_stokes)| を
決めるだけなので 1% 動いても収束判定の閾値が 1% 動くだけ。擬似時間対角のない鞍点系は ILU に重い
（288×192 で 1e-8 まで解くと 16 s、1e-2 なら 2.3 s）。収束判定・SER は定常残差（Δτ 非依存）で行う（上記）。

SNESComputeJacobianDefaultColor の注意: 直接 `snes.computeJacobian` を呼ぶときは `snes.setSolution(X)` で解ベクトルを
登録しておく（未登録だと基準関数値の比較で NULL 参照して落ちる）。SNES 内部の MatFDColoring はオプション prefix を
継がないので `-mat_fd_coloring_err` などは `petsc_options_global` で入れる。

## インストール（この PC の手順、2026-09-08）

- `pip install petsc` は `--download-mpich` 付きだと隔離ビルド環境で MPICH の make に失敗し、システム OpenMPI 5
  （Ubuntu）を使うと **mpiexec 無しのシングルトン起動で MPI_Init が prte 待ちでハングする**（`mpiexec -n 1` なら動く）。
- 採った手順: sdist を展開して手で configure → make → prefix install → petsc4py を `--no-build-isolation` で組む。

```bash
pip download --no-deps --no-binary :all: petsc==3.25.5 -d ~/opt && cd ~/opt && tar xzf petsc-3.25.5.tar.gz && cd petsc-3.25.5
./configure PETSC_ARCH=arch-mpich --prefix=$HOME/opt/petsc-mpich --with-debugging=0 \
    --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-mpich-device=ch3:nemesis \
    --download-hypre --download-openblas --with-make-np=16 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
make PETSC_DIR=$PWD PETSC_ARCH=arch-mpich all && make PETSC_DIR=$PWD PETSC_ARCH=arch-mpich install
PETSC_DIR=$HOME/opt/petsc-mpich pip install --no-build-isolation petsc4py==3.25.5   # cython>=3, numpy が要る
```

mpi4py は要らない（大域 min は PETSc の Vec で取る）。`mpiexec` は `$HOME/opt/petsc-mpich/bin/mpiexec`
（`nsbp.launch.mpiexec_path()` が petsc4py の設定から引く）。
