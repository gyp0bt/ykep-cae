# status-32: nsb の xkep_cae_fluid からの切り離し（コピー方式）と高速化の効果見積り（実測）

[<- README](../../README.md) | [<- status-index](status-index.md) | [nsb/README](../../nsb/README.md) | [前: status-31](status-31.md)

**日付**: 2026-09-04
**ブランチ**: `claude/nsb-performance-optimization-ajmpkf`
**テスト数**: 520（`tests/test_nsb_standalone.py` +4。本セッション環境（pyamg / numba 未導入）で `pytest tests/` は 501 passed / 8 skipped / 1 xfailed / 10 failed、失敗 10 件はすべて pyamg・numba の ImportError で nsb と無関係）
**契約違反**: 0 件（登録プロセス 19）

## 目的

1. `nsb/` を **xkep_cae_fluid に依存せず単体で持ち出せる**ようにする（共有離散化はコピーで持ち、xkep 側にも残す）
2. nsb 高速化の候補（for ループ削減 / pypardiso 置換 / JAX 化）が**それぞれどの程度効くか**を実測で見積もる

## 1. 切り離し（コピー方式）

### 実装

- `xkep_cae_fluid/brinkman_flow/{data,assembly}.py` → `nsb/{data,assembly}.py` にコピー。差分は import 行のみ
  （`from xkep_cae_fluid.brinkman_flow.` → `from nsb.`）。xkep 側は無変更で残す
- `nsb/{core,geo,solver,adjoint,utils}.py` の import を `nsb.data` / `nsb.assembly` に切替。
  `nsb/__init__.py` からマスク補助（`west_span` 等、`disk_mask`、`smooth_disk`）、`BoundaryKind`、
  `ConvectionSchemeType`、`BrinkmanDiscretization` を再エクスポート
- `scripts/sync_nsb_from_xkep.py`: コピー同期（`--reverse` で nsb → xkep、`--check` で乖離検査）
- `tests/test_nsb_standalone.py`（+4）: (a) コピーが import 行以外で xkep 側と一致、(b) コピー内に `xkep_cae_fluid` 文字列なし、
  (c) 書き換え規則の純粋性、(d) **xkep_cae_fluid を import 不能にした別プロセス**で nsb 全モジュールが読み込め `make_case` が動く
- `tests/test_nsb.py`, `tests/test_nsb_adjoint.py`, `experiments/nsb/*.py`, `nsb/README.md` の import を nsb 側へ
- Process ソルバーとの一致テストは `to_xkep_flow_input()` で名前詰め替え（`nsb.data.BoundaryKind` と xkep 側は別クラスで
  `is` 比較が跨げないため）

### 運用ルール

- xkep 側 `brinkman_flow/{data,assembly}.py` を変えたら `python scripts/sync_nsb_from_xkep.py` を実行する。
  忘れると `tests/test_nsb_standalone.py::test_copies_match_xkep_modulo_imports` が落ちる
- nsb 側で先に直した場合は `--reverse`
- 持ち出し: `nsb/` ディレクトリをコピーし `pip install numpy scipy`（`nsb/README.md`）

## 2. 高速化の効果見積り（実測）

### 計測条件

- 環境: 本セッションのコンテナ（4 コア、実機は 18 コア 32 GB）。numpy 2.4.6 / scipy 1.17.1 / pypardiso（MKL）
- ケース: flat、U=1、推奨構成（`velocity_floor=0.1`, `init_field="stokes"`, `alpha_u=1.0`, cfl 0.5、JFNK: GMRES + LU(J1) 前処理）
- スクリプト: `experiments/nsb/profile_stages.py`（段別の累積時間）、`experiments/nsb/bench_linear_solver.py`（同じ J1 での LU 比較）
- ログ: `experiments/nsb/logs/profile-stages-flat-r124.log`、`experiments/nsb/logs/bench-linear-solver-flat-r124.log`
- 実行: `python experiments/nsb/profile_stages.py 2>&1 | tee ...`、`python experiments/nsb/bench_linear_solver.py 2>&1 | tee ...`

### 段別の内訳（solve_steady 全体に対する割合）

| 格子 | 未知数 3N | 反復 | 全体 | splu（LU 分解） | GMRES（前処理 lu.solve 含む） | compute_state（残差の面値・RC） | jacobian 組立 | residual_from_state |
|---|---|---|---|---|---|---|---|---|
| 72×48 | 10,368 | 13 | 3.7 s | 2.61 s (70%) | 0.81 s (22%) | 0.20 s (5%) | 0.16 s (4%) | 0.05 s (1%) |
| 144×96 | 41,472 | 18 | 40.3 s | 30.6 s (76%) | 8.6 s (21%) | 1.06 s (3%) | 0.46 s (1%) | 0.26 s (1%) |
| 288×192 | 165,888 | 31 | 757 s | 613 s (81%) | 134 s (18%) | 11.9 s (1.6%) | 3.5 s (0.5%) | 2.8 s (0.4%) |
| 72×48（手元構成、80 反復未収束） | 10,368 | 80 | 19.6 s | 13.9 s (71%) | 4.1 s (21%) | 0.96 s (5%) | 0.79 s (4%) | 0.25 s (1%) |

- **LU 分解が 70〜81%**、GMRES が約 20%（その大半は前処理の三角解 `lu.solve`、残差評価は compute_state に計上済み）。
- 配列演算部分（残差・ヤコビアン組立）は合計 **5〜10%** に留まる。格子が細かいほど LU の比率が上がる（fill-in が超線形に増えるため）。

### 同じ J1 での LU 比較（`bench_linear_solver.py`、4 コア）

| 格子 | nnz(J1) | splu(COLAMD) 分解 | splu 三角解 | fill-in (L+U) | pypardiso 分解 | pypardiso 解 | 分解の比 |
|---|---|---|---|---|---|---|---|
| 72×48 | 159,552 | 0.178 s | 3.8 ms | 3.0 M | 0.141 s | 2.9 ms | 1.3× |
| 144×96 | 643,968 | 1.54 s | 25.7 ms | 18.6 M | 0.307 s | 16.1 ms | **5.0×** |
| 288×192 | 2,587,392 | 17.1 s | 145 ms | 104 M | 4.61 s | 155 ms | **3.7×** |

- `permc_spec="MMD_AT_PLUS_A"` は 72×48 で 15.9 s（fill 27 M）と COLAMD の 90 倍遅く不適。`MMD_ATA` は COLAMD と同程度
- pypardiso は 4 コアで分解 4〜5 倍速。三角解（前処理適用）は splu と同程度で速くならない
- 実機 18 コアではさらに分解が速くなる見込み（PARDISO のスレッド並列は 8〜16 コアまで比較的よく伸びる）。ただし三角解は伸びにくい

### 各案の見積り

| 案 | 効く部分 | 見込み（推奨構成、4 コア実測ベース） | 備考 |
|---|---|---|---|
| **for ループ削減** | 残差・ヤコビアン組立 | **ほぼ 0%**（全体の 5〜10% の中のさらに一部） | `assembly.py` は既に配列演算・疎行列オペレータ合成で書かれており、セル単位の Python ループは無い。残るループは Newton 反復、`adjoint.colored_fd_jacobian` の彩色（150 回残差評価 = 構造上必要）、境界 4 辺の小ループのみ |
| **pypardiso 置換**（`splu` → PARDISO） | LU 分解（70〜81%） | 144×96: 40 s → **約 15 s（2.7×）**、288×192: 757 s → **約 290 s（2.6×）**。18 コアなら 3〜4× | 三角解（GMRES 前処理、約 20%）は速くならないので上限はそこで決まる。`splu` の LU オブジェクトの代わりに `PyPardisoSolver.factorize` + `solve` を `LinearOperator` に包めばよく、変更は `solve_linear` 内に局所化できる |
| **JAX 化** | 残差・ヤコビアン組立（5〜10%） | 速度面は **ほぼ効かない**（配列部分が既に小さい）。むしろ JIT のコンパイルと XLA→scipy の往復（疎 LU は JAX に無い）で遅くなる恐れ | 価値は速度ではなく **autodiff**: 彩色 FD ヤコビアン（150 残差評価、`adjoint.py`）を `jax.jacfwd` / 疎 JVP で厳密化、∂R/∂θ の中心差分を解消。外側の最適化を JAX で書くなら custom VJP と相性が良い。GPU に載せるなら線形ソルバーも反復法（GMRES + ブロック前処理）に変える必要があり、別プロジェクト規模 |

### pypardiso と組み合わせるとさらに効くアルゴリズム側の手（追加見積り）

- **前処理 LU の遅延更新（lagged preconditioner）**: JFNK なので前処理は近似でよい。LU を毎 Newton 反復ではなく k 反復ごと（残差比が悪化したら再分解）にすれば
  分解回数が 1/2〜1/4 になる。GMRES 反復は数回増える。**LU 分解 70〜81% → 20〜40%** が見込め、pypardiso と独立に効く
- **収束後半の Newton 反復数**: 288×192 で 31 反復（72×48 は 13）。格子が細かいほど反復が増えるのは SER の CFL 成長が残差比で頭打ちになるため。
  `cfl_max` / `ser_growth` の調整か、粗格子解の補間を初期場にする（マルチレベル継続）で反復数自体を減らせる
- **随伴（`adjoint.py`）**: forward + 勾配 3.8 s（72×48）の内訳は LU 2 回 + 150 残差評価。ここは pypardiso よりも彩色回数の削減（1 次風上なら radius=1 で 3·9·2=54 回）が効く

### 結論（優先順）

1. **pypardiso**: 効果最大かつ変更が局所（`nsb/solver.py::solve_linear` の LU 生成を差し替え）。2.5〜4× を見込む。依存が増える（MKL）ので `NSBSettings` にバックエンド選択を置き、未インストール時は splu にフォールバックする形が妥当
2. **前処理 LU の遅延更新**: pypardiso と独立に分解回数を減らせる。実装 30 行程度、収束性の確認が必要
3. **for ループ削減**: 対象がない（既に配列演算）。着手不要
4. **JAX 化**: 速度目的では非推奨。随伴の厳密化・外側最適化との統合が目的なら検討

いずれも本 status では**見積りのみ**で、ソルバー本体は無変更。

## 次にやること

- [ ] `NSBSettings.lu_backend = "splu" | "pardiso"` を追加し、`solve_linear` で pypardiso を使う（フォールバック付き）。18 コア実機で再計測
- [ ] 前処理 LU の遅延更新（残差比で再分解を判断）と収束性の確認
- [ ] 288×192 の Newton 反復数増加（31 回）の原因切り分け（SER の頭打ち / 粗格子初期場）
- status-31 からの持ち越し: 境界 inlet の連続化、冷却設計向け目的関数、熱ソルバー連携、Process 側への Stokes 初期場反映

## ファイル

- `nsb/{data,assembly}.py`（コピー新設）、`nsb/{__init__,core,geo,solver,adjoint,utils}.py`（import 切替）、`nsb/README.md`
- `scripts/sync_nsb_from_xkep.py`（新設）、`tests/test_nsb_standalone.py`（+4）
- `tests/test_nsb.py`（`to_xkep_flow_input` 追加）、`tests/test_nsb_adjoint.py`、`experiments/nsb/{inlet_sweep,manifold_demo,manifold_optimize}.py`（import 切替）
- `experiments/nsb/profile_stages.py`、`experiments/nsb/bench_linear_solver.py`（新設）+ `logs/profile-stages-flat-r124.log`、`logs/bench-linear-solver-flat-r124.log`
