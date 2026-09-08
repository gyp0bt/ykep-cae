# nsbm: nsb の初期解を学習で出す（教師あり代理、72×48 固定）設計

[<- README](../../../README.md) | [nsb/README](../../../nsb/README.md) | [status-index](../../status/status-index.md)

日付: 2026-09-09 / ブランチ: `claude/nsbm-learned-init`（`claude/nsbp-petsc` から分岐）
依頼: 「NSB ソルバーに torch でメタ学習を追加し、最良の初期解を出して反復回数を削減する。LX, LY, NX, NY は固定、
h と u_in と inlet/outlet 位置を変える条件で学習する。反復にかかる時間は置いておき反復回数だけ見る。格子は 72×48、
教師あり。空きコア数だけ並行で流す。h は uturn だけでなくランダム・2 次元 sin・2 次関数・細いパターン流路まで網羅的に」

## 1. 目的と成功の線

- 目的: 形状パラメータ θ = (h 場, u_in, inlet, outlet) から nsb の収束解 (u, v, p) を近似する写像を学習し、
  それを `NSBInput.u0/v0/p0` に入れて **Newton 反復数**を減らす。時間は評価対象にしない。
- 成功の線: 学習に使っていない θ（テスト集合）で、UNet 初期場の Newton 反復数が
  (a) Stokes 発進より少なく、(b) kNN 補間（データセット近傍の解の重み付き平均）よりも少ないこと。
  (b) に負けるなら「学習の価値なし」と報告する。
- 収束の定義は nsb と同じ（`newton_tol=1e-6`、参照は Stokes 場の完全 NS 残差）。学習器は収束判定を変えない。

## 2. 全体像

```
 [families]  θ をサンプル: h 場ファミリ(7) × u_in × inlet 壁/区間 × outlet 壁/区間
     │        閉塞系は連結性チェック（scipy.ndimage.label）で棄却
     ▼
 [dataset]   multiprocessing（ワーカー 1 スレッド × 空きコア） → nsb.solve_steady（Stokes 発進、既定設定）
     │        → npz シャード: 入力画像 X (C,72,48)、正解 Y (3,72,48)、θ の記述、n_iter、converged
     ▼
 [model]     UNet(4 段) : X → Ŷ（正規化した u, v, p）、MSE、CPU torch
     ▼
 [evaluate]  テスト θ で solve_steady を 3 通りの初期場で回し Newton 反復数を比較
             Stokes 発進 / kNN 補間 / UNet
```

## 3. パラメータ空間（`nsbm/families.py`）

固定: `nx, ny = 72, 48`、`lx, ly = 0.7, 0.4`、`rho=1000, mu=1e-3, mu_b=1e-3`（nsb 既定）。

乱数（`numpy.random.Generator(seed)`、サンプル id ごとに独立シード）:

- `h0`（基準厚さ）: 対数一様 [0.3e-3, 3e-3] m
- `u_in`: 対数一様 [0.1, 2] m/s
- inlet, outlet: 壁を {west, east, north, south} から独立に選び、区間長 [0.05, 0.15] m、位置は壁内で一様。
  同じ壁なら重ならないよう再抽選（最大 20 回）。BC は `BC.velocity_inlet` / `BC.pressure_outlet` に
  `west_span/east_span/north_span/south_span` のマスク
- h 場ファミリ（等確率、`h_blocked = h0/100`）:

| family | 生成 |
|---|---|
| `uniform` | h0 一様 |
| `sin2d` | h0·exp(a·sin(2π kx x/LX + φ)·sin(2π ky y/LY + ψ))、a∈[0.2, 1.0]、kx, ky∈[0.5, 4]、φ, ψ∈[0, 2π) |
| `quad` | h0·exp(a·((x−x0)²/LX² + s·(y−y0)²/LY²))、a∈[−1.5, 1.5]、s∈{+1, −1}、(x0, y0) 領域内一様 |
| `grf` | 対数正規: h0·exp(σ·G)、G は白色雑音をガウシアン平滑（相関長 [0.03, 0.2] m）し分散 1 に規格化、σ∈[0.3, 1.0] |
| `uturn` | 既存 `make_uturn_h` 型: inlet/outlet を west に強制し区間・折返し幅 [0.05, 0.2] を乱数化 |
| `serpentine` | 蛇行流路: 往復数 [2, 5]、流路高さ [0.03, 0.1] m、inlet/outlet 区間に接続する開口を付ける |
| `pins` / `blobs` | ピンフィン（円柱閉塞、ピッチ [0.05, 0.15]、径/ピッチ [0.3, 0.6]）または grf の閾値 2 値化（開口率 [0.4, 0.8]） |

閉塞系（uturn, serpentine, pins, blobs）は inlet 区間セルと outlet 区間セルが `h > h_blocked` の連結成分で
繋がっていることを `scipy.ndimage.label`（4 近傍）で確認し、繋がっていなければ棄却して再抽選。
inlet/outlet に接するセルは必ず開ける（境界 1 セル幅の帯を h0 にする）。

サンプル数: 2000（train/val/test = 80/10/10、θ 単位で分割、ファミリ層化）。

## 4. データ生成（`nsbm/dataset.py`）

- `Sample` dataclass: `x: (C, nx, ny) float32`、`y: (3, nx, ny) float32`、`meta: dict`（family, h0, u_in, 壁・区間, seed,
  n_iter, converged, n_gmres_total, residual_ref, elapsed）
- ワーカーは `OMP_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1` を **numpy import 前**に設定
  （生成スクリプト先頭で環境変数を設定してから import、`multiprocessing` の spawn context で子に継承）。
  ワーカー数は `os.cpu_count() − 予約`（既定 16）。
- ソルバー設定は `NSBSettings()` 既定（`newton_max_iter=80`）。未収束は `converged=False` で保存し学習ラベルから除外、
  評価集計では別枠で数える。
- 出力は `experiments/nsbm/data/shard-XXXX.npz`（256 件/シャード、`np.savez_compressed`）。生成ログは tee でファイルへ。
- 生成は `~/.claude/hooks/memcap` 下で走らせる（メモリ方針）。

## 5. 入力画像と正規化（`nsbm/model.py`）

入力 8 チャネル（すべて (72, 48)）:

| ch | 内容 |
|---|---|
| 0 | log(h/h0)（閉塞セルは log(1/100) = −4.6） |
| 1 | log(h0/1e-3) の定数面 |
| 2 | log(u_in) の定数面 |
| 3, 4 | inlet 境界セルに u_in·n_x, u_in·n_y（内向き法線）、他は 0 |
| 5 | outlet 境界セルに 1、他は 0 |
| 6, 7 | x/LX, y/LY（CoordConv） |

出力 3 チャネル: `u/u_in, v/u_in, p/p_ref`、`p_ref = 12 μ u_in LX/h0² + ρ u_in²`。
スケールは入力から決まるので推論時に逆変換できる。逆変換 `denormalize(yhat, meta) → (u0, v0, p0)`。

UNet: 4 段（72×48 → 36×24 → 18×12 → 9×6）、各段 conv3×3×2 + GELU + GroupNorm、チャネル 32-64-128-256、
skip 結合、出力 1×1 conv。約 2 M パラメータ。

## 6. 学習（`nsbm/train.py`）

- 損失: MSE（3 チャネル等重み）。閉塞セルも含めて学習する（速度 ≈ 0 を学ぶ）
- AdamW lr 1e-3、cosine 減衰、batch 32、epoch 200 目安（val loss で early stop、best を保存）
- CPU、`torch.set_num_threads(空きコア)`。データは全件メモリ常駐（2000 × 11 × 72×48 × 4 B ≈ 300 MB）
- チェックポイント `experiments/nsbm/runs/<name>/best.pt`、学習曲線 CSV

## 7. 評価（`nsbm/evaluate.py`）

テスト集合の各 θ で `solve_steady` を 3 通りの初期場で回し、`n_iter`、`converged`、初期残差比
`steady_residual_history[0] / residual_ref`、`cfl_history[0]` を記録:

1. Stokes 発進（`u0/v0/p0 = None`、データ生成時の値をそのまま使う）
2. kNN: 入力画像 X の平坦化ベクトルで train 集合の最近傍 k=4、距離の逆数重みで Y を平均 → 逆変換
3. UNet: `denormalize(model(X))`

集計: 全体・ファミリ別・u_in 三分位別の Newton 反復数の中央値/四分位、Stokes 比、勝敗数。
図: 反復数の散布（Stokes vs UNet）、ファミリ別箱ひげ、代表ケースの場の比較（正解・UNet・差）。
報告は md → mdview → Artifact。

## 8. テスト（`tests/test_nsbm_*.py`）

- families: 各ファミリの h が正・形状 (72,48)、閉塞系の連結性が成立、inlet/outlet が重ならない、シード再現
- dataset: 1 件を直列で生成して `solve_steady` が走り Sample が組める（refine 1 なので数秒）、npz 往復
- model: 正規化⇄逆変換の往復が 1e-6、UNet の入出力形状、1 バッチの backward が通る
- torch 未導入環境では `pytest.importorskip("torch")`

## 9. 配置

`nsbm/`（torch 依存の別パッケージ、nsb は変更しない）: `families.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py`, `README.md`。実行スクリプトとログは `experiments/nsbm/`（`gen.py`, `train.py`, `eval.py`, `logs/`, `data/`, `runs/`）。
`data/` と `runs/` は `.gitignore`（結果の要約 YAML と図は commit）。

## 10. 非目標

- 時間の短縮（初期場の推論・kNN の時間は測るが目標にしない）
- 格子・領域サイズの変更（72×48 固定。粗→細の転用は将来の TODO）
- 残差駆動の学習・MAML（教師ありで価値が出た後に検討）
