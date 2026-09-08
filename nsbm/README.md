# nsbm: nsb の初期場を学習で出す（72×48 固定、教師あり UNet）

[<- README](../README.md) | [nsb/README](../nsb/README.md) | [設計書](../docs/superpowers/specs/2026-09-09-nsbm-learned-initializer-design.md) | [status-43](../docs/status/status-43.md)

形状パラメータ θ = (厚さ場 h, 流入速度 u_in, inlet, outlet) から nsb の収束解 (u, v, p) を UNet で近似し、
`NSBInput.u0/v0/p0` に入れて **Newton 反復数**を減らす。格子 72×48・領域 0.7×0.4 m は固定。
時間ではなく反復数だけを見る（gyp さんの指定）。

```
 families ──▶ dataset ──▶ model/train ──▶ evaluate
 θ を乱数で    並列に nsb を   UNet(8ch → 3ch)   テスト θ で Stokes / kNN / UNet の
 8 ファミリ    解き npz へ     MSE, CPU torch     初期場から solve_steady、n_iter を比較
```

## 使い方

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU 版で足りる
bash experiments/nsbm/gen_chunks.sh 4000 250        # 空きコア追従でデータ生成（experiments/nsbm/data/）
python experiments/nsbm/train.py --epochs 120 --threads 8 --out experiments/nsbm/runs/unet-a
python experiments/nsbm/eval.py --run experiments/nsbm/runs/unet-a
```

```python
from nsbm import sample_theta, build_input
from nsbm.dataset import solve_sample
th = sample_theta(seed=7, families=["serpentine"])   # 決定的（seed → θ）
inp = build_input(th)                                 # nsb.NSBInput（Stokes 発進）
s = solve_sample(7)                                   # 解いて Sample（x: 8ch 入力画像, y: 正規化した解）
```

| ファイル | 役割 |
|---|---|
| `families.py` | `Theta`（JSON 化可）、`sample_theta`（閉塞系は inlet-outlet 連結まで再抽選）、`build_h` / `build_bc` / `build_input`、`port_cells` / `connected` |
| `features.py` | 入力 8 ch（log(h/h0)、log h0、log u_in、inlet の u_in·n、outlet マスク、x/LX、y/LY）、出力の正規化 u/u_in, v/u_in, p/p_ref（p_ref = 12μu_in LX/h0² + ρu_in²） |
| `dataset.py` | `solve_sample` / `generate`（spawn Pool、ワーカー 1 スレッド）/ `save_shard` / `load_shards` |
| `model.py` | `UNet`（4 段 72×48→9×6、GroupNorm+GELU、約 2M パラメータ） |
| `train.py` | `split_by_family`（収束サンプルのみ、ファミリ層化 80/10/10、`split.json`）、`train`（AdamW + cosine、val 最良を `best.pt`）、`load_model` |
| `evaluate.py` | `knn_predict`（標準化した入力画像の L2 近傍 k=4、距離逆数重み）、`run_with_init`、`evaluate`（並列）、`summarize` |

## h 場のファミリ（`families.py`）

| family | 生成 | 閉塞 |
|---|---|---|
| uniform | h0 一様 | なし |
| sin2d | h0·exp(a sin(2πkx x/LX+φ) sin(2πky y/LY+ψ))、a∈[0.2,1]、k∈[0.5,4] | なし |
| quad | h0·exp(a((x−x0)²/LX² ± (y−y0)²/LY²))、a∈[−1,1] | なし |
| grf | 対数正規ガウス場（相関長 [0.03,0.2] m、σ∈[0.3,1]） | なし |
| uturn | 既存 `make_uturn_h`（inlet/outlet は west、折返し幅 [0.05,0.2]） | h0/100 |
| serpentine | 2〜5 往復の蛇行流路（流路高さ [0.03,0.1]、端で交互に連結） | h0/100 |
| pins | 円柱ピンフィン配列（正方/六方、ピッチ [0.05,0.15]、径/ピッチ [0.3,0.6]） | h0/100 |
| blobs | ガウス場の分位点 2 値化（開口率 [0.4,0.8]） | h0/100 |

共通: h0 対数一様 [0.3e-3, 3e-3] m、u_in 対数一様 [0.1, 2] m/s、inlet/outlet は 4 壁いずれかに長さ [0.05, 0.15] m
（同じ壁なら重ならない）。滑らか系は log(h/h0) を ±2 に留める。閉塞系はポートから内側へ最初の開きセルまで
廊下を切り、`scipy.ndimage.label` で inlet-outlet の連結を確認する（棄却率 約 3%）。

## 結果（status-43）

テスト 357 件（72×48、`cfl_init` 0.25）で、UNet 初期場は Stokes 発進に **41 勝 17 分 299 敗**（Newton 中央値 19 vs 13）、
kNN 補間も 34 勝 302 敗。効くのは Stokes 発進が遅い裾と、Stokes で未収束だった 60 件のうち 8 件の救済だけ。
機構は 3 つ: (a) 閉塞セルに残る速度が Brinkman 抗力 12μ/h² で 1e4 倍に増幅され残差比 1000 になる（`mask_blocked` で 2〜7）、
(b) 72×48 の反復数は SER の CFL 梯子（0.25 → 数百）で決まり、予測場の残差比 2〜8 では出発 CFL が上がらない、
(c) 予測場は Newton の吸引域の外（減衰なしの 1 歩 `nsbm/project.py` は 6 割で残差が増えて棄却）。
`cfl_init` 4 の方が効く（Stokes 発進 13 → 7、未収束 3%）。詳細と TODO は [status-43](../docs/status/status-43.md)。
