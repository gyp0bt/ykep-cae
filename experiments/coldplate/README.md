# 冷却流路 概念設計ソルバ (coldplate v2)

[<- リポジトリ README](../../README.md)

**注意**: 本ディレクトリは xkep-cae-fluid 本体 (FDM/FVM ソルバー・Process Architecture)
とはほぼ独立した実験タスク。CI 対象外で、追加依存に torch を使う
(`pip install torch` が別途必要)。

## 概要

格子グラフ上の回路方程式 (Hagen-Poiseuille + Kirchhoff) と鮮度スカラー輸送で、
発熱体アレイ (4×8) に対する冷却流路レイアウトを秒〜分オーダーで生成する
「エセトポロジー最適化」(ground structure 法 + SIMP)。
定式化・較正パラメータ・踏んだ罠は `coldplate.py` 冒頭 docstring と
引き継ぎメモ (会話ログ) を参照。

## v2 での修正 (v1 からの差分)

### P0: 「孤立している流路にも液が流れる」偽解の排除

v1 は SIMP の `rho_floor = 1e-4` により、消したはずの辺にも
`g = rho_floor^p · w^4 / L > 0` の伝導度が残り、Kirchhoff 解が全域に微小流を
配っていた。節点レベルの質量保存 `div q = b` 自体は破れていない
(体積保存「制約」の問題ではなく、**ゼロであるべき辺に流れが乗る**モデル構造の問題)。
この漏れ流れが除熱の得点として目的関数に混入し、`connectivity_check() = False`
の分断解が最適に見えていた。小規模検証では総流量 Q=1 に対し
`rho<0.5` の辺が計 1.38 の |q| を運んでいた (`leak_inactive`)。

対策 (引き継ぎメモ P0-(a) の実装 = 逐次刈り込み):

1. `prune()` — `rho > 0.5` (不成立なら閾値を段階的に緩和) の辺で連結成分を取り、
   inlet/outlet の重みを両方含む主成分だけを残す。さらに厳密再解して
   `|q| < 1e-6·Q_tot` の死枝 (無流量のヒゲ・孤立リング) を除去。
2. `decode(mask=...)` — 主成分外の辺は **rho = 0 (厳密ゼロ)**、主成分内は
   **rho = 1** に凍結。ポート softmax も主成分内に制限 (自動再正規化)。
3. `physics(edge_on=...)` — アクティブ節点のみで Kirchhoff を解く。
   凍結辺は g = 0 なので流量は**厳密にゼロ** (数値ノイズすらない)。
4. `optimise(mask=...)` — 凍結位相の上で幅 w とポート配分を再最適化。
5. 1-4 を マスクが安定するまで反復 (`solve_pipeline`, 既定 3 ラウンド)。

検証 (`mass_check`):

- `resid_max` — 節点質量残差 max|div q − b| → 1e-9 以下
- `leak_inactive` — 凍結辺の流量合計 → **厳密に 0.0**

### P1: grey (中間 rho) の排除

- stage A で penal 1→2→3 / softmin β 4→6→8 / mu_bin 0.3→1→3 の continuation
- 刈り込み後は rho ∈ {0, 1} に凍結するため **grey = 0 が構造的に保証**される

### 均一性

- ブロック別除熱の `block_cv` (変動係数)・`block_min_over_mean` を `report()` に追加
- 刈り込み後ラウンドは softmin β = 12 に強化して worst ブロックを直接押し上げ
- v1 の「不均一に見える流れ」の一因は漏れ流れの得点混入 (見かけの除熱が
  分断領域にも付く)。厳密ゼロ化後の数字が設計間比較に使える値
- 追加の均一性罰則 `Config.mu_uni` (ブロック別 log 除熱の分散) をオプションで用意

## 実行

```bash
cd experiments/coldplate
python coldplate.py 2>&1 | tee logs/log-coldplate-v2-$(date +%s).log  # フル実行 (数分)
python -m pytest test_coldplate.py -v                                  # テスト (~40 秒)
```

主要 API (v1 互換 + v2 追加):

```python
prob = make_problem(ntu_coef=1e-2)          # 4x8 発熱体、左辺ポート
cfg  = Config(vol_frac=0.25)
x_raw, x, mask = solve_pipeline(prob, cfg, log_p_max=5.0)  # continuation + 刈り込み
r    = report(prob, cfg, x, mask)           # cooling, block_cv, leak, ...
mass_check(prob, cfg, x, mask)              # ★ leak_inactive == 0.0 を必ず確認
connectivity_check(prob, cfg, x, mask=mask) # ★ connected == True を必ず確認
check_gradient(prob, cfg, mask=mask)        # 1e-4 以下なら健全
plot(prob, cfg, x, fig, ax, mask=mask)      # 実スケール描画
```

## 残課題 (引き継ぎメモの番号)

- P3: `ntu_coef` の実機校正 (現状 φ_out ≈ 0.3 になる便宜値)
- P4: 伝導スプレッディング未モデル化 (footprint 4 辺のみで除熱判定)
- P5: ヘッダ部の幅制約緩和 (並列細管でヘッダを代用する不自然さ)
- 刈り込みで到達不能になったブロックは softmin から除外して orphan 報告する
  設計のため、被覆保証はない (`blocks_covered` を必ず確認)。被覆を厳密に
  保証するには P0-(c) の整数計画 (被覆制約付き Steiner tree) が必要。
