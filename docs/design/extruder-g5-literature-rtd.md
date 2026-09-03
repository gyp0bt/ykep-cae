# 単軸押出 ゲート G5 — 文献 RTD モデルとの照合（設計）

日付: 2026-09-04 / 対象ブランチ: `claude/single-screw-extruder-impl`
親文書: `docs/design/single-screw-extruder.md`（§3 検証の階段、§7 Phase 計画）

## 0. 何を差し替えるのか

実装計画 §D と roadmap は「Phase 2（粘性発熱）に入る前に実機データと突き合わせる」を
前提にしていた。実機も想定機も無いので、この前提を **文献の古典 RTD モデルとの照合**
に差し替える。押出量 Q–ΔP 側は形状係数 F_d/F_p の級数を自前実装し Tadmor & Gogos の
値を pytest で固定済み（G1/G2）なので、文献に繋がっていないのは **RTD の形** だけである。

```
 Pinto–Tadmor 1970（文献モデル）              ykep 2.5D（Phase 1/1.5 実装済み）
 無限幅・等温ニュートン・漏れ無し              有限幅（フライト）・隙間 δ・同じ流体
 上層 ξ>2/3 ↔ 下層 を往復する閉じた流線        Picard 結合 u,v,w → 粒子追跡 → RTD
        │  自前で再導出                                │  H/W = 0.117 → 0.06 → 0.03
        ▼                                              ▼
   F_PT(t/t̄)  r によらない 1 本の曲線   ←── 比較 ──→  F_ykep(t/t̄_theory) が浅溝極限で F_PT に収束
```

比較の物差しは **各自の理論平均 t̄ = V/Q で規格化した縮約曲線**。ykep 側の t̄ は
G4b の厳密関係 `t_mean_theory = z_axial·A_free/(sinφ·Q_axial)` を使う（標本平均は
裾支配で信用できない、§3.1）。文献モデルは浅溝極限なので、**H/W を下げるにつれて
単調に近づき、最浅で許容内に入る**ことをゲートにする。

## 1. 文献モデル（真値の供給源）

Pinto, G. & Tadmor, Z. (1970) *Polym. Eng. Sci.* 10, 279. Tadmor & Gogos
*Principles of Polymer Processing* §7 に再掲。仮定: 無限幅平板（側壁無し）、
等温ニュートン、漏れ無し、フライトでの折返しは瞬時。ξ = y/H（0 スクリュー根元、
1 バレル）、r = Q_p/Q_d ∈ (−1, 0]（0 = 純引きずり、−1 = 閉塞）。

```
  下流    ŵ(ξ) = w/V_z = ξ + 3r·ξ(1−ξ)
  横断    û(ξ) = u/V_x = 3ξ² − 2ξ         （正味流量 0、ξ=2/3 で符号反転）
```

上層 ξ ∈ (2/3, 1) の粒子はフライトで折り返し、横断流束の保存で決まる下層の高さ
ξ_c ∈ (0, 2/3) に移る:

```
  ∫_ξ^1 û dξ' = −∫_0^{ξ_c} û dξ'   ⇔   g(ξ) = g(ξ_c),  g(s) = s²(1−s)
```

1 周の時間重み（横断 1 回 ∝ 1/|û|）で平均した下流速度 w̄ と滞留時間:

```
  w̄/V_z = ( ŵ/û + ŵ_c/|û_c| ) / ( 1/û + 1/|û_c| )        t = L/w̄
  流線対の運ぶ流量  dQ ∝ [ ŵ + ŵ_c·|dξ_c/dξ| ] dξ,   |dξ_c/dξ| = û/|û_c|
```

**普遍性の機構（本設計で確認した恒等式）。** 3ξ(1−ξ) = ξ − (3ξ²−2ξ)、すなわち
圧力流れ分布は「引きずり分布 − 横断分布」に恒等的に等しい。横断速度は閉じた
流線 1 周で変位ゼロ（往って戻る）なので周平均が消え、どの流線でも
w̄ = (1+r)·w̄_drag、dQ = (1+r)·dQ_drag。t も t̄ も同じ因子で割られ **F(t/t̄) は r に
依存しない**。t̄ = HWL/Q は流管の体積÷流束として厳密に出る（t̄·V_z/L = 2/(1+r)）。

**数値評価は下層 ξ_c ∈ (0, 2/3) で標本化する（中点則）。** 上層 ξ で標本化すると
ξ→1 の粒子が根元 ξ_c ~ √(1−ξ) に張り付いて t ~ 1/√(1−ξ) と発散し、中点則が
O(1/√n) にしか収束しない（実測 1000→16000 点で 0.37%→0.09%）。下層で標本化して
流線対の流量 dQ = [ŵ(ξ_c) + ŵ(ξ)·|û_c|/û] dξ_c を重みにすると被積分関数が滑らかになり、
1000 点で t̄ が 1e-7、普遍性が 1e-6 で成立する。相手 ξ は g(ξ) = g(ξ_c) を
(2/3, 1) で二分法（ベクトル化、g は単調減少）で解く。ξ_c→0 で t → ∞（根元に
張り付く）なので F は 1 に漸近する裾を持つ。r < −1/3 では根元付近の ŵ が負（逆流）に
なるが、流線対の正味流量は (1+r)×引きずり対流量 > 0 なので重みは常に正。

再導出の数値（n_ξ = 4000, 本設計時に確認）: **t_min/t̄ = 0.750000**（文献の 3/4 は
厳密値。最速の粒子はバレル面ではなく再循環の停留高さ ξ = 2/3 に居て一度も
折り返さない粒子で、t_min = L/(⅔V_z) = ¾·(2L/V_z)）、p10/t̄ = 0.7524、
p50/t̄ = 0.8225、p90/t̄ = 1.3247。

## 2. 部品と境界

| 部品 | 場所 | 依存 | 公開 I/F |
|---|---|---|---|
| 文献モデル | `xkep_cae_fluid/extruder/pinto_tadmor.py` | numpy のみ | `pinto_tadmor_rtd(r: float = 0.0, n_xi: int = 4000) -> PintoTadmorRTD` |
| 結果型 | 同上 | — | `PintoTadmorRTD(t_over_tbar, F, t_min_ratio, t_p10_ratio, t_p50_ratio, t_p90_ratio, tbar_over_L_Vz)` frozen dataclass。`t_over_tbar` 昇順、`F` 同長で 0→1 |
| 重み付き ECDF | `xkep_cae_fluid/extruder/rtd.py` に `weighted_ecdf(values, weights) -> (sorted_values, F)` を追加 | — | ykep 側の F 曲線をビン幅に依存せず作る。既存 `weighted_quantile` と同じ流儀（区間中点の累積） |
| モデル単体テスト | `tests/test_extruder_pinto_tadmor.py` | 上 2 つ | §3 |
| ゲート G5 | `tests/test_extruder_literature_rtd.py` | 既存パイプライン | §3 |
| 実験スクリプト | `experiments/extruder/g5_literature.py` | 既存パイプライン | `--out DIR` に `result.json` |
| レポート生成 | `experiments/extruder/g5_report.py` | `result.json` | `docs/reports/extruder/g5-literature-rtd.md` |

ソルバー・追跡・RTD Process のロジックには触れない。`shape_factors.py` と同じく
`pinto_tadmor.py` は「真値の供給源」であり、検証とレポートだけが使う。
`__init__.py` に `pinto_tadmor_rtd`, `PintoTadmorRTD`, `weighted_ecdf` を追加。

エラー: r ∉ (−1, 0] は `ValueError`（r=−1 は Q=0 で t̄ 発散）。n_xi < 16 は `ValueError`。
`weighted_ecdf` は重み総和 ≤ 0 で `ValueError`（`weighted_quantile` と同じ）。

## 3. 判定（すべて閾値規格化比、比 < 1.00 で合格）

### 3.1 モデル単体（`test_extruder_pinto_tadmor.py`）

| 検査 | 真値 | 許容 |
|---|---|---|
| t̄·V_z/L | 2/(1+r)（体積÷流量の厳密値） | \|比−1\| / 1e-3 |
| 普遍性 | r ∈ {0, −0.3, −0.7} で p10/p50/p90 比が一致 | \|差\| / 1e-5 |
| t_min/t̄ | 3/4（厳密値） | \|差\| / 1e-4 |
| F の性質 | 単調非減少、F[0]=0 近傍、F[-1]=1 | 厳密 |
| 引数検査 | r=−1, r=0.1, n_xi=8 → ValueError | — |

### 3.2 ゲート G5（`test_extruder_literature_rtd.py`）

40 mm 機の D, lead, e を保ち H = 4, 2, 1 mm（H/W = 0.117, 0.059, 0.029）、
δ/H = 0.025、G = 0、μ = 1000 Pa·s、z_axial = 50 mm。格子は G4b と同じ粗さ
（ny_bulk=16, n_gap=6, nx_channel=40, nx_land=12）から始め、必要なら 1 段細かく。

| 検査 | 閾値 | 比の定義 |
|---|---|---|
| p50/t̄ の PT 比が最浅で 1 に入る | 0.03 | \|p50_ykep/t̄_theory ÷ p50_PT − 1\| / 0.03 |
| p10/t̄ 同上 | 0.03 | 同様 |
| p90/t̄ 同上 | 0.05 | 同様 |
| 単調接近 | — | 上の偏差が H/W = 0.117 → 0.029 で単調減少 |
| 曲線全体 | 0.05 | max_{F≤0.9} \|F_ykep − F_PT\|（t/t̄ 軸で補間）/ 0.05、最浅のみ |

閾値 3% / 5% の根拠: 側壁の影響が及ぶ幅は概ね H なので、断面幅に占める割合は
2H/W ≈ 6%（H/W=0.029）。分位点への影響はその半分程度と見積もる。フル解像度で
「単調には近づくが 3% に届かない」場合は、H/W=0.015 を 1 本足して側壁効果の残りか
格子かを切り分け、閾値を変えるならレポートに理由を書く（黙って緩めない）。
粗格子で 30 s を超えるテストは `@pytest.mark.slow`。

### 3.3 実験（合否に入れない観察）

- 同系列をフル解像度（ny_bulk=60, n_gap=20, nx_channel=200, nx_land=48）で流し、
  F(t/t̄) 重ね描きと分位点比の表を作る。
- 最浅 H/W で r = −0.3 に相当する G を 1 本流し、**有限幅・隙間ありでも普遍性が
  保たれるか**を観察する（G は Q_p/Q_d = −H²·G·F_p/(6μ·w_barrel·F_d) から逆算）。
- 閉チャネル（δ=0）の最浅 1 本で、側壁の裾伸びと隙間の短時間側の広がりを分離する。

## 4. レポート（`docs/reports/extruder/g5-literature-rtd.md`）

G3 レポートと同じ型（`g3_report.py`）。構成:

1. 全体像（§0 の図を inline SVG に）と結果表（閾値規格化比）
2. F(t/t̄) 重ね描き（inline SVG polyline: PT 1 本 + ykep 3 本 + r=−0.3 1 本）
3. メカニズム: (a) なぜ r によらないか（§1 の恒等式）、(b) 側壁が裾を伸ばす
   （Bigg & Middleman 1974 が有限幅で数値的に示した機構: 側壁近傍の遅い流線が
   分布の長時間側に足される）、(c) 隙間が短時間側を広げる（G4b の知見の再掲）、
   (d) 実測との鎖: Pinto–Tadmor 曲線は Wolf & White 1976 の放射性トレーサ実験で
   計量部について確認されている、と記す（数値の転記はしない）
4. 再現手順（`g5_literature.py --out /tmp/of-g5` → `g5_report.py`）

mdview で HTML 化し `Artifact` で公開、URL を `docs/reports/extruder/README.md` に載せる。

## 5. 文書更新

- `docs/design/single-screw-extruder.md` §3 の表に G5 行、§7 Phase 2 の前提を
  「G5 文献照合 ✅」に。
- `docs/roadmap.md` Phase 7: `[ ] 実機データとの突き合わせ` → `[x] 文献 RTD 照合 G5`。
- `docs/plans/2026-09-02-single-screw-extruder-impl.md` §D の前提文を差し替え。
- `docs/status/status-29.md` + `status-index.md` + `README.md` テスト数更新。

## 6. 範囲外

- 実測 RTD 点列の転記（案 B）。A が通ってから判断。
- べき乗則での文献比較（Pinto–Tadmor はニュートンのみ）。
- 新しい数値ソルバー・Phase 2 の実装。
