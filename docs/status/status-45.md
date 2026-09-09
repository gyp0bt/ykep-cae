# status-45: nsb 蛇行流路（messi trama パターン）の収束不良 — 高 Re・階段・前処理の 3 観点で可視化

[<- README](../../README.md) | [status-index](status-index.md) | [status-44](status-44.md) | [図解レポート](../reports/nsb-trama-convergence.md) | [roadmap](../roadmap.md)

日付: 2026-09-09 / ブランチ: `claude/nsbm-learned-init` / gyp さんの指示（21:23）「NSB の収束性悪化の調査。
`../tmp/pattern.json` の左端 2 か所を inner cell の inlet/outlet、ρ 1000、μ 3e-3、ṁ 0.15、600×350、Δx 1.5、
h 3.8e-3 / 1e-5。高 Re による条件数悪化・斜め流路を格子で区切った不連続性・前処理の弱点の 3 観点で可視化」
+ 追加（21:30）「inlet/outlet をセル内に付けるか壁面に付けるかの条件数の違いも知りたい」。

本文・図・数字は全て [docs/reports/nsb-trama-convergence.md](../reports/nsb-trama-convergence.md) にある。
ここには結論と成果物だけ書く。**コードの修正はしていない**（調査と可視化のみ）。

## 1. 結論

| 観点 | 結果（0.0015 → 0.15 kg/s、最初の Newton 反復、cfl 0.25） | 判定 |
|---|---|---|
| 高 Re | κ₂(J1+τ) は 1.6e12 → 2.1e12 でほぼ不変（両端とも閉塞領域が決める: σ_max = 12μ/h²·V = 811、σ_min ≈ 4e-10 の純圧力モード）。変わるのは RC 結合 d = V/a_P（50 倍弱まる）→ 厳密 J1 歩の圧力成分 60 → 8.4e6（ṁ^2.6）→ その方向で真の作用素と食い違い、真の残差比 1.16 → **81** | 主因。「条件数」ではなく「J1 の圧力方向が平らになり Newton 歩が場の外へ飛ぶ」 |
| 階段 | 階段セル 203（0.8%）の残差分担 ≈ 1%。直交化版も未収束。直線流路は 0/15/30/45° どれも健全（同じ Re・ポート・格子で Ritz 0.03〜1.03、棄却ゼロ） | 無関係。引き金は蛇行の U ターンで Stokes 出発点が遠いこと（\|R_ref\| 190 vs 直線 7.2） |
| 前処理 | SIMPLE の Schur 近似が高 Re で崩れ Ritz 1.6e-4〜2.8e6。JFNK と組むと Krylov 基底が閉塞域・出口円板の圧力レベル方向（1/σ_min = 2e9 倍）に汚れ、Givens 0.077 / 真の残差 **30**。LU(J1) なら 3.3e-3 だが、その Newton 一歩で残差 **5060 倍** | 主因（増幅器）。線形解を直しても歩の方向が悪い |
| ポート（追加） | κ は内部 2.1e12 / 壁 2.3e12 で同等。厳密歩の真の残差比は内部 81 / 壁 1.6（0.0015 では 1.16 / 0.15）。壁ポート 0.0015 は 2 反復で rel 4e-2、内部は it=1 で 6.6 倍に跳ねる | 条件数は変わらない。内部ポート（pressure sink の運動量項 + 円板縁の RC 結合）が食い違いを 30〜50 倍増やす |

- CFL を縮めても直らない理由: τ は u, v 行だけを太らせ、圧力の谷底は τ に依らない。cfl 1e-6 でも SIMPLE 真の残差 4.5、LU(J1) 22。
- 副次: JFNK の有限差分 matvec が τ を 2 重に数えている（真の作用素 J+2τ、前処理 J1+τ、実測比 2.000）。発散の原因ではないが実効 CFL が半分。
- 0.0015（1/100）は最初の一歩で 6.6 倍に跳ねて CFL 0.03 で立て直し、80 反復で rel 3.7e-6（判定 1e-6 にあと一歩）。「1/100 なら収束」と整合。

## 2. 成果物

| 種別 | パス |
|---|---|
| ケースビルダー（trama JSON → 厚さ場、内部/壁ポート、ortho 変種、直線流路） | `experiments/nsb/trama_case.py` |
| 最初の線形系の取り出し（Stokes 場、τ、J1、有限差分 matvec） | `experiments/nsb/trama_lin.py` |
| 特異値・厳密歩・τ 検算 | `experiments/nsb/trama_sv.py` |
| GMRES 4 組（{J1, 有限差分} × {SIMPLE, LU}）・Ritz 値・残差マップ | `experiments/nsb/trama_diag.py` |
| ε 掃引（差分商がノイズか導関数か） | `experiments/nsb/trama_epsscan.py` |
| 図 | `experiments/nsb/trama_plots.py` → `experiments/nsb/results/trama_figs/*.png` |
| 走行結果（yaml + 場） | `experiments/nsb/results/trama_*.yaml / *.npz / *.json` |
| ログ | `experiments/nsb/logs/trama-*.log` |

走行一覧（全て 400×233、Δx 1.5 mm、jfnk_simple / sou / cfl_init 0.25、`--max-iter` 40〜80）:

| 走行 | 結果 |
|---|---|
| 内部ポート 0.15 | it=1 棄却（真の残差比 4.09）→ 棄却連鎖、CFL 1e-38、max_iter |
| 内部ポート 0.0015 | 80 反復で rel 3.7e-6（未収束 max_iter、あと一歩） |
| 壁ポート 0.15 | it=1 通過（残差 9.3 倍）→ it=2 から棄却連鎖 |
| 壁ポート 0.0015 | it=2 で rel 4.3e-2（走行中に打ち切り、収束に向かう） |
| 斜め区間を直交化 0.15 | 未収束（残差 1.5〜4 で停滞、CFL 1e-5） |
| 残差も 1 次風上 0.15 | 内部ポートと同じ棄却連鎖 |
| 前処理 LU(J1) PARDISO 0.15 | it=1 線形解 1.6e-3 で通過、残差 5060 倍、40 反復で rel 1700 |
| 直線流路 0/15/30/45° 0.15 | 棄却ゼロ、40 反復で rel 9.2e-2 / 7.6e-3 / 7.4e-4 / 6.3e-3 |

## 3. 前提と仮定

- 座標: trama 単位を一様 4.93 mm/unit で 600×350 mm の中央に置いた（bbox 62×57 unit を非等方に伸ばすと流路幅が方向で変わる）。
- 出口は `interior_pressure_sink`（C = 1e-3 kg/(s·Pa)、p = 0）。
- 乱流モデルなし（指示どおり）。

## 4. 次にやること（roadmap に転記）

1. 圧力レベル方向で J1 と真の作用素を揃える（出口 sink 項・RC 結合の厳密化、または defect correction）。
2. 閉塞セル圧力の谷底を埋める（h_blocked 1e-5 → 1e-4、または閉塞セルの圧力緩和項）。
3. 高 Re 用の Schur 近似（SIMPLEC / PCD / LSC）。
4. τ の 2 重カウント修正。
5. 圧力歩のガード（|δp| が場の数倍を超えたら縮小）。
