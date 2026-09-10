# status-49: 領域内ポートを実パッチにする — 刳り抜きポート（nsb）

[← README](../../README.md) ｜ [status 一覧](status-index.md) ｜ [図解レポート §13](../reports/nsb-trama-convergence.md)

- 日付: 2026-09-10（夜）
- ブランチ: `claude/nsbm-learned-init`
- 依頼: gyp さん「nsb の inner cell 境界条件を openfoam と合わせたい。outlet はセル内圧力固定境界を実装したい」
- 前提: [status-48](status-48.md)（壁セル）、[status-47](status-47.md)（OpenFOAM 検算）

## 1 行で

内部ポートを「セル内の体積ソース／コンダクタンス sink」から
**「円板セルを刳り抜き、露出したリング面を inlet / outlet の実パッチにする」** に変えた。
OpenFOAM の `cylinderToCell` + `subsetMesh` と 1 対 1。
0.005 kg/s の検算で、速度 L2 の**距離依存が消えた**（内部ポート 11.9→1.8% の減衰 → 刳り抜き 0.98→0.90% の平坦）。

## 何を作ったか

面ごとの情報を **向き × 種別** の 2 枚に分けた。status-48 は向き（どちら側が流体か）しか
持てず、種別は「壁」に固定だった。

| 配列 | 意味 |
|---|---|
| `wall_x` / `wall_y` | **向き**。0 開 / 1 右・上が流体 / 2 左・下が流体 / 3 両側固体（既存のまま） |
| `pkind_x` / `pkind_y` | **種別**。0 = 壁、1 = inlet、2 = outlet |
| `pun_x` / `pun_y` | inlet 面の法線流入速度 [m/s]（+x / +y を正とする符号つき） |
| `pp_x` / `pp_y` | outlet 面の指定圧力 [Pa] |

面 1 枚の役割は 4 辺で既に書いてある式をそのまま内部面に持ってきただけ:

- **inlet 面**: 面速度 = 法線に u_n、面圧力 = 流体側セル値、面勾配は片側 2 倍（速度 Dirichlet）、
  質量流束は RC 補正なしの ρ A u_n、対流面値は面値
- **outlet 面**: 面速度 = 流体側セル値（ゼロ勾配）、面圧力 = 指定値、面勾配 0、
  質量流束は RC 補正なしの ρ A u_P

`u_n = ṁ / (ρ Σ_f h_f A_f)` は 4 辺の `MASS_FLOW_INLET` と同一の式で、OpenFOAM の
`flowRateInletVelocity`（`u_n = Q/(L_perim·tz)`、`tz = h_channel`）と単位まで一致する。

### API

```python
BC.port_inlet(disk_mask(cx, cy, r), mass_flow)   # PORT_MASS_FLOW_INLET
BC.port_outlet(disk_mask(cx, cy, r), p=0.0)      # PORT_PRESSURE_OUTLET
```

マスクはセル中心で評価（`cylinderToCell` と同じ規則）。滑らかな `weight` は受け付けない
（刳り抜きは離散的なので設計変数に対して滑らかでない）。既存の `INTERIOR_MASS_SOURCE` /
`INTERIOR_PRESSURE_SINK` は**別の物理**（紙面垂直方向のマニホールド、status-31 の設計感度）
なのでそのまま残した。`trama_case.py` に `--port carve` を追加（`interior` / `wall` は再現性のため不変）。

### outlet が「セル内圧力固定」になる意味

従来の outlet は Robin 条件 `q = C (p − p_out)` で、`max(q_c, 0)` の折れが残差に入っていた。
今回は圧力 Dirichlet（`C → ∞` の極限）なので、レポート §6 が
「内部ポートは出口 sink セルの非線形性が加わる」と書いた折れが消える。

逆流の扱いだけは OF と揃えていない。OF の outlet は `U inletOutlet`（逆流時 `inletValue (0 0 0)`）
だが、nsb は 4 辺の outlet と揃えて**逆流時もゼロ勾配**にした。折れを 1 つ消して別の折れを
入れ直すのは筋が悪いという判断。0.005 kg/s の検算では逆流は起きていない。

## 検算（OpenFOAM `walls` 変種、0.005 kg/s、dx 1.5 mm、壁セル `h_solid=1e-4`）

§12.5 とまったく同じ条件で、ポートの与え方だけ差し替えた。ポート半径も OF に合わせて
w/2 − 2Δx = 14.25 mm（`port_shrink_cells=2` と同じ縮め方）。

| ポートからの除外 | セル数 | 速度 L2（内部ポート） | 速度 L2（**刳り抜き**） | 圧力 L2（内部） | 圧力 L2（**刳り抜き**） |
|---|---|---|---|---|---|
| 3 セル（4.5 mm） | 23992 | 11.93% | **0.98%** | 0.241% | **0.075%** |
| 6 セル | 23832 | 9.21% | **0.97%** | 0.179% | **0.074%** |
| 10 セル | 23628 | 6.52% | **0.95%** | 0.122% | **0.073%** |
| 15 セル | 23384 | 4.21% | **0.94%** | 0.089% | **0.072%** |
| 25 セル（≈ 流路幅 1 つ） | 22910 | 1.76% | **0.90%** | 0.071% | **0.070%** |

**読み方は「小さくなった」ではなく「距離依存が消えた」。** 内部ポート版の単調減衰は
差の源がポートにあることの証拠だった（§12.5）。刳り抜き版は平坦で、ポートがもう誤差源ではない。
残る 0.9% は離散化の地の差（nsb の Newton + Venkatakrishnan SOU 対 OF の SIMPLE + cellLimited SOU）。

噴流の再現（比較領域の最大流速 [m/s]）:

| 除外 | 内部ポート | **刳り抜き** | OpenFOAM |
|---|---|---|---|
| 3 セル | 0.0789 | **0.1491** | 0.1516 |
| 6 セル | 0.0789 | **0.1169** | 0.1191 |
| 10 セル | 0.0789 | **0.0927** | 0.0945 |
| 15 セル 以遠 | 0.0789 | 0.0789 | 0.0801 |

内部ポート版はポート直近でも流路本体の最大流速 0.0789 しか出ておらず**噴流がそもそも無い**。
刳り抜き版は 0.1491 で OF の 0.1516 に **1.6%** まで詰まる。圧力 span も 145.1 vs 145.0 Pa。

## 費用

| 構成 | 段 0.0015 | 段 0.005 | 合計反復 | 合計時間 |
|---|---|---|---|---|
| 内部ポート | 11 | 19 | 30 | 44.6 s |
| **刳り抜きポート** | 10 | 24 | **34** | **49.1 s** |

sink の非線形性が消えるぶん楽になるかと思ったが、0.005 段は 19 → 24 反復に増えた。
機構は噴流そのもので、リング面から法線に吹き出す流れは体積ソースより局所 Re が高く
（最大流速 0.079 → 0.170 m/s、2.2 倍）Newton が歩きにくい。物理を正しく入れた代償。

## テスト

`tests/test_nsb_ports.py`（15 件、新規）。

- 構造: 円板セルが未知数から外れる / リング面の質量流束合計が ṁ/h に機械精度一致 /
  固体行の分離 / numba 経路と numpy 経路の一致 /
  **J1 が色分け FD ヤコビアンと一致**（貫通項支配、面種別を 1 つでも取りこぼすと落ちる）
- 例外: 圧力基準なし / 空マスク / 固体に食い込むポート / 滑らかな weight
- 物理: 質量保存・圧力基準 / リング面から放射状に吹き出す /
  体積ソース版との差がポートからの距離とともに単調に減る /
  **流路内ポートなら下流の発達断面が体積ソース版と 2% 以内で一致**
- 噛み合わせ: 非定常（後退 Euler）が定常解に落ちる / nsbp・随伴が明示的に拒否する

回帰: `test_nsb*` + `test_nsbp*` + `test_brinkman_flow` = **153 passed / 0 failed**。

## 未対応（意図的）

- **nsbp（PETSc）**: 面種別を `nsbp/kernels.py` に通していないので `make_discretization` が
  `NotImplementedError`（黙って違う問題を解かせない）
- **随伴の圧損目的関数**: `source_mean_pressure_objective` は `q_src` 重みなので刳り抜きでは使えない。
  設計感度は `INTERIOR_MASS_SOURCE` + `smooth_disk` のまま
- **0.15 kg/s 非定常の再検算**: §12.6 の場の L2 58% がどこまで落ちるかは未測定（1 走行 2.5 時間）

## ついでに直した

`nsb/utils.inlet_cells` が 4 辺の inlet 面しか見ておらず、内部ポート系では常に空マスク →
`p_inlet_mean` が nan になっていた（status-45 以降の trama の YAML は全部 nan）。
領域内ソースと刳り抜き inlet リング面も見るようにした。

## 再現コマンド

```bash
python experiments/nsb/run_trama_of.py --variant walls --mass 0.005 --out <of>/walls-m0005 --end-time 8000
~/.claude/hooks/memcap -m 24G -- python experiments/nsb/trama_case.py --mass 0.005 --dx 1.5 \
    --port carve --continuation 0.0015,0.005 --freeze 1e-3 --steady-ser --max-iter 120 \
    --linear-solver jfnk --h-solid 1e-4 --tag CARVE-oracle-m0005
for EX in 3 6 10 15 25; do
  python experiments/nsb/trama_of_verify.py --exclude-cells $EX \
    --nsb experiments/nsb/results/trama_CARVE-oracle-m0005_fields.npz --of <of>/walls-m0005
done
```
