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

## 付録: N 判定が uturn でも同じ位置に出る（gyp さんの追加質問）

「0.7 × 0.4、流路幅 0.1、u_in = 1、288×182、ρ=1000、μ=1e-3、**h_channel = 4e-3** の uturn は何反復で収束するか」。
まず判定式（status-47）に入れると `L_drag = ρuh²/(12μ) = 1.333 m`、`N = L_drag/w = 13.3`、
`Re_h = 8000`、`Re_w = 1e5`。**trama の 0.15 kg/s と同じ N** で、境目（N ≈ 2）の 8 倍向こう。

実測（SOU + Venkatakrishnan、上限 80）:

| 構成 | 結果 | 定常残差比 | 時間 |
|---|---|---|---|
| `jfnk_simple`（既定） | **発散** | 27.7（`rel_min` 1.00 = 一度も参照を下回らない） | 221 s |
| `jfnk` | 未収束 | 5.4e-2 で平坦 | 80 s |
| `jfnk` + 壁セル `h_solid=1e-4` | 未収束 | 2.0e-1 | 19.8 s |

継続法（U を上げる、壁セル + リミター凍結 1e-3 + 定常残差 SER、上限 200）:

| U | **N** | 結果 | 反復 | 時間 |
|---|---|---|---|---|
| 0.125 | **1.67** | **収束** | **161** | 103 s |
| 0.25 | **3.33** | 停滞（rel_steady 1.4e-1） | 200 未達 | 42 s |

ny = 192 でも同じ（161 → 163 反復、停滞位置も同じ）ので、ny の端数（流路幅 0.1 が 45.5 セル）は原因ではない。

**境目は N = 1.67 収束 / 3.33 停滞の間**で、status-47 が OpenFOAM の蛇行流路で測った
**N 1.33 収束 / 2.22 停滞** と同じ位置に出る。**別の流路形状・別のソルバーで境目が一致**したので、
N = L_drag/w は形状によらない判定と見てよい。

### FOU で収束するのは数値拡散が粘度を 1200 倍にするから

gyp さんの手元では FOU に切り替えると 100 反復程度で定常解に落ちるとのこと。1 次風上の切り捨て誤差は
主流方向の拡散そのもので `μ_num ≈ ½ρ|u|Δx`。このケースでは Δx = 2.43 mm なので

| 量 | 値 |
|---|---|
| セル Reynolds 数 `Re_Δ = ρuΔx/μ` | 2431 |
| `μ_num = ½ρuΔx` | **1.22 Pa·s** |
| `μ_phys` | 1.0e-3 Pa·s |
| **比** | **≈ 1200 倍** |
| 実効 `Re_w = ρuw/μ_eff` | 1e5 → **82** |

渦放出に要る目安は Re_w ≳ 2000（status-47）なので、FOU では**渦がそもそも立たない**。
status-47 で「乱流モデルで定常化させるのに必要」と測った渦粘性 ν_t = 4.2e-4 m²/s（隙間乱流の目安の
22 倍なので棄却した）に対し、FOU が供給しているのは ν_num = 1.2e-3 で**その 3 倍**。
**乱流モデルとして棄却した安定化を、FOU は名前を変えて 3 倍多く入れている。**

切り分けは格子細分で決まる（μ_num ∝ Δx）。物理の解なら Δx → 0 で残り、数値拡散の産物なら
細かくするほど実効 Re_w が上がって収束しなくなる（Re_w,eff = 2000 を跨ぐのは Δx ≈ 1e-4 m ＝ nx ≈ 7000 相当なので、
288 → 576 → 1152 で u_max と Δp が単調に動き続けるかを見れば足りる）。
商用ソルバーとの突き合わせでも、**両者のスキーム次数と実効セル Re を揃えない限り「同じ定常解が出た」は物理の裏付けにならない**。

### 再現

```bash
python - <<'PY'
from nsb.core import NSBInput, NSBSettings
from nsb.geo import LX, LY, make_uturn_h, uturn_bc_preset
from nsb.solver import solve_steady
NX, NY, U = 288, 182, 1.0
h = make_uturn_h(NX, NY, h_channel=4e-3, h_blocked=1e-5, width=0.1)
inp = NSBInput(nx=NX, ny=NY, lx=LX, ly=LY, h=h, bc=uturn_bc_preset(NY, u_in=U),
               h_solid=1e-4, rho=1000.0, mu=1e-3, mu_b=1e-3,
               settings=NSBSettings(linear_solver="jfnk", newton_max_iter=80))
r = solve_steady(inp, log=None)
print(r.converged, r.n_iter, r.rel_steady_residual)
PY
```
