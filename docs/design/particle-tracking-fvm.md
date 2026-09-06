# 非構造メッシュの粒子追跡と滞留時間分布（Pollock 型）

[<- README](../../README.md) | [<- docs](../README.md) | [<- 設計文書一覧](README.md) | [<- roadmap](../roadmap.md)

- 実装: [`xkep_cae_fluid/post/tracking.py`](../../xkep_cae_fluid/post/tracking.py)
  （`ParticleTrackFVMProcess`）、[`xkep_cae_fluid/post/rtd.py`](../../xkep_cae_fluid/post/rtd.py)
  （`ResidenceTimeProcess`）、[`xkep_cae_fluid/post/statistics.py`](../../xkep_cae_fluid/post/statistics.py)
- テスト: [`tests/test_particle_tracking_fvm.py`](../../tests/test_particle_tracking_fvm.py)、
  [`tests/test_extruder_inp_export.py`](../../tests/test_extruder_inp_export.py)
- 例: [`examples/extruder_generic_rtd.py`](../../examples/extruder_generic_rtd.py)
- 関連: [inp-generic-extrusion.md](inp-generic-extrusion.md) /
  [single-screw-extruder.md](single-screw-extruder.md) / [fvm-layer.md](fvm-layer.md)

---

## 1. 何のためのものか

押出（および混練一般）の第一の目的は**混練性と滞留時間分布（RTD）**で、流速場そのものでは
ない。E(t) の広がり・累積せん断ひずみ γ = ∫γ̇ dt・混合指数 λ が品質を決める。

構造格子の展開チャネル専用トラッカー
（[`extruder/tracker.py`](../../xkep_cae_fluid/extruder/tracker.py)）は
**節点流れ関数 ψ の双一次補間**で発散ゼロを担保しているが、ψ は 2 次元の構造格子でしか
作れない。汎用記法（`*NODE` / `*ELEMENT`）で書いた .inp の非構造メッシュでは使えないので、
面流束から追跡する Pollock 型を用意する。

## 2. なぜセル中心速度を補間してはいけないか

セル中心速度を線形補間して RK4 で流すと、離散的な発散ゼロが壊れる。壊れた場では

- 渦の中心に粒子が落ち込む（時計回りに巻き込まれて出て来ない）
- 壁際に貼り付く（法線方向の残留速度が消えない）

ので、RTD の裾（最も知りたいところ）が偽物になる。**FVM が実際に満たしているのは
セル中心速度の発散ではなく面流束の総和 Σ_f q_f = 0** なので、その面流束を出発点にする。

## 3. セル内の再構成: 面流束を厳密に再現するアフィン場

セル `c` の内部で速度をアフィン場

```
u(x) = a_c + B_c (x − x_c)
```

とし、**そのセルの全ての面について流束を厳密に再現する**ように係数を決める:

```
(a_c + B_c d_f)·S_f = q_f        d_f = x_f − x_c、S_f = 外向き面ベクトル、q_f = 外向き流束
```

未知数は nd + nd² 個（3 次元で 12）、拘束は面数（六面体で 6）なので不足決定になる。
残る自由度は**最小ノルム解**で閉じる（`np.linalg.pinv`）。ノルムは無次元化した
`B̃ = B·L_c`（`L_c` はセルの代表長さ）に対して取る — a [m/s] と B [1/s] を素のまま並べると
次元が揃わず、セルの大きさで結果が変わってしまう。

この閉じ方には 2 つの良い性質がある。

- **直交六面体では Pollock（1988）の再構成そのものになる。**
  対向する 2 面の拘束が軸ごとに分離して `u_x` が x だけの 1 次関数……となり、
  非対角成分は拘束に現れないので最小ノルムがゼロにする
- **四面体では最低次 Raviart–Thomas（RT0）と一致する。**
  拘束 4 本が「定数ベクトル + 等方な B」を決め、非圧縮（Σq_f = 0）なら B = 0 の
  **一定速度**になる

発散は `∇·u = tr(B_c) = Σ_f q_f / V_c` なので、**離散連続式を満たす面流束を渡す限り
セル内で恒等的にゼロ**になる。セル形状は問わない（六面体・楔・四面体・角錐が混在してよい）。

**Pollock 型の精度の性格**を誤解しないこと。せん断流 `u = S y` を直交格子で再構成すると、
x 方向流束は ±x 面にしか現れないので `u_x` は y に**階段状**（セル内で一定）になる。
y 方向の分解能は 1 次で、構造格子の ψ 双一次補間より粗い。代わりに得られるのが
「どんなセル形状でも局所質量保存が厳密」という性質で、粒子が詰まらない・抜けない。
精度は格子細分化で回復する（§8 の実測）。

## 4. セルからセルへの受け渡し

1 ステップの刻みは**面平面までの残り時間**で決める。アフィン場なので直線近似の到達時刻

```
τ_f = −s_f / (u·n̂_f)          s_f = 面平面までの符号付き距離（内側が負）
```

が良い予測になる。`dt = cfl·min_f τ_f` で RK4 を 1 歩進め、面を跨いだら false position
（4 回）で面上まで戻し、法線方向に厳密に落としてから隣接セルへ渡す。

> **面拘束で刻んだステップは面に「ちょうど」乗る**（丸めで `s = 0`）。
> 不等号 `s > 0` だけで跨いだ判定をすると受け渡しが起きず、以降 `dt = 0` のまま
> 動かなくなる（実装中に踏んだ）。刻みが面拘束で決まったときは「到達」も跨いだ扱いにする。

セル内で場が厳密にアフィンなので、**RK4 は解の 4 次 Taylor 打ち切りそのもの**になり、
刻み幅は精度ではなく面の検出のためだけに要る。淀み点（|u| → 0）では τ が発散するので、
理論平均滞留時間に対する比 `dt_max_fraction`（既定 0.02）で上限を置く。

### 4.1 周期面

周期面（`MeshData.face_offset`）は内部面としてそのまま跨げる。跨ぐときに位置へ並進 T を
掛け、`shift_total ← shift_total − T` を持ち回るので `x + shift_total` が連続な
**巻き戻さない座標**になる。

周期方向のセルが 1 層しか無いとき（押出 2.5D の z 方向）、周期対は
**owner == neighbour の自己面**に併合される。セル → 面のテーブルはその面を符号 ±1 で
2 項目持ち、面平面の位置が並進の分だけ違うので**別の平面**として正しく扱える。

### 4.2 境界面

流束が（総流束比で）`wall_flux_tol` 以下の境界面は**壁**とみなし、跨いだ粒子を領域内へ
押し戻す。壁面の法線流束は再構成が厳密にゼロにするので、これは面の隅で RK4 の打ち切り
誤差が出たときの安全網でしかない。流束のある境界面を跨いだら**流出**として脱出させ、
どのパッチから出たかを記録する。

## 5. 種まきと脱出条件

| `seed` | 種まき | 重み | 脱出 |
|---|---|---|---|
| `"patch"` | 流入する境界面 1 枚につき 1 粒子（面中心） | 流入流束 [m³/s] | 流束のある境界面を跨ぐ |
| `"axial"` | 流体セル 1 個につき 1 粒子（セル中心） | `max(u_c·â, 0)·V_c` | 進行度 ζ = â·(x + shift_total) が `length` に達する |
| `"explicit"` | 呼び出し側が位置・重み・セルを与える | 任意 | 同上（`axis` を与えたとき） |

種まき速度は**追跡に使うのと同じ再構成場**から取る（セル中心では `u = a_c`）。生のセル中心
速度で重みを作ると場が食い違って数 % ずれる。乱数配置ではなく決定論的な求積なので
モンテカルロ誤差はゼロで、残るのは空間求積誤差だけ。

ステップ上限に達した粒子は進行率から外挿して閉じる（`t ← t·length/ζ`）。ただし
**ζ ≈ 0 の粒子は外挿してはいけない** — 淀みや二次渦に捕まった粒子は「定常な周回で ζ が t に
比例する」という前提を満たしておらず、係数が発散して ⟨t⟩ を桁で壊す。
`extrapolation_min_progress`（既定 0.1）未満は未解決として正直に報告する。

## 6. 厳密関係 ⟨t⟩ = length·V / Σw

`seed="axial"` の理論平均滞留時間は

```
⟨t⟩ = length · V_total / Σ_c (u_c·â) V_c
```

導出: ζ=const 面の断面積を `A_ζ`、周期の ζ 長さを `Δζ` とすると `V_total = A_ζ·Δζ`、
ζ 面を通る流束は `Q = Σ_c (u_c·â) V_c / Δζ`。定常・非圧縮なら (x, y, ζ) 空間で
「体積 ÷ 流束」が流束重み付き平均滞留時間に一致するので
`⟨t⟩ = length·A_ζ/Q = length·V_total/Σ_c (u_c·â)V_c`。**`Δζ` を知らなくても書ける**のが要点。

構造格子の展開チャネル版 `⟨t⟩ = z_axial·A_free/(sinφ·Q_axial)` はこの式の 2.5D 特殊形になる
（`V_total = A_free·dz`、`Σ(u·â)V_c = dz·sinφ·Q_axial`）。

`seed="patch"` なら単純に `⟨t⟩ = V_total / Q_in`。

**これが RTD の最も鋭い検査。** 再構成の誤り・種まき重みの誤り・面の受け渡しの取りこぼし・
周期の巻き戻しの記帳ミス・脱出時刻の内挿ミスを**同時に**捕まえる。文献の RTD 曲線との
目視比較よりはるかに強い。

## 7. RTD の集計

`ResidenceTimeProcess` が流束重み付きで E(t)（確率密度）/ F(t)（累積）/ 分位点 /
経路積分スカラーの統計を作る。ビン幅に依存しない重み付き経験分布 `t_ecdf` / `F_ecdf` も
返すので、文献曲線との `max|ΔF|` 比較に使える。

`rate_scalars` に入れた名前は `∫s dt / t`（経路に沿った時間平均）として扱う。混合指数 λ の
ように「平均値」に意味がある量に使い、累積せん断ひずみ γ = ∫γ̇ dt は積算のまま扱う。

`unresolved_weight_fraction`（脱出も外挿もできなかった重み割合）が大きい結果は信用できない。
`extrapolated_weight_fraction` と合わせて必ず見ること。

## 8. 使い方

```python
from xkep_cae_fluid.post.rtd import ResidenceTimeInput, ResidenceTimeProcess
from xkep_cae_fluid.post.tracking import ParticleTrackFVMInput, ParticleTrackFVMProcess

flow = NavierStokesFVMProcess().execute(ns_input)      # 収束済みの非圧縮 NS
track = ParticleTrackFVMProcess().execute(
    ParticleTrackFVMInput(
        mesh=mesh,
        face_flux=flow.mass_flux,      # ρ u·n A（Rhie–Chow。density で体積流束に直す）
        density=ns_input.rho,
        seed="axial",
        axis=(math.cos(phi), 0.0, math.sin(phi)),   # 軸方向 ζ
        length=0.05,                                 # ζ >= 0.05 m で脱出
        scalars={"gamma": flow.strain_rate, "lam": flow.mixing_index},
    )
)
rtd = ResidenceTimeProcess().execute(
    ResidenceTimeInput(track=track, rate_scalars=("lam",))
)
print(rtd.t_mean, rtd.t_mean_theory, rtd.spread, rtd.scalar_mean["gamma"])
```

`NavierStokesFVMResult` の `strain_rate`（γ̇ = sqrt(2 D:D)）と `mixing_index`
（λ = |D|/(|D|+|Ω|)。0: 純回転、0.5: 単純せん断、1: 純伸長）は**粘度モデルの有無に関わらず**
収束後の速度勾配から作る。`.inp` の `*OUTPUT` では `GAMMA` / `LAMBDA` で出せる。

## 9. 検証（実測）

### 9.1 再構成

| 検査 | 結果 |
|---|---|
| 全ての面の流束を再現 | 相対残差 < 1e-12 |
| 一様流 | `a` が厳密、`B = 0`（< 1e-12） |
| せん断 `u = (S y, 0, 0)` | `a_x = S y_c` が厳密、`B = 0`（セル内で y に階段状 = Pollock の正しい振る舞い） |
| 線形発散ゼロ場 `u = (S x, −S y, 0)` | `a`・`B` とも厳密（< 1e-10） |
| 発散 | `tr(B) = Σq_f/V < 1e-10` |

### 9.2 軌跡と滞留時間

| ケース | 結果 |
|---|---|
| 一様流（プラグフロー） | 全粒子が同じ滞留時間（ptp < 1e-12）、`t = L/U` と 1e-5、広がり 1.0 |
| 単純せん断（ゲート G4a 相当） | 軌跡が直線（Δy < 1e-12）、`t = L/(S y)` と 1e-5 |
| 単純せん断の RTD | `F(t) = 1 − (t_min/t)²` と max|ΔF| < 3e-2 |
| 蓋駆動キャビティ（Stokes） | 全粒子が領域内に残る（壁を跨がない）、脱出ゼロ |
| 周期 Poiseuille（ゲート G4b 相当） | **⟨t⟩ = length·V/Σw と 1e-12**（厳密関係）。解析解 V/Q とは ny=12 で 1.4%、ny=24 で 0.35%（2 次収束）。粒子ごとの `t = L/u(y)` は 8e-2 → 2.2e-2 |

### 9.3 押出の展開チャネル（構造格子トラッカーとの照合）

40 mm 機の計量部（D=40 mm, リード 40 mm, H=4 mm, e=4 mm, δ=0.2 mm, N=1 s⁻¹, μ=1000 Pa·s,
G=1e5 Pa/m, z_axial=0.05 m）。参照は専用 2.5D ソルバー + ψ 双一次補間トラッカー
（ゲート G4a/G4b/G5 通過済み）。**流れ場の作り方も追跡の原理も別物**なので、
揃えば両方の実装を同時に検証したことになる。

| 量 | 1184 セル | 4736 セル |
|---|---|---|
| `⟨t⟩` | 5.1e-3 | 3.9e-3 |
| `t_p10` | 8.4e-3 | 2.2e-4 |
| `t_p50` | 6.5e-3 | 3.8e-4 |
| `t_p90` | 1.1e-2 | 9.6e-4 |
| `γ = ∫γ̇dt` | 8.3e-2 | 1.3e-2 |
| 混合指数 `λ` | 5.3e-5 | 4.7e-6 |

γ の差が大きいのは γ̇ の評価が違うため（汎用は最小二乗の速度勾配、参照は構造格子差分）で、
細分化で 1.3e-2 まで縮む。λ はどちらも 0.4996（単純せん断が支配的）。

## 10. 限界と TODO

- **精度は Pollock 型の 1 次**（セル内でせん断方向に階段状）。ψ 双一次補間より粗いので、
  同じ精度を出すには細かい格子が要る。局所質量保存と引き換え
- **計算量**が周期方向のセル数に効く。押出 2.5D は z が 1 セルなので下流方向に進むたびに
  自己面を跨ぐ（例題で 1 粒子あたり数百〜数万ステップ、1920 粒子で 80 s 前後）。
  自己面の複数回横断をまとめる最適化は未実装
- **非定常流には未対応**（定常場のみ）。時間依存の追跡は面流束の時間補間が要る
- **点からセルを探す機能が無い**ので `seed="explicit"` は初期セルも呼び出し側が与える
- `.inp` から RTD を出す**キーワード（`*RTD` 相当）は未実装**。いまは Python API か
  [`examples/extruder_generic_rtd.py`](../../examples/extruder_generic_rtd.py) を使う
- セルが**凸**であることを仮定している（面平面の符号で内外を判定する）
