# 汎用記法（.inp）で押出級の流れを書く

[<- README](../../README.md) | [<- docs](../README.md) | [<- 設計文書索引](README.md) | [<- roadmap](../roadmap.md)

**状態**: 実装済み（Phase 12、status-36）
**目的**: 単軸押出の展開チャネル 2.5D を、押出専用のキーワードではなく
**汎用記法**（`*NODE` / `*ELEMENT` + `*NAVIER STOKES`）で書けるようにする。
形状・境界条件を .inp の編集だけで変えられるようにするのが狙い。

関連: [inp-format.md](inp-format.md)（.inp 全体の仕様）、
[navier-stokes-fvm.md](navier-stokes-fvm.md)（非構造 NS ソルバー）、
[fvm-layer.md](fvm-layer.md)（面ベース FVM 低レイヤー）、
[single-screw-extruder.md](single-screw-extruder.md)（専用 2.5D ソルバーと検証ゲート）

---

## 1. なぜ専用キーワードにしなかったのか

`*SCREW` / `*EXTRUDER` のような専用手続きを足せば、諸元（D, リード, H, e, δ, N）から
メッシュも境界条件も自動で決まるので **書く量は最小**になる。しかしそれは
「設計変数を諸元の 6 個に固定する」ことでもある。溝付きバレル・途切れたフライト・
非対称断面・混練エレメントの前段などは、諸元では表せない。

汎用記法で書ければ、同じソルバーとフォーマットのまま

- 断面形状を任意（`*NODE` / `*ELEMENT` を差し替えるだけ）
- フライトの本数・幅・隙間を要素単位で
- 3D 螺旋 1 ピッチへの拡張（並進周期を軸方向 1 リードに変えるだけ）

にできる。専用記法は、汎用記法が動いた後に「諸元 → `*NODE`/`*ELEMENT`/`*BOUNDARY`
を生成するプリプロセス」として被せれば両立する。実際その位置に置いたのが
[`ExtruderChannelInpProcess`](../../xkep_cae_fluid/extruder/inp_export.py) で、
出力は**ただの汎用記法 .inp** なので、生成後に手で編集できる。

## 2. 足りなかった 5 つと、その埋め方

| # | 欠けていたもの | 追加したもの |
|---|---|---|
| 1 | **周期境界**（1 ピッチで閉じる。漏れ流れの駆動） | `*BOUNDARY, TYPE=PERIODIC` + `MeshData.face_offset` |
| 2 | **圧力跳び** `Δp = G·L_turn` | `P = βx + p̃` 分解 → `*DLOAD` の一様体積力 |
| 3 | **非ニュートン粘度** μ(γ̇) | `*VISCOSITY, TYPE=POWER LAW / CARREAU` + 非構造 γ̇ の Picard |
| 4 | **回転する壁**（3D 円筒バレル） | `*ORIENTATION` + `*MPC` + 参照節点の自由度 4-6 |
| 5 | **慣性項を落とす**（クリープ流れ） | `CONVECTION=NONE`（Stokes）と `PRESSURE_VELOCITY=COUPLED` |

### 2.1 周期境界（`*BOUNDARY, TYPE=PERIODIC`）

```
*BOUNDARY, TYPE=PERIODIC
 master_surface, slave_surface[, tx, ty, tz]
```

対になる 2 つの `*SURFACE`（または予約面名）の面中心を並進 `t` で照合し、
**master 面を内部面に昇格**させて slave 面を消す。並進を省くと両面の面中心の平均差から
自動で決める。照合できないとき（分割が違う、並進が違う、法線が反平行でない）は
ずれの最大値を添えてエラーにする。

`MeshData` 側は内部面ごとの並進ベクトル `face_offset` を 1 本持つだけ:

```
neighbour_centers(mesh) = cell_centers[neighbour] + face_offset
```

fvm 層で P–N ベクトルを使うところ（補間重み・スキュー・over-relaxed 分解・
最小二乗勾配・TVD の上流距離・Rhie–Chow）はすべてこの関数を通すので、
**周期面は普通の内部面として扱われる**。拡散も対流も圧力補正も分岐が要らない。

対応するのは**並進周期**だけ（回転周期・螺旋周期は未対応）。3D 螺旋 1 ピッチは
軸方向 1 リードの並進なので、これで書ける。

**2.5D の要点**: z 方向 1 セルの両端を周期にすると `∂/∂z = 0` が厳密になり、
**w が自由になる**。対称面（`TYPE=SYMMETRY`）にすると z 方向にも壁ができ、
下流方向速度が厚さ `lz` に依存する偽の解になる。これが
「2.5D を汎用記法で書く」の正体で、テスト
`TestPeriodicAndBodyForcePhysics::test_z_periodic_gives_exact_2p5d_third_component`
が両者を対比している。

### 2.2 圧力跳びを体積力に落とす（`*DLOAD`）

チャネル 1 ピッチの周期は「隣のチャネル = 同じチャネルの 1 周後」なので、
圧力が周期ではなく `P(x + W_t) = P(x) + Δp` になる（[設計 §2.1](single-screw-extruder.md)）。
これを `P = βx + p̃`（`p̃` は周期）と分解すると、`p̃` は普通の周期場になり、
`β` は**一様体積力**として運動量に入る。

```
*DLOAD
 CHANNEL, BF, fx, fy, fz     ** ベクトル [N/m³]
 CHANNEL, BX, fx             ** 1 成分だけ（BY / BZ も同じ）
```

展開チャネルでは `f = (−G·cotφ, 0, −G)`（横断方向が圧力跳びの分、下流方向が `dp/dz`）。
`GRAV` は従来どおり Boussinesq 浮力に写るので、体積力とは別の経路。

### 2.3 非ニュートン粘度（`*VISCOSITY, TYPE=`）

```
*VISCOSITY, TYPE=POWER LAW
 K, n[, gamma_min, mu_max]        ** μ = K γ̇^(n−1)
*VISCOSITY, TYPE=CARREAU
 mu_0, mu_inf, lambda, n
```

粘度モデルは `xkep_cae_fluid.fvm.viscosity` の Strategy（押出専用モジュールから
fvm 層に移して再輸出）。非構造メッシュのせん断速度は**最小二乗の速度勾配テンソル**から

```
γ̇ = sqrt(2 D:D),   D = ½(∇u + ∇uᵀ)
```

で作る（構造格子専用の `extruder/viscosity.py::strain_rate` の非構造版）。
外部反復ごとに μ を更新する Picard 結合で、緩和は `*CONTROLS, PARAMETERS=RELAXATION`
の `VISCOSITY=`（既定 0.5）。変粘度では拡散項 `∇·(μ∇u)` に入らない
`∇·(μ∇uᵀ)` の余剰項 `Σ_j ∂_i u_j ∂_j μ` を陽的ソースに足している。

`gamma_min` / `mu_max` は **数値上の安全弁であって物理ではない**ので、
結果がこれらに依存しないことをテストで確認している
（`test_gamma_min_clamp_does_not_change_the_answer`）。

### 2.4 回転する壁（`*ORIENTATION` + `*MPC` + 自由度 4-6）

3D で円筒バレルをそのまま扱うときは、面ごとに `Ω × r` を与える必要がある。
Abaqus 流に **参照節点を回す**書き方にした。

```
*NSET, NSET=REF
 900001                                   ** 軸上の節点
*ORIENTATION, NAME=SPIN, SYSTEM=CYLINDRICAL
 0., 0., 0.,  0., 0., 1.                  ** 軸上の 2 点（局所 3 軸が軸方向）
*MPC
 BEAM, BARREL, REF                        ** 面 BARREL を参照節点の剛体運動に拘束
...
*BOUNDARY, ORIENTATION=SPIN
 REF, 6, 6, 12.566                        ** 自由度 6 = 局所 3 軸まわりの角速度 [rad/s]
 REF, 1, 3, 0.0                           ** 自由度 1-3 = 並進速度 [m/s]（任意）
```

面上の速度は `u(x) = v_ref + ω_ref × (x − x_ref)`。回転中心は**参照節点の座標**なので、
軸上に置く。`*ORIENTATION` は速度・角速度の**成分の解釈**だけを変える
（`CYLINDRICAL` は軸上の 2 点、`RECTANGULAR` は Abaqus と同じ局所 1 軸・1–2 平面の点）。

`*MPC` の種別は `BEAM` / `RIGID` / `TIE`（すべて同じ剛体リンク）。
回転面が回転軸まわりの回転面になっていないと壁が「吹く」ので、法線速度が接線速度の
1e-6 倍を超えたら警告する。

展開チャネル 2.5D では並進周期と定数速度で足りるので、この経路は使わない。

### 2.5 Stokes モードと速度–圧力連成

樹脂の `Re = ρVH/μ ~ 10⁻³` なのでクリープ流れ。

```
*CONTROLS, PARAMETERS=DISCRETIZATION
 CONVECTION=NONE, PRESSURE_VELOCITY=COUPLED
```

- `CONVECTION=NONE`: 運動量の対流項を落とす（エネルギー・追加スカラーは風上のまま輸送）。
  密度を変えても解が動かないことをテストで確認している
- `PRESSURE_VELOCITY=COUPLED`: 速度 nd 成分と圧力を **1 つの線形系**にして直接解く。
  圧力勾配を最小二乗勾配の作用素で、連続式を Rhie–Chow 流束で、どちらも陰的に書く。
  Stokes なら**外部反復 2 回**（1 回目で厳密解、2 回目で残差確認）で終わる。
  SIMPLE 系と同じ解に落ちることを、Stokes キャビティと Re=100 キャビティで確認している

COUPLED は鞍点系なので直接法で解く（大規模では SIMPLE 系のほうが省メモリ）。
`OUTFLOW`（対流流出）境界は未対応。

## 3. 展開チャネル 2.5D を汎用記法で書く

```
*NODE / *ELEMENT             ** 断面 (x,y) を z 方向 1 セルに押し出した C3D8。フライトは要素ごと抜く
*SURFACE, NAME=XPER0 / XPER1 ** x = 0 / x = W_t
*SURFACE, NAME=ZPER0 / ZPER1 ** z = 0 / z = dz
*SURFACE, NAME=BARREL        ** y = H
*BOUNDARY, TYPE=PERIODIC
 XPER0, XPER1, W_t, 0., 0.   ** 1 ピッチ
*BOUNDARY, TYPE=PERIODIC
 ZPER0, ZPER1, 0., 0., dz    ** ∂/∂z = 0（w を自由にする）
*MATERIAL / *VISCOSITY / *FLUID SECTION
*STEP
*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=NONE
*CONTROLS, PARAMETERS=DISCRETIZATION
 CONVECTION=NONE, PRESSURE_VELOCITY=COUPLED
*BOUNDARY, TYPE=VELOCITY
 BARREL, -V sinφ, 0., +V cosφ
*DLOAD
 CHANNEL, BF, -G cotφ, 0., -G
*END STEP
```

スクリュー根元・フライト側壁・フライト頂部は**書かない**（既定が no-slip 壁）。
例題は [examples/inp/extruder-channel-1.inp](../../examples/inp/extruder-channel-1.inp)
（`ExtruderChannelInpProcess` が生成、2016 セル、COUPLED で 2 反復・0.18 s）。

## 4. 検証: 専用ソルバーと解析解が汎用経路のリファレンス

専用の 2.5D ソルバー（`ExtruderFlowProcess`）はゲート G1〜G5 を通しているので、
そのまま汎用経路の参照解になる。40 mm 機の計量部（D=40 mm, リード 40 mm, H=4 mm,
e=4 mm, δ=0.2 mm, N=1 s⁻¹, μ=1000 Pa·s, G=1e5 Pa/m）で照合した結果:

| 量 | 汎用 vs 参照 | 備考 |
|---|---|---|
| `Q`（下流方向 ∫∫w dA、閉チャネル） | 解析解と 1.7e-3（40×20）→ 7.3e-4（60×24） | 形状係数 `F_d` / `F_p`（G1 / G2） |
| `Q`（隙間あり） | 専用ソルバーと **1e-15** | 同じ可変係数 Poisson になるので機械精度 |
| `Q_leak`（漏れ） | 3.9e-2（40×20）→ 2.2e-2（60×24）→ 7.2e-3（120×48） | MAC 千鳥格子 vs Rhie–Chow 同位置格子の離散化差。細かくすると縮む |
| `Q_axial = Q + L_turn·Q_leak` | 2.5e-3 → 1.4e-3 → 4.7e-4 | 実際の押出量 |

`Q_leak` だけ差が大きいのは、断面内 Stokes の離散化が違う（専用は MAC 千鳥格子、
汎用は同位置格子 + Rhie–Chow）ため。格子細分化で縮むことをテスト
`test_clearance_agreement_improves_with_refinement` で固定している。

テストは [tests/test_extruder_inp_export.py](../../tests/test_extruder_inp_export.py)。

## 5. 制限と次

- 周期は**並進のみ**（回転周期・螺旋周期は未対応。3D 螺旋 1 ピッチは軸方向並進で書ける）
- `COUPLED` は直接法固定・`OUTFLOW` 非対応
- 非構造メッシュの**粒子追跡 / RTD** は
  [particle-tracking-fvm.md](particle-tracking-fvm.md)（`ParticleTrackFVMProcess` /
  `ResidenceTimeProcess`、面流束から再構成した Pollock 型）で対応済み。
  ただし `.inp` のキーワードからは呼べず、Python API か
  [`examples/extruder_generic_rtd.py`](../../examples/extruder_generic_rtd.py) を使う
- 粘性発熱 `Φ = μγ̇²` と温度依存粘度は未対応（専用ソルバー側も Phase 2）
- `*MPC` は面 → 参照節点の剛体拘束だけ（節点対節点の一般 MPC ではない）
