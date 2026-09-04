# ゲート G3 — OpenFOAM による独立検算

[<- docs](../../README.md) | [<- 設計文書](../../design/single-screw-extruder.md) | [<- 図解レポート](README.md)

**実行**: 2026-09-04 / branch `claude/single-screw-extruder-impl` @ `cbe0d2b` / image `opencfd/openfoam-run:2312`
**コマンド**: `OMP_NUM_THREADS=2 experiments/extruder/run_g3.sh /tmp/of-g3`（実行時の作業ディレクトリ: `of-g3`）
**判定**: G3a 合格 / G3b 合格 / 1D 較正 合格（全て比 < 1.00 が合格）

## 1. 何を検算しているか

ykep-cae の `ExtruderFlowProcess`（自作の MAC Stokes + FV Poisson）が出す断面解を、
**別の離散化・別の解法**である OpenFOAM `simpleFoam` で再現できるかを見る。
解析解ゲート G1/G2 は形状係数（積分量）しか見ていないので、断面の**分布**と
**隙間を越える漏れ流れ**は G3 で初めて第三者に当たる。


<figure>
<svg viewBox="0 0 760 300" role="img" aria-label="同じ格子を 2 つのソルバーに食わせて、流量・漏れ流量・断面分布を突き合わせる" style="max-width:100%;height:auto;font-family:sans-serif;font-size:12px">
  <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0,0 8,4 0,8" fill="currentColor"/></marker></defs>
  <rect x="20" y="110" width="170" height="80" rx="6" fill="none" stroke="currentColor"/>
  <text x="105" y="140" text-anchor="middle" font-weight="bold">ScrewGeometryProcess</text>
  <text x="105" y="158" text-anchor="middle">dx / dy 幅配列（等比区間）</text>
  <text x="105" y="176" text-anchor="middle">フライト = 固体マスク</text>

  <line x1="190" y1="135" x2="300" y2="60" stroke="currentColor" marker-end="url(#ah)"/>
  <text x="235" y="88" text-anchor="middle">同じ配列</text>
  <line x1="190" y1="165" x2="300" y2="240" stroke="currentColor" marker-end="url(#ah)"/>
  <text x="235" y="215" text-anchor="middle">grading + topoSet</text>

  <rect x="300" y="20" width="200" height="80" rx="6" fill="none" stroke="#B24714" stroke-width="1.5"/>
  <text x="400" y="45" text-anchor="middle" font-weight="bold">ykep-cae</text>
  <text x="400" y="63" text-anchor="middle">w: FV Poisson（直接解）</text>
  <text x="400" y="81" text-anchor="middle">u,v,p: MAC Stokes（splu）</text>

  <rect x="300" y="200" width="200" height="80" rx="6" fill="none" stroke="currentColor"/>
  <text x="400" y="225" text-anchor="middle" font-weight="bold">OpenFOAM simpleFoam</text>
  <text x="400" y="243" text-anchor="middle">z 1 セル cyclic、x cyclic</text>
  <text x="400" y="261" text-anchor="middle">体積力 (−β, 0, −G)、SIMPLEC α=0.999</text>

  <line x1="500" y1="60" x2="590" y2="130" stroke="currentColor" marker-end="url(#ah)"/>
  <line x1="500" y1="240" x2="590" y2="170" stroke="currentColor" marker-end="url(#ah)"/>

  <rect x="590" y="110" width="150" height="80" rx="6" fill="none" stroke="currentColor"/>
  <text x="665" y="135" text-anchor="middle" font-weight="bold">突き合わせ</text>
  <text x="665" y="153" text-anchor="middle">Q, Q_leak, w(y), u(y)</text>
  <text x="665" y="171" text-anchor="middle">閾値 1% → 比 &lt; 1.00</text>
</svg>
<figcaption>図 1: 同じ幅配列から両方の格子を作り、セル中心座標の一致を検査してから比較する。格子差を切り分けから消すのが G3 の設計。</figcaption>
</figure>


| 項目 | 設定 |
|---|---|
| 諸元 | 40 mm 機（設計文書 §6）: D=40, lead=40, H=4, e=4, δ=0.1 mm, N=100 rpm |
| 背圧勾配 | G = 5.0e+06 Pa/m（β = G·cotφ = 1.571e+07 Pa/m） |
| 格子 | 248 × 80、流体セル 16960、セル中心ずれ 1.4e-09 セル幅 |
| G3a | ニュートン μ = 1000 Pa·s |
| G3b | べき乗則 K = 2×10⁴, n = 0.4（γ̇_min = 0.01 s⁻¹ クランプ） |
| OpenFOAM | ρ = 1 の運動学的単位、x/z cyclic、体積力 `vectorSemiImplicitSource`、SIMPLEC（U 0.999 / p 1.0） |

## 2. 結果

### G3a ニュートン（OpenFOAM 15519 反復、residualControl 到達: True）

| 量 | 比（閾値 1%） | 相対差 | 生値 |
|---|---|---|---|
| Q = ∫∫w dA | **5.3e-10** ✅ | 5.31e-12 | ykep 1.194917e-05 / OF 1.194917e-05 |
| Q_leak（x=0 面流束） | **0.399** ✅ | 3.99e-03 | ykep -3.241085e-06 / OF -3.254004e-06 |
| w(y) @ x≈0 | **3.1e-10** ✅ | 3.12e-12 | —  |
| u(y) @ x≈0 | **0.006** ✅ | 5.59e-05 | —  |
| Q_axial = Q + L_turn·Q_leak（参考） | **0.013** ✅ | 1.34e-04 | ykep 1.156107e-05 / OF 1.155952e-05 |

全域 L2 相対誤差（参考）: u 2.68e-03 / v 1.12e-02 / w 6.59e-10

### G3b べき乗則（OpenFOAM 15469 反復、residualControl 到達: True）

| 量 | 比（閾値 1%） | 相対差 | 生値 |
|---|---|---|---|
| Q = ∫∫w dA | **0.148** ✅ | 1.48e-03 | ykep 1.054281e-05 / OF 1.055839e-05 |
| Q_leak（x=0 面流束） | **0.519** ✅ | 5.19e-03 | ykep -3.466509e-06 / OF -3.484512e-06 |
| w(y) @ x≈0 | **0.046** ✅ | 4.57e-04 | —  |
| u(y) @ x≈0 | **0.088** ✅ | 8.80e-04 | —  |
| Q_axial = Q + L_turn·Q_leak（参考） | **0.133** ✅ | 1.33e-03 | ykep 1.012772e-05 / OF 1.014114e-05 |

全域 L2 相対誤差（参考）: u 3.85e-03 / v 1.80e-02 / w 2.46e-03
ykep 側 Picard 反復 43 回、25.1 s。

### 1D 較正（G3b の前提）: OpenFOAM powerLaw の (k, n) ↔ ykep の (K, n)

平行平板 Poiseuille（H = 4 mm, K = 2×10⁴, n = 0.4, G = 5×10⁷ Pa/m, ny = 100）を
γ̇_min クランプ込みの厳密解と比較。

| 量 | 比（閾値 0.5%） | 相対差 |
|---|---|---|
| 中心速度 u_max | **0.025** ✅ | 1.26e-04 |
| 流量 q | **0.040** ✅ | 2.01e-04 |
| 分布 L2 | **0.048** ✅ | 2.42e-04 |

**確定した対応づけ**: ρ = 1 で `k = K`, `n = n`。OpenFOAM の `strainRate() = √2·|symm(∇U)|` は
ykep の `γ̇ = √(2 D:D)` と同じ定義。`nuMax = K·γ̇_min^(n−1)` にすればクランプの掛かり方まで一致する。

## 3. なぜそうなるか（メカニズム）

### 3.1 w は 10⁻¹² まで一致する — 同じ離散方程式だから

下流方向 w は ∂/∂z = 0 のとき圧力と切り離された Poisson 方程式
∇·(μ∇w) = −G になる。両者とも同じ格子で、面の粘度を調和平均、壁を半セル距離の
Dirichlet で離散化するので、**代数方程式が同一**になり、差は線形ソルバーの丸めだけになる。
これは「一致した」というより「同じものを 2 回解いた」に近い。だから G3 の実質的な
検算対象は **u, v, p の Stokes 連成と漏れ流れ**の方にある。

べき乗則（G3b）では w の一致が 2.5e-03 に落ちる。方程式は同じでも
**係数 μ(γ̇) の作り方が違う**ためで、ykep は Green-Gauss（面値の差 ÷ セル幅、境界セルで
壁値をそのまま面に置く）、OpenFOAM は `fvc::grad` の Gauss linear（境界セルでは片側）で
γ̇ を評価する。差が出るのはせん断が最大のバレル直下の 1 セル層で、n = 0.4 だと
γ̇ の 1% 差が μ の 0.6% 差になる。この層は隙間の入口でもあるので Q_leak が最も敏感
（5.2e-03）。それでも閾値の半分に収まる。

### 3.2 u, v, Q_leak の差は離散化の違いから来る

ykep は MAC スタガード格子（速度は面、圧力はセル中心）、OpenFOAM はコロケート格子 +
Rhie–Chow 補間。同じ格子でも**圧力–速度の結合の仕方が違う**ので、u, v は解像度分の
差を持つ。隙間（δ = 0.1 mm に 20 セル）を通る漏れ流量 Q_leak が最も敏感で、
G3a で 3.99e-03、G3b で 5.19e-03。
どちらも閾値 1% の内側で、しかも**両者とも格子を細かくすれば同じ極限に向かう**ことが
ykep 側の 2 次収束テスト（`test_extruder_solver.py`）と合わせて言える。

### 3.3 simpleFoam は緩和係数 0.999 でないと収束しない

緩和係数 α は擬似時間刻み Δτ = α/(1−α)·V/a_P を与える。クリープ流れでは
a_P ≈ 2ν/Δy² が支配し、最も滑らかな誤差モード（固有値 ν(π/H)²）の 1 反復あたりの減衰率は

    g = 1 / (1 + α/(1−α) · (πΔy/H)²/2)

隙間セル Δy = 5 μm では (πΔy/H)²/2 ≈ 8×10⁻⁶。α = 0.9 なら g ≈ 1 − 7×10⁻⁵ で
数十万反復、α = 0.999 でも g ≈ 1 − 8×10⁻³ で数千反復かかる。
1D で実測: ny=100 で α=0.9 は 5000 反復後も残差 10⁻³、α=0.999 は 20 反復で 10⁻¹¹。
α = 1 は SIMPLEC の 1/(1/rAU − H1) が純拡散でゼロになり sigFpe で落ちる（実測）ので
0.999 が上限。**a02 の教訓「完全発達に simpleFoam は遅い」の定量版**がこれ。

### 3.4 べき乗則はニュートン解から始める

U = 0 から始めると γ̇ = 0 → ν = nuMax（3×10⁵）で a_P が 150 倍になり、
擬似時間刻みがさらに潰れる。粘度が下がるには速度が育つ必要があり、速度が育つには
粘度が下がる必要がある、という**鶏と卵**で 3.3 の反復数がもう 1 桁増える。
G3b は G3a の収束解を初期値にして、この立ち上がりを飛ばしている。
ykep 側の Picard（ω = 0.5）も同じ不動点反復だが、線形解が毎回厳密なので
擬似時間の制約が無く、43 回で収束する。

## 4. 再現

```bash
cd ~/work/ykep-cae
OMP_NUM_THREADS=2 experiments/extruder/run_g3.sh /tmp/of-g3
```

- `experiments/extruder/of_case.py` — ケース生成（blockMesh 多区間 grading で ykep の dx/dy を再現）
- `experiments/extruder/of_powerlaw_check.py` — 1D 較正
- `experiments/extruder/compare_openfoam.py` — 突き合わせ（格子一致検査つき）
- `experiments/extruder/g3_report.py` — この文書の生成
