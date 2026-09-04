# nsb 数理ノート: Brinkman 補正 NS の FVM 離散化と Newton–擬似時間反復（総和規約）

[<- README](../README.md) | [<- nsb/README](README.md) | [設計文書](../docs/design/brinkman-flow-fvm.md) | [status-28](../docs/status/status-28.md)

本ノートは status-28 の再現実験で使った数理を、実装（`xkep_cae_fluid/brinkman_flow/assembly.py`、
`nsb/solver.py`）と 1 対 1 に対応する形で総和規約で書き出したものである。
「なぜ落ちるか」の機構解析（§9）が主眼で、そこに至る定義（§1–§8）を先に固定する。

## 0. 記法

- 空間添字 $i, j, k \in \{1, 2\}$（$x_1 = x$, $x_2 = y$）。同じ添字が 2 回現れたら和をとる（総和規約）。$\partial_i \equiv \partial/\partial x_i$。
- セル添字 $P$（着目セル）、$N$（面 $f$ を挟む隣接セル）、$U$（面 $f$ の風上セル）。$\sum_f$ はセル $P$ の全面（東西南北）の和で、こちらは明示的に書く。
- 面 $f$ の外向き単位法線 $n^f_i$、面積 $A_f$（$x$ 面は $\Delta y$、$y$ 面は $\Delta x$）、セル体積 $V = \Delta x \Delta y$（深さ平均量なので単位深さあたり）。
- 面 $f$ の中心とセル中心の相対位置 $r^f_i$（$x$ 面なら $r^f_1 = \pm \Delta x/2$）。
- Newton 反復添字 $k$ は上付き $x^{(k)}$。離散未知数ベクトル $x = [u_1, u_2, p]^\top \in \mathbb{R}^{3n}$、$n = n_x n_y$。
- $\|\cdot\|$ はユークリッドノルム。$\langle \cdot \rangle_f$ は面への線形補間 $\tfrac12(\phi_P + \phi_N)$。

## 1. 支配方程式（深さ平均 Brinkman 補正 Navier–Stokes）

厚さ $h(x_1, x_2)$ の薄い流路を深さ方向に平均した 2 次元場で、深さ方向の粘性抵抗を Brinkman 項として持つ定常非圧縮流れ:

$$
\rho\, \partial_j (u_j u_i) = -\partial_i p + \mu\, \partial_j \partial_j u_i - \kappa\, u_i,
\qquad \partial_i u_i = 0,
\qquad \kappa \equiv \frac{12\,\mu_b}{h^2}.
$$

- 係数 12 は平行平板 Poiseuille 分布 $u(z) = 6 \bar u\, z(h-z)/h^2$ の壁せん断を深さ平均した Hele-Shaw の値。
- $h = 10^{-3}$ で $\kappa = 1.2\times10^4$、$h = 10^{-5}$ で $\kappa = 1.2\times10^8$。後者は実質的な壁（U ターンモデルの閉塞部）。
- 連続式は $\partial_i u_i = 0$ のまま（$h$ を連続式に入れない近似。設計文書参照）。
- 対流を落とした $-\partial_i p + \mu \partial_j\partial_j u_i - \kappa u_i = 0$ が **Stokes–Brinkman**（Darcy–Brinkman）方程式で、§8 の初期場に使う。

無次元数: inlet 高さ $H = 0.1$ 基準の $\mathrm{Re} = \rho U H/\mu = 10^4 U$（$U = 0.1, 1, 2$ で $10^3$〜$2\times10^5$）。
Darcy 数 $\mathrm{Da} = \mu/(\kappa H^2) = 8\times10^{-6}$ で、圧力損失は Hele-Shaw の $\Delta p = \kappa U L$ が支配的。

## 2. 有限体積離散化（同位置格子）

セル $P$ で式 (1) を体積積分し、Gauss の定理で面積分にする。面 $f$ の質量流束を

$$
F_f \equiv \rho\, u^f_j n^f_j A_f
$$

とすると、運動量成分 $i$ と連続式の離散残差は

$$
R^{u_i}_P = \sum_f F_f\, \tilde u^f_i
\;-\; \sum_f \mu\, (\partial_j u_i)^f n^f_j A_f
\;+\; \sum_f p^f n^f_i A_f
\;+\; \kappa_P V u_{i,P},
\qquad
R^p_P = \sum_f F_f .
$$

$\tilde u^f_i$ は**対流面値**（§2.3、スキーム依存）、$u^f_j$（$F_f$ の中の速度）は **Rhie–Chow 面速度**（§2.4）で、両者は別物である。
未知数 $x$ に対して $R(x) = [R^{u_1}, R^{u_2}, R^p]^\top$ を組む。

### 2.1 境界条件

領域 $[0, L_x]\times[0, L_y]$、境界はすべて左壁（$x_1 = 0$）にある。

| 境界 | 速度 | 圧力 |
|---|---|---|
| 速度 inlet（左壁 $y \in [0.25, 0.35]$） | $u^f_1 = U_\mathrm{in}$, $u^f_2 = 0$（Dirichlet） | $p^f = p_P$（ゼロ勾配） |
| 圧力 outlet（左壁 $y \in [0.05, 0.15]$） | $u^f_i = u_{i,P}$（ゼロ勾配） | $p^f = 0$ |
| 壁（その他） | $u^f_i = 0$（no-slip） | $p^f = p_P$ |

### 2.2 拡散項

内部面: $(\partial_j \phi)^f n^f_j = (\phi_N - \phi_P)/d_{PN}$（$d_{PN} = \Delta x$ または $\Delta y$）。
Dirichlet 境界面: $(\phi^f - \phi_P)/(d_{PN}/2)$（壁・inlet）。outlet はゼロ勾配なので 0。
拡散の対角寄与（§2.4 の $a_P$ に使う）は $\sum_f \mu A_f / d_f$ で、Dirichlet 面は $2\mu A_f/d_{PN}$。

### 2.3 対流面値 $\tilde u^f_i$

1 次風上（FOU）: $\tilde\phi^f = \phi_U$、$U$ は $F_f \ge 0$ なら $P$、そうでなければ $N$。

2 次風上（SOU）: 風上セルからの線形外挿にリミターを掛ける。

$$
\tilde\phi^f = \phi_U + \psi_U\, (\partial_j \phi)_U\, r^{f}_j ,
\qquad
(\partial_j \phi)_P = \frac{1}{V} \sum_f \langle\phi\rangle_f\, n^f_j A_f
\quad(\text{Green–Gauss、線形面値}).
$$

Venkatakrishnan リミター $\psi_P \in [0, 1]$: 各面への外挿量 $\Delta_f \equiv (\partial_j\phi)_P r^f_j$ と、
隣接セル（境界では境界面値）との最大・最小差 $\Delta^+ = \max(\max_N \phi_N - \phi_P, 0)$、
$\Delta^- = \min(\min_N \phi_N - \phi_P, 0)$ を使い、$\Delta_f > 0$ なら $\Delta = \Delta^+$、
$\Delta_f < 0$ なら $\Delta = \Delta^-$ として

$$
\psi_f = \frac{(\Delta^2 + \varepsilon^2) + 2\Delta_f \Delta}{\Delta^2 + 2\Delta_f^2 + \Delta_f \Delta + \varepsilon^2},
\qquad
\psi_P = \min_f \operatorname{clip}(\psi_f, 0, 1),
\qquad
\varepsilon^2 = (K \min(\Delta x, \Delta y))^3,\ K = 5 .
$$

$\varepsilon^2$ が滑らかな領域でリミターを効かせない（微分可能に保つ）ための項で、Newton と相性が良い。
本実装では**残差は SOU、ヤコビアンは FOU**で組む（§3.1）。

### 2.4 Rhie–Chow 面速度

同位置格子のチェッカーボードを抑えるため、質量流束の面速度に圧力勾配の面補正を入れる:

$$
u^f_j n^f_j = \langle u_j \rangle_f n^f_j
\;-\; d_f \left[ \frac{p_N - p_P}{d_{PN}} - \big\langle (\partial_j p)\, n^f_j \big\rangle_f \right],
\qquad
d_f = \Big\langle \frac{V}{a_P} \Big\rangle_f ,
$$

$$
a_P = \sum_f \max(\pm F^{\mathrm{lin}}_f, 0) \;+\; \sum_f \frac{\mu A_f}{d_f} \;+\; \kappa_P V .
$$

$F^{\mathrm{lin}}_f$ は線形面値による質量流束（RC 補正前）、$\pm$ は流出面で $+$、流入面で $-$（FOU の運動量対角）。
$\langle (\partial_j p) n^f_j \rangle_f$ はセル中心圧力勾配（線形面圧力の差分）の面平均。
**$a_P$ には緩和も擬似時間項も含めない**（§6, §7 で含めた場合を扱う）。
これにより残差 $R(x)$ は $\Delta\tau$ にも $\alpha_u$ にも依存せず、収束解は反復パラメータと独立になる。

## 3. 非線形反復（Newton–Krylov）

$R(x) = 0$ を Newton 法で解く。$k$ 反復目の更新 $\delta^{(k)}$ は

$$
\big( J^{(k)} + D^{(k)} \big)\, \delta^{(k)} = -R(x^{(k)}),
\qquad x^{(k+1)} = x^{(k)} + \delta^{(k)},
$$

$D^{(k)}$ は対角補強（§4 擬似時間 + §5 緩和、運動量行のみ、連続式行は 0）。

### 3.1 1 次風上ヤコビアン $J_1$

FOU 残差（§2.3）を $x$ で微分した解析ヤコビアン。RC 係数 $d_f$ は凍結し、$F_f$ の $u$ 依存と $p$ 依存を含める:

$$
\frac{\partial F_f}{\partial u_{j,P}} = \rho\, \tfrac12 n^f_j A_f,
\qquad
\frac{\partial F_f}{\partial p_Q} = -\rho A_f\, d_f\, \frac{\partial}{\partial p_Q}\left[ \frac{p_N - p_P}{d_{PN}} - \big\langle (\partial_j p) n^f_j \big\rangle_f \right].
$$

ブロック形（$[u_1, u_2, p]$ 順）:

$$
J_1 = \begin{bmatrix}
C + \mathcal{N}_{11} - L + \mathcal{K} & \mathcal{N}_{12} & G_1 + \mathcal{N}_{1p} \\
\mathcal{N}_{21} & C + \mathcal{N}_{22} - L + \mathcal{K} & G_2 + \mathcal{N}_{2p} \\
B_1 & B_2 & S_{pp}
\end{bmatrix},
$$

- $C$: 風上セレクタによる対流 $\sum_f F_f\, \partial \tilde u^f/\partial u$（$F_f$ 凍結）
- $\mathcal{N}$: Newton 項 $\sum_f \tilde u^f_i\, \partial F_f/\partial(u, p)$（$F_f$ の速度・圧力依存）
- $L$: 拡散、$\mathcal{K} = \operatorname{diag}(\kappa_P V)$: Brinkman 抵抗、$G_i = \sum_f n^f_i A_f\, \partial p^f/\partial p$: 圧力勾配
- $B_j = \sum_f \partial F_f/\partial u_j$: 発散、$S_{pp} = \sum_f \partial F_f/\partial p$: RC による圧力ラプラシアン（これがあるので鞍点系が可解になる）

有限差分との整合をテストで確認済み（`tests/test_brinkman_flow.py`）。
SOU 残差に対しては $J_1$ は厳密なヤコビアンではなく、以下の 2 通りで使う。

### 3.2 JFNK（既定）と defect correction

**JFNK**: 真のヤコビアン $J_2$（SOU 残差の微分）の行列ベクトル積を有限差分で近似し、GMRES を $\mathrm{LU}(J_1 + D)$ で前処理する。

$$
(J_2 + D)\, v \;\approx\; \frac{R(x + \epsilon v) - R(x)}{\epsilon} + D v,
\qquad
\epsilon = \sqrt{\epsilon_{\mathrm{mach}}}\, \frac{\sqrt{1 + \|x\|}}{\|v\|} .
$$

GMRES は相対残差 $10^{-3}$、restart 40、最大 5 サイクル。前処理は完全 LU（`scipy.sparse.linalg.splu`）。
**Defect correction**: $(J_1 + D)\,\delta = -R_{\mathrm{SOU}}(x)$ を LU で直接解く。$J_1 \ne J_2$ なので線形収束。

### 3.3 収束判定

$$
\frac{\|R(x^{(k)})\|}{\|R(x^{(0)})\|} < 10^{-6}
\quad\text{または}\quad \|R(x^{(k)})\| < 10^{-10} .
$$

$x^{(0)} = 0$ のとき $R(0)$ は inlet 面の流束項（$R^p$ に $\rho U_\mathrm{in} \Delta y$、$R^{u_1}$ に $\rho U_\mathrm{in}^2 \Delta y$ と拡散の Dirichlet 項）**だけ**からなり、
物理的に意味のある場の残差よりずっと小さい。相対判定は実質的に絶対判定 $10^{-6}\|R(0)\|$ である（§9.1）。

## 4. 擬似時間（pseudo-transient continuation）

運動量式に擬似時間微分 $\rho\, \partial u_i/\partial\tau$ を加え、後退 Euler を 1 Newton ステップで解くと

$$
\left( \frac{\rho V}{\Delta\tau_P} \delta_{ij} + J \right) \delta = -R(x^{(k)}),
\qquad
D_P = \frac{\rho V}{\Delta\tau_P}\quad(\text{運動量行のみ、連続式行は } 0).
$$

$D \to 0$ で Newton、$D \to \infty$ で $\delta \to 0$（運動量成分のみ。圧力は §9.3）。Levenberg–Marquardt の減衰と同型で、
$D$ は $J$ の固有値を右にずらして $(J + D)^{-1}$ の作用を運動量方向に縮める。

### 4.1 局所 Δτ と速度下限

$$
\Delta\tau_P = \mathrm{CFL}\, \frac{\min(\Delta x, \Delta y)}{\max\big(|u_{1,P}| + |u_{2,P}|,\; r\, U_\mathrm{in}\big)},
\qquad r = \text{`velocity_floor_ratio`}\ (0.1) .
$$

$r = 0$（下限なし）だと静止セル・閉塞部・流路外で $\Delta\tau_P \to \infty$、$D_P \to 0$ になり、そこでは素の Newton になる。
静止初期場では**全セル**がそうなる。

### 4.2 大域 Δτ

$$
\Delta\tau = \min_P \Delta\tau_P
$$

を全セルに使う。流れが発達すると最小値は噴流部（$|u| \approx U_\mathrm{in}$ 以上）で決まり、遠方（$|u| \approx r U_\mathrm{in}$）の $D$ は局所 Δτ より
$U_\mathrm{in}/(r U_\mathrm{in}) = 1/r = 10$ 倍大きい。同じ CFL で減衰が 10 倍強い分、高 CFL に寛容（status-28: cfl 5 で局所は発散、大域は収束）。

### 4.3 SER（switched evolution relaxation）

$$
\mathrm{CFL}^{(k+1)} = \min\!\left( \mathrm{CFL}_{\max},\ \mathrm{CFL}^{(k)} \operatorname{clip}\!\left( \frac{\|R^{(k)}\|}{\|R^{(k+1)}\|},\ 0.1,\ 2 \right) \right).
$$

残差が減れば CFL を（最大 2 倍まで）増やし、増えれば（最小 0.1 倍まで）減らす。
残差が増え続けると CFL は幾何級数的に落ち、$D$ が巨大になって $\delta \to 0$、残差が動かず SER も動かない**停滞**に入る（§9.2）。

## 5. 陰的緩和

運動量対角を $a_P/\alpha_u$ に置き換える SIMPLE 系の under-relaxation を、Newton 系では対角補強として書く:

$$
D^{\mathrm{relax}}_P = \frac{1 - \alpha_u}{\alpha_u}\, a_P,
\qquad D = D^{\tau} + D^{\mathrm{relax}} .
$$

残差は変えないので収束解は $\alpha_u$ に依存しない。$a_P$ の主要部は対流 $\rho |u_P| \Delta y$ なので、
擬似時間項 $\rho V/\Delta\tau_P = \rho |u_P| \Delta y / \mathrm{CFL}$ と比べると

$$
D^{\mathrm{relax}}_P \approx \frac{1-\alpha_u}{\alpha_u}\, \rho |u_P| \Delta y
\;\Longleftrightarrow\;
\mathrm{CFL}_{\mathrm{eq}} = \frac{\alpha_u}{1 - \alpha_u}\quad(\alpha_u = 0.7 \text{ で } 2.3),
$$

つまり陰的緩和は**速度下限のない局所 Δτ（CFL ≈ 2.3）と同じもの**で、静止・低速セルでは $\sum_f \mu A_f/d_f + \kappa V$ の分しか残らない。
流路部で $\kappa V = 1.2$、$\rho U_\mathrm{in} \Delta y = 16$（72×48, U=2）なので、緩和は Δτ の速度下限の代わりにならない（status-28 の α_u 実験）。

## 6. 擬似時間項を残差にも含める場合（dual-time 型）

$$
R_\tau(x) = R(x) + D^{\tau}\, (x - x_{\mathrm{prev}})\quad(\text{運動量成分}),
\qquad
(J + D)\,\delta = -R_\tau(x^{(k)}) .
$$

- $x_{\mathrm{prev}} = x^{(k)}$（毎ステップ更新）なら $R_\tau(x^{(k)}) = R(x^{(k)})$ で更新式は §4 と同一。
  違いは更新後に評価する残差で、線形化のもとで $R(x^{(k+1)}) \approx D\delta$、$R_\tau(x^{(k+1)}) \approx 2 D \delta$。
  収束判定・SER がこの大きめの残差で動くため SER が鈍る（flat U=1 で 62 反復 → 80 反復未収束）。
- $x_{\mathrm{prev}} = x^{(0)}$（初期場に固定）なら不動点は $R(x) + D(x - x^{(0)}) = 0$ で、**初期場へのペナルティ付きの別の解**に収束する。
  $D$ が局所 Δτ で場所ごとに違うので歪みも場所依存になる。
- sub_iters $> 1$（$x_{\mathrm{prev}}$ を凍結して Newton を複数回）は真の dual-time で、各擬似ステップの後退 Euler を収束させる。

## 7. Rhie–Chow 係数に擬似時間項を含める場合

$$
d_f = \Big\langle \frac{V}{a_P + \rho V/\Delta\tau_P} \Big\rangle_f .
$$

質量流束が $\Delta\tau$ 場に依存するので残差 $R(x; \Delta\tau)$ も、したがって収束解も $\Delta\tau$ 場に依存する。
$\Delta\tau \to 0$ で $d_f \to 0$（RC 補正が消えチェッカーボードが戻る）、$\Delta\tau \to \infty$ で §2.4 に戻る。
本問題では致命的ではなく（定常残差との差 $3.1\times10^{-5}$ vs $3.9\times10^{-5}$）、
むしろ 1 反復目の増幅が小さくなる（$d_f$ が小さい分、圧力の応答が鈍る）。
ただし収束判定は $\Delta\tau$ 非依存の $R(x)$ で行うべきで、`steady_residual_ratio` として別に報告している。

## 8. 初期場: Stokes–Brinkman 解

$R(x)$ から運動量の対流項を落とした線形残差 $R_S(x)$（$\sum_f F_f \tilde u^f_i$ を除く。$F_f$ は連続式には残す）と、
対応する線形ヤコビアン $J_S$（$C$, $\mathcal{N}$ を除く）で

$$
x^{(0)} = -J_S^{-1} R_S(0)
$$

を 1 回の LU で得る。RC 係数 $d_f$ はゼロ場の $a_P = \sum_f \mu A_f/d_f + \kappa V$ で評価するので、これは離散 Stokes–Brinkman 問題の厳密解である。

**落とし穴**: 対流項込みの $R(0)$ と $J_1(0)$ で同じことをすると、速度ゼロでも inlet 面の運動量流束
$\sum_f F_f \tilde u^f_1 = \rho U_\mathrm{in}^2 \Delta y$ が inlet 隣接セルの残差に残る。これが Darcy 抵抗とだけ釣り合うので

$$
\kappa V u_1 \approx \rho U_\mathrm{in}^2 \Delta y
\;\Rightarrow\;
u_1 \approx \frac{\rho U_\mathrm{in}^2}{\kappa\, \Delta x}
= \frac{1000 \cdot 4}{1.2\times10^4 \cdot 9.7\times10^{-3}} \approx 34\ \mathrm{m/s}
$$

（実測 28.6 m/s、$U_\mathrm{in}$ の 14 倍）。正しい Stokes 場は最大 4.9 m/s（flat）/ 4.3 m/s（uturn）、NS 収束解は 3.4 m/s。
14 倍の噴流から NS を始めると 2 反復目で残差が $2\times10^6$ 倍になり発散した。

## 9. なぜ落ちるか（機構解析）

### 9.1 静止初期場の残差は「見かけ上」小さい

$R(0)$ は inlet 面の流束項のみで、$\|R(0)\| = 129$（72×48, U=2）。一方、連続式を満たす任意の速度場の運動量残差は
対流項 $\rho U^2 \Delta y$ 程度がセル数分あり、Stokes 場で $\|R\| \approx 5\times10^3$。
したがって**静止場からの 1 歩目は残差が必ず数十倍に増える**。これは発散ではなく必要な離脱で、
- 残差比で判断する大域化（Armijo ラインサーチ、backtracking）は 1 歩目を必ず棄却する
- 相対収束判定 $10^{-6}\|R(0)\|$ は実質 $10^{-6} \times 129$ の絶対判定になる
- 増幅率 $\|R^{(1)}\|/\|R^{(0)}\|$ は $U$ と細分化で増える（uturn U=2: 27×→42×→124×、flat U=2: 51×→178×）

Stokes 初期場を使うと基準残差が物理的な大きさになり、1 歩目の比は 0.7〜1.9 に落ちる。

### 9.2 速度下限なしの局所 Δτ: 1 歩目が素の Newton、以後は停滞

静止初期場では $|u_P| = 0$ なので $r = 0$ だと全セル $\Delta\tau_P = \infty$、$D = 0$。
1 歩目は $\mathrm{Re} = 2\times10^5$ の NS を減衰なしの Newton で 1 回解くことになり、しかもその線形化は $x = 0$ での $J_1(0) = J_S$（対流なし）。
つまり **1 歩目は §8 の「落とし穴付き Stokes 解」そのもの**で、14 倍の噴流が立つ。
2 歩目以降は噴流部で $\Delta\tau$ が小さく遠方で大きい（範囲 $[10^{-5}, 10^{2}]$ s、7 桁）。
残差が増え続けて SER が CFL を 0.02 まで落とし、$D$ が巨大になって $\delta \approx 0$、残差が動かず CFL も動かない停滞に入る。
uturn 72×48 U=2 で 80 反復後も $\|R\|/\|R^{(0)}\| = 35$（初期残差を一度も下回らない）。

速度下限 $r U_\mathrm{in}$ を入れると初期場での $\Delta\tau_P = \mathrm{CFL}\,\Delta x/(r U_\mathrm{in})$ が有限になり、1 歩目が減衰付きになる。
これだけで全ケースが単調減少に転じる。

### 9.3 Δτ が小さすぎると圧力が発散する

鞍点系の運動量ブロックを $A = J_{uu} + D$、発散を $B$、RC 圧力項を $S_{pp}$ と書くと、Schur 補元は

$$
S = S_{pp} - B A^{-1} B^\top .
$$

$\Delta\tau \to 0$ で $A \approx D \propto 1/\Delta\tau$、$A^{-1} \propto \Delta\tau$ なので $B A^{-1} B^\top \propto \Delta\tau \to 0$ であり、
$S \to S_{pp}$（RC 項のみ）。圧力更新は

$$
\delta p = S^{-1}\!\left( -R^p + B A^{-1} R^u \right)
$$

で、速度がほぼ凍結される（$\delta u \approx -D^{-1}(R^u + B^\top \delta p) \to 0$）一方、連続式残差 $R^p$ を $S_{pp}$ だけで消そうとする。
$S_{pp}$ は $d_f \propto 1/a_P$ を係数に持つ弱いラプラシアンなので、非ソレノイダルな場（静止場に inlet 流束が入った状態）では
$\delta p$ が大きくなり、運動量残差の $\sum_f p^f n^f_i A_f$ が跳ね上がる。
backtracking で CFL を $10^{-13}$ まで下げた実験で残差が $4\times10^{8}$ まで増えたのはこれで、
**CFL には下限が要る**（`cfl_min`）。SER の 0.1 倍クリップも同じ理由で、CFL が落ち続ける経路には保護がない。

### 9.4 大域 Δτ が高 CFL に寛容な理由

§4.2 の通り、遠方の $D$ が局所 Δτ の $1/r = 10$ 倍。同じ CFL=5 で局所は 3 反復で発散、大域は収束した。
逆に CFL=0.5 では局所（下限あり）の方が速い（噴流部以外を過剰に減衰しない）。
Fluent の pseudo-transient automatic time scale が領域長さと境界速度で決まる大域的な値であることと整合する。

### 9.5 陰的緩和が「効いたり効かなかったり」する理由

§5 の通り、緩和は速度下限なしの局所 Δτ と同じ盲点を持つ。下限なしでは α_u の効きがモデルごとに逆転し
（uturn は α=1 が通り 0.7 が停滞、flat は逆）、下限ありでは α=1（緩和なし）が最速で緩和は Newton を鈍らせるだけだった
（uturn 36 反復 vs 102 反復）。

### 9.6 Stokes 初期場と backtracking の効果

- Stokes 初期場（正しい版）: 1 歩目の増幅を消し、反復を 25〜35% 減らす。4 ケース全て収束。
- backtracking（残差 2 倍超で棄却、CFL 半減、1 歩目は対象外、cfl_min あり）: flat は発動せず、uturn は 3 回の棄却で CFL を下げた分だけ遅れて未収束。
  残差増加の主因が「必要な離脱」と「GMRES 未収束の序盤」なので、棄却より $\Delta\tau$ の下限が本質。

## 10. 実験結果の要約（詳細は status-28）

| 線 | 影響 | 対処 |
|---|---|---|
| 局所 Δτ に速度下限なし | 静止場で $D = 0$、1 歩目が素の Newton。停滞の主因。U・細分化で悪化 | $\max(|u|, 0.1 U_\mathrm{in})$ |
| CFL が大きい（局所 Δτ で 5） | 3 反復で発散 | 0.5、または大域 Δτ |
| 擬似時間項を残差に含める | $x_{\mathrm{prev}}$ 更新が正しければ解は同じ。SER が鈍る | 対角のみに加える |
| $x_{\mathrm{prev}}$ を初期場に固定 | 初期場へのペナルティ付きの別解に収束 | 毎ステップ更新 |
| RC 係数に擬似時間項 | 収束解が Δτ 場に依存（本問題では小） | $d_f = V/a_P$、判定は $R(x)$ で |
| Darcy 初期場を対流込み残差で作る | inlet 運動量流束が残り $|u| \approx 14 U_\mathrm{in}$ の噴流、発散 | 対流項を残差・ヤコビアンから落とす |
| Δτ を下げすぎる | 圧力が $1/\Delta\tau$ で発散 | `cfl_min` |
| 陰的緩和 | Δτ 未管理では効きが読めない。管理後は Newton を鈍らせる | α_u = 1 |

## 11. 記号表

| 記号 | 意味 | 実装 |
|---|---|---|
| $u_i, p$ | 深さ平均速度 [m/s]、圧力 [Pa] | `x = [u, v, p]` |
| $\rho, \mu, \mu_b$ | 密度、粘度、Brinkman 粘度 | `rho, mu, mu_b` |
| $\kappa = 12\mu_b/h^2$ | Brinkman 抵抗係数 [kg/(m³s)] | `disc.drag` |
| $F_f$ | 面質量流束 [kg/s]（単位深さ） | `st.fx, st.fy` |
| $\tilde u^f_i$ | 対流面値（FOU/SOU） | `st.conv_ufx` 等 |
| $a_P, d_f$ | RC 用運動量対角、RC 係数 | `st.a_p, st.dfx, st.dfy` |
| $\psi_P, K$ | Venkatakrishnan リミター、定数 | `_venkatakrishnan`, `venkat_k` |
| $J_1, J_2$ | FOU 解析ヤコビアン、SOU 有限差分 J·v | `jacobian_first_order`, `_make_jfnk_matvec` |
| $D^\tau_P = \rho V/\Delta\tau_P$ | 擬似時間対角 | `tau_diag` |
| $D^{\mathrm{relax}}_P$ | 陰的緩和対角 | `relax` |
| $\mathrm{CFL}, r$ | 擬似 CFL、速度下限比 | `cfl_init, velocity_floor` |
| $R_S, J_S$ | Stokes–Brinkman 残差・ヤコビアン | `convection=False` |
