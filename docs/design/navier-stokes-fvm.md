# 非圧縮 Navier–Stokes ソルバー（面ベース FVM）設計仕様 — `NavierStokesFVMProcess`

[← README](../../README.md) | [設計文書一覧](README.md) | [FVM 層の設計](fvm-layer.md) | [構造格子版（NaturalConvectionFDM）](natural-convection-fdm.md) | [棚卸しと計画](../plans/2026-09-05-solver-layering.md)

## 位置付け

`NaturalConvectionFDMProcess`（構造格子、SIMPLE + Rhie–Chow、Boussinesq、固体マスク）と
`BrinkmanFlowFVMProcess`（2D、Brinkman 抵抗）を、面ベース FVM 共通低レイヤー
[`xkep_cae_fluid.fvm`](fvm-layer.md) の上に **1 つの方程式ファミリー**として載せ替えたもの。
`MeshData`（構造格子 / polyMesh / `.inp` の任意六面体）を同じ経路で解き、境界条件はパッチ名 →
`FlowPatchBC`（速度 + 温度）で与える。

運動量・圧力連成のカーネル（速度境界の展開、成分ごとの運動量行列、Rhie–Chow 面流束、
圧力補正、流束修正）は `xkep_cae_fluid/fvm/momentum.py` にあり、方程式ファミリー非依存。

## 支配方程式

- 連続: ∇·u = 0
- 運動量: ρ ∂u/∂t + ∇·(ρ u u) = −∇p + ∇·(μ∇u) − (μ/K) u − ρ β (T − T_ref) g + f
  （K: 透過率、inf で抵抗なし。β = 0 で浮力なし。f: 一様 / セル別の体積力 `body_force` [N/m³]）
- 対流項は `convection="none"`（Stokes）で落とせる。粘度は `viscosity_model` で μ(γ̇) にできる
  （γ̇ = sqrt(2 D:D) を最小二乗の速度勾配から評価する Picard。変粘度では ∇·(μ∇uᵀ) の余剰項
  Σ_j ∂_i u_j ∂_j μ を陽的ソースに足す）
- エネルギー（`solve_energy`）: ρC ∂T/∂t + ∇·(ρC u T) = ∇·(k∇T) + q（固体セルは u = 0、k = k_solid）

## 離散化（同位置格子 SIMPLE / SIMPLEC / PISO / COUPLED）

1. 圧力勾配 ∇p: 最小二乗 `cell_gradient_lsq`（OUTLET は Dirichlet、他はゼロ勾配）
2. 運動量成分ごとに: 対流 1 次風上（`assemble_convection(bounded=True)`、有界形）+ TVD 遅延補正
   （`convection="tvd"`、van Leer / Superbee、`tvd_deferred_correction`）+ 拡散 μ（over-relaxed
   非直交補正付き）+ 時間項（陰的 Euler / BDF2）+ 圧力勾配 −∂p/∂x_c V + 浮力 + 抵抗 μ/K V（対角）、
   陰的緩和 α_u → u*
3. Rhie–Chow 面質量流束: ṁ_f = ρ[ ū_f·S_f − D⁰_f((p_N − p_P)|E_f|/d_PN − (∇p)_f·E_f) ]、
   D⁰_f = interp(V/a⁰_P)、a⁰_P = α_u a_P は**緩和前**の対角（Majumdar 1988。緩和後の a_P を使うと収束解が
   α_u に依存する。16×16 の Stokes 的キャビティで α_u = 0.5 と 0.7 の差が 2% → 3e-8 に）。
   境界: INLET ρ u_in·S、OUTLET ρ u_P·S、壁・対称 0、OUTFLOW は ρ u_P·S を他の境界の正味流入と釣り合うよう
   スケーリング
4. 圧力補正 Σ_f a_f (p'_P − p'_N) = −Σ_f ṁ_f + Σ_f c_f、a_f = ρ D_f |E_f|/d_PN、D_f = interp(V/a_P)
   （SIMPLEC は V/(a_P − Σ|a_nb|)）。**非直交補正**: c_f = ρ D_f (∇p')_f·T_f を前回の p'
   （最小二乗勾配）で陽的に評価し、`n_nonorthogonal_correctors` 回（既定 2、直交メッシュでは 1 回）
   p' を解き直す。OUTLET は p' = 0、内部吐出・吸入セルは p' = 0 に固定、どちらも無い閉領域はセル 0 を基準
5. p += α_p p'、u −= (V/a_P)∇p'、ṁ_f −= a_f (p'_N − p'_P) + c_f（同じ c_f を使うので修正後の流束の
   発散は解いた線形系と厳密に整合する）
   - PISO（`coupling="piso"`、α_p = 1）: 修正した速度で隣接項 H(u) = b − A_off u を再評価し、新しい圧力勾配で
     u** = H/a_P を作って 3–4 を繰り返す（`n_piso_correctors` 回、Issa 1986）。非定常で外部反復 1 回の
     時間進行に使う
6. エネルギー: `assemble_scalar_transport`（ṁ、k、q、時間項、有界形、TVD）に陰的緩和 α_T
7. 追加スカラー（`scalars: ScalarSpec`）: 同じ ṁ・スキーム、拡散係数 Γ とパッチ境界（`PatchBC`）は個別

速度境界（`VelocityPatchBC`）:

| 種別 | 運動量 | 質量流束 | 圧力 |
|---|---|---|---|
| WALL（`velocity` で動く壁） | Dirichlet | 0 | ゼロ勾配 |
| INLET | Dirichlet | ρ u_in·S | ゼロ勾配 |
| OUTLET（`FlowPatchBC.outlet`） | ゼロ勾配 | ρ u_P·S（p' で修正） | Dirichlet |
| OUTFLOW（`FlowPatchBC.outflow`、対流流出） | ゼロ勾配 | ρ u_P·S を流入と釣り合わせてスケーリング | ゼロ勾配（p' もゼロ勾配。基準はセル 0 か内部吸入セル） |
| SLIP / 対称面 | 軸に平行な面: 法線成分 Dirichlet 0・接線成分ゼロ勾配（陰的）。傾いた面: owner 速度の接線射影を Dirichlet（遅延） | 0 | ゼロ勾配 |

固体セル（`solid_mask`）: 運動量行は u = 0 に固定、接する面の流束と圧力補正係数は 0。

内部セル境界（`internal_bcs: InternalCellBC`、構造格子版 `InternalFaceBC` 相当）: INLET セルは運動量行を
指定速度に、エネルギー行を `temperature` に固定し（`fix_rows`）、圧力補正を p' = 0 に固定して質量の
湧き出しを許す。OUTLET セルは p' = 0 の固定だけ（吸い込み）。これらのセルの質量不整合は残差に数えない。
対流を有界形（∇·(ṁφ) − φ∇·ṁ）にしているので、吸い込みセルで φ が流入値の範囲を超えない。

収束判定: 運動量 3 成分の相対初期残差 ‖b − A u‖/max(‖b‖, ‖A u‖, ‖a_P‖U_ref)、質量不整合
Σ|Σ_f ṁ_f| /(Σ_f|ṁ_f|/2)、エネルギー連成なら温度残差、の最大値が `tol` 未満（2 反復目から）。
追加スカラーは判定に含めない（構造格子版と同じ）。NaN か 1e20 超で発散として打ち切る。
反復 1〜5 と 10 反復ごとに残差をログに出す（`ykep ... int` で端末にも）。

適応緩和（`adaptive_relaxation`）: 構造格子版と同じ規則を [`fvm/relaxation.py`](fvm-layer.md) に切り出して
共有する。前回比 0.8 未満で α_u, α_p を 1.1 倍（上限 0.9 / 0.5）、1.2 超で 0.8 倍（下限 0.1 / 0.05）、
加えて**最小残差の 5 倍を超えたら保守化**（前回比は小さいがじわじわ発散する型。cavity-nc-2 で
旧規則は 75 → 474 反復に悪化していた）、SIMPLE では α_p ≤ 1 − α_u。`alpha_history` に反復ごとの値を残す。
実測: cavity-nc-2（非直交 14°）75 → 62 反復、cavity-nc-1 の非構造経路 275 → 219 反復。

### 速度–圧力の連成（`coupling="coupled"`）

SIMPLE 系の分離解法に代えて、速度 nd 成分と圧力を 1 つの線形系にまとめて直接解く
（`fvm/momentum.assemble_coupled`）。

```
[ A_u        V ∂/∂x G_p ] [u]   [b_u]
[ A_v        V ∂/∂y G_p ] [v] = [b_v]
[ ρ Div F_u  ρ Div F_p   ] [p]   [b_c]
```

- 圧力勾配は最小二乗勾配の**線形作用素**（`geometry.lsq_gradient_operator`。
  `cell_gradient_lsq` と同じ係数を疎行列 + Dirichlet 境界の定数項として返す）
- 連続式は Rhie–Chow 面流束（D_f = interp(V/a_P)、a_P は緩和前の対角）を u と p の両方について陰的に
- 緩和係数を使わない。Stokes（`convection="none"`）なら**外部反復 2 回**で収束し、
  対流があっても面流束の Picard だけで回る

鞍点系なので直接法（`DirectSolver`）で解く。`OUTFLOW`（対流流出）と `adaptive_relaxation` は使えない。
Stokes キャビティ（16×16）と Re=100 キャビティ（20×20）で SIMPLE と同じ解（1e-6 / 1e-4）に、
反復数は 273 → 2、197 → 10 になる。

### 周期境界

`MeshData.face_offset` がある内部面（`.inp` の `*BOUNDARY, TYPE=PERIODIC`）は、
neighbour セル中心を並進で戻した位置で幾何を評価するだけで、拡散・対流・圧力補正・Rhie–Chow は
そのまま通る（[fvm-layer.md](fvm-layer.md)、[inp-generic-extrusion.md](inp-generic-extrusion.md)）。
圧力跳びは `body_force` に移す。

## 入出力

| | 内容 |
|---|---|
| `NavierStokesFVMInput` | `mesh`、`rho`、`mu`、`bcs`（パッチ → `FlowPatchBC`）、`solve_energy`、`Cp`、`k_fluid`、`beta`、`T_ref`、`gravity`、`T0`/`u0`/`p0`、`solid_mask`、`k_solid`、`heat_source`、`permeability`、`dt`/`t_end`、`max_outer_iter`、`tol`、`alpha_u`/`alpha_p`/`alpha_T`、`adaptive_relaxation`、`coupling`（simple / simplec / piso / coupled）、`viscosity_model`、`alpha_mu`、`body_force`、`n_piso_correctors`、`n_nonorthogonal_correctors`（既定 2）、`convection`（upwind / tvd / none）、`limiter`（van_leer / superbee）、`time_scheme`（euler / bdf2）、`scalars`（`ScalarSpec`）、`internal_bcs`（`InternalCellBC`）、`linear_solver`/`pressure_solver`、`tol_inner`/`max_inner_iter` |
| `NavierStokesFVMResult` | `velocity (n_cells, 3)`、`p`、`T`、`mass_flux (n_faces,)`、`scalars`（名前 → 場）、`converged`、`n_outer_iterations`、`n_timesteps`、`residual_history`（u/v/w/T/mass/スカラー名）、`residual_fields`（res_u/res_v/res_w/res_T/res_mass/res_<名前>）、`alpha_history`（適応緩和のときの alpha_u / alpha_p）、`viscosity` / `strain_rate`（`viscosity_model` のときの μ と γ̇） |

`FlowPatchBC.wall(temperature=, heat_flux=, film=(h, T_inf), velocity=)` /
`rotating_wall(angular_velocity, center=, velocity=, …)`（u = v + ω × (x − center)）/
`inlet(velocity, temperature=)` / `outlet(pressure=, temperature=)` / `outflow(temperature=)` / `symmetry()`。
`InternalCellBC.inlet(mask, velocity, temperature=)` / `outlet(mask)`。
`ScalarSpec(name, diffusivity, phi0=, source=, bcs={パッチ: PatchBC}, alpha=)`。

## テスト（`tests/test_navier_stokes_fvm.py`）

- API: メタ、入力検証（coupling / convection / limiter / time_scheme / スカラー名の重複 / mask 形状）、
  結果の形と残差マップ、全オプション（TVD Superbee + BDF2 + PISO 3 回 + 対流流出 + スカラー + 内部セル）の通し
- 物理:
  - 平行平板 Poiseuille（箱格子 / せん断メッシュ / 対流流出 OUTFLOW）: 出口の放物線分布と圧力勾配 12 μ U/H²、流入 = 流出
  - Brinkman 流路（すべり壁、一様 K）: 一様速度と圧力降下 μ U L/K
  - 蓋駆動キャビティ Re=100（24×24）: 1 次風上は中心線 u の極小値が Ghia (1982) の範囲内（−0.185）、
    TVD van Leer は −0.211（Ghia −0.2109）
  - 差分加熱キャビティ Ra=10³（z 対称面）: Nu が de Vahl Davis 1.118 の 20% 以内、構造格子 FDM 版と整合
  - PISO: Stokes 的キャビティ（μ = 1）の 1 ステップを外部反復 1 回で解き、連成解との差が補正 1 → 2 → 3 回で
    6% → 1.5% → 0.8% と減る（残りは遅延評価の面流束）
  - BDF2: 静止流体の 1D 熱伝導 sin(πx) の減衰で Euler の 1/5 未満の誤差（0.049 → 0.0065）
  - 内部セル吐出・吸入: 吐出セルの速度・温度が固定、他のセルは質量保存（1e-12）、湧き出し = 吸い込み、
    温度が [初期, 吐出] に収まる（有界形）
  - 追加スカラー: 流路で流入値が全域に運ばれる、閉じたキャビティのトレーサ（非定常、TVD）の総量保存と 0 ≤ c ≤ 1
  - 圧力補正の非直交補正: せん断 0.6（31°）の Stokes 的キャビティ 1 ステップを α_u = 0.8, α_p = 0.5 で解くと、
    補正 1 回は 60 反復で収束せず、2 回は直交メッシュ並みの反復数で収束して解は保守的な緩和の収束解と 1e-5 で一致。
    直交メッシュでは補正回数によらず同じ反復数（余分な圧力解法をしない）
  - 適応緩和: Poiseuille 流路で収束し、`alpha_history` が反復ごとに記録され、上下限と α_p ≤ 1 − α_u を守る
  - 周期境界 + 体積力: 周期流路の Poiseuille が解析解と 3e-3、圧力は周期方向に一様（跳びは体積力に移してある）。
    1 セル厚の z を周期にすると w が厚さに依らない厳密な 2.5D になり、対称面にすると厚さに依存する偽の解になる
  - Stokes モード: `convection="none"` は密度を変えても解が動かない
  - COUPLED: Stokes キャビティを 2 反復で解き SIMPLE（273 反復）と 1e-6 一致、Re=100 でも 1e-4 一致
  - 回転壁: Taylor–Couette（円環 16×96、外周回転）が解析解 u_θ = Ar + B/r と 5e-3、半径方向速度は 1e-9 以下
  - 非ニュートン: べき乗則流路が解析解と 5e-3、`gamma_min` / `alpha_mu` を変えても不動点が動かない、
    Carreau の n=1・μ_0=μ_∞ がニュートンに退化する

実測（`docs/status/status-35.md`）: 圧力補正の非直交補正は定常 SIMPLE の**収束解を変えず**（p' → 0）、
効くのは緩和が強いとき・歪みが大きいときの安定性。せん断 0.6 / 1.0（31° / 45°）で α = (0.8, 0.5) は
補正 1 回だと発散、2 回で 28〜30 反復。**3 回は 45° で発散**（遅延補正の反復の縮小率が tan θ ≈ 1 で
縮小しない）ので既定は 2。cavity-nc-2（14°）は補正回数によらず 75 反復。

## 制限と TODO

- 圧力補正の非直交補正は無制限（unlimited）の遅延補正。45° を超えるメッシュでは OpenFOAM の `limited ψ`
  相当（T_f を ψ 倍）が要る
- OUTFLOW は流出流束のスケーリング（Fluent の outflow 相当）で、非定常の非反射条件 ∂u/∂t + U_c ∂u/∂n = 0 ではない
- 乱流モデルなし。CFL 適応 dt は構造格子版のみ
- COUPLED は直接法固定（鞍点系の前処理付き Krylov 法は未実装）で `OUTFLOW` 非対応。大規模では SIMPLE 系が省メモリ
- 周期境界は**並進のみ**（回転周期・螺旋周期は未対応）
- 非ニュートン粘度の Picard は残差で収束判定する（粘度場の変化量では見ていない）。粘性発熱 Φ = μγ̇² は未対応
- 構造格子版 `NaturalConvectionFDMProcess` の Rhie–Chow は緩和後の a_P を使っている（収束解の α_u 依存が
  残っている可能性。空気実物性の不安定化調査と合わせて確認する）
- 内部セル境界は `.inp` では要素集合を target にした `*BOUNDARY`（TYPE=VELOCITY / PRESSURE / TEMPERATURE）で与える
  （[inp-format.md](inp-format.md)）。内部面（`*SURFACE`）はバッフル（厚さゼロの壁、両側同条件。
  WALL / SLIP / SYMMETRY / TEMPERATURE と `*DFLUX` / `*SFILM`）として `InpMeshProcess` が両側の境界面に
  分割する（[unstructured-inp-mesh.md](unstructured-inp-mesh.md)）。内部面の流入・流出（ファン・圧力ジャンプ）は無い
