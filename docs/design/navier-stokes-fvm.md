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
- 運動量: ρ ∂u/∂t + ∇·(ρ u u) = −∇p + ∇·(μ∇u) − (μ/K) u − ρ β (T − T_ref) g
  （K: 透過率、inf で抵抗なし。β = 0 で浮力なし）
- エネルギー（`solve_energy`）: ρC ∂T/∂t + ∇·(ρC u T) = ∇·(k∇T) + q（固体セルは u = 0、k = k_solid）

## 離散化（同位置格子 SIMPLE / SIMPLEC / PISO）

1. 圧力勾配 ∇p: 最小二乗 `cell_gradient_lsq`（OUTLET は Dirichlet、他はゼロ勾配）
2. 運動量成分ごとに: 対流 1 次風上（`assemble_convection(bounded=True)`、有界形）+ TVD 遅延補正
   （`convection="tvd"`、van Leer / Superbee、`tvd_deferred_correction`）+ 拡散 μ（over-relaxed
   非直交補正付き）+ 時間項（陰的 Euler / BDF2）+ 圧力勾配 −∂p/∂x_c V + 浮力 + 抵抗 μ/K V（対角）、
   陰的緩和 α_u → u*
3. Rhie–Chow 面質量流束: ṁ_f = ρ[ ū_f·S_f − D_f((p_N − p_P)|E_f|/d_PN − (∇p)_f·E_f) ]、
   D_f = interp(V/a_P)（SIMPLEC は V/(a_P − Σ|a_nb|)）。境界: INLET ρ u_in·S、OUTLET ρ u_P·S、壁・対称 0、
   OUTFLOW は ρ u_P·S を他の境界の正味流入と釣り合うようスケーリング
4. 圧力補正 Σ_f a_f (p'_P − p'_N) = −Σ_f ṁ_f、a_f = ρ D_f |E_f|/d_PN。OUTLET は p' = 0、内部吐出・吸入セルは
   p' = 0 に固定、どちらも無い閉領域はセル 0 を基準
5. p += α_p p'、u −= (V/a_P)∇p'、ṁ_f −= a_f (p'_N − p'_P)
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
追加スカラーは判定に含めない（構造格子版と同じ）。

## 入出力

| | 内容 |
|---|---|
| `NavierStokesFVMInput` | `mesh`、`rho`、`mu`、`bcs`（パッチ → `FlowPatchBC`）、`solve_energy`、`Cp`、`k_fluid`、`beta`、`T_ref`、`gravity`、`T0`/`u0`/`p0`、`solid_mask`、`k_solid`、`heat_source`、`permeability`、`dt`/`t_end`、`max_outer_iter`、`tol`、`alpha_u`/`alpha_p`/`alpha_T`、`coupling`（simple / simplec / piso）、`n_piso_correctors`、`convection`（upwind / tvd）、`limiter`（van_leer / superbee）、`time_scheme`（euler / bdf2）、`scalars`（`ScalarSpec`）、`internal_bcs`（`InternalCellBC`）、`linear_solver`/`pressure_solver`、`tol_inner`/`max_inner_iter` |
| `NavierStokesFVMResult` | `velocity (n_cells, 3)`、`p`、`T`、`mass_flux (n_faces,)`、`scalars`（名前 → 場）、`converged`、`n_outer_iterations`、`n_timesteps`、`residual_history`（u/v/w/T/mass/スカラー名）、`residual_fields`（res_u/res_v/res_w/res_T/res_mass/res_<名前>） |

`FlowPatchBC.wall(temperature=, heat_flux=, film=(h, T_inf), velocity=)` / `inlet(velocity, temperature=)` /
`outlet(pressure=, temperature=)` / `outflow(temperature=)` / `symmetry()`。
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

## 制限と TODO

- 圧力補正に非直交補正は入れていない（歪んだメッシュでは外部反復で吸収）
- OUTFLOW は流出流束のスケーリング（Fluent の outflow 相当）で、非定常の非反射条件 ∂u/∂t + U_c ∂u/∂n = 0 ではない
- `ADAPTIVE`（適応緩和）、乱流モデルなし
- 内部セル境界は `.inp` では要素集合を target にした `*BOUNDARY`（TYPE=VELOCITY / PRESSURE / TEMPERATURE）で与える
  （[inp-format.md](inp-format.md)）。内部面（`*SURFACE`）単位の指定は無い（セル単位で十分なので保留）
