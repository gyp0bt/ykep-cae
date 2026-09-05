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

## 離散化（同位置格子 SIMPLE / SIMPLEC）

1. 圧力勾配 ∇p: Green–Gauss（OUTLET は Dirichlet、他はゼロ勾配 + 接線外挿）
2. 運動量成分ごとに: 対流 1 次風上（`assemble_convection`）+ 拡散 μ（over-relaxed 非直交補正付き）
   + 時間項 + 圧力勾配 −∂p/∂x_c V + 浮力 + 抵抗 μ/K V（対角）、陰的緩和 α_u → u*
3. Rhie–Chow 面質量流束: ṁ_f = ρ[ ū_f·S_f − D_f((p_N − p_P)|E_f|/d_PN − (∇p)_f·E_f) ]、
   D_f = interp(V/a_P)（SIMPLEC は V/(a_P − Σ|a_nb|)）。境界: INLET ρ u_in·S、OUTLET ρ u_P·S、壁・対称 0
4. 圧力補正 Σ_f a_f (p'_P − p'_N) = −Σ_f ṁ_f、a_f = ρ D_f |E_f|/d_PN。OUTLET は p' = 0、
   OUTLET が無い閉領域はセル 0 を基準
5. p += α_p p'、u −= (V/a_P)∇p'、ṁ_f −= a_f (p'_N − p'_P)
6. エネルギー: `assemble_scalar_transport`（ṁ、k、q、時間項）に陰的緩和 α_T

速度境界（`VelocityPatchBC`）:

| 種別 | 運動量 | 質量流束 | 圧力 |
|---|---|---|---|
| WALL（`velocity` で動く壁） | Dirichlet | 0 | ゼロ勾配 |
| INLET | Dirichlet | ρ u_in·S | ゼロ勾配 |
| OUTLET | ゼロ勾配 | ρ u_P·S（p' で修正） | Dirichlet |
| SLIP / 対称面 | owner 速度の接線射影を Dirichlet（遅延） | 0 | ゼロ勾配 |

固体セル（`solid_mask`）: 運動量行は u = 0 に固定、接する面の流束と圧力補正係数は 0。

収束判定: 運動量 3 成分の相対初期残差 ‖b − A u‖/‖b‖ と質量不整合 Σ|Σ_f ṁ_f| /(Σ_f|ṁ_f|/2) の
最大値が `tol` 未満。温度残差は判定に含めない（構造格子版と同じ）。

## 入出力

| | 内容 |
|---|---|
| `NavierStokesFVMInput` | `mesh`、`rho`、`mu`、`bcs`（パッチ → `FlowPatchBC`）、`solve_energy`、`Cp`、`k_fluid`、`beta`、`T_ref`、`gravity`、`T0`/`u0`/`p0`、`solid_mask`、`k_solid`、`heat_source`、`permeability`、`dt`/`t_end`、`max_outer_iter`、`tol`、`alpha_u`/`alpha_p`/`alpha_T`、`coupling`（simple / simplec）、`linear_solver`/`pressure_solver`、`tol_inner`/`max_inner_iter` |
| `NavierStokesFVMResult` | `velocity (n_cells, 3)`、`p`、`T`、`mass_flux (n_faces,)`、`converged`、`n_outer_iterations`、`n_timesteps`、`residual_history`（u/v/w/T/mass）、`residual_fields`（res_u/res_v/res_w/res_T/res_mass） |

`FlowPatchBC.wall(temperature=, heat_flux=, film=(h, T_inf), velocity=)` / `inlet(velocity, temperature=)` /
`outlet(pressure=, temperature=)` / `symmetry()`。

## テスト（`tests/test_navier_stokes_fvm.py`）

- API: メタ、入力検証、結果の形と残差マップ
- 物理:
  - 平行平板 Poiseuille（箱格子 / せん断メッシュ）: 出口の放物線分布と圧力勾配 12 μ U/H²、流入 = 流出
  - Brinkman 流路（すべり壁、一様 K）: 一様速度と圧力降下 μ U L/K
  - 蓋駆動キャビティ Re=100: 中心線速度の極小値と位置が Ghia (1982) と整合（1 次風上・粗格子の範囲で）
  - 差分加熱キャビティ Ra=10³（z 対称面）: Nu が de Vahl Davis 1.118 の 20% 以内、構造格子 FDM 版と整合

## 制限と TODO

- 対流は 1 次風上のみ（TVD の遅延補正は未移植）。時間積分は陰的 Euler のみ
- `InternalFaceBC`（水槽内部の吐出・吸入）、追加スカラー（`extra_scalars`）、PISO は未移植
- 圧力補正に非直交補正は入れていない（歪んだメッシュでは外部反復で吸収）
- 乱流モデルなし
