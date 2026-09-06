# Darcy 流れ（`DarcyFlowProcess`、面ベース FVM）設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← FVM 層](fvm-layer.md) | [← .inp フォーマット](inp-format.md)

## 概要

多孔質媒体の定常 Darcy 流れ

$$
\nabla\cdot\mathbf{u} = S,\qquad \mathbf{u} = -\frac{K}{\mu}\nabla p
$$

を圧力のポアソン方程式 $\nabla\cdot\left(\frac{K}{\mu}\nabla p\right) = -S$ として解く。
`*DARCY` 方程式ファミリー（`.inp`）の実行先で、面ベース FVM 低レイヤー
（[fvm-layer.md](fvm-layer.md)）の上に書かれた最初の「非構造格子で動く」ソルバー。
構造格子（`StructuredMeshProcess`）、`.inp` の任意六面体メッシュ（`InpMeshProcess`）、
OpenFOAM polyMesh（`PolyMeshReaderProcess`）のどの `MeshData` でも同じ経路で解ける。

参考: `experiments/coldplate/darcy.py` の面リスト `(fi, fj)` 実装（torch、随伴付き）は本 Process の
原型で、実験側に残す。

## 入出力

- `DarcyFlowInput(mesh, permeability, viscosity, density=1000, bcs={patch: DarcyPatchBC}, source=None, p0=None, linear_solver="direct", tol, max_iter, max_nonorthogonal_iter, forchheimer=0, max_picard_iter, picard_tol, specific_storage=0, dt=0, t_end=0, output_interval)`
  - `forchheimer` β [1/m]（スカラーかセル配列）: −∇p = (μ/K) u + β ρ |u| u。実効移動度 Γ = (K/μ)/(1 + β ρ K |u|/μ) を
    セル速度 |u| で更新する Picard 反復（速度の相対変化 < `picard_tol`）
  - `specific_storage` S_s [1/Pa] と `dt > 0` で非定常 S_s ∂p/∂t + ∇·u = S（陰的 Euler）。閉じた領域でも
    時間項があれば圧力基準は不要
  - `permeability` K [m²]: スカラーかセル配列（正の値。不透過は極小値か WALL 境界）
  - `DarcyPatchBC`: `PRESSURE`（p 固定）/ `VELOCITY`（法線流入速度 u_n、正 = 流入）/ `WALL`（不透過、未指定パッチの既定）
  - 圧力の基準として PRESSURE パッチが 1 つ以上必要（無ければ `ValueError`）
- `DarcyFlowResult(p, velocity (n_cells,3), face_flux (n_faces,), mass_residual (n_cells,), converged, residual, inflow, outflow, n_nonorthogonal_iter, n_picard_iter, n_timesteps, time_history, p_history)`

## 離散化

- 拡散係数 Γ = K/μ（面は調和平均）で `fvm.assemble_scalar_transport(gamma=Γ, source=S)` を組む
- 境界: PRESSURE → Dirichlet、VELOCITY → Neumann（拡散流束 J = −Γ∇p = u なので u_n がそのまま流入量）、WALL → ゼロ勾配
- 面の体積流量 q_f = −Γ_f A_f ∂p/∂n（Γ_f は調和平均、境界は Dirichlet 値 / 流入速度）を評価し、
  セル速度は u_P = (1/V_P) Σ_f q_f (x_f − x_P) で再構成する（定数速度で厳密。透過率が不連続な
  界面セルでも面流束と整合。Green–Gauss 勾配 × Γ_P では界面セルの速度が両側の混合になる）
- セルごとの Σ_f q_f − S V を `mass_residual` に返す
  （直接法なら丸め誤差程度）

## テスト（`tests/test_darcy.py`）

- API: メタ情報、戻り値の形、K ≤ 0 / 圧力基準なし / パッチ名不正の拒否
- 物理: 1D 圧力差の一様流 u = K Δp/(μ L)（構造格子・せん断メッシュ）、流入速度指定の線形圧力分布、
  2 層透過率の直列則、質量保存（各セルの不整合が丸め誤差、流入 = 流出）、ソース項の総流出、
  Forchheimer の圧力勾配 μu/K + βρu²（Picard 収束）、閉領域 + 一様ソースの非定常 p = S t/S_s、
  非定常が定常解へ緩和

## `.inp` からの実行

```
*MATERIAL, NAME=SAND
*VISCOSITY
 1e-3
*PERMEABILITY
 1e-10
*FLUID SECTION, ELSET=ALL, MATERIAL=SAND
*STEP
*DARCY, STEADY STATE
*BOUNDARY, TYPE=PRESSURE
 XM, 1000.
 XP, 0.
*CONTROLS, PARAMETERS=SOLVER
 METHOD=DIRECT
*OUTPUT, FIELD, FORMAT=VTK
*END STEP
```

材料に `*FORCHHEIMER`（β [1/m]）を書くと慣性補正、`*SPECIFIC STORAGE`（S_s [1/Pa]）と
`*DARCY` のデータ行 `dt, time_period`（`STEADY STATE` なし）で非定常。`*CONTROLS, PARAMETERS=SOLVER` に
`MAX_PICARD` / `PICARD_TOL`、`PARAMETERS=TIME INCREMENTATION` に `OUTPUT_INTERVAL`。

`InpCaseRunnerProcess` は `*DARCY` ステップでは `StructuredGridRecoveryProcess` の代わりに
`InpMeshProcess` でメッシュを組み、`InpToDarcyProcess` が `DarcyFlowInput` に写す。
出力は非構造版（NPZ に `node_coords` / `connectivity`、VTK は `UNSTRUCTURED_GRID`、HTML は
`MiradorExportProcess` の `mesh=` 入力）。

## 制限 / 今後

- ~~定常のみ。Forchheimer 慣性補正は未実装~~ → 比貯留項の非定常と Forchheimer の Picard 反復を追加（2026-09-06）。
  Brinkman 粘性項（速度が未知数になる）はこのファミリーには入れず、`NavierStokesFVMProcess(permeability=)` の
  Brinkman 抵抗で扱う
- ~~非直交補正なし~~ → over-relaxed 分解の遅延補正を `solve_corrected` で反復（[fvm-layer.md](fvm-layer.md)）。`DarcyFlowInput.max_nonorthogonal_iter`、`DarcyFlowResult.n_nonorthogonal_iter`。質量不整合は tol × 流量のオーダー
