# 面ベース FVM 共通低レイヤー（`xkep_cae_fluid.fvm`）設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md) | [← roadmap](../roadmap.md) | [棚卸しと分離計画](../plans/2026-09-05-solver-layering.md)

## 位置付け

ykep のソルバーを 3 層に分ける。

| 層 | 役割 | 代表 |
|---|---|---|
| **Geo 層**（PreProcess） | 幾何・領域・境界パッチを `MeshData` に載せる | `StructuredMeshProcess`、`PolyMeshReaderProcess`、`InpMeshProcess` |
| **Solver 低レイヤー**（本文書） | 面リストの上の離散化演算子・境界条件・線形ソルバー | `xkep_cae_fluid.fvm` |
| **方程式ファミリー**（SolverProcess） | 低レイヤーを組み合わせる薄い層 | `ScalarTransportFVMProcess`、`DarcyFlowProcess` |

構造格子（`StructuredMeshProcess`）も polyMesh も `.inp` の六面体メッシュも、同じ `MeshData`
（内部面 → 境界面の順、`boundary_patches`）に落ちるので、方程式ファミリーは格子の種類を知らない。

## モジュール

### `fvm/boundary.py` — 境界パッチ条件

- `BCKind`: `ZERO_GRADIENT` / `DIRICHLET` / `NEUMANN` / `ROBIN`
- `PatchBC(kind, value, flux, h, phi_inf)`: 1 パッチの条件。`value` / `flux` は面ごとの配列でも可
- `resolve_boundary(mesh, {patch_name: PatchBC}, default=...) -> BoundaryFaces`
  — パッチ名を境界面ごとの配列（種別コード、値、面積、法線距離 d_b）に展開する。
  メッシュに無いパッチ名は `KeyError`、未指定パッチは `default`（既定はゼロ勾配）

離散化（owner P、境界面 b、面積 A_b、セル中心から面中心までの法線距離 d_b）:

| 種別 | 対角 | 右辺 |
|---|---|---|
| Dirichlet φ_b | Γ_P A_b / d_b | Γ_P A_b φ_b / d_b |
| Neumann（Γ ∂φ/∂n_in = flux、正 = 流入） | — | flux · A_b |
| Robin J = h (φ_inf − φ_b) | U A_b、U = Γ_P h / (Γ_P + h d_b) | U A_b φ_inf |
| ゼロ勾配 | — | — |

構造格子の既存 FDM（`2Γ/d²`、`U_eff = 2Γh/(2Γ + hd)`）とは d_b = d/2 で一致する。

### `fvm/geometry.py` — 幾何演算

- `face_interpolation_weights(mesh)`: 内部面の距離重み（等間隔で 0.5）
- `face_diffusivity(mesh, gamma)`: 内部面の調和平均
- `face_mass_flux(mesh, velocity, rho, blocked_cells=, boundary_normal_velocity=)`:
  ṁ_f = ρ (u_f·n_f) A_f（全面）。固体セルに接する面はゼロ
- `internal_face_values` / `boundary_face_values`: 面の φ
- `cell_gradient(mesh, phi, bfaces, gamma=)`: Green–Gauss 勾配（線形場で厳密）

### `fvm/assembly.py` — 係数行列

体積積分形（行 = Σ_f F_f − S V_P）で組む。

- `assemble_diffusion(mesh, gamma, bfaces) -> (A, b)`: 内部面 a_f = Γ_f A_f / d_PN（`core/strategies` の中心差分と同じ）+ 境界
- `assemble_convection(mesh, mass_flux, bfaces) -> (A, b)`: 1 次風上。境界は流出 → φ_P、流入 → Dirichlet なら φ_b（右辺）、それ以外は φ_P
- `assemble_scalar_transport(mesh, gamma=, bfaces=, mass_flux=, source=, rho=, dt=, phi_old=)`: 上 2 つ + ソース S V_P + 陰的 Euler ρ V_P / dt

### 非直交補正（`geometry.face_decomposition` / `assembly.nonorthogonal_correction`）

内部面の面ベクトル S_f = n_f A_f を over-relaxed 分解 S_f = E_f + T_f する
（E_f ∥ e_f = (x_N − x_P)/d_PN、|E_f| = A_f/(n_f·e_f)、T_f = S_f − E_f）。

- 陰的部分: `assemble_diffusion` の係数 a_f = Γ_f |E_f|/d_PN（直交メッシュでは A_f/d_PN）
- 陽的部分（遅延補正）: `nonorthogonal_correction(mesh, φ, Γ, bfaces)` が
  Γ_f (∇φ)_f·T_f を owner に +、neighbour に − で右辺に足す。(∇φ)_f は Green–Gauss セル勾配
  `cell_gradient`（Dirichlet 以外の境界面では φ_b に接線外挿 ∇φ_P·t_b を加え 2 回反復）の線形補間
- 傾いた境界面（Dirichlet / Robin）: 法線勾配を (φ_b − φ_P − ∇φ_P·t_b)/d_b と評価する接線補正
  −c_b (∇φ_P·t_b)（`boundary_tangent`、c_b = Γ_P A_b/d_b または U A_b）
- `diffusive_face_flux(mesh, φ, Γ, bfaces)` は同じ分解で全面の J_f = −Γ∇φ·S_f を返す
  （Darcy の面流量、熱収支の検査に使う）
- `solve_corrected(mesh, build, solver, φ0, max_iter, tol)` が「補正を前回の φ で評価 → 解く」を
  ‖Δφ‖/‖φ‖ < tol まで反復する。`is_orthogonal(mesh)` なら 1 回で終わる。
  各方程式ファミリーの Input に `max_nonorthogonal_iter`（既定 20）がある

一様せん断した六面体メッシュでは、全面 Dirichlet の線形場が補正の反復で厳密に再現され、
面フラックスも厳密（`tests/test_fvm_layer.py::TestNonorthogonalPhysics`）。
補正は遅延評価なので、収束解の質量不整合（Darcy）は tol × 流量のオーダーになる。

### `fvm/momentum.py` — 運動量・圧力連成カーネル

`VelocityPatchBC`（WALL / INLET / OUTLET / SLIP）→ `resolve_velocity_boundary`、成分ごとの運動量行列
`assemble_momentum`（対流・拡散 + 非直交補正・時間項・圧力勾配・抵抗・陰的緩和）、Rhie–Chow 面質量流束
`rhie_chow_mass_flux`、圧力補正 `pressure_correction_coefficients` / `assemble_pressure_correction` /
`correct_mass_flux`。圧力勾配には最小二乗勾配 `geometry.cell_gradient_lsq`（境界セルでも線形場で厳密）を使う。
詳細は [navier-stokes-fvm.md](navier-stokes-fvm.md)。

### `fvm/linear.py` — 線形ソルバー Strategy

`LinearSolverStrategy` Protocol（`solve(A, b)`）の具象:
`DirectSolver`（spsolve）、`BiCGSTABSolver`（ILU 前処理）、`AMGSolver`（PyAMG、行列構造でキャッシュ）、
`make_linear_solver(name)`。`heat_transfer` / `natural_convection` / `scalar_transport` に散っていた
同種のラッパーの置き換え先。

## 方程式ファミリーの書き方（`ScalarTransportFVMProcess`）

```python
from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.fvm import PatchBC
from xkep_cae_fluid.scalar_transport.fvm import ScalarTransportFVMInput, ScalarTransportFVMProcess

mesh = StructuredMeshProcess().execute(StructuredMeshInput(Lx=1, Ly=1, Lz=1, nx=20, ny=10, nz=5)).mesh
res = ScalarTransportFVMProcess().execute(
    ScalarTransportFVMInput(
        mesh=mesh,
        phi0=np.zeros(mesh.n_cells),
        diffusivity=1e-3,
        velocity=u_cells,               # (n_cells, 3)、または mass_flux=(n_faces,)
        bcs={"XM": PatchBC.dirichlet(1.0), "XP": PatchBC.zero_gradient(), "ZP": PatchBC.robin(h, phi_inf)},
        dt=0.1, t_end=5.0, linear_solver="bicgstab",
    )
)
```

`ScalarTransportFVMResult.phi` は `(n_cells,)`。構造格子なら `reshape(nx, ny, nz)`
（セル添字は i 最遅・k 最速で既存ソルバーの `ravel()` と一致）。

既存の構造格子版 `ScalarTransportProcess`（FDM）との回帰は `tests/test_scalar_transport_fvm.py`
で取る: 同じ箱格子・同じ面速度（内部面はセル平均、境界面ゼロ）・同じ境界条件で定常・非定常ともに
解が一致する（相対 1e-8）。既存 FDM は境界面の対流項を持たないため、境界を横切る流れがある問題では
FVM 側（境界流出入を含む）とは一致しない。

## 既存ソルバーからの移行方針

1. `ScalarTransportFVMProcess`（本文書、パイロット）
2. `DarcyFlowProcess`（[darcy-flow-fvm.md](darcy-flow-fvm.md)、新ファミリー、最初の非構造ケース）
3. `HeatTransferFVMProcess`（[heat-transfer-fvm.md](heat-transfer-fvm.md)、構造格子 FDM と 1e-8 で一致、
   `*HEAT TRANSFER` の非箱格子 / `--mesh=unstructured` 経路）— 済
4. `NavierStokesFVMProcess`（[navier-stokes-fvm.md](navier-stokes-fvm.md)）— `BrinkmanFlowFVMProcess` の
   Brinkman 抵抗と `NaturalConvectionFDMProcess` の SIMPLE + Rhie–Chow + Boussinesq + 固体マスクを
   1 ファミリーに。運動量・圧力連成カーネルは `fvm/momentum.py` — 済（1 次風上、SIMPLE/SIMPLEC）
5. 構造格子版に残る機能（TVD 遅延補正、BDF2、PISO、`InternalFaceBC`、追加スカラー）の移植

## テスト

- `tests/test_fvm_layer.py`: 1D 拡散の線形解（等間隔・不等間隔）、Neumann の勾配、Robin の平衡と
  既存 FDM 係数との一致、2 領域物性の直列抵抗、風上の一様流輸送、対流拡散の単調性、時間項・ソース項、
  Green–Gauss 勾配の厳密性、線形ソルバー 3 種の一致、非直交分解（直交で退化、せん断で角度）、
  せん断メッシュの線形場と面フラックスの厳密性
- `tests/test_scalar_transport_fvm.py`: `ScalarTransportFVMProcess` の API と構造格子 FDM との回帰
- `tests/test_heat_transfer_fvm.py`: `HeatTransferFVMProcess` の API、FDM との回帰、せん断メッシュの線形場・熱収支
- `tests/test_navier_stokes_fvm.py`: `NavierStokesFVMProcess` の API、Poiseuille（箱 / せん断）、Brinkman 流路、
  蓋駆動キャビティ Re=100、差分加熱キャビティ Ra=10³
