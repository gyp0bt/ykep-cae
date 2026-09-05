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
3. `HeatTransferFDMProcess` → 面カーネル版（境界条件式 5 か所・調和平均 5 か所の複製を吸収）
4. `BrinkmanFlowFVMProcess`（演算子合成 `Dx@diag(f)@W` を owner/neighbour で組み直す）
5. `NaturalConvectionFDMProcess`（SIMPLE、Rhie–Chow を面リストで）

## テスト

- `tests/test_fvm_layer.py`: 1D 拡散の線形解（等間隔・不等間隔）、Neumann の勾配、Robin の平衡と
  既存 FDM 係数との一致、2 領域物性の直列抵抗、風上の一様流輸送、対流拡散の単調性、時間項・ソース項、
  Green–Gauss 勾配の厳密性、線形ソルバー 3 種の一致
- `tests/test_scalar_transport_fvm.py`: `ScalarTransportFVMProcess` の API と構造格子 FDM との回帰
