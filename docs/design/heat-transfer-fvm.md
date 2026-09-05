# 伝熱ソルバー（面ベース FVM）設計仕様 — `HeatTransferFVMProcess`

[← README](../../README.md) | [設計文書一覧](README.md) | [FVM 層の設計](fvm-layer.md) | [構造格子 FDM 版](heat-transfer-fdm.md) | [棚卸しと計画](../plans/2026-09-05-solver-layering.md)

## 位置付け

`HeatTransferFDMProcess`（構造格子、`(nx, ny, nz)` 配列、6 面固定の境界条件）を
面ベース FVM 共通低レイヤー [`xkep_cae_fluid.fvm`](fvm-layer.md) の上に載せ替えたもの。
`MeshData`（構造格子 / polyMesh / `.inp` 由来の任意六面体・四辺形押し出し）を同じ経路で解き、
境界条件はパッチ名 → `PatchBC` で与える。FDM 版に散っていた境界係数式（Dirichlet `2k/d²`、
Neumann `q/d`、Robin `2kh/(2k+hd)`）と調和平均は低レイヤーに 1 か所だけ置く。

`.inp` の `*HEAT TRANSFER` は、メッシュが軸平行の箱格子なら従来どおり FDM 版、
箱格子でなければ（または `ykep --mesh=unstructured`）本プロセスで解く（[inp-format.md](inp-format.md)）。

## 支配方程式と離散化

ρC ∂T/∂t − ∇·(k∇T) = q を体積積分形で陰的 Euler:

    (ρC V_P / Δt)(T_P − T_P^old) + Σ_f k_f A_f (T_P − T_N)/d_PN = q V_P + 境界寄与

- 内部面: k_f は owner / neighbour の調和平均、d_PN はセル中心間距離（非直交補正なし、現状）
- 境界面（owner セル P、距離 d_b）:
  - Dirichlet `PatchBC.dirichlet(T_w)`: k_P A_b/d_b を対角、k_P A_b T_w/d_b を右辺
  - Neumann `PatchBC.neumann(q)`（q [W/m²] 正 = 流入）: q A_b を右辺
  - Robin `PatchBC.robin(h, T_inf)`: U = k_P h/(k_P + h d_b)、U A_b を対角、U A_b T_inf を右辺
  - 未指定 / `zero_gradient`: 断熱

構造格子では d_b = d/2 なので FDM 版と同じ係数（行ごとの体積倍を除く）になる。

## 入出力

| | 内容 |
|---|---|
| `HeatTransferFVMInput` | `mesh`、`conductivity`（スカラー / `(n_cells,)`）、`T0`、`heat_capacity`（ρC、非定常用）、`heat_source`（W/m³）、`bcs`、`dt`/`t_end`/`output_interval`、`linear_solver`（direct / bicgstab / amg）、`tol`、`max_iter` |
| `HeatTransferFVMResult` | `T (n_cells,)`、`converged`、`n_timesteps`、`residual_history`（各ソルブの相対残差）、`time_history` / `T_history`、`elapsed_seconds`、`residual_fields["res_T"]`（定常のみ、\|b − A T\|/‖b‖） |

## 使い方

```python
from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.fvm import PatchBC
from xkep_cae_fluid.heat_transfer.fvm import HeatTransferFVMInput, HeatTransferFVMProcess

mesh = StructuredMeshProcess().execute(StructuredMeshInput(Lx=0.2, Ly=0.1, Lz=0.05, nx=20, ny=10, nz=5)).mesh
res = HeatTransferFVMProcess().execute(
    HeatTransferFVMInput(
        mesh=mesh, conductivity=200.0, T0=np.full(mesh.n_cells, 300.0),
        bcs={"XM": PatchBC.dirichlet(350.0), "XP": PatchBC.neumann(2000.0), "ZP": PatchBC.robin(25.0, 300.0)},
        heat_source=q_cells, linear_solver="direct",
    )
)
T = res.T.reshape(20, 10, 5)  # 構造格子なら i 最遅・k 最速で FDM 版の配列順と一致
```

## テスト（`tests/test_heat_transfer_fvm.py`）

- API: メタ情報、入力検証（長さ不一致、非正の k）、定常の `res_T`、非定常の履歴
- 回帰: 同じ箱格子（等間隔・不等間隔）、同じ物性分布・発熱・Dirichlet/Neumann/Robin/断熱で
  `HeatTransferFDMProcess(method="direct")` と定常・非定常ともに相対 1e-8 で一致
- 物理: せん断した非構造六面体メッシュ（`InpMeshProcess`）で 1D 定常熱伝導が線形分布、
  Robin 壁の平衡温度、発熱ありの全体熱収支（Σ q V = Σ 境界流出）

## 制限と TODO

- 非直交補正なし（歪んだメッシュでは中心差分の交差拡散項を落としている。roadmap Phase 11）
- 温度依存物性、輻射境界は未対応（FDM 版と同じ）
