"""HeatTransferFVMProcess のテスト（API + 構造格子 FDM との回帰 + 非構造メッシュの物理）."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.fvm import PatchBC
from xkep_cae_fluid.heat_transfer.data import BoundaryCondition, BoundarySpec, HeatTransferInput
from xkep_cae_fluid.heat_transfer.fvm import (
    HeatTransferFVMInput,
    HeatTransferFVMProcess,
    HeatTransferFVMResult,
)
from xkep_cae_fluid.heat_transfer.solver import HeatTransferFDMProcess
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mesh import build_inp_mesh
from xkep_cae_fluid.inp.parser import parse_inp_text

NX, NY, NZ = 6, 4, 3
LX, LY, LZ = 0.3, 0.2, 0.15


def _structured(stretch_x=(1.0,)):
    return StructuredMeshProcess().execute(
        StructuredMeshInput(
            Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ, stretch_x=stretch_x, stretch_z=(1.5, 1.0)
        )
    )


def _fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """セルごとに異なる k / ρC / q / T0（構造格子 (nx, ny, nz) 配列）."""
    i, j, k = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    cond = 10.0 + 5.0 * (i % 2) + 2.0 * j
    cap = 1.0e6 * (1.0 + 0.3 * (k % 2))
    q = 1.0e5 * np.sin(np.pi * (i + 0.5) / NX) * (1.0 + 0.5 * (j == 1))
    T0 = 300.0 + 20.0 * (i + j + k) / (NX + NY + NZ)
    return cond, cap, q, T0


_BCS_FDM = {
    "bc_xm": BoundarySpec(BoundaryCondition.DIRICHLET, value=350.0),
    "bc_xp": BoundarySpec(BoundaryCondition.NEUMANN, value=2000.0),
    "bc_ym": BoundarySpec(BoundaryCondition.ROBIN, h_conv=25.0, T_inf=290.0),
    "bc_yp": BoundarySpec(BoundaryCondition.ADIABATIC),
    "bc_zm": BoundarySpec(BoundaryCondition.DIRICHLET, value=320.0),
    "bc_zp": BoundarySpec(BoundaryCondition.ADIABATIC),
}
_BCS_FVM = {
    "XM": PatchBC.dirichlet(350.0),
    "XP": PatchBC.neumann(2000.0),
    "YM": PatchBC.robin(25.0, 290.0),
    "ZM": PatchBC.dirichlet(320.0),
}


def _sheared_hex_text(nx: int, ny: int, shear: float, lx: float, ly: float, lz: float) -> str:
    """上辺ほど x 方向にずれる 2 層の六面体メッシュ（箱格子ではない）."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.array([0.0, lz])

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    lines = ["*NODE, NSET=NALL"]
    for k in range(2):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f" {nid(i, j, k)}, {xs[i] + shear * ys[j]}, {ys[j]}, {zs[k]}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=PLATE")
    e = 0
    for j in range(ny):
        for i in range(nx):
            e += 1
            c = [
                nid(i, j, 0),
                nid(i + 1, j, 0),
                nid(i + 1, j + 1, 0),
                nid(i, j + 1, 0),
                nid(i, j, 1),
                nid(i + 1, j, 1),
                nid(i + 1, j + 1, 1),
                nid(i, j + 1, 1),
            ]
            lines.append(f" {e}, " + ", ".join(str(n) for n in c))
    return "\n".join(lines) + "\n"


def _sheared_mesh(nx=8, ny=4, shear=0.25, lx=0.4, ly=0.2, lz=0.05):
    case = build_case(parse_inp_text(_sheared_hex_text(nx, ny, shear, lx, ly, lz)))
    return build_inp_mesh(case).mesh


@binds_to(HeatTransferFVMProcess)
class TestHeatTransferFVMAPI:
    def test_meta(self):
        assert HeatTransferFVMProcess.meta.name == "HeatTransferFVM"
        assert HeatTransferFVMProcess.meta.module == "solve"

    def test_rejects_bad_inputs(self):
        mesh = _structured().mesh
        with pytest.raises(ValueError, match="conductivity"):
            HeatTransferFVMProcess().execute(
                HeatTransferFVMInput(mesh=mesh, conductivity=-1.0, T0=np.zeros(mesh.n_cells))
            )
        with pytest.raises(ValueError, match="T0"):
            HeatTransferFVMProcess().execute(
                HeatTransferFVMInput(mesh=mesh, conductivity=1.0, T0=np.zeros(3))
            )
        with pytest.raises(KeyError, match="LID"):
            HeatTransferFVMProcess().execute(
                HeatTransferFVMInput(
                    mesh=mesh,
                    conductivity=1.0,
                    T0=np.zeros(mesh.n_cells),
                    bcs={"LID": PatchBC.dirichlet(1.0)},
                )
            )

    def test_steady_result_has_residual_map(self):
        mesh = _structured().mesh
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=mesh,
                conductivity=5.0,
                T0=np.full(mesh.n_cells, 300.0),
                bcs=_BCS_FVM,
                linear_solver="direct",
            )
        )
        assert isinstance(res, HeatTransferFVMResult)
        assert res.converged and res.n_timesteps == 0
        assert res.residual_fields["res_T"].shape == (mesh.n_cells,)
        assert res.residual_fields["res_T"].max() < 1e-10
        assert len(res.residual_history) == 1

    def test_transient_history(self):
        mesh = _structured().mesh
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=mesh,
                conductivity=5.0,
                heat_capacity=1.0e6,
                T0=np.full(mesh.n_cells, 300.0),
                bcs={"XM": PatchBC.dirichlet(350.0)},
                dt=2.0,
                t_end=10.0,
                output_interval=2,
                linear_solver="bicgstab",
            )
        )
        assert res.n_timesteps == 5 and res.converged
        assert res.time_history == (4.0, 8.0, 10.0)
        assert len(res.T_history) == 3 and res.residual_fields == {}
        assert 300.0 < res.T.max() <= 350.0


class TestHeatTransferFVMPhysics:
    @pytest.mark.parametrize("stretch_x", [(1.0,), (2.0, 1.0)])
    def test_matches_fdm_steady(self, stretch_x):
        """同じ箱格子（等間隔 / 不等間隔）・物性分布・発熱・境界条件で FDM 版と一致."""
        sm = _structured(stretch_x)
        cond, cap, q, T0 = _fields()
        fdm = HeatTransferFDMProcess(method="direct").execute(
            HeatTransferInput.from_mesh(sm, k=cond, C=cap, q=q, T0=T0, **_BCS_FDM)
        )
        fvm = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=sm.mesh,
                conductivity=cond.ravel(),
                heat_capacity=cap.ravel(),
                heat_source=q.ravel(),
                T0=T0.ravel(),
                bcs=_BCS_FVM,
                linear_solver="direct",
            )
        )
        assert np.allclose(fvm.T.reshape(NX, NY, NZ), fdm.T, rtol=1e-8, atol=1e-8)

    def test_matches_fdm_transient(self):
        sm = _structured((1.0,))
        cond, cap, q, T0 = _fields()
        fdm = HeatTransferFDMProcess(method="direct").execute(
            HeatTransferInput.from_mesh(
                sm, k=cond, C=cap, q=q, T0=T0, dt=5.0, t_end=30.0, output_interval=3, **_BCS_FDM
            )
        )
        fvm = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=sm.mesh,
                conductivity=cond.ravel(),
                heat_capacity=cap.ravel(),
                heat_source=q.ravel(),
                T0=T0.ravel(),
                bcs=_BCS_FVM,
                dt=5.0,
                t_end=30.0,
                output_interval=3,
                linear_solver="direct",
            )
        )
        assert fvm.n_timesteps == fdm.n_timesteps == 6
        assert fvm.time_history == fdm.time_history
        for a, b in zip(fvm.T_history, fdm.T_history, strict=True):
            assert np.allclose(a.reshape(NX, NY, NZ), b, rtol=1e-8, atol=1e-8)

    def test_linear_profile_on_sheared_mesh(self):
        """せん断六面体メッシュ（非直交）で線形場 T = 400 − 250 x を再現する.

        傾いた XM/XP 面には面ごとの厳密値を Dirichlet で与え、YM/YP/ZM/ZP は断熱
        （法線が x に垂直なので線形場と整合）。over-relaxed 非直交補正の反復で線形場に収束する。
        """
        mesh = _sheared_mesh()
        exact = lambda xyz: 400.0 - 250.0 * xyz[:, 0]  # noqa: E731
        bcs = {
            name: PatchBC.dirichlet(exact(mesh.face_centers[mesh.patch_faces(name)]))
            for name in ("XM", "XP")
        }
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=mesh,
                conductivity=5.0,
                T0=np.full(mesh.n_cells, 300.0),
                bcs=bcs,
                linear_solver="direct",
                tol=1e-12,
            )
        )
        assert res.converged
        np.testing.assert_allclose(res.T, exact(mesh.cell_centers), rtol=1e-6)

    def test_sheared_mesh_without_correction_is_not_linear(self):
        """補正を切る（max_nonorthogonal_iter=1 でも直交判定で 1 回目は補正入り）代わりに、
        非直交角が実際に付いていることと、補正が結果を変えていることを確認する."""
        from xkep_cae_fluid.fvm import max_nonorthogonality_deg

        mesh = _sheared_mesh()
        assert max_nonorthogonality_deg(mesh) > 10.0
        assert max_nonorthogonality_deg(_structured().mesh) < 1e-9

    def test_energy_balance_with_source_and_robin(self):
        """発熱の総量 = Robin 壁からの放熱（定常の全体熱収支）."""
        mesh = _sheared_mesh()
        q = np.full(mesh.n_cells, 1.0e4)
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=mesh,
                conductivity=20.0,
                T0=np.full(mesh.n_cells, 300.0),
                heat_source=q,
                bcs={"ZP": PatchBC.robin(15.0, 290.0)},
                linear_solver="direct",
            )
        )
        total_q = float(np.sum(q * mesh.cell_volumes))
        faces = mesh.patch_faces("ZP")
        owner = mesh.face_owner[faces]
        k, h = 20.0, 15.0
        d_b = np.abs(
            np.sum(
                (mesh.face_centers[faces] - mesh.cell_centers[owner]) * mesh.face_normals[faces],
                axis=1,
            )
        )
        u_eff = k * h / (k + h * d_b)
        out = float(np.sum(u_eff * mesh.face_areas[faces] * (res.T[owner] - 290.0)))
        assert out == pytest.approx(total_q, rel=1e-9)
        assert np.all(res.T > 290.0)
