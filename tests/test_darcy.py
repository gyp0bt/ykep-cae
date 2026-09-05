"""DarcyFlowProcess のテスト（API + 物理）."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.darcy import (
    DarcyBCKind,
    DarcyFlowInput,
    DarcyFlowProcess,
    DarcyFlowResult,
    DarcyPatchBC,
)
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess
from xkep_cae_fluid.inp.parser import parse_inp_text

K, MU = 2.0e-10, 1.0e-3


def _box(nx=10, ny=2, nz=2, Lx=1.0, **kw):
    return (
        StructuredMeshProcess()
        .execute(StructuredMeshInput(Lx=Lx, Ly=0.2, Lz=0.2, nx=nx, ny=ny, nz=nz, **kw))
        .mesh
    )


@binds_to(DarcyFlowProcess)
class TestDarcyFlowAPI:
    def test_meta(self):
        assert DarcyFlowProcess.meta.name == "DarcyFlowFVM"
        assert DarcyFlowProcess.meta.module == "solve"

    def test_returns_result(self):
        mesh = _box()
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(
                mesh=mesh,
                permeability=K,
                viscosity=MU,
                bcs={"XM": DarcyPatchBC.pressure_bc(10.0), "XP": DarcyPatchBC.pressure_bc(0.0)},
            )
        )
        assert isinstance(res, DarcyFlowResult)
        assert res.p.shape == (mesh.n_cells,)
        assert res.velocity.shape == (mesh.n_cells, 3)
        assert res.face_flux.shape == (mesh.n_faces,)
        assert res.converged and res.residual < 1e-10

    def test_validation(self):
        mesh = _box(4, 1, 1)
        pbc = {"XM": DarcyPatchBC.pressure_bc(1.0)}
        with pytest.raises(ValueError, match="圧力の基準"):
            DarcyFlowProcess().execute(DarcyFlowInput(mesh=mesh, permeability=K, viscosity=MU))
        with pytest.raises(ValueError, match="permeability"):
            DarcyFlowProcess().execute(
                DarcyFlowInput(mesh=mesh, permeability=np.zeros(4), viscosity=MU, bcs=pbc)
            )
        with pytest.raises(ValueError, match="viscosity"):
            DarcyFlowProcess().execute(
                DarcyFlowInput(mesh=mesh, permeability=K, viscosity=0.0, bcs=pbc)
            )
        with pytest.raises(KeyError):
            DarcyFlowProcess().execute(
                DarcyFlowInput(
                    mesh=mesh,
                    permeability=K,
                    viscosity=MU,
                    bcs={"INLET": DarcyPatchBC.pressure_bc(1.0)},
                )
            )
        assert DarcyPatchBC.wall().kind == DarcyBCKind.WALL


class TestDarcyFlowPhysics:
    def test_1d_pressure_drop_uniform_velocity(self):
        """圧力差 Δp の 1D Darcy 流れ: u = K Δp/(μ L)、p は線形、質量保存."""
        mesh = _box(nx=10)
        dp = 500.0
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(
                mesh=mesh,
                permeability=K,
                viscosity=MU,
                bcs={"XM": DarcyPatchBC.pressure_bc(dp), "XP": DarcyPatchBC.pressure_bc(0.0)},
            )
        )
        u_exact = K * dp / (MU * 1.0)
        np.testing.assert_allclose(res.velocity[:, 0], u_exact, rtol=1e-10)
        np.testing.assert_allclose(res.velocity[:, 1:], 0.0, atol=1e-12 * u_exact)
        np.testing.assert_allclose(res.p, dp * (1.0 - mesh.cell_centers[:, 0]), rtol=1e-10)
        np.testing.assert_allclose(res.mass_residual, 0.0, atol=1e-14 * u_exact)
        area = 0.2 * 0.2
        np.testing.assert_allclose(res.inflow, u_exact * area, rtol=1e-10)
        np.testing.assert_allclose(res.outflow, u_exact * area, rtol=1e-10)

    def test_velocity_inlet_gives_linear_pressure(self):
        """流入速度指定 + 出口圧力 0: p = μ u (L − x)/K、流出 = 流入."""
        mesh = _box(nx=8, Lx=2.0)
        u_in = 1.5e-4
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(
                mesh=mesh,
                permeability=K,
                viscosity=MU,
                bcs={"XM": DarcyPatchBC.velocity_bc(u_in), "XP": DarcyPatchBC.pressure_bc(0.0)},
            )
        )
        np.testing.assert_allclose(
            res.p, MU * u_in * (2.0 - mesh.cell_centers[:, 0]) / K, rtol=1e-10
        )
        np.testing.assert_allclose(res.velocity[:, 0], u_in, rtol=1e-10)
        np.testing.assert_allclose(res.outflow, res.inflow, rtol=1e-10)

    def test_two_layer_permeability_series(self):
        """透過率が 2 層で異なると、流量は直列則 1/K_eff = (L1/K1 + L2/K2)/L."""
        mesh = _box(nx=10)
        k = np.where(mesh.cell_centers[:, 0] < 0.5, K, 4 * K)
        dp = 100.0
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(
                mesh=mesh,
                permeability=k,
                viscosity=MU,
                bcs={"XM": DarcyPatchBC.pressure_bc(dp), "XP": DarcyPatchBC.pressure_bc(0.0)},
            )
        )
        k_eff = 1.0 / (0.5 / K + 0.5 / (4 * K))
        u_exact = k_eff * dp / MU
        np.testing.assert_allclose(res.velocity[:, 0], u_exact, rtol=1e-10)
        np.testing.assert_allclose(res.mass_residual, 0.0, atol=1e-14 * u_exact)

    def test_source_term_total_outflow(self):
        """一様ソース S の総流出 = S × 体積（全面 p=0）."""
        mesh = _box(nx=4, ny=4, nz=4)
        S = 3.0e-3
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(
                mesh=mesh,
                permeability=K,
                viscosity=MU,
                bcs={n: DarcyPatchBC.pressure_bc(0.0) for n in mesh.boundary_patches},
                source=np.full(mesh.n_cells, S),
            )
        )
        np.testing.assert_allclose(
            res.outflow - res.inflow, S * mesh.cell_volumes.sum(), rtol=1e-10
        )
        np.testing.assert_allclose(res.mass_residual, 0.0, atol=1e-12 * S)

    def test_sheared_inp_mesh_gives_same_flow(self):
        """せん断で歪んだ .inp 六面体メッシュ（非直交）でも一様流 u = K Δp/(μ L) が厳密に出る.

        x += 0.15 y のせん断で XM/XP 面が傾くので、面ごとに p = Δp(1 − x) を Dirichlet で与える
        （YM/YP は水平のまま = 不透過壁と整合）。非直交補正の反復で線形圧力・一様速度に収束する。
        """
        nx, ny, nz = 6, 2, 2
        text = _lattice_text(nx, ny, nz, 1.0 / nx, 0.1, 0.1)
        case = build_case(parse_inp_text(text))
        coords = case.nodes.coords.copy()
        coords[:, 0] += 0.15 * coords[:, 1]
        case = replace(case, nodes=replace(case.nodes, coords=coords))
        mesh = InpMeshProcess().execute(InpMeshInput(case=case)).mesh
        dp = 200.0
        bcs = {
            name: DarcyPatchBC.pressure_bc(
                dp * (1.0 - mesh.face_centers[mesh.patch_faces(name), 0])
            )
            for name in ("XM", "XP")
        }
        res = DarcyFlowProcess().execute(
            DarcyFlowInput(mesh=mesh, permeability=K, viscosity=MU, bcs=bcs, tol=1e-12)
        )
        u_exact = K * dp / MU
        assert res.n_nonorthogonal_iter > 1
        np.testing.assert_allclose(res.p, dp * (1.0 - mesh.cell_centers[:, 0]), rtol=1e-8)
        np.testing.assert_allclose(res.velocity[:, 0], u_exact, rtol=1e-8)
        np.testing.assert_allclose(res.velocity[:, 1:], 0.0, atol=1e-8 * u_exact)
        np.testing.assert_allclose(res.inflow, res.outflow, rtol=1e-10)
        np.testing.assert_allclose(res.mass_residual, 0.0, atol=1e-12 * u_exact)


def _lattice_text(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> str:
    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    lines = ["*NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f" {nid(i, j, k)}, {i * dx}, {j * dy}, {k * dz}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALL")
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e += 1
                c = [
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]
                lines.append(f" {e}, " + ", ".join(map(str, c)))
    return "\n".join(lines) + "\n"
