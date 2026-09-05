"""ScalarTransportFVMProcess のテスト（API + 構造格子 FDM との回帰）."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.fvm import PatchBC
from xkep_cae_fluid.scalar_transport.data import (
    ScalarBoundaryCondition,
    ScalarBoundarySpec,
    ScalarFieldSpec,
    ScalarTransportInput,
)
from xkep_cae_fluid.scalar_transport.fvm import (
    ScalarTransportFVMInput,
    ScalarTransportFVMProcess,
    ScalarTransportFVMResult,
)
from xkep_cae_fluid.scalar_transport.solver import ScalarTransportProcess

NX, NY, NZ = 5, 4, 3
LX, LY, LZ = 1.0, 0.8, 0.6


def _mesh():
    return (
        StructuredMeshProcess()
        .execute(StructuredMeshInput(Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ))
        .mesh
    )


def _velocity_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑らかな速度場（構造格子 (nx, ny, nz) 配列）."""
    x = (np.arange(NX) + 0.5) / NX * LX
    y = (np.arange(NY) + 0.5) / NY * LY
    z = (np.arange(NZ) + 0.5) / NZ * LZ
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u = 0.3 * np.sin(np.pi * X / LX) * np.cos(np.pi * Y / LY)
    v = -0.2 * np.cos(np.pi * X / LX) * np.sin(np.pi * Y / LY)
    w = 0.1 * Z / LZ
    return u, v, w


def _structured_face_flux(mesh, u, v, w, rho) -> np.ndarray:
    """既存 FDM と同じ面速度（内部面はセル平均、境界面はゼロ）を面質量流束にする."""
    n_int = mesh.n_internal_faces
    vel = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)
    owner = mesh.face_owner[:n_int]
    nb = mesh.face_neighbour
    u_face = 0.5 * (vel[owner] + vel[nb])
    mf = np.zeros(mesh.n_faces)
    mf[:n_int] = rho * np.sum(u_face * mesh.face_normals[:n_int], axis=1) * mesh.face_areas[:n_int]
    return mf


_BCS_FDM = {
    "bc_xm": ScalarBoundarySpec(ScalarBoundaryCondition.DIRICHLET, value=1.0),
    "bc_xp": ScalarBoundarySpec(ScalarBoundaryCondition.NEUMANN, flux=0.4),
    "bc_ym": ScalarBoundarySpec(ScalarBoundaryCondition.ROBIN, h_mass=0.7, phi_inf=2.0),
    "bc_yp": ScalarBoundarySpec(ScalarBoundaryCondition.ADIABATIC),
    "bc_zm": ScalarBoundarySpec(ScalarBoundaryCondition.DIRICHLET, value=0.0),
    "bc_zp": ScalarBoundarySpec(ScalarBoundaryCondition.ADIABATIC),
}
_BCS_FVM = {
    "XM": PatchBC.dirichlet(1.0),
    "XP": PatchBC.neumann(0.4),
    "YM": PatchBC.robin(0.7, 2.0),
    "ZM": PatchBC.dirichlet(0.0),
}


@binds_to(ScalarTransportFVMProcess)
class TestScalarTransportFVMAPI:
    def test_meta(self):
        assert ScalarTransportFVMProcess.meta.name == "ScalarTransportFVM"
        assert ScalarTransportFVMProcess.meta.module == "solve"

    def test_returns_result_and_shape(self):
        mesh = _mesh()
        res = ScalarTransportFVMProcess().execute(
            ScalarTransportFVMInput(
                mesh=mesh, phi0=np.zeros(mesh.n_cells), diffusivity=1.0, bcs=_BCS_FVM
            )
        )
        assert isinstance(res, ScalarTransportFVMResult)
        assert res.phi.shape == (mesh.n_cells,)
        assert res.converged
        assert res.n_timesteps == 0

    def test_bad_phi0_length(self):
        mesh = _mesh()
        with pytest.raises(ValueError, match="n_cells"):
            ScalarTransportFVMProcess().execute(
                ScalarTransportFVMInput(mesh=mesh, phi0=np.zeros(3), diffusivity=1.0)
            )

    def test_unknown_patch_rejected(self):
        mesh = _mesh()
        with pytest.raises(KeyError):
            ScalarTransportFVMProcess().execute(
                ScalarTransportFVMInput(
                    mesh=mesh,
                    phi0=np.zeros(mesh.n_cells),
                    diffusivity=1.0,
                    bcs={"INLET": PatchBC.dirichlet(1.0)},
                )
            )

    def test_direct_solver_option(self):
        mesh = _mesh()
        res = ScalarTransportFVMProcess().execute(
            ScalarTransportFVMInput(
                mesh=mesh,
                phi0=np.zeros(mesh.n_cells),
                diffusivity=1.0,
                bcs=_BCS_FVM,
                linear_solver="direct",
            )
        )
        assert res.converged and res.residual_history[0] < 1e-10


class TestScalarTransportFVMRegression:
    """構造格子の FDM 版 ScalarTransportProcess と同じ解になること."""

    @pytest.mark.parametrize("transient", [False, True])
    def test_matches_structured_fdm(self, transient: bool):
        mesh = _mesh()
        u, v, w = _velocity_cells()
        rho, gamma = 1.2, 0.05
        source = 0.3 * np.ones((NX, NY, NZ))
        phi0 = np.linspace(0.0, 1.0, NX * NY * NZ).reshape(NX, NY, NZ)
        dt, t_end = (0.2, 0.6) if transient else (0.0, 0.0)

        fdm = ScalarTransportProcess().execute(
            ScalarTransportInput(
                Lx=LX,
                Ly=LY,
                Lz=LZ,
                nx=NX,
                ny=NY,
                nz=NZ,
                rho=rho,
                u=u,
                v=v,
                w=w,
                field=ScalarFieldSpec("c", gamma, phi0, source=source),
                dt=dt,
                t_end=t_end,
                tol=1e-12,
                max_iter=2000,
                **_BCS_FDM,
            )
        )
        fvm = ScalarTransportFVMProcess().execute(
            ScalarTransportFVMInput(
                mesh=mesh,
                phi0=phi0.ravel(),
                diffusivity=gamma,
                rho=rho,
                mass_flux=_structured_face_flux(mesh, u, v, w, rho),
                bcs=_BCS_FVM,
                source=source.ravel(),
                dt=dt,
                t_end=t_end,
                linear_solver="direct",
            )
        )
        assert fvm.n_timesteps == fdm.n_timesteps
        np.testing.assert_allclose(fvm.phi, fdm.phi.ravel(), rtol=1e-8, atol=1e-10)

    def test_solid_mask_matches_when_flow_is_zero_around_solid(self):
        """固体セルの対流無効化: 固体周りが静止していれば FDM と一致する."""
        mesh = _mesh()
        u, v, w = _velocity_cells()
        solid = np.zeros((NX, NY, NZ), dtype=bool)
        solid[2, 1:3, 1] = True
        # 固体とその周囲 1 セルの速度をゼロにして、対流の扱いの差（片側 vs 両側）を消す
        u2, v2, w2 = u.copy(), v.copy(), w.copy()
        for arr in (u2, v2, w2):
            arr[1:4, 0:4, 0:3] = 0.0
        phi0 = np.zeros((NX, NY, NZ))
        fdm = ScalarTransportProcess().execute(
            ScalarTransportInput(
                Lx=LX,
                Ly=LY,
                Lz=LZ,
                nx=NX,
                ny=NY,
                nz=NZ,
                rho=1.0,
                u=u2,
                v=v2,
                w=w2,
                field=ScalarFieldSpec("c", 0.1, phi0),
                solid_mask=solid,
                tol=1e-12,
                max_iter=2000,
                **_BCS_FDM,
            )
        )
        fvm = ScalarTransportFVMProcess().execute(
            ScalarTransportFVMInput(
                mesh=mesh,
                phi0=phi0.ravel(),
                diffusivity=0.1,
                rho=1.0,
                mass_flux=_structured_face_flux(mesh, u2, v2, w2, 1.0),
                bcs=_BCS_FVM,
                solid_mask=solid.ravel(),
                linear_solver="direct",
            )
        )
        np.testing.assert_allclose(fvm.phi, fdm.phi.ravel(), rtol=1e-8, atol=1e-10)

    def test_velocity_input_uses_boundary_outflow(self):
        """セル速度入力では境界面の流出入も入る（FDM には無い項）: 一様流で流入値が全域に伝わる."""
        mesh = _mesh()
        vel = np.zeros((mesh.n_cells, 3))
        vel[:, 0] = 0.8
        res = ScalarTransportFVMProcess().execute(
            ScalarTransportFVMInput(
                mesh=mesh,
                phi0=np.zeros(mesh.n_cells),
                diffusivity=1e-6,
                rho=1.0,
                velocity=vel,
                bcs={"XM": PatchBC.dirichlet(2.0)},
                linear_solver="direct",
            )
        )
        assert res.mass_flux is not None and res.mass_flux.shape == (mesh.n_faces,)
        np.testing.assert_allclose(res.phi, 2.0, rtol=1e-6)
