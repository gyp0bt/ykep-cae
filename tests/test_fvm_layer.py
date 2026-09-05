"""面ベース FVM 低レイヤー（xkep_cae_fluid.fvm）のテスト.

構造格子（StructuredMeshProcess）を面リストとして使い、解析解と比較する。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.fvm import (
    AMGSolver,
    BiCGSTABSolver,
    DirectSolver,
    PatchBC,
    assemble_convection,
    assemble_diffusion,
    assemble_scalar_transport,
    cell_gradient,
    face_mass_flux,
    make_linear_solver,
    relative_residual,
    resolve_boundary,
)


def _box(nx: int, ny: int = 1, nz: int = 1, **kw):
    return (
        StructuredMeshProcess()
        .process(StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz, **kw))
        .mesh
    )


class TestBoundaryResolution:
    def test_unknown_patch_raises(self):
        mesh = _box(3)
        with pytest.raises(KeyError, match="TOP"):
            resolve_boundary(mesh, {"TOP": PatchBC.dirichlet(1.0)})

    def test_default_is_zero_gradient_and_distance_is_half_cell(self):
        mesh = _box(4, 2, 2)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(1.0)})
        assert bf.n == mesh.n_boundary_faces
        assert bf.is_dirichlet.sum() == 4
        assert (bf.kind == 0).sum() == bf.n - 4
        np.testing.assert_allclose(bf.distance[bf.is_dirichlet], 0.125)

    def test_per_face_dirichlet_array(self):
        mesh = _box(2, 3, 1)
        vals = np.arange(3, dtype=float)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(vals)})
        np.testing.assert_allclose(bf.value[bf.is_dirichlet], vals)
        with pytest.raises(ValueError, match="長さ"):
            resolve_boundary(mesh, {"XM": PatchBC.dirichlet(np.ones(2))})


class TestDiffusion:
    @pytest.mark.parametrize("stretch", [(1.0,), (4.0, 1.0)])
    def test_1d_dirichlet_linear_profile(self, stretch):
        """両端 Dirichlet の 1D 定常拡散は線形（不等間隔でも 2 次精度で厳密）."""
        mesh = _box(12, stretch_x=stretch)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(1.0), "XP": PatchBC.dirichlet(3.0)})
        A, b = assemble_diffusion(mesh, 2.0, bf)
        phi = DirectSolver().solve(A, b)
        x = mesh.cell_centers[:, 0]
        np.testing.assert_allclose(phi, 1.0 + 2.0 * x, rtol=1e-12)

    def test_neumann_flux_sets_gradient(self):
        """左 Dirichlet + 右 Neumann（流入 q）→ ∂φ/∂x = −q/Γ（右端で流入なので勾配は負）."""
        mesh = _box(10)
        gamma, q = 0.5, 2.0
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.neumann(q)})
        A, b = assemble_diffusion(mesh, gamma, bf)
        phi = DirectSolver().solve(A, b)
        x = mesh.cell_centers[:, 0]
        # 右端で Γ ∂φ/∂n_in = q, n_in = −x なので ∂φ/∂x = −q/Γ … 符号: 流入で φ が増える
        np.testing.assert_allclose(phi, (q / gamma) * x, rtol=1e-12)

    def test_robin_equilibrium(self):
        """全面 Robin で同じ φ_inf なら φ ≡ φ_inf."""
        mesh = _box(3, 3, 3)
        bc = PatchBC.robin(h=5.0, phi_inf=7.0)
        bf = resolve_boundary(mesh, {n: bc for n in mesh.boundary_patches})
        A, b = assemble_diffusion(mesh, 1.3, bf)
        phi = DirectSolver().solve(A, b)
        np.testing.assert_allclose(phi, 7.0, rtol=1e-12)

    def test_robin_matches_structured_formula(self):
        """Robin の合成コンダクタンスが既存 FDM の 2Γh/(2Γ+hd) と一致する."""
        mesh = _box(4)
        gamma, h, d = 0.7, 3.0, 0.25
        bf = resolve_boundary(mesh, {"XM": PatchBC.robin(h, 0.0)})
        A, _ = assemble_diffusion(mesh, gamma, bf)
        # 左端セルの対角 = 内部面 a_f + Robin。内部面 a_f = Γ A/d = 0.7 * 1 / 0.25
        a_internal = gamma * 1.0 / d
        u_eff = 2.0 * gamma * h / (2.0 * gamma + h * d)  # 体積で割った形の係数 × d² … 面積分形
        expected = a_internal + u_eff * 1.0  # 面積 1
        np.testing.assert_allclose(A[0, 0], expected, rtol=1e-12)

    def test_heterogeneous_gamma_harmonic(self):
        """物性が 2 領域で異なる 1D 拡散: 界面で流束連続（調和平均）."""
        mesh = _box(10)
        gamma = np.where(mesh.cell_centers[:, 0] < 0.5, 1.0, 4.0)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(1.0)})
        A, b = assemble_diffusion(mesh, gamma, bf)
        phi = DirectSolver().solve(A, b)
        # 直列抵抗: R = 0.5/1 + 0.5/4 = 0.625, 流束 = 1/0.625 = 1.6
        x = mesh.cell_centers[:, 0]
        left = x < 0.5
        np.testing.assert_allclose(phi[left], 1.6 * x[left], rtol=1e-12)
        np.testing.assert_allclose(phi[~left], 0.8 + 0.4 * (x[~left] - 0.5), rtol=1e-12)


class TestConvection:
    def test_uniform_flow_upwind_transports_inlet_value(self):
        """一様流 + 流入 Dirichlet + 流出ゼロ勾配 → 純風上で全域が流入値."""
        mesh = _box(8, 2, 2)
        u = np.zeros((mesh.n_cells, 3))
        u[:, 0] = 1.5
        mf = face_mass_flux(mesh, u, rho=2.0)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(3.0)})
        A, b = assemble_convection(mesh, mf, bf)
        phi = DirectSolver().solve(A, b)
        np.testing.assert_allclose(phi, 3.0, rtol=1e-12)

    def test_mass_flux_blocked_cells(self):
        mesh = _box(4)
        u = np.zeros((mesh.n_cells, 3))
        u[:, 0] = 1.0
        blocked = np.array([False, True, False, False])
        mf = face_mass_flux(mesh, u, blocked_cells=blocked)
        # 内部面 0 (0-1), 1 (1-2) はゼロ、2 (2-3) は非ゼロ
        assert mf[0] == 0.0 and mf[1] == 0.0 and mf[2] != 0.0

    def test_convection_diffusion_peclet_profile(self):
        """1D 対流拡散（両端 Dirichlet）: 風上の離散解は指数則で表せる."""
        n = 20
        mesh = _box(n)
        rho, u_val, gamma = 1.0, 1.0, 0.1
        u = np.zeros((mesh.n_cells, 3))
        u[:, 0] = u_val
        mf = face_mass_flux(mesh, u, rho=rho)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(1.0)})
        A, b = assemble_scalar_transport(mesh, gamma=gamma, bfaces=bf, mass_flux=mf)
        phi = DirectSolver().solve(A, b)
        assert np.all(np.diff(phi) >= -1e-12)  # 単調
        assert phi[0] < 0.05 and phi[-1] > 0.5


class TestTransientAndSource:
    def test_no_flux_holds_initial(self):
        mesh = _box(3, 3, 3)
        bf = resolve_boundary(mesh, {})
        phi0 = np.full(mesh.n_cells, 2.5)
        A, b = assemble_scalar_transport(mesh, gamma=1.0, bfaces=bf, rho=1.0, dt=0.1, phi_old=phi0)
        phi = DirectSolver().solve(A, b)
        np.testing.assert_allclose(phi, 2.5, rtol=1e-12)

    def test_uniform_source_neumann_zero_transient(self):
        """断熱 + 一様ソース S の 1 ステップ: φ = φ0 + S dt/ρ."""
        mesh = _box(4, 2, 1)
        bf = resolve_boundary(mesh, {})
        phi0 = np.zeros(mesh.n_cells)
        A, b = assemble_scalar_transport(
            mesh,
            gamma=1.0,
            bfaces=bf,
            source=np.full(mesh.n_cells, 3.0),
            rho=2.0,
            dt=0.5,
            phi_old=phi0,
        )
        phi = DirectSolver().solve(A, b)
        np.testing.assert_allclose(phi, 0.75, rtol=1e-12)


class TestGradient:
    def test_green_gauss_linear_field_exact(self):
        mesh = _box(4, 3, 2)
        g = np.array([1.0, -2.0, 0.5])
        phi = mesh.cell_centers @ g + 3.0
        bf = resolve_boundary(
            mesh,
            {
                n: PatchBC.dirichlet(mesh.face_centers[idx] @ g + 3.0)
                for n, idx in mesh.boundary_patches.items()
            },
        )
        grad = cell_gradient(mesh, phi, bf)
        np.testing.assert_allclose(grad, np.tile(g, (mesh.n_cells, 1)), atol=1e-12)


class TestLinearSolvers:
    def _system(self):
        mesh = _box(6, 5, 4)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "ZP": PatchBC.dirichlet(1.0)})
        return assemble_diffusion(mesh, 1.0, bf)

    def test_direct_and_bicgstab_agree(self):
        A, b = self._system()
        x_d = DirectSolver().solve(A, b)
        x_b = BiCGSTABSolver(tol=1e-12).solve(A, b)
        np.testing.assert_allclose(x_b, x_d, atol=1e-8)
        assert relative_residual(A, x_d, b) < 1e-12

    def test_amg_agrees_or_skips(self):
        pytest.importorskip("pyamg")
        A, b = self._system()
        x_d = DirectSolver().solve(A, b)
        solver = AMGSolver(tol=1e-12)
        x_a = solver.solve(A, b)
        np.testing.assert_allclose(x_a, x_d, atol=1e-8)
        # 同じ行列で 2 回目はキャッシュ利用
        x_a2 = solver.solve(A, b)
        np.testing.assert_allclose(x_a2, x_d, atol=1e-8)

    def test_factory(self):
        assert isinstance(make_linear_solver("direct"), DirectSolver)
        assert isinstance(make_linear_solver("BiCGSTAB", tol=1e-6), BiCGSTABSolver)
        with pytest.raises(ValueError, match="未知"):
            make_linear_solver("gmres")
