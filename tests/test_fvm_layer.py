"""面ベース FVM 低レイヤー（xkep_cae_fluid.fvm）のテスト.

構造格子（StructuredMeshProcess）を面リストとして使い、解析解と比較する。
"""

from __future__ import annotations

from dataclasses import replace

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
    cell_gradient_lsq,
    diffusive_face_flux,
    face_decomposition,
    face_mass_flux,
    face_skewness,
    is_orthogonal,
    make_linear_solver,
    max_nonorthogonality_deg,
    relative_residual,
    resolve_boundary,
    solve_corrected,
    time_derivative_terms,
    tvd_deferred_correction,
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

    def test_specified_patch_wins_over_overlapping_default(self):
        """*SURFACE 名と予約面名が同じ面を指すとき、指定した方が default に上書きされない."""
        mesh = _box(3, 2, 2)
        patches = dict(mesh.boundary_patches)
        patches["INLET"] = patches["XM"].copy()  # 予約面 XM と同じ面
        mesh2 = replace(mesh, boundary_patches=patches)
        bf = resolve_boundary(mesh2, {"INLET": PatchBC.dirichlet(5.0)})
        assert bf.is_dirichlet.sum() == 4
        np.testing.assert_allclose(bf.value[bf.is_dirichlet], 5.0)
        # 指定パッチ同士の重なりは後のものが優先
        bf2 = resolve_boundary(mesh2, {"INLET": PatchBC.dirichlet(5.0), "XM": PatchBC.neumann(1.0)})
        assert bf2.is_dirichlet.sum() == 0 and bf2.is_neumann.sum() == 4

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


class TestTVDConvectionPhysics:
    """TVD 遅延補正（van Leer / Superbee）: 1D 対流拡散で風上より高精度、単調、保存."""

    def _peclet_case(self, n: int = 20):
        mesh = _box(n)
        rho, u_val, gamma = 1.0, 1.0, 0.1
        u = np.zeros((mesh.n_cells, 3))
        u[:, 0] = u_val
        mf = face_mass_flux(mesh, u, rho=rho)
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(1.0)})
        x = mesh.cell_centers[:, 0]
        pe = rho * u_val / gamma
        exact = (np.exp(pe * x) - 1.0) / (np.exp(pe) - 1.0)
        return mesh, mf, bf, gamma, exact

    @pytest.mark.parametrize("limiter", ["van_leer", "superbee"])
    def test_tvd_more_accurate_than_upwind_and_monotone(self, limiter: str):
        mesh, mf, bf, gamma, exact = self._peclet_case()
        A, b = assemble_scalar_transport(mesh, gamma=gamma, bfaces=bf, mass_flux=mf)
        phi_up = DirectSolver().solve(A, b)
        phi = phi_up.copy()
        for _ in range(100):
            A2, b2 = assemble_scalar_transport(
                mesh,
                gamma=gamma,
                bfaces=bf,
                mass_flux=mf,
                phi_correction=phi,
                convection="tvd",
                limiter=limiter,
            )
            new = DirectSolver().solve(A2, b2)
            done = np.linalg.norm(new - phi) < 1e-12
            phi = new
            if done:
                break
        err_up = np.abs(phi_up - exact).max()
        err_tvd = np.abs(phi - exact).max()
        assert err_tvd < err_up / 2.5, f"upwind {err_up:.3e}, tvd {err_tvd:.3e}"
        assert np.all(np.diff(phi) >= -1e-12)

    def test_correction_is_conservative_and_zero_for_upwind(self):
        mesh, mf, bf, _gamma, _exact = self._peclet_case(12)
        phi = np.sin(3.0 * mesh.cell_centers[:, 0])
        grad = cell_gradient(mesh, phi, bf)
        corr = tvd_deferred_correction(mesh, phi, mf, grad, "van_leer")
        assert abs(corr.sum()) < 1e-14  # 内部面の補正は owner/neighbour で相殺
        assert np.any(np.abs(corr) > 1e-6)
        with pytest.raises(ValueError, match="limiter"):
            tvd_deferred_correction(mesh, phi, mf, grad, "minmod")
        with pytest.raises(ValueError, match="phi_correction"):
            assemble_scalar_transport(mesh, gamma=1.0, bfaces=bf, mass_flux=mf, convection="tvd")

    def test_bounded_form_keeps_sink_cell_within_inflow_values(self):
        """面流束が保存的でない（セル 1 が吸い込み）とき、有界形なら流入値を超えない."""
        mesh = _box(4)
        mf = np.zeros(mesh.n_faces)
        mf[:3] = [1.0, 0.5, 0.25]  # 内部面 0-1, 1-2, 2-3（owner → neighbour）
        mf[mesh.patch_faces("XP")] = 0.25
        mf[mesh.patch_faces("XM")] = -1.0
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(3.0)})
        A, b = assemble_convection(mesh, mf, bf)
        phi = DirectSolver().solve(A, b)
        assert phi[1] > 3.0 + 1e-9  # 非保存的な流束で発散形はオーバーシュートする
        A_b, b_b = assemble_convection(mesh, mf, bf, bounded=True)
        phi_b = DirectSolver().solve(A_b, b_b)
        np.testing.assert_allclose(phi_b, 3.0, rtol=1e-12)


class TestTimeSchemesPhysics:
    """時間項: 陰的 Euler と BDF2（sin(πx) e^{−π²t} の減衰）."""

    def _run(self, scheme: str, n: int = 40, dt: float = 0.01, t_end: float = 0.1) -> float:
        mesh = _box(n)
        x = mesh.cell_centers[:, 0]
        bf = resolve_boundary(mesh, {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(0.0)})
        phi_old = np.sin(np.pi * x)
        phi_old2 = None
        t = 0.0
        while t < t_end - 1e-12:
            t += dt
            A, b = assemble_scalar_transport(
                mesh,
                gamma=1.0,
                bfaces=bf,
                rho=1.0,
                dt=dt,
                phi_old=phi_old,
                phi_old2=phi_old2 if scheme == "bdf2" else None,
            )
            phi_old2, phi_old = phi_old, DirectSolver().solve(A, b)
        exact = np.sin(np.pi * x) * np.exp(-(np.pi**2) * t_end)
        return float(np.abs(phi_old - exact).max() / exact.max())

    def test_bdf2_is_more_accurate_than_euler(self):
        err_euler = self._run("euler")
        err_bdf2 = self._run("bdf2")
        assert err_bdf2 < err_euler / 5.0, f"euler {err_euler:.3e}, bdf2 {err_bdf2:.3e}"

    def test_time_terms_formulas(self):
        mesh = _box(2)
        old = np.array([1.0, 2.0])
        old2 = np.array([0.5, 0.5])
        c, r = time_derivative_terms(mesh, 2.0, 0.5, old)
        np.testing.assert_allclose(c, 2.0 * mesh.cell_volumes / 0.5)
        np.testing.assert_allclose(r, c * old)
        c2, r2 = time_derivative_terms(mesh, 2.0, 0.5, old, old2)
        np.testing.assert_allclose(c2, 1.5 * c)
        np.testing.assert_allclose(r2, c * (2.0 * old - 0.5 * old2))


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
    def test_skew_corrected_green_gauss_on_tets(self, kuhn_tets):
        """四面体（面中心が P–N 直線から外れる）でも反復スキュー補正で線形場の勾配が出る."""
        mesh = kuhn_tets(3, 3, 3)
        _t, skew = face_skewness(mesh)
        assert np.abs(skew).max() > 0.02  # 実際にスキューがある（セル幅 1/3 に対して 8%）
        phi = (
            1.0
            + 2.0 * mesh.cell_centers[:, 0]
            - 0.5 * mesh.cell_centers[:, 1]
            + 3.0 * mesh.cell_centers[:, 2]
        )
        bcs = {
            nm: PatchBC.dirichlet(
                1.0
                + 2.0 * mesh.face_centers[mesh.patch_faces(nm), 0]
                - 0.5 * mesh.face_centers[mesh.patch_faces(nm), 1]
                + 3.0 * mesh.face_centers[mesh.patch_faces(nm), 2]
            )
            for nm in ("XM", "XP", "YM", "YP", "ZM", "ZP")
        }
        bf = resolve_boundary(mesh, bcs)
        g1 = cell_gradient(mesh, phi, bf, n_iter=1)
        g = cell_gradient(mesh, phi, bf)
        exact = np.array([2.0, -0.5, 3.0])
        assert np.abs(g1 - exact).max() > 0.5  # 補正なしの Green–Gauss は大きく外す
        exact_all = np.broadcast_to(exact, g.shape)
        np.testing.assert_allclose(g, exact_all, atol=1e-8)
        np.testing.assert_allclose(cell_gradient_lsq(mesh, phi, bf), exact_all, atol=1e-10)

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


def _sheared_box(nx: int, ny: int, nz: int, shear: float):
    """構造格子の節点を x += shear·y でせん断した非直交メッシュ（面リストは組み直す）."""
    from dataclasses import replace

    from xkep_cae_fluid.inp.builder import build_case
    from xkep_cae_fluid.inp.mesh import build_inp_mesh
    from xkep_cae_fluid.inp.parser import parse_inp_text

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    lines = ["*NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f" {nid(i, j, k)}, {i / nx}, {j / ny}, {k / nz}")
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
                lines.append(f" {e}, " + ", ".join(str(n) for n in c))
    case = build_case(parse_inp_text("\n".join(lines) + "\n"))
    coords = case.nodes.coords.copy()
    coords[:, 0] += shear * coords[:, 1]
    case = replace(case, nodes=replace(case.nodes, coords=coords))
    return build_inp_mesh(case).mesh


class TestNonorthogonalPhysics:
    def test_decomposition_reduces_to_area_on_orthogonal_mesh(self):
        mesh = _box(4, 3, 2)
        e_mag, t_vec, _d = face_decomposition(mesh)
        np.testing.assert_allclose(e_mag, mesh.face_areas[: mesh.n_internal_faces], rtol=1e-12)
        np.testing.assert_allclose(t_vec, 0.0, atol=1e-14)
        assert is_orthogonal(mesh) and max_nonorthogonality_deg(mesh) < 1e-9

    def test_sheared_mesh_has_nonorthogonality(self):
        mesh = _sheared_box(4, 3, 2, 0.3)
        assert not is_orthogonal(mesh)
        # 面法線 (±1, 0, 0) と e = (0.3, 1, 0)/|..| の間の y 面: atan(0.3) ≈ 16.7°
        assert max_nonorthogonality_deg(mesh) == pytest.approx(np.degrees(np.arctan(0.3)), abs=1e-6)

    def test_linear_field_exact_with_correction(self):
        """全面 Dirichlet の線形場は、補正の反復で厳密に再現され、面フラックスも厳密."""
        mesh = _sheared_box(5, 4, 2, 0.3)
        g = np.array([1.5, -0.7, 0.4])
        exact = mesh.cell_centers @ g + 2.0
        bf = resolve_boundary(
            mesh,
            {
                n: PatchBC.dirichlet(mesh.face_centers[idx] @ g + 2.0)
                for n, idx in mesh.boundary_patches.items()
            },
        )
        gamma = 3.0

        def build(phi_corr):
            return assemble_scalar_transport(mesh, gamma=gamma, bfaces=bf, phi_correction=phi_corr)

        phi, resid, n_iter = solve_corrected(
            mesh, build, DirectSolver(), np.zeros(mesh.n_cells), max_iter=50, tol=1e-13
        )
        assert n_iter > 1 and resid < 1e-10
        np.testing.assert_allclose(phi, exact, rtol=1e-9)
        flux = diffusive_face_flux(mesh, phi, gamma, bf)
        s_f = mesh.face_normals * mesh.face_areas[:, None]
        np.testing.assert_allclose(flux, -gamma * (s_f @ g), atol=1e-9)

    def test_uncorrected_flux_conserves_but_is_biased(self):
        mesh = _sheared_box(5, 4, 2, 0.3)
        g = np.array([1.0, 0.0, 0.0])
        phi = mesh.cell_centers @ g
        bf = resolve_boundary(mesh, {})
        full = diffusive_face_flux(mesh, phi, 1.0, bf, corrected=True)
        part = diffusive_face_flux(mesh, phi, 1.0, bf, corrected=False)
        n_int = mesh.n_internal_faces
        assert np.abs(full[:n_int] - part[:n_int]).max() > 1e-6
