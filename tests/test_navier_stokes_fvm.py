"""NavierStokesFVMProcess のテスト（API + 物理ベンチマーク）."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.fvm import PatchBC, diffusive_face_flux, resolve_boundary
from xkep_cae_fluid.fvm.momentum import VelocityPatchBC, resolve_velocity_boundary
from xkep_cae_fluid.fvm.viscosity import CarreauViscosity, PowerLawViscosity
from xkep_cae_fluid.incompressible import (
    FlowPatchBC,
    InternalCellBC,
    NavierStokesFVMInput,
    NavierStokesFVMProcess,
    NavierStokesFVMResult,
    ScalarSpec,
)
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mesh import build_inp_mesh
from xkep_cae_fluid.inp.parser import parse_inp_text


def _box(nx, ny, nz, lx, ly, lz):
    return (
        StructuredMeshProcess()
        .execute(StructuredMeshInput(Lx=lx, Ly=ly, Lz=lz, nx=nx, ny=ny, nz=nz))
        .mesh
    )


def _sheared(nx, ny, lx, ly, lz, shear):
    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    lines = ["*NODE"]
    for k in range(2):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f" {nid(i, j, k)}, {i / nx * lx}, {j / ny * ly}, {k * lz}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALL")
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
            lines.append(f" {e}, " + ", ".join(map(str, c)))
    case = build_case(parse_inp_text("\n".join(lines) + "\n"))
    coords = case.nodes.coords.copy()
    coords[:, 0] += shear * coords[:, 1]
    case = replace(case, nodes=replace(case.nodes, coords=coords))
    return build_inp_mesh(case).mesh


U_IN, H, LX, RHO, MU = 0.01, 0.1, 1.0, 1.0, 0.01


def _channel_bcs(walls: str = "wall") -> dict[str, FlowPatchBC]:
    wall = FlowPatchBC.wall() if walls == "wall" else FlowPatchBC.symmetry()
    return {
        "XM": FlowPatchBC.inlet((U_IN, 0.0, 0.0)),
        "XP": FlowPatchBC.outlet(0.0),
        "YM": wall,
        "YP": wall,
        "ZM": FlowPatchBC.symmetry(),
        "ZP": FlowPatchBC.symmetry(),
    }


def _run(mesh, bcs, **kw) -> NavierStokesFVMResult:
    base = dict(
        mesh=mesh,
        rho=RHO,
        mu=MU,
        bcs=bcs,
        max_outer_iter=800,
        tol=1e-6,
        alpha_u=0.7,
        alpha_p=0.3,
        linear_solver="direct",
        pressure_solver="direct",
    )
    base.update(kw)
    return NavierStokesFVMProcess().execute(NavierStokesFVMInput(**base))


@binds_to(NavierStokesFVMProcess)
class TestNavierStokesFVMAPI:
    def test_meta(self):
        assert NavierStokesFVMProcess.meta.name == "NavierStokesFVM"
        assert NavierStokesFVMProcess.meta.module == "solve"

    def test_validation(self):
        mesh = _box(4, 2, 1, 0.4, 0.2, 0.1)
        with pytest.raises(ValueError, match="rho"):
            _run(mesh, {}, rho=-1.0, max_outer_iter=1)
        with pytest.raises(ValueError, match="coupling"):
            _run(mesh, {}, coupling="block", max_outer_iter=1)
        with pytest.raises(ValueError, match="convection"):
            _run(mesh, {}, convection="quick", max_outer_iter=1)
        with pytest.raises(ValueError, match="limiter"):
            _run(mesh, {}, convection="tvd", limiter="minmod", max_outer_iter=1)
        with pytest.raises(ValueError, match="time_scheme"):
            _run(mesh, {}, time_scheme="rk2", max_outer_iter=1)
        with pytest.raises(ValueError, match="重複"):
            specs = (ScalarSpec("c", 1e-3), ScalarSpec("c", 1e-3))
            _run(mesh, {}, scalars=specs, max_outer_iter=1)
        with pytest.raises(KeyError, match="LID"):
            _run(mesh, {"LID": FlowPatchBC.wall()}, max_outer_iter=1)

    def test_result_shapes_and_residual_maps(self):
        mesh = _box(6, 3, 1, 0.6, 0.1, 0.1)
        res = _run(mesh, _channel_bcs(), max_outer_iter=5, tol=1e-12)
        assert isinstance(res, NavierStokesFVMResult)
        assert res.velocity.shape == (18, 3) and res.p.shape == (18,) and res.T.shape == (18,)
        assert res.mass_flux.shape == (mesh.n_faces,)
        assert res.n_outer_iterations == 5 and not res.converged
        assert set(res.residual_fields) == {"res_u", "res_v", "res_w", "res_T", "res_mass"}
        assert len(res.residual_history["mass"]) == 5

    def test_transient_runs(self):
        mesh = _box(6, 3, 1, 0.6, 0.1, 0.1)
        res = _run(mesh, _channel_bcs(), dt=0.5, t_end=1.0, max_outer_iter=50, tol=1e-6)
        assert res.n_timesteps == 2 and res.converged
        assert res.time_history == (0.5, 1.0)

    def test_options_run_and_report(self):
        """TVD / BDF2 / PISO / 対流流出 / 追加スカラー / 内部セル BC が API として通る."""
        mesh = _box(6, 3, 1, 0.6, 0.1, 0.1)
        bcs = _channel_bcs()
        bcs["XP"] = FlowPatchBC.outflow()
        inl = np.zeros(mesh.n_cells, dtype=bool)
        inl[4] = True
        out = np.zeros(mesh.n_cells, dtype=bool)
        out[13] = True
        res = _run(
            mesh,
            bcs,
            dt=0.5,
            t_end=1.5,
            max_outer_iter=20,
            tol=1e-6,
            convection="tvd",
            limiter="superbee",
            time_scheme="bdf2",
            coupling="piso",
            n_piso_correctors=3,
            scalars=(ScalarSpec("c", 1e-3, 1.0, bcs={"XM": PatchBC.dirichlet(2.0)}),),
            internal_bcs=(
                InternalCellBC.inlet(inl, (0.02, 0.0, 0.0), temperature=310.0, label="pump"),
                InternalCellBC.outlet(out),
            ),
            solve_energy=True,
        )
        assert res.n_timesteps == 3
        assert set(res.scalars) == {"c"} and res.scalars["c"].shape == (mesh.n_cells,)
        assert "c" in res.residual_history and "res_c" in res.residual_fields
        np.testing.assert_allclose(res.velocity[4], [0.02, 0.0, 0.0])
        assert res.T[4] == pytest.approx(310.0)
        assert np.isfinite(res.velocity).all() and np.isfinite(res.p).all()
        with pytest.raises(ValueError, match="mask"):
            _run(mesh, {}, internal_bcs=(InternalCellBC.outlet(np.zeros(3, dtype=bool)),))


class TestNavierStokesFVMPhysics:
    def _check_poiseuille(self, mesh, res, ny: int, rtol_profile: float, rtol_dp: float):
        assert res.converged
        u = res.velocity
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        # 出口付近（x が最大の列）の断面
        col = x >= np.sort(np.unique(np.round(x, 9)))[-1] - 1e-9
        assert col.sum() == ny
        exact = 1.5 * U_IN * (1.0 - (2.0 * (y[col] - H / 2.0) / H) ** 2)
        # 壁セルは半セル距離の境界フラックス評価で放物線から 3% ほどずれる（離散化誤差）
        np.testing.assert_allclose(u[col, 0], exact, rtol=rtol_profile)
        assert u[col, 0].mean() == pytest.approx(U_IN, rel=1e-6)  # 断面平均 = 流入速度
        np.testing.assert_allclose(u[col, 1], 0.0, atol=1e-3 * U_IN)
        # 流入 = 流出
        mf = res.mass_flux[mesh.n_internal_faces :]
        inflow = -mf[mf < 0].sum()
        outflow = mf[mf > 0].sum()
        assert inflow == pytest.approx(
            RHO * U_IN * H * mesh.cell_volumes.sum() / (LX * H), rel=1e-6
        )
        assert outflow == pytest.approx(inflow, rel=1e-6)
        # 圧力勾配 dp/dx = 12 μ U / H²（発達した領域で）
        dpdx = 12.0 * MU * U_IN / H**2
        mid = (x > 0.4 * LX) & (x < 0.8 * LX)
        coef = np.polyfit(x[mid], res.p[mid], 1)
        assert -coef[0] == pytest.approx(dpdx, rel=rtol_dp)

    def test_poiseuille_box(self):
        nx, ny = 30, 10
        mesh = _box(nx, ny, 1, LX, H, 0.02)
        res = _run(mesh, _channel_bcs())
        self._check_poiseuille(mesh, res, ny, rtol_profile=0.04, rtol_dp=0.03)

    def test_poiseuille_sheared_mesh(self):
        """x += 0.3 y のせん断メッシュ（非直交 16.7°）でも Poiseuille 分布が出る."""
        nx, ny = 30, 10
        mesh = _sheared(nx, ny, LX, H, 0.02, 0.3)
        res = _run(mesh, _channel_bcs())
        assert res.converged
        y = mesh.cell_centers[:, 1]
        x = mesh.cell_centers[:, 0]
        # 流路中央の列（要素番号 i = nx/2）で評価する。出口面は傾いており、そこに一様圧力を
        # 課すと出口近傍の流れが横方向に歪む（境界条件の帰結で、ソルバーの誤差ではない）
        col = np.zeros(mesh.n_cells, dtype=bool)
        col[np.arange(nx // 2, mesh.n_cells, nx)] = True
        exact = 1.5 * U_IN * (1.0 - (2.0 * (y[col] - H / 2.0) / H) ** 2)
        np.testing.assert_allclose(res.velocity[col, 0], exact, rtol=0.05)
        mf = res.mass_flux[mesh.n_internal_faces :]
        assert -mf[mf < 0].sum() == pytest.approx(mf[mf > 0].sum(), rel=1e-6)
        dpdx = 12.0 * MU * U_IN / H**2
        mid = (x > 0.4 * LX) & (x < 0.8 * LX)
        # 圧力は x にほぼ線形（y 方向には一様）
        coef = np.polyfit(x[mid], res.p[mid], 1)
        assert -coef[0] == pytest.approx(dpdx, rel=0.05)

    def test_brinkman_channel_uniform_flow_and_pressure_drop(self):
        """すべり壁 + 一様透過率: 速度一様、圧力降下 μ U L / K."""
        nx, ny = 20, 4
        mesh = _box(nx, ny, 1, LX, H, 0.02)
        K = 1.0e-6
        res = _run(mesh, _channel_bcs("slip"), permeability=np.full(mesh.n_cells, K), tol=1e-8)
        assert res.converged
        np.testing.assert_allclose(res.velocity[:, 0], U_IN, rtol=1e-6)
        np.testing.assert_allclose(res.velocity[:, 1:], 0.0, atol=1e-8 * U_IN)
        x = mesh.cell_centers[:, 0]
        dp_exact = MU * U_IN * LX / K
        np.testing.assert_allclose(res.p, dp_exact * (1.0 - x / LX), rtol=1e-6)

    def test_lid_driven_cavity_re100(self):
        """蓋駆動キャビティ Re=100（24×24、1 次風上）: 中心線速度が Ghia (1982) と整合."""
        n = 24
        mesh = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        bcs = {
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        res = _run(mesh, bcs, mu=0.01, max_outer_iter=3000, tol=1e-5)
        assert res.converged
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        # 鉛直中心線 x ≈ 0.5（2 列の平均）
        vert = np.abs(x - 0.5) < 1.0 / n
        u_line = res.velocity[vert, 0]
        y_line = y[vert]
        i_min = int(np.argmin(u_line))
        # Ghia: u_min ≈ −0.21 at y ≈ 0.46。粗格子 + 1 次風上なので緩い許容
        assert -0.26 < u_line[i_min] < -0.12
        assert 0.3 < y_line[i_min] < 0.65
        horiz = np.abs(y - 0.5) < 1.0 / n
        v_line = res.velocity[horiz, 1]
        x_line = x[horiz]
        # Ghia: v_max ≈ 0.18 at x ≈ 0.23、v_min ≈ −0.25 at x ≈ 0.81
        assert 0.09 < v_line.max() < 0.24 and x_line[np.argmax(v_line)] < 0.4
        assert -0.32 < v_line.min() < -0.13 and x_line[np.argmin(v_line)] > 0.6
        # 蓋のすぐ下は正、質量保存
        assert res.velocity[np.argmax(y), 0] > 0.0
        assert res.residual_history["mass"][-1] < 1e-5

    def test_differentially_heated_cavity_ra1e3_nusselt(self):
        """差分加熱キャビティ Ra=10³（12×12、z 対称面）: Nu が de Vahl Davis 1.118 の 20% 以内."""
        L, T_hot, T_cold, T_ref = 0.1, 310.0, 290.0, 300.0
        rho, mu, Cp, k = 1.0, 0.01, 1000.0, 1.0
        g = 9.81
        nu = mu / rho
        alpha_th = k / (rho * Cp)
        beta = 1000.0 * nu * alpha_th / (g * (T_hot - T_cold) * L**3)
        n = 12
        mesh = _box(n, n, 1, L, L, L / n)
        bcs = {
            "XM": FlowPatchBC.wall(temperature=T_hot),
            "XP": FlowPatchBC.wall(temperature=T_cold),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        res = _run(
            mesh,
            bcs,
            rho=rho,
            mu=mu,
            solve_energy=True,
            Cp=Cp,
            k_fluid=k,
            beta=beta,
            T_ref=T_ref,
            gravity=(0.0, -g, 0.0),
            T0=np.full(mesh.n_cells, T_ref),
            max_outer_iter=3000,
            tol=1e-4,
            alpha_u=0.5,
            alpha_p=0.2,
            alpha_T=0.7,
        )
        assert res.converged
        assert T_cold - 1e-9 <= res.T.min() and res.T.max() <= T_hot + 1e-9
        tb = resolve_boundary(mesh, {k_: v.thermal for k_, v in bcs.items() if v.thermal})
        flux = diffusive_face_flux(mesh, res.T, k, tb)
        hot = mesh.patch_faces("XM")
        q_in = -flux[hot].sum()  # owner から出る向きが正なので、壁から流体へは負
        area = mesh.face_areas[hot].sum()
        Nu = q_in * L / (k * (T_hot - T_cold) * area)
        assert abs(Nu - 1.118) / 1.118 < 0.20, f"Nu={Nu:.3f}"
        # 高温壁側で上昇流（v > 0）、低温壁側で下降流
        x = mesh.cell_centers[:, 0]
        assert res.velocity[x < L / 4, 1].mean() > 0.0 > res.velocity[x > 3 * L / 4, 1].mean()

    def test_poiseuille_outflow_boundary(self):
        """対流流出（速度・圧力ゼロ勾配、流束を流入と釣り合わせる）でも Poiseuille 分布と圧力勾配が出る."""
        nx, ny = 30, 10
        mesh = _box(nx, ny, 1, LX, H, 0.02)
        bcs = _channel_bcs()
        bcs["XP"] = FlowPatchBC.outflow()
        res = _run(mesh, bcs)
        self._check_poiseuille(mesh, res, ny, rtol_profile=0.04, rtol_dp=0.03)
        # 圧力の基準はセル 0（OUTLET が無い）
        assert res.p[0] == pytest.approx(0.0, abs=1e-12 * abs(res.p).max())

    def test_lid_driven_cavity_tvd_matches_ghia_closely(self):
        """蓋駆動キャビティ Re=100 を TVD（van Leer）で: 中心線 u の極小値が Ghia (1982) の −0.2109 に近い.

        1 次風上（24×24）は −0.185 で 12% 低く出るが、TVD は −0.211（コミット時の実測値）
        """
        n = 24
        mesh = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        bcs = {
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        res = _run(mesh, bcs, mu=0.01, max_outer_iter=3000, tol=1e-5, convection="tvd")
        assert res.converged
        x = mesh.cell_centers[:, 0]
        vert = np.abs(x - 0.5) < 1.0 / n
        u_min = float(res.velocity[vert, 0].min())
        assert abs(u_min + 0.2109) < 0.02, f"u_min={u_min:.4f}"

    def test_piso_correctors_reduce_splitting_error(self):
        """外部反復 1 回の非定常ステップで、PISO の追加補正が連成解（外部反復収束）との差を減らす.

        Stokes 的な蓋駆動キャビティ（μ = 1、1 ステップ）: 補正 1 回（SIMPLE 相当）→ 2 回 → 3 回で誤差が単調に減る。
        残る差は遅延評価の面質量流束による
        """
        n = 16
        mesh = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        bcs = {
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        kw = dict(mu=1.0, dt=0.002, t_end=0.002)
        ref = _run(mesh, bcs, max_outer_iter=500, tol=1e-11, alpha_u=0.8, alpha_p=0.5, **kw)
        assert ref.converged
        scale = np.abs(ref.velocity).max()
        errs = []
        for nc in (1, 2, 3):
            res = _run(
                mesh,
                bcs,
                max_outer_iter=1,
                tol=1e-12,
                coupling="piso",
                n_piso_correctors=nc,
                alpha_u=1.0,
                alpha_p=1.0,
                **kw,
            )
            errs.append(float(np.abs(res.velocity - ref.velocity).max() / scale))
        assert errs[1] < errs[0] / 3.0 and errs[2] < errs[1], f"errs={errs}"
        assert errs[2] < 0.01

    def test_pressure_nonorthogonal_correction_stabilizes_sheared_cavity(self):
        """せん断 0.6（非直交 31°）の Stokes 的キャビティ 1 ステップ: 圧力補正の非直交補正で収束が安定する.

        α_u = 0.8, α_p = 0.5 では補正なし（1 回）だと 60 反復で収束せず、2 回で直交メッシュ並みの
        反復数で収束する。収束解は補正回数によらない（保守的な緩和で収束させた解と一致）。
        非直交角 45° 付近では遅延補正の反復自体が縮小しないので 3 回にはしない（設計文書）。
        """
        n = 16
        mesh = _sheared(n, n, 1.0, 1.0, 1.0 / n, 0.6)
        bcs = {
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        kw = dict(mu=1.0, dt=0.002, t_end=0.002, tol=1e-9)
        ref = _run(mesh, bcs, max_outer_iter=300, alpha_u=0.7, alpha_p=0.3, **kw)
        assert ref.converged
        one = _run(
            mesh,
            bcs,
            max_outer_iter=60,
            alpha_u=0.8,
            alpha_p=0.5,
            n_nonorthogonal_correctors=1,
            **kw,
        )
        two = _run(
            mesh,
            bcs,
            max_outer_iter=60,
            alpha_u=0.8,
            alpha_p=0.5,
            n_nonorthogonal_correctors=2,
            **kw,
        )
        assert not one.converged
        assert two.converged and two.n_outer_iterations <= 40, two.n_outer_iterations
        scale = np.abs(ref.velocity).max()
        assert np.abs(two.velocity - ref.velocity).max() / scale < 1e-5
        # 直交メッシュでは補正回数によらず同じ反復数（余分な圧力解法をしない）
        box = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        a = _run(
            box,
            bcs,
            max_outer_iter=100,
            alpha_u=0.8,
            alpha_p=0.5,
            n_nonorthogonal_correctors=1,
            **kw,
        )
        b = _run(
            box,
            bcs,
            max_outer_iter=100,
            alpha_u=0.8,
            alpha_p=0.5,
            n_nonorthogonal_correctors=3,
            **kw,
        )
        assert a.converged and b.converged and a.n_outer_iterations == b.n_outer_iterations
        np.testing.assert_allclose(a.velocity, b.velocity)

    def test_adaptive_relaxation_converges_and_records_history(self):
        """適応緩和: Poiseuille 流路で収束し、α の履歴が反復ごとに記録され、上下限と SIMPLE の目安を守る."""
        mesh = _box(16, 8, 1, LX, H, 0.01)
        res = _run(mesh, _channel_bcs(), adaptive_relaxation=True, alpha_u=0.5, alpha_p=0.2)
        assert res.converged
        au = np.array(res.alpha_history["alpha_u"])
        ap = np.array(res.alpha_history["alpha_p"])
        assert len(au) == len(ap) == res.n_outer_iterations
        assert np.all((au >= 0.1) & (au <= 0.9)) and np.all((ap >= 0.05) & (ap <= 0.5))
        assert np.all(ap <= 1.0 - au + 1e-12)
        assert au.max() > 0.5  # 収束が順調なので積極化された
        plain = _run(mesh, _channel_bcs(), alpha_u=0.5, alpha_p=0.2)
        assert plain.converged and not plain.alpha_history
        scale = np.abs(plain.velocity).max()
        assert np.abs(res.velocity - plain.velocity).max() / scale < 1e-4

    def test_bdf2_conduction_more_accurate_than_euler(self):
        """静止流体の 1D 熱伝導（sin(πx) の減衰）: BDF2 の誤差が Euler の 1/5 未満."""
        n = 20
        mesh = _box(n, 1, 1, 1.0, 0.1, 0.1)
        x = mesh.cell_centers[:, 0]
        bcs = {"XM": FlowPatchBC.wall(temperature=0.0), "XP": FlowPatchBC.wall(temperature=0.0)}
        errs = {}
        for scheme in ("euler", "bdf2"):
            res = _run(
                mesh,
                bcs,
                rho=1.0,
                mu=0.01,
                solve_energy=True,
                Cp=1.0,
                k_fluid=1.0,
                T0=np.sin(np.pi * x),
                T_ref=0.0,
                dt=0.01,
                t_end=0.1,
                max_outer_iter=50,
                tol=1e-10,
                time_scheme=scheme,
                alpha_T=1.0,
            )
            assert res.converged and res.n_timesteps == 10
            exact = np.sin(np.pi * x) * np.exp(-(np.pi**2) * 0.1)
            errs[scheme] = float(np.abs(res.T - exact).max() / exact.max())
        assert errs["bdf2"] < errs["euler"] / 5.0, errs

    def test_internal_cell_inlet_outlet(self):
        """閉じた箱の内部に吐出セル（u = 0.1、T = 350）と吸入セル: 湧き出し・吸い込み以外は質量保存."""
        n = 16
        mesh = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        inl = np.zeros(mesh.n_cells, dtype=bool)
        inl[np.argmin((x - 0.2) ** 2 + (y - 0.5) ** 2)] = True
        out = np.zeros(mesh.n_cells, dtype=bool)
        out[np.argmin((x - 0.8) ** 2 + (y - 0.5) ** 2)] = True
        bcs = {"ZM": FlowPatchBC.symmetry(), "ZP": FlowPatchBC.symmetry()}
        res = _run(
            mesh,
            bcs,
            mu=0.01,
            max_outer_iter=2000,
            tol=1e-5,
            internal_bcs=(
                InternalCellBC.inlet(inl, (0.1, 0.0, 0.0), temperature=350.0),
                InternalCellBC.outlet(out),
            ),
            solve_energy=True,
            T0=np.full(mesh.n_cells, 300.0),
            k_fluid=0.01,
            Cp=1.0,
        )
        assert res.converged
        np.testing.assert_allclose(res.velocity[inl], [[0.1, 0.0, 0.0]])
        assert res.T[inl] == pytest.approx(350.0)
        mf = res.mass_flux
        imb = np.zeros(mesh.n_cells)
        np.add.at(imb, mesh.face_owner, mf)
        np.add.at(imb, mesh.face_neighbour, -mf[: mesh.n_internal_faces])
        free = ~(inl | out)
        assert np.abs(imb[free]).max() < 1e-12 * np.abs(mf).max()
        assert imb[inl][0] > 0.0 and imb[out][0] == pytest.approx(-imb[inl][0], rel=1e-9)
        # 吐出 → 吸入へ向かう流れ、温度は有界（有界形の対流）
        line = (np.abs(y - y[inl][0]) < 1e-9) & (x > x[inl][0]) & (x < x[out][0])
        assert res.velocity[line, 0].min() > 0.0
        assert 300.0 - 1e-9 <= res.T.min() and res.T.max() <= 350.0 + 1e-9

    def test_scalar_transported_with_flow(self):
        """流路の追加スカラー: 流入 Dirichlet 2.0 が定常で全域に運ばれる（境界流出はゼロ勾配）."""
        mesh = _box(20, 6, 1, LX, H, 0.02)
        spec = ScalarSpec("c", 1e-4, 0.0, bcs={"XM": PatchBC.dirichlet(2.0)})
        res = _run(mesh, _channel_bcs(), scalars=(spec,))
        assert res.converged
        np.testing.assert_allclose(res.scalars["c"], 2.0, rtol=1e-9)

    def test_scalar_closed_domain_conserved_and_bounded(self):
        """蓋駆動キャビティのトレーサ（非定常、TVD）: 総量保存、0 ≤ c ≤ 1."""
        n = 10
        mesh = _box(n, n, 1, 1.0, 1.0, 1.0 / n)
        x = mesh.cell_centers[:, 0]
        c0 = np.where(x < 0.5, 1.0, 0.0)
        bcs = {
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        res = _run(
            mesh,
            bcs,
            mu=0.01,
            scalars=(ScalarSpec("c", 1e-3, c0),),
            dt=0.1,
            t_end=1.0,
            max_outer_iter=100,
            tol=1e-6,
            convection="tvd",
        )
        assert res.converged
        c = res.scalars["c"]
        total0 = float((c0 * mesh.cell_volumes).sum())
        assert float((c * mesh.cell_volumes).sum()) == pytest.approx(total0, rel=1e-10)
        assert c.min() >= -1e-9 and c.max() <= 1.0 + 1e-9
        assert 0.05 < c[x > 0.5].mean() < 0.95  # 流れで再分配されている


def _periodic_box(nx, ny, lx, ly, lz=0.05, *, x_periodic=True, z_periodic=True):
    """箱格子を .inp 経由で作り、x / z 方向を周期にする（体積力駆動の検証用）."""
    lines = [f"*GRID, NX={nx}, NY={ny}, NZ=1, LX={lx}, LY={ly}, LZ={lz}"]
    if x_periodic:
        lines.append("*BOUNDARY, TYPE=PERIODIC\n XM, XP")
    if z_periodic:
        lines.append("*BOUNDARY, TYPE=PERIODIC\n ZM, ZP")
    return build_inp_mesh(build_case(parse_inp_text("\n".join(lines) + "\n"))).mesh


def _annulus(nr, nt, r_in, r_out, depth=0.05):
    """全周の円環（六面体）。内周 ``INNER`` / 外周 ``OUTER``."""
    import math

    def nid(i, j, k):
        return 1 + i + (nr + 1) * (j % nt) + (nr + 1) * nt * k

    lines = ["*NODE"]
    for k in range(2):
        for j in range(nt):
            th = 2 * math.pi * j / nt
            for i in range(nr + 1):
                r = r_in + (r_out - r_in) * i / nr
                lines.append(
                    f" {nid(i, j, k)}, {r * math.cos(th)}, {r * math.sin(th)}, {k * depth}"
                )
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALL")
    e, inner, outer = 0, [], []
    for j in range(nt):
        for i in range(nr):
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
            lines.append(f" {e}, " + ", ".join(map(str, c)))
            (inner if i == 0 else outer if i == nr - 1 else []).append(e)
    for name, ids in (("EIN", inner), ("EOUT", outer)):
        lines.append(f"*ELSET, ELSET={name}")
        lines += [" " + ", ".join(map(str, ids[a : a + 8])) for a in range(0, len(ids), 8)]
    lines.append("*SURFACE, NAME=INNER\n EIN, S6")
    lines.append("*SURFACE, NAME=OUTER\n EOUT, S4")
    return build_inp_mesh(build_case(parse_inp_text("\n".join(lines) + "\n"))).mesh


class TestPeriodicAndBodyForcePhysics:
    """周期境界 + 一様体積力（押出の圧力跳びの分解）と Stokes モード."""

    def test_periodic_channel_matches_poiseuille(self):
        h, f, mu = 0.1, 2.0, 0.01
        mesh = _periodic_box(4, 24, 0.4, h)
        res = _run(
            mesh,
            {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()},
            mu=mu,
            body_force=(f, 0.0, 0.0),
            coupling="coupled",
            tol=1e-10,
            max_outer_iter=20,
        )
        assert res.converged
        y = mesh.cell_centers[:, 1]
        u_exact = f / (2.0 * mu) * y * (h - y)
        assert np.max(np.abs(res.velocity[:, 0] - u_exact)) / u_exact.max() < 3e-3
        # 圧力は周期方向に一様（跳びは体積力に移してある）
        assert np.ptp(res.p) < 1e-10
        assert res.residual_history["mass"][-1] < 1e-12

    def test_z_periodic_gives_exact_2p5d_third_component(self):
        """1 セル厚の z を周期にすると ∂/∂z = 0 が厳密になり w が y だけの関数になる.

        対称面にすると z 方向にも壁ができ、w が厚さ ``lz`` に依存する偽の解になる
        （2.5D の展開チャネルで下流方向速度を出すには周期にしなければならない）。
        """
        h, mu, f = 0.1, 0.01, 1.0
        kw = dict(mu=mu, body_force=(0.0, 0.0, f), coupling="coupled", tol=1e-10, max_outer_iter=20)
        mesh = _periodic_box(4, 16, 0.4, h)
        res = _run(mesh, {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()}, **kw)
        y = mesh.cell_centers[:, 1]
        w_exact = f / (2.0 * mu) * y * (h - y)
        assert res.converged
        assert np.max(np.abs(res.velocity[:, 2] - w_exact)) / w_exact.max() < 1e-2
        # 厚さを変えても解は動かない（∂/∂z = 0）
        thick = _periodic_box(4, 16, 0.4, h, lz=0.2)
        res_thick = _run(thick, {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()}, **kw)
        assert np.max(np.abs(res_thick.velocity[:, 2] - res.velocity[:, 2])) < 1e-10
        # 対称面にすると z 方向の壁が効いて w が厚さに依存する
        sym_bcs = {
            "YM": FlowPatchBC.wall(),
            "YP": FlowPatchBC.wall(),
            "ZM": FlowPatchBC.symmetry(),
            "ZP": FlowPatchBC.symmetry(),
        }
        a = _run(_periodic_box(4, 16, 0.4, h, z_periodic=False), sym_bcs, **kw)
        b = _run(_periodic_box(4, 16, 0.4, h, lz=0.2, z_periodic=False), sym_bcs, **kw)
        assert a.velocity[:, 2].max() < 0.5 * w_exact.max()
        assert b.velocity[:, 2].max() > 1.5 * a.velocity[:, 2].max()

    def test_stokes_mode_drops_convection(self):
        """``convection="none"`` は慣性項を落とす（ρ を変えても解が動かない）."""
        mesh = _periodic_box(4, 16, 0.4, 0.1)
        bcs = {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall(velocity=(0.5, 0.0, 0.0))}
        kw = dict(mu=0.01, coupling="coupled", convection="none", tol=1e-10, max_outer_iter=20)
        a = _run(mesh, bcs, rho=1.0, **kw)
        b = _run(mesh, bcs, rho=1000.0, **kw)
        assert a.converged and b.converged
        assert np.max(np.abs(a.velocity - b.velocity)) < 1e-12


class TestCoupledSolverPhysics:
    """速度–圧力の連成（``coupling="coupled"``）: SIMPLE と同じ解に 1 回の直接解で届く."""

    def test_stokes_cavity_matches_simple_in_two_iterations(self):
        mesh = _periodic_box(16, 16, 1.0, 1.0, x_periodic=False)
        bcs = {
            "XM": FlowPatchBC.wall(),
            "XP": FlowPatchBC.wall(),
            "YM": FlowPatchBC.wall(),
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
        }
        kw = dict(mu=0.01, convection="none", tol=1e-9, max_outer_iter=800)
        c = _run(mesh, bcs, coupling="coupled", **kw)
        s = _run(mesh, bcs, coupling="simple", **kw)
        assert c.converged and s.converged
        assert c.n_outer_iterations == 2 and s.n_outer_iterations > 50
        assert np.max(np.abs(c.velocity - s.velocity)) < 1e-6
        dp = c.p - s.p
        assert np.max(np.abs(dp - dp.mean())) < 1e-7 * max(np.ptp(c.p), 1e-30)

    def test_re100_cavity_matches_simple(self):
        mesh = _periodic_box(20, 20, 1.0, 1.0, x_periodic=False)
        bcs = {
            "XM": FlowPatchBC.wall(),
            "XP": FlowPatchBC.wall(),
            "YM": FlowPatchBC.wall(),
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
        }
        kw = dict(mu=0.01, rho=1.0, tol=1e-7, convection="upwind")
        c = _run(mesh, bcs, coupling="coupled", max_outer_iter=60, **kw)
        s = _run(mesh, bcs, coupling="simple", max_outer_iter=800, **kw)
        assert c.converged and s.converged
        assert c.n_outer_iterations < s.n_outer_iterations / 5
        assert np.max(np.abs(c.velocity - s.velocity)) < 1e-4

    def test_outflow_rejected(self):
        mesh = _periodic_box(4, 4, 0.4, 0.1, x_periodic=False)
        with pytest.raises(ValueError, match="OUTFLOW"):
            _run(
                mesh,
                {"XM": FlowPatchBC.inlet((0.01, 0.0, 0.0)), "XP": FlowPatchBC.outflow()},
                coupling="coupled",
                max_outer_iter=2,
            )


class TestRotatingWallPhysics:
    """``VelocityPatchBC.rotating_wall``: 参照点まわりの剛体回転する壁（Taylor–Couette）."""

    def test_taylor_couette_matches_analytic(self):
        r1, r2, omega = 0.5, 1.0, 2.0
        mesh = _annulus(16, 96, r1, r2)
        res = _run(
            mesh,
            {
                "INNER": FlowPatchBC.wall(),
                "OUTER": FlowPatchBC.rotating_wall((0.0, 0.0, omega)),
                "ZM": FlowPatchBC.symmetry(),
                "ZP": FlowPatchBC.symmetry(),
            },
            mu=1.0,
            convection="none",
            coupling="coupled",
            tol=1e-9,
            max_outer_iter=30,
            n_nonorthogonal_correctors=3,
        )
        assert res.converged
        xc = mesh.cell_centers
        r = np.linalg.norm(xc[:, :2], axis=1)
        th = np.arctan2(xc[:, 1], xc[:, 0])
        a = omega * r2**2 / (r2**2 - r1**2)
        u_exact = a * r - a * r1**2 / r
        u_th = -res.velocity[:, 0] * np.sin(th) + res.velocity[:, 1] * np.cos(th)
        u_r = res.velocity[:, 0] * np.cos(th) + res.velocity[:, 1] * np.sin(th)
        assert np.max(np.abs(u_th - u_exact)) / np.abs(u_exact).max() < 5e-3
        assert np.max(np.abs(u_r)) < 1e-9 * np.abs(u_th).max()

    def test_rotation_reduces_to_translation_far_from_axis(self):
        """回転中心を遠ざけると面ごとの速度が一様並進に近づく（ω × r の実装確認）."""
        mesh = _periodic_box(4, 8, 0.4, 0.1, x_periodic=False)
        vb = resolve_velocity_boundary(
            mesh, {"YP": VelocityPatchBC.rotating_wall((0.0, 0.0, 1e-6), (0.0, -1.0e6, 0.0))}
        )
        top = mesh.boundary_patches["YP"] - mesh.n_internal_faces
        # ω × r = (0, 0, ω) × (x, +1e6, 0) = (−ω·1e6, ω x, 0)
        assert np.allclose(vb.velocity[top, 0], -1.0, rtol=1e-6)
        assert np.max(np.abs(vb.velocity[top, 1])) < 1e-6


class TestNonNewtonianPhysics:
    """``viscosity_model``: γ̇ から μ を更新する Picard（べき乗則の解析解と照合）."""

    def _power_law_channel(self, k, n, gamma_min=1e-2, ny=32, **kw):
        h, f = 0.1, 2.0
        mesh = _periodic_box(4, ny, 0.4, h)
        res = _run(
            mesh,
            {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()},
            mu=k,
            viscosity_model=PowerLawViscosity(K=k, n=n, gamma_min=gamma_min),
            body_force=(f, 0.0, 0.0),
            convection="none",
            coupling="coupled",
            tol=1e-9,
            max_outer_iter=200,
            **kw,
        )
        y = mesh.cell_centers[:, 1]
        half = h / 2.0
        u_exact = (
            n
            / (n + 1.0)
            * (f / k) ** (1.0 / n)
            * (half ** ((n + 1.0) / n) - np.abs(y - half) ** ((n + 1.0) / n))
        )
        return res, u_exact, mesh

    def test_power_law_channel_matches_analytic(self):
        res, u_exact, _ = self._power_law_channel(0.05, 0.5)
        assert res.converged
        assert np.max(np.abs(res.velocity[:, 0] - u_exact)) / u_exact.max() < 5e-3
        # 壁で最もせん断が強く粘度が下がる（せん断減粘）
        assert res.viscosity is not None and res.strain_rate is not None
        assert res.viscosity.min() < res.viscosity.max()
        assert res.strain_rate.max() == pytest.approx((2.0 * 0.05 / 0.05) ** 2.0, rel=0.1)

    def test_gamma_min_clamp_does_not_change_the_answer(self):
        a, _, _ = self._power_law_channel(0.05, 0.5, gamma_min=1e-2)
        b, u_exact, _ = self._power_law_channel(0.05, 0.5, gamma_min=1e-4)
        assert a.converged and b.converged
        assert np.max(np.abs(a.velocity - b.velocity)) / u_exact.max() < 5e-3

    def test_relaxation_does_not_change_the_fixed_point(self):
        a, u_exact, _ = self._power_law_channel(0.05, 0.7, alpha_mu=1.0)
        b, _, _ = self._power_law_channel(0.05, 0.7, alpha_mu=0.3)
        assert a.converged and b.converged
        # 収束判定は残差なので不動点そのものは残差レベルまでしか一致しない
        assert np.max(np.abs(a.velocity - b.velocity)) / u_exact.max() < 1e-5

    def test_newtonian_limit_matches_constant_viscosity(self):
        mesh = _periodic_box(4, 16, 0.4, 0.1)
        bcs = {"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()}
        kw = dict(
            body_force=(2.0, 0.0, 0.0),
            convection="none",
            coupling="coupled",
            tol=1e-10,
            max_outer_iter=100,
        )
        a = _run(mesh, bcs, mu=0.02, **kw)
        b = _run(
            mesh,
            bcs,
            mu=0.02,
            viscosity_model=CarreauViscosity(mu_0=0.02, mu_inf=0.02, lam=1.0, n=1.0),
            **kw,
        )
        assert a.converged and b.converged
        assert np.max(np.abs(a.velocity - b.velocity)) < 1e-10

    def test_rejects_bad_viscosity_model(self):
        class _Bad:
            def viscosity(self, gamma_dot):
                return np.zeros_like(gamma_dot)

        mesh = _periodic_box(4, 4, 0.4, 0.1)
        with pytest.raises(ValueError, match="非正または非有限"):
            _run(mesh, {}, viscosity_model=_Bad(), max_outer_iter=2)
