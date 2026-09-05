"""NavierStokesFVMProcess のテスト（API + 物理ベンチマーク）."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.fvm import diffusive_face_flux, resolve_boundary
from xkep_cae_fluid.incompressible import (
    FlowPatchBC,
    NavierStokesFVMInput,
    NavierStokesFVMProcess,
    NavierStokesFVMResult,
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
            _run(mesh, {}, coupling="piso", max_outer_iter=1)
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
        res = _run(mesh, _channel_bcs("slip"), permeability=np.full(mesh.n_cells, K))
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
