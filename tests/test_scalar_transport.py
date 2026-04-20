"""ScalarTransportProcess テスト.

API テスト（Process 契約準拠）、収束テスト（1D 純拡散解析解）、
物理テスト（1D 純対流の質量保存）を含む。
"""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.scalar_transport.data import (
    ScalarBoundaryCondition,
    ScalarBoundarySpec,
    ScalarFieldSpec,
    ScalarTransportInput,
    ScalarTransportResult,
)
from xkep_cae_fluid.scalar_transport.solver import ScalarTransportProcess


def _zero_velocity(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (nx, ny, nz)
    return np.zeros(shape), np.zeros(shape), np.zeros(shape)


# ---------------------------------------------------------------------------
# API テスト
# ---------------------------------------------------------------------------


@binds_to(ScalarTransportProcess)
class TestScalarTransportAPI:
    """ScalarTransportProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        """ProcessMeta が定義されていること."""
        assert ScalarTransportProcess.meta.name == "ScalarTransportFDM"
        assert ScalarTransportProcess.meta.module == "solve"

    def test_process_returns_result(self):
        """process() が ScalarTransportResult を返すこと."""
        nx, ny, nz = 3, 3, 3
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=1.0,
            phi0=np.ones((nx, ny, nz)),
        )
        inp = ScalarTransportInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)
        assert isinstance(result, ScalarTransportResult)
        assert result.phi.shape == (nx, ny, nz)

    def test_zero_velocity_adiabatic_no_source_holds_initial(self):
        """ゼロ速度・全面断熱・ソース無しの非定常で初期値が維持されること.

        定常の adiabatic+無ソースは解が定数倍の自由度をもつ不良設定なので
        非定常で任意ステップ進めても初期値が維持されることを検証する。
        """
        nx, ny, nz = 4, 4, 4
        phi_init = 2.5
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=1.0,
            phi0=np.full((nx, ny, nz), phi_init),
        )
        inp = ScalarTransportInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            dt=0.1,
            t_end=0.3,
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)
        np.testing.assert_allclose(result.phi, phi_init, atol=1e-6)

    def test_transient_returns_n_steps(self):
        """非定常解析でタイムステップ数が返ること."""
        nx, ny, nz = 3, 3, 3
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=0.01,
            phi0=np.zeros((nx, ny, nz)),
        )
        inp = ScalarTransportInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            dt=0.1,
            t_end=0.5,
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)
        assert result.n_timesteps == 5


# ---------------------------------------------------------------------------
# 収束テスト: 1D 純拡散解析解
# ---------------------------------------------------------------------------


class TestScalarTransportConvergence:
    """解析解と突き合わせる収束テスト."""

    def test_1d_steady_pure_diffusion_linear(self):
        """1D 純拡散・両端 Dirichlet → 線形プロファイル.

        支配方程式: Γ d²φ/dx² = 0, φ(0)=φ_L, φ(Lx)=φ_R
        解析解: φ(x) = φ_L + (φ_R - φ_L) · x / Lx
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        phi_L, phi_R = 10.0, 30.0
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=1.0,
            phi0=np.zeros((nx, ny, nz)),
        )
        inp = ScalarTransportInput(
            Lx=Lx,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            bc_xm=ScalarBoundarySpec(condition=ScalarBoundaryCondition.DIRICHLET, value=phi_L),
            bc_xp=ScalarBoundarySpec(condition=ScalarBoundaryCondition.DIRICHLET, value=phi_R),
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)

        # セル中心位置 x = (i + 0.5) * dx
        dx = Lx / nx
        x_centers = (np.arange(nx) + 0.5) * dx
        phi_analytic = phi_L + (phi_R - phi_L) * x_centers / Lx
        np.testing.assert_allclose(result.phi[:, 0, 0], phi_analytic, atol=1e-6, rtol=1e-4)

    def test_neumann_flux_adds_expected_increment(self):
        """1D: xm Dirichlet + xp Neumann(流入 flux) で勾配一致.

        定常: Γ dφ/dx = flux (一定) → φ(x) = φ_L + (flux/Γ)·x
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        phi_L = 5.0
        Gamma = 2.0
        flux = 4.0  # 正=領域への流入
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=Gamma,
            phi0=np.zeros((nx, ny, nz)),
        )
        inp = ScalarTransportInput(
            Lx=Lx,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            bc_xm=ScalarBoundarySpec(condition=ScalarBoundaryCondition.DIRICHLET, value=phi_L),
            bc_xp=ScalarBoundarySpec(condition=ScalarBoundaryCondition.NEUMANN, flux=flux),
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)

        # xp 面での勾配: (φ_xp_face - φ_last_cell) / (dx/2) = flux / Γ
        # 線形プロファイルなので勾配一定: dφ/dx = flux/Γ
        dx = Lx / nx
        phi_cells = result.phi[:, 0, 0]
        slopes = np.diff(phi_cells) / dx
        np.testing.assert_allclose(slopes, flux / Gamma, atol=1e-4)


# ---------------------------------------------------------------------------
# 物理テスト: 1D 対流の質量保存 + Robin BC
# ---------------------------------------------------------------------------


class TestScalarTransportPhysics:
    """物理的妥当性テスト."""

    def test_robin_bc_equilibrium(self):
        """全面 Robin BC（ヘンリー則風）で平衡濃度 phi_inf に漸近.

        ソース無し・速度ゼロ・全面 Robin BC なら、定常解は φ_inf で一様。
        """
        nx, ny, nz = 6, 6, 6
        phi_inf = 42.0
        u, v, w = _zero_velocity(nx, ny, nz)
        field = ScalarFieldSpec(
            name="CO2",
            diffusivity=1.0,
            phi0=np.zeros((nx, ny, nz)),
        )
        robin = ScalarBoundarySpec(
            condition=ScalarBoundaryCondition.ROBIN, h_mass=1.0, phi_inf=phi_inf
        )
        inp = ScalarTransportInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            bc_xm=robin,
            bc_xp=robin,
            bc_ym=robin,
            bc_yp=robin,
            bc_zm=robin,
            bc_zp=robin,
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)
        np.testing.assert_allclose(result.phi, phi_inf, atol=1e-4)

    def test_1d_pure_convection_conserves_integral(self):
        """1D 純対流（拡散を極小）で領域全体のスカラー合計が保存する.

        上流 Dirichlet=0、下流 Neumann=0（流出）、初期に矩形波を注入、
        ソース無し、定常的に運搬される場を短時間で計算。
        ここでは「質量の湧き出しが生じていないこと」を一般的に確認する:
        境界を通過した分を除き、内部領域でのスカラー生成は起きない。
        """
        nx, ny, nz = 40, 1, 1
        Lx = 1.0
        dx = Lx / nx
        u_val = 0.1
        u = np.full((nx, ny, nz), u_val)
        v = np.zeros_like(u)
        w = np.zeros_like(u)

        # 中央に矩形波の初期プロファイル
        phi0 = np.zeros((nx, ny, nz))
        phi0[15:25, 0, 0] = 1.0

        field = ScalarFieldSpec(
            name="tracer",
            diffusivity=1e-6,  # ほぼ純対流
            phi0=phi0,
        )
        inp = ScalarTransportInput(
            Lx=Lx,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            u=u,
            v=v,
            w=w,
            field=field,
            bc_xm=ScalarBoundarySpec(condition=ScalarBoundaryCondition.DIRICHLET, value=0.0),
            bc_xp=ScalarBoundarySpec(condition=ScalarBoundaryCondition.NEUMANN, flux=0.0),
            dt=0.1,
            t_end=0.2,  # 2 ステップだけ
        )
        solver = ScalarTransportProcess()
        result = solver.process(inp)

        # 境界外への流出がまだ小さい時刻では、合計濃度は減少のみ（増加はなし）。
        total_initial = phi0.sum() * dx
        total_final = result.phi.sum() * dx
        assert total_final <= total_initial + 1e-10, (
            "内部領域でスカラーが湧き出している (質量保存違反)"
        )
        # 短時間なら大部分が残っている
        assert total_final > 0.5 * total_initial
