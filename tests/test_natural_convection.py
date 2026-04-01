"""NaturalConvectionFDMProcess テスト.

API テスト（契約準拠）と物理テスト（基本物理量の妥当性検証）を含む。
"""

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    NaturalConvectionResult,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess

# ---------------------------------------------------------------------------
# API テスト
# ---------------------------------------------------------------------------


@binds_to(NaturalConvectionFDMProcess)
class TestNaturalConvectionAPI:
    """NaturalConvectionFDMProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        """ProcessMeta が定義されていること."""
        assert NaturalConvectionFDMProcess.meta.name == "NaturalConvectionFDMProcess"
        assert NaturalConvectionFDMProcess.meta.module == "solve"

    def test_process_returns_result(self):
        """process() が NaturalConvectionResult を返すこと."""
        nx, ny, nz = 3, 3, 3
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            max_simple_iter=5,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)
        assert isinstance(result, NaturalConvectionResult)
        assert result.u.shape == (nx, ny, nz)
        assert result.v.shape == (nx, ny, nz)
        assert result.w.shape == (nx, ny, nz)
        assert result.p.shape == (nx, ny, nz)
        assert result.T.shape == (nx, ny, nz)

    def test_data_schema_properties(self):
        """NaturalConvectionInput のプロパティが正しく計算されること."""
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=2.0,
            Lz=0.5,
            nx=10,
            ny=20,
            nz=5,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=0.6,
            beta=0.001,
            T_ref=300.0,
        )
        assert inp.dx == pytest.approx(0.1)
        assert inp.dy == pytest.approx(0.1)
        assert inp.dz == pytest.approx(0.1)
        assert inp.nu == pytest.approx(0.01)
        assert inp.alpha_thermal == pytest.approx(0.6 / 1000.0)
        assert inp.Pr == pytest.approx(0.01 / 0.0006)
        assert not inp.is_transient

    def test_boundary_spec_defaults(self):
        """FluidBoundarySpec のデフォルト値が正しいこと."""
        bc = FluidBoundarySpec()
        assert bc.condition == FluidBoundaryCondition.NO_SLIP
        assert bc.thermal == ThermalBoundaryCondition.ADIABATIC

    def test_result_frozen(self):
        """NaturalConvectionResult が frozen であること."""
        result = NaturalConvectionResult(
            u=np.zeros((2, 2, 2)),
            v=np.zeros((2, 2, 2)),
            w=np.zeros((2, 2, 2)),
            p=np.zeros((2, 2, 2)),
            T=np.ones((2, 2, 2)) * 300.0,
            converged=True,
        )
        with pytest.raises(AttributeError):
            result.converged = False  # type: ignore[misc]

    def test_q_vol_field_default_none(self):
        """q_vol のデフォルト値が None であること."""
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=3,
            ny=3,
            nz=3,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
        )
        assert inp.q_vol is None

    def test_execute_input_immutability(self):
        """execute() が入力データを変更しないこと (C9)."""
        nx, ny, nz = 3, 3, 3
        T0 = np.full((nx, ny, nz), 300.0)
        T0_copy = T0.copy()
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            T0=T0,
            max_simple_iter=3,
        )
        solver = NaturalConvectionFDMProcess()
        solver.execute(inp)
        np.testing.assert_array_equal(T0, T0_copy)


# ---------------------------------------------------------------------------
# 物理テスト
# ---------------------------------------------------------------------------


class TestNaturalConvectionPhysics:
    """物理的妥当性の検証テスト."""

    def test_no_gravity_no_flow(self):
        """重力なしで速度場がほぼゼロであること."""
        nx, ny, nz = 5, 5, 3
        T_hot = 400.0
        T_cold = 300.0
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            gravity=(0.0, 0.0, 0.0),  # 重力なし
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            max_simple_iter=50,
            tol_simple=1e-4,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # 重力がないので速度はほぼゼロ
        assert np.max(np.abs(result.u)) < 0.1
        assert np.max(np.abs(result.v)) < 0.1
        assert np.max(np.abs(result.w)) < 0.1

    def test_pure_conduction_temperature(self):
        """浮力なしの場合、温度場が純粋な伝導解（線形分布）になること."""
        nx, ny, nz = 8, 4, 3
        T_hot = 400.0
        T_cold = 300.0
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,  # 浮力なし
            T_ref=300.0,
            gravity=(0.0, -9.81, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            max_simple_iter=100,
            tol_simple=1e-6,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # beta=0 なので速度はゼロ、温度は x 方向に線形
        # セル中心の温度: T(x) = T_hot - (T_hot-T_cold)*x/Lx
        # FDM のゴーストセル離散化で若干ずれるため許容誤差 15K
        dx = inp.dx
        for i in range(nx):
            x_center = (i + 0.5) * dx
            T_expected = T_hot - (T_hot - T_cold) * x_center / inp.Lx
            T_actual = result.T[i, ny // 2, nz // 2]
            assert abs(T_actual - T_expected) < 15.0, (
                f"i={i}: T_actual={T_actual:.1f}, T_expected={T_expected:.1f}"
            )
        # 単調減少の確認（より厳密な物理チェック）
        T_profile = result.T[:, ny // 2, nz // 2]
        assert np.all(np.diff(T_profile) < 0), "温度プロファイルが単調減少でない"

    def test_buoyancy_drives_vertical_flow(self):
        """浮力により垂直方向の流れが発生すること."""
        nx, ny, nz = 5, 8, 3
        T_hot = 310.0
        T_cold = 290.0
        T_ref = 300.0
        inp = NaturalConvectionInput(
            Lx=0.1,
            Ly=0.1,
            Lz=0.1,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=0.6,
            beta=0.001,
            T_ref=T_ref,
            gravity=(0.0, -9.81, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            max_simple_iter=200,
            tol_simple=1e-4,
            alpha_u=0.3,
            alpha_p=0.1,
            alpha_T=0.7,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # 差分加熱により垂直方向の速度が生じるべき
        # 高温側で上昇、低温側で下降 → v成分が非ゼロ
        v_max = np.max(np.abs(result.v))
        assert v_max > 1e-6, f"v_max={v_max}: 浮力による流れが発生していない"

    def test_solid_region_zero_velocity(self):
        """固体領域で速度がゼロであること."""
        nx, ny, nz = 6, 6, 3
        solid_mask = np.zeros((nx, ny, nz), dtype=bool)
        solid_mask[:2, :, :] = True  # x方向の最初の2セルが固体
        k_solid = np.full((nx, ny, nz), 50.0)  # 固体の熱伝導率

        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=0.6,
            beta=0.001,
            T_ref=300.0,
            gravity=(0.0, -9.81, 0.0),
            solid_mask=solid_mask,
            k_solid=k_solid,
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=400.0,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=300.0,
            ),
            max_simple_iter=50,
            tol_simple=1e-3,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # 固体領域で速度=0
        assert np.allclose(result.u[:2, :, :], 0.0), "固体領域でu≠0"
        assert np.allclose(result.v[:2, :, :], 0.0), "固体領域でv≠0"
        assert np.allclose(result.w[:2, :, :], 0.0), "固体領域でw≠0"

    def test_temperature_bounded(self):
        """温度が境界値の範囲内に収まること."""
        nx, ny, nz = 5, 5, 3
        T_hot = 310.0
        T_cold = 290.0
        inp = NaturalConvectionInput(
            Lx=0.1,
            Ly=0.1,
            Lz=0.1,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=0.6,
            beta=0.001,
            T_ref=300.0,
            gravity=(0.0, -9.81, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            max_simple_iter=100,
            tol_simple=1e-4,
            alpha_u=0.1,
            alpha_p=0.05,
            alpha_T=0.5,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # 温度は境界値の近傍に収まるべき（少しのマージン許容）
        assert not np.any(np.isnan(result.T)), "温度にNaNが含まれる"
        margin = 15.0
        assert np.min(result.T) > T_cold - margin, f"T_min={np.min(result.T):.1f}"
        assert np.max(result.T) < T_hot + margin, f"T_max={np.max(result.T):.1f}"

    def test_q_vol_raises_temperature(self):
        """体積熱生成で温度が上昇すること."""
        nx, ny, nz = 5, 5, 3
        T_ref = 300.0
        q_vol = np.zeros((nx, ny, nz))
        q_vol[2, 2, :] = 10000.0  # 中心に発熱

        inp = NaturalConvectionInput(
            Lx=0.1,
            Ly=0.1,
            Lz=0.1,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=T_ref,
            gravity=(0.0, 0.0, 0.0),
            q_vol=q_vol,
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_ref,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_ref,
            ),
            bc_ym=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_ref,
            ),
            bc_yp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_ref,
            ),
            max_simple_iter=100,
            tol_simple=1e-4,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # 中心セルの温度が基準温度より高いこと
        T_center = result.T[2, 2, nz // 2]
        assert T_center > T_ref + 0.1, f"T_center={T_center:.2f}: q_volによる温度上昇がない"

    def test_q_vol_none_backward_compatible(self):
        """q_vol=None で既存動作に影響がないこと."""
        nx, ny, nz = 3, 3, 3
        inp_no_q = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
            max_simple_iter=10,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp_no_q)

        # q_vol=None でも正常動作
        assert isinstance(result, NaturalConvectionResult)
        assert not np.any(np.isnan(result.T))

    def test_residual_history_populated(self):
        """残差履歴が記録されること."""
        nx, ny, nz = 3, 3, 3
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            max_simple_iter=10,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        assert "u" in result.residual_history
        assert "v" in result.residual_history
        assert "mass" in result.residual_history
        assert len(result.residual_history["u"]) > 0

    def test_transient_timestep(self):
        """非定常解析でタイムステップが正しく進むこと."""
        nx, ny, nz = 3, 3, 3
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            dt=0.1,
            t_end=0.3,
            max_simple_iter=5,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        assert result.n_timesteps == 3

    def test_symmetry_boundary(self):
        """対称境界条件が機能すること."""
        nx, ny, nz = 5, 5, 3
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
            bc_zm=FluidBoundarySpec(
                condition=FluidBoundaryCondition.SYMMETRY,
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            bc_zp=FluidBoundarySpec(
                condition=FluidBoundaryCondition.SYMMETRY,
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            max_simple_iter=10,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # エラーなく完了すること
        assert isinstance(result, NaturalConvectionResult)

    @pytest.mark.slow
    def test_differentially_heated_cavity_nusselt(self):
        """差分加熱キャビティのNusselt数が de Vahl Davis (1983) の値に近いこと.

        Ra=1000 のケース: Nu ≈ 1.118
        粗いメッシュのため大きめの許容誤差を設定。
        """
        # Ra = g*beta*deltaT*L^3 / (nu*alpha)
        # Ra = 1000 を実現するパラメータ
        L = 0.1
        T_hot = 310.0
        T_cold = 290.0
        delta_T = T_hot - T_cold
        T_ref = 300.0
        rho = 1.0
        mu = 0.01
        Cp = 1000.0
        k_fluid = 1.0
        nu = mu / rho
        alpha_th = k_fluid / (rho * Cp)
        # beta = Ra * nu * alpha / (g * delta_T * L^3)
        g = 9.81
        Ra_target = 1000.0
        beta = Ra_target * nu * alpha_th / (g * delta_T * L**3)

        nx, ny, nz = 12, 12, 3
        inp = NaturalConvectionInput(
            Lx=L,
            Ly=L,
            Lz=L,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=rho,
            mu=mu,
            Cp=Cp,
            k_fluid=k_fluid,
            beta=beta,
            T_ref=T_ref,
            gravity=(0.0, -g, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            bc_ym=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            bc_yp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            bc_zm=FluidBoundarySpec(
                condition=FluidBoundaryCondition.SYMMETRY,
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            bc_zp=FluidBoundarySpec(
                condition=FluidBoundaryCondition.SYMMETRY,
                thermal=ThermalBoundaryCondition.ADIABATIC,
            ),
            max_simple_iter=3000,
            tol_simple=1e-4,
            alpha_u=0.2,
            alpha_p=0.05,
            alpha_T=0.5,
        )
        solver = NaturalConvectionFDMProcess()
        result = solver.process(inp)

        # Nusselt数の計算: Nu = -L/(delta_T) * ∂T/∂x|_{x=0} の面平均
        # 高温壁面での温度勾配
        dx = inp.dx
        # 壁面での dT/dx ≈ (T[0,:,:] - T_hot) / (dx/2) ... ゴーストセル法
        dTdx_hot = (result.T[0, :, :] - T_hot) / (dx / 2.0)
        # Nu = -k * dT/dx * L / (k * delta_T)
        Nu_local = -dTdx_hot * L / delta_T
        Nu_avg = np.mean(Nu_local)

        # de Vahl Davis (1983): Ra=1000 → Nu ≈ 1.118
        # 粗いメッシュ・FDM なので許容誤差は大きめ（20%以内）
        Nu_ref = 1.118
        assert abs(Nu_avg - Nu_ref) / Nu_ref < 0.20, (
            f"Nu_avg={Nu_avg:.3f}, Nu_ref={Nu_ref:.3f}, "
            f"相対誤差={abs(Nu_avg - Nu_ref) / Nu_ref * 100:.1f}%"
        )


def _run_cavity_benchmark(Ra, nx=12, ny=12, nz=3, max_iter=3000):
    """差分加熱キャビティのベンチマーク実行ヘルパー."""
    L = 0.1
    T_hot, T_cold = 310.0, 290.0
    delta_T = T_hot - T_cold
    T_ref = 300.0
    rho, mu = 1.0, 0.01
    Cp, k_fluid = 1000.0, 1.0
    nu = mu / rho
    alpha_th = k_fluid / (rho * Cp)
    g = 9.81
    beta = Ra * nu * alpha_th / (g * delta_T * L**3)

    inp = NaturalConvectionInput(
        Lx=L,
        Ly=L,
        Lz=L,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=rho,
        mu=mu,
        Cp=Cp,
        k_fluid=k_fluid,
        beta=beta,
        T_ref=T_ref,
        gravity=(0.0, -g, 0.0),
        bc_xm=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_hot,
        ),
        bc_xp=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.DIRICHLET,
            temperature=T_cold,
        ),
        bc_ym=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_yp=FluidBoundarySpec(
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_zm=FluidBoundarySpec(
            condition=FluidBoundaryCondition.SYMMETRY,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        bc_zp=FluidBoundarySpec(
            condition=FluidBoundaryCondition.SYMMETRY,
            thermal=ThermalBoundaryCondition.ADIABATIC,
        ),
        max_simple_iter=max_iter,
        tol_simple=1e-5,
        alpha_u=0.15,
        alpha_p=0.05,
        alpha_T=0.5,
    )
    result = NaturalConvectionFDMProcess().process(inp)
    dx = inp.dx
    dTdx_hot = (result.T[0, :, :] - T_hot) / (dx / 2.0)
    Nu_local = -dTdx_hot * L / delta_T
    return float(np.mean(Nu_local)), result


class TestCavityBenchmark:
    """差分加熱キャビティベンチマーク（de Vahl Davis 1983）の定量検証."""

    @pytest.mark.slow
    def test_ra_1e3_nusselt(self):
        """Ra=10³: Nu≈1.118 (20%以内)."""
        Nu, result = _run_cavity_benchmark(1000)
        assert abs(Nu - 1.118) / 1.118 < 0.20, (
            f"Ra=1000: Nu={Nu:.3f} (ref=1.118, err={abs(Nu - 1.118) / 1.118 * 100:.1f}%)"
        )

    @pytest.mark.slow
    def test_ra_1e4_nusselt(self):
        """Ra=10⁴: Nu≈2.243 (20%以内)."""
        Nu, result = _run_cavity_benchmark(10000)
        assert abs(Nu - 2.243) / 2.243 < 0.20, (
            f"Ra=10000: Nu={Nu:.3f} (ref=2.243, err={abs(Nu - 2.243) / 2.243 * 100:.1f}%)"
        )

    @pytest.mark.slow
    def test_ra_1e3_temperature_bounded(self):
        """Ra=10³: 温度が境界値範囲内."""
        Nu, result = _run_cavity_benchmark(1000)
        assert result.T.min() >= 290.0 - 5.0
        assert result.T.max() <= 310.0 + 5.0

    @pytest.mark.slow
    def test_ra_1e4_flow_symmetry(self):
        """Ra=10⁴: 流れ場が概ね反対称."""
        _, result = _run_cavity_benchmark(10000)
        nx, ny, nz = result.u.shape
        # x方向速度は中心面(x=L/2)で概ねゼロ付近
        mid_x = nx // 2
        u_mid = np.abs(result.u[mid_x, :, :]).mean()
        u_max = np.abs(result.u).max()
        # 中心面のuは最大値の50%以下
        assert u_mid < 0.5 * u_max or u_max < 1e-10


class TestTransientNaturalConvection:
    """非定常自然対流テスト."""

    def test_transient_temperature_evolution(self):
        """非定常: 温度場が時間とともに定常解に近づく."""
        nx, ny, nz = 5, 5, 3
        T_hot, T_cold = 310.0, 290.0
        inp = NaturalConvectionInput(
            Lx=0.1,
            Ly=0.1,
            Lz=0.1,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
            gravity=(0.0, 0.0, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            dt=0.01,
            t_end=0.5,
            max_simple_iter=20,
            tol_simple=1e-4,
            alpha_u=0.3,
            alpha_p=0.1,
            alpha_T=0.7,
        )
        result = NaturalConvectionFDMProcess().process(inp)

        # 浮力なし → 純粋な伝導 → 温度は境界値範囲内
        assert not np.any(np.isnan(result.T))
        assert result.T.min() >= T_cold - 1.0
        assert result.T.max() <= T_hot + 1.0
        # タイムステップが進んでいる
        assert result.n_timesteps >= 10

    def test_transient_buoyancy_onset(self):
        """非定常: 浮力項がある場合に流れが発生する."""
        nx, ny, nz = 5, 5, 3
        T_hot, T_cold = 320.0, 280.0
        inp = NaturalConvectionInput(
            Lx=0.1,
            Ly=0.1,
            Lz=0.1,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.001,
            T_ref=300.0,
            gravity=(0.0, -9.81, 0.0),
            bc_xm=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_hot,
            ),
            bc_xp=FluidBoundarySpec(
                thermal=ThermalBoundaryCondition.DIRICHLET,
                temperature=T_cold,
            ),
            dt=0.01,
            t_end=0.2,
            max_simple_iter=20,
            tol_simple=1e-4,
            alpha_u=0.2,
            alpha_p=0.05,
            alpha_T=0.5,
        )
        result = NaturalConvectionFDMProcess().process(inp)

        # 浮力により流れが発生しているはず
        max_vel = max(np.abs(result.u).max(), np.abs(result.v).max())
        assert max_vel > 1e-6, f"流れが発生していない: max_vel={max_vel:.2e}"

    def test_transient_residual_history_length(self):
        """非定常: 残差履歴がタイムステップ×SIMPLE反復分記録される."""
        nx, ny, nz = 3, 3, 3
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
            dt=0.1,
            t_end=0.5,
            max_simple_iter=5,
            tol_simple=1e-10,
        )
        result = NaturalConvectionFDMProcess().process(inp)

        # 5 timesteps × (1以上のSIMPLE反復) = 5以上のエントリ
        assert len(result.residual_history["u"]) >= 5
        assert result.n_timesteps == 5
