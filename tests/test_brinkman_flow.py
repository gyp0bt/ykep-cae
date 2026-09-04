"""BrinkmanFlowFVMProcess / UTurnThicknessProcess テスト.

API テスト（Process 契約・ヤコビアン整合）、収束テスト（U ターン基本ケース）、
物理テスト（質量保存・Hele-Shaw 圧力損失・閉塞部の無流れ）を含む。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.brinkman_flow import (
    BrinkmanFlowFVMProcess,
    BrinkmanFlowInput,
    BrinkmanFlowResult,
    BrinkmanGeometry,
    BrinkmanSolverSettings,
    ConvectionSchemeType,
    JacobianMode,
    ThicknessInput,
    ThicknessModel,
    ThicknessResult,
    ThicknessSpec,
    UTurnThicknessProcess,
)
from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization
from xkep_cae_fluid.core.testing import binds_to


def _thickness(nx: int, ny: int, model: ThicknessModel, **kw) -> ThicknessResult:
    return UTurnThicknessProcess().execute(
        ThicknessInput(nx, ny, ThicknessSpec(model=model, **kw), BrinkmanGeometry())
    )


def _input(
    nx: int, ny: int, model: ThicknessModel, u_inlet: float, **settings
) -> BrinkmanFlowInput:
    th = _thickness(nx, ny, model).thickness
    return BrinkmanFlowInput(
        nx=nx, ny=ny, thickness=th, u_inlet=u_inlet, settings=BrinkmanSolverSettings(**settings)
    )


# ---------------------------------------------------------------------------
# UTurnThicknessProcess
# ---------------------------------------------------------------------------


@binds_to(UTurnThicknessProcess)
class TestUTurnThicknessAPI:
    """厚さ場ビルダーの API テスト."""

    def test_meta(self):
        assert UTurnThicknessProcess.meta.name == "UTurnThickness"
        assert UTurnThicknessProcess.meta.module == "pre"

    def test_flat_is_uniform(self):
        res = _thickness(36, 24, ThicknessModel.FLAT)
        assert res.thickness.shape == (36, 24)
        assert np.all(res.thickness == 1.0e-3)
        assert res.channel_mask.all()

    def test_uturn_connects_inlet_to_outlet(self):
        res = _thickness(72, 48, ThicknessModel.UTURN)
        h = res.thickness
        yc = (np.arange(48) + 0.5) * (0.4 / 48)
        inlet = (yc > 0.25) & (yc < 0.35)
        outlet = (yc > 0.05) & (yc < 0.15)
        assert np.all(h[0, inlet] == 1.0e-3)
        assert np.all(h[0, outlet] == 1.0e-3)
        assert np.all(h[0, ~(inlet | outlet)] == 1.0e-5)
        # 中央帯（inlet と outlet の間）は左側で閉塞、右端の折返しで開通
        mid = (yc > 0.15) & (yc < 0.25)
        assert np.all(h[0, mid] == 1.0e-5)
        assert np.all(h[-1, mid] == 1.0e-3)
        assert 0.3 < res.channel_mask.mean() < 0.7


# ---------------------------------------------------------------------------
# BrinkmanFlowFVMProcess
# ---------------------------------------------------------------------------


@binds_to(BrinkmanFlowFVMProcess)
class TestBrinkmanFlowAPI:
    """Process 契約とヤコビアン整合."""

    def test_meta(self):
        assert BrinkmanFlowFVMProcess.meta.name == "BrinkmanFlowFVM"
        assert BrinkmanFlowFVMProcess.meta.module == "solve"

    def test_input_validation(self):
        with pytest.raises(ValueError):
            BrinkmanFlowInput(nx=4, ny=4, thickness=np.ones((4, 3)))
        with pytest.raises(ValueError):
            BrinkmanFlowInput(nx=4, ny=4, thickness=np.zeros((4, 4)))

    def test_first_order_jacobian_matches_finite_difference(self):
        """RC 係数が速度に依存しない状態（貫通項支配）で J1 が FD ヤコビアンと一致すること."""
        nx, ny = 12, 8
        th = np.full((nx, ny), 1.0e-5)
        inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, u_inlet=0.1)
        disc = BrinkmanDiscretization(inp)
        n = disc.n
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(0, 10.0, n)])
        sch = ConvectionSchemeType.FIRST_ORDER_UPWIND
        st = disc.compute_state(x, sch, 5.0)
        J = disc.jacobian_first_order(st).toarray()
        Jfd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jfd[:, k] = (disc.residual(x + e, sch, 5.0) - disc.residual(x - e, sch, 5.0)) / (2 * hk)
        scale = np.abs(Jfd).max(axis=0) + 1e-12
        assert np.abs(J - Jfd).max(axis=0).max() < 1e-4 * scale.max()
        # 各列相対誤差
        assert np.all(np.abs(J - Jfd).max(axis=0) / scale < 1e-3)

    def test_returns_result_without_raising_on_failure(self):
        """反復上限に達しても例外を出さず converged=False を返すこと."""
        inp = _input(24, 16, ThicknessModel.UTURN, 0.1, newton_max_iter=1, newton_tol=1e-12)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert isinstance(res, BrinkmanFlowResult)
        assert res.converged is False
        assert res.failure_reason == "max_iter"
        assert len(res.residual_history) == 2


class TestBrinkmanFlowConvergence:
    """基本ケースの収束."""

    @pytest.mark.parametrize("mode", [JacobianMode.JFNK, JacobianMode.DEFECT_CORRECTION])
    def test_uturn_low_speed_converges(self, mode: JacobianMode):
        inp = _input(36, 24, ThicknessModel.UTURN, 0.1, jacobian_mode=mode, newton_max_iter=60)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.residual_history[-1] / res.residual_history[0] < 1e-6

    def test_first_order_scheme_converges(self):
        inp = _input(
            36,
            24,
            ThicknessModel.UTURN,
            0.1,
            convection_scheme=ConvectionSchemeType.FIRST_ORDER_UPWIND,
            newton_max_iter=60,
        )
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason


class TestBrinkmanFlowPhysics:
    """物理的妥当性."""

    @pytest.fixture(scope="class")
    def uturn_result(self) -> tuple[BrinkmanFlowInput, BrinkmanFlowResult]:
        inp = _input(72, 48, ThicknessModel.UTURN, 0.1, newton_max_iter=60)
        return inp, BrinkmanFlowFVMProcess().execute(inp)

    def test_mass_conservation(self, uturn_result):
        inp, res = uturn_result
        assert res.converged
        m_expected = inp.rho * inp.u_inlet * 0.1
        assert abs(res.mass_in - m_expected) / m_expected < 1e-6
        assert abs(res.mass_out - res.mass_in) / m_expected < 1e-5

    def test_no_flow_in_blocked_region(self, uturn_result):
        inp, res = uturn_result
        blocked = inp.thickness < 1.0e-4
        speed = np.hypot(res.u, res.v)
        assert speed[blocked].max() < 1e-2 * inp.u_inlet

    def test_hele_shaw_pressure_drop(self, uturn_result):
        """低速では圧力損失 ≈ (12 mu_b / h²) U × 経路長（Hele-Shaw、慣性項無視）."""
        inp, res = uturn_result
        k = inp.brinkman_factor * inp.mu_brinkman / 1.0e-6
        # 経路長: 往路 (0.7-0.05) + 折返し 0.2 + 復路 (0.7-0.05) ≈ 1.5 m（近似）
        dp_est = k * inp.u_inlet * 1.5
        yc = (np.arange(inp.ny) + 0.5) * inp.dy
        inlet = (yc > 0.25) & (yc < 0.35)
        p_in = res.p[0, inlet].mean()
        assert 0.5 * dp_est < p_in < 1.5 * dp_est
