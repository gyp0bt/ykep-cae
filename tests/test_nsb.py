"""nsb（手元構成ミラー）のテスト: 形状・BC 変換・収束・Process ソルバーとの一致."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nsb import BC, FaceType, NSBInput, NSBSettings, make_case, make_uturn_h, solve_steady
from nsb.geo import LX, LY, uturn_bc_preset
from xkep_cae_fluid.brinkman_flow import BrinkmanFlowFVMProcess, BrinkmanSolverSettings


class TestNSBAPI:
    def test_uturn_h_and_bc(self):
        h = make_uturn_h(72, 48)
        assert h.shape == (72, 48)
        assert h[0, 30] == pytest.approx(1e-3)  # inlet 往路
        assert h[0, 12] == pytest.approx(1e-3)  # outlet 復路
        assert h[0, 20] == pytest.approx(1e-5)  # 間の閉塞部
        bc = uturn_bc_preset(48, 1.0)
        g = bc.to_geometry(LX, LY)
        assert (g.inlet_y0, g.inlet_y1) == pytest.approx((0.25, 0.35))
        assert (g.outlet_y0, g.outlet_y1) == pytest.approx((0.05, 0.15))

    def test_bc_requires_contiguous_inlet(self):
        west = np.array([FaceType.WALL] * 8, dtype=object)
        west[1] = FaceType.VELOCITY_INLET
        west[3] = FaceType.VELOCITY_INLET
        west[6] = FaceType.PRESSURE_OUTLET
        with pytest.raises(ValueError):
            BC(west=west, u_inlet=1.0).to_geometry(LX, LY)

    def test_returns_without_raising_on_failure(self):
        inp = make_case("uturn", 1, 2.0, NSBSettings(newton_max_iter=1))
        res = solve_steady(inp, log=None)
        assert not res.converged
        assert res.failure_reason == "max_iter"
        assert len(res.residual_history) == 2


class TestNSBConvergence:
    def test_fixed_config_matches_process_solver(self):
        """速度下限あり・擬似時間項は対角のみ、の構成は Process ソルバーと同じ解に収束する."""
        u_in = 0.1
        inp = make_case(
            "uturn",
            1,
            u_in,
            NSBSettings(velocity_floor=0.1 * u_in, pseudo_time_in_residual=False, cfl_init=5.0),
        )
        res = solve_steady(inp, log=None)
        assert res.converged, res.failure_reason
        ref = BrinkmanFlowFVMProcess().execute(
            replace(inp.to_flow_input(), settings=BrinkmanSolverSettings())
        )
        assert ref.converged
        scale = np.abs(ref.u).max()
        assert np.abs(res.u - ref.u).max() < 1e-4 * scale
        assert np.abs(res.p - ref.p).max() < 1e-4 * np.abs(ref.p).max()

    def test_pseudo_time_in_residual_converges_to_same_steady_state(self):
        """残差に擬似時間項を含めても収束すれば定常残差も小さい（u_prev 更新が正しい）."""
        u_in = 0.1
        inp = make_case(
            "uturn",
            1,
            u_in,
            NSBSettings(velocity_floor=0.1 * u_in, pseudo_time_in_residual=True, cfl_init=5.0),
        )
        res = solve_steady(inp, log=None)
        assert res.converged, res.failure_reason
        assert res.rel_steady_residual < 1e-5


class TestNSBInputValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            NSBInput(
                nx=4, ny=4, lx=LX, ly=LY, h=np.ones((3, 4)) * 1e-3, bc=uturn_bc_preset(4, 1.0)
            ).to_flow_input()
