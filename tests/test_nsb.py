"""nsb（手元構成ミラー）のテスト: 形状・BC 変換・収束・Process ソルバーとの一致."""

from __future__ import annotations

import numpy as np
import pytest

from nsb import (
    BC,
    NSBInput,
    NSBSettings,
    east_span,
    make_case,
    make_uturn_h,
    north_span,
    solve_steady,
)
from nsb.assembly import BrinkmanDiscretization
from nsb.geo import LX, LY, uturn_bc_preset
from nsb.utils import inlet_velocity, mass_balance
from xkep_cae_fluid import brinkman_flow as xkep_bf
from xkep_cae_fluid.brinkman_flow import BrinkmanFlowFVMProcess, BrinkmanSolverSettings


def to_xkep_flow_input(inp: NSBInput) -> xkep_bf.BrinkmanFlowInput:
    """nsb（独立コピー）の入力を xkep_cae_fluid 側の型に変換する（Process ソルバー比較用）.

    nsb.data と xkep_cae_fluid.brinkman_flow.data は同一内容のコピーだが別クラスなので、
    Enum メンバの同一性比較（``kind is BoundaryKind.X``）が跨げない。名前で詰め替える。
    """
    fi = inp.to_flow_input()
    patches = tuple(
        xkep_bf.BoundaryPatch(
            kind=xkep_bf.BoundaryKind[p.kind.name],
            mask=p.mask,
            velocity=p.velocity,
            mass_flow=p.mass_flow,
            pressure=p.pressure,
            conductance=p.conductance,
            weight=p.weight,
            name=p.name,
        )
        for p in fi.boundaries
    )
    return xkep_bf.BrinkmanFlowInput(
        nx=fi.nx,
        ny=fi.ny,
        thickness=fi.thickness,
        geometry=xkep_bf.BrinkmanGeometry(lx=fi.geometry.lx, ly=fi.geometry.ly),
        rho=fi.rho,
        mu=fi.mu,
        mu_brinkman=fi.mu_brinkman,
        brinkman_factor=fi.brinkman_factor,
        u_inlet=fi.u_inlet,
        boundaries=patches,
        settings=BrinkmanSolverSettings(),
    )


class TestNSBAPI:
    def test_uturn_h_and_bc(self):
        h = make_uturn_h(72, 48)
        assert h.shape == (72, 48)
        assert h[0, 30] == pytest.approx(1e-3)  # inlet 往路
        assert h[0, 12] == pytest.approx(1e-3)  # outlet 復路
        assert h[0, 20] == pytest.approx(1e-5)  # 間の閉塞部
        bc = uturn_bc_preset(48, u_in=1.0)
        inp = NSBInput(nx=72, ny=48, lx=LX, ly=LY, h=h, bc=bc)
        W = BrinkmanDiscretization(inp.to_flow_input()).sides["W"]
        yc = (np.arange(48) + 0.5) * LY / 48
        assert np.array_equal(W.is_inlet, (yc > 0.25) & (yc < 0.35))
        assert np.array_equal(W.is_outlet, (yc > 0.05) & (yc < 0.15))

    def test_bc_requires_exactly_one_inlet_spec(self):
        with pytest.raises(ValueError):
            uturn_bc_preset(8)
        with pytest.raises(ValueError):
            uturn_bc_preset(8, u_in=1.0, mass_flow=1.0)

    def test_mass_flow_inlet_moves_with_position(self):
        """流量固定で inlet の位置・幅を変えると換算速度が幅に反比例する."""
        mdot = 0.02
        a = make_case("flat", 1, mass_flow=mdot, inlet_y=(0.25, 0.35))
        b = make_case("flat", 1, mass_flow=mdot, inlet_y=(0.10, 0.30))
        assert inlet_velocity(b) == pytest.approx(0.5 * inlet_velocity(a), rel=1e-12)
        assert inlet_velocity(a) == pytest.approx(mdot / (1000.0 * 1e-3 * 0.1), rel=1e-12)

    def test_returns_without_raising_on_failure(self):
        inp = make_case("uturn", 1, 2.0, NSBSettings(newton_max_iter=1))
        res = solve_steady(inp, log=None)
        assert not res.converged
        assert res.failure_reason == "max_iter"
        assert len(res.residual_history) == 2


class TestNSBConvergence:
    @pytest.mark.slow
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
        ref = BrinkmanFlowFVMProcess().execute(to_xkep_flow_input(inp))
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


class TestNSBBoundaryPatches:
    def test_mass_flow_inlet_on_top_wall_outlet_on_right_wall(self):
        mdot = 0.01
        bc = BC(
            patches=(
                BC.mass_flow_inlet(north_span(0.1, 0.2, LY), mdot),
                BC.pressure_outlet(east_span(0.1, 0.2, LX)),
            )
        )
        inp = make_case(
            "flat",
            1,
            bc=bc,
            settings=NSBSettings(
                velocity_floor=0.01, pseudo_time_in_residual=False, newton_max_iter=60
            ),
        )
        res = solve_steady(inp, log=None)
        assert res.converged, res.failure_reason
        assert res.mass_in == pytest.approx(mdot / 1e-3, rel=1e-6)
        assert mass_balance(res) == pytest.approx(1.0, rel=1e-6)


class TestNSBInputValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            NSBInput(
                nx=4, ny=4, lx=LX, ly=LY, h=np.ones((3, 4)) * 1e-3, bc=uturn_bc_preset(4, 1.0)
            ).to_flow_input()
