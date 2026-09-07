"""nsb.fastres（残差評価の numba カーネル）が numpy 経路と同じ値を返すことの検査."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nsb import BC, NSBSettings, disk_mask, make_case, north_span, solve_steady, west_span
from nsb.assembly import BrinkmanDiscretization
from nsb.data import ConvectionSchemeType

pytestmark = pytest.mark.skipif(
    not __import__("nsb.fastres", fromlist=["HAVE_NUMBA"]).HAVE_NUMBA, reason="numba が無い"
)


def _cases():
    s = NSBSettings(velocity_floor=0.1, init_field="stokes", alpha_u=1.0)
    yield "flat", make_case("flat", 1, 1.0, settings=s)
    yield "uturn", make_case("uturn", 1, 1.0, settings=s)
    # 質量流入 + 領域内マニホールド（q_src / c_sink の分岐を通す）
    bc = BC(
        patches=(
            BC.mass_flow_inlet(north_span(0.3, 0.4, 0.4), 0.1),
            BC.pressure_outlet(west_span(0.05, 0.15)),
            BC.interior_source(disk_mask(0.15, 0.2, 0.05), 0.05),
            BC.interior_pressure_sink(disk_mask(0.55, 0.2, 0.05), 1e-4, p=0.0),
        )
    )
    yield "manifold", make_case("flat", 1, bc=bc, settings=s)


class TestFastResidualPhysics:
    @pytest.mark.parametrize(
        "name,inp", list(_cases()), ids=lambda c: c if isinstance(c, str) else ""
    )
    @pytest.mark.parametrize(
        "scheme",
        [ConvectionSchemeType.SECOND_ORDER_UPWIND, ConvectionSchemeType.FIRST_ORDER_UPWIND],
    )
    @pytest.mark.parametrize("pseudo", [False, True])
    @pytest.mark.parametrize("convection", [True, False])
    def test_matches_numpy_path(self, name, inp, scheme, pseudo, convection):
        disc = BrinkmanDiscretization(inp.to_flow_input())
        rng = np.random.default_rng(0)
        x = rng.standard_normal(3 * disc.n)
        pd = np.abs(rng.standard_normal((disc.nx, disc.ny))) if pseudo else None
        ref = disc.residual_from_state(
            x, disc.compute_state(x, scheme, 5.0, pd), convection=convection
        )
        fast = disc.residual_fast(x, scheme, 5.0, pd, convection=convection)
        assert fast.shape == ref.shape
        assert np.abs(fast - ref).max() <= 1e-13 * np.abs(ref).max()


class TestFastResidualSolverAPI:
    def test_solver_gives_same_solution_with_and_without_fast_residual(self):
        s = NSBSettings(
            velocity_floor=0.1,
            init_field="stokes",
            alpha_u=1.0,
            linear_solver="jfnk_simple",
            gmres_tol=1e-2,
        )
        inp = make_case("flat", 1, 1.0, settings=s)
        res_fast = solve_steady(inp, log=None)
        res_ref = solve_steady(replace(inp, settings=replace(s, fast_residual=False)), log=None)
        assert res_fast.converged and res_ref.converged
        assert res_fast.n_iter == res_ref.n_iter
        assert np.abs(res_fast.u - res_ref.u).max() <= 1e-8 * np.abs(res_ref.u).max()
