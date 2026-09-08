"""nsb.nested（入れ子反復: 粗→細の補間と段の駆動）の検査."""

from __future__ import annotations

import numpy as np
import pytest

from nsb import NSBSettings, make_case, prolong_bilinear, prolong_inject, solve_nested, solve_steady
from nsb.nested import prolong_fields


class TestProlongationAPI:
    def test_inject_shape_and_constant(self):
        a = np.full((3, 4), 2.5)
        out = prolong_inject(a)
        assert out.shape == (6, 8)
        assert np.all(out == 2.5)

    def test_bilinear_reproduces_linear_field_in_interior(self):
        nx, ny = 6, 5
        xc = (np.arange(nx) + 0.5)[:, None]
        yc = (np.arange(ny) + 0.5)[None, :]
        a = 2.0 * xc - 3.0 * yc + 1.0
        out = prolong_bilinear(a)
        xf = (np.arange(2 * nx) + 0.5)[:, None] / 2.0
        yf = (np.arange(2 * ny) + 0.5)[None, :] / 2.0
        exact = 2.0 * xf - 3.0 * yf + 1.0
        # 境界 1 セル（細格子で 2 セル）は最近傍外挿なので除く
        assert np.allclose(out[2:-2, 2:-2], exact[2:-2, 2:-2])

    def test_bilinear_preserves_constant_and_mean(self):
        a = np.full((4, 4), -1.25)
        assert np.allclose(prolong_bilinear(a), -1.25)
        rng = np.random.default_rng(1)
        b = rng.standard_normal((8, 6))
        # 内部では重みの和が 1 で対称なので、4 セル平均は元の値に近い（境界を除く）
        out = prolong_bilinear(b)
        avg = out.reshape(8, 2, 6, 2).mean(axis=(1, 3))
        assert np.abs(avg - b)[1:-1, 1:-1].max() < np.abs(b).max()

    def test_prolong_fields_rejects_non_power_of_two(self):
        s = NSBSettings(newton_tol=1e-2, newton_max_iter=5)
        res = solve_steady(make_case("flat", 1, 1.0, settings=s), log=None)
        with pytest.raises(ValueError):
            prolong_fields(res, 3 * 72, 3 * 48)


class TestSolveNestedConvergence:
    @pytest.mark.parametrize("prolongation", ["inject", "bilinear"])
    def test_two_levels_converge_and_match_direct(self, prolongation):
        s = NSBSettings(newton_max_iter=80)
        nested = solve_nested(
            lambda r: make_case("flat", r, 1.0),
            [1, 2],
            coarse_tol=1e-4,
            prolongation=prolongation,
            settings=s,
            log=None,
        )
        assert len(nested.levels) == 2
        assert nested.levels[0].newton_tol == 1e-4
        assert nested.levels[1].newton_tol == s.newton_tol
        assert nested.result.converged
        direct = solve_steady(make_case("flat", 2, 1.0, settings=s), log=None)
        assert direct.converged
        du = np.abs(nested.result.u - direct.u).max() / np.abs(direct.u).max()
        assert du < 1e-4
        assert nested.n_iter_total == sum(lv.result.n_iter for lv in nested.levels)
        assert nested.elapsed_total > 0.0
