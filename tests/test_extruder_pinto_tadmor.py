"""Pinto–Tadmor 型 RTD モデル（真値の供給源）のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.extruder.pinto_tadmor import PintoTadmorRTD, pinto_tadmor_rtd


class TestExactRelations:
    @pytest.mark.parametrize("r", [0.0, -0.3, -0.7])
    def test_mean_is_volume_over_flow(self, r):
        """t̄·V_z/L = 2/(1+r)（流管の体積÷流束の厳密値）に 1e-3 で一致."""
        res = pinto_tadmor_rtd(r)
        assert abs(res.tbar_over_L_Vz / (2.0 / (1.0 + r)) - 1.0) / 1.0e-3 < 1.0

    def test_minimum_is_three_quarters(self):
        """t_min/t̄ = 3/4。最速は再循環の停留高さ ξ=2/3 に居て折り返さない粒子."""
        res = pinto_tadmor_rtd(0.0)
        assert abs(res.t_min_ratio - 0.75) / 1.0e-4 < 1.0

    def test_reduced_curve_is_independent_of_back_pressure(self):
        """F(t/t̄) は r によらない（3ξ(1−ξ) = ξ − (3ξ²−2ξ) と 1 周の横断変位 0）."""
        a, b, c = (pinto_tadmor_rtd(r) for r in (0.0, -0.3, -0.7))
        for key in ("t_p10_ratio", "t_p50_ratio", "t_p90_ratio"):
            assert abs(getattr(a, key) - getattr(b, key)) / 1.0e-5 < 1.0
            assert abs(getattr(a, key) - getattr(c, key)) / 1.0e-5 < 1.0

    def test_reference_quantiles(self):
        """設計時に確認した分位点（p10 0.7524, p50 0.8225, p90 1.3247）."""
        res = pinto_tadmor_rtd(0.0)
        assert res.t_p10_ratio == pytest.approx(0.7524, abs=5e-4)
        assert res.t_p50_ratio == pytest.approx(0.8225, abs=5e-4)
        assert res.t_p90_ratio == pytest.approx(1.3247, abs=5e-4)


class TestCurveShape:
    def test_cumulative_is_monotone_from_zero_to_one(self):
        res = pinto_tadmor_rtd(0.0)
        assert isinstance(res, PintoTadmorRTD)
        assert np.all(np.diff(res.t_over_tbar) >= 0.0)
        assert np.all(np.diff(res.F) >= 0.0)
        assert res.F[0] < 1.0e-3 and res.F[-1] > 1.0 - 1.0e-3
        assert res.t_over_tbar[0] == pytest.approx(res.t_min_ratio)

    def test_converged_in_n_xi(self):
        """n_ξ を 4 倍にしても分位点が 1e-4 で動かない."""
        a, b = pinto_tadmor_rtd(0.0, n_xi=1000), pinto_tadmor_rtd(0.0, n_xi=4000)
        assert abs(a.t_p50_ratio - b.t_p50_ratio) / 1.0e-4 < 1.0


class TestArguments:
    @pytest.mark.parametrize("r", [-1.0, -1.5, 0.1])
    def test_rejects_back_pressure_ratio_out_of_range(self, r):
        with pytest.raises(ValueError, match="Q_p/Q_d"):
            pinto_tadmor_rtd(r)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="n_xi"):
            pinto_tadmor_rtd(0.0, n_xi=8)
