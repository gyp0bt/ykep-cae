"""流束重み付きの分位点・経験分布（`xkep_cae_fluid.post.statistics`）のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.post.statistics import weighted_ecdf, weighted_quantile


class TestWeightedQuantile:
    def test_uniform_weights_match_numpy(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = weighted_quantile(v, np.ones_like(v), np.array([0.5]))
        assert q[0] == pytest.approx(3.0)

    def test_weight_shifts_the_median(self):
        v = np.array([1.0, 10.0])
        assert weighted_quantile(v, np.array([99.0, 1.0]), 0.5) < 2.0
        assert weighted_quantile(v, np.array([1.0, 99.0]), 0.5) > 9.0

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="重み"):
            weighted_quantile(np.array([1.0]), np.array([0.0]), 0.5)


class TestWeightedEcdf:
    """重み付き経験分布。`weighted_quantile` と同じ中点流儀なので分位点が逆算で一致する."""

    def test_matches_weighted_quantile(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(1.0, 3.0, 500)
        w = rng.uniform(0.1, 1.0, 500)
        t, f = weighted_ecdf(v, w)
        assert np.all(np.diff(t) >= 0.0)
        assert 0.0 < f[0] < f[-1] < 1.0
        for q in (0.1, 0.5, 0.9):
            assert np.interp(q, f, t) == pytest.approx(float(weighted_quantile(v, w, q)))

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="重み"):
            weighted_ecdf(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
