"""nsbm.project: 収束解の近傍からの Newton 1 歩が残差を大きく減らすこと."""

from __future__ import annotations

import numpy as np
import pytest

from nsbm.dataset import solve_sample
from nsbm.families import build_input
from nsbm.project import newton_project


@pytest.mark.slow
def test_newton_step_reduces_residual_near_solution():
    s = solve_sample(seed=1, families=["uniform"])
    u, v, p = s.fields()
    rng = np.random.default_rng(0)
    init = (u * (1 + 0.05 * rng.normal(size=u.shape)), v * (1 + 0.05 * rng.normal(size=v.shape)), p)
    inp = build_input(s.theta)
    (u1, v1, p1), info = newton_project(inp, init, s.x)
    assert info["steps_taken"] == 1
    assert (
        info["r_after"] < 0.5 * info["r_before"]
    )  # SOU リミターの折れ点があるので 1 歩では 2 次収束にならない
    assert (
        np.isfinite(u1).all() and u1.shape == u.shape
    )  # 残差は減るが場の最大誤差が減る保証はない（吸引域の縁）
