"""nsbm.galerkin: Stokes + 近傍解の局所基底で残差が Stokes 以下になること."""

from __future__ import annotations

import numpy as np
import pytest

from nsb.core import NSBSettings
from nsb.solver import solve_steady
from nsbm.families import build_input, sample_theta
from nsbm.galerkin import galerkin_init, stokes_field


@pytest.mark.slow
def test_galerkin_residual_never_worse_than_stokes():
    theta = sample_theta(3, ("uniform",))
    st = NSBSettings(newton_max_iter=40)
    inp = build_input(theta, st)
    x_s, disc = stokes_field(inp)
    res = solve_steady(
        inp, log=None
    )  # 収束解を「近傍」として 1 本入れる（部分空間に解が入る極端な例）
    rng = np.random.default_rng(0)
    noisy = (res.u * (1 + 0.2 * rng.standard_normal(res.u.shape)), res.v, res.p)
    init, info = galerkin_init(inp, [noisy, (res.u, res.v, res.p)], blocked=None, steps=6)
    assert info["n_basis"] == 3
    assert info["r_final"] <= info["r_stokes"]
    assert info["r_final"] < 0.1 * info["r_stokes"]  # 解が張られているので大きく落ちる
    assert init[0].shape == (inp.nx, inp.ny)
    # Stokes 解だけを基底にすると何もできない（残差は Stokes のまま、係数は 1）
    _, info0 = galerkin_init(inp, [], blocked=None, steps=2)
    assert info0["r_final"] <= info0["r_stokes"] and info0["n_basis"] == 1
