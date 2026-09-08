"""nsbm.features: 入力画像と正規化のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from nsbm.families import build_h, port_cells, sample_theta
from nsbm.features import IN_CH, denormalize_y, make_x, normalize_y, p_ref


def test_roundtrip():
    th = sample_theta(5)
    rng = np.random.default_rng(0)
    u, v, p = rng.normal(size=(3, 72, 48))
    p *= p_ref(th)
    y = normalize_y(th, u, v, p)
    assert y.shape == (3, 72, 48) and y.dtype == np.float32
    u2, v2, p2 = denormalize_y(th, y)
    assert np.allclose(u, u2, atol=1e-5 * th.u_in)
    assert np.allclose(v, v2, atol=1e-5 * th.u_in)
    assert np.allclose(p, p2, atol=1e-5 * p_ref(th))


def test_p_ref_positive_and_scales():
    th = sample_theta(5)
    assert p_ref(th) > 0
    assert p_ref(th) >= 1000.0 * th.u_in**2


def test_make_x_channels():
    th = sample_theta(5)
    h = build_h(th)
    x = make_x(th, h)
    assert x.shape == (IN_CH, 72, 48) and x.dtype == np.float32
    assert np.allclose(x[0], np.log(h / th.h0), atol=1e-6)
    assert np.allclose(x[1], np.log(th.h0 / 1e-3)) and np.allclose(x[2], np.log(th.u_in))
    i, j = port_cells(th.inlet)
    speed = np.hypot(x[3], x[4])
    assert speed[i, j].max() == pytest.approx(th.u_in, rel=1e-6)
    assert np.count_nonzero(speed) == len(i)
    io, jo = port_cells(th.outlet)
    assert x[5].sum() == len(io) and x[5][io, jo].all()
    assert x[6].min() > 0 and x[6].max() < 1 and x[7].min() > 0 and x[7].max() < 1


def test_mask_blocked_zeroes_velocity_in_blocked_cells():
    from nsbm.features import blocked_mask_from_x, mask_blocked

    th = sample_theta(seed=0, families=["serpentine"])
    h = build_h(th)
    x = make_x(th, h)
    b = blocked_mask_from_x(x)
    assert np.array_equal(b, h <= th.h_blocked * 1.5) and b.any() and not b.all()
    u = np.ones((72, 48))
    v = np.ones((72, 48))
    p = np.full((72, 48), 3.0)
    u2, v2, p2 = mask_blocked(x, (u, v, p))
    assert (u2[b] == 0).all() and (v2[b] == 0).all() and (u2[~b] == 1).all()
    assert np.array_equal(p2, p)
