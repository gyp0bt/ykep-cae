"""nsbm.families: θ のサンプルと h 場 / BC 構築のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from nsbm.families import (
    FAMILIES,
    Port,
    Theta,
    build_h,
    build_input,
    connected,
    port_cells,
    sample_theta,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_build_h_shape_positive(family):
    th = sample_theta(seed=11, families=[family])
    h = build_h(th)
    assert h.shape == (72, 48)
    assert np.all(h > 0) and np.isfinite(h).all()
    assert th.family == family


@pytest.mark.parametrize("family", ["uturn", "serpentine", "pins", "blobs"])
def test_blocked_families_connected(family):
    for seed in range(5):
        th = sample_theta(seed=seed, families=[family])
        h = build_h(th)
        assert connected(h, th.inlet, th.outlet, th.h0 / 100)
        assert (h <= th.h0 / 100 * 1.001).any()


@pytest.mark.parametrize("family", ["uniform", "sin2d", "quad", "grf"])
def test_smooth_families_span(family):
    """滑らかなファミリは全域開いていて、h0 の周りで振れる."""
    th = sample_theta(seed=2, families=[family])
    h = build_h(th)
    assert h.min() > th.h0 / 100 * 1.5
    assert np.isclose(np.exp(np.log(h / th.h0).mean()), 1.0, atol=1.5)


def test_ports_do_not_overlap_same_wall():
    for seed in range(60):
        th = sample_theta(seed)
        if th.inlet.wall == th.outlet.wall:
            a, b = th.inlet, th.outlet
            assert a.s1 <= b.s0 or b.s1 <= a.s0
        for p in (th.inlet, th.outlet):
            assert 0.05 - 1e-9 <= p.s1 - p.s0 <= 0.15 + 1e-9


def test_seed_reproducible_and_roundtrip():
    a, b = sample_theta(3), sample_theta(3)
    assert a == b
    assert Theta.from_dict(a.to_dict()) == a


def test_port_cells_on_boundary():
    i, j = port_cells(Port("east", 0.1, 0.2))
    assert np.all(i == 71) and len(j) > 0
    i, j = port_cells(Port("north", 0.3, 0.4))
    assert np.all(j == 47) and len(i) > 0


@pytest.mark.slow
def test_build_input_solves():
    from nsb import solve_steady

    th = sample_theta(seed=1, families=["uniform"])
    inp = build_input(th)
    assert inp.nx == 72 and inp.ny == 48
    res = solve_steady(inp, log=None)
    assert res.converged


def test_sample_ports_fallback_never_raises_on_one_wall():
    """west 壁だけ（uturn）で 500 シード引いても例外にならず、重ならない."""
    from nsbm.families import _sample_ports

    for seed in range(500):
        rng = np.random.default_rng([seed, 0])
        a, b = _sample_ports(rng, ("west",))
        assert a.s1 <= b.s0 or b.s1 <= a.s0
        assert 0 <= a.s0 and a.s1 <= 0.4 and 0 <= b.s0 and b.s1 <= 0.4
