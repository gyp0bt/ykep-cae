"""nsbm.dataset: 1 件生成とシャード I/O のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from nsbm.dataset import Sample, load_shards, save_shard, solve_sample
from nsbm.families import sample_theta


def _fake_sample(seed: int) -> Sample:
    th = sample_theta(seed)
    rng = np.random.default_rng(seed)
    return Sample(
        theta=th,
        x=rng.normal(size=(8, 72, 48)).astype(np.float32),
        y=rng.normal(size=(3, 72, 48)).astype(np.float32),
        n_iter=int(seed),
        converged=bool(seed % 2),
        n_gmres_total=7,
        residual_ref=1.5,
        elapsed=0.1,
    )


def test_shard_roundtrip(tmp_path):
    samples = [_fake_sample(s) for s in range(3)]
    path = save_shard(tmp_path / "shard-0000.npz", samples)
    back = load_shards(tmp_path)
    assert [b.theta for b in back] == [s.theta for s in samples]
    for a, b in zip(samples, back, strict=True):
        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
        assert (a.n_iter, a.converged, a.n_gmres_total) == (b.n_iter, b.converged, b.n_gmres_total)
        assert a.residual_ref == pytest.approx(b.residual_ref)
    assert path.exists()


@pytest.mark.slow
def test_solve_sample_uniform():
    s = solve_sample(seed=1, families=["uniform"])
    assert s.converged and s.n_iter > 0
    assert s.x.shape == (8, 72, 48) and s.y.shape == (3, 72, 48)
    u, v, p = s.fields()
    assert np.hypot(u, v).max() > 0.5 * s.theta.u_in
    assert s.residual_ref > 0
