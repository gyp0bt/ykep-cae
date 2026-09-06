"""nsb.linalg（PARDISO ラッパ）と前処理 LU の遅延更新のテスト."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from nsb import NSBSettings, make_case, solve_steady
from nsb.linalg import PardisoLU, pardiso_solve
from nsb.solver import LaggedPreconditioner


def _spd_like(n: int, seed: int = 0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    A = sparse.random(n, n, density=0.02, random_state=rng, format="csr")
    return (A + sparse.eye(n, format="csr") * n).tocsr()


class TestPardisoLUAPI:
    def test_factorize_solve_matches_dense(self):
        A = _spd_like(300)
        b = np.arange(300, dtype=float)
        with PardisoLU() as lu:
            x = lu.factorize(A).solve(b)
            assert lu.is_factorized
            assert lu.shape == (300, 300)
            x2 = lu.solve(b[:, None])  # 複数右辺の形状も維持
        assert np.linalg.norm(A @ x - b) < 1e-9 * np.linalg.norm(b)
        assert x2.shape == (300, 1)
        assert np.allclose(x2[:, 0], x)
        assert not lu.is_factorized  # with を抜けると解放

    def test_solve_before_factorize_raises(self):
        lu = PardisoLU()
        with pytest.raises(RuntimeError):
            lu.solve(np.ones(3))

    def test_pardiso_solve_one_shot_and_transpose(self):
        A = _spd_like(200, seed=1)
        b = np.ones(200)
        x = pardiso_solve(A, b)
        xt = pardiso_solve(A.T.tocsr(), b)
        assert np.linalg.norm(A @ x - b) < 1e-9 * np.linalg.norm(b)
        assert np.linalg.norm(A.T @ xt - b) < 1e-9 * np.linalg.norm(b)

    def test_refactorize_replaces_previous(self):
        A1 = _spd_like(100, seed=2)
        A2 = (A1 + sparse.eye(100, format="csr") * 50.0).tocsr()
        b = np.ones(100)
        with PardisoLU() as lu:
            x1 = lu.factorize(A1).solve(b)
            x2 = lu.factorize(A2).solve(b)
        assert np.linalg.norm(A1 @ x1 - b) < 1e-9 * np.linalg.norm(b)
        assert np.linalg.norm(A2 @ x2 - b) < 1e-9 * np.linalg.norm(b)

    def test_thread_split_defaults(self):
        lu = PardisoLU()
        assert lu.solve_threads == 1
        assert 1 <= lu.factor_threads <= max(1, lu.max_threads)
        lu.free()


class TestLaggedPreconditionerAPI:
    def test_refresh_rules(self):
        s = NSBSettings(precond_lag=3, precond_refresh_gmres=10, precond_cfl_ratio=4.0)
        pc = LaggedPreconditioner(s)
        assert pc.needs_refresh()  # 未分解
        pc.cfl = 1.0
        pc.refresh(_spd_like(50))
        assert pc.n_factorizations == 1 and pc.age == 0
        assert not pc.needs_refresh()
        pc.age = 3
        assert pc.needs_refresh()  # age >= lag
        pc.age = 1
        pc.last_gmres = 11
        assert pc.needs_refresh()  # GMRES 反復が多すぎる
        pc.last_gmres = 0
        pc.cfl = 4.0
        assert pc.needs_refresh()  # CFL が 4 倍
        pc.cfl = 2.0
        assert not pc.needs_refresh()
        assert pc.needs_refresh(force=True)
        pc.free()

    def test_lu_mode_always_refreshes(self):
        pc = LaggedPreconditioner(NSBSettings(linear_solver="lu", precond_lag=10))
        pc.cfl = 1.0
        pc.refresh(_spd_like(50))
        assert pc.needs_refresh()
        pc.free()


@pytest.mark.slow
class TestLaggedPreconditionerConvergence:
    @pytest.mark.parametrize("lag", [1, 4])
    def test_lagged_pc_reaches_same_steady_state(self, lag: int):
        """前処理の遅延更新は解を変えない（同じ定常解に収束し、分解回数だけ減る）."""
        u_in = 1.0
        base = NSBSettings(
            velocity_floor=0.1 * u_in,
            init_field="stokes",
            alpha_u=1.0,
            newton_tol=1e-8,
            precond_lag=lag,
            precond_cfl_ratio=2.0,
        )
        inp = make_case("flat", 1, u_in, settings=base)
        res = solve_steady(inp, log=None)
        assert res.converged, res.failure_reason
        assert res.n_gmres_total > 0
        if lag == 1:
            assert res.n_factorizations == res.n_iter + 1  # Stokes 初期場 + 毎反復
        else:
            assert res.n_factorizations < res.n_iter + 1
        ref = solve_steady(replace(inp, settings=replace(base, precond_lag=1)), log=None)
        scale = np.abs(ref.u).max()
        assert np.abs(res.u - ref.u).max() < 1e-5 * scale
        assert np.abs(res.p - ref.p).max() < 1e-5 * np.abs(ref.p).max()
