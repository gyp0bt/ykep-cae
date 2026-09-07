"""nsb.precond（SIMPLE 型ブロック前処理）と linear_solver="jfnk_simple" / "dc_simple" のテスト."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from nsb import NSBSettings, make_case, solve_steady
from nsb.assembly import BrinkmanDiscretization
from nsb.linalg import PardisoLU
from nsb.precond import SimpleBlockPreconditioner
from nsb.solver import LaggedPreconditioner


def _stokes_jacobian(
    refine: int = 1,
) -> tuple[BrinkmanDiscretization, sparse.csr_matrix, np.ndarray]:
    """Stokes–Brinkman の J0 と右辺（線形問題。前処理の単体検査用）."""
    s = NSBSettings(velocity_floor=0.1, init_field="stokes", alpha_u=1.0)
    inp = make_case("flat", refine, 1.0, settings=s)
    disc = BrinkmanDiscretization(inp.to_flow_input())
    x = np.zeros(3 * disc.n)
    st0 = disc.compute_state(x, s.scheme, s.venkat_k)
    J0 = disc.jacobian_first_order(st0, convection=False, x=x).tocsr()
    b = -disc.residual_from_state(x, st0, convection=False)
    return disc, J0, b


class TestSimpleBlockPreconditionerAPI:
    def test_solve_before_factorize_raises(self):
        pc = SimpleBlockPreconditioner(10)
        assert not pc.is_factorized
        with pytest.raises(RuntimeError):
            pc.solve(np.ones(30))

    def test_bad_momentum_raises(self):
        with pytest.raises(ValueError):
            SimpleBlockPreconditioner(10, momentum="gs")

    def test_shape_mismatch_raises(self):
        pc = SimpleBlockPreconditioner(10)
        with pytest.raises(ValueError):
            pc.factorize(sparse.eye(20, format="csr"))

    def test_factorize_solve_shapes_and_free(self):
        disc, J0, b = _stokes_jacobian()
        with SimpleBlockPreconditioner(disc.n) as pc:
            pc.factorize(J0)
            assert pc.is_factorized
            assert pc.shape == J0.shape
            assert pc.J.shape == J0.shape
            y = pc.solve(b)
            y2 = pc.solve(b[:, None])
            assert y.shape == b.shape and y2.shape == (b.size, 1)
            assert np.all(np.isfinite(y))
            assert np.allclose(y2[:, 0], y)
            # Schur 補元は −div(ρ d ∇p) 型: 対角は正
            assert pc.schur.shape == (disc.n, disc.n)
            assert float(np.mean(pc.schur.diagonal())) > 0.0
        assert not pc.is_factorized

    def test_solve_is_linear(self):
        """GMRES の前提: 前処理は線形写像（AMG は固定サイクル数、ILU は固定分解）."""
        disc, J0, b = _stokes_jacobian()
        pc = SimpleBlockPreconditioner(disc.n).factorize(J0)
        rng = np.random.default_rng(0)
        v, w = rng.standard_normal(b.size), rng.standard_normal(b.size)
        lhs = pc.solve(2.0 * v - 3.0 * w)
        rhs = 2.0 * pc.solve(v) - 3.0 * pc.solve(w)
        assert np.linalg.norm(lhs - rhs) < 1e-10 * np.linalg.norm(rhs)
        pc.free()

    @pytest.mark.parametrize("momentum", ["ilu", "jacobi"])
    def test_preconditioned_gmres_converges_far_faster_than_unpreconditioned(self, momentum: str):
        """Stokes–Brinkman（72×48）を前処理付き GMRES で解くと 1e-8 まで 100 反復以内に収束する."""
        disc, J0, b = _stokes_jacobian()
        pc = SimpleBlockPreconditioner(disc.n, momentum=momentum).factorize(J0)
        count = [0]
        M = spla.LinearOperator(J0.shape, matvec=pc.solve, dtype=float)
        x, info = spla.gmres(
            J0,
            b,
            M=M,
            rtol=1e-8,
            atol=0.0,
            restart=40,
            maxiter=5,
            callback=lambda _: count.__setitem__(0, count[0] + 1),
            callback_type="pr_norm",
        )
        pc.free()
        assert info == 0
        assert count[0] <= 100, count[0]
        assert np.linalg.norm(J0 @ x - b) < 1e-6 * np.linalg.norm(b)
        with PardisoLU() as lu:
            x_ref = lu.factorize(J0).solve(b)
        assert np.abs(x - x_ref).max() < 1e-5 * np.abs(x_ref).max()


class TestLaggedPreconditionerSimpleAPI:
    def test_simple_mode_requires_n(self):
        with pytest.raises(ValueError):
            LaggedPreconditioner(NSBSettings(linear_solver="jfnk_simple"))

    def test_unknown_linear_solver_raises(self):
        with pytest.raises(ValueError):
            LaggedPreconditioner(NSBSettings(linear_solver="amg"))

    def test_simple_mode_holds_block_preconditioner(self):
        pc = LaggedPreconditioner(
            NSBSettings(linear_solver="dc_simple", simple_momentum="jacobi"), n=12
        )
        assert isinstance(pc.fac, SimpleBlockPreconditioner)
        assert pc.fac.momentum == "jacobi"
        assert pc.needs_refresh()  # 未組立
        pc.free()


class TestSimpleModesConvergence:
    @pytest.mark.parametrize("mode", ["jfnk_simple", "dc_simple"])
    def test_reaches_same_steady_state_as_pardiso(self, mode: str):
        """SIMPLE 型前処理（jfnk_simple / dc_simple）は PARDISO 前処理の jfnk と同じ定常解に収束する."""
        u_in = 1.0
        base = NSBSettings(
            velocity_floor=0.1 * u_in,
            init_field="stokes",
            alpha_u=1.0,
            newton_tol=1e-8,
            precond_cfl_ratio=2.0,
        )
        inp = make_case("flat", 1, u_in, settings=base)
        ref = solve_steady(inp, log=None)
        assert ref.converged, ref.failure_reason
        res = solve_steady(replace(inp, settings=replace(base, linear_solver=mode)), log=None)
        assert res.converged, res.failure_reason
        assert res.n_gmres_total > 0
        assert res.n_factorizations >= 1
        scale = np.abs(ref.u).max()
        assert np.abs(res.u - ref.u).max() < 1e-5 * scale
        assert np.abs(res.p - ref.p).max() < 1e-5 * np.abs(ref.p).max()
        if mode == "jfnk_simple":
            assert res.n_iter == ref.n_iter  # 前処理は Newton の経路を変えない
        else:
            assert res.n_iter >= ref.n_iter  # defect correction は線形収束

    def test_stokes_init_uses_preconditioned_gmres(self):
        """Stokes 初期場は SIMPLE 前処理付き GMRES で解かれ、ログに反復数が出る."""
        lines: list[str] = []
        inp = make_case(
            "flat",
            1,
            1.0,
            settings=NSBSettings(
                velocity_floor=0.1,
                init_field="stokes",
                alpha_u=1.0,
                linear_solver="jfnk_simple",
                newton_max_iter=1,
            ),
        )
        solve_steady(inp, log=lines.append)
        init = [ln for ln in lines if "stokes init" in ln]
        assert len(init) == 1
        assert "gmres=" in init[0]
        assert "not converged" not in init[0]
