"""nsb.precond（SIMPLE 型ブロック前処理）と linear_solver="jfnk_simple" のテスト."""

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
    s = NSBSettings(velocity_floor_ratio=0.1, alpha_u=1.0)
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

    def test_preconditioned_gmres_converges_far_faster_than_unpreconditioned(self):
        """Stokes–Brinkman（72×48）を前処理付き GMRES で解くと 1e-8 まで 100 反復以内に収束する."""
        disc, J0, b = _stokes_jacobian()
        pc = SimpleBlockPreconditioner(disc.n).factorize(J0)
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


class TestSimpleBlockPreconditionerILURetry:
    def test_zero_pivot_retries_with_tighter_drop_tol(self, monkeypatch):
        """spilu が零ピボットを投げたら drop_tol 1/10・fill_factor 2 倍で組み直す（最大 3 回）."""
        disc, J0, b = _stokes_jacobian()
        real = spla.spilu
        calls: list[tuple[float, float]] = []

        def flaky(A, drop_tol, fill_factor, **kw):
            calls.append((drop_tol, fill_factor))
            if len(calls) == 1:
                raise RuntimeError("Factor is exactly singular")
            return real(A, drop_tol=drop_tol, fill_factor=fill_factor, **kw)

        monkeypatch.setattr("nsb.precond.spla.spilu", flaky)
        pc = SimpleBlockPreconditioner(disc.n, ilu_drop_tol=1e-2, ilu_fill_factor=1.5).factorize(J0)
        assert calls == [(1e-2, 1.5), (1e-3, 3.0)]
        assert pc.n_ilu_retries == 1
        assert np.all(np.isfinite(pc.solve(b)))
        pc.free()

    def test_persistent_zero_pivot_raises(self, monkeypatch):
        disc, J0, b = _stokes_jacobian()

        def always(*_a, **_k):
            raise RuntimeError("Factor is exactly singular")

        monkeypatch.setattr("nsb.precond.spla.spilu", always)
        pc = SimpleBlockPreconditioner(disc.n)
        with pytest.raises(RuntimeError, match="3 回"):
            pc.factorize(J0)
        assert pc.n_ilu_retries == 3


class TestLaggedPreconditionerSimpleAPI:
    def test_simple_mode_requires_n(self):
        with pytest.raises(ValueError):
            LaggedPreconditioner(NSBSettings(linear_solver="jfnk_simple"))

    def test_unknown_linear_solver_raises(self):
        with pytest.raises(ValueError):
            LaggedPreconditioner(NSBSettings(linear_solver="amg"))

    def test_simple_mode_holds_block_preconditioner(self):
        pc = LaggedPreconditioner(NSBSettings(linear_solver="jfnk_simple"), n=12)
        assert isinstance(pc.fac, SimpleBlockPreconditioner)
        assert pc.needs_refresh()  # 未組立
        pc.free()


class TestSimpleModesConvergence:
    def test_reaches_same_steady_state_as_pardiso(self):
        """SIMPLE 型前処理（jfnk_simple）は PARDISO 前処理の jfnk と同じ定常解に収束する."""
        u_in = 1.0
        mode = "jfnk_simple"
        base = NSBSettings(
            linear_solver="jfnk",
            velocity_floor_ratio=0.1,
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
        assert res.n_iter == ref.n_iter  # 前処理は Newton の経路を変えない

    def test_stokes_init_uses_preconditioned_gmres(self):
        """Stokes 初期場は SIMPLE 前処理付き GMRES で解かれ、ログに反復数が出る."""
        lines: list[str] = []
        inp = make_case(
            "flat",
            1,
            1.0,
            settings=NSBSettings(
                velocity_floor_ratio=0.1,
                alpha_u=1.0,
                linear_solver="jfnk_simple",
                newton_max_iter=1,
            ),
        )
        solve_steady(inp, log=lines.append)
        init = [ln for ln in lines if "stokes ref" in ln]
        assert len(init) == 1
        assert "gmres=" in init[0]
        assert "not converged" not in init[0]


class TestSimpleBlockPreconditionerHierarchyReuse:
    def test_reuse_keeps_aggregation_and_regalerkins(self):
        """2 回目の factorize は集約（P, R）を固定し、粗格子行列だけ Galerkin 積で組み直す."""
        disc, J0, b = _stokes_jacobian()
        pc = SimpleBlockPreconditioner(disc.n).factorize(J0)
        assert pc.n_hierarchy_builds == 1
        ml = pc._ml_schur
        P0 = [lv.P.copy() for lv in ml.levels[:-1]]
        # 擬似時間対角を足した別の行列で組み直す
        n = disc.n
        diag = np.concatenate([np.full(2 * n, 5.0), np.zeros(n)])
        J1 = (J0 + sparse.diags(diag)).tocsr()
        pc.factorize(J1)
        assert pc.n_hierarchy_builds == 1
        assert pc._ml_schur is ml
        for lv, P in zip(ml.levels[:-1], P0, strict=True):
            assert (lv.P != P).nnz == 0
        # 粗格子行列は新しい Ŝ の Galerkin 積
        S1 = pc.schur * pc._schur_sign
        assert np.abs((ml.levels[0].A - S1).data).max(initial=0.0) < 1e-12
        A1 = ml.levels[0].R @ S1 @ ml.levels[0].P
        assert np.abs((ml.levels[1].A - A1).data).max(initial=0.0) < 1e-10
        # 組み直した前処理で J1 の GMRES が収束する
        count = [0]
        x, info = spla.gmres(
            J1,
            b,
            M=spla.LinearOperator(J1.shape, matvec=pc.solve, dtype=float),
            rtol=1e-8,
            atol=0.0,
            restart=40,
            maxiter=5,
            callback=lambda _: count.__setitem__(0, count[0] + 1),
            callback_type="pr_norm",
        )
        assert info == 0 and count[0] <= 100
        assert np.linalg.norm(J1 @ x - b) < 1e-6 * np.linalg.norm(b)
        pc.free()
        assert pc._ml_schur is None

    def test_vcycle_matches_pyamg_solve(self):
        """直接辿る V サイクルは pyamg `MultilevelSolver.solve(maxiter=1)` と同じ写像."""
        disc, J0, _ = _stokes_jacobian()
        pc = SimpleBlockPreconditioner(disc.n).factorize(J0)
        r = np.random.default_rng(0).standard_normal(disc.n)
        mine = pc._solve_schur(r)
        ref = pc._ml_schur.solve(pc._schur_sign * r, tol=1e-12, maxiter=1, cycle="V", accel=None)
        assert np.allclose(mine, ref, rtol=1e-10, atol=1e-12)
        pc.free()


class TestSimpleBlockPreconditionerDeterminism:
    def test_same_matrix_gives_identical_hierarchy(self):
        disc, J0, b = _stokes_jacobian()
        pc1 = SimpleBlockPreconditioner(disc.n).factorize(J0)
        y1 = pc1.solve(b)
        np.random.seed(12345)  # グローバル乱数の状態に依らないこと
        pc2 = SimpleBlockPreconditioner(disc.n).factorize(J0)
        y2 = pc2.solve(b)
        for l1, l2 in zip(pc1._ml_schur.levels[:-1], pc2._ml_schur.levels[:-1], strict=True):
            assert (l1.P != l2.P).nnz == 0
        assert np.array_equal(y1, y2)
        pc1.free()
        pc2.free()
