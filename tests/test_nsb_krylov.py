"""nsb.krylov.fgmres（右前処理 FGMRES）のテスト."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from nsb.krylov import fgmres


def _convection_diffusion(n: int = 40, peclet: float = 20.0) -> sparse.csr_matrix:
    """2D 1 次風上の移流拡散（非対称・非特異）."""
    e = np.ones(n)
    L1 = sparse.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n, n))
    U1 = sparse.diags([-e, e], [-1, 0], shape=(n, n))
    idn = sparse.eye(n)
    A = (
        sparse.kron(idn, L1)
        + sparse.kron(L1, idn)
        + peclet / n * (sparse.kron(idn, U1) + sparse.kron(U1, idn))
    )
    return A.tocsr()


class TestFGMRESAPI:
    def test_solves_to_requested_tolerance_and_reports_iterations(self):
        A = _convection_diffusion()
        b = np.arange(A.shape[0], dtype=float)
        hist: list[float] = []
        x, n_iter, ok = fgmres(
            lambda v: A @ v, b, rtol=1e-8, restart=60, maxiter=20, callback=hist.append
        )
        assert ok
        assert n_iter == len(hist) > 0
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)
        assert all(
            h2 <= h1 * (1 + 1e-12) for h1, h2 in zip(hist, hist[1:], strict=False)
        )  # 残差は単調非増加

    def test_stops_at_tolerance_not_far_below(self):
        """指定した rtol に届いたら止まる（scipy のように勝手に締めない）."""
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        x, _, ok = fgmres(lambda v: A @ v, b, rtol=1e-2, restart=60, maxiter=20)
        assert ok
        rel = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert 1e-3 < rel <= 1e-2

    def test_right_preconditioning_reduces_iterations(self):
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
        _, n_plain, ok_plain = fgmres(lambda v: A @ v, b, rtol=1e-8, restart=60, maxiter=20)
        x, n_pc, ok_pc = fgmres(
            lambda v: A @ v, b, precond=ilu.solve, rtol=1e-8, restart=60, maxiter=20
        )
        assert ok_plain and ok_pc
        assert n_pc < n_plain / 3
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)

    def test_restart_and_x0(self):
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        x1, n1, ok1 = fgmres(lambda v: A @ v, b, rtol=1e-6, restart=10, maxiter=200)
        assert ok1 and n1 > 10  # 再出発を跨いで収束
        x2, n2, ok2 = fgmres(lambda v: A @ v, b, rtol=1e-6, restart=10, maxiter=200, x0=x1)
        assert ok2 and n2 == 0  # 既に収束している初期値
        assert np.allclose(x1, x2)

    def test_not_converged_flag(self):
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        x, n_iter, ok = fgmres(lambda v: A @ v, b, rtol=1e-12, restart=5, maxiter=1)
        assert not ok and n_iter == 5
        assert np.all(np.isfinite(x))

    def test_zero_rhs(self):
        A = _convection_diffusion()
        x, n_iter, ok = fgmres(lambda v: A @ v, np.zeros(A.shape[0]))
        assert ok and n_iter == 0 and not x.any()

    def test_bad_args_raise(self):
        with pytest.raises(ValueError):
            fgmres(lambda v: v, np.ones(3), restart=0)

    def test_flexible_variable_preconditioner(self):
        """前処理が反復ごとに変わっても（内側に反復解法を置く等）正しく解ける."""
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        calls = [0]

        def var_precond(v: np.ndarray) -> np.ndarray:
            calls[0] += 1
            y, _ = spla.bicgstab(A, v, rtol=1e-1 if calls[0] % 2 else 1e-3, maxiter=50)
            return y

        x, n_iter, ok = fgmres(
            lambda v: A @ v, b, precond=var_precond, rtol=1e-8, restart=30, maxiter=5
        )
        assert ok
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)


class TestFGMRESNonlinearMatvec:
    def test_givens_only_stops_without_true_residual_check(self):
        """matvec に微小な非線形雑音があると真の残差は許容を割れないが、Givens 推定で止められる."""
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        rng = np.random.default_rng(1)
        noise = 3e-3

        def noisy_matvec(v: np.ndarray) -> np.ndarray:
            y = A @ v
            return y + noise * np.linalg.norm(y) * rng.standard_normal(y.size) / np.sqrt(y.size)

        ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
        x1, n1, ok1 = fgmres(noisy_matvec, b, precond=ilu.solve, rtol=1e-3, restart=30, maxiter=5)
        x2, n2, ok2 = fgmres(
            noisy_matvec,
            b,
            precond=ilu.solve,
            rtol=1e-3,
            restart=30,
            maxiter=5,
            check_true_residual=False,
        )
        assert not ok1 and n1 > n2  # 真の残差確認は雑音の床で空回り
        assert ok2
        assert np.linalg.norm(A @ x2 - b) <= 10 * noise * np.linalg.norm(
            b
        )  # 解の質は雑音の数倍以内


class TestFGMRESInfoAPI:
    def test_info_reports_final_true_residual_ratio(self):
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        info: dict[str, float] = {}
        x, _, ok = fgmres(lambda v: A @ v, b, rtol=1e-8, restart=60, maxiter=20, info=info)
        assert ok
        ratio = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert info["resid_ratio"] == pytest.approx(ratio, rel=1e-6)
        assert info["resid_ratio"] <= 1e-8

    def test_info_when_not_converged(self):
        A = _convection_diffusion()
        b = np.ones(A.shape[0])
        info: dict[str, float] = {}
        _, _, ok = fgmres(lambda v: A @ v, b, rtol=1e-12, restart=2, maxiter=1, info=info)
        assert not ok
        assert info["resid_ratio"] > 1e-12
