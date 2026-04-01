"""TVD 対流スキームのテスト.

TVDConvectionScheme (van Leer / Superbee) が ConvectionSchemeStrategy Protocol を満たし、
MeshData ベースの FVM 離散化が正しく動作することを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.strategies.protocols import ConvectionSchemeStrategy
from xkep_cae_fluid.core.strategies.tvd_convection import (
    TVDConvectionScheme,
    TVDLimiter,
    _limiter_superbee,
    _limiter_van_leer,
)


def _make_mesh(nx=4, ny=4, nz=4):
    return StructuredMeshProcess().process(
        StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz)
    ).mesh


class TestTVDConvectionAPI:
    """API テスト."""

    @pytest.mark.parametrize("limiter", [TVDLimiter.VAN_LEER, TVDLimiter.SUPERBEE])
    def test_satisfies_protocol(self, limiter):
        """Protocol を満たしているか."""
        scheme = TVDConvectionScheme(limiter=limiter)
        assert isinstance(scheme, ConvectionSchemeStrategy)

    def test_matrix_shape(self):
        """行列のサイズが n_cells x n_cells."""
        mesh = _make_mesh(3, 3, 3)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0

        scheme = TVDConvectionScheme()
        A = scheme.matrix_coefficients(velocity, mesh)
        assert A.shape == (mesh.n_cells, mesh.n_cells)

    def test_flux_shape(self):
        """フラックスのサイズが n_cells."""
        mesh = _make_mesh(3, 3, 3)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.ones(mesh.n_cells)

        scheme = TVDConvectionScheme()
        f = scheme.flux(phi, velocity, mesh)
        assert f.shape == (mesh.n_cells,)

    def test_deferred_correction_shape(self):
        """遅延修正ベクトルのサイズが n_cells."""
        mesh = _make_mesh(3, 3, 3)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.linspace(0, 1, mesh.n_cells)

        scheme = TVDConvectionScheme()
        corr = scheme.deferred_correction(phi, velocity, mesh)
        assert corr.shape == (mesh.n_cells,)

    def test_limiter_property(self):
        """リミッタ種別プロパティ."""
        assert TVDConvectionScheme(TVDLimiter.VAN_LEER).limiter == TVDLimiter.VAN_LEER
        assert (
            TVDConvectionScheme(TVDLimiter.SUPERBEE).limiter == TVDLimiter.SUPERBEE
        )


class TestTVDLimiterFunctions:
    """リミッタ関数の数学的性質テスト."""

    def test_van_leer_symmetry(self):
        """van Leer: ψ(r)/r = ψ(1/r) (対称性)."""
        r = np.array([0.5, 1.0, 2.0, 5.0])
        psi = _limiter_van_leer(r)
        psi_inv = _limiter_van_leer(1.0 / r)
        np.testing.assert_allclose(psi / r, psi_inv, rtol=1e-12)

    def test_van_leer_values(self):
        """van Leer の既知の値."""
        # r=0 → ψ=0 (風上)
        assert _limiter_van_leer(np.array([0.0]))[0] == pytest.approx(0.0)
        # r=1 → ψ=1 (中心差分)
        assert _limiter_van_leer(np.array([1.0]))[0] == pytest.approx(1.0)
        # r<0 → ψ=0
        assert _limiter_van_leer(np.array([-1.0]))[0] == pytest.approx(0.0)

    def test_superbee_values(self):
        """Superbee の既知の値."""
        # r=0 → ψ=0
        assert _limiter_superbee(np.array([0.0]))[0] == pytest.approx(0.0)
        # r=1 → ψ=1
        assert _limiter_superbee(np.array([1.0]))[0] == pytest.approx(1.0)
        # r=0.5 → max(0, min(1.0, 1), min(0.5, 2)) = max(0, 1, 0.5) = 1
        assert _limiter_superbee(np.array([0.5]))[0] == pytest.approx(1.0)
        # r=2 → max(0, min(4, 1), min(2, 2)) = max(0, 1, 2) = 2
        assert _limiter_superbee(np.array([2.0]))[0] == pytest.approx(2.0)
        # r<0 → ψ=0
        assert _limiter_superbee(np.array([-1.0]))[0] == pytest.approx(0.0)

    def test_tvd_region(self):
        """リミッタが Sweby の TVD 領域内にあるか."""
        r = np.linspace(0.01, 10, 100)
        for func in [_limiter_van_leer, _limiter_superbee]:
            psi = func(r)
            # TVD 条件: 0 ≤ ψ(r) ≤ min(2r, 2)
            assert np.all(psi >= -1e-14)
            upper = np.minimum(2.0 * r, 2.0)
            assert np.all(psi <= upper + 1e-12)


class TestTVDConvectionPhysics:
    """物理テスト."""

    @pytest.mark.parametrize("limiter", [TVDLimiter.VAN_LEER, TVDLimiter.SUPERBEE])
    def test_uniform_field_zero_flux(self, limiter):
        """一様場では対流フラックスの内部セル寄与がゼロ."""
        mesh = _make_mesh(5, 5, 5)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.ones(mesh.n_cells) * 42.0

        scheme = TVDConvectionScheme(limiter=limiter)
        f = scheme.flux(phi, velocity, mesh)

        nx, ny, nz = mesh.dimensions
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    idx = i * ny * nz + j * nz + k
                    assert abs(f[idx]) < 1e-10

    @pytest.mark.parametrize("limiter", [TVDLimiter.VAN_LEER, TVDLimiter.SUPERBEE])
    def test_zero_velocity_zero_flux(self, limiter):
        """速度ゼロではフラックスもゼロ."""
        mesh = _make_mesh(4, 4, 4)
        velocity = np.zeros((mesh.n_cells, 3))
        phi = np.random.default_rng(42).uniform(0, 100, mesh.n_cells)

        scheme = TVDConvectionScheme(limiter=limiter)
        f = scheme.flux(phi, velocity, mesh)
        np.testing.assert_allclose(f, 0.0, atol=1e-14)

    @pytest.mark.parametrize("limiter", [TVDLimiter.VAN_LEER, TVDLimiter.SUPERBEE])
    def test_deferred_correction_zero_for_uniform(self, limiter):
        """一様場では遅延修正がゼロ."""
        mesh = _make_mesh(4, 4, 4)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.ones(mesh.n_cells) * 5.0

        scheme = TVDConvectionScheme(limiter=limiter)
        corr = scheme.deferred_correction(phi, velocity, mesh)
        np.testing.assert_allclose(corr, 0.0, atol=1e-12)

    def test_linear_field_correction_small(self):
        """線形場では遅延修正が小さい（2次精度の効果）."""
        mesh = _make_mesh(8, 4, 4)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        # x 方向に線形場
        phi = mesh.cell_centers[:, 0]

        scheme = TVDConvectionScheme(TVDLimiter.VAN_LEER)
        corr = scheme.deferred_correction(phi, velocity, mesh)
        # 線形場では TVD と風上の差は限定的（リミッタが r≈1 になる）
        assert np.max(np.abs(corr)) < 1.0

    def test_velocity_scaling(self):
        """速度2倍で行列要素も2倍."""
        mesh = _make_mesh(3, 3, 3)
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0

        scheme = TVDConvectionScheme()
        A1 = scheme.matrix_coefficients(velocity, mesh)
        A2 = scheme.matrix_coefficients(2.0 * velocity, mesh)
        np.testing.assert_allclose(A2.toarray(), 2.0 * A1.toarray(), rtol=1e-14)
