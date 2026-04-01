"""非直交補正付き拡散スキームのテスト.

CorrectedDiffusionScheme が DiffusionSchemeStrategy Protocol を満たし、
直交格子上では CentralDiffusionScheme と同等の結果を返すことを検証する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.strategies.corrected_diffusion import (
    CorrectedDiffusionScheme,
)
from xkep_cae_fluid.core.strategies.diffusion import CentralDiffusionScheme
from xkep_cae_fluid.core.strategies.protocols import DiffusionSchemeStrategy


def _make_mesh(nx=4, ny=4, nz=4):
    return (
        StructuredMeshProcess()
        .process(StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz))
        .mesh
    )


class TestCorrectedDiffusionAPI:
    """API テスト."""

    def test_satisfies_protocol(self):
        """DiffusionSchemeStrategy Protocol を満たしているか."""
        scheme = CorrectedDiffusionScheme()
        assert isinstance(scheme, DiffusionSchemeStrategy)

    def test_matrix_shape(self):
        """行列のサイズが n_cells x n_cells."""
        mesh = _make_mesh(3, 3, 3)
        scheme = CorrectedDiffusionScheme()
        A = scheme.matrix_coefficients(1.0, mesh)
        assert A.shape == (mesh.n_cells, mesh.n_cells)

    def test_flux_shape(self):
        """フラックスのサイズが n_cells."""
        mesh = _make_mesh(3, 3, 3)
        phi = np.ones(mesh.n_cells)
        scheme = CorrectedDiffusionScheme()
        f = scheme.flux(phi, 1.0, mesh)
        assert f.shape == (mesh.n_cells,)

    def test_deferred_correction_shape(self):
        """遅延修正ベクトルのサイズが n_cells."""
        mesh = _make_mesh(3, 3, 3)
        phi = np.linspace(0, 1, mesh.n_cells)
        scheme = CorrectedDiffusionScheme()
        corr = scheme.deferred_correction(phi, 1.0, mesh)
        assert corr.shape == (mesh.n_cells,)

    def test_max_corrections_property(self):
        """max_non_ortho_corrections プロパティ."""
        assert CorrectedDiffusionScheme(3).max_non_ortho_corrections == 3
        assert CorrectedDiffusionScheme(0).max_non_ortho_corrections == 0


class TestCorrectedDiffusionPhysics:
    """物理テスト."""

    def test_orthogonal_matches_central(self):
        """直交格子上では CentralDiffusionScheme と一致."""
        mesh = _make_mesh(5, 5, 5)
        diff = 2.5

        central = CentralDiffusionScheme()
        corrected = CorrectedDiffusionScheme(max_non_ortho_corrections=0)

        A_central = central.matrix_coefficients(diff, mesh)
        A_corrected = corrected.matrix_coefficients(diff, mesh)
        np.testing.assert_allclose(A_corrected.toarray(), A_central.toarray(), rtol=1e-10)

    def test_uniform_field_zero_flux(self):
        """一様場では拡散フラックスがゼロ."""
        mesh = _make_mesh(5, 5, 5)
        phi = np.ones(mesh.n_cells) * 42.0
        scheme = CorrectedDiffusionScheme()
        f = scheme.flux(phi, 1.0, mesh)
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_uniform_field_zero_correction(self):
        """一様場では遅延修正がゼロ."""
        mesh = _make_mesh(4, 4, 4)
        phi = np.ones(mesh.n_cells) * 10.0
        scheme = CorrectedDiffusionScheme()
        corr = scheme.deferred_correction(phi, 1.0, mesh)
        np.testing.assert_allclose(corr, 0.0, atol=1e-10)

    def test_no_correction_mode(self):
        """補正なしモードでは遅延修正が常にゼロ."""
        mesh = _make_mesh(4, 4, 4)
        phi = np.random.default_rng(42).uniform(0, 100, mesh.n_cells)
        scheme = CorrectedDiffusionScheme(max_non_ortho_corrections=0)
        corr = scheme.deferred_correction(phi, 1.0, mesh)
        np.testing.assert_allclose(corr, 0.0, atol=1e-14)

    def test_orthogonal_correction_small(self):
        """直交格子では非直交補正がほぼゼロ."""
        mesh = _make_mesh(5, 5, 5)
        phi = mesh.cell_centers[:, 0]  # x方向線形場
        scheme = CorrectedDiffusionScheme()
        corr = scheme.deferred_correction(phi, 1.0, mesh)
        # 直交格子では非直交成分がゼロなので補正もゼロ
        np.testing.assert_allclose(corr, 0.0, atol=1e-10)

    def test_diffusivity_scaling(self):
        """拡散係数2倍で行列要素も2倍."""
        mesh = _make_mesh(3, 3, 3)
        scheme = CorrectedDiffusionScheme()
        A1 = scheme.matrix_coefficients(1.0, mesh)
        A2 = scheme.matrix_coefficients(2.0, mesh)
        np.testing.assert_allclose(A2.toarray(), 2.0 * A1.toarray(), rtol=1e-14)

    def test_heterogeneous_diffusivity(self):
        """不均一拡散係数での行列の対角優位性."""
        mesh = _make_mesh(4, 4, 4)
        rng = np.random.default_rng(42)
        diff = rng.uniform(0.1, 10.0, mesh.n_cells)
        scheme = CorrectedDiffusionScheme()
        A = scheme.matrix_coefficients(diff, mesh)
        A_dense = A.toarray()
        # 対角成分は非負
        assert np.all(A_dense.diagonal() >= -1e-14)
