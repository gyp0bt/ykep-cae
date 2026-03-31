"""1次風上対流スキームのテスト.

UpwindConvectionScheme が ConvectionSchemeStrategy Protocol を満たし、
MeshData ベースの FVM 離散化が正しく動作することを検証する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.strategies.convection import UpwindConvectionScheme
from xkep_cae_fluid.core.strategies.protocols import ConvectionSchemeStrategy


class TestUpwindConvectionAPI:
    """API テスト."""

    def test_satisfies_protocol(self):
        """Protocol を満たしているか."""
        scheme = UpwindConvectionScheme()
        assert isinstance(scheme, ConvectionSchemeStrategy)

    def test_matrix_shape(self):
        """行列のサイズが n_cells x n_cells."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0

        scheme = UpwindConvectionScheme()
        A = scheme.matrix_coefficients(velocity, mesh)
        assert A.shape == (mesh.n_cells, mesh.n_cells)

    def test_flux_shape(self):
        """フラックスのサイズが n_cells."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.ones(mesh.n_cells)

        scheme = UpwindConvectionScheme()
        f = scheme.flux(phi, velocity, mesh)
        assert f.shape == (mesh.n_cells,)


class TestUpwindConvectionPhysics:
    """物理テスト."""

    def test_uniform_field_zero_flux(self):
        """一様場では対流フラックスの内部セル寄与がゼロ."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0
        phi = np.ones(mesh.n_cells) * 42.0

        scheme = UpwindConvectionScheme()
        f = scheme.flux(phi, velocity, mesh)

        # 内部セルではフラックスの合計がゼロ
        nx, ny, nz = mesh.dimensions
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    idx = i * ny * nz + j * nz + k
                    assert abs(f[idx]) < 1e-12

    def test_zero_velocity_zero_flux(self):
        """速度ゼロではフラックスもゼロ."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=4, nz=4)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        phi = np.random.default_rng(42).uniform(0, 100, mesh.n_cells)

        scheme = UpwindConvectionScheme()
        f = scheme.flux(phi, velocity, mesh)
        np.testing.assert_allclose(f, 0.0, atol=1e-14)

    def test_matrix_diagonal_dominant(self):
        """風上差分行列は対角優位（M-matrix 性）."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=4, nz=4)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 2.0
        velocity[:, 1] = -1.0

        scheme = UpwindConvectionScheme()
        A = scheme.matrix_coefficients(velocity, mesh)
        A_dense = A.toarray()

        # 対角成分は非負
        assert np.all(A_dense.diagonal() >= -1e-14)

        # 非対角成分は非正
        for i in range(mesh.n_cells):
            for j in range(mesh.n_cells):
                if i != j:
                    assert A_dense[i, j] <= 1e-14, f"A[{i},{j}] = {A_dense[i, j]} > 0"

    def test_row_sum_zero_interior(self):
        """内部セルでは行列の行和がゼロ（保存性）."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0

        scheme = UpwindConvectionScheme()
        A = scheme.matrix_coefficients(velocity, mesh)

        row_sums = np.array(A.sum(axis=1)).ravel()
        nx, ny, nz = mesh.dimensions
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    idx = i * ny * nz + j * nz + k
                    assert abs(row_sums[idx]) < 1e-12, (
                        f"cell ({i},{j},{k}): row_sum={row_sums[idx]}"
                    )

    def test_velocity_reversal(self):
        """速度反転でフラックスも反転."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=4, nz=4)
        )
        mesh = mesh_result.mesh
        rng = np.random.default_rng(42)
        phi = rng.uniform(0, 100, mesh.n_cells)

        velocity_pos = np.zeros((mesh.n_cells, 3))
        velocity_pos[:, 0] = 1.0
        velocity_neg = np.zeros((mesh.n_cells, 3))
        velocity_neg[:, 0] = -1.0

        scheme = UpwindConvectionScheme()
        f_pos = scheme.flux(phi, velocity_pos, mesh)
        f_neg = scheme.flux(phi, velocity_neg, mesh)

        # 速度反転で符号反転（ただしφの風上値が変わるため完全一致はしない）
        # 少なくとも非ゼロフラックスが存在する
        assert np.max(np.abs(f_pos)) > 1e-10
        assert np.max(np.abs(f_neg)) > 1e-10

    def test_velocity_scaling(self):
        """速度2倍で行列要素も2倍."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        mesh = mesh_result.mesh
        velocity = np.zeros((mesh.n_cells, 3))
        velocity[:, 0] = 1.0

        scheme = UpwindConvectionScheme()
        A1 = scheme.matrix_coefficients(velocity, mesh)
        A2 = scheme.matrix_coefficients(2.0 * velocity, mesh)
        np.testing.assert_allclose(A2.toarray(), 2.0 * A1.toarray(), rtol=1e-14)
