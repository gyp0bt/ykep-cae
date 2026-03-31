"""StructuredMeshProcess のテスト.

API テスト + 物理テスト（メッシュ幾何学的整合性）。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import (
    StructuredMeshInput,
    StructuredMeshProcess,
    _compute_cell_widths,
)
from xkep_cae_fluid.core.testing import binds_to


@binds_to(StructuredMeshProcess)
class TestStructuredMeshAPI:
    """StructuredMeshProcess の API テスト."""

    def test_registry_registered(self):
        """レジストリに登録されていること."""
        from xkep_cae_fluid.core.registry import ProcessRegistry

        reg = ProcessRegistry.default()
        assert "StructuredMeshProcess" in reg

    def test_meta_fields(self):
        """メタ情報が正しいこと."""
        proc = StructuredMeshProcess()
        assert proc.meta.name == "StructuredMesh"
        assert proc.meta.module == "pre"

    def test_uniform_3d(self):
        """等間隔3Dメッシュの基本動作."""
        inp = StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=3, nz=2)
        result = StructuredMeshProcess().process(inp)
        m = result.mesh

        assert m.n_cells == 4 * 3 * 2
        assert m.n_nodes == 5 * 4 * 3
        assert m.ndim == 3
        assert m.is_structured
        assert m.dimensions == (4, 3, 2)
        np.testing.assert_allclose(m.cell_volumes.sum(), 1.0)

    def test_uniform_cell_widths(self):
        """等間隔メッシュのセル幅が均一であること."""
        inp = StructuredMeshInput(Lx=2.0, Ly=1.0, Lz=0.5, nx=4, ny=2, nz=1)
        result = StructuredMeshProcess().process(inp)
        np.testing.assert_allclose(result.dx, 0.5)
        np.testing.assert_allclose(result.dy, 0.5)
        np.testing.assert_allclose(result.dz, 0.5)

    def test_cell_centers_within_domain(self):
        """セル中心が領域内にあること."""
        inp = StructuredMeshInput(Lx=1.0, Ly=2.0, Lz=0.5, nx=3, ny=4, nz=2)
        result = StructuredMeshProcess().process(inp)
        cc = result.mesh.cell_centers
        assert np.all(cc[:, 0] >= 0) and np.all(cc[:, 0] <= 1.0)
        assert np.all(cc[:, 1] >= 0) and np.all(cc[:, 1] <= 2.0)
        assert np.all(cc[:, 2] >= 0) and np.all(cc[:, 2] <= 0.5)

    def test_connectivity_shape(self):
        """connectivity が (n_cells, 8) であること（六面体）."""
        inp = StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
        result = StructuredMeshProcess().process(inp)
        assert result.mesh.connectivity.shape == (8, 8)

    def test_internal_face_count(self):
        """内部面数が正しいこと."""
        nx, ny, nz = 3, 4, 5
        inp = StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz)
        result = StructuredMeshProcess().process(inp)
        expected = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
        assert len(result.mesh.face_owner) == expected

    def test_face_normals_unit(self):
        """面法線が単位ベクトルであること."""
        inp = StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        result = StructuredMeshProcess().process(inp)
        norms = np.linalg.norm(result.mesh.face_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0)

    def test_origin_offset(self):
        """原点オフセットが正しく反映されること."""
        inp = StructuredMeshInput(
            Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2, origin=(10.0, 20.0, 30.0)
        )
        result = StructuredMeshProcess().process(inp)
        cc = result.mesh.cell_centers
        assert cc[:, 0].min() > 10.0
        assert cc[:, 1].min() > 20.0
        assert cc[:, 2].min() > 30.0
        assert cc[:, 0].max() < 11.0


class TestStructuredMeshPhysics:
    """StructuredMeshProcess の物理テスト（幾何学的整合性）."""

    def test_volume_sum_equals_domain(self):
        """セル体積の合計 = 領域体積."""
        Lx, Ly, Lz = 2.0, 0.5, 0.3
        inp = StructuredMeshInput(Lx=Lx, Ly=Ly, Lz=Lz, nx=10, ny=5, nz=3)
        result = StructuredMeshProcess().process(inp)
        np.testing.assert_allclose(result.mesh.cell_volumes.sum(), Lx * Ly * Lz, rtol=1e-12)

    def test_stretched_volume_sum(self):
        """ストレッチメッシュでも体積合計 = 領域体積."""
        Lx, Ly, Lz = 1.0, 0.5, 0.2
        inp = StructuredMeshInput(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            nx=8,
            ny=4,
            nz=3,
            stretch_x=(5.0, 1.0),
            stretch_y=(3.0, -1.0),
        )
        result = StructuredMeshProcess().process(inp)
        np.testing.assert_allclose(result.mesh.cell_volumes.sum(), Lx * Ly * Lz, rtol=1e-12)

    def test_geometric_stretching_ratio(self):
        """幾何級数ストレッチングの比率が正しいこと."""
        ratio = 4.0
        n = 10
        dx = _compute_cell_widths(1.0, n, (ratio, 1.0))
        actual_ratio = dx[-1] / dx[0]
        np.testing.assert_allclose(actual_ratio, ratio, rtol=1e-10)
        np.testing.assert_allclose(dx.sum(), 1.0, rtol=1e-12)

    def test_reverse_stretching(self):
        """逆方向ストレッチングの確認."""
        ratio = 4.0
        n = 10
        dx_fwd = _compute_cell_widths(1.0, n, (ratio, 1.0))
        dx_rev = _compute_cell_widths(1.0, n, (ratio, -1.0))
        np.testing.assert_allclose(dx_rev, dx_fwd[::-1], rtol=1e-12)

    def test_direct_ratio_specification(self):
        """直接比率指定のテスト."""
        ratios = (1.0, 2.0, 3.0)
        dx = _compute_cell_widths(1.2, 3, ratios)
        np.testing.assert_allclose(dx.sum(), 1.2, rtol=1e-12)
        np.testing.assert_allclose(dx[1] / dx[0], 2.0, rtol=1e-12)
        np.testing.assert_allclose(dx[2] / dx[0], 3.0, rtol=1e-12)

    def test_face_owner_neighbour_consistency(self):
        """面のオーナー < 隣接 であること（構造化格子の場合）."""
        inp = StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=3, nz=2)
        result = StructuredMeshProcess().process(inp)
        assert np.all(result.mesh.face_owner < result.mesh.face_neighbour)

    def test_face_area_times_normal_consistent(self):
        """面面積と法線が整合すること."""
        inp = StructuredMeshInput(Lx=1.0, Ly=2.0, Lz=0.5, nx=3, ny=4, nz=2)
        result = StructuredMeshProcess().process(inp)
        m = result.mesh
        # 法線ベクトルの各成分の絶対値が0か1
        for i in range(m.face_normals.shape[0]):
            n_abs = np.abs(m.face_normals[i])
            assert np.isclose(np.max(n_abs), 1.0)
            assert np.count_nonzero(n_abs > 0.5) == 1


class TestComputeCellWidths:
    """_compute_cell_widths ユーティリティのテスト."""

    def test_uniform(self):
        dx = _compute_cell_widths(2.0, 4, (1.0,))
        np.testing.assert_allclose(dx, 0.5)

    def test_single_cell(self):
        dx = _compute_cell_widths(3.0, 1, (5.0, 1.0))
        np.testing.assert_allclose(dx, [3.0])

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="正の値"):
            _compute_cell_widths(1.0, 5, (-1.0, 1.0))

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="長さが不正"):
            _compute_cell_widths(1.0, 5, (1.0, 2.0, 3.0))

    def test_both_ends_concentration(self):
        """両端集中ストレッチング: 中央が粗く、両端が細かい."""
        dx = _compute_cell_widths(1.0, 10, (4.0, 0.0))
        # 両端が細かく、中央が粗い
        assert dx[0] < dx[4]  # 左端 < 中央付近
        assert dx[-1] < dx[5]  # 右端 < 中央付近
        np.testing.assert_allclose(dx.sum(), 1.0, rtol=1e-12)
