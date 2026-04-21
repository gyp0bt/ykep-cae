"""AquariumGeometryProcess テスト (Phase 6.2a).

API テスト（Process 契約）と物理/整合性テスト（マスク形状・体積保存・
底床 refinement）を含む。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.aquarium import (
    AquariumGeometryInput,
    AquariumGeometryProcess,
    AquariumGeometryResult,
)
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.core.testing import binds_to


@binds_to(AquariumGeometryProcess)
class TestAquariumGeometryAPI:
    """AquariumGeometryProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        assert AquariumGeometryProcess.meta.name == "AquariumGeometryProcess"
        assert AquariumGeometryProcess.meta.module == "pre"

    def test_default_input_returns_result(self):
        """デフォルト入力（90×30×45 cm）で AquariumGeometryResult が返される."""
        res = AquariumGeometryProcess().process(AquariumGeometryInput())
        assert isinstance(res, AquariumGeometryResult)
        assert isinstance(res.mesh, MeshData)
        # デフォルト値
        nx, ny, nz = 36, 12, 30
        assert res.substrate_mask.shape == (nx, ny, nz)
        assert res.glass_mask.shape == (nx, ny, nz)
        assert res.water_mask.shape == (nx, ny, nz)
        assert res.solid_mask.shape == (nx, ny, nz)

    def test_solid_mask_is_union_of_substrate_and_glass(self):
        """solid_mask == substrate_mask | glass_mask の整合性."""
        inp = AquariumGeometryInput(
            Lx=0.9,
            Ly=0.3,
            Lz=0.45,
            nx=10,
            ny=6,
            nz=10,
            substrate_depth=0.05,
            glass_thickness=0.01,
        )
        res = AquariumGeometryProcess().process(inp)
        expected = res.substrate_mask | res.glass_mask
        assert np.array_equal(res.solid_mask, expected)
        # water = ~solid
        assert np.array_equal(res.water_mask, ~res.solid_mask)

    def test_gravity_vector_is_vertical_z(self):
        """推奨 gravity は z 軸下向き."""
        res = AquariumGeometryProcess().process(AquariumGeometryInput())
        assert res.gravity == (0.0, 0.0, -9.81)

    def test_negative_substrate_depth_raises(self):
        with pytest.raises(ValueError):
            AquariumGeometryProcess().process(AquariumGeometryInput(substrate_depth=-0.01))

    def test_substrate_filling_domain_raises(self):
        """substrate_depth >= Lz で底床が水槽全体を満たす場合は ValueError."""
        with pytest.raises(ValueError):
            AquariumGeometryProcess().process(AquariumGeometryInput(Lz=0.45, substrate_depth=0.45))


class TestAquariumGeometryPhysics:
    """物理的整合性テスト."""

    def test_no_substrate_no_glass(self):
        """substrate_depth=0 / glass=0 で固体なし."""
        inp = AquariumGeometryInput(
            Lx=0.9,
            Ly=0.3,
            Lz=0.45,
            nx=6,
            ny=4,
            nz=6,
            substrate_depth=0.0,
            glass_thickness=0.0,
        )
        res = AquariumGeometryProcess().process(inp)
        assert not res.substrate_mask.any()
        assert not res.glass_mask.any()
        assert res.water_mask.all()

    def test_substrate_confined_to_lower_z(self):
        """底床セルは z 座標下端 substrate_depth 以下."""
        sd = 0.05
        inp = AquariumGeometryInput(
            Lx=0.9,
            Ly=0.3,
            Lz=0.45,
            nx=6,
            ny=4,
            nz=10,
            substrate_depth=sd,
        )
        res = AquariumGeometryProcess().process(inp)
        # 底床セル (i, j, k) の z 中心は sd 以下（セル幅分の余裕あり）
        for k in range(res.substrate_mask.shape[2]):
            if res.substrate_mask[:, :, k].any():
                # このセル層は底床に含まれるべき
                assert res.z_centers[k] <= sd + 1e-12
            else:
                # 底床外のセル層は中心が sd より上
                assert res.z_centers[k] > sd + 1e-12

    def test_total_volume_conserved(self):
        """セル体積の合計が Lx*Ly*Lz."""
        Lx, Ly, Lz = 0.9, 0.3, 0.45
        inp = AquariumGeometryInput(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            nx=6,
            ny=4,
            nz=8,
            substrate_refinement_ratio=3.0,
        )
        res = AquariumGeometryProcess().process(inp)
        total = float(res.mesh.cell_volumes.sum())
        assert abs(total - Lx * Ly * Lz) / (Lx * Ly * Lz) < 1e-10

    def test_refinement_bottom_finer_than_top(self):
        """substrate_refinement_ratio > 1 で底床側 dz が上面側より小さい."""
        inp = AquariumGeometryInput(
            Lx=0.9,
            Ly=0.3,
            Lz=0.45,
            nx=4,
            ny=4,
            nz=10,
            substrate_refinement_ratio=3.0,
        )
        res = AquariumGeometryProcess().process(inp)
        assert res.dz[0] < res.dz[-1]
        # 比率が指定通り（厳密ではないがおおむね ratio 倍）
        assert res.dz[-1] / res.dz[0] == pytest.approx(3.0, rel=0.05)

    def test_no_refinement_is_uniform(self):
        """substrate_refinement_ratio=1.0 で dz は等間隔."""
        inp = AquariumGeometryInput(
            Lx=0.9,
            Ly=0.3,
            Lz=0.45,
            nx=4,
            ny=4,
            nz=10,
            substrate_refinement_ratio=1.0,
        )
        res = AquariumGeometryProcess().process(inp)
        assert np.allclose(res.dz, res.dz[0])

    def test_glass_on_xy_edges(self):
        """ガラス厚>0 で x/y 両端がガラス、z 端はガラスなし."""
        # 1 セル層が確実にガラスに入る厚さを採用（dx=0.0625 なので glass=0.1）
        Lx = 1.0
        Ly = 0.3
        inp = AquariumGeometryInput(
            Lx=Lx,
            Ly=Ly,
            Lz=0.45,
            nx=16,
            ny=6,
            nz=4,
            substrate_depth=0.0,
            glass_thickness=0.08,
        )
        res = AquariumGeometryProcess().process(inp)
        # dx = Lx/nx = 0.0625 <= 0.08 なので最初と最後の x 層はガラス
        assert res.glass_mask[0, :, :].all()
        assert res.glass_mask[-1, :, :].all()
        # dy = Ly/ny = 0.05 <= 0.08 なので最初と最後の y 層もガラス
        assert res.glass_mask[:, 0, :].all()
        assert res.glass_mask[:, -1, :].all()
        # 水槽の中心部（x, y 内側、z 中央）は水領域
        mid_x, mid_y, mid_z = inp.nx // 2, inp.ny // 2, inp.nz // 2
        assert res.water_mask[mid_x, mid_y, mid_z]
        # z 方向端はガラスにしない（上面=水面、下面=底床）
        assert not res.glass_mask[mid_x, mid_y, 0]
        assert not res.glass_mask[mid_x, mid_y, -1]

    def test_x_centers_within_domain(self):
        """セル中心座標が [origin, origin+L] 内."""
        inp = AquariumGeometryInput(Lx=0.9, Ly=0.3, Lz=0.45, nx=6, ny=4, nz=8)
        res = AquariumGeometryProcess().process(inp)
        assert (res.x_centers > 0).all() and (res.x_centers < 0.9).all()
        assert (res.y_centers > 0).all() and (res.y_centers < 0.3).all()
        assert (res.z_centers > 0).all() and (res.z_centers < 0.45).all()


class TestAquariumGeometryIntegration:
    """NaturalConvectionInput との互換性."""

    def test_solid_mask_shape_matches_natconvection(self):
        """solid_mask の形状・dtype が NaturalConvectionInput.solid_mask と互換."""
        res = AquariumGeometryProcess().process(
            AquariumGeometryInput(Lx=0.9, Ly=0.3, Lz=0.45, nx=6, ny=4, nz=6)
        )
        assert res.solid_mask.dtype == bool
        assert res.solid_mask.ndim == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
