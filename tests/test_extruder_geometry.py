"""ScrewGeometryProcess のテスト（幾何恒等式 + 格子生成）.

幾何恒等式は D・リードを振ったパラメトリックテストで固定する。
設計文書 docs/design/single-screw-extruder.md §2.1.1 に対応。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess


def spec_40mm(**kw) -> ScrewSpec:
    """設計文書 §6 の 40mm 押出機（仮諸元）."""
    base = {
        "D": 0.040,
        "lead": 0.040,
        "H": 0.004,
        "e": 0.004,
        "delta": 1.0e-4,
        "N": 100.0 / 60.0,
    }
    base.update(kw)
    return ScrewSpec(**base)


@binds_to(ScrewGeometryProcess)
class TestScrewGeometryAPI:
    """ScrewGeometryProcess の API テスト."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ScrewGeometryProcess" in ProcessRegistry.default()

    def test_meta_module(self):
        assert ScrewGeometryProcess.meta.module == "pre"

    def test_rejects_gap_larger_than_depth(self):
        with pytest.raises(ValueError, match="delta"):
            ScrewGeometryProcess().process(spec_40mm(delta=0.005))

    def test_rejects_flight_wider_than_pitch(self):
        with pytest.raises(ValueError, match="フライト幅"):
            ScrewGeometryProcess().process(spec_40mm(e=0.050))

    def test_zero_flight_gives_uniform_parallel_plate(self):
        """e=0 はフライト無しの平行平板（1D 厳密解の検証で使う正式なケース）."""
        g = ScrewGeometryProcess().process(spec_40mm(e=0.0, delta=0.0, nx_channel=8, ny_bulk=32))
        assert not g.solid.any()
        assert np.allclose(g.dx, g.dx[0], rtol=1e-12)
        assert np.allclose(g.dy, g.dy[0], rtol=1e-12)


class TestScrewGeometryPhysics:
    """幾何恒等式。D に依らないものは D を振って確認する."""

    def test_helix_angle(self):
        s = spec_40mm()
        assert s.phi == pytest.approx(math.atan(0.040 / (math.pi * 0.040)))
        assert math.degrees(s.phi) == pytest.approx(17.6568, abs=1e-4)

    def test_channel_pitch_matches_design_doc(self):
        """W_t = πD sinφ。設計文書の W=34.1mm, e=4mm と整合すること."""
        s = spec_40mm()
        assert s.W_t == pytest.approx(0.0381156, rel=1e-6)
        assert s.W == pytest.approx(0.0341156, rel=1e-6)

    def test_l_turn_is_pi_d_cos_phi(self):
        """L_turn = πD cosφ。旧版の πD/sinφ = 414mm は誤り."""
        s = spec_40mm()
        assert s.L_turn == pytest.approx(0.1197438, rel=1e-6)
        assert abs(s.L_turn - 0.414) > 0.2

    @pytest.mark.parametrize("D", [0.020, 0.040, 0.090, 0.150])
    @pytest.mark.parametrize("lead_ratio", [0.5, 1.0, 1.5])
    def test_wt_over_lturn_is_tan_phi(self, D, lead_ratio):
        """恒等式 W_t / L_turn = tanφ。D にもリードにも依らない."""
        s = spec_40mm(D=D, lead=D * lead_ratio)
        assert s.W_t / s.L_turn == pytest.approx(math.tan(s.phi), rel=1e-12)

    @pytest.mark.parametrize("D", [0.020, 0.040, 0.150])
    def test_beta_is_g_cot_phi(self, D):
        """β = G·L_turn/W_t = G·cotφ。D に依らない."""
        s = spec_40mm(D=D)
        G = 5.0e6
        assert s.beta(G) == pytest.approx(G / math.tan(s.phi), rel=1e-12)
        assert s.beta(G) == pytest.approx(G * s.L_turn / s.W_t, rel=1e-12)

    def test_pressure_gradient_is_purely_axial(self):
        """全圧力勾配 (β, G) が展開平面の軸方向 ζ̂=(cosφ, sinφ) と平行であること.

        これが本モデルの整合性の要。L_turn を πD/sinφ にすると壊れる。
        """
        s = spec_40mm()
        G = 3.0e6
        grad = np.array([s.beta(G), G])
        zeta_hat = np.array([math.cos(s.phi), math.sin(s.phi)])
        cross = grad[0] * zeta_hat[1] - grad[1] * zeta_hat[0]
        assert abs(cross) < 1e-9 * float(np.linalg.norm(grad))
        assert float(np.linalg.norm(grad)) == pytest.approx(G / math.sin(s.phi), rel=1e-12)

    def test_barrel_velocity_signs(self):
        """u_barrel = -V sinφ（-x 向き）、w_barrel = +V cosφ（下流向き）."""
        s = spec_40mm()
        assert s.V == pytest.approx(math.pi * 0.040 * (100.0 / 60.0), rel=1e-12)
        assert s.u_barrel == pytest.approx(-0.063526, rel=1e-4)
        assert s.w_barrel == pytest.approx(0.199573, rel=1e-4)
        assert s.u_barrel < 0.0 < s.w_barrel


class TestChannelGridPhysics:
    """格子生成の物理的整合性."""

    def test_gap_is_resolved(self):
        """隙間 delta に n_gap セル以上が入ること.

        1a/a02 のボクセルメッシュ品質ベンチの結論（誤差は最狭方向のセル数だけで
        決まる。1% なら 20 セル）を隙間に適用する。
        """
        s = spec_40mm(n_gap=20)
        g = ScrewGeometryProcess().process(s)
        y_face = np.concatenate([[0.0], np.cumsum(g.dy)])
        n_in_gap = int(np.sum(y_face[1:] > s.H - s.delta - 1e-15))
        assert n_in_gap >= 20

    def test_grid_sums_to_domain(self):
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert g.dx.sum() == pytest.approx(s.W_t, rel=1e-12)
        assert g.dy.sum() == pytest.approx(s.H, rel=1e-12)

    def test_adjacent_cell_ratio_is_bounded(self):
        """隣接セル幅比が 1.3 以下であること（急な格子変化は 2 次精度を壊す）."""
        for spec in (spec_40mm(), spec_40mm(e=0.0, nx_channel=8, nx_land=1)):
            g = ScrewGeometryProcess().process(spec)
            for w in (g.dx, g.dy):
                ratio = np.maximum(w[1:] / w[:-1], w[:-1] / w[1:])
                assert ratio.max() < 1.3 + 1e-9, f"最大セル比 {ratio.max()}"

    def test_flight_is_centred_and_periodic_faces_match(self):
        """周期境界 x=0 / x=W_t の列が両方とも流体（チャネル中央）であること.

        フライト側に周期境界を置くと両端の断面形状が一致せず周期条件が破綻する。
        """
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert not g.solid[0, :].any()
        assert not g.solid[-1, :].any()

    def test_flight_block_dimensions(self):
        """固体セルの面積が e × (H - delta) に一致すること."""
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        area = (g.dx[:, None] * g.dy[None, :])[g.solid].sum()
        assert area == pytest.approx(s.e * (s.H - s.delta), rel=2e-2)

    def test_delta_zero_closes_the_channel(self):
        """delta=0 で固体が y=H まで届き、閉チャネルになること（G1/G2 の形）."""
        s = spec_40mm(delta=0.0)
        g = ScrewGeometryProcess().process(s)
        i_mid = g.nx // 2
        assert g.solid[i_mid, :].all()
        assert g.area_free == pytest.approx(s.W * s.H, rel=2e-2)

    def test_delta_zero_gives_uniform_y_grid(self):
        """delta=0 では y が等間隔になること（1D 厳密解テストの前提）."""
        g = ScrewGeometryProcess().process(spec_40mm(delta=0.0, ny_bulk=32))
        assert np.allclose(g.dy, g.dy[0], rtol=1e-12)

    def test_mesh_data_is_produced(self):
        """StructuredMeshProcess 経由で MeshData が付いてくること."""
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert g.mesh.n_cells == g.nx * g.ny
        assert g.mesh.is_structured

    def test_cell_centres_are_consistent_with_widths(self):
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert np.allclose(g.xc, np.cumsum(g.dx) - g.dx / 2.0)
        assert np.allclose(g.yc, np.cumsum(g.dy) - g.dy / 2.0)
