"""RTDProcess のテスト。ゲート G4b を含む."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    RTDInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.rtd import RTDProcess, weighted_ecdf, weighted_quantile
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
Z_AXIAL = 0.050
"""検証用の軸方向長さ [m]。実機の計量部（5D = 200mm）より短くして計算を軽くする."""

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def spec_closed(ny: int = 16, nx_channel: int = 40) -> ScrewSpec:
    return replace(_BASE, nx_channel=nx_channel, nx_land=8, ny_bulk=ny, n_gap=0)


def spec_gap(ny: int = 16, n_gap: int = 6, nx_channel: int = 40) -> ScrewSpec:
    return replace(_BASE, delta=1.0e-4, nx_channel=nx_channel, nx_land=12, ny_bulk=ny, n_gap=n_gap)


def pipeline(spec: ScrewSpec, G: float, z_axial: float = Z_AXIAL, **track_kw):
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    track = ParticleTrackerProcess().process(
        ParticleTrackInput(flow=flow, z_axial=z_axial, **track_kw)
    )
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=z_axial, n_bins=100))
    return flow, track, rtd


class TestWeightedQuantile:
    """重み付き分位点のヘルパー."""

    def test_uniform_weights_match_numpy(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones_like(v)
        q = weighted_quantile(v, w, np.array([0.5]))
        assert q[0] == pytest.approx(3.0)

    def test_weight_shifts_the_median(self):
        v = np.array([1.0, 10.0])
        assert weighted_quantile(v, np.array([99.0, 1.0]), 0.5) < 2.0
        assert weighted_quantile(v, np.array([1.0, 99.0]), 0.5) > 9.0

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="重み"):
            weighted_quantile(np.array([1.0]), np.array([0.0]), 0.5)


class TestWeightedEcdf:
    """重み付き経験分布。`weighted_quantile` と同じ中点流儀なので分位点が逆算で一致する."""

    def test_matches_weighted_quantile(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(1.0, 3.0, 500)
        w = rng.uniform(0.1, 1.0, 500)
        t, F = weighted_ecdf(v, w)
        assert np.all(np.diff(t) >= 0.0)
        assert 0.0 < F[0] < F[-1] < 1.0
        for q in (0.1, 0.5, 0.9):
            assert np.interp(q, F, t) == pytest.approx(float(weighted_quantile(v, w, q)))

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="重み"):
            weighted_ecdf(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


@binds_to(RTDProcess)
class TestRTDAPI:
    """API と分布の整合性."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "RTDProcess" in ProcessRegistry.default()

    def test_meta_module(self):
        assert RTDProcess.meta.module == "post"

    def test_density_integrates_to_one(self):
        _, _, rtd = pipeline(spec_closed(), 1.0e6)
        assert float(np.sum(rtd.E * np.diff(rtd.t_edges))) == pytest.approx(1.0, rel=1e-12)

    def test_cumulative_is_monotone_and_reaches_one(self):
        _, _, rtd = pipeline(spec_closed(), 1.0e6)
        assert np.all(np.diff(rtd.F) >= -1e-15)
        assert rtd.F[0] == 0.0
        assert rtd.F[-1] == pytest.approx(1.0, rel=1e-12)

    def test_percentiles_are_ordered(self):
        _, _, rtd = pipeline(spec_closed(), 1.0e6)
        assert rtd.t_min <= rtd.t_p10 < rtd.t_p50 < rtd.t_p90
        assert rtd.gamma_p10 < rtd.gamma_p50 < rtd.gamma_p90
        assert rtd.spread == pytest.approx(rtd.t_p90 / rtd.t_p10)


class TestGateG4b:
    """G4b: 厳密関係 ⟨t⟩ = z_axial·A_free/(sinφ·Q) との照合.

    **隙間ありの構成で判定する。** 隙間ゼロの閉チャネルは壁沿いの閉じた流線が
    周方向へ逃げ場を持たず、隅で周期が発散するデッドゾーンになるため、平均が
    裾に支配されて格子細分でも収束しない（TestDeadZone 参照）。実機のスクリューには
    必ず隙間があるので、判定は隙間ありで行うのが妥当。
    """

    def test_mean_residence_time_matches_volume_over_flux(self):
        """平均滞留時間が体積÷流束と 1.5% 以内で一致すること.

        補間誤差・脱出時刻の内挿ミス・種まき重みの誤りを同時に捕まえる。
        """
        _, _, rtd = pipeline(spec_gap(), 5.0e6)
        assert rtd.t_mean == pytest.approx(rtd.t_mean_theory, rel=1.5e-2)

    def test_mean_converges_with_grid(self):
        """格子細分で ⟨t⟩ が理論値に近づくこと.

        誤差は 1% 前後から出発して細分で半分になる（実測 0.99% → 1.00% → 0.52%）。
        中間格子で一旦横ばうのは、隙間セル数とバルクセル数の比が格子ごとに
        変わり自己相似な細分になっていないため。粗→細で確実に縮むことを見る。
        """
        errs = []
        for ny, n_gap, nxc in ((16, 6, 40), (32, 10, 80), (48, 16, 120)):
            _, _, rtd = pipeline(spec_gap(ny=ny, n_gap=n_gap, nx_channel=nxc), 5.0e6)
            errs.append(abs(rtd.t_mean / rtd.t_mean_theory - 1.0))
        assert errs[-1] < 0.6 * errs[0], f"収束していない: {errs}"
        assert max(errs) < 1.5e-2

    def test_scales_linearly_with_section_length(self):
        """軸方向長さを 2 倍にすれば平均滞留時間も 2 倍になること."""
        _, _, a = pipeline(spec_gap(), 5.0e6, z_axial=0.025)
        _, _, b = pipeline(spec_gap(), 5.0e6, z_axial=0.050)
        assert b.t_mean == pytest.approx(2.0 * a.t_mean, rel=3e-2)

    def test_minimum_residence_time_bound(self):
        """最短滞留時間が z_axial / max(dζ/dt) を下回らないこと.

        比較対象は追跡に使う**補間場**の最大値（セル中心値の最大ではない。
        ψ から作る u は面平均、w の節点にはバレル値が入るので、補間場の方が
        速い点を持ちうる）。
        """
        import math

        from xkep_cae_fluid.extruder.tracker import _Interpolator

        flow, _, rtd = pipeline(spec_gap(), 5.0e6)
        g = flow.grid
        s = g.spec
        interp = _Interpolator(flow)
        xs = np.linspace(0.0, s.W_t, 401)[:-1]
        ys = np.linspace(0.0, s.H, 401)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        u, _v, w = interp.velocity(xx.ravel(), yy.ravel())
        axial_max = float((u * math.cos(s.phi) + w * math.sin(s.phi)).max())
        assert rtd.t_min >= Z_AXIAL / axial_max * (1.0 - 1e-6)
        assert rtd.t_min < rtd.t_p10

    def test_unresolved_weight_is_negligible(self):
        """脱出も外挿もできなかった粒子の重みが無視できること."""
        _, _, rtd = pipeline(spec_gap(), 5.0e6)
        assert rtd.unresolved_weight_fraction < 1e-3


class TestPercentilesConverge:
    """パーセンタイルは平均と違って格子に対して安定であること.

    平均は壁境界層の長い裾に支配されるが、工程設計で使うのは分布の広がりと
    累積せん断の中央値なので、そちらが収束していることが実用上の要件になる。
    """

    @pytest.mark.parametrize("with_gap", [False, True])
    def test_median_and_spread_are_grid_converged(self, with_gap):
        results = []
        for ny, nxc in ((16, 40), (32, 80)):
            spec = (
                spec_gap(ny=ny, n_gap=max(4, ny // 3), nx_channel=nxc)
                if with_gap
                else spec_closed(ny=ny, nx_channel=nxc)
            )
            _, _, rtd = pipeline(spec, 5.0e6 if with_gap else 1.0e6)
            results.append(rtd)
        a, b = results
        assert b.t_p50 / b.t_mean_theory == pytest.approx(a.t_p50 / a.t_mean_theory, rel=3e-2)
        assert b.t_p10 / b.t_mean_theory == pytest.approx(a.t_p10 / a.t_mean_theory, rel=3e-2)
        assert b.gamma_p50 == pytest.approx(a.gamma_p50, rel=5e-2)


class TestDeadZone:
    """隙間ゼロの理想化はデッドゾーンを作るという知見."""

    def test_zero_clearance_has_a_heavier_tail(self):
        """隙間ゼロの方が「平均の 10 倍以上」滞留する材料の割合が大きいこと.

        閉チャネルでは壁沿いの閉じた流線が周方向へ抜けられず、隅で周期が発散する。
        隙間があると壁際の材料がフライトを越えて逃げられるのでトラップが壊れる。
        実機のスクリューに隙間があることが、淀みを作らない条件になっている。
        """
        _, tr_closed, r_closed = pipeline(spec_closed(ny=32, nx_channel=80), 1.0e6)
        _, tr_gap, r_gap = pipeline(spec_gap(ny=32, n_gap=10, nx_channel=80), 5.0e6)

        def tail_fraction(track, rtd):
            ok = track.escaped
            long = ok & (track.t_res > 10.0 * rtd.t_mean_theory)
            return float(track.weight[long].sum() / track.weight[ok].sum())

        assert tail_fraction(tr_closed, r_closed) > tail_fraction(tr_gap, r_gap)

    def test_zero_clearance_mean_is_not_reliable(self):
        """閉チャネルの平均は裾支配で理論値から大きく外れうること（既知の限界）.

        この構成で ⟨t⟩ を判定に使ってはいけない、という記録。
        パーセンタイルは同じ構成でも収束する（TestPercentilesConverge）。
        """
        _, _, rtd = pipeline(spec_closed(ny=32, nx_channel=80), 1.0e6)
        assert rtd.t_mean > 1.2 * rtd.t_mean_theory


class TestRTDPhysics:
    """混練性の物理."""

    def test_clearance_broadens_the_short_time_end(self):
        """隙間があると隙間を突っ切る速い経路が生まれ、最短滞留時間が縮むこと."""
        _, _, closed = pipeline(spec_closed(), 5.0e6)
        _, _, gap = pipeline(spec_gap(), 5.0e6)
        assert gap.t_min / gap.t_mean_theory < closed.t_min / closed.t_mean_theory

    def test_cumulative_shear_correlates_with_residence_time(self):
        """累積せん断が滞留時間と正の相関を持つこと（長く居れば混ざる）."""
        _, track, _ = pipeline(spec_gap(), 5.0e6)
        ok = track.escaped
        r = np.corrcoef(track.t_res[ok], track.gamma_total[ok])[0, 1]
        assert r > 0.5

    def test_mixing_index_is_shear_dominated(self):
        """混合指数の平均が 0.5 付近（せん断主体）に来ること.

        押出のチャネル流れはほぼ単純せん断なので λ ≈ 0.5。伸長流れが
        支配的なら 1 に寄る。
        """
        _, _, rtd = pipeline(spec_gap(), 5.0e6)
        assert 0.4 < rtd.lambda_mean < 0.6

    def test_back_pressure_lengthens_residence(self):
        """背圧を上げると押出量が減り、平均滞留時間が延びること."""
        _, _, low = pipeline(spec_gap(), 1.0e6)
        _, _, high = pipeline(spec_gap(), 6.0e6)
        assert high.t_mean_theory > low.t_mean_theory
