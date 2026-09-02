"""ParticleTrackerProcess のテスト。ゲート G4a を含む."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def flow_of(spec: ScrewSpec, G: float):
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    return proc.process(ExtruderFlowInput(spec=spec, G=G))


def flow_parallel_plate(ny: int = 24, G: float = -2.0e6):
    """フライト無しの平行平板。1D 厳密解が使えるので G4a の土台になる.

    G<0（下流ほど低圧）にしないと軸方向流れが後戻りになる。フライトが無ければ
    引きずりだけでは押し出せない（u cotφ + w ≡ 0）ので、圧力だけが駆動源。
    """
    spec = replace(_BASE, e=0.0, delta=0.0, nx_channel=8, nx_land=1, ny_bulk=ny, n_gap=0)
    return flow_of(spec, G)


def flow_with_gap(ny: int = 16, n_gap: int = 6, nx_channel: int = 40):
    spec = replace(_BASE, delta=1.0e-4, nx_channel=nx_channel, nx_land=12, ny_bulk=ny, n_gap=n_gap)
    return flow_of(spec, 5.0e6)


def track(flow, z_axial: float, **kw):
    return ParticleTrackerProcess().process(ParticleTrackInput(flow=flow, z_axial=z_axial, **kw))


@binds_to(ParticleTrackerProcess)
class TestParticleTrackerAPI:
    """API・入力検証."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ParticleTrackerProcess" in ProcessRegistry.default()

    def test_meta_module(self):
        assert ParticleTrackerProcess.meta.module == "post"

    def test_rejects_bad_length(self):
        flow = flow_parallel_plate(ny=8)
        with pytest.raises(ValueError, match="z_axial"):
            track(flow, 0.0)

    def test_rejects_bad_stride(self):
        flow = flow_parallel_plate(ny=8)
        with pytest.raises(ValueError, match="stride"):
            track(flow, 0.01, stride=0)


class TestSeeding:
    """流束重み付き種まき."""

    def test_weight_sum_matches_axial_throughput(self):
        """重みの総和が離散ソルバーの Q_axial と 2% 以内で一致すること.

        重みは補間場から作るので、離散場との差（壁際の節点値の扱い）が残る。
        差は格子細分で 1 次で消える（下のテストで確認）。
        """
        flow = flow_with_gap()
        tr = track(flow, 0.005)
        assert tr.weight.sum() == pytest.approx(flow.Q_axial, rel=2e-2)

    def test_interpolated_flux_converges_to_discrete(self):
        """補間場の流束が格子細分とともに離散 Q_axial に収束すること.

        節点場は壁で値を打ち切る（w=0 / w_barrel）ので、壁際の双一次補間は
        セル中心値の並びと厳密には一致しない。実測で誤差は
        1.0e-2 → 7.6e-3 → 4.6e-3 → 2.9e-3 と 1 次で縮む。
        """
        errs = []
        for ny, n_gap, nxc in ((16, 6, 40), (30, 12, 90), (40, 16, 120)):
            spec = replace(_BASE, delta=1.0e-4, nx_channel=nxc, nx_land=12, ny_bulk=ny, n_gap=n_gap)
            flow = flow_of(spec, 5.0e6)
            tr = track(flow, 0.002, max_steps=50)
            errs.append(abs(tr.weight.sum() / flow.Q_axial - 1.0))
        assert errs[0] > errs[1] > errs[2], f"収束していない: {errs}"
        assert errs[-1] < 5e-3

    def test_weights_are_positive(self):
        tr = track(flow_with_gap(), 0.02)
        assert np.all(tr.weight > 0.0)

    def test_stride_preserves_total_weight(self):
        """間引いても総流束が保たれること（重みを stride² 倍している）."""
        flow = flow_with_gap()
        full = track(flow, 0.005, stride=1)
        thin = track(flow, 0.005, stride=2)
        assert thin.weight.sum() == pytest.approx(full.weight.sum(), rel=0.1)
        assert thin.t_res.shape[0] < full.t_res.shape[0]

    def test_seeds_are_in_fluid(self):
        flow = flow_with_gap()
        tr = track(flow, 0.005)
        g = flow.grid
        i = np.clip(np.searchsorted(np.cumsum(g.dx), tr.x0), 0, g.nx - 1)
        j = np.clip(np.searchsorted(np.cumsum(g.dy), tr.y0), 0, g.ny - 1)
        assert not g.solid[i, j].any()


class TestGateG4a:
    """G4a: 1D 厳密解に対する軌跡の検証."""

    def test_y_is_conserved_in_one_dimensional_flow(self):
        """v≡0 なので粒子の y が機械精度で保存すること."""
        flow = flow_parallel_plate()
        tr = track(flow, 0.02)
        assert np.max(np.abs(tr.y - tr.y0)) < 1e-12 * flow.grid.spec.H

    def test_residence_time_matches_constant_axial_speed(self):
        """1D 流れでは dζ/dt が経路上一定なので t_res = z_axial/(dζ/dt) が厳密.

        dζ/dt は種まき重みから逆算できる（weight = (dζ/dt)/sinφ · dA）ので、
        追跡結果だけで閉じた検証になる。RK4・ζ 積分・脱出時刻の内挿・
        x 周期の折り返しを一度に検証する。
        """
        flow = flow_parallel_plate()
        z_axial = 0.02
        tr = track(flow, z_axial)
        g = flow.grid
        s = g.spec

        i = np.clip(np.searchsorted(np.cumsum(g.dx), tr.x0), 0, g.nx - 1)
        j = np.clip(np.searchsorted(np.cumsum(g.dy), tr.y0), 0, g.ny - 1)
        area = g.dx[i] * g.dy[j]
        dzeta_dt = tr.weight * math.sin(s.phi) / area
        expect = z_axial / dzeta_dt

        ok = tr.escaped & ~tr.extrapolated
        assert ok.sum() > 0
        rel = np.abs(tr.t_res[ok] / expect[ok] - 1.0)
        assert np.max(rel) < 1e-8, f"最大相対誤差 {np.max(rel):.2e}"

    def test_axial_speed_matches_analytic_profile(self):
        """1D 解 dζ/dt = (Hy − y²)·(−G)/(2μ sinφ) と一致すること.

        u = u_b·η + (β/2μ)(y²−Hy), w = w_b·η + (G/2μ)(y²−Hy), β = G cotφ から
        引きずり成分が u_b cosφ + w_b sinφ = 0 で消え、圧力成分だけが残る。
        """
        G = -2.0e6
        flow = flow_parallel_plate(ny=64, G=G)
        g = flow.grid
        s = g.spec
        axial = flow.u * math.cos(s.phi) + flow.w * math.sin(s.phi)
        y = g.yc
        expect = (s.H * y - y**2) * (-G) / (2.0 * MU * math.sin(s.phi))
        rel = np.abs(axial[0, :] / expect - 1.0)
        # 壁隣接セルは半セル Dirichlet の O(h) 誤差が相対的に効く（放物線が
        # そこで 0 に向かうため）。内部は 1 桁良い。
        assert np.max(rel[1:-1]) < 3e-3, f"内部の最大相対誤差 {np.max(rel[1:-1]):.2e}"
        assert np.max(rel) < 1e-2, f"全体の最大相対誤差 {np.max(rel):.2e}"

    def test_pure_drag_has_no_axial_flow(self):
        """フライト無し・G=0 では軸方向速度が全域で厳密にゼロになること.

        u cotφ + w = (y/H)(−V sinφ·cotφ + V cosφ) = 0。
        フライトが無ければ材料は周方向に回るだけで押し出されない。
        """
        flow = flow_parallel_plate(G=0.0)
        s = flow.grid.spec
        axial = flow.u * math.cos(s.phi) + flow.w * math.sin(s.phi)
        assert np.max(np.abs(axial)) < 1e-12 * s.V

    def test_cumulative_shear_matches_constant_rate(self):
        """1D 流れでは γ̇ が経路上一定なので γ_total = γ̇·t_res になること."""
        flow = flow_parallel_plate()
        tr = track(flow, 0.02)
        ok = tr.escaped & ~tr.extrapolated
        rate = tr.gamma_total[ok] / tr.t_res[ok]
        # 同じ y から出た粒子は同じ γ̇ を持つ（x に依らない）
        for y_val in np.unique(np.round(tr.y0[ok], 12))[:5]:
            same = np.isclose(tr.y0[ok], y_val)
            if same.sum() > 1:
                assert np.ptp(rate[same]) < 1e-8 * np.mean(rate[same])


class TestTrackerPhysics:
    """隙間ありの一般ケース."""

    def test_particles_do_not_enter_solid(self):
        flow = flow_with_gap()
        tr = track(flow, 0.02)
        g = flow.grid
        i = np.clip(np.searchsorted(np.cumsum(g.dx), tr.x), 0, g.nx - 1)
        j = np.clip(np.searchsorted(np.cumsum(g.dy), tr.y), 0, g.ny - 1)
        assert not g.solid[i, j].any()

    def test_wraps_are_backward_dominated_when_pumping(self):
        """背圧下では x 周期の跨ぎが −x（上流）側に偏ること."""
        flow = flow_with_gap()
        tr = track(flow, 0.02)
        assert tr.n_wraps.min() < 0
        assert np.average(tr.n_wraps, weights=tr.weight) < 0.0

    def test_cumulative_shear_is_positive_and_finite(self):
        tr = track(flow_with_gap(), 0.02)
        assert np.all(tr.gamma_total > 0.0)
        assert np.all(np.isfinite(tr.gamma_total))

    def test_mixing_index_is_in_range(self):
        tr = track(flow_with_gap(), 0.02)
        ok = tr.escaped
        assert np.all(tr.lambda_mean[ok] >= 0.0)
        assert np.all(tr.lambda_mean[ok] <= 1.0)

    def test_extrapolation_is_flagged(self):
        """ステップ上限を切れば外挿された粒子が出て、フラグが立つこと."""
        flow = flow_with_gap()
        tr = track(flow, 0.05, max_steps=200)
        assert tr.extrapolated.any()
        assert np.all(tr.escaped[tr.extrapolated])
