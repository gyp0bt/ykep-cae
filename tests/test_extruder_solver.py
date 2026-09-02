"""ExtruderFlowProcess のテスト（Picard による粘度結合）."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ScrewSpec
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
)

MU = 1000.0
_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def spec_closed(ny: int = 64, nx_channel: int = 200) -> ScrewSpec:
    """delta=0 の閉チャネル（x も y も等間隔になるようランド幅を揃える）."""
    half = 0.5 * (_BASE.W_t - _BASE.e)
    n_half = max(2, nx_channel // 2)
    return replace(
        _BASE,
        nx_channel=nx_channel,
        nx_land=max(1, round(_BASE.e / (half / n_half))),
        ny_bulk=ny,
        n_gap=0,
    )


def spec_gap(ny: int = 40, n_gap: int = 16, nx_channel: int = 160) -> ScrewSpec:
    return replace(_BASE, delta=1.0e-4, nx_channel=nx_channel, nx_land=40, ny_bulk=ny, n_gap=n_gap)


def run(spec: ScrewSpec, G: float, model, **kw):
    proc = ExtruderFlowProcess()
    proc.viscosity = model
    return proc.process(ExtruderFlowInput(spec=spec, G=G, **kw))


@binds_to(ExtruderFlowProcess)
class TestExtruderFlowAPI:
    """API・Strategy 注入の契約."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ExtruderFlowProcess" in ProcessRegistry.default()

    def test_strategy_slot_is_required(self):
        with pytest.raises(AttributeError, match="viscosity"):
            ExtruderFlowProcess().process(
                ExtruderFlowInput(spec=spec_gap(ny=8, n_gap=4, nx_channel=20), G=0.0)
            )

    def test_effective_uses_includes_viscosity_strategy(self):
        proc = ExtruderFlowProcess()
        proc.viscosity = NewtonianViscosity(mu=MU)
        assert NewtonianViscosity in proc.effective_uses()

    def test_rejects_bad_relaxation(self):
        with pytest.raises(ValueError, match="relax_mu"):
            run(
                spec_gap(ny=8, n_gap=4, nx_channel=20), 0.0, NewtonianViscosity(mu=MU), relax_mu=0.0
            )

    def test_declared_uses(self):
        from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
        from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
        from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

        assert set(ExtruderFlowProcess.uses) == {
            ScrewGeometryProcess,
            DownChannelFlowProcess,
            CrossChannelStokesProcess,
        }


class TestNewtonianConsistency:
    """ニュートン極限での整合性."""

    def test_converges_in_one_iteration(self):
        """粘度が動かないので 1 反復で収束すること."""
        res = run(spec_closed(ny=24, nx_channel=80), 0.0, NewtonianViscosity(mu=MU))
        assert res.converged
        assert res.n_iter == 1
        assert res.mu_history == (0.0,)

    def test_reproduces_gate_g2(self):
        """結合ソルバー経由でも G2 の解析解と 0.1% 以内で一致すること."""
        spec = spec_closed(ny=96, nx_channel=384)
        G = 1.0e6
        res = run(spec, G, NewtonianViscosity(mu=MU))
        fd = shape_factor_drag(spec.H / spec.W)
        fp = shape_factor_pressure(spec.H / spec.W)
        q = metering_flow_rate(spec.w_barrel, spec.W, spec.H, MU, G, F_d=fd, F_p=fp)
        assert res.Q == pytest.approx(q, rel=1e-3)

    def test_carreau_n1_matches_newtonian(self):
        """Carreau で n=1, μ0=μ∞ ならニュートンと同じ Q になること."""
        spec = spec_closed(ny=32, nx_channel=120)
        a = run(spec, 1.0e6, NewtonianViscosity(mu=MU))
        b = run(spec, 1.0e6, CarreauViscosity(mu_0=MU, mu_inf=MU, lam=1.0, n=1.0))
        assert b.Q == pytest.approx(a.Q, rel=1e-12)

    def test_power_law_n1_matches_newtonian(self):
        """べき乗則で n=1, K=μ ならニュートンと同じ Q になること."""
        spec = spec_closed(ny=32, nx_channel=120)
        a = run(spec, 1.0e6, NewtonianViscosity(mu=MU))
        b = run(spec, 1.0e6, PowerLawViscosity(K=MU, n=1.0))
        assert b.Q == pytest.approx(a.Q, rel=1e-12)


class TestNonNewtonian:
    """べき乗則の Picard 収束と挙動."""

    def test_power_law_converges(self):
        res = run(spec_gap(), 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        assert res.converged, f"収束しなかった: n_iter={res.n_iter}, 履歴末尾={res.mu_history[-3:]}"
        assert res.n_iter < 100

    def test_mu_history_decreases(self):
        """粘度場の変化量が単調に縮んでいくこと（不動点反復の健全性）."""
        res = run(spec_gap(), 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        hist = np.array(res.mu_history)
        assert hist[-1] < hist[len(hist) // 2] < hist[0]

    def test_thins_where_shear_is_highest(self):
        """隙間はせん断速度が最大なので、粘度がチャネル中心より小さいこと."""
        spec = spec_gap()
        res = run(spec, 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        g = res.grid
        i_land = g.nx // 2
        assert res.gamma_dot[i_land, -1] > res.gamma_dot[0, g.ny // 2]
        assert res.mu[i_land, -1] < res.mu[0, g.ny // 2]

    def test_gamma_min_does_not_change_the_answer(self):
        """γ̇ クランプは数値上の安全弁であり、結果を動かさないこと."""
        spec = spec_gap()
        qs = [
            run(spec, 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=gm)).Q
            for gm in (1e-3, 1e-2, 1e-1)
        ]
        assert qs[1] == pytest.approx(qs[0], rel=1e-3)
        assert qs[2] == pytest.approx(qs[0], rel=1e-3)

    def test_relaxation_does_not_change_the_fixed_point(self):
        """緩和係数を変えても収束先（不動点）は同じであること.

        一致の精度は収束判定 tol で決まる（tol=1e-6 で 5.5e-6、tol=1e-9 で 3.1e-8 と
        実測で連動する）ので、tol を締めて比較する。
        """
        spec = spec_gap()
        model = PowerLawViscosity(K=2.0e4, n=0.4)
        a = run(spec, 5.0e6, model, relax_mu=0.5, tol=1e-9, max_iter=200)
        b = run(spec, 5.0e6, model, relax_mu=0.8, tol=1e-9, max_iter=200)
        assert a.converged and b.converged
        assert b.Q == pytest.approx(a.Q, rel=1e-7)
        assert b.Q_axial == pytest.approx(a.Q_axial, rel=1e-7)

    def test_reports_non_convergence_honestly(self):
        """反復上限を 2 に切れば converged=False を返すこと（収束の詐称をしない）."""
        res = run(spec_gap(), 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4), max_iter=2)
        assert not res.converged
        assert res.n_iter == 2


class TestLeakage:
    """隙間による漏れ流れ."""

    def test_leak_flux_is_negative(self):
        """漏れ流れは −x（上流へ戻る）向きであること."""
        res = run(spec_gap(), 5.0e6, NewtonianViscosity(mu=MU))
        assert res.Q_leak < 0.0

    def test_axial_throughput_identity(self):
        """Q_axial = Q + L_turn·Q_leak が成り立つこと."""
        res = run(spec_gap(), 5.0e6, NewtonianViscosity(mu=MU))
        expect = res.Q + res.grid.spec.L_turn * res.Q_leak
        assert res.Q_axial == pytest.approx(expect, rel=1e-14)

    def test_cross_flux_is_independent_of_x(self):
        """どの x 面を通る横断流束も同じであること（断面内 2D 非圧縮の帰結）.

        Q_axial = Q + L_turn·Q_leak の導出で ∫∫u dA = Q_leak·W_t を使うが、
        それはこの性質に依っている。
        """
        res = run(spec_gap(), 5.0e6, NewtonianViscosity(mu=MU))
        g = res.grid
        fluxes = np.array([float(np.sum(res.u_face[i, :] * g.dy)) for i in range(g.nx)])
        assert np.max(np.abs(fluxes - fluxes.mean())) < 1e-10 * abs(fluxes.mean())
        area_int = float(np.sum(res.u * (g.dx[:, None] * g.dy[None, :])))
        assert area_int == pytest.approx(res.Q_leak * g.spec.W_t, rel=1e-8)

    def test_clearance_raises_Q_but_lowers_throughput(self):
        """隙間は Q(=∫∫w dA) を増やすが、押出量 Q_axial は減らすこと.

        隙間は w が最大になるバレル直下に e×δ の流路を足すので Q は増える。
        しかし漏れが材料を L_turn だけ戻すので、軸方向の正味流量は減る。
        「隙間は押出量を減らす」という古典的描像は Q ではなく Q_axial の話。
        """
        model = NewtonianViscosity(mu=MU)
        closed = run(spec_closed(ny=40, nx_channel=160), 5.0e6, model)
        gap = run(spec_gap(), 5.0e6, model)
        assert closed.Q_leak == pytest.approx(0.0, abs=1e-18)
        assert closed.Q_axial == pytest.approx(closed.Q, rel=1e-14)
        assert gap.Q > closed.Q
        assert gap.Q_axial < closed.Q_axial
        # 40mm 機・G=5e6 で −3% 程度
        assert -0.06 < gap.Q_axial / closed.Q_axial - 1.0 < -0.01

    def test_divergence_stays_clean(self):
        res = run(spec_gap(), 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        assert res.div_max < 1e-9


class TestGridConvergence:
    """1a/a02 の結論「誤差は最狭方向のセル数だけで決まる」を隙間に適用して検証."""

    def test_leak_converges_second_order_in_gap_resolution(self):
        """漏れ量 Q_leak が隙間セル数に対して 2 次で収束すること.

        隙間の解像度が直接効くのは Q_leak（Q_axial は下流方向の寄与が支配的で
        隙間解像度への感度が低い）。最細 n_gap=63 を基準にした相対差は実測で
        n_gap=5:1.7e-3 → 10:4.4e-4 → 20:1.2e-4 → 40:2.4e-5 と、
        セル数 2 倍ごとにおよそ 1/4 になる。
        """
        model = NewtonianViscosity(mu=MU)
        ref = run(spec_gap(ny=60, n_gap=63), 5.0e6, model).Q_leak
        errs = []
        for n in (5, 10, 20, 40):
            r = run(spec_gap(ny=60, n_gap=n), 5.0e6, model)
            errs.append(abs(r.Q_leak / ref - 1.0))
        order = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert all(1.6 < o < 2.6 for o in order), f"観測次数 {order}, 誤差 {errs}"

    def test_gap_resolution_beats_the_a02_criterion(self):
        """a02 基準（1% なら 20 セル）を大きく上回る精度が出ること.

        1a/a02 のボクセルメッシュ品質ベンチは「誤差は最狭方向のセル数だけで
        決まる。1% なら 20 セル、0.1% なら 63 セル」だった。ここでは n_gap=20 の
        Q_leak が最細格子と 0.02% 以内で一致する。差が出るのは格子の性質で、
        a02 は**等間隔ボクセル**で隙間を刻むしかなかったのに対し、ここは
        境界適合の等比格子で隙間の内側を直接刻めるため。
        a02 の基準は十分条件であって、ここでは保守的すぎるということ。
        """
        model = NewtonianViscosity(mu=MU)
        ref = run(spec_gap(ny=60, n_gap=63), 5.0e6, model)
        r20 = run(spec_gap(ny=60, n_gap=20), 5.0e6, model)
        assert r20.Q_leak == pytest.approx(ref.Q_leak, rel=1e-3)
        assert r20.Q_axial == pytest.approx(ref.Q_axial, rel=1e-3)

    def test_20_cells_in_gap_is_within_1_percent(self):
        """n_gap=20 の Q_axial が最細格子と 1% 以内（a02 基準の合格確認）."""
        model = NewtonianViscosity(mu=MU)
        q20 = run(spec_gap(ny=60, n_gap=20), 5.0e6, model).Q_axial
        q63 = run(spec_gap(ny=60, n_gap=63), 5.0e6, model).Q_axial
        assert q20 == pytest.approx(q63, rel=1e-2)
