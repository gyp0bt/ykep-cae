"""リミター凍結（`NSBSettings.limiter_freeze_rel`）と隙間の摩擦則（`NSBInput.friction_re_crit`）の検査.

- numba カーネルと numpy 経路が凍結 ψ・摩擦則込みで一致する
- 凍結 ψ を「現在の場の ψ」にすると残差は凍結前と同じ（連続性）
- 摩擦則の倍率は層流域で 1、Re_c で 2^(1/m) に近く、Blasius の漸近 (Re/Re_c)^0.75 に乗る
- uturn で凍結を入れても収束し、解凍後の真の残差が報告される
"""

from __future__ import annotations

import numpy as np

from nsb import NSBInput, NSBSettings, make_case, solve_steady
from nsb.assembly import BrinkmanDiscretization
from nsb.data import ConvectionSchemeType

SOU = ConvectionSchemeType.SECOND_ORDER_UPWIND


def _with(inp: NSBInput, **kw) -> NSBInput:
    return NSBInput(**{**inp.__dict__, **kw})


class TestLimiterFreezePhysics:
    def test_frozen_psi_matches_numpy_and_is_continuous(self):
        inp = make_case("uturn", 1, 1.0)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        rng = np.random.default_rng(1)
        x = rng.standard_normal(3 * disc.n)
        psi = disc.limiter(x, 5.0)
        assert psi[0].shape == (disc.nx, disc.ny) and (psi[0] <= 1.0).all() and (psi[0] >= 0).all()
        r_free = disc.residual_fast(x, SOU, 5.0)
        r_frozen = disc.residual_fast(x, SOU, 5.0, psi=psi)
        assert np.abs(r_frozen - r_free).max() <= 1e-12 * np.abs(r_free).max()
        # 別の場では凍結の有無で残差が変わる（ψ が本当に使われている）
        x2 = x + 0.3 * rng.standard_normal(3 * disc.n)
        r2_free = disc.residual_fast(x2, SOU, 5.0)
        r2_frozen = disc.residual_fast(x2, SOU, 5.0, psi=psi)
        assert np.abs(r2_frozen - r2_free).max() > 1e-6 * np.abs(r2_free).max()
        # numpy 経路と一致
        st = disc.compute_state(x2, SOU, 5.0, psi=psi)
        r2_np = disc.residual_from_state(x2, st)
        assert np.abs(r2_frozen - r2_np).max() <= 1e-13 * np.abs(r2_np).max()


class TestLimiterFreezeConvergence:
    def test_uturn_converges_with_freeze_and_reports_unfrozen_residual(self):
        s = NSBSettings(limiter_freeze_rel=1e-3, newton_max_iter=60)
        inp = make_case("uturn", 1, 1.0, settings=s)
        res = solve_steady(inp, log=None)
        assert res.converged, res.failure_reason
        assert res.limiter_frozen_at >= 0
        # 凍結問題の解の真の残差は 1e-4 程度（凍結時点の ψ は解の ψ と違う）。解凍残差が報告されること
        assert 0.0 <= res.residual_unfrozen / res.residual_ref < 1e-3
        base = solve_steady(_with(inp, settings=NSBSettings(newton_max_iter=60)), log=None)
        assert base.converged
        # 凍結解と真の解の差は速度で 1e-3 程度（uturn r1 U=1 で 6.4e-4）
        assert np.abs(res.u - base.u).max() < 3e-3 * np.abs(base.u).max()

    def test_refreeze_picard_reduces_unfrozen_residual(self):
        s = NSBSettings(limiter_freeze_rel=1e-3, limiter_refreeze_max=3, newton_max_iter=60)
        res = solve_steady(make_case("uturn", 1, 1.0, settings=s), log=None)
        s0 = NSBSettings(limiter_freeze_rel=1e-3, newton_max_iter=60)
        res0 = solve_steady(make_case("uturn", 1, 1.0, settings=s0), log=None)
        assert res.converged and res0.converged
        assert res.residual_unfrozen < res0.residual_unfrozen


class TestFrictionLawPhysics:
    def test_factor_limits(self):
        inp = _with(make_case("flat", 1, 1.0), friction_re_crit=2040.0)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        h = float(disc.thickness.mean())
        u_c = 2040.0 * disc.mu / (disc.rho * 2.0 * h)
        zeros = np.zeros((disc.nx, disc.ny))
        f_lam = disc.drag_factor(np.full_like(zeros, 0.01 * u_c), zeros)
        f_crit = disc.drag_factor(np.full_like(zeros, u_c), zeros)
        f_turb = disc.drag_factor(np.full_like(zeros, 100.0 * u_c), zeros)
        assert np.allclose(f_lam, 1.0, atol=1e-6)
        assert np.allclose(f_crit, 2.0 ** (1.0 / 4.0), rtol=1e-12)
        assert np.allclose(f_turb, 100.0**0.75, rtol=1e-3)

    def test_fast_matches_numpy_with_friction(self):
        inp = _with(make_case("uturn", 1, 1.0), friction_re_crit=200.0)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        rng = np.random.default_rng(2)
        x = rng.standard_normal(3 * disc.n)
        st = disc.compute_state(x, SOU, 5.0)
        assert st.drag_fac.max() > 1.5  # 摩擦則が効く速度域にある
        ref = disc.residual_from_state(x, st)
        fast = disc.residual_fast(x, SOU, 5.0)
        assert np.abs(fast - ref).max() <= 1e-13 * np.abs(ref).max()
        # J1 も倍率込みの抗力を持つ（u 列の対角に drag·V·f が入る）。摩擦則は a_P → RC 係数 →
        # 質量流束 → 対流項まで変えるので、対流を落とした J1 で抗力の差だけを見る
        J = disc.jacobian_first_order(st, x=x, convection=False).tocsr()
        J0 = BrinkmanDiscretization(make_case("uturn", 1, 1.0).to_flow_input())
        st0 = J0.compute_state(x, SOU, 5.0)
        J_lam = J0.jacobian_first_order(st0, x=x, convection=False).tocsr()
        d = (J.diagonal() - J_lam.diagonal())[: disc.n]
        expect = (disc.drag * disc.vol * (st.drag_fac - 1.0)).ravel()
        # 閉塞セルは drag·V が大きく差分で桁落ちするので絶対許容は対角の最大値で規格化する
        assert np.allclose(d, expect, rtol=1e-6, atol=1e-12 * np.abs(J.diagonal()).max())

    def test_uturn_converges_with_friction_and_needs_more_pressure(self):
        base = solve_steady(make_case("uturn", 1, 1.0), log=None)
        inp = _with(make_case("uturn", 1, 1.0), friction_re_crit=50.0)
        res = solve_steady(inp, log=None)
        assert base.converged and res.converged, (base.failure_reason, res.failure_reason)
        assert res.p.max() - res.p.min() > 1.2 * (base.p.max() - base.p.min())
        assert abs(res.mass_in - res.mass_out) < 1e-5 * abs(res.mass_in)


class TestUnsteadyPhysics:
    def test_uturn_backward_euler_reaches_steady_state(self):
        from nsb.unsteady import solve_unsteady

        inp = make_case("uturn", 1, 1.0, settings=NSBSettings(newton_tol=1e-5))
        base = solve_steady(inp, log=None)
        res = solve_unsteady(inp, dt=0.05, n_steps=200, log=None, newton_max=6, save_every=5)
        assert res.failure_reason == ""
        assert res.reached_steady, res.steady_residual[-5:]
        assert len(res.snapshots) >= 2
        # 定常残差は単調に近く落ち、到達した場は定常 Newton の解と一致する
        assert res.steady_residual[-1] < 1e-2 * res.steady_residual[0]
        assert np.abs(res.u - base.u).max() < 1e-2 * np.abs(base.u).max()
        assert abs(res.probes["mass_out"][-1] - base.mass_out) < 1e-3 * abs(base.mass_out)
