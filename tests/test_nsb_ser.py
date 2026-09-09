"""SER 制御の安全規則（status-41）: 線形解が実質失敗したステップは棄却され CFL が下がる."""

from __future__ import annotations

import numpy as np
import pytest

import nsb.solver as solver_mod
from nsb import NSBSettings, make_case, solve_steady


class TestLinearFailureRejectionAPI:
    def test_failed_linear_solve_is_rejected_and_cfl_shrinks(self, monkeypatch):
        """最初の 1 回だけ線形解を「失敗」（残差比 1.0）に差し替え、その修正量が捨てられることを見る."""
        real = solver_mod.solve_linear
        calls = {"n": 0}

        def fake(disc, st, x, r_tau, diag_aug, resid_fn, s, pc, force_refresh=False, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return np.full(x.size, 1e6), 200, False, 1.0  # ごみ修正量
            return real(disc, st, x, r_tau, diag_aug, resid_fn, s, pc, force_refresh)

        monkeypatch.setattr(solver_mod, "solve_linear", fake)
        lines: list[str] = []
        s = NSBSettings(newton_max_iter=150, reject_lin_ratio=0.3, ser_shrink=0.1)
        res = solve_steady(make_case("flat", 1, 1.0, settings=s), log=lines.append)
        rejected = [ln for ln in lines if "rejected" in ln]
        assert len(rejected) == 1
        assert "|b-Ax|/|b|=1.00e+00" in rejected[0]
        # 棄却されたステップは残差履歴に入らず、CFL は 0.25 → 0.025 に下がってから再出発する
        assert res.cfl_history[0] == pytest.approx(0.025)
        assert res.converged
        assert np.isfinite(res.u).all()

    def test_rejection_disabled_accepts_step(self, monkeypatch):
        real = solver_mod.solve_linear
        calls = {"n": 0}

        def fake(disc, st, x, r_tau, diag_aug, resid_fn, s, pc, force_refresh=False, **kw):
            calls["n"] += 1
            d, n, ok, ratio = real(disc, st, x, r_tau, diag_aug, resid_fn, s, pc, force_refresh, **kw)
            return d, n, ok, (1.0 if calls["n"] == 1 else ratio)

        monkeypatch.setattr(solver_mod, "solve_linear", fake)
        lines: list[str] = []
        s = NSBSettings(newton_max_iter=60, reject_lin_ratio=0.0)
        solve_steady(make_case("flat", 1, 1.0, settings=s), log=lines.append)
        assert not any("rejected" in ln for ln in lines)
