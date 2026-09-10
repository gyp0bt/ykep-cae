"""非定常の Rhie–Chow 係数に時間微分の対角を入れる切替（`NSBSettings.rc_with_pseudo_time`）の検査.

Rhie–Chow の面速度補正は d_f = V/a_P に比例する。a_P は運動量方程式の対角なので、後退 Euler
なら本来 ρV/Δt を含む（OpenFOAM の 1/A と同じ形）。含めないと**低速域（a_P ≪ ρV/Δt）で d_f が
過大**になり、圧力平滑化が Δt を小さくしても減らない。剥離域はまさに a_P が小さい場所なので、
そこだけ人工散逸が乗る。

- 係数の関係 d_f(入り) = d_f(無し)·a_P/(a_P + ρV/Δt) が厳密に成り立つ
- Δt → ∞ で切替は無効化される（rc_diag → 0）
- 実用的な Δt では場が測れるだけ変わる（＝ 効いている）
"""

from __future__ import annotations

import numpy as np

from nsb import NSBInput, NSBSettings, make_case
from nsb.assembly import BrinkmanDiscretization
from nsb.data import ConvectionSchemeType
from nsb.unsteady import solve_unsteady

SOU = ConvectionSchemeType.SECOND_ORDER_UPWIND


def _with(inp: NSBInput, **kw) -> NSBInput:
    return NSBInput(**{**inp.__dict__, **kw})


class TestRhieChowTransientDiagonal:
    def test_coefficient_shrinks_by_exactly_the_diagonal_ratio(self):
        """d_f は a_P/(a_P + ρV/Δt) 倍になる（面は両側セルの平均なので係数も平均で縮む）."""
        inp = make_case("uturn", 1, 1.0)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        rng = np.random.default_rng(0)
        x = disc.mask_state(0.1 * rng.standard_normal(3 * disc.n))
        dt = 1.0e-3
        tau = (inp.rho * disc.vol / dt) * np.ones((disc.nx, disc.ny))

        free = disc.compute_state(x, SOU, 5.0)
        with_dt = disc.compute_state(x, SOU, 5.0, tau)
        d_free = disc.vol / free.a_p
        d_with = disc.vol / (free.a_p + tau)
        for got, want in ((with_dt.dfx[1:-1], 0.5 * (d_with[:-1] + d_with[1:])),):
            m = np.abs(want) > 0
            assert np.allclose(got[m], want[m], rtol=1e-12)
        # 縮む向きと大きさ: どこでも 1 以下で、低速域では大きく縮む
        ratio = d_with / d_free
        assert (ratio <= 1.0 + 1e-12).all()
        assert ratio.min() < 0.5, f"時間微分の対角が効いていない（最小比 {ratio.min():.3f}）"

    def test_large_time_step_makes_the_switch_a_no_op(self):
        """Δt → ∞ で ρV/Δt → 0 なので、面係数が切替の有無で一致する（時間発展は定常に戻る）."""
        inp = make_case("uturn", 1, 1.0)
        disc = BrinkmanDiscretization(inp.to_flow_input())
        rng = np.random.default_rng(0)
        x = disc.mask_state(0.1 * rng.standard_normal(3 * disc.n))
        tau = (inp.rho * disc.vol / 1.0e12) * np.ones((disc.nx, disc.ny))
        free = disc.compute_state(x, SOU, 5.0)
        huge = disc.compute_state(x, SOU, 5.0, tau)
        assert np.abs(huge.dfx - free.dfx).max() <= 1e-9 * np.abs(free.dfx).max()
        assert np.abs(huge.fx - free.fx).max() <= 1e-9 * np.abs(free.fx).max()

    def test_practical_time_step_changes_the_field(self):
        """実用的な Δt では場が測れるだけ変わる（無害な no-op ではない）."""
        inp = make_case("uturn", 1, 1.0)
        kw = {"newton_max_iter": 30, "linear_solver": "jfnk"}
        off = solve_unsteady(
            _with(inp, settings=NSBSettings(**kw)), dt=1.0e-3, n_steps=20, log=None
        )
        on = solve_unsteady(
            _with(inp, settings=NSBSettings(rc_with_pseudo_time=True, **kw)),
            dt=1.0e-3,
            n_steps=20,
            log=None,
        )
        rel = np.linalg.norm(np.hypot(on.u - off.u, on.v - off.v)) / np.linalg.norm(
            np.hypot(off.u, off.v)
        )
        assert rel > 1.0e-4, f"切替が場に効いていない（相対差 {rel:.2e}）"
