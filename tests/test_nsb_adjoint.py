"""nsb.adjoint: 彩色 FD ヤコビアンと陰関数定理による設計感度のテスト."""

from __future__ import annotations

import numpy as np

from nsb import (
    BC,
    ImplicitSolve,
    NSBInput,
    NSBSettings,
    colored_fd_jacobian,
    source_mean_pressure_objective,
)
from nsb.geo import LX, LY
from xkep_cae_fluid.brinkman_flow import ConvectionSchemeType, disk_mask, smooth_disk
from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization

MDOT, COND = 0.02, 1e-5


def _bc(theta: np.ndarray, eps: float) -> BC:
    """θ = (cx, cy, r): 吸出マニホールドの位置と径（滑らかな窓）。注入は固定."""
    cx, cy, r = theta
    return BC(
        patches=(
            BC.interior_source(disk_mask(0.15, 0.2, 0.05), MDOT),
            BC.interior_pressure_sink(None, COND, p=0.0, weight=smooth_disk(cx, cy, r, eps)),
        )
    )


def _build(nx: int, ny: int, tol: float):
    eps = LX / nx

    def build(theta: np.ndarray) -> NSBInput:
        h = np.full((nx, ny), 1e-3)
        return NSBInput(
            nx=nx,
            ny=ny,
            lx=LX,
            ly=LY,
            h=h,
            bc=_bc(theta, eps),
            settings=NSBSettings(
                velocity_floor=0.02,
                pseudo_time_in_residual=False,
                alpha_u=1.0,
                init_field="stokes",
                newton_tol=tol,
                newton_max_iter=120,
            ),
        )

    return build


class TestColoredJacobian:
    def test_matches_dense_fd_for_sou_with_manifolds(self):
        """2 次風上 + リミター + RC + マニホールドの残差で、彩色 FD が密 FD と一致する（半径 2 で十分）."""
        nx, ny = 12, 8
        build = _build(nx, ny, 1e-6)
        inp = build(np.array([0.5, 0.2, 0.06]))
        disc = BrinkmanDiscretization(inp.to_flow_input())
        n = disc.n
        rng = np.random.default_rng(5)
        x = np.concatenate([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.normal(0, 5.0, n)])
        sch = ConvectionSchemeType.SECOND_ORDER_UPWIND

        def resid(xx):
            return disc.residual(xx, sch, 5.0)

        J = colored_fd_jacobian(disc, resid, x, radius=2).toarray()
        Jd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jd[:, k] = (resid(x + e) - resid(x - e)) / (2 * hk)
        scale = np.abs(Jd).max() + 1e-12
        assert np.abs(J - Jd).max() / scale < 1e-8


class TestImplicitSensitivity:
    def test_gradient_matches_finite_difference_of_full_solve(self):
        """dΔp/d(cx, cy, r) を随伴で求め、全体を解き直す中心差分と比較する（36×24）."""
        build = _build(36, 24, 1e-10)
        prob = ImplicitSolve(build, jac_radius=2)
        theta0 = np.array([0.50, 0.20, 0.06])
        res, x = prob.forward(theta0)
        assert res.converged, res.failure_reason
        obj = source_mean_pressure_objective()
        f0, g = prob.gradient(theta0, x, obj)
        assert f0 > 0.0

        g_fd = np.zeros(3)
        for k, h in enumerate([2e-3, 2e-3, 1e-3]):
            e = np.zeros(3)
            e[k] = h
            rp, xp = prob.forward(theta0 + e, init=res)
            rm, xm = prob.forward(theta0 - e, init=res)
            assert rp.converged and rm.converged
            g_fd[k] = (obj.value(xp, build(theta0 + e)) - obj.value(xm, build(theta0 - e))) / (
                2 * h
            )
        # 吸出を注入から遠ざける（cx 増）と圧損は増え、径を大きくすると減る
        assert g[0] > 0.0 and g[2] < 0.0
        assert np.allclose(g, g_fd, rtol=2e-2, atol=1e-3 * np.abs(g_fd).max())
