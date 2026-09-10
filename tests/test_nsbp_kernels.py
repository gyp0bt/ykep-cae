"""nsbp.kernels（ゴースト付きパッチの残差）が nsb の残差と一致することの検査（petsc4py 不要）."""

from __future__ import annotations

import numpy as np
import pytest

from nsb.data import ConvectionSchemeType
from nsb.geo import make_case
from nsbp.kernels import dtau_patch, limiter_psi, residual_patch
from nsbp.problem import Patch, PatchCoefficients, make_discretization


def _random_state(disc, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = disc.n
    u = 0.1 * rng.standard_normal(n)
    v = 0.1 * rng.standard_normal(n)
    p = 10.0 * rng.standard_normal(n)
    return np.concatenate([u, v, p])


def _kernel_on_patch(disc, patch: Patch, x: np.ndarray, scheme, venkat_k: float, cs: float = 1.0):
    coef = PatchCoefficients(disc, patch)
    u, v, p = disc.split(x)
    gx, gy = slice(patch.gxs, patch.gxe), slice(patch.gys, patch.gye)
    r_u, r_v, r_p = residual_patch(
        np.ascontiguousarray(u[gx, gy]),
        np.ascontiguousarray(v[gx, gy]),
        np.ascontiguousarray(p[gx, gy]),
        coef.rho,
        coef.mu,
        coef.dx,
        coef.dy,
        *coef.args_bc,
        scheme is ConvectionSchemeType.FIRST_ORDER_UPWIND,
        venkat_k,
        cs,
        np.ones((patch.gnx, patch.gny)),
        np.ones((patch.gnx, patch.gny)),
        False,
        coef.wall_x,
        coef.wall_y,
    )
    ox, oy = patch.own
    return r_u[ox, oy], r_v[ox, oy], r_p[ox, oy]


@pytest.mark.parametrize("model", ["flat", "uturn"])
@pytest.mark.parametrize(
    "scheme", [ConvectionSchemeType.SECOND_ORDER_UPWIND, ConvectionSchemeType.FIRST_ORDER_UPWIND]
)
class TestResidualPatchAPI:
    def test_full_grid_matches_nsb_residual(self, model, scheme):
        inp = make_case(model, 1, u_in=1.0)
        disc = make_discretization(inp)
        x = _random_state(disc)
        ref = disc.residual_fast(x, scheme, 5.0)
        nx, ny = inp.nx, inp.ny
        patch = Patch(nx, ny, 0, nx, 0, ny, 0, nx, 0, ny)
        r_u, r_v, r_p = _kernel_on_patch(disc, patch, x, scheme, 5.0)
        got = np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())

    def test_stokes_flag_matches_nsb_convection_off(self, model, scheme):
        inp = make_case(model, 1, u_in=1.0)
        disc = make_discretization(inp)
        x = _random_state(disc, seed=3)
        ref = disc.residual_fast(x, scheme, 5.0, convection=False)
        nx, ny = inp.nx, inp.ny
        patch = Patch(nx, ny, 0, nx, 0, ny, 0, nx, 0, ny)
        r_u, r_v, r_p = _kernel_on_patch(disc, patch, x, scheme, 5.0, cs=0.0)
        got = np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())

    def test_split_patches_with_ghost_width_2_reproduce_full_residual(self, model, scheme):
        """2×2 に分割した各パッチ（ゴースト幅 2）の所有セル残差が全体計算と一致する（切れ目の扱いの検査）."""
        inp = make_case(model, 1, u_in=1.0)
        disc = make_discretization(inp)
        x = _random_state(disc, seed=7)
        ref_u, ref_v, ref_p = (a for a in np.split(disc.residual_fast(x, scheme, 5.0), 3))
        nx, ny = inp.nx, inp.ny
        ref_u, ref_v, ref_p = ref_u.reshape(nx, ny), ref_v.reshape(nx, ny), ref_p.reshape(nx, ny)
        cuts_x = [(0, 31), (31, nx)]
        cuts_y = [(0, 20), (20, ny)]
        for xs, xe in cuts_x:
            for ys, ye in cuts_y:
                patch = Patch(
                    nx,
                    ny,
                    xs,
                    xe,
                    ys,
                    ye,
                    max(xs - 2, 0),
                    min(xe + 2, nx),
                    max(ys - 2, 0),
                    min(ye + 2, ny),
                )
                r_u, r_v, r_p = _kernel_on_patch(disc, patch, x, scheme, 5.0)
                for got, ref in ((r_u, ref_u), (r_v, ref_v), (r_p, ref_p)):
                    assert np.allclose(
                        got, ref[xs:xe, ys:ye], rtol=1e-12, atol=1e-12 * np.abs(ref).max()
                    )


class TestDtauPatchAPI:
    def test_matches_nsb_compute_dtau(self):
        from nsb.core import NSBSettings
        from nsb.solver import compute_dtau

        rng = np.random.default_rng(1)
        u = rng.standard_normal((12, 9))
        v = rng.standard_normal((12, 9))
        u[0, 0] = v[0, 0] = 0.0  # 静止セル
        ref = compute_dtau(u, v, 0.01, 0.02, 3.0, NSBSettings(), 0.05)
        got = dtau_patch(u, v, 0.01, 0.02, 3.0, 0.05)
        assert np.allclose(got, ref, rtol=1e-14)


class TestLimiterFreezeAPI:
    def test_frozen_psi_from_same_state_reproduces_residual(self):
        """その場で計算した ψ を凍結して渡すと、生の Venkatakrishnan と同じ残差になる."""
        inp = make_case("uturn", 1, u_in=1.0)
        disc = make_discretization(inp)
        x = _random_state(disc, seed=11)
        nx, ny = inp.nx, inp.ny
        patch = Patch(nx, ny, 0, nx, 0, ny, 0, nx, 0, ny)
        coef = PatchCoefficients(disc, patch)
        u, v, p = (np.ascontiguousarray(a) for a in disc.split(x))
        pu, pv = limiter_psi(u, v, p, coef.args_bc, coef.dx, coef.dy, 5.0, coef.wall_x, coef.wall_y)
        assert (
            pu.min() >= 0.0 and pu.max() <= 1.0 and pu.min() < 1.0
        )  # 乱数場なので一部は制限される
        scheme = ConvectionSchemeType.SECOND_ORDER_UPWIND
        ref = disc.residual_fast(x, scheme, 5.0)
        r = residual_patch(
            u,
            v,
            p,
            coef.rho,
            coef.mu,
            coef.dx,
            coef.dy,
            *coef.args_bc,
            False,
            5.0,
            1.0,
            pu,
            pv,
            True,
            coef.wall_x,
            coef.wall_y,
        )
        got = np.concatenate([a.ravel() for a in r])
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


class TestSolidCellsPatchAPI:
    """[壁セル] h <= h_solid を固体にした場合もパッチカーネルが nsb の残差と一致する."""

    @staticmethod
    def _case():
        from nsb.core import NSBInput
        from nsb.geo import LX, LY, make_uturn_h, uturn_bc_preset

        nx, ny = 72, 48
        return NSBInput(
            nx=nx,
            ny=ny,
            lx=LX,
            ly=LY,
            h=make_uturn_h(nx, ny, h_channel=1e-3, h_blocked=1e-5),
            bc=uturn_bc_preset(ny, u_in=1.0),
            h_solid=1e-4,
        )

    @pytest.mark.parametrize(
        "scheme",
        [ConvectionSchemeType.SECOND_ORDER_UPWIND, ConvectionSchemeType.FIRST_ORDER_UPWIND],
    )
    def test_full_grid_matches_nsb_residual(self, scheme):
        inp = self._case()
        disc = make_discretization(inp)
        assert disc.has_solid
        x = disc.mask_state(_random_state(disc))
        ref = disc.residual_fast(x, scheme, 5.0)
        nx, ny = inp.nx, inp.ny
        patch = Patch(nx, ny, 0, nx, 0, ny, 0, nx, 0, ny)
        r_u, r_v, r_p = _kernel_on_patch(disc, patch, x, scheme, 5.0)
        # パッチカーネルは固体セルを落とさない（落とすのは nsbp.solver 側）ので流体セルだけ比べる
        got = np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])
        live = disc.live3
        assert np.allclose(
            got[live], ref[live], rtol=1e-12, atol=1e-12 * max(np.abs(ref).max(), 1e-300)
        )

    def test_split_patches_reproduce_full_residual(self):
        inp = self._case()
        disc = make_discretization(inp)
        x = disc.mask_state(_random_state(disc, seed=3))
        scheme = ConvectionSchemeType.SECOND_ORDER_UPWIND
        ref = disc.residual_fast(x, scheme, 5.0)
        nx, ny = inp.nx, inp.ny
        r_u = np.zeros((nx, ny))
        r_v = np.zeros((nx, ny))
        r_p = np.zeros((nx, ny))
        for xs, xe in ((0, nx // 2), (nx // 2, nx)):
            for ys, ye in ((0, ny // 2), (ny // 2, ny)):
                patch = Patch(
                    nx,
                    ny,
                    xs,
                    xe,
                    ys,
                    ye,
                    max(0, xs - 2),
                    min(nx, xe + 2),
                    max(0, ys - 2),
                    min(ny, ye + 2),
                )
                a, b, c = _kernel_on_patch(disc, patch, x, scheme, 5.0)
                r_u[xs:xe, ys:ye] = a
                r_v[xs:xe, ys:ye] = b
                r_p[xs:xe, ys:ye] = c
        got = np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])
        live = disc.live3
        assert np.allclose(
            got[live], ref[live], rtol=1e-10, atol=1e-10 * max(np.abs(ref).max(), 1e-300)
        )
