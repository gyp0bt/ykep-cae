"""粘度モデル Strategy とせん断速度評価のテスト."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.extruder.data import ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
    ViscosityModelStrategy,
    strain_rate,
)

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def parallel_plate(ny: int = 64, nx: int = 8):
    """フライト無しの平行平板（等間隔）."""
    return ScrewGeometryProcess().process(
        replace(_BASE, e=0.0, delta=0.0, nx_channel=nx, nx_land=1, ny_bulk=ny, n_gap=0)
    )


def channel_with_gap():
    return ScrewGeometryProcess().process(
        replace(
            _BASE,
            delta=1.0e-4,
            nx_channel=40,
            nx_land=8,
            ny_bulk=20,
            n_gap=8,
        )
    )


class TestViscosityAPI:
    """Protocol 適合と入力検証."""

    @pytest.mark.parametrize(
        "model",
        [
            NewtonianViscosity(mu=1000.0),
            PowerLawViscosity(K=2.0e4, n=0.4),
            CarreauViscosity(mu_0=1.0e5, mu_inf=10.0, lam=1.0, n=0.4),
        ],
    )
    def test_satisfies_protocol(self, model):
        assert isinstance(model, ViscosityModelStrategy)

    def test_shape_is_preserved(self):
        g = np.zeros((5, 7))
        for model in (
            NewtonianViscosity(mu=1.0),
            PowerLawViscosity(K=1.0, n=0.5),
            CarreauViscosity(mu_0=10.0, mu_inf=1.0, lam=1.0, n=0.5),
        ):
            assert model.viscosity(g).shape == (5, 7)

    @pytest.mark.parametrize(
        "kwargs",
        [{"K": 1.0, "n": 0.0}, {"K": -1.0, "n": 0.5}, {"K": 1.0, "n": 0.5, "gamma_min": 0.0}],
    )
    def test_power_law_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError, match="必要"):
            PowerLawViscosity(**kwargs)


class TestViscosityPhysics:
    """粘度モデルの物理的挙動."""

    def test_newtonian_is_constant(self):
        m = NewtonianViscosity(mu=1000.0)
        assert np.allclose(m.viscosity(np.array([0.0, 1.0, 1e4])), 1000.0)

    def test_power_law_known_value(self):
        """K γ̇^(n-1) を手計算と突き合わせる."""
        m = PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=1e-6)
        assert m.viscosity(np.array([100.0]))[0] == pytest.approx(2.0e4 * 100.0**-0.6)

    def test_power_law_is_shear_thinning(self):
        m = PowerLawViscosity(K=2.0e4, n=0.4)
        mu = m.viscosity(np.array([1.0, 10.0, 100.0, 1000.0]))
        assert all(a > b for a, b in zip(mu, mu[1:], strict=False))

    def test_power_law_n1_is_newtonian(self):
        m = PowerLawViscosity(K=1000.0, n=1.0)
        assert np.allclose(m.viscosity(np.array([0.1, 10.0, 1e5])), 1000.0)

    def test_power_law_clamped_at_zero_shear(self):
        """γ̇=0 で発散せず有限に留まること."""
        m = PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=1e-2, mu_max=1e8)
        mu0 = m.viscosity(np.array([0.0]))[0]
        assert np.isfinite(mu0)
        assert mu0 == pytest.approx(min(2.0e4 * 1e-2**-0.6, 1e8))

    def test_power_law_mu_max_caps(self):
        m = PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=1e-12, mu_max=1e5)
        assert m.viscosity(np.array([0.0]))[0] == pytest.approx(1e5)

    def test_carreau_limits(self):
        m = CarreauViscosity(mu_0=1.0e5, mu_inf=10.0, lam=1.0, n=0.4)
        assert m.viscosity(np.array([1e-8]))[0] == pytest.approx(1.0e5, rel=1e-6)
        assert m.viscosity(np.array([1e12]))[0] == pytest.approx(10.0, rel=1e-3)

    def test_carreau_n1_is_newtonian(self):
        m = CarreauViscosity(mu_0=1000.0, mu_inf=1000.0, lam=1.0, n=1.0)
        assert np.allclose(m.viscosity(np.array([0.1, 10.0, 1e5])), 1000.0)


class TestStrainRate:
    """せん断速度の離散評価."""

    def test_pure_drag_shear_rate_is_V_over_H(self):
        """純引きずり（両成分とも線形）の γ̇ が厳密に V/H になること.

        u = u_barrel·y/H, w = w_barrel·y/H は両方ともバレル BC と整合する。
        このとき γ̇² = (u_barrel² + w_barrel²)/H² = V²/H² で、
        **リード角 φ が消える**。バレルは速さ V で滑っているのだから当然だが、
        u = -V sinφ / w = +V cosφ の符号と Green-Gauss の境界処理が両方
        正しくないとこの相殺は起きない。境界セルも含めて厳密に成立する。
        """
        grid = parallel_plate()
        s = grid.spec
        eta = grid.yc[None, :] / s.H
        u = np.broadcast_to(s.u_barrel * eta, (grid.nx, grid.ny)).copy()
        w = np.broadcast_to(s.w_barrel * eta, (grid.nx, grid.ny)).copy()
        v = np.zeros_like(u)
        gd = strain_rate(u, v, w, grid)
        expect = s.V / s.H
        assert np.max(np.abs(gd - expect)) < 1e-9 * expect

    def test_zero_field_with_static_barrel_gives_zero(self):
        """バレルが止まっている（N=0）なら零速度場の γ̇ は 0."""
        grid = ScrewGeometryProcess().process(
            replace(
                _BASE,
                e=0.0,
                delta=0.0,
                N=0.0,
                nx_channel=8,
                nx_land=1,
                ny_bulk=64,
                n_gap=0,
            )
        )
        z = np.zeros((grid.nx, grid.ny))
        assert np.max(np.abs(strain_rate(z, z, z, grid))) < 1e-12

    def test_zero_field_with_moving_barrel_is_not_zero(self):
        """逆に、バレルが動いていれば零速度場は BC と不整合で γ̇ > 0 になること.

        境界値が本当に効いていることの確認（効いていなければ 0 になってしまう）。
        """
        grid = parallel_plate()
        z = np.zeros((grid.nx, grid.ny))
        gd = strain_rate(z, z, z, grid)
        assert gd[0, -1] > 0.0
        assert np.max(np.abs(gd[:, :-1])) < 1e-12

    def test_is_nonnegative_and_finite(self):
        grid = channel_with_gap()
        rng = np.random.default_rng(0)
        u = rng.normal(size=(grid.nx, grid.ny))
        v = rng.normal(size=(grid.nx, grid.ny))
        w = rng.normal(size=(grid.nx, grid.ny))
        gd = strain_rate(u, v, w, grid)
        assert np.all(gd >= 0.0)
        assert np.all(np.isfinite(gd))

    def test_solid_cells_are_zero(self):
        grid = channel_with_gap()
        rng = np.random.default_rng(1)
        u = rng.normal(size=(grid.nx, grid.ny))
        gd = strain_rate(u, u, u, grid)
        assert np.all(gd[grid.solid] == 0.0)
