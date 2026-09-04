"""CrossChannelStokesProcess のテスト。ゲート G2b を含む."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.cross_channel import (
    CrossChannelStokesProcess,
    StokesLayout,
    build_stokes_system,
)
from xkep_cae_fluid.extruder.data import CrossChannelInput, ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

MU = 1000.0

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def parallel_plate(ny: int = 40, nx: int = 8):
    """フライト無しの平行平板。x も y も等間隔."""
    return ScrewGeometryProcess().process(
        replace(_BASE, e=0.0, delta=0.0, nx_channel=nx, nx_land=1, ny_bulk=ny, n_gap=0)
    )


def closed_channel(ny: int = 32, nx_channel: int = 80):
    """delta=0 の閉チャネル（フライトが y=H まで届く）."""
    half = 0.5 * (_BASE.W_t - _BASE.e)
    n_half = max(2, nx_channel // 2)
    return ScrewGeometryProcess().process(
        replace(
            _BASE,
            nx_channel=nx_channel,
            nx_land=max(1, round(_BASE.e / (half / n_half))),
            ny_bulk=ny,
            n_gap=0,
        )
    )


def channel_with_gap(ny: int = 32, n_gap: int = 16, nx_channel: int = 120, nx_land: int = 32):
    return ScrewGeometryProcess().process(
        replace(
            _BASE,
            delta=1.0e-4,
            nx_channel=nx_channel,
            nx_land=nx_land,
            ny_bulk=ny,
            n_gap=n_gap,
        )
    )


def solve(grid, G: float, mu: float = MU):
    return CrossChannelStokesProcess().process(
        CrossChannelInput(grid=grid, mu=np.full((grid.nx, grid.ny), mu), G=G)
    )


@binds_to(CrossChannelStokesProcess)
class TestCrossChannelAPI:
    """API・契約のテスト."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "CrossChannelStokesProcess" in ProcessRegistry.default()

    def test_solid_cells_are_zero(self):
        grid = channel_with_gap()
        res = solve(grid, 1.0e6)
        assert np.all(res.u[grid.solid] == 0.0)
        assert np.all(res.v[grid.solid] == 0.0)

    def test_rejects_shape_mismatch(self):
        grid = parallel_plate(ny=8, nx=4)
        with pytest.raises(ValueError, match="形状が格子と不一致"):
            CrossChannelStokesProcess().process(
                CrossChannelInput(grid=grid, mu=np.ones((3, 3)), G=0.0)
            )

    def test_input_is_not_mutated(self):
        """C9: process() が入力の numpy 配列を変更しないこと."""
        grid = parallel_plate(ny=16, nx=4)
        mu = np.full((grid.nx, grid.ny), MU)
        before = mu.copy()
        CrossChannelStokesProcess().execute(CrossChannelInput(grid=grid, mu=mu, G=1.0e6))
        assert np.array_equal(mu, before)

    def test_layout_counts(self):
        """フライト無しなら u は全 x 面、v は内部 y 面だけが自由度になること."""
        grid = parallel_plate(ny=10, nx=6)
        lay = StokesLayout(grid)
        assert lay.nu == grid.nx * grid.ny
        assert lay.nv == grid.nx * (grid.ny - 1)
        assert lay.npr == grid.nx * grid.ny


class TestStokesOperator:
    """離散作用素そのものの性質."""

    def test_viscous_block_is_symmetric_without_flight(self):
        """フライト無しなら粘性ブロックが対称になること.

        完全応力 Stokes 作用素は Dirichlet 条件下で対称。MAC 離散が
        「せん断の微分距離 = CV 寸法」を満たす限り対称性が保たれる。
        u-v 結合の係数が両方向で一致しないと崩れるので、せん断項の
        組み立てミスを一発で捕まえる。

        （フライトがあると壁位置を正しく置くために半セル距離を使うので
        対称性は意図的に崩す。cross_channel.py の docstring 参照。）
        """
        grid = parallel_plate(ny=12, nx=6)
        mu = np.full((grid.nx, grid.ny), MU)
        A, _rhs, lay, _alpha = build_stokes_system(grid, mu, 1.0e6)
        n = lay.nu + lay.nv
        vel = A[:n, :n].tocsr()
        asym = abs(vel - vel.T)
        assert asym.max() < 1e-9 * abs(vel).max()

    def test_gradient_is_minus_divergence_transpose(self):
        """勾配ブロックが発散ブロックの厳密な転置 × (−1) であること.

        これが成り立たないと離散的な発散ゼロが機械精度で出ない。
        ピン留めした 1 行だけは除いて比較する。
        """
        grid = parallel_plate(ny=10, nx=6)
        mu = np.full((grid.nx, grid.ny), MU)
        A, _rhs, lay, _alpha = build_stokes_system(grid, mu, 1.0e6)
        n = lay.nu + lay.nv
        div = A[lay.p_offset + 1 :, :n].tocsr()
        grad = A[:n, lay.p_offset + 1 :].tocsr()
        assert abs(grad + div.T).max() < 1e-9 * abs(div).max()


class TestCrossChannelPhysics:
    """断面内流れの物理."""

    def test_pure_drag_is_linear_exact(self):
        """体積力なし（G=0）の平行平板で u(y) = u_barrel·(y/H) を機械精度で再現.

        1 次関数なので離散化が厳密に表せる。バレルの符号と、せん断ステンシルの
        境界処理をここで固定する。
        """
        grid = parallel_plate(ny=40)
        res = solve(grid, 0.0)
        expect = grid.spec.u_barrel * grid.yc / grid.spec.H
        assert np.max(np.abs(res.u[0, :] - expect)) < 1e-12 * abs(grid.spec.u_barrel)

    def test_body_force_pushes_in_minus_x_when_pumping(self):
        """G>0（背圧）で f_x = −G·cotφ < 0 なので横断流量が減ること."""
        grid = parallel_plate(ny=40)
        flux0 = float(np.sum(solve(grid, 0.0).u_face[0, :] * grid.dy))
        flux1 = float(np.sum(solve(grid, 5.0e6).u_face[0, :] * grid.dy))
        assert flux1 < flux0

    def test_g2b_closed_channel_profile_converges(self):
        """G2b: 正味横断流量ゼロの 1D 解 u(y) = U(3η²−2η) に 2 次収束すること.

        β = 6μU/H² を与えると正味流量ゼロの条件がちょうど満たされる。
        セル中心 FV の半セル Dirichlet は 2 次多項式には厳密でないので
        （下流方向 Poiseuille と同じ理由）、収束次数で判定する。
        """
        s0 = parallel_plate(ny=8).spec
        U = s0.u_barrel
        beta_needed = 6.0 * MU * U / s0.H**2
        G = beta_needed * math.tan(s0.phi)

        errs = []
        for ny in (20, 40, 80, 160):
            grid = parallel_plate(ny=ny)
            res = solve(grid, G)
            eta = grid.yc / grid.spec.H
            expect = U * (3.0 * eta**2 - 2.0 * eta)
            errs.append(float(np.max(np.abs(res.u[0, :] - expect)) / abs(U)))
        order = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert all(1.95 < o < 2.05 for o in order), f"観測次数 {order}, 誤差 {errs}"
        assert errs[-1] < 5e-5

    def test_g2b_net_cross_flux_converges_to_zero(self):
        """G2b の条件で正味横断流量が 2 次で 0 に収束すること.

        β = 6μU/H² は連続の解に対する条件なので、離散解の正味流束は
        厳密には 0 にならず O(h²) で消える。閉チャネル（フライトで塞がれた
        場合）の機械精度ゼロとは意味が違うことに注意。
        """
        s0 = parallel_plate(ny=8).spec
        U = s0.u_barrel
        G = (6.0 * MU * U / s0.H**2) * math.tan(s0.phi)
        fluxes = []
        for ny in (20, 40, 80, 160):
            grid = parallel_plate(ny=ny)
            res = solve(grid, G)
            flux = float(np.sum(res.u_face[0, :] * grid.dy))
            fluxes.append(abs(flux) / (abs(U) * grid.spec.H))
        order = [np.log2(fluxes[i] / fluxes[i + 1]) for i in range(len(fluxes) - 1)]
        assert all(1.95 < o < 2.05 for o in order), f"観測次数 {order}, 流束 {fluxes}"
        assert fluxes[-1] < 5e-5

    def test_zero_net_cross_flux_in_closed_channel(self):
        """閉チャネル（フライトが y=H まで届く）では正味横断流量が機械精度でゼロ.

        両端をフライトで塞がれているので、どの断面でも通過流量は 0 でなければ
        ならない。離散的な発散ゼロが厳密に成り立っていることの帰結。
        """
        grid = closed_channel()
        res = solve(grid, 5.0e6)
        for i in (0, grid.nx // 4, grid.nx - 1):
            flux = float(np.sum(res.u_face[i, :] * grid.dy))
            assert abs(flux) < 1e-13 * abs(grid.spec.u_barrel) * grid.spec.H

    def test_discretely_divergence_free(self):
        grid = channel_with_gap()
        res = solve(grid, 5.0e6)
        assert res.div_max < 1e-10, f"div_max={res.div_max:.2e}"

    def test_streamfunction_is_consistent_with_faces(self):
        """ψ の y 差分が u 面流束、x 差分が −v 面流束と一致すること."""
        grid = channel_with_gap()
        res = solve(grid, 5.0e6)
        dpsi_y = res.psi[:-1, 1:] - res.psi[:-1, :-1]
        assert np.allclose(dpsi_y, res.u_face * grid.dy[None, :], rtol=1e-10, atol=1e-18)
        dpsi_x = res.psi[1:, :] - res.psi[:-1, :]
        assert np.allclose(dpsi_x, -res.v_face * grid.dx[:, None], rtol=1e-10, atol=1e-18)

    def test_streamfunction_is_single_valued(self):
        """x を 1 周しても ψ が戻ること（質量保存の帰結）."""
        grid = channel_with_gap()
        res = solve(grid, 5.0e6)
        assert res.psi_periodicity < 1e-10, f"{res.psi_periodicity:.2e}"

    def test_leakage_is_backward_when_pumping(self):
        """G>0 のとき隙間を通る正味横断流量が −x（上流へ戻る）向きであること.

        引きずり成分 u_barrel<0 と圧力成分 f_x<0 が同じ向きなので、
        漏れは必ず後戻りになる（押出量を減らす）。
        """
        grid = channel_with_gap()
        res = solve(grid, 5.0e6)
        i_land = grid.nx // 2
        flux = float(np.sum(res.u_face[i_land, :] * grid.dy))
        assert flux < 0.0

    def test_leakage_grows_with_back_pressure(self):
        grid = channel_with_gap()
        i_land = grid.nx // 2

        def leak(G):
            r = solve(grid, G)
            return float(np.sum(r.u_face[i_land, :] * grid.dy))

        assert leak(1.0e7) < leak(5.0e6) < leak(0.0)

    def test_recirculation_exists_in_closed_channel(self):
        """閉チャネルの断面内に循環（u の符号反転）があること — 混練の主機構."""
        grid = closed_channel()
        res = solve(grid, 0.0)
        col = res.u[0, :]
        assert col.min() < 0.0 < col.max()

    def test_two_layer_viscosity_series_resistance(self):
        """y 方向 2 層粘度の引きずり流れが直列抵抗則に従うこと（節点粘度の検証）."""
        ny = 64
        grid = parallel_plate(ny=ny)
        mu = np.where(grid.yc[None, :] < grid.spec.H / 2.0, 500.0, 2000.0)
        mu = np.broadcast_to(mu, (grid.nx, grid.ny)).copy()
        res = CrossChannelStokesProcess().process(CrossChannelInput(grid=grid, mu=mu, G=0.0))
        h_half = grid.spec.H / 2.0
        tau = grid.spec.u_barrel / (h_half / 500.0 + h_half / 2000.0)
        u_int = tau * h_half / 500.0
        y = grid.yc
        expect = np.where(y < h_half, tau * y / 500.0, u_int + tau * (y - h_half) / 2000.0)
        assert np.max(np.abs(res.u[0, :] - expect)) < 1e-10 * abs(grid.spec.u_barrel)
