"""DownChannelFlowProcess のテスト。ゲート G1 / G2 を含む.

G1/G2 を通らなければ先へ進まない（設計文書 §3）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import DownChannelInput, ScrewSpec
from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)

MU = 1000.0
"""ニュートン粘度 [Pa·s]（設計文書 §6）."""

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def closed_channel(ny: int, nx_channel: int):
    """delta=0 の閉チャネル（幅 W、両側 no-slip）。G1/G2 の解析解と同じ問題.

    ランドのセル幅をチャネル側と揃えることで x 方向も等間隔になり、
    ny と nx_channel を同時に倍にすると自己相似な格子細分になる
    （観測次数を測るために必要）。
    """
    half = 0.5 * (_BASE.W_t - _BASE.e)
    n_half = max(2, nx_channel // 2)
    nx_land = max(1, round(_BASE.e / (half / n_half)))
    spec = replace(_BASE, nx_channel=nx_channel, nx_land=nx_land, ny_bulk=ny, n_gap=0)
    return ScrewGeometryProcess().process(spec)


def parallel_plate(ny: int, *, N: float = 100.0 / 60.0):
    """フライト無しの平行平板（1D 厳密解の検証用）。y 等間隔."""
    spec = replace(_BASE, e=0.0, delta=0.0, N=N, nx_channel=8, nx_land=1, ny_bulk=ny, n_gap=0)
    return ScrewGeometryProcess().process(spec)


def solve(grid, G: float, mu: float = MU):
    mu_field = np.full((grid.nx, grid.ny), mu)
    return DownChannelFlowProcess().process(DownChannelInput(grid=grid, mu=mu_field, G=G))


@binds_to(DownChannelFlowProcess)
class TestDownChannelAPI:
    """API・契約のテスト."""

    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "DownChannelFlowProcess" in ProcessRegistry.default()

    def test_solid_cells_are_zero(self):
        grid = closed_channel(40, 160)
        res = solve(grid, 0.0)
        assert np.all(res.w[grid.solid] == 0.0)

    def test_rejects_shape_mismatch(self):
        grid = closed_channel(16, 32)
        with pytest.raises(ValueError, match="形状が格子と不一致"):
            DownChannelFlowProcess().process(DownChannelInput(grid=grid, mu=np.ones((3, 3)), G=0.0))

    def test_rejects_nonpositive_viscosity(self):
        grid = closed_channel(16, 32)
        mu = np.full((grid.nx, grid.ny), MU)
        mu[0, 0] = 0.0
        with pytest.raises(ValueError, match="粘度"):
            DownChannelFlowProcess().process(DownChannelInput(grid=grid, mu=mu, G=0.0))

    def test_input_is_not_mutated(self):
        """C9: process() が入力の numpy 配列を変更しないこと."""
        grid = closed_channel(24, 96)
        mu = np.full((grid.nx, grid.ny), MU)
        before = mu.copy()
        DownChannelFlowProcess().execute(DownChannelInput(grid=grid, mu=mu, G=1.0e6))
        assert np.array_equal(mu, before)


class TestDownChannel1DExact:
    """1D 厳密解。離散化の符号をここで確定させる."""

    def test_couette_1d_exact(self):
        """引きずりのみ（G=0）で w(y) = w_barrel·(y/H) を機械精度で再現.

        等間隔格子なら 3 点ラプラシアン + 半セル Dirichlet は 1 次関数を厳密に表す。
        引きずり側の符号（w_barrel > 0 で下流向き）をここで固定する。
        """
        grid = parallel_plate(32)
        res = solve(grid, 0.0)
        expect = grid.spec.w_barrel * grid.yc / grid.spec.H
        assert np.max(np.abs(res.w[0, :] - expect)) < 1e-12 * grid.spec.w_barrel

    def test_poiseuille_sign_and_convergence(self):
        """圧力のみ（引きずり無し）で w(y) = (G/2μ)(y²−Hy) に 2 次収束すること.

        セル中心 FV の半セル Dirichlet は境界面の勾配に O(h) 誤差を持つので、
        2 次多項式に対して厳密にはならない（1 次の Couette とはここが違う）。
        G>0 で w<0（背圧が押出量を減らす向き）という符号をここで固定する。
        """
        G = 2.0e6
        errs = []
        for ny in (16, 32, 64):
            grid = parallel_plate(ny, N=0.0)
            res = solve(grid, G)
            y = grid.yc
            expect = G / (2.0 * MU) * (y**2 - grid.spec.H * y)
            assert np.all(res.w[0, :] < 0.0), "G>0 では w<0 でなければならない"
            errs.append(float(np.max(np.abs(res.w[0, :] - expect)) / np.max(np.abs(expect))))
        order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
        assert all(1.7 < o < 2.3 for o in order), f"観測次数 {order}, 誤差 {errs}"
        assert errs[-1] < 1e-3

    def test_two_layer_viscosity_matches_series_resistance(self):
        """μ を y 方向に 2 層にした引きずり流れが直列抵抗則と一致すること.

        せん断応力 τ が y に依らないので、境界の速度比は 1/μ の比で決まる。
        """
        ny = 64
        grid = parallel_plate(ny)
        mu = np.where(grid.yc[None, :] < grid.spec.H / 2.0, 500.0, 2000.0)
        mu = np.broadcast_to(mu, (grid.nx, grid.ny)).copy()
        res = DownChannelFlowProcess().process(DownChannelInput(grid=grid, mu=mu, G=0.0))
        # 厳密解: τ 一定の折れ線。調和平均の面粘度ならこれを厳密に再現する
        h_half = grid.spec.H / 2.0
        tau = grid.spec.w_barrel / (h_half / 500.0 + h_half / 2000.0)
        w_int = tau * h_half / 500.0
        y = grid.yc
        expect = np.where(y < h_half, tau * y / 500.0, w_int + tau * (y - h_half) / 2000.0)
        assert w_int / grid.spec.w_barrel == pytest.approx(0.8, rel=1e-12)
        assert np.max(np.abs(res.w[0, :] - expect)) < 1e-12 * grid.spec.w_barrel


class TestGateG1:
    """G1: ニュートン・隙間無し・純引きずり流れ（G=0）."""

    def test_drag_flow_matches_shape_factor(self):
        """Q = V_z W H F_d / 2 と 0.1% 以内で一致すること."""
        grid = closed_channel(96, 384)
        s = grid.spec
        res = solve(grid, 0.0)
        fd = shape_factor_drag(s.H / s.W)
        q_exact = s.w_barrel * s.W * s.H * fd / 2.0
        assert res.Q == pytest.approx(q_exact, rel=1e-3)

    def test_convergence_order_is_reduced_by_corner_singularity(self):
        """引きずり流れの観測次数が 2 をやや下回ること（実測 1.7〜1.8）.

        バレル（w=w_barrel）と側壁（w=0）が出会う上部 2 隅で速度が不連続になり、
        厳密解がそこで特異になる。キャビティ流と同型の角部特異性で、離散化が
        2 次でも大域の収束次数はこれに引きずられて下がる。バグではない。
        離散化そのものが 2 次であることは
        TestGateG2::test_pressure_flow_is_cleanly_second_order が示す。
        """
        errs = []
        for ny in (24, 48, 96):
            grid = closed_channel(ny, ny * 4)
            s = grid.spec
            res = solve(grid, 0.0)
            fd = shape_factor_drag(s.H / s.W)
            q_exact = s.w_barrel * s.W * s.H * fd / 2.0
            errs.append(abs(res.Q / q_exact - 1.0))
        order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
        assert all(1.5 < o < 2.1 for o in order), f"観測次数 {order}, 誤差 {errs}"
        assert errs[-1] < 1e-4


class TestGateG2:
    """G2: ニュートン・隙間無し・引きずり＋圧力."""

    @pytest.mark.parametrize("G", [-2.0e6, -1.0e6, 0.0, 1.0e6, 2.0e6])
    def test_drag_plus_pressure(self, G):
        """引きずり＋圧力の直線全体が解析解と 0.1% 以内で一致すること."""
        grid = closed_channel(96, 384)
        s = grid.spec
        res = solve(grid, G)
        fd = shape_factor_drag(s.H / s.W)
        fp = shape_factor_pressure(s.H / s.W)
        q_exact = metering_flow_rate(s.w_barrel, s.W, s.H, MU, G, F_d=fd, F_p=fp)
        assert res.Q == pytest.approx(q_exact, rel=1e-3)

    def test_pressure_flow_is_cleanly_second_order(self):
        """引きずり無しの純圧力流れは観測次数がきっちり 2 になること.

        移動壁が無いので角部特異性が消え、離散化本来の 2 次精度が見える。
        引きずり流れの次数低下が離散化のバグではないことの対照実験。
        """
        G = 2.0e6
        errs = []
        for ny in (24, 48, 96):
            spec = replace(_BASE, nx_channel=ny * 4, ny_bulk=ny, n_gap=0, N=0.0)
            half = 0.5 * (_BASE.W_t - _BASE.e)
            n_half = max(2, (ny * 4) // 2)
            spec = replace(spec, nx_land=max(1, round(_BASE.e / (half / n_half))))
            grid = ScrewGeometryProcess().process(spec)
            s = grid.spec
            res = solve(grid, G)
            fp = shape_factor_pressure(s.H / s.W)
            q_exact = -s.W * s.H**3 * G * fp / (12.0 * MU)
            errs.append(abs(res.Q / q_exact - 1.0))
        order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
        assert all(1.9 < o < 2.1 for o in order), f"観測次数 {order}, 誤差 {errs}"

    def test_linearity_in_G(self):
        """Q が G の厳密な一次関数であること（線形問題なので機械精度で成立）."""
        grid = closed_channel(48, 192)
        qs = [solve(grid, g).Q for g in (0.0, 1.0e6, 2.0e6)]
        assert qs[0] - qs[1] == pytest.approx(qs[1] - qs[2], rel=1e-10)

    def test_closure_point(self):
        """解析解の閉塞点 G で数値解の Q も 0 近傍になること."""
        grid = closed_channel(96, 384)
        s = grid.spec
        fd = shape_factor_drag(s.H / s.W)
        fp = shape_factor_pressure(s.H / s.W)
        q_drag = s.w_barrel * s.W * s.H * fd / 2.0
        g_closed = q_drag * 12.0 * MU / (s.W * s.H**3 * fp)
        res = solve(grid, g_closed)
        assert abs(res.Q) < 1e-3 * q_drag
