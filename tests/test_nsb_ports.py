"""[刳り抜きポート] 領域内ポートを実パッチ（リング面 BC）として扱う機能のテスト.

OpenFOAM の `subsetMesh` で円板セルを刳り抜き、露出面を inlet / outlet パッチにするのと
1 対 1 に対応させる（`experiments/nsb/trama_of_case.py` 参照）。既存の `INTERIOR_MASS_SOURCE` /
`INTERIOR_PRESSURE_SINK`（セル内の体積ソース / コンダクタンス sink）は別物として残す。

- `Test...API`     : 構造検査（刳り抜き・面種別・ヤコビアン・例外）
- `Test...Physics` : 物理検査（質量保存・圧力基準・体積ソース版との一致）
"""

from __future__ import annotations

import numpy as np
import pytest

from nsb import BC, NSBInput, NSBSettings, disk_mask, solve_steady
from nsb.assembly import BrinkmanDiscretization
from nsb.data import ConvectionSchemeType
from nsb.utils import mass_balance

LX, LY = 0.30, 0.20
R_PORT = 0.030
IN_C = (0.075, 0.10)
OUT_C = (0.225, 0.10)
MDOT = 2.0e-3  # [kg/s]（厚さ込みの 3 次元値）
H = 1.0e-3


def _case(
    nx: int = 48,
    ny: int = 32,
    mode: str = "carve",
    mdot: float = MDOT,
    h: np.ndarray | None = None,
    h_solid: float = 0.0,
    **settings_kw,
) -> NSBInput:
    """一様厚さの矩形領域に inlet / outlet の円板ポートを 2 つ置いたケース.

    mode="carve"    : 刳り抜きポート（リング面 BC）
    mode="interior" : 既存の体積ソース / コンダクタンス sink
    """
    if mode == "carve":
        patches = (
            BC.port_inlet(disk_mask(*IN_C, R_PORT), mdot),
            BC.port_outlet(disk_mask(*OUT_C, R_PORT), p=0.0),
        )
    elif mode == "interior":
        patches = (
            BC.interior_source(disk_mask(*IN_C, R_PORT), mdot),
            BC.interior_pressure_sink(disk_mask(*OUT_C, R_PORT), 1.0e-3, p=0.0),
        )
    else:  # pragma: no cover - テストの書き間違い検出
        raise ValueError(mode)
    return NSBInput(
        nx=nx,
        ny=ny,
        lx=LX,
        ly=LY,
        h=np.full((nx, ny), H) if h is None else h,
        bc=BC(patches=patches),
        h_solid=h_solid,
        settings=NSBSettings(linear_solver="jfnk", newton_max_iter=60, **settings_kw),
    )


def _disc(**kw) -> BrinkmanDiscretization:
    return BrinkmanDiscretization(_case(**kw).to_flow_input())


# --- 流路内ポート（実際の用途）: 直線流路の中にポートを 2 つ置く ---
CH_LX, CH_LY = 0.60, 0.15
CH_NX, CH_NY = 120, 30
CH_W = 0.070  # 流路幅
CH_R = 0.025  # ポート半径（w/2 = 35 mm より 2 セルぶん内側）
CH_IN, CH_OUT = (0.10, 0.075), (0.50, 0.075)


def _channel_case(mode: str = "carve", mdot: float = MDOT) -> NSBInput:
    yc = (np.arange(CH_NY) + 0.5) * (CH_LY / CH_NY)
    band = np.abs(yc - CH_LY / 2) < CH_W / 2
    h = np.where(band[None, :], H, 1.0e-6) * np.ones((CH_NX, 1))
    if mode == "carve":
        patches = (
            BC.port_inlet(disk_mask(*CH_IN, CH_R), mdot),
            BC.port_outlet(disk_mask(*CH_OUT, CH_R), p=0.0),
        )
    else:
        patches = (
            BC.interior_source(disk_mask(*CH_IN, CH_R), mdot),
            BC.interior_pressure_sink(disk_mask(*CH_OUT, CH_R), 1.0e-3, p=0.0),
        )
    return NSBInput(
        nx=CH_NX,
        ny=CH_NY,
        lx=CH_LX,
        ly=CH_LY,
        h=h,
        bc=BC(patches=patches),
        h_solid=1.0e-5,
        settings=NSBSettings(linear_solver="jfnk", newton_max_iter=60),
    )


class TestNSBCarvePortAPI:
    """[刳り抜きポート] 構造検査."""

    def test_port_cells_are_carved_out(self):
        """円板セルは未知数から外れ、その周りのリング面に inlet / outlet 種別が立つ."""
        disc = _disc()
        xc = (np.arange(disc.nx) + 0.5) * disc.dx
        yc = (np.arange(disc.ny) + 0.5) * disc.dy
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        want_in = disk_mask(*IN_C, R_PORT)(X, Y)
        want_out = disk_mask(*OUT_C, R_PORT)(X, Y)
        assert want_in.any() and want_out.any()
        # 円板セルは固体扱い（active=False）
        assert not disc.active[want_in].any()
        assert not disc.active[want_out].any()
        assert disc.n_active == disc.n - int((want_in | want_out).sum())
        assert disc.has_solid
        # リング面に種別が立つ（1: inlet, 2: outlet）
        assert (disc.pkind_x == 1).any() and (disc.pkind_y == 1).any()
        assert (disc.pkind_x == 2).any() and (disc.pkind_y == 2).any()
        # 種別が立つのは「片側だけ流体」の内部面に限る
        for pk, wall in ((disc.pkind_x, disc.wall_x), (disc.pkind_y, disc.wall_y)):
            assert np.all(wall[pk != 0] != 0)

    def test_inlet_ring_carries_exactly_the_prescribed_mass_flow(self):
        """リング面の質量流束の合計が ṁ/h（単位深さ換算）に機械精度で一致する."""
        disc = _disc()
        m = disc.rho * (
            np.abs(disc.pun_x[disc.pkind_x == 1] * disc.dy).sum()
            + np.abs(disc.pun_y[disc.pkind_y == 1] * disc.dx).sum()
        )
        assert m == pytest.approx(MDOT / H, rel=1e-12)

    def test_solid_rows_are_decoupled_and_kernels_agree(self):
        """ポートセルの残差は 0、numba 経路と numpy 経路が一致、J は固体行が単位行."""
        import scipy.sparse as sp

        disc = _disc()
        rng = np.random.default_rng(0)
        x = disc.mask_state(rng.normal(size=3 * disc.n) * 0.1)
        s = NSBSettings()
        r = disc.residual_fast(x, s.scheme, s.venkat_k)
        assert np.abs(r[disc.dead3]).max() == 0.0
        assert np.abs(r - disc.residual(x, s.scheme, s.venkat_k)).max() < 1e-12 * max(
            np.abs(r).max(), 1e-300
        )
        x2 = x.copy()
        x2[disc.dead3] = rng.normal(size=int(disc.dead3.sum()))
        r2 = disc.residual_fast(x2, s.scheme, s.venkat_k)
        assert np.abs(r - r2)[disc.live3].max() < 1e-12
        j = disc.jacobian_first_order(disc.compute_state(x, s.scheme, s.venkat_k), x=x).tocsr()
        assert j[:, disc.dead3][disc.live3, :].nnz == 0
        j_d = j[disc.dead3, :]
        assert j_d[:, disc.live3].nnz == 0
        assert (j_d[:, disc.dead3] - sp.identity(int(disc.dead3.sum()), format="csr")).nnz == 0

    def test_first_order_jacobian_matches_finite_difference(self):
        """貫通項支配（RC 係数が速度に依存しない）で J1 が FD ヤコビアンと一致する."""
        nx, ny = 20, 14
        disc = _disc(nx=nx, ny=ny, h=np.full((nx, ny), 1.0e-5))
        n = disc.n
        rng = np.random.default_rng(1)
        x = disc.mask_state(
            np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(0, 10.0, n)])
        )
        sch = ConvectionSchemeType.FIRST_ORDER_UPWIND
        st = disc.compute_state(x, sch, 5.0)
        J = disc.jacobian_first_order(st, x=x).toarray()
        Jfd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jfd[:, k] = (disc.residual(x + e, sch, 5.0) - disc.residual(x - e, sch, 5.0)) / (2 * hk)
        # 死んだセル（刳り抜いたポート）の列は残差が動かないので FD では出ない。J は単位行で
        # そこを埋めているので、比較は流体セルの行・列だけで行う
        live = disc.live3
        J, Jfd = J[np.ix_(live, live)], Jfd[np.ix_(live, live)]
        scale = np.abs(Jfd).max(axis=0) + 1e-12
        assert np.all(np.abs(J - Jfd).max(axis=0) / scale < 1e-3)

    def test_port_inlet_without_pressure_reference_raises(self):
        """流出はあるが圧力基準が無い（ポート inlet + 流量指定の吸出）と例外."""
        inp = NSBInput(
            nx=32,
            ny=24,
            lx=LX,
            ly=LY,
            h=np.full((32, 24), H),
            bc=BC(
                patches=(
                    BC.port_inlet(disk_mask(*IN_C, R_PORT), MDOT),
                    BC.interior_sink(disk_mask(*OUT_C, R_PORT), MDOT),
                )
            ),
        )
        with pytest.raises(ValueError, match="圧力の基準"):
            BrinkmanDiscretization(inp.to_flow_input())

    def test_empty_port_mask_raises(self):
        inp = NSBInput(
            nx=16,
            ny=12,
            lx=LX,
            ly=LY,
            h=np.full((16, 12), H),
            bc=BC(
                patches=(
                    BC.port_inlet(disk_mask(-1.0, -1.0, 1e-6), MDOT),
                    BC.port_outlet(disk_mask(*OUT_C, R_PORT)),
                )
            ),
        )
        with pytest.raises(ValueError, match="ポート"):
            BrinkmanDiscretization(inp.to_flow_input())

    def test_port_touching_solid_raises(self):
        """リング面が固体セルに接するとリング面積が変わるので例外にする."""
        nx, ny = 48, 32
        h = np.full((nx, ny), H)
        h[:, :12] = 1e-6  # 下半分を塞ぐ → inlet 円板が固体に食い込む
        with pytest.raises(ValueError, match="固体"):
            _disc(nx=nx, ny=ny, h=h, h_solid=1e-5)

    def test_smooth_weight_is_rejected_for_ports(self):
        """刳り抜きは離散的なので滑らかな窓関数は受け付けない（設計感度用途は INTERIOR_* のまま）."""
        from nsb.data import BoundaryKind, BoundaryPatch, smooth_disk

        with pytest.raises(ValueError, match="mask"):
            BoundaryPatch(
                BoundaryKind.PORT_MASS_FLOW_INLET,
                mask=None,
                mass_flow=MDOT,
                weight=smooth_disk(*IN_C, R_PORT, 1e-3),
            )


class TestNSBCarvePortIntegration:
    """[刳り抜きポート] 他の経路との噛み合わせ（非定常・未対応レイヤーのガード）."""

    def test_unsteady_converges_to_the_steady_solution(self):
        """非定常（後退 Euler）でもポート面が効き、十分回すと定常解に落ちる."""
        from nsb.unsteady import solve_unsteady

        steady = solve_steady(_case(), log=None)
        assert steady.converged
        res = solve_unsteady(_case(), dt=0.05, n_steps=40, log=None)
        d = _disc()
        num = np.hypot(res.u - steady.u, res.v - steady.v)[d.active]
        den = np.hypot(steady.u, steady.v)[d.active]
        assert np.linalg.norm(num) / np.linalg.norm(den) < 0.02
        assert np.abs(res.u[~d.active]).max() == 0.0

    def test_nsbp_rejects_carve_ports(self):
        """nsbp（PETSc）は面種別を通していないので、黙って違う問題を解かせない."""
        from nsbp.problem import make_discretization

        with pytest.raises(NotImplementedError, match="刳り抜きポート"):
            make_discretization(_case())

    def test_adjoint_objective_rejects_carve_ports(self):
        """随伴の圧損目的関数は q_src 重みなので刳り抜きポートには使えない."""
        from nsb.adjoint import source_mean_pressure_objective

        obj = source_mean_pressure_objective()
        inp = _case()
        x = np.zeros(3 * inp.nx * inp.ny)
        with pytest.raises(NotImplementedError, match="刳り抜きポート"):
            obj.value(x, inp)


class TestNSBCarvePortPhysics:
    """[刳り抜きポート] 物理検査."""

    @staticmethod
    def _solved(mode: str):
        res = solve_steady(_case(mode=mode), log=None)
        assert res.converged, res.failure_reason
        return res

    def test_mass_balance_and_pressure_reference(self):
        res = self._solved("carve")
        assert res.mass_in == pytest.approx(MDOT / H, rel=1e-6)
        assert mass_balance(res) == pytest.approx(1.0, rel=1e-6)
        # outlet ポートのリング面が圧力基準になるので、そこの圧力は 0 近傍
        disc = _disc()
        ring = np.zeros((disc.nx, disc.ny), dtype=bool)
        ring[:-1] |= disc.pkind_x[1:-1] == 2
        ring[1:] |= disc.pkind_x[1:-1] == 2
        ring &= disc.active
        assert np.abs(res.p[ring]).max() < 0.05 * np.ptp(res.p[disc.active])

    def test_flow_leaves_inlet_ring_outward(self):
        """inlet 円板の右側では +x、左側では -x に流れる（放射状に吹き出す）."""
        res = self._solved("carve")
        disc = _disc()
        xc = (np.arange(disc.nx) + 0.5) * disc.dx
        yc = (np.arange(disc.ny) + 0.5) * disc.dy
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        band = (np.abs(Y - IN_C[1]) < 0.01) & disc.active
        right = band & (X > IN_C[0] + R_PORT) & (X < IN_C[0] + R_PORT + 0.02)
        left = band & (X < IN_C[0] - R_PORT) & (X > IN_C[0] - R_PORT - 0.02)
        assert res.u[right].mean() > 0.0
        assert res.u[left].mean() < 0.0

    def test_difference_from_volume_source_decays_with_distance(self):
        """体積ソース版との差はポートから離れるほど小さくなる（差の入口はポートだけ）."""
        carve = self._solved("carve")
        inter = self._solved("interior")
        disc = _disc()
        xc = (np.arange(disc.nx) + 0.5) * disc.dx
        yc = (np.arange(disc.ny) + 0.5) * disc.dy
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        l2 = []
        for k in (1.2, 1.5, 2.0, 2.5, 3.0):
            far = disc.active.copy()
            for cx, cy in (IN_C, OUT_C):
                far &= (X - cx) ** 2 + (Y - cy) ** 2 > (k * R_PORT) ** 2
            num = np.hypot(carve.u - inter.u, carve.v - inter.v)[far]
            den = np.hypot(inter.u, inter.v)[far]
            l2.append(float(np.linalg.norm(num) / np.linalg.norm(den)))
        assert all(a > b for a, b in zip(l2, l2[1:], strict=False)), l2
        # 圧力損失（開放領域なので円板が塞ぐぶん刳り抜き版がやや高い）は 1 割の中
        dp = lambda r: float(np.ptp(r.p[disc.active]))  # noqa: E731
        assert dp(carve) == pytest.approx(dp(inter), rel=0.10)

    def test_channel_profile_matches_volume_source_downstream(self):
        """流路内に置いたポートなら、下流で発達した断面は体積ソース版と一致する.

        流量も流路も同じなので、ポートの与え方の差はポート近傍にしか残らない
        （trama の検算で「除外距離を伸ばすと単調に落ちる」と見たのと同じ構造）。
        """
        carve = solve_steady(_channel_case("carve"), log=None)
        inter = solve_steady(_channel_case("interior"), log=None)
        assert carve.converged and inter.converged
        disc = BrinkmanDiscretization(_channel_case("carve").to_flow_input())
        xc = (np.arange(CH_NX) + 0.5) * (CH_LX / CH_NX)
        mid = disc.active & ((xc > 0.22) & (xc < 0.38))[:, None]
        assert mid.sum() > 100
        num = np.hypot(carve.u - inter.u, carve.v - inter.v)[mid]
        den = np.hypot(inter.u, inter.v)[mid]
        assert np.linalg.norm(num) / np.linalg.norm(den) < 0.02
        # 流量も一致
        assert carve.mass_in == pytest.approx(MDOT / H, rel=1e-6)
        assert mass_balance(carve) == pytest.approx(1.0, rel=1e-6)
