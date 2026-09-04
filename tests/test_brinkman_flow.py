"""BrinkmanFlowFVMProcess / UTurnThicknessProcess テスト.

API テスト（Process 契約・ヤコビアン整合）、収束テスト（U ターン基本ケース）、
物理テスト（質量保存・Hele-Shaw 圧力損失・閉塞部の無流れ）を含む。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.brinkman_flow import (
    BoundaryKind,
    BoundaryPatch,
    BrinkmanFlowFVMProcess,
    BrinkmanFlowInput,
    BrinkmanFlowResult,
    BrinkmanGeometry,
    BrinkmanSolverSettings,
    ConvectionSchemeType,
    JacobianMode,
    PseudoTimeMode,
    ThicknessInput,
    ThicknessModel,
    ThicknessResult,
    ThicknessSpec,
    UTurnThicknessProcess,
    disk_mask,
    east_span,
    north_span,
    rect_mask,
    south_span,
    west_span,
)
from xkep_cae_fluid.brinkman_flow.assembly import BrinkmanDiscretization
from xkep_cae_fluid.core.testing import binds_to


def _thickness(nx: int, ny: int, model: ThicknessModel, **kw) -> ThicknessResult:
    return UTurnThicknessProcess().execute(
        ThicknessInput(nx, ny, ThicknessSpec(model=model, **kw), BrinkmanGeometry())
    )


def _input(
    nx: int, ny: int, model: ThicknessModel, u_inlet: float, **settings
) -> BrinkmanFlowInput:
    th = _thickness(nx, ny, model).thickness
    return BrinkmanFlowInput(
        nx=nx, ny=ny, thickness=th, u_inlet=u_inlet, settings=BrinkmanSolverSettings(**settings)
    )


# ---------------------------------------------------------------------------
# UTurnThicknessProcess
# ---------------------------------------------------------------------------


@binds_to(UTurnThicknessProcess)
class TestUTurnThicknessAPI:
    """厚さ場ビルダーの API テスト."""

    def test_meta(self):
        assert UTurnThicknessProcess.meta.name == "UTurnThickness"
        assert UTurnThicknessProcess.meta.module == "pre"

    def test_flat_is_uniform(self):
        res = _thickness(36, 24, ThicknessModel.FLAT)
        assert res.thickness.shape == (36, 24)
        assert np.all(res.thickness == 1.0e-3)
        assert res.channel_mask.all()

    def test_uturn_connects_inlet_to_outlet(self):
        res = _thickness(72, 48, ThicknessModel.UTURN)
        h = res.thickness
        yc = (np.arange(48) + 0.5) * (0.4 / 48)
        inlet = (yc > 0.25) & (yc < 0.35)
        outlet = (yc > 0.05) & (yc < 0.15)
        assert np.all(h[0, inlet] == 1.0e-3)
        assert np.all(h[0, outlet] == 1.0e-3)
        assert np.all(h[0, ~(inlet | outlet)] == 1.0e-5)
        # 中央帯（inlet と outlet の間）は左側で閉塞、右端の折返しで開通
        mid = (yc > 0.15) & (yc < 0.25)
        assert np.all(h[0, mid] == 1.0e-5)
        assert np.all(h[-1, mid] == 1.0e-3)
        assert 0.3 < res.channel_mask.mean() < 0.7


# ---------------------------------------------------------------------------
# BrinkmanFlowFVMProcess
# ---------------------------------------------------------------------------


@binds_to(BrinkmanFlowFVMProcess)
class TestBrinkmanFlowAPI:
    """Process 契約とヤコビアン整合."""

    def test_meta(self):
        assert BrinkmanFlowFVMProcess.meta.name == "BrinkmanFlowFVM"
        assert BrinkmanFlowFVMProcess.meta.module == "solve"

    def test_input_validation(self):
        with pytest.raises(ValueError):
            BrinkmanFlowInput(nx=4, ny=4, thickness=np.ones((4, 3)))
        with pytest.raises(ValueError):
            BrinkmanFlowInput(nx=4, ny=4, thickness=np.zeros((4, 4)))

    def test_first_order_jacobian_matches_finite_difference(self):
        """RC 係数が速度に依存しない状態（貫通項支配）で J1 が FD ヤコビアンと一致すること."""
        nx, ny = 12, 8
        th = np.full((nx, ny), 1.0e-5)
        inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, u_inlet=0.1)
        disc = BrinkmanDiscretization(inp)
        n = disc.n
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(0, 10.0, n)])
        sch = ConvectionSchemeType.FIRST_ORDER_UPWIND
        st = disc.compute_state(x, sch, 5.0)
        J = disc.jacobian_first_order(st).toarray()
        Jfd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jfd[:, k] = (disc.residual(x + e, sch, 5.0) - disc.residual(x - e, sch, 5.0)) / (2 * hk)
        scale = np.abs(Jfd).max(axis=0) + 1e-12
        assert np.abs(J - Jfd).max(axis=0).max() < 1e-4 * scale.max()
        # 各列相対誤差
        assert np.all(np.abs(J - Jfd).max(axis=0) / scale < 1e-3)

    def test_returns_result_without_raising_on_failure(self):
        """反復上限に達しても例外を出さず converged=False を返すこと."""
        inp = _input(24, 16, ThicknessModel.UTURN, 0.1, newton_max_iter=1, newton_tol=1e-12)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert isinstance(res, BrinkmanFlowResult)
        assert res.converged is False
        assert res.failure_reason == "max_iter"
        assert len(res.residual_history) == 2


class TestBrinkmanFlowConvergence:
    """基本ケースの収束."""

    @pytest.mark.parametrize("mode", [JacobianMode.JFNK, JacobianMode.DEFECT_CORRECTION])
    def test_uturn_low_speed_converges(self, mode: JacobianMode):
        inp = _input(36, 24, ThicknessModel.UTURN, 0.1, jacobian_mode=mode, newton_max_iter=60)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.residual_history[-1] / res.residual_history[0] < 1e-6

    def test_first_order_scheme_converges(self):
        inp = _input(
            36,
            24,
            ThicknessModel.UTURN,
            0.1,
            convection_scheme=ConvectionSchemeType.FIRST_ORDER_UPWIND,
            newton_max_iter=60,
        )
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason

    def test_global_pseudo_time_converges(self):
        inp = _input(
            36,
            24,
            ThicknessModel.UTURN,
            0.1,
            pseudo_time_mode=PseudoTimeMode.GLOBAL,
            newton_max_iter=80,
        )
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        # 既定（RC 係数に擬似時間項を含めない）では定常残差 = 最終残差
        rel = res.residual_history[-1] / res.residual_history[0]
        assert res.steady_residual_ratio == pytest.approx(rel, rel=1e-9)

    def test_rhie_chow_pseudo_time_reports_steady_residual(self):
        """RC 係数に擬似時間項を含める変種は Δτ 非依存の定常残差を別途報告する."""
        inp = _input(
            36,
            24,
            ThicknessModel.UTURN,
            0.1,
            rhie_chow_pseudo_time=True,
            newton_max_iter=80,
        )
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert np.isfinite(res.steady_residual_ratio)
        assert len(res.residual_history) == res.n_newton + 1


class TestBrinkmanFlowPhysics:
    """物理的妥当性."""

    @pytest.fixture(scope="class")
    def uturn_result(self) -> tuple[BrinkmanFlowInput, BrinkmanFlowResult]:
        inp = _input(72, 48, ThicknessModel.UTURN, 0.1, newton_max_iter=60)
        return inp, BrinkmanFlowFVMProcess().execute(inp)

    def test_mass_conservation(self, uturn_result):
        inp, res = uturn_result
        assert res.converged
        m_expected = inp.rho * inp.u_inlet * 0.1
        assert abs(res.mass_in - m_expected) / m_expected < 1e-6
        assert abs(res.mass_out - res.mass_in) / m_expected < 1e-5

    def test_no_flow_in_blocked_region(self, uturn_result):
        inp, res = uturn_result
        blocked = inp.thickness < 1.0e-4
        speed = np.hypot(res.u, res.v)
        assert speed[blocked].max() < 1e-2 * inp.u_inlet

    def test_hele_shaw_pressure_drop(self, uturn_result):
        """低速では圧力損失 ≈ (12 mu_b / h²) U × 経路長（Hele-Shaw、慣性項無視）."""
        inp, res = uturn_result
        k = inp.brinkman_factor * inp.mu_brinkman / 1.0e-6
        # 経路長: 往路 (0.7-0.05) + 折返し 0.2 + 復路 (0.7-0.05) ≈ 1.5 m（近似）
        dp_est = k * inp.u_inlet * 1.5
        yc = (np.arange(inp.ny) + 0.5) * inp.dy
        inlet = (yc > 0.25) & (yc < 0.35)
        p_in = res.p[0, inlet].mean()
        assert 0.5 * dp_est < p_in < 1.5 * dp_est


# ---------------------------------------------------------------------------
# 座標マスク境界 / 質量流入境界
# ---------------------------------------------------------------------------


def _flat_input(nx: int, ny: int, boundaries, **settings) -> BrinkmanFlowInput:
    th = np.full((nx, ny), 1.0e-3)
    return BrinkmanFlowInput(
        nx=nx,
        ny=ny,
        thickness=th,
        boundaries=boundaries,
        settings=BrinkmanSolverSettings(**settings),
    )


class TestBoundaryPatchAPI:
    """座標マスク境界の解決と質量流入境界の換算."""

    def test_default_boundaries_match_geometry(self):
        """boundaries=None は従来の左壁 inlet/outlet と同じ面配置になる."""
        th = np.full((36, 24), 1.0e-3)
        inp = BrinkmanFlowInput(nx=36, ny=24, thickness=th, u_inlet=0.3)
        disc = BrinkmanDiscretization(inp)
        yc = (np.arange(24) + 0.5) * inp.dy
        W = disc.sides["W"]
        assert np.array_equal(W.is_inlet, (yc > 0.25) & (yc < 0.35))
        assert np.array_equal(W.is_outlet, (yc > 0.05) & (yc < 0.15))
        assert W.un[W.is_inlet] == pytest.approx(0.3)
        assert disc.u_scale == pytest.approx(0.3)
        for k in ("E", "S", "N"):
            assert not disc.sides[k].is_inlet.any() and not disc.sides[k].is_outlet.any()

    def test_mass_flow_inlet_velocity(self):
        """u_n = mass_flow / (ρ Σ h_f A_f)。inlet 3 面（Δy=0.4/24）なら Σ h A = 3·1e-3·Δy."""
        nx, ny = 36, 24
        th = np.full((nx, ny), 1.0e-3)
        mdot = 0.02  # kg/s
        bnd = (
            BoundaryPatch(BoundaryKind.MASS_FLOW_INLET, north_span(0.5, 0.55, 0.4), mass_flow=mdot),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, west_span(0.05, 0.15)),
        )
        inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, boundaries=bnd)
        disc = BrinkmanDiscretization(inp)
        N = disc.sides["N"]
        n_faces = int(N.is_inlet.sum())
        xc = (np.arange(nx) + 0.5) * inp.dx
        assert n_faces == int(((xc > 0.5) & (xc < 0.55)).sum()) == 2
        u_expected = mdot / (inp.rho * n_faces * 1.0e-3 * inp.dx)
        assert N.un[N.is_inlet] == pytest.approx(u_expected)
        assert disc.v_n[N.is_inlet] == pytest.approx(-u_expected)  # 上壁からの流入は -y 方向

    def test_missing_inlet_raises(self):
        bnd = (BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, west_span(0.05, 0.15)),)
        with pytest.raises(ValueError):
            BrinkmanDiscretization(_flat_input(12, 8, bnd))

    def test_jacobian_matches_fd_with_patches_on_all_walls(self):
        """inlet が上壁と右壁、outlet が下壁と左壁の構成で J1 が FD ヤコビアンと一致する."""
        nx, ny = 12, 8
        th = np.full((nx, ny), 1.0e-5)
        bnd = (
            BoundaryPatch(BoundaryKind.VELOCITY_INLET, north_span(0.1, 0.3, 0.4), velocity=0.1),
            BoundaryPatch(BoundaryKind.MASS_FLOW_INLET, east_span(0.2, 0.3, 0.7), mass_flow=1e-3),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, south_span(0.4, 0.6), pressure=5.0),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, west_span(0.05, 0.15)),
        )
        inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, boundaries=bnd)
        disc = BrinkmanDiscretization(inp)
        n = disc.n
        rng = np.random.default_rng(2)
        x = np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(0, 10.0, n)])
        sch = ConvectionSchemeType.FIRST_ORDER_UPWIND
        st = disc.compute_state(x, sch, 5.0)
        J = disc.jacobian_first_order(st).toarray()
        Jfd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jfd[:, k] = (disc.residual(x + e, sch, 5.0) - disc.residual(x - e, sch, 5.0)) / (2 * hk)
        scale = np.abs(Jfd).max(axis=0) + 1e-12
        assert np.all(np.abs(J - Jfd).max(axis=0) / scale < 1e-3)


class TestBoundaryPatchPhysics:
    """任意壁の inlet/outlet と質量流入境界での質量保存."""

    def test_mass_flow_inlet_top_wall_outlet_right_wall(self):
        mdot = 0.01  # kg/s（h=1e-3 なので単位深さ流量は 10 kg/s）
        bnd = (
            BoundaryPatch(BoundaryKind.MASS_FLOW_INLET, north_span(0.1, 0.2, 0.4), mass_flow=mdot),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, east_span(0.1, 0.2, 0.7)),
        )
        inp = _flat_input(36, 24, bnd, newton_max_iter=60)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.mass_in == pytest.approx(mdot / 1.0e-3, rel=1e-6)
        assert res.mass_out == pytest.approx(res.mass_in, rel=1e-6)
        # 流入は -y 方向、流出は +x 方向
        assert res.v[:, -1][(np.arange(36) + 0.5) * inp.dx > 0.1].min() < 0.0
        assert res.u[-1].max() > 0.0

    def test_two_inlets_one_outlet(self):
        """左壁 2 か所の質量流入（合計固定）を 1 つのマスクで指定しても流量が保存される."""
        mask = lambda x, y: (x <= 1e-12) & (((y > 0.05) & (y < 0.1)) | ((y > 0.3) & (y < 0.35)))  # noqa: E731
        bnd = (
            BoundaryPatch(BoundaryKind.MASS_FLOW_INLET, mask, mass_flow=0.005),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, east_span(0.15, 0.25, 0.7)),
        )
        inp = _flat_input(36, 24, bnd, newton_max_iter=60)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.mass_in == pytest.approx(0.005 / 1.0e-3, rel=1e-6)
        assert res.mass_out == pytest.approx(res.mass_in, rel=1e-6)


# ---------------------------------------------------------------------------
# 領域内マニホールド（紙面垂直方向の注入 / 吸出）
# ---------------------------------------------------------------------------


class TestInteriorManifoldAPI:
    def test_source_distribution_and_pressure_reference_required(self):
        nx, ny = 36, 24
        th = np.full((nx, ny), 1.0e-3)
        src = BoundaryPatch(
            BoundaryKind.INTERIOR_MASS_SOURCE, disk_mask(0.2, 0.2, 0.04), mass_flow=0.01
        )
        # 圧力基準なし（流量指定の吸出だけ）は ValueError
        with pytest.raises(ValueError):
            BrinkmanDiscretization(
                BrinkmanFlowInput(
                    nx=nx,
                    ny=ny,
                    thickness=th,
                    boundaries=(
                        src,
                        BoundaryPatch(
                            BoundaryKind.INTERIOR_MASS_SINK,
                            disk_mask(0.5, 0.2, 0.04),
                            mass_flow=0.01,
                        ),
                    ),
                )
            )
        disc = BrinkmanDiscretization(
            BrinkmanFlowInput(
                nx=nx,
                ny=ny,
                thickness=th,
                boundaries=(
                    src,
                    BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, west_span(0.05, 0.15)),
                ),
            )
        )
        # 単位深さ総ソース = ṁ / h（h 一様）
        assert disc.q_src.sum() == pytest.approx(0.01 / 1.0e-3)
        assert disc.interior_mask.sum() == (disc.q_src > 0).sum() > 0
        assert disc.u_scale > 0.0

    def test_jacobian_matches_fd_with_interior_patches(self):
        """注入 + 流量指定吸出 + 圧力指定マニホールドの全種で J1 が FD と一致する（吸出の運動量項・圧力結合含む）."""
        nx, ny = 12, 8
        th = np.full((nx, ny), 1.0e-5)
        bnd = (
            BoundaryPatch(
                BoundaryKind.INTERIOR_MASS_SOURCE, rect_mask(0.05, 0.2, 0.1, 0.3), mass_flow=1e-3
            ),
            BoundaryPatch(
                BoundaryKind.INTERIOR_MASS_SINK, rect_mask(0.3, 0.4, 0.05, 0.15), mass_flow=5e-4
            ),
            BoundaryPatch(
                BoundaryKind.INTERIOR_PRESSURE_SINK,
                rect_mask(0.5, 0.65, 0.2, 0.35),
                conductance=1e-6,
                pressure=2.0,
            ),
        )
        inp = BrinkmanFlowInput(nx=nx, ny=ny, thickness=th, boundaries=bnd)
        disc = BrinkmanDiscretization(inp)
        n = disc.n
        rng = np.random.default_rng(3)
        # 圧力は p_sink=2 をまたぐ値にして吸出/逆流の両側を含める
        x = np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), rng.normal(2.0, 10.0, n)])
        sch = ConvectionSchemeType.FIRST_ORDER_UPWIND
        st = disc.compute_state(x, sch, 5.0)
        J = disc.jacobian_first_order(st, x=x).toarray()
        Jfd = np.zeros_like(J)
        for k in range(3 * n):
            e = np.zeros(3 * n)
            hk = 1e-6 * max(1.0, abs(x[k]))
            e[k] = hk
            Jfd[:, k] = (disc.residual(x + e, sch, 5.0) - disc.residual(x - e, sch, 5.0)) / (2 * hk)
        scale = np.abs(Jfd).max(axis=0) + 1e-12
        assert np.all(np.abs(J - Jfd).max(axis=0) / scale < 1e-3)


class TestInteriorManifoldPhysics:
    def test_source_disk_to_boundary_outlet(self):
        """中央の注入マニホールド → 左壁 outlet。質量保存と、マニホールドから放射状に出る流れ."""
        mdot = 0.01
        bnd = (
            BoundaryPatch(
                BoundaryKind.INTERIOR_MASS_SOURCE, disk_mask(0.35, 0.2, 0.05), mass_flow=mdot
            ),
            BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, west_span(0.05, 0.15)),
        )
        inp = _flat_input(36, 24, bnd, newton_max_iter=80)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.mass_in == pytest.approx(mdot / 1.0e-3, rel=1e-6)
        assert res.mass_out == pytest.approx(res.mass_in, rel=1e-6)
        # マニホールドの右側では +x、左側では -x に流れる
        i_c, j_c = int(0.35 / inp.dx), int(0.2 / inp.dy)
        assert res.u[i_c + 4, j_c] > 0.0 and res.u[i_c - 4, j_c] < 0.0

    def test_source_to_pressure_manifold_only(self):
        """境界 outlet なし: 注入マニホールド → 圧力指定マニホールド。圧力基準は p_manifold + q/C."""
        mdot = 0.01
        p_man = 100.0
        cond = 1e-6  # kg/(s·Pa)
        bnd = (
            BoundaryPatch(
                BoundaryKind.INTERIOR_MASS_SOURCE, disk_mask(0.15, 0.2, 0.05), mass_flow=mdot
            ),
            BoundaryPatch(
                BoundaryKind.INTERIOR_PRESSURE_SINK,
                disk_mask(0.55, 0.2, 0.05),
                conductance=cond,
                pressure=p_man,
            ),
        )
        inp = _flat_input(36, 24, bnd, newton_max_iter=80)
        res = BrinkmanFlowFVMProcess().execute(inp)
        assert res.converged, res.failure_reason
        assert res.mass_in == pytest.approx(mdot / 1.0e-3, rel=1e-6)
        assert res.mass_out == pytest.approx(res.mass_in, rel=1e-6)
        # 吸出セルの圧力は p_man より高く、3 次元流量 ṁ = C (p̄ - p_man) を満たす
        disc = BrinkmanDiscretization(inp)
        sink = disc.c_sink > 0
        p_bar = float((res.p[sink] * disc.c_sink[sink]).sum() / disc.c_sink[sink].sum())
        assert p_bar > p_man
        assert cond * (p_bar - p_man) == pytest.approx(mdot, rel=1e-6)
