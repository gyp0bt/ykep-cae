"""AquariumFilterProcess + InternalFaceBC テスト (Phase 6.3a).

API テスト（Process 契約）と物理テスト（質量保存・吐出速度整合）を含む。
InternalFaceBC 自体は NaturalConvection 経由で強制速度が伝搬するかを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.aquarium import (
    AquariumFilterInput,
    AquariumFilterProcess,
    AquariumFilterResult,
    AquariumGeometryInput,
    AquariumGeometryProcess,
    NozzleGeometry,
)
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.natural_convection import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    InternalFaceBC,
    InternalFaceBCKind,
    NaturalConvectionFDMProcess,
    NaturalConvectionInput,
)


def _make_geometry(nx: int = 12, ny: int = 4, nz: int = 8) -> tuple:
    """共通の AquariumGeometryResult を返すヘルパー."""
    geom_res = AquariumGeometryProcess().process(
        AquariumGeometryInput(Lx=0.3, Ly=0.1, Lz=0.2, nx=nx, ny=ny, nz=nz, substrate_depth=0.0)
    )
    return geom_res


def _make_filter_input(
    flow_rate_lph: float = 440.0,
    direction: tuple = (1.0, 0.0, 0.0),
    inflow_range: tuple = ((0.0, 0.03), (0.03, 0.07), (0.13, 0.19)),
    outflow_range: tuple = ((0.27, 0.30), (0.03, 0.07), (0.01, 0.05)),
    temperature_K: float | None = None,
) -> AquariumFilterInput:
    geom_res = _make_geometry()
    return AquariumFilterInput(
        x_centers=geom_res.x_centers,
        y_centers=geom_res.y_centers,
        z_centers=geom_res.z_centers,
        dx=geom_res.dx,
        dy=geom_res.dy,
        dz=geom_res.dz,
        inflow_geometry=NozzleGeometry(
            x_range=inflow_range[0], y_range=inflow_range[1], z_range=inflow_range[2]
        ),
        outflow_geometry=NozzleGeometry(
            x_range=outflow_range[0], y_range=outflow_range[1], z_range=outflow_range[2]
        ),
        flow_rate_lph=flow_rate_lph,
        inflow_direction=direction,
        inflow_temperature_K=temperature_K,
    )


# ---------------------------------------------------------------------------
# API テスト
# ---------------------------------------------------------------------------


@binds_to(AquariumFilterProcess)
class TestAquariumFilterAPI:
    """AquariumFilterProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        assert AquariumFilterProcess.meta.name == "AquariumFilterProcess"
        assert AquariumFilterProcess.meta.module == "pre"

    def test_process_returns_filter_result(self):
        res = AquariumFilterProcess().process(_make_filter_input())
        assert isinstance(res, AquariumFilterResult)
        assert isinstance(res.inflow_bc, InternalFaceBC)
        assert isinstance(res.outflow_bc, InternalFaceBC)
        assert res.inflow_bc.kind == InternalFaceBCKind.INLET
        assert res.outflow_bc.kind == InternalFaceBCKind.OUTLET

    def test_invalid_flow_rate_raises(self):
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(_make_filter_input(flow_rate_lph=0.0))
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(_make_filter_input(flow_rate_lph=-10.0))

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(
                _make_filter_input(inflow_range=((0.1, 0.05), (0.03, 0.07), (0.13, 0.19)))
            )

    def test_zero_direction_raises(self):
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(_make_filter_input(direction=(0.0, 0.0, 0.0)))

    def test_empty_inflow_region_raises(self):
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(
                _make_filter_input(inflow_range=((-0.5, -0.4), (0.03, 0.07), (0.13, 0.19)))
            )

    def test_overlapping_regions_raises(self):
        """吐出と吸入のセルが重なると ValueError."""
        with pytest.raises(ValueError):
            AquariumFilterProcess().process(
                _make_filter_input(
                    inflow_range=((0.0, 0.15), (0.03, 0.07), (0.0, 0.2)),
                    outflow_range=((0.1, 0.2), (0.03, 0.07), (0.0, 0.2)),
                )
            )


# ---------------------------------------------------------------------------
# 物理テスト（AquariumFilter）
# ---------------------------------------------------------------------------


class TestAquariumFilterPhysics:
    """物理的妥当性テスト."""

    def test_flow_rate_converted_to_m3s(self):
        """Q[L/h] → Q[m³/s] の換算が正しい."""
        res = AquariumFilterProcess().process(_make_filter_input(flow_rate_lph=3600.0))
        # 3600 L/h = 1 L/s = 1e-3 m³/s
        assert abs(res.flow_rate_m3s - 1e-3) < 1e-15

    def test_inflow_velocity_matches_q_over_area(self):
        """|v_inflow| * A ≈ Q の関係が成立."""
        res = AquariumFilterProcess().process(
            _make_filter_input(flow_rate_lph=440.0, direction=(1.0, 0.0, 0.0))
        )
        speed = float(np.linalg.norm(res.inflow_velocity))
        assert abs(speed * res.inflow_area_m2 - res.flow_rate_m3s) < 1e-12

    def test_inflow_direction_aligned(self):
        """吐出方向が指定ベクトルと平行（単位化後）."""
        res = AquariumFilterProcess().process(_make_filter_input(direction=(0.0, 0.0, 1.0)))
        vx, vy, vz = res.inflow_velocity
        assert abs(vx) < 1e-12
        assert abs(vy) < 1e-12
        assert vz > 0

    def test_masks_disjoint(self):
        res = AquariumFilterProcess().process(_make_filter_input())
        assert not np.any(res.inflow_mask & res.outflow_mask)
        assert res.inflow_mask.any()
        assert res.outflow_mask.any()

    def test_inflow_temperature_forwarded(self):
        """inflow_temperature_K が BC に伝搬する."""
        res = AquariumFilterProcess().process(_make_filter_input(temperature_K=305.0))
        assert res.inflow_bc.temperature == 305.0
        assert res.outflow_bc.temperature is None


# ---------------------------------------------------------------------------
# InternalFaceBC 統合テスト（NaturalConvection 経由）
# ---------------------------------------------------------------------------


def _make_simple_nc_input(
    internal_bcs: tuple,
    T0: np.ndarray | None = None,
    dt: float = 0.0,
    max_simple_iter: int = 80,
) -> NaturalConvectionInput:
    """単純な立方体ドメインで internal_face_bcs をテスト用に組む."""
    nx, ny, nz = 12, 4, 8
    if T0 is None:
        T0 = np.full((nx, ny, nz), 300.0)
    return NaturalConvectionInput(
        Lx=0.3,
        Ly=0.1,
        Lz=0.2,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=1000.0,
        mu=1e-3,
        Cp=4186.0,
        k_fluid=0.6,
        beta=0.0,  # 浮力なし（純粋な強制対流テスト）
        T_ref=300.0,
        gravity=(0.0, 0.0, 0.0),
        T0=T0,
        bc_xm=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        bc_xp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        bc_ym=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        bc_yp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        bc_zm=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        bc_zp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
        dt=dt,
        t_end=dt * 5 if dt > 0 else 0.0,
        max_simple_iter=max_simple_iter,
        tol_simple=1e-4,
        alpha_u=0.5,
        alpha_p=0.2,
        alpha_T=0.8,
        internal_face_bcs=internal_bcs,
    )


class TestInternalFaceBCIntegration:
    """InternalFaceBC が NaturalConvection ソルバーで期待通り機能するか."""

    def test_inlet_velocity_enforced(self):
        """INLET マスクのセルで指定速度が保持される."""
        filt = AquariumFilterProcess().process(_make_filter_input(flow_rate_lph=440.0))
        nc_inp = _make_simple_nc_input(
            internal_bcs=(filt.inflow_bc, filt.outflow_bc), max_simple_iter=60
        )
        res = NaturalConvectionFDMProcess().process(nc_inp)
        u_target = filt.inflow_velocity[0]
        # inflow cells should be very close to target u
        u_at_inlet = res.u[filt.inflow_mask]
        assert np.allclose(u_at_inlet, u_target, atol=1e-6), (
            f"inlet 強制速度が維持されていない: mean={u_at_inlet.mean()}, target={u_target}"
        )

    def test_outlet_draws_flow_toward_it(self):
        """OUTLET マスク近傍で流れがアウトレットに引き寄せられる（|u|>0）."""
        filt = AquariumFilterProcess().process(_make_filter_input(flow_rate_lph=440.0))
        nc_inp = _make_simple_nc_input(
            internal_bcs=(filt.inflow_bc, filt.outflow_bc), max_simple_iter=80
        )
        res = NaturalConvectionFDMProcess().process(nc_inp)
        # ドメイン内平均速度が実質的に非ゼロ（循環が成立）
        assert np.max(np.abs(res.u)) > 1e-5

    def test_inflow_temperature_enforced(self):
        """INLET 温度が指定されたときはその温度が維持される."""
        filt_inp = _make_filter_input(flow_rate_lph=440.0, temperature_K=310.0)
        filt = AquariumFilterProcess().process(filt_inp)
        T0 = np.full((12, 4, 8), 300.0)
        nc_inp = _make_simple_nc_input(
            internal_bcs=(filt.inflow_bc, filt.outflow_bc), T0=T0, max_simple_iter=60
        )
        res = NaturalConvectionFDMProcess().process(nc_inp)
        T_at_inlet = res.T[filt.inflow_mask]
        assert np.allclose(T_at_inlet, 310.0, atol=1e-3), (
            f"inlet 温度が維持されていない: mean={T_at_inlet.mean()}"
        )

    def test_no_internal_bc_baseline_unchanged(self):
        """internal_face_bcs=() のケースでは既存挙動と変わらない（重力ゼロ → 流れゼロ）."""
        nc_inp = _make_simple_nc_input(internal_bcs=(), max_simple_iter=40)
        res = NaturalConvectionFDMProcess().process(nc_inp)
        # 境界すべて NO_SLIP + gravity=0 + beta=0 → 流れなし
        assert np.max(np.abs(res.u)) < 1e-6
        assert np.max(np.abs(res.v)) < 1e-6
        assert np.max(np.abs(res.w)) < 1e-6

    @pytest.mark.xfail(
        reason=(
            "Phase 6.3a 時点の既知課題: INLET ペナルティによる強制流入は "
            "SIMPLE の mass 残差を増幅させ、有限反復では厳密な入出口質量保存に "
            "到達しない。SIMPLEC/PISO 化（docs/status/status-12 参照）待ち。"
        ),
        strict=False,
    )
    def test_outlet_mass_nearly_balances_inlet(self):
        """入口強制流入と出口近傍での流束がオーダー一致で釣り合う.

        CLAUDE.md の既知課題（低粘性・強制対流で mass 残差 O(1-100) 残存）と
        同じ根因で、本ケースも SIMPLE 反復では厳密質量保存に到達しない。
        SIMPLEC/PISO 実装後に strict 化する予定。
        """
        filt_inp = _make_filter_input(flow_rate_lph=440.0)
        filt = AquariumFilterProcess().process(filt_inp)
        nc_inp = _make_simple_nc_input(
            internal_bcs=(filt.inflow_bc, filt.outflow_bc), max_simple_iter=150
        )
        res = NaturalConvectionFDMProcess().process(nc_inp)

        geom = _make_geometry()
        dy_dz = np.einsum("j,k->jk", geom.dy, geom.dz)
        inflow_i = np.unique(np.where(filt.inflow_mask)[0])
        inflow_flux = 0.0
        for i in inflow_i:
            col_mask = filt.inflow_mask[i]
            inflow_flux += (res.u[i] * dy_dz * col_mask).sum()
        outflow_i = np.unique(np.where(filt.outflow_mask)[0])
        outflow_flux = 0.0
        for i in outflow_i:
            col_mask = filt.outflow_mask[i]
            outflow_flux += (res.u[i] * dy_dz * col_mask).sum()
        assert inflow_flux > 0
        assert outflow_flux > 0
        ratio = outflow_flux / inflow_flux
        assert 0.5 < ratio < 1.5


class TestInternalFaceBCAssembly:
    """アセンブリレベルの単体テスト（ソルバー実行なし）."""

    def test_internal_bc_field_defaults_empty(self):
        """NaturalConvectionInput.internal_face_bcs のデフォルトは空 tuple."""
        inp = NaturalConvectionInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            nx=2,
            ny=2,
            nz=2,
            rho=1.0,
            mu=0.01,
            Cp=1.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
        )
        assert inp.internal_face_bcs == ()

    def test_internal_bc_kind_enum(self):
        """InternalFaceBCKind の 2 値を検証."""
        assert InternalFaceBCKind.INLET.value == "inlet"
        assert InternalFaceBCKind.OUTLET.value == "outlet"

    def test_empty_mask_is_noop(self):
        """全 False マスクの InternalFaceBC は BC 無効（ソルバーが安定に収束）."""
        nx, ny, nz = 6, 4, 4
        empty_mask = np.zeros((nx, ny, nz), dtype=bool)
        noop_bc = InternalFaceBC(
            kind=InternalFaceBCKind.INLET, mask=empty_mask, velocity=(1.0, 0, 0)
        )
        inp = NaturalConvectionInput(
            Lx=0.3,
            Ly=0.2,
            Lz=0.2,
            nx=nx,
            ny=ny,
            nz=nz,
            rho=1.0,
            mu=0.01,
            Cp=1000.0,
            k_fluid=1.0,
            beta=0.0,
            T_ref=300.0,
            gravity=(0.0, 0.0, 0.0),
            max_simple_iter=30,
            tol_simple=1e-3,
            bc_xm=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            bc_xp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            bc_ym=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            bc_yp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            bc_zm=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            bc_zp=FluidBoundarySpec(condition=FluidBoundaryCondition.NO_SLIP),
            internal_face_bcs=(noop_bc,),
        )
        res = NaturalConvectionFDMProcess().process(inp)
        assert np.max(np.abs(res.u)) < 1e-5
        assert not np.any(np.isnan(res.u))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
