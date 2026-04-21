"""NaturalConvection + ExtraScalar 統合テスト（Phase 6.1b）.

`NaturalConvectionInput.extra_scalars` 経由で追加スカラー（トレーサー等）を
SIMPLE 外部反復内で温度と同時輸送する機能を検証する。

- API: `NaturalConvectionResult.extra_scalars` に指定した名前の場が入る
- 物理: 閉じた系（全面 No-Slip + Adiabatic + ゼロソース）で総量が保存される
- 物理: 温度場と同じ流れ場でトレーサーが運ばれる
"""

import numpy as np
import pytest

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess
from xkep_cae_fluid.scalar_transport.data import (
    ExtraScalarSpec,
    ScalarFieldSpec,
)


def _make_base_input(
    nx: int = 6,
    ny: int = 6,
    nz: int = 3,
    dt: float = 0.0,
    t_end: float = 0.0,
    max_simple_iter: int = 20,
    T0: np.ndarray | None = None,
    extra_scalars: tuple[ExtraScalarSpec, ...] = (),
    wall_dirichlet: bool = False,
) -> NaturalConvectionInput:
    """差分加熱キャビティ（左右 Dirichlet / 上下断熱）の入力を組み立てる.

    wall_dirichlet=False のときは全面断熱＋左右のみ温度差で自然対流を誘起する。
    T0 が None のときは一様 T_ref とし、左右に温度勾配を与える。
    """
    T_ref = 300.0
    if T0 is None:
        T0 = np.full((nx, ny, nz), T_ref)
    # 左壁 301K, 右壁 299K（温度勾配で自然対流）
    bc_xm = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref + 1.0,
    )
    bc_xp = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref - 1.0,
    )
    bc_other = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )
    return NaturalConvectionInput(
        Lx=0.05,
        Ly=0.05,
        Lz=0.025,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=1.0,
        mu=0.01,
        Cp=1000.0,
        k_fluid=1.0,
        beta=0.001,
        T_ref=T_ref,
        gravity=(0.0, -9.81, 0.0),
        T0=T0,
        bc_xm=bc_xm,
        bc_xp=bc_xp,
        bc_ym=bc_other,
        bc_yp=bc_other,
        bc_zm=bc_other,
        bc_zp=bc_other,
        dt=dt,
        t_end=t_end,
        max_simple_iter=max_simple_iter,
        max_inner_iter=50,
        tol_simple=1e-4,
        tol_inner=1e-8,
        alpha_u=0.7,
        alpha_p=0.3,
        alpha_T=0.9,
        extra_scalars=extra_scalars,
    )


class TestNaturalConvectionScalarAPI:
    """extra_scalars の API 契約."""

    def test_extra_scalars_default_is_empty(self):
        """extra_scalars を指定しないときは結果 dict も空."""
        inp = _make_base_input(max_simple_iter=3)
        res = NaturalConvectionFDMProcess().process(inp)
        assert res.extra_scalars == {}

    def test_extra_scalar_appears_in_result(self):
        """指定したスカラー名が結果 dict に含まれ、形状が一致する."""
        nx, ny, nz = 5, 5, 3
        phi0 = np.ones((nx, ny, nz)) * 1.0
        spec = ExtraScalarSpec(
            field=ScalarFieldSpec(name="tracer", diffusivity=1e-5, phi0=phi0),
        )
        inp = _make_base_input(
            nx=nx,
            ny=ny,
            nz=nz,
            max_simple_iter=5,
            extra_scalars=(spec,),
        )
        res = NaturalConvectionFDMProcess().process(inp)
        assert "tracer" in res.extra_scalars
        assert res.extra_scalars["tracer"].shape == (nx, ny, nz)

    def test_residual_history_contains_scalar_key(self):
        """追加スカラー名 `phi_<name>` のキーが残差履歴に出現する."""
        nx, ny, nz = 5, 5, 3
        phi0 = np.ones((nx, ny, nz))
        spec = ExtraScalarSpec(
            field=ScalarFieldSpec(name="co2", diffusivity=1e-5, phi0=phi0),
        )
        inp = _make_base_input(
            nx=nx,
            ny=ny,
            nz=nz,
            max_simple_iter=3,
            extra_scalars=(spec,),
        )
        res = NaturalConvectionFDMProcess().process(inp)
        assert "phi_co2" in res.residual_history
        assert len(res.residual_history["phi_co2"]) >= 1


class TestNaturalConvectionScalarPhysics:
    """温度と同時輸送されるトレーサーの物理的妥当性."""

    def test_closed_domain_tracer_mass_conservation(self):
        """全面断熱＋ゼロソース＋自然対流で、トレーサー総量が保存される.

        差分加熱（左右 Dirichlet）で自然対流を誘起するが、トレーサーは
        全面 Adiabatic・ソースなしなので合計濃度が不変でなければならない。
        """
        nx, ny, nz = 6, 6, 3
        # 左側半分に初期濃度を偏らせる（トレーサーの運搬を観測しやすく）
        phi0 = np.zeros((nx, ny, nz))
        phi0[: nx // 2, :, :] = 1.0
        spec = ExtraScalarSpec(
            field=ScalarFieldSpec(name="tracer", diffusivity=1e-6, phi0=phi0),
        )
        total_initial = float(phi0.sum())

        inp = _make_base_input(
            nx=nx,
            ny=ny,
            nz=nz,
            dt=0.05,
            t_end=0.1,  # 2 タイムステップ
            max_simple_iter=30,
            extra_scalars=(spec,),
        )
        res = NaturalConvectionFDMProcess().process(inp)
        total_final = float(res.extra_scalars["tracer"].sum())
        # 全面 Adiabatic なので出入りは 0、セル数ベースの相対誤差 < 1%
        rel_err = abs(total_final - total_initial) / max(total_initial, 1e-30)
        assert rel_err < 1e-2, f"total changed: {total_initial} -> {total_final}"

    def test_tracer_gets_redistributed_by_flow(self):
        """自然対流で非一様な初期分布のトレーサーが運搬される.

        定常 SIMPLE では速度場が立ち上がる必要があるので非定常で数ステップ進める。
        """
        nx, ny, nz = 6, 6, 3
        phi0 = np.zeros((nx, ny, nz))
        phi0[0, :, :] = 1.0  # 左壁側（高温側）に集中
        spec = ExtraScalarSpec(
            field=ScalarFieldSpec(name="tracer", diffusivity=1e-6, phi0=phi0),
        )
        inp = _make_base_input(
            nx=nx,
            ny=ny,
            nz=nz,
            dt=0.05,
            t_end=0.25,  # 5 タイムステップ
            max_simple_iter=30,
            extra_scalars=(spec,),
        )
        res = NaturalConvectionFDMProcess().process(inp)
        phi_final = res.extra_scalars["tracer"]
        # 何らかの運搬が起きて分布が一様化されてはいないが、最左列以外にも
        # 伝搬していることを確認（数値拡散と対流を含む）。
        non_left_mass = float(phi_final[1:, :, :].sum())
        assert non_left_mass > 1e-3, "tracer did not spread from source column"


class TestNaturalConvectionScalarMultiple:
    """複数スカラーの同時輸送."""

    def test_two_scalars_transported(self):
        """2 つの独立なスカラーが両方とも結果に入る."""
        nx, ny, nz = 4, 4, 3
        phi_a = np.ones((nx, ny, nz)) * 0.5
        phi_b = np.zeros((nx, ny, nz))
        phi_b[-1, :, :] = 1.0
        specs = (
            ExtraScalarSpec(field=ScalarFieldSpec(name="O2", diffusivity=1e-5, phi0=phi_a)),
            ExtraScalarSpec(field=ScalarFieldSpec(name="CO2", diffusivity=1e-5, phi0=phi_b)),
        )
        inp = _make_base_input(
            nx=nx,
            ny=ny,
            nz=nz,
            max_simple_iter=5,
            extra_scalars=specs,
        )
        res = NaturalConvectionFDMProcess().process(inp)
        assert set(res.extra_scalars.keys()) == {"O2", "CO2"}
        assert res.extra_scalars["O2"].shape == (nx, ny, nz)
        assert res.extra_scalars["CO2"].shape == (nx, ny, nz)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
