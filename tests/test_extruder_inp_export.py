"""ExtruderChannelInpProcess（展開チャネル 2.5D → 汎用記法 .inp）のテスト.

汎用経路（``*NODE`` / ``*ELEMENT`` + ``*NAVIER STOKES``）が、専用の 2.5D ソルバー
（:class:`~xkep_cae_fluid.extruder.solver.ExtruderFlowProcess`）と解析解（形状係数）を
再現することを確かめる。専用ソルバーは既にゲート G1〜G5 を通しているので、
これがそのまま汎用経路のリファレンスになる。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ScrewSpec
from xkep_cae_fluid.extruder.inp_export import (
    ExtruderChannelInpProcess,
    ExtruderInpInput,
    ExtruderInpResult,
    axial_throughput,
)
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.fvm.viscosity import CarreauViscosity, NewtonianViscosity, PowerLawViscosity
from xkep_cae_fluid.incompressible import NavierStokesFVMProcess
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mapping import InpMeshMappingInput, InpToNavierStokesFVMProcess
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess
from xkep_cae_fluid.inp.parser import parse_inp_text

MU = 1000.0


def _spec(f: int = 1, delta: float = 2.0e-4) -> ScrewSpec:
    """40 mm 機の計量部（設計文書 §6 の諸元。解像度は ``f`` 倍）."""
    return ScrewSpec(
        D=0.04,
        lead=0.04,
        H=0.004,
        e=0.004,
        delta=delta,
        N=1.0,
        nx_channel=40 * f,
        nx_land=8 * f,
        ny_bulk=20 * f,
        n_gap=8 * f if delta > 0 else 0,
    )


def _export(spec: ScrewSpec, G: float, **kw) -> ExtruderInpResult:
    return ExtruderChannelInpProcess().execute(
        ExtruderInpInput(
            spec=spec, G=G, viscosity=kw.pop("viscosity", NewtonianViscosity(MU)), **kw
        )
    )


def _solve(res: ExtruderInpResult) -> tuple[float, float, float]:
    """書き出した .inp をそのまま非構造 NS で解いて (Q, Q_leak, Q_axial) を返す."""
    case = build_case(parse_inp_text(res.text))
    mesh = InpMeshProcess().execute(InpMeshInput(case=case))
    ns = InpToNavierStokesFVMProcess().execute(
        InpMeshMappingInput(case=case, mesh=mesh, step_index=0)
    )
    out = NavierStokesFVMProcess().execute(ns)
    assert out.converged, "汎用経路が収束しませんでした"
    return axial_throughput(out.velocity, mesh.mesh.cell_volumes, res.depth_z, res.grid.spec)


def _exact_Q(spec: ScrewSpec, mu: float, G: float) -> float:
    h = spec.H / spec.W
    return metering_flow_rate(
        spec.w_barrel,
        spec.W,
        spec.H,
        mu,
        G,
        F_d=shape_factor_drag(h),
        F_p=shape_factor_pressure(h),
    )


@binds_to(ExtruderChannelInpProcess)
class TestExtruderChannelInpAPI:
    def test_meta(self):
        meta = ExtruderChannelInpProcess.meta
        assert meta.name == "ExtruderChannelInp" and meta.module == "pre"

    def test_writes_expected_keywords(self):
        res = _export(_spec(), 1.0e5)
        text = res.text
        for kw in (
            "*NODE",
            "*ELEMENT, TYPE=C3D8",
            "*SURFACE, NAME=BARREL",
            "*BOUNDARY, TYPE=PERIODIC",
            "*DLOAD",
            "*NAVIER STOKES, STEADY STATE",
            "CONVECTION=NONE",
            "PRESSURE_VELOCITY=COUPLED",
        ):
            assert kw in text, kw
        assert text.count("*BOUNDARY, TYPE=PERIODIC") == 2  # x 1 ピッチ と z 1 セル厚
        assert res.n_cells > 0 and res.depth_z == pytest.approx(res.grid.spec.H / 10.0)

    def test_body_force_is_the_pressure_jump_decomposition(self):
        spec, G = _spec(), 1.0e5
        res = _export(spec, G)
        # P = βx + p̃ の分解（設計文書 §2.1）: f = (−G cotφ, 0, −G)
        assert res.body_force == pytest.approx((-spec.beta(G), 0.0, -G))
        assert res.barrel_velocity == pytest.approx((spec.u_barrel, 0.0, spec.w_barrel))

    def test_flight_cells_are_removed(self):
        spec = _spec()
        res = _export(spec, 0.0)
        assert res.n_cells == int((~res.grid.solid).sum())
        assert res.n_cells < res.grid.nx * res.grid.ny

    @pytest.mark.parametrize(
        "model,keyword",
        [
            (NewtonianViscosity(MU), "*VISCOSITY\n"),
            (PowerLawViscosity(K=5000.0, n=0.4), "*VISCOSITY, TYPE=POWER LAW"),
            (CarreauViscosity(mu_0=1e4, mu_inf=10.0, lam=1.0, n=0.5), "*VISCOSITY, TYPE=CARREAU"),
        ],
    )
    def test_viscosity_keywords(self, model, keyword: str):
        assert keyword in _export(_spec(), 0.0, viscosity=model).text

    def test_rejects_unknown_viscosity_model(self):
        class _Bad:
            def viscosity(self, gamma_dot):
                return np.ones_like(gamma_dot)

        with pytest.raises(ValueError, match="\\.inp に書けません"):
            _export(_spec(), 0.0, viscosity=_Bad())

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="depth_z"):
            _export(_spec(), 0.0, depth_z=-1.0)

    def test_roundtrip_parses_into_a_periodic_mesh(self):
        res = _export(_spec(), 1.0e5)
        case = build_case(parse_inp_text(res.text))
        mesh = InpMeshProcess().execute(InpMeshInput(case=case))
        assert mesh.mesh.n_cells == res.n_cells
        assert mesh.periodic_surfaces == ("XPER0", "XPER1", "ZPER0", "ZPER1")
        assert mesh.mesh.has_periodic_faces
        assert "BARREL" in (mesh.mesh.boundary_patches or {})


class TestExtruderGenericPathPhysics:
    """汎用記法で書いた展開チャネルが解析解・専用ソルバーと一致する."""

    def test_g1_drag_flow_matches_shape_factor(self):
        """G1: 閉チャネル（隙間なし）の純引きずり流れ Q = V H W F_d / 2."""
        spec = _spec(delta=0.0)
        q, _, _ = _solve(_export(spec, 0.0))
        assert q == pytest.approx(_exact_Q(spec, MU, 0.0), rel=3e-3)

    @pytest.mark.parametrize("G", [5.0e4, 1.0e5])
    def test_g2_pressure_flow_matches_shape_factor(self, G: float):
        """G2: 背圧を加えた直線関係 Q = (VHW/2)F_d − (WH³/12μ)G F_p."""
        spec = _spec(delta=0.0)
        q, _, _ = _solve(_export(spec, G))
        assert q == pytest.approx(_exact_Q(spec, MU, G), rel=3e-3)

    def test_clearance_matches_specialised_solver(self):
        """隙間あり（解析解なし）: 専用 2.5D ソルバーと押出量・漏れ量が一致する."""
        spec = _spec()
        ref_proc = ExtruderFlowProcess()
        ref_proc.viscosity = NewtonianViscosity(MU)
        ref = ref_proc.execute(ExtruderFlowInput(spec=spec, G=1.0e5))
        assert ref.converged
        q, q_leak, q_axial = _solve(_export(spec, 1.0e5))
        # 下流方向 w は同じ可変係数 Poisson になるので機械精度で一致する
        assert q == pytest.approx(ref.Q, rel=1e-10)
        # 断面内は MAC 千鳥格子（専用）と Rhie–Chow 同位置格子（汎用）で離散化が違う
        assert q_leak == pytest.approx(ref.Q_leak, rel=6e-2)
        assert q_axial == pytest.approx(ref.Q_axial, rel=5e-3)
        assert q_leak < 0.0  # 漏れは上流側（−x）へ

    def test_clearance_agreement_improves_with_refinement(self):
        """漏れ量の差は離散化差なので、格子を細かくすると縮む."""
        errors = []
        for f in (1, 2):
            spec = _spec(f)
            ref_proc = ExtruderFlowProcess()
            ref_proc.viscosity = NewtonianViscosity(MU)
            ref = ref_proc.execute(ExtruderFlowInput(spec=spec, G=1.0e5))
            _, q_leak, _ = _solve(_export(spec, 1.0e5))
            errors.append(abs(q_leak - ref.Q_leak) / abs(ref.Q_leak))
        assert errors[1] < 0.6 * errors[0], errors

    def test_power_law_runs_and_thins_where_shear_is_high(self):
        """べき乗則: 汎用経路でも Picard が収束し、バレル直下で粘度が下がる."""
        spec = _spec()
        res = _export(spec, 1.0e5, viscosity=PowerLawViscosity(K=5000.0, n=0.5))
        case = build_case(parse_inp_text(res.text))
        mesh = InpMeshProcess().execute(InpMeshInput(case=case))
        ns = InpToNavierStokesFVMProcess().execute(
            InpMeshMappingInput(case=case, mesh=mesh, step_index=0)
        )
        out = NavierStokesFVMProcess().execute(ns)
        assert out.converged and out.viscosity is not None
        y = mesh.mesh.cell_centers[:, 1]
        near_barrel = y > spec.H - 1.5 * spec.delta
        assert out.viscosity[near_barrel].mean() < out.viscosity[~near_barrel].mean()
        q, _, q_axial = axial_throughput(out.velocity, mesh.mesh.cell_volumes, res.depth_z, spec)
        assert q > 0.0 and q_axial > 0.0
