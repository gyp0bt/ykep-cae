"""ykep .inp フォーマット: ソルバー Input へのマッピングのテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.darcy.data import DarcyBCKind
from xkep_cae_fluid.fvm import BCKind
from xkep_cae_fluid.heat_transfer.data import BoundaryCondition as HTBC
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.grid import recover_structured_grid
from xkep_cae_fluid.inp.mapping import (
    InpMappingInput,
    InpMeshMappingInput,
    InpToDarcyProcess,
    InpToHeatTransferFVMProcess,
    InpToHeatTransferProcess,
    InpToNaturalConvectionProcess,
    UnsupportedFeatureError,
)
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess
from xkep_cae_fluid.inp.parser import parse_inp_text
from xkep_cae_fluid.natural_convection.data import FluidBoundaryCondition, ThermalBoundaryCondition

NS_HEAD = """\
*GRID, NX=4, NY=3, NZ=2, LX=0.4, LY=0.3, LZ=0.2
*ELSET, ELSET=SOLID_CELLS
 1, 2
*ELSET, ELSET=FLUID_CELLS, GENERATE
 3, 24
*ELSET, ELSET=HEATER
 24
*MATERIAL, NAME=AIR
*DENSITY
 1.2
*VISCOSITY
 1.8e-5
*SPECIFIC HEAT
 1005.
*CONDUCTIVITY
 0.026
*EXPANSION, ZERO=293.
 3.4e-3
*MATERIAL, NAME=CU
*CONDUCTIVITY
 400.
*FLUID SECTION, ELSET=FLUID_CELLS, MATERIAL=AIR
*SOLID SECTION, ELSET=SOLID_CELLS, MATERIAL=CU
*INITIAL CONDITIONS, TYPE=TEMPERATURE
 ALL, 293.
 HEATER, 320.
"""

NS_STEP = """\
*STEP
*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=COUPLED
*DLOAD
 ALL, GRAV, 9.81, 0, 0, -1
*BOUNDARY, TYPE=VELOCITY
 XM, 0.5, 0, 0
*BOUNDARY, TYPE=PRESSURE
 XP, 0.
*BOUNDARY, TYPE=SYMMETRY
 YM
*BOUNDARY, TYPE=OUTLET
 YP
*BOUNDARY, TYPE=TEMPERATURE
 XM, 300.
*DFLUX
 ZP, S, 50.
 HEATER, BF, 1e4
*CONTROLS, PARAMETERS=DISCRETIZATION
 CONVECTION=SUPERBEE, TIME=BDF2, PRESSURE_VELOCITY=PISO, PISO_CORRECTORS=3
*CONTROLS, PARAMETERS=RELAXATION
 VELOCITY=0.6, PRESSURE=0.25, TEMPERATURE=0.8, ADAPTIVE=YES
*CONTROLS, PARAMETERS=SOLVER
 PRESSURE=AMG, MAX_OUTER=77, MAX_INNER=30, TOL=1e-3, TOL_INNER=1e-7, MAX_PRESSURE_ITER=120
*CONTROLS, PARAMETERS=TIME INCREMENTATION
 OUTPUT_INTERVAL=4
*OUTPUT, FIELD
*END STEP
"""


def _mapping_input(text: str) -> InpMappingInput:
    case = build_case(parse_inp_text(text))
    return InpMappingInput(case=case, grid=recover_structured_grid(case))


@binds_to(InpToNaturalConvectionProcess)
class TestInpToNaturalConvectionAPI:
    def test_full_mapping(self):
        inp = InpToNaturalConvectionProcess().execute(_mapping_input(NS_HEAD + NS_STEP))
        assert (inp.nx, inp.ny, inp.nz) == (4, 3, 2)
        assert (inp.Lx, inp.Ly, inp.Lz) == pytest.approx((0.4, 0.3, 0.2))
        assert (inp.rho, inp.mu, inp.Cp, inp.k_fluid) == (1.2, 1.8e-5, 1005.0, 0.026)
        assert inp.beta == pytest.approx(3.4e-3) and inp.T_ref == 293.0
        assert inp.gravity == pytest.approx((0.0, 0.0, -9.81))
        # 固体マスク: 要素 1, 2 = (0,0,0), (1,0,0)
        assert inp.solid_mask is not None and inp.solid_mask.sum() == 2
        assert inp.solid_mask[0, 0, 0] and inp.solid_mask[1, 0, 0]
        assert inp.k_solid[0, 0, 0] == 400.0 and inp.k_solid[2, 0, 0] == 0.026
        # 初期温度: HEATER（要素 24 = (3,2,1)）だけ 320
        assert inp.T0[3, 2, 1] == 320.0 and inp.T0[0, 0, 0] == 293.0
        assert inp.q_vol is not None and inp.q_vol[3, 2, 1] == 1e4 and inp.q_vol.sum() == 1e4
        # 境界
        assert inp.bc_xm.condition == FluidBoundaryCondition.INLET_VELOCITY
        assert inp.bc_xm.velocity == (0.5, 0.0, 0.0)
        assert (
            inp.bc_xm.thermal == ThermalBoundaryCondition.DIRICHLET
            and inp.bc_xm.temperature == 300.0
        )
        assert inp.bc_xp.condition == FluidBoundaryCondition.OUTLET_PRESSURE
        assert inp.bc_ym.condition == FluidBoundaryCondition.SYMMETRY
        assert inp.bc_yp.condition == FluidBoundaryCondition.OUTLET_CONVECTIVE
        assert inp.bc_zm.condition == FluidBoundaryCondition.NO_SLIP  # 3D の既定
        assert inp.bc_zp.thermal == ThermalBoundaryCondition.NEUMANN and inp.bc_zp.heat_flux == 50.0
        # 離散化・緩和・ソルバー
        assert inp.convection_scheme == "superbee" and inp.time_scheme == "bdf2"
        assert inp.coupling_method == "piso" and inp.n_piso_correctors == 3
        assert (inp.alpha_u, inp.alpha_p, inp.alpha_T) == (
            0.6,
            0.25,
            0.8,
        ) and inp.adaptive_relaxation
        assert inp.pressure_solver == "amg"
        assert (inp.max_simple_iter, inp.max_inner_iter, inp.max_pressure_iter) == (77, 30, 120)
        assert (inp.tol_simple, inp.tol_inner) == (1e-3, 1e-7)
        assert inp.output_interval == 4
        assert inp.dt == 0.0 and not inp.is_transient

    def test_transient_and_defaults(self):
        step = "*STEP, INC=25\n*NAVIER STOKES\n 0.01, 0.5\n*END STEP\n"
        inp = InpToNaturalConvectionProcess().execute(_mapping_input(NS_HEAD + step))
        assert inp.is_transient and inp.dt == 0.01 and inp.t_end == 0.5
        assert inp.max_simple_iter == 25  # *STEP, INC= が MAX_OUTER の既定
        assert inp.convection_scheme == "upwind" and inp.coupling_method == "simple"
        assert inp.gravity == (0.0, 0.0, 0.0)  # *DLOAD, GRAV なし → 無重力
        assert inp.beta == 0.0  # HEAT TRANSFER 省略 = NONE → 等温（浮力なし）
        assert inp.bc_xm.condition == FluidBoundaryCondition.NO_SLIP

    def test_isothermal_ignores_thermal_bcs(self, caplog):
        step = "*STEP\n*NAVIER STOKES, STEADY STATE\n*BOUNDARY, TYPE=TEMPERATURE\n XM, 350.\n*END STEP\n"
        inp = InpToNaturalConvectionProcess().execute(_mapping_input(NS_HEAD + step))
        assert inp.beta == 0.0 and inp.q_vol is None
        assert np.all(inp.T0 == inp.T0.flat[0])
        assert inp.bc_xm.thermal == ThermalBoundaryCondition.ADIABATIC
        assert "無視" in caplog.text

    def test_2d_elements_default_symmetry_in_z(self):
        text = (
            "*NODE\n 1,0,0\n 2,1,0\n 3,2,0\n 4,0,1\n 5,1,1\n 6,2,1\n*ELEMENT, TYPE=CPS4\n 1,1,2,5,4\n 2,2,3,6,5\n"
            "*MATERIAL, NAME=W\n*DENSITY\n 1000.\n*VISCOSITY\n 1e-3\n*FLUID SECTION, ELSET=ALL, MATERIAL=W\n"
            "*STEP\n*NAVIER STOKES, STEADY STATE\n*END STEP\n"
        )
        inp = InpToNaturalConvectionProcess().execute(_mapping_input(text))
        assert inp.nz == 1 and inp.bc_zm.condition == FluidBoundaryCondition.SYMMETRY
        assert inp.bc_zp.condition == FluidBoundaryCondition.SYMMETRY

    @pytest.mark.parametrize(
        ("step", "match"),
        [
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE, TURBULENCE=K-EPSILON\n*END STEP\n",
                "TURBULENCE",
            ),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE\n*CONTROLS, PARAMETERS=DISCRETIZATION\n CONVECTION=QUICK\n*END STEP\n",
                "CONVECTION",
            ),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE\n*CONTROLS, PARAMETERS=DISCRETIZATION\n LIMITER=VENKATAKRISHNAN\n*END STEP\n",
                "LIMITER",
            ),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE\n*CONTROLS, PARAMETERS=RELAXATION\n VELOCTY=0.5\n*END STEP\n",
                "未知のキー",
            ),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE\n*CONTROLS, PARAMETERS=SOLVER\n PRESSURE=GMRES\n*END STEP\n",
                "PRESSURE",
            ),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=COUPLED\n*SFILM\n XM, F, 300., 10.\n*END STEP\n",
                "SFILM",
            ),
            ("*STEP\n*HEAT TRANSFER, STEADY STATE\n*END STEP\n", "NAVIER STOKES 用"),
            (
                "*STEP\n*NAVIER STOKES, STEADY STATE\n*BOUNDARY, TYPE=WALL\n NOPE\n*END STEP\n",
                "SURFACE でも",
            ),
        ],
    )
    def test_unsupported(self, step: str, match: str):
        with pytest.raises(UnsupportedFeatureError, match=match):
            InpToNaturalConvectionProcess().execute(_mapping_input(NS_HEAD + step))

    def test_nonuniform_grid_rejected(self):
        text = (
            "*NODE\n 1,0,0,0\n 2,1,0,0\n 3,3,0,0\n 4,0,1,0\n 5,1,1,0\n 6,3,1,0\n 7,0,0,1\n 8,1,0,1\n 9,3,0,1\n 10,0,1,1\n 11,1,1,1\n 12,3,1,1\n"
            "*ELEMENT, TYPE=C3D8\n 1,1,2,5,4,7,8,11,10\n 2,2,3,6,5,8,9,12,11\n"
            "*MATERIAL, NAME=W\n*DENSITY\n 1.\n*VISCOSITY\n 1.\n*FLUID SECTION, ELSET=ALL, MATERIAL=W\n"
            "*STEP\n*NAVIER STOKES, STEADY STATE\n*END STEP\n"
        )
        with pytest.raises(UnsupportedFeatureError, match="等間隔"):
            InpToNaturalConvectionProcess().execute(_mapping_input(text))

    def test_section_coverage_errors(self):
        head = NS_HEAD.replace("*SOLID SECTION, ELSET=SOLID_CELLS, MATERIAL=CU\n", "")
        with pytest.raises(UnsupportedFeatureError, match="未割当"):
            InpToNaturalConvectionProcess().execute(
                _mapping_input(head + "*STEP\n*NAVIER STOKES, STEADY STATE\n*END STEP\n")
            )


HT_TEXT = """\
*NODE
 1,0,0,0
 2,1,0,0
 3,3,0,0
 4,0,1,0
 5,1,1,0
 6,3,1,0
 7,0,0,1
 8,1,0,1
 9,3,0,1
 10,0,1,1
 11,1,1,1
 12,3,1,1
*ELEMENT, TYPE=C3D8, ELSET=E
 1,1,2,5,4,7,8,11,10
 2,2,3,6,5,8,9,12,11
*NSET, NSET=LEFTN
 1, 4, 7, 10
*ELSET, ELSET=E2
 2
*SURFACE, NAME=LEFT
 1, S6
*MATERIAL, NAME=A
*DENSITY
 2000.
*SPECIFIC HEAT
 500.
*CONDUCTIVITY
 10.
*MATERIAL, NAME=B
*CONDUCTIVITY
 1.
*SOLID SECTION, ELSET=E2, MATERIAL=B
*ELSET, ELSET=E1
 1
*SOLID SECTION, ELSET=E1, MATERIAL=A
*INITIAL CONDITIONS, TYPE=TEMPERATURE
 ALL, 300.
 LEFTN, 400.
*STEP
*HEAT TRANSFER, STEADY STATE
*BOUNDARY
 LEFT, 11, 11, 450.
*DFLUX
 XP, S, 100.
 E2, BF, 7.
*SFILM
 YP, F, 290., 12.
*CONTROLS, PARAMETERS=SOLVER
 METHOD=DIRECT, MAX_ITER=5, TOL=1e-9
*END STEP
"""


@binds_to(InpToHeatTransferProcess)
class TestInpToHeatTransferAPI:
    def test_full_mapping(self):
        mapped = InpToHeatTransferProcess().execute(_mapping_input(HT_TEXT))
        inp = mapped.input
        assert mapped.method == "direct"
        assert (inp.nx, inp.ny, inp.nz) == (2, 1, 1)
        assert inp.k[0, 0, 0] == 10.0 and inp.k[1, 0, 0] == 1.0
        assert inp.C[0, 0, 0] == 2000.0 * 500.0
        assert inp.q[1, 0, 0] == 7.0 and inp.q[0, 0, 0] == 0.0
        # 節点集合 LEFTN の 4 節点は要素 1 の半分 → 要素平均 350
        assert inp.T0[0, 0, 0] == 350.0 and inp.T0[1, 0, 0] == 300.0
        assert inp.bc_xm.condition == HTBC.DIRICHLET and inp.bc_xm.value == 450.0
        assert inp.bc_xp.condition == HTBC.NEUMANN and inp.bc_xp.value == 100.0
        assert inp.bc_yp.condition == HTBC.ROBIN and (inp.bc_yp.h_conv, inp.bc_yp.T_inf) == (
            12.0,
            290.0,
        )
        assert inp.bc_ym.condition == HTBC.ADIABATIC
        assert inp.is_nonuniform and inp.dx_array.tolist() == [1.0, 2.0]
        assert inp.max_iter == 5 and inp.tol == 1e-9 and inp.dt == 0.0

    def test_transient_requires_density_and_cp(self):
        text = HT_TEXT.replace("*HEAT TRANSFER, STEADY STATE\n", "*HEAT TRANSFER\n 0.1, 1.0\n")
        with pytest.raises(ValueError, match="density"):
            InpToHeatTransferProcess().execute(_mapping_input(text))

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [
            (
                "*BOUNDARY\n LEFT, 11, 11, 450.\n",
                "*BOUNDARY, TYPE=VELOCITY\n LEFT, 1, 0, 0\n",
                "使えません",
            ),
            (" METHOD=DIRECT, MAX_ITER=5, TOL=1e-9\n", " METHOD=PARDISO\n", "METHOD"),
            (
                "*HEAT TRANSFER, STEADY STATE\n",
                "*HEAT TRANSFER, STEADY STATE\n*CONTROLS, PARAMETERS=RELAXATION\n VELOCITY=0.5\n",
                "未対応",
            ),
            ("*DFLUX\n XP, S, 100.\n", "*DFLUX\n LEFT, S, 100.\n", "同時"),
        ],
    )
    def test_unsupported(self, old: str, new: str, match: str):
        with pytest.raises(UnsupportedFeatureError, match=match):
            InpToHeatTransferProcess().execute(_mapping_input(HT_TEXT.replace(old, new)))


# ---------------------------------------------------------------------------
# Darcy
# ---------------------------------------------------------------------------

DARCY_TEXT = """\
*GRID, NX=4, NY=2, NZ=2, LX=0.4, LY=0.2, LZ=0.2
*ELSET, ELSET=TIGHT
 1, 2
*ELSET, ELSET=LOOSE, GENERATE
 3, 16
*MATERIAL, NAME=WATER_SAND
*DENSITY
 998.
*VISCOSITY
 1e-3
*PERMEABILITY
 1e-10
*MATERIAL, NAME=CLAY
*PERMEABILITY
 1e-13
*FLUID SECTION, ELSET=LOOSE, MATERIAL=WATER_SAND
*SOLID SECTION, ELSET=TIGHT, MATERIAL=CLAY
*INITIAL CONDITIONS, TYPE=PRESSURE
 ALL, 50.
*STEP
*DARCY, STEADY STATE
*BOUNDARY, TYPE=PRESSURE
 XM, 100.
*BOUNDARY, TYPE=VELOCITY
 XP, -2e-4, 0, 0
*BOUNDARY, TYPE=VELOCITY
 YM, 3e-5
*BOUNDARY, TYPE=SYMMETRY
 ZM
*CONTROLS, PARAMETERS=SOLVER
 METHOD=BICGSTAB, TOL=1e-9, MAX_ITER=50
*END STEP
"""


def _darcy_input(text: str) -> InpMeshMappingInput:
    case = build_case(parse_inp_text(text))
    return InpMeshMappingInput(case=case, mesh=InpMeshProcess().execute(InpMeshInput(case=case)))


@binds_to(InpToDarcyProcess)
class TestInpToDarcyAPI:
    def test_full_mapping(self):
        inp = InpToDarcyProcess().execute(_darcy_input(DARCY_TEXT))
        assert inp.mesh.n_cells == 16 and inp.viscosity == 1e-3 and inp.density == 998.0
        # 要素 1, 2 が CLAY
        k = np.asarray(inp.permeability)
        assert np.sum(k == 1e-13) == 2 and np.sum(k == 1e-10) == 14
        assert np.all(inp.p0 == 50.0)
        assert inp.bcs["XM"].kind == DarcyBCKind.PRESSURE and inp.bcs["XM"].pressure == 100.0
        # XP の外向き法線は +x。速度 (-2e-4, 0, 0) は領域へ向かう → 内向き法線成分 +2e-4（流入）
        assert inp.bcs["XP"].kind == DarcyBCKind.VELOCITY
        assert inp.bcs["XP"].velocity == pytest.approx(2e-4)
        assert inp.bcs["YM"].velocity == pytest.approx(3e-5)
        assert inp.bcs["ZM"].kind == DarcyBCKind.WALL
        assert (inp.linear_solver, inp.tol, inp.max_iter) == ("bicgstab", 1e-9, 50)

    @pytest.mark.parametrize(
        "old,new,match",
        [
            ("*DARCY, STEADY STATE", "*DARCY\n 0.1, 1.0", "定常"),
            ("*BOUNDARY, TYPE=SYMMETRY\n ZM", "*BOUNDARY, TYPE=TEMPERATURE\n ZM, 300.", "DARCY"),
            ("*BOUNDARY, TYPE=SYMMETRY\n ZM", "*BOUNDARY, TYPE=SYMMETRY\n LID", "予約面名"),
            ("*PERMEABILITY\n 1e-13", "*CONDUCTIVITY\n 1.", "permeability"),
            ("METHOD=BICGSTAB", "METHOD=NUMBA", "METHOD"),
            (
                "*CONTROLS, PARAMETERS=SOLVER",
                "*CONTROLS, PARAMETERS=RELAXATION\n VELOCITY=0.5\n*CONTROLS, PARAMETERS=SOLVER",
                "RELAXATION",
            ),
        ],
    )
    def test_unsupported(self, old: str, new: str, match: str):
        assert old in DARCY_TEXT
        with pytest.raises((UnsupportedFeatureError, ValueError), match=match):
            InpToDarcyProcess().execute(_darcy_input(DARCY_TEXT.replace(old, new)))


HT_SKEW_TEXT = """\
*HEADING
 heat transfer on a box mesh via the unstructured path
*GRID, NX=4, NY=2, NZ=2, LX=0.4, LY=0.2, LZ=0.2
*MATERIAL, NAME=M
*DENSITY
 1000.
*SPECIFIC HEAT
 2.
*CONDUCTIVITY
 5.
*SOLID SECTION, ELSET=ALL, MATERIAL=M
*INITIAL CONDITIONS, TYPE=TEMPERATURE
 ALL, 310.
*STEP
*HEAT TRANSFER, STEADY STATE
*BOUNDARY, TYPE=TEMPERATURE
 XM, 400.
*DFLUX
 XP, S, 250.
 ALL, BF, 1000.
*SFILM
 ZP, F, 290., 12.
*CONTROLS, PARAMETERS=SOLVER
 METHOD=BICGSTAB, TOL=1e-9, MAX_ITER=50
*END STEP
"""


@binds_to(InpToHeatTransferFVMProcess)
class TestInpToHeatTransferFVMAPI:
    def test_full_mapping(self):
        inp = InpToHeatTransferFVMProcess().execute(_darcy_input(HT_SKEW_TEXT))
        assert inp.mesh.n_cells == 16
        assert np.all(np.asarray(inp.conductivity) == 5.0)
        assert np.all(np.asarray(inp.heat_capacity) == 2000.0)
        assert np.all(inp.T0 == 310.0) and np.all(inp.heat_source == 1000.0)
        assert inp.bcs["XM"].kind == BCKind.DIRICHLET and inp.bcs["XM"].value == 400.0
        assert inp.bcs["XP"].kind == BCKind.NEUMANN and inp.bcs["XP"].flux == 250.0
        assert inp.bcs["ZP"].kind == BCKind.ROBIN and (inp.bcs["ZP"].h, inp.bcs["ZP"].phi_inf) == (
            12.0,
            290.0,
        )
        assert set(inp.bcs) == {"XM", "XP", "ZP"}
        assert (inp.linear_solver, inp.tol, inp.max_iter) == ("bicgstab", 1e-9, 50)
        assert inp.dt == 0.0

    @pytest.mark.parametrize(
        "old,new,match",
        [
            (
                "*BOUNDARY, TYPE=TEMPERATURE\n XM, 400.",
                "*BOUNDARY, TYPE=TEMPERATURE\n LID, 400.",
                "予約面名",
            ),
            ("*BOUNDARY, TYPE=TEMPERATURE\n XM, 400.", "*BOUNDARY, TYPE=WALL\n XM", "TEMPERATURE"),
            (" XP, S, 250.", " XM, S, 250.", "同時"),
            ("METHOD=BICGSTAB", "METHOD=JACOBI", "METHOD"),
            (
                "*CONTROLS, PARAMETERS=SOLVER",
                "*CONTROLS, PARAMETERS=RELAXATION\n TEMPERATURE=0.5\n*CONTROLS, PARAMETERS=SOLVER",
                "RELAXATION",
            ),
        ],
    )
    def test_unsupported(self, old: str, new: str, match: str):
        assert old in HT_SKEW_TEXT
        with pytest.raises(UnsupportedFeatureError, match=match):
            InpToHeatTransferFVMProcess().execute(_darcy_input(HT_SKEW_TEXT.replace(old, new)))

    def test_transient_needs_density_and_specific_heat(self):
        text = HT_SKEW_TEXT.replace("*HEAT TRANSFER, STEADY STATE", "*HEAT TRANSFER\n 1., 10.")
        inp = InpToHeatTransferFVMProcess().execute(_darcy_input(text))
        assert inp.dt == 1.0 and inp.t_end == 10.0
        with pytest.raises(ValueError, match="density"):
            InpToHeatTransferFVMProcess().execute(
                _darcy_input(text.replace("*DENSITY\n 1000.\n", ""))
            )
