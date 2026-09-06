"""ykep .inp フォーマット: トークナイザ（InpKeywordParseProcess）と組み立て（InpCaseBuildProcess）のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.inp.builder import InpCaseBuildProcess, build_case, generate_grid
from xkep_cae_fluid.inp.case import (
    BoundaryKind,
    CaseDefinition,
    ControlCategory,
    EquationFamily,
    FluxLabel,
    InitialConditionKind,
    OutputFormat,
    SectionKind,
)
from xkep_cae_fluid.inp.parameters import ParameterError, safe_eval, substitute
from xkep_cae_fluid.inp.parser import (
    InpKeywordParseProcess,
    InpParseInput,
    InpSyntaxError,
    parse_inp_text,
    parse_keyword_line,
)

MINIMAL_CASE = """\
*HEADING
 minimal case
 second line
*PARAMETER
 n = 3
 L = 0.3
 dx = L / n
*GRID, NX=<n>, NY=<n>, NZ=1, LX=<L>, LY=<L>, LZ=<dx>
*MATERIAL, NAME=Water
*DENSITY
 1000.0
*VISCOSITY
 1e-3
*CONDUCTIVITY
 0.6
*SPECIFIC HEAT
 4180.0
*EXPANSION, ZERO=300.0
 2.1e-4
*FLUID SECTION, ELSET=ALL, MATERIAL=water
*INITIAL CONDITIONS, TYPE=TEMPERATURE
 ALL, 300.0
*STEP, NAME=Flow, INC=50
*NAVIER STOKES, TURBULENCE=LAMINAR, STEADY STATE, HEAT TRANSFER=COUPLED
*BOUNDARY, TYPE=TEMPERATURE
 XM, 310.0
 XP, 290.0
*BOUNDARY
 YP, 1, 3, 0.0
*BOUNDARY, TYPE=SYMMETRY
 ZM
 ZP
*DLOAD
 ALL, GRAV, 9.81, 0., -1., 0.
*CONTROLS, PARAMETERS=DISCRETIZATION
 CONVECTION=VAN LEER, TIME=BDF2, PRESSURE_VELOCITY=SIMPLEC
*CONTROLS, PARAMETERS=RELAXATION
 VELOCITY=0.5, PRESSURE=0.2
*OUTPUT, FIELD, FORMAT=VTK, FREQUENCY=5
*ELEMENT OUTPUT
 U, P, T
*END STEP
"""


@binds_to(InpKeywordParseProcess)
class TestInpKeywordParseAPI:
    """トークナイザの契約テスト."""

    def test_meta(self):
        assert InpKeywordParseProcess.meta.module == "pre"

    def test_keyword_line_normalization(self):
        kw, params = parse_keyword_line("*Solid   Section, elset=Plate , MATERIAL = Al, Generate")
        assert kw == "SOLID SECTION"
        assert params == {"ELSET": "Plate", "MATERIAL": "Al", "GENERATE": ""}

    def test_process_returns_blocks_and_parameters(self):
        res = InpKeywordParseProcess().execute(InpParseInput(path="<mem>", text=MINIMAL_CASE))
        names = [b.keyword for b in res.blocks]
        assert names[0] == "HEADING"
        assert "PARAMETER" not in names  # 評価されてブロックには残らない
        assert res.parameters == {"n": 3, "L": 0.3, "dx": pytest.approx(0.1)}
        grid = next(b for b in res.blocks if b.keyword == "GRID")
        assert grid.params["NX"] == "3"
        assert grid.params["LZ"] == "0.09999999999999999" or float(
            grid.params["LZ"]
        ) == pytest.approx(0.1)
        heading = res.blocks[0]
        assert heading.raw_lines == ("minimal case", "second line")
        assert heading.data == ()

    def test_flags_and_get(self):
        res = parse_inp_text(MINIMAL_CASE)
        proc = next(b for b in res.blocks if b.keyword == "NAVIER STOKES")
        assert proc.has("steady state")
        assert proc.get("turbulence") == "LAMINAR"
        assert proc.get("missing") is None
        with pytest.raises(InpSyntaxError):
            proc.require("missing")

    def test_comments_and_blank_lines_skipped(self):
        text = "** comment\n\n*NODE\n** inner comment\n 1, 0, 0, 0\n\n 2, 1, 0, 0\n"
        res = parse_inp_text(text)
        assert len(res.blocks) == 1
        assert res.blocks[0].data == (("1", "0", "0", "0"), ("2", "1", "0", "0"))

    def test_continuation_lines(self):
        text = "*ELEMENT, TYPE=C3D8,\n ELSET=A\n 1, 1, 2, 3, 4,\n 5, 6, 7, 8\n"
        res = parse_inp_text(text)
        b = res.blocks[0]
        assert b.params["ELSET"] == "A"
        assert b.data == (("1", "1", "2", "3", "4", "5", "6", "7", "8"),)

    def test_parameter_expression_substitution(self):
        text = "*PARAMETER\n a = 2\n b = a ** 3 + sqrt(16)\n*GRID, NX=<a>, LX=<b / 2>, NY=<int(a * 1.5)>\n"
        res = parse_inp_text(text)
        assert res.parameters["b"] == pytest.approx(12.0)
        assert res.blocks[0].params == {"NX": "2", "LX": "6.0", "NY": "3"}

    def test_parameter_can_be_overridden_from_outside(self):
        text = "*GRID, NX=<n>, LX=1\n"
        res = parse_inp_text(text, parameters={"n": 7})
        assert res.blocks[0].params["NX"] == "7"
        # .inp 内の定義が優先
        text2 = "*PARAMETER\n n = 4\n*GRID, NX=<n>, LX=1\n"
        assert parse_inp_text(text2, parameters={"n": 7}).blocks[0].params["NX"] == "4"

    def test_parameter_errors_have_location(self):
        with pytest.raises(InpSyntaxError, match=r"<mem>:2"):
            parse_inp_text("*PARAMETER\n x = undefined_name + 1\n", path="<mem>")
        with pytest.raises(InpSyntaxError, match="未定義"):
            parse_inp_text("*GRID, NX=<nope>\n")

    def test_safe_eval_rejects_unsafe_syntax(self):
        assert safe_eval("2 * pi", {}) == pytest.approx(6.283185307)
        assert safe_eval("max(1, 2, 3) if 1 < 2 else 0", {}) == 3
        for bad in (
            "__import__('os')",
            "(1).real",
            "[x for x in range(3)]",
            "lambda: 1",
            "open('f')",
        ):
            with pytest.raises(ParameterError):
                safe_eval(bad, {})
        assert substitute("A=<a>, B=<a*2>", {"a": 3}) == "A=3, B=6"

    def test_include_and_cycle(self, tmp_path: Path):
        mesh = tmp_path / "mesh.inp"
        mesh.write_text("*NODE\n 1, 0, 0, 0\n", encoding="utf-8")
        main = tmp_path / "main.inp"
        main.write_text(
            "*HEADING\n x\n*INCLUDE, INPUT=mesh.inp\n*ELEMENT, TYPE=C3D8\n 1, 1,1,1,1,1,1,1,1\n",
            encoding="utf-8",
        )
        res = InpKeywordParseProcess().execute(InpParseInput(path=str(main)))
        assert [b.keyword for b in res.blocks] == ["HEADING", "NODE", "ELEMENT"]
        assert res.blocks[1].source == str(mesh)
        cyc = tmp_path / "cyc.inp"
        cyc.write_text("*INCLUDE, INPUT=cyc.inp\n", encoding="utf-8")
        with pytest.raises(InpSyntaxError, match="循環"):
            InpKeywordParseProcess().execute(InpParseInput(path=str(cyc)))

    def test_data_before_keyword_is_error(self):
        with pytest.raises(InpSyntaxError, match="キーワード行の前"):
            parse_inp_text(" 1, 2, 3\n*NODE\n")


@binds_to(InpCaseBuildProcess)
class TestInpCaseBuildAPI:
    """組み立ての契約テスト."""

    def _case(self, text: str = MINIMAL_CASE) -> CaseDefinition:
        return InpCaseBuildProcess().execute(parse_inp_text(text))

    def test_minimal_case_structure(self):
        case = self._case()
        assert case.heading.startswith("minimal case")
        assert case.nodes.n_nodes == 4 * 4 * 2
        assert case.n_elements == 9
        assert set(case.elsets) == {"ALL"}
        assert set(case.nsets) == {"ALL"}
        mat = case.materials["WATER"]
        assert (mat.density, mat.viscosity, mat.conductivity) == (1000.0, 1e-3, 0.6)
        assert mat.expansion == pytest.approx(2.1e-4)
        assert mat.reference_temperature == 300.0
        assert case.sections[0].kind == SectionKind.FLUID
        assert case.sections[0].material == "WATER"  # 名前は大文字正規化
        ic = case.initial_conditions[0]
        assert ic.kind == InitialConditionKind.TEMPERATURE and ic.target == "ALL"
        assert case.parameters["n"] == 3

    def test_step_contents(self):
        case = self._case()
        assert len(case.steps) == 1
        step = case.steps[0]
        assert step.name == "Flow" and step.max_increments == 50
        proc = step.procedure
        assert proc.family == EquationFamily.NAVIER_STOKES
        assert proc.steady and proc.heat_transfer == "COUPLED" and proc.turbulence == "LAMINAR"
        kinds = [(b.target, b.kind, b.values) for b in step.boundaries]
        assert (("XM", BoundaryKind.TEMPERATURE, (310.0,))) in kinds
        assert (("YP", BoundaryKind.WALL, ())) in kinds  # 自由度 1-3 = 0 は WALL
        assert (("ZM", BoundaryKind.SYMMETRY, ())) in kinds
        assert step.loads[0].vector == pytest.approx((0.0, -9.81, 0.0))
        disc = step.control_values(ControlCategory.DISCRETIZATION)
        assert disc == {"CONVECTION": "VAN LEER", "TIME": "BDF2", "PRESSURE_VELOCITY": "SIMPLEC"}
        assert step.control_values(ControlCategory.RELAXATION)["VELOCITY"] == "0.5"
        out = step.outputs[0]
        assert out.variables == ("U", "P", "T")
        assert OutputFormat.VTK in out.formats and OutputFormat.NPZ in out.formats
        assert out.frequency == 5

    def test_output_format_html_combined(self):
        case = build_case(
            parse_inp_text(
                "*GRID, NX=2, LX=1\n*STEP\n*NAVIER STOKES, STEADY STATE\n"
                "*OUTPUT, FIELD, FORMAT=VTK+HTML\n*END STEP\n"
            )
        )
        out = case.steps[0].outputs[0]
        assert out.formats == (OutputFormat.NPZ, OutputFormat.VTK, OutputFormat.HTML)
        assert out.formats_explicit
        case = build_case(
            parse_inp_text(
                "*GRID, NX=2, LX=1\n*STEP\n*NAVIER STOKES, STEADY STATE\n*OUTPUT, FIELD\n*END STEP\n"
            )
        )
        assert not case.steps[0].outputs[0].formats_explicit

    def test_dof_boundary_forms(self):
        text = (
            "*GRID, NX=2, LX=1\n*STEP\n*HEAT TRANSFER, STEADY STATE\n*BOUNDARY\n"
            " XM, 11, 11, 350.\n XP, 8, 8, 0.\n YM, 1, 1, 0.5\n*BOUNDARY, TYPE=WALL\n YP, SLIP\n*END STEP\n"
        )
        step = self._case(text).steps[0]
        by_target = {b.target: b for b in step.boundaries}
        assert by_target["XM"].kind == BoundaryKind.TEMPERATURE
        assert by_target["XP"].kind == BoundaryKind.PRESSURE
        assert by_target["YM"].kind == BoundaryKind.VELOCITY and by_target["YM"].values == (
            0.5,
            0.0,
            0.0,
        )
        assert by_target["YP"].kind == BoundaryKind.SLIP

    def test_node_element_sets_surfaces(self):
        text = (
            "*NODE, NSET=N1\n 1, 0, 0\n 2, 1, 0\n 3, 1, 1\n 4, 0, 1\n"
            "*ELEMENT, TYPE=CPS4, ELSET=E1\n 1, 1, 2, 3, 4\n"
            "*ELSET, ELSET=E2\n E1, 1\n"
            "*NSET, NSET=N2, GENERATE\n 1, 3, 2\n"
            "*SURFACE, NAME=S_LEFT\n E1, S4\n"
            "*MATERIAL, NAME=M\n*CONDUCTIVITY\n 1.\n*SOLID SECTION, ELSET=E2, MATERIAL=M\n"
            "*DFLUX\n S_LEFT, S, 10.\n E1, BF, 5.\n*SFILM\n S_LEFT, F, 300., 20.\n"
        )
        case = self._case(text)
        assert case.nodes.coords.shape == (4, 3) and np.all(case.nodes.coords[:, 2] == 0.0)
        assert case.elements[0].element_type == "CPS4" and not case.is_3d
        assert case.elsets["E2"].ids.tolist() == [1]
        assert case.nsets["N2"].ids.tolist() == [1, 3]
        assert case.surfaces["S_LEFT"].entries[0].face == "S4"
        assert [(f.label, f.magnitude) for f in case.fluxes] == [
            (FluxLabel.SURFACE, 10.0),
            (FluxLabel.BODY, 5.0),
        ]
        assert case.films[0].h == 20.0 and case.films[0].t_inf == 300.0
        assert case.element_ids_of("e1").tolist() == [1]
        assert case.node_ids_of("N1").tolist() == [1, 2, 3, 4]

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("*GRID, NX=2, LX=1\n*STEP\n*HEAT TRANSFER, STEADY STATE\n", "END STEP"),
            ("*GRID, NX=2, LX=1\n*NAVIER STOKES, STEADY STATE\n", "STEP の中"),
            ("*GRID, NX=2, LX=1\n*STEP\n*END STEP\n", "手続きキーワード"),
            ("*GRID, NX=2, LX=1\n*STEP\n*HEAT TRANSFER\n*END STEP\n", "非定常"),
            ("*GRID, NX=2, LX=1\n*BOUNDARY, TYPE=MAGIC\n XM\n", "未対応"),
            ("*GRID, NX=2, LX=1\n*NODE\n 1, 0, 0, 0\n", "併用"),
            ("*ELEMENT, TYPE=C3D3\n 1, 1, 2, 3\n", "未対応"),
            ("*ELEMENT, TYPE=CPS5\n 1, 1, 2, 3, 4, 5\n", "未対応"),
            ("*GRID, NX=2, LX=1\n*FLUID SECTION, ELSET=NOPE, MATERIAL=M\n", "未定義"),
            (
                "*GRID, NX=2, LX=1\n*STEP\n*NAVIER STOKES, STEADY STATE\n*CONTROLS, PARAMETERS=FOO\n*END STEP\n",
                "未対応",
            ),
            (
                "*GRID, NX=2, LX=1\n*STEP\n*NAVIER STOKES, STEADY STATE\n*OUTPUT, FIELD, FORMAT=CSV\n*END STEP\n",
                "FORMAT",
            ),
        ],
    )
    def test_errors(self, text: str, match: str):
        with pytest.raises(InpSyntaxError, match=match):
            self._case(text)

    def test_transient_procedure_data_line(self):
        text = "*GRID, NX=2, LX=1\n*STEP\n*HEAT TRANSFER\n 0.1, 2.0\n*END STEP\n"
        proc = self._case(text).steps[0].procedure
        assert not proc.steady and proc.dt == 0.1 and proc.time_period == 2.0

    def test_generate_grid_node_ordering(self):
        nodes, elems = generate_grid(2, 1, 1, 2.0, 1.0, 1.0)
        assert nodes.n_nodes == 12 and elems.n_elements == 2
        # 要素 1 の節点順序が Abaqus C3D8（下面 1-2-3-4 反時計回り、上面 5-8）
        c = elems.connectivity[0]
        xyz = nodes.coords[c - 1]
        assert xyz[0].tolist() == [0.0, 0.0, 0.0]
        assert xyz[1].tolist() == [1.0, 0.0, 0.0]
        assert xyz[2].tolist() == [1.0, 1.0, 0.0]
        assert xyz[3].tolist() == [0.0, 1.0, 0.0]
        assert xyz[4].tolist() == [0.0, 0.0, 1.0]

    def test_build_case_function_equals_process(self):
        parsed = parse_inp_text(MINIMAL_CASE)
        assert build_case(parsed).n_elements == InpCaseBuildProcess().execute(parsed).n_elements


class TestGenericExtrusionKeywords:
    """汎用記法（周期境界・体積力・非ニュートン粘度・座標系・MPC・回転自由度）の構文."""

    def _case(self, extra: str):
        text = "*GRID, NX=2, NY=2, NZ=1, LX=0.2, LY=0.2, LZ=0.1\n" + extra
        return build_case(parse_inp_text(text))

    def test_periodic_pair(self):
        case = self._case("*BOUNDARY, TYPE=PERIODIC\n WEST, EAST\n")
        assert len(case.periodic) == 1
        # 別名 WEST/EAST は予約面名 XM/XP に正規化される
        assert case.periodic[0].master == "XM" and case.periodic[0].slave == "XP"
        assert case.periodic[0].translation is None
        assert not case.boundaries

    def test_periodic_with_translation(self):
        case = self._case("*BOUNDARY, TYPE=PERIODIC\n XM, XP, 0.2, 0.0, 0.0\n")
        assert case.periodic[0].translation == (0.2, 0.0, 0.0)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ("*BOUNDARY, TYPE=PERIODIC\n XM\n", "master_surface, slave_surface"),
            ("*BOUNDARY, TYPE=PERIODIC\n XM, XM\n", "2 面が同じ"),
            ("*BOUNDARY, TYPE=PERIODIC\n XM, NOPE\n", "予約面名でもありません"),
            (
                "*BOUNDARY, TYPE=PERIODIC\n XM, XP\n*BOUNDARY, TYPE=PERIODIC\n XP, YM\n",
                "複数の周期境界",
            ),
            (
                "*STEP, NAME=s\n*NAVIER STOKES, STEADY STATE\n"
                "*BOUNDARY, TYPE=PERIODIC\n XM, XP\n*END STEP\n",
                "\\*STEP の外",
            ),
        ],
    )
    def test_periodic_errors(self, extra: str, match: str):
        with pytest.raises(InpSyntaxError, match=match):
            self._case(extra)

    @pytest.mark.parametrize(
        "row,label,vector",
        [
            (" ALL, BX, 2.5", "BX", (2.5, 0.0, 0.0)),
            (" ALL, BY, -3.0", "BY", (0.0, -3.0, 0.0)),
            (" ALL, BZ, 4.0", "BZ", (0.0, 0.0, 4.0)),
            (" ALL, BF, 1.0, 2.0, 3.0", "BF", (1.0, 2.0, 3.0)),
        ],
    )
    def test_dload_body_force(self, row: str, label: str, vector: tuple):
        case = self._case(f"*DLOAD\n{row}\n")
        ld = case.loads[0]
        assert ld.label == label and ld.is_body_force and ld.vector == vector

    def test_dload_grav_is_not_body_force(self):
        case = self._case("*DLOAD\n ALL, GRAV, 9.81, 0., -1., 0.\n")
        assert not case.loads[0].is_body_force
        assert case.loads[0].vector == (0.0, -9.81, 0.0)

    def test_dload_unknown_label(self):
        with pytest.raises(InpSyntaxError, match="GRAV / BX / BY / BZ / BF"):
            self._case("*DLOAD\n ALL, PX, 1.0\n")

    def test_viscosity_power_law_and_carreau(self):
        case = self._case(
            "*MATERIAL, NAME=M1\n*VISCOSITY, TYPE=POWER LAW\n 5000., 0.4, 1e-3, 1e7\n"
            "*MATERIAL, NAME=M2\n*VISCOSITY, TYPE=CARREAU\n 1e4, 10., 1.0, 0.5\n"
            "*MATERIAL, NAME=M3\n*VISCOSITY\n 1000.\n"
        )
        law = case.materials["M1"].viscosity_law
        assert law.model.value == "POWER LAW" and law.parameters == (5000.0, 0.4, 1e-3, 1e7)
        assert case.materials["M1"].viscosity == 5000.0  # 参照粘度 = K
        assert case.materials["M2"].viscosity_law.parameters == (1e4, 10.0, 1.0, 0.5)
        assert case.materials["M3"].viscosity_law is None
        assert case.materials["M3"].viscosity == 1000.0

    @pytest.mark.parametrize(
        "extra,match",
        [
            ("*MATERIAL, NAME=M\n*VISCOSITY, TYPE=BINGHAM\n 1., 2.\n", "TYPE=BINGHAM"),
            ("*MATERIAL, NAME=M\n*VISCOSITY, TYPE=POWER LAW\n 5000.\n", "K, n"),
            ("*MATERIAL, NAME=M\n*VISCOSITY, TYPE=CARREAU\n 1., 2., 3.\n", "mu_0"),
            ("*MATERIAL, NAME=M\n*VISCOSITY, TYPE=POWER LAW\n 5000., -0.4\n", "正の値"),
        ],
    )
    def test_viscosity_errors(self, extra: str, match: str):
        with pytest.raises(InpSyntaxError, match=match):
            self._case(extra)

    def test_orientation_and_mpc(self):
        case = self._case(
            "*NSET, NSET=REF\n 1\n"
            "*ORIENTATION, NAME=SPIN, SYSTEM=CYLINDRICAL\n 0., 0., 0., 0., 0., 1.\n"
            "*MPC\n BEAM, XP, REF\n"
        )
        ori = case.orientations["SPIN"]
        assert ori.system.value == "CYLINDRICAL" and ori.point_b == (0.0, 0.0, 1.0)
        assert np.allclose(ori.basis()[2], [0.0, 0.0, 1.0])
        assert case.mpcs[0].slave == "XP" and case.mpcs[0].master == "REF"

    def test_rotation_dofs_4_to_6(self):
        case = self._case(
            "*NSET, NSET=REF\n 1\n*MPC\n BEAM, XP, REF\n"
            "*ORIENTATION, NAME=SPIN, SYSTEM=CYLINDRICAL\n 0., 0., 0., 0., 0., 1.\n"
            "*BOUNDARY, ORIENTATION=SPIN\n REF, 6, 6, 12.5\n REF, 1, 3, 0.5\n"
        )
        rot = [b for b in case.boundaries if b.kind is BoundaryKind.ROTATION][0]
        assert rot.values == (0.0, 0.0, 12.5) and rot.orientation == "SPIN"
        vel = [b for b in case.boundaries if b.kind is BoundaryKind.VELOCITY][0]
        assert vel.values == (0.5, 0.5, 0.5)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ("*NSET, NSET=REF\n 1\n*MPC\n BEAM, NOPE, REF\n", "従属面"),
            ("*MPC\n BEAM, XP, NOSUCH\n", "参照節点"),
            ("*NSET, NSET=REF\n 1\n*MPC\n SPRING, XP, REF\n", "BEAM, RIGID, TIE"),
            ("*ORIENTATION, NAME=A, SYSTEM=SPHERICAL\n 0.,0.,0., 0.,0.,1.\n", "SYSTEM=SPHERICAL"),
            ("*ORIENTATION, NAME=A\n 0., 0., 0.\n", "ax, ay, az"),
            ("*BOUNDARY, ORIENTATION=NOPE\n XP, 1, 3, 0.\n", "ORIENTATION=NOPE"),
            ("*BOUNDARY\n XP, 7, 7, 0.\n", "4-6: 角速度"),
        ],
    )
    def test_orientation_mpc_errors(self, extra: str, match: str):
        with pytest.raises(InpSyntaxError, match=match):
            self._case(extra)
