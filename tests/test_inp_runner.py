"""ykep .inp フォーマット: 出力（InpOutputWriterProcess）・ジョブ実行（InpCaseRunnerProcess）・CLI のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.case import EquationFamily, OutputFormat, OutputRequest
from xkep_cae_fluid.inp.cli import main, parse_args
from xkep_cae_fluid.inp.grid import recover_structured_grid
from xkep_cae_fluid.inp.output import FieldOutputInput, InpOutputWriterProcess, dump_yaml
from xkep_cae_fluid.inp.parser import parse_inp_text
from xkep_cae_fluid.inp.runner import InpCaseRunnerProcess, InpJobInput

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "inp"

try:
    import messi  # noqa: F401

    _HAS_MESSI = True
except ImportError:  # pragma: no cover - 環境依存
    _HAS_MESSI = False


def _grid(nx=2, ny=2, nz=1):
    case = build_case(parse_inp_text(f"*GRID, NX={nx}, NY={ny}, NZ={nz}, LX=1, LY=1, LZ=0.5\n"))
    return recover_structured_grid(case)


@binds_to(InpOutputWriterProcess)
class TestInpOutputWriterAPI:
    def test_writes_npz_yaml_vtk(self, tmp_path: Path):
        grid = _grid()
        fields = {
            "U": np.zeros((2, 2, 1, 3)),
            "P": np.arange(4.0).reshape(2, 2, 1),
            "T": np.full((2, 2, 1), 300.0),
        }
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name="j",
                output_dir=str(tmp_path / "out"),
                grid=grid,
                fields=fields,
                summary={
                    "converged": True,
                    "n": 3,
                    "name": "a: b",
                    "hist": [1.0, 2.0],
                    "nested": {"x": None},
                },
                requests=(
                    OutputRequest(
                        variables=("NT11", "P"),
                        formats=(OutputFormat.NPZ, OutputFormat.VTK),
                        formats_explicit=True,
                    ),
                ),
            )
        )
        names = sorted(Path(p).name for p in res.paths)
        assert names == ["j.npz", "j.vtk", "j.yaml"]
        data = np.load(tmp_path / "out" / "j.npz")
        assert set(data.files) == {
            "x_lines",
            "y_lines",
            "z_lines",
            "T",
            "P",
        }  # NT11 → T、U は未選択
        vtk = (tmp_path / "out" / "j.vtk").read_text()
        assert "RECTILINEAR_GRID" in vtk and "CELL_DATA 4" in vtk and "SCALARS P" in vtk
        yaml = pytest.importorskip("yaml")
        loaded = yaml.safe_load((tmp_path / "out" / "j.yaml").read_text())
        assert loaded["converged"] is True and loaded["n"] == 3 and loaded["name"] == "a: b"
        assert loaded["hist"] == [1.0, 2.0] and loaded["nested"] == {"x": None}
        assert loaded["variables"] == ["T", "P"]

    def test_unstructured_npz_vtk(self, tmp_path: Path):
        from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess

        mesh = (
            StructuredMeshProcess()
            .execute(StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=1, nz=1))
            .mesh
        )
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name="u",
                output_dir=str(tmp_path),
                grid=None,
                mesh=mesh,
                fields={"P": np.array([1.0, 2.0]), "U": np.zeros((2, 3))},
                requests=(
                    OutputRequest(
                        formats=(OutputFormat.NPZ, OutputFormat.VTK), formats_explicit=True
                    ),
                ),
            )
        )
        assert sorted(Path(p).name for p in res.paths) == ["u.npz", "u.vtk", "u.yaml"]
        with np.load(tmp_path / "u.npz") as data:
            assert data["connectivity"].shape == (2, 8) and data["cell_types"].tolist() == [12, 12]
        text = (tmp_path / "u.vtk").read_text()
        assert "POINTS 12 double" in text and "CELLS 2 18" in text and "CELL_DATA 2" in text
        with pytest.raises(ValueError, match="grid か mesh"):
            InpOutputWriterProcess().execute(
                FieldOutputInput(job_name="x", output_dir=str(tmp_path), grid=None, fields={})
            )

    def test_default_all_variables_and_unknown_variable(self, tmp_path: Path):
        grid = _grid()
        fields = {"T": np.zeros((2, 2, 1))}
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(job_name="k", output_dir=str(tmp_path), grid=grid, fields=fields)
        )
        # FORMAT 未指定: messi があれば HTML も自動で付く
        expected = ["k.npz", "k.html", "k.yaml"] if _HAS_MESSI else ["k.npz", "k.yaml"]
        assert [Path(p).name for p in res.paths] == expected
        with pytest.raises(ValueError, match="未対応"):
            InpOutputWriterProcess().execute(
                FieldOutputInput(
                    job_name="k",
                    output_dir=str(tmp_path),
                    grid=grid,
                    fields=fields,
                    requests=(OutputRequest(variables=("VORTICITY",)),),
                )
            )

    def test_html_output_via_mirador(self, tmp_path: Path):
        pytest.importorskip("messi")
        grid = _grid(3, 3, 1)
        fields = {"T": np.arange(9.0).reshape(3, 3, 1), "U": np.zeros((3, 3, 1, 3))}
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name="h",
                output_dir=str(tmp_path),
                grid=grid,
                fields=fields,
                title="html test",
                requests=(OutputRequest(formats=(OutputFormat.HTML,)),),
            )
        )
        assert [Path(p).name for p in res.paths] == ["h.npz", "h.html", "h.yaml"]
        html = (tmp_path / "h.html").read_text(encoding="utf-8")
        assert "html test" in html and '"fieldNames": ["T", "|U|"' in html
        yaml = pytest.importorskip("yaml")
        loaded = yaml.safe_load((tmp_path / "h.yaml").read_text())
        assert loaded["output_files"] == ["h.npz", "h.html", "h.yaml"]

    def test_html_auto_when_format_unspecified_and_residual_alias(self, tmp_path: Path):
        pytest.importorskip("messi")
        grid = _grid(3, 3, 1)
        fields = {
            "T": np.zeros((3, 3, 1)),
            "res_T": np.ones((3, 3, 1)),
            "res_mass": np.ones((3, 3, 1)),
        }
        # FORMAT 未指定（*OUTPUT なし）→ messi があれば HTML も自動
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(job_name="a", output_dir=str(tmp_path), grid=grid, fields=fields)
        )
        assert [Path(p).name for p in res.paths] == ["a.npz", "a.html", "a.yaml"]
        # FORMAT=NPZ を明示 → HTML は出さない
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name="b",
                output_dir=str(tmp_path),
                grid=grid,
                fields=fields,
                requests=(OutputRequest(formats=(OutputFormat.NPZ,), formats_explicit=True),),
            )
        )
        assert [Path(p).name for p in res.paths] == ["b.npz", "b.yaml"]
        # VARIABLE=T,RES → 残差マップ全部を展開
        res = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name="c",
                output_dir=str(tmp_path),
                grid=grid,
                fields=fields,
                requests=(OutputRequest(variables=("T", "RES"), formats_explicit=True),),
            )
        )
        assert set(np.load(tmp_path / "c.npz").files) >= {"T", "res_T", "res_mass"}

    def test_dump_yaml_roundtrip(self):
        yaml = pytest.importorskip("yaml")
        data = {"a": 1, "b": [1, "x y", {"c": 2.5}], "d": {}, "e": [], "f": "plain", "g": False}
        assert yaml.safe_load(dump_yaml(data)) == data


TINY_NS = """\
*HEADING
 tiny cavity for pipeline test
*GRID, NX=4, NY=4, NZ=2, LX=0.1, LY=0.1, LZ=0.05
*MATERIAL, NAME=F
*DENSITY
 1.0
*VISCOSITY
 0.01
*SPECIFIC HEAT
 1000.
*CONDUCTIVITY
 1.
*EXPANSION, ZERO=300.
 1e-3
*FLUID SECTION, ELSET=ALL, MATERIAL=F
*STEP, NAME=S1
*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=COUPLED
*DLOAD
 ALL, GRAV, 9.81, 0, -1, 0
*BOUNDARY, TYPE=TEMPERATURE
 XM, 310.
 XP, 290.
*CONTROLS, PARAMETERS=SOLVER
 MAX_OUTER=3, TOL=1e-4
*OUTPUT, FIELD, FORMAT=VTK
*END STEP
"""


TINY_DARCY = """\
*HEADING
 tiny darcy for pipeline test
*GRID, NX=4, NY=2, NZ=2, LX=0.4, LY=0.2, LZ=0.2
*MATERIAL, NAME=SAND
*DENSITY
 1000.
*VISCOSITY
 1e-3
*PERMEABILITY
 1e-10
*FLUID SECTION, ELSET=ALL, MATERIAL=SAND
*STEP, NAME=S1
*DARCY, STEADY STATE
*BOUNDARY, TYPE=PRESSURE
 XM, 100.
 XP, 0.
*CONTROLS, PARAMETERS=SOLVER
 METHOD=DIRECT
*OUTPUT, FIELD, FORMAT=VTK
*END STEP
"""


@binds_to(InpCaseRunnerProcess)
class TestInpCaseRunnerAPI:
    def test_heat_transfer_example_runs_and_converges(self, tmp_path: Path):
        res = InpCaseRunnerProcess().execute(
            InpJobInput(path=str(EXAMPLES / "plate-ht-1.inp"), output_dir=str(tmp_path))
        )
        assert res.job_name == "plate-ht-1" and res.converged
        step = res.steps[0]
        assert step.family == EquationFamily.HEAT_TRANSFER
        assert sorted(Path(p).name for p in step.output_paths) == [
            "plate-ht-1.npz",
            "plate-ht-1.vtk",
            "plate-ht-1.yaml",
        ]
        assert step.summary["parameters"]["T_left"] == 350.0
        T = np.load(tmp_path / "plate-ht-1.npz")["T"]
        assert T.shape == (4, 2, 2) and T.min() > 350.0

    def test_heat_transfer_unstructured_matches_structured(self, tmp_path: Path):
        """箱格子の plate-ht-1 を mesh_mode=unstructured で解くと FDM 版と同じ温度場（非構造出力）."""
        ref = InpCaseRunnerProcess().execute(
            InpJobInput(path=str(EXAMPLES / "plate-ht-1.inp"), output_dir=str(tmp_path / "s"))
        )
        res = InpCaseRunnerProcess().execute(
            InpJobInput(
                path=str(EXAMPLES / "plate-ht-1.inp"),
                output_dir=str(tmp_path / "u"),
                mesh_mode="unstructured",
            )
        )
        assert res.grid is None and res.mesh is not None and res.converged
        step = res.steps[0]
        assert step.summary["solver"]["process"] == "HeatTransferFVMProcess"
        assert step.summary["mesh"]["max_nonorthogonality_deg"] == pytest.approx(0.0, abs=1e-9)
        # セル順は要素順（plate-mesh.inp は i 最速）なので、構造格子の (i, j, k) に戻して比較
        T_u = step.result.T
        T_s = ref.steps[0].result.T
        lookup = {int(e): idx for idx, e in enumerate(ref.grid.element_ids.tolist())}
        ijk = ref.grid.element_ijk[[lookup[int(e)] for e in res.mesh.element_ids.tolist()]]
        assert np.allclose(T_u, T_s[ijk[:, 0], ijk[:, 1], ijk[:, 2]], rtol=1e-8)
        with np.load(tmp_path / "u" / "plate-ht-1.npz") as data:
            assert data["T"].shape == (16,) and data["connectivity"].shape == (16, 8)
        assert "DATASET UNSTRUCTURED_GRID" in (tmp_path / "u" / "plate-ht-1.vtk").read_text()

    def test_heat_transfer_sheared_example_auto_falls_back(self, tmp_path: Path):
        """plate-ht-2（せん断メッシュ）は auto で非構造経路に落ち、structured 強制なら拒否."""
        res = InpCaseRunnerProcess().execute(
            InpJobInput(path=str(EXAMPLES / "plate-ht-2.inp"), output_dir=str(tmp_path))
        )
        assert res.grid is None and res.mesh is not None and res.mesh.n_cells == 32
        step = res.steps[0]
        assert res.converged and step.summary["solver"]["process"] == "HeatTransferFVMProcess"
        assert step.summary["mesh"]["max_nonorthogonality_deg"] == pytest.approx(
            np.degrees(np.arctan(0.3)), abs=1e-6
        )
        assert step.summary["temperature_range"][0] > 350.0
        with pytest.raises(ValueError, match="箱格子"):
            InpCaseRunnerProcess().execute(
                InpJobInput(
                    path=str(EXAMPLES / "plate-ht-2.inp"),
                    output_dir=str(tmp_path),
                    mesh_mode="structured",
                )
            )

    def test_navier_stokes_rejected_on_unstructured(self, tmp_path: Path):
        inp = tmp_path / "ns.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        with pytest.raises(ValueError, match="NAVIER STOKES"):
            InpCaseRunnerProcess().execute(
                InpJobInput(path=str(inp), mesh_mode="unstructured", check_only=True)
            )
        with pytest.raises(ValueError, match="mesh_mode"):
            InpCaseRunnerProcess().execute(InpJobInput(path=str(inp), mesh_mode="polyhedral"))

    def test_navier_stokes_pipeline(self, tmp_path: Path):
        inp = tmp_path / "tiny.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        res = InpCaseRunnerProcess().execute(InpJobInput(path=str(inp)))
        step = res.steps[0]
        assert step.family == EquationFamily.NAVIER_STOKES
        assert step.n_iterations == 3  # MAX_OUTER=3 で打ち切り（収束は要求しない）
        assert (tmp_path / "tiny.npz").exists() and (tmp_path / "tiny.vtk").exists()
        data = np.load(tmp_path / "tiny.npz")
        assert data["U"].shape == (4, 4, 2, 3)
        assert data["res_mass"].shape == (4, 4, 2)  # 残差マップも出力に含まれる
        assert step.summary["solver"]["process"] == "NaturalConvectionFDMProcess"
        assert "final_residuals" in step.summary

    def test_check_only_and_darcy(self, tmp_path: Path):
        inp = tmp_path / "chk.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        res = InpCaseRunnerProcess().execute(InpJobInput(path=str(inp), check_only=True))
        assert (
            res.steps[0].output_paths == ()
            and res.steps[0].summary["solver"]["coupling"] == "simple"
        )
        # *DARCY は *PERMEABILITY が無いとマッピングで拒否される
        darcy = tmp_path / "darcy.inp"
        darcy.write_text(
            TINY_NS.replace(
                "*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=COUPLED", "*DARCY, STEADY STATE"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="permeability"):
            InpCaseRunnerProcess().execute(InpJobInput(path=str(darcy)))

    def test_darcy_pipeline_unstructured_output(self, tmp_path: Path):
        """*DARCY: 非構造メッシュ経由で解き、NPZ（node_coords/connectivity）と VTK UNSTRUCTURED_GRID を書く."""
        inp = tmp_path / "d.inp"
        inp.write_text(TINY_DARCY, encoding="utf-8")
        res = InpCaseRunnerProcess().execute(InpJobInput(path=str(inp), output_dir=str(tmp_path)))
        assert res.grid is None and res.mesh is not None and res.mesh.n_cells == 16
        step = res.steps[0]
        assert step.family == EquationFamily.DARCY and step.converged
        assert step.summary["mesh"]["n_cells"] == 16 and step.summary["solver"]["process"] == (
            "DarcyFlowProcess"
        )
        names = sorted(Path(p).name for p in step.output_paths)
        assert names[:3] == ["d.npz", "d.vtk", "d.yaml"] or names == [
            "d.html",
            "d.npz",
            "d.vtk",
            "d.yaml",
        ]
        with np.load(tmp_path / "d.npz") as data:
            assert data["connectivity"].shape == (16, 8) and data["node_coords"].shape == (45, 3)
            assert data["P"].shape == (16,) and data["U"].shape == (16, 3)
            p = data["P"]
        vtk = (tmp_path / "d.vtk").read_text()
        assert "DATASET UNSTRUCTURED_GRID" in vtk and "CELL_TYPES 16" in vtk
        assert "VECTORS U double" in vtk and "SCALARS P double 1" in vtk
        # 1D 圧力差 100 → 0 で線形（x 方向 4 セル）
        assert p.max() == pytest.approx(87.5) and p.min() == pytest.approx(12.5)
        u_exact = 1e-10 * 100.0 / (1e-3 * 0.4)
        assert step.summary["max_abs_velocity"] == pytest.approx(u_exact, rel=1e-8)
        assert step.summary["inflow_m3s"] == pytest.approx(step.summary["outflow_m3s"], rel=1e-8)
        # 後追いの view（非構造 NPZ → HTML）は messi があるときだけ
        if _HAS_MESSI:
            html = tmp_path / "d.html"
            html.unlink(missing_ok=True)
            code = main(["-j", str(inp), "view", f"-o={tmp_path}", "--cut=x=0.2"])
            assert code == 0 and html.exists()
            assert (
                main(["-j", str(inp), "view", f"-o={tmp_path}", "--slice=x=0.2"]) == 2
            )  # 非構造では --slice 不可

    def test_parameter_override(self, tmp_path: Path):
        inp = tmp_path / "p.inp"
        inp.write_text(TINY_NS.replace("MAX_OUTER=3", "MAX_OUTER=<iters>"), encoding="utf-8")
        res = InpCaseRunnerProcess().execute(InpJobInput(path=str(inp), parameters={"iters": 2}))
        assert res.steps[0].n_iterations == 2


class TestYkepCli:
    def test_parse_args_abaqus_style(self, tmp_path: Path):
        inp = tmp_path / "nsb-1.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        args = parse_args(
            [f"-j={inp.with_suffix('')}", "int", "-p", "iters=4", f"-o={tmp_path / 'o'}"]
        )
        assert args.interactive
        job = args.job
        assert job.path == str(inp)
        assert job.parameters == {"iters": 4}
        assert job.output_dir == str(tmp_path / "o")
        args2 = parse_args([f"job={inp}", "--check"])
        assert args2.job.check_only and not args2.interactive
        assert args2.job.mesh_mode == "auto"
        assert parse_args([f"-j={inp}", "--mesh=Unstructured"]).job.mesh_mode == "unstructured"
        with pytest.raises(SystemExit):
            parse_args([f"-j={inp}", "--mesh=polyhedral"])
        assert parse_args(["--help"]) is None

    def test_main_check_and_errors(self, tmp_path: Path, capsys):
        inp = tmp_path / "c.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        assert main([f"-j={inp}", "--check"]) == 0
        assert "CHECK OK" in capsys.readouterr().out
        assert (tmp_path / "c.log").exists()
        assert main(["-j=/nonexistent/x.inp"]) == 2
        assert main([]) == 2
        bad = tmp_path / "bad.inp"
        bad.write_text("*GRID, NX=2, LX=1\n", encoding="utf-8")
        assert main([f"-j={bad}"]) == 1
        assert "ERROR" in capsys.readouterr().err

    def test_main_runs_heat_transfer_example(self, tmp_path: Path, capsys):
        code = main([f"-j={EXAMPLES / 'plate-ht-1'}", "int", f"-o={tmp_path}"])
        assert code == 0
        out = capsys.readouterr().out
        assert "CONVERGED" in out and "plate-ht-1.yaml" in out

    def test_parse_view_args(self, tmp_path: Path):
        inp = tmp_path / "v.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        args = parse_args(
            [
                f"-j={inp}",
                "view",
                "--slice=x=0.05",
                "--slice",
                "z=1i",
                "--no-vectors",
                "--collapse-panel",
                "--cut=z=0.5",
            ]
        )
        assert args.view and args.no_vectors and not args.no_slices
        assert args.collapse_panel
        assert args.cut == ((0.0, 0.0, 1.0), 0.5)
        assert parse_args([f"-j={inp}", "view"]).cut is None
        assert parse_args([f"-j={inp}", "view", "--cut", "-x=0.1"]).cut == ((-1.0, 0.0, 0.0), -0.1)
        assert parse_args([f"-j={inp}", "view", "--cut=1, 1, 0, 0.3"]).cut == ((1.0, 1.0, 0.0), 0.3)
        for bad in ("w=1", "z=abc", "1,2,3", "0,0,0,1"):
            with pytest.raises(SystemExit):
                parse_args([f"-j={inp}", "view", f"--cut={bad}"])
        assert [(s.axis, s.position, s.index) for s in args.slices] == [
            ("x", 0.05, None),
            ("z", None, 1),
        ]
        with pytest.raises(SystemExit):
            parse_args([f"-j={inp}", "view", "--slice=w=1"])
        with pytest.raises(SystemExit):
            parse_args([f"-j={inp}", "view", "--slice=x=abc"])

    def test_main_view_from_npz(self, tmp_path: Path, capsys):
        pytest.importorskip("messi")
        inp = tmp_path / "tiny.inp"
        inp.write_text(TINY_NS, encoding="utf-8")
        # NPZ が無ければ案内して終了コード 2
        assert main([f"-j={inp}", "view"]) == 2
        assert "npz" in capsys.readouterr().err
        assert main([f"-j={inp}"]) == 3  # MAX_OUTER=3 で打ち切り = NOT CONVERGED（NPZ は出る）
        capsys.readouterr()
        assert main([f"-j={inp}", "view", "--slice=x=0.05", "--no-slices"]) == 0
        out = capsys.readouterr().out
        assert "VIEW:" in out and (tmp_path / "tiny.html").exists()
        html = (tmp_path / "tiny.html").read_text(encoding="utf-8")
        assert '"groupNames": ["domain", "x=' in html  # --no-slices でも明示 --slice は有効
        # --cut: 任意平面の断面を有効にして開く（自動スラブ無し、外皮は隠さない）
        assert main([f"-j={inp}", "view", "--cut=y=0.5"]) == 0
        capsys.readouterr()
        html = (tmp_path / "tiny.html").read_text(encoding="utf-8")
        assert '"cut": {"normal": [0.0, 1.0, 0.0], "d": 0.5}' in html
        assert '"groupNames": ["domain"]' in html and '"hiddenGroups": []' in html


class TestInpPhysics:
    """物理テスト: .inp 経由でも 1D 定常熱伝導が線形分布になること."""

    def test_darcy_example_mass_balance(self, tmp_path: Path):
        """darcy-1.inp（せん断メッシュ + 低透過率ブロック）: 流入 = 流出、圧力は境界値の範囲内."""
        res = InpCaseRunnerProcess().execute(
            InpJobInput(path=str(EXAMPLES / "darcy-1.inp"), output_dir=str(tmp_path))
        )
        assert res.grid is None and res.mesh is not None and res.mesh.n_cells == 144
        s = res.steps[0].summary
        assert res.converged and s["inflow_m3s"] == pytest.approx(s["outflow_m3s"], rel=1e-9)
        assert 0.0 < s["pressure_range"][0] < s["pressure_range"][1] < 1000.0
        # 非直交補正は遅延補正なので質量不整合は tol（1e-10）× 流量のオーダー
        assert s["max_mass_residual"] < 1e-9 * s["inflow_m3s"]
        # 低透過率ブロック内の速度は砂の部分より遥かに小さい
        d = res.steps[0].result
        clay = res.mesh.mask_for_elements(res.case.elsets["CLAY"].ids)
        speed = np.linalg.norm(d.velocity, axis=1)
        assert speed[clay].max() < 0.05 * speed[~clay].mean()

    def test_linear_conduction_profile(self, tmp_path: Path):
        text = (
            "*GRID, NX=10, NY=1, NZ=1, LX=1.0, LY=0.1, LZ=0.1\n"
            "*MATERIAL, NAME=M\n*CONDUCTIVITY\n 5.\n*SOLID SECTION, ELSET=ALL, MATERIAL=M\n"
            "*STEP\n*HEAT TRANSFER, STEADY STATE\n*BOUNDARY, TYPE=TEMPERATURE\n XM, 400.\n XP, 300.\n"
            "*CONTROLS, PARAMETERS=SOLVER\n METHOD=DIRECT\n*END STEP\n"
        )
        inp = tmp_path / "rod.inp"
        inp.write_text(text, encoding="utf-8")
        res = InpCaseRunnerProcess().execute(InpJobInput(path=str(inp)))
        T = res.steps[0].result.T[:, 0, 0]
        x = 0.5 * (res.grid.x_lines[:-1] + res.grid.x_lines[1:])
        assert np.allclose(T, 400.0 - 100.0 * x, atol=1e-6)
