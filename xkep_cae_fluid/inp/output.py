"""``*OUTPUT, FIELD`` の書き出し（NPZ / YAML サマリ / VTK legacy）.

- ``<job>.npz``: 格子線と場（u, v, w, p, T …）。必ず出力
- ``<job>.yaml``: 収束情報・残差・実行条件（STA2 防止ルール: ログと照合できる形）
- ``<job>.vtk``: ``FORMAT=VTK`` 指定時。RECTILINEAR_GRID + CELL_DATA（依存ライブラリなし）。
  非構造格子（``*DARCY``）では UNSTRUCTURED_GRID、NPZ には ``node_coords`` / ``connectivity``
- ``<job>.html``: ``FORMAT=HTML`` 指定時。messi mirador（three.js）3D ビューア
  （:class:`MiradorExportProcess`。messi 未導入なら警告して他の出力は続行）。
  ``FORMAT=`` を書いていない（``*OUTPUT`` 自体が無い場合も含む）ときは、messi が
  import できる環境なら自動で書く（明示した FORMAT はそのまま尊重）
"""

from __future__ import annotations

import subprocess
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.inp.case import OutputFormat, OutputRequest
from xkep_cae_fluid.inp.grid import StructuredGridMap
from xkep_cae_fluid.post.mirador import (
    MiradorExportInput,
    MiradorExportProcess,
    MiradorUnavailableError,
)

# 変数名の別名（Abaqus 風 → 内部名）
VARIABLE_ALIASES: dict[str, str] = {
    "U": "U",
    "V": "U",
    "VELOCITY": "U",
    "P": "P",
    "PRESSURE": "P",
    "T": "T",
    "NT": "T",
    "NT11": "T",
    "TEMP": "T",
    "TEMPERATURE": "T",
    "MU": "MU",  # 非ニュートン粘度 μ(γ̇)（*VISCOSITY, TYPE=POWER LAW / CARREAU のとき）
    "VISCOSITY": "MU",
    "GAMMA": "GAMMA",  # せん断速度 γ̇
    "SR": "GAMMA",
    "STRAIN_RATE": "GAMMA",
    "LAMBDA": "LAMBDA",  # 混合指数 λ = |D|/(|D|+|Ω|)
    "MIX": "LAMBDA",
    "MIXING_INDEX": "LAMBDA",
    "RES": "res_*",  # 残差マップ全部（res_u / res_v / res_w / res_T / res_mass …）
    "RESIDUAL": "res_*",
}


@dataclass(frozen=True)
class FieldOutputInput:
    """:class:`InpOutputWriterProcess` の入力.

    Parameters
    ----------
    job_name : str
        出力ファイルのベース名
    output_dir : str
        出力ディレクトリ（無ければ作成）
    grid : StructuredGridMap | None
        構造格子（``*NAVIER STOKES`` / ``*HEAT TRANSFER``）。非構造なら None
    mesh : MeshData | None
        非構造格子（``*DARCY``）。``grid`` が None のとき必須
    fields : Mapping[str, np.ndarray]
        内部変数名 → (nx, ny, nz) 配列。ベクトルは "U" → (nx, ny, nz, 3)。
        非構造では (n_cells,) / (n_cells, 3)
    summary : Mapping[str, Any]
        YAML に書く実行サマリ（収束・反復・残差など）
    requests : tuple[OutputRequest, ...]
        ``*OUTPUT`` 要求。空なら NPZ で全変数
    """

    job_name: str
    output_dir: str
    grid: StructuredGridMap | None
    fields: Mapping[str, np.ndarray]
    summary: Mapping[str, Any] = field(default_factory=dict)
    requests: tuple[OutputRequest, ...] = ()
    title: str | None = None  # HTML のページタイトル（None なら job_name）
    mesh: MeshData | None = None


@dataclass(frozen=True)
class FieldOutputResult:
    paths: tuple[str, ...]


def _selected_variables(requests: tuple[OutputRequest, ...], available: list[str]) -> list[str]:
    wanted: list[str] = []
    for req in requests:
        for v in req.variables:
            name = VARIABLE_ALIASES.get(v.upper())
            if name is None:
                raise ValueError(
                    f"*OUTPUT の変数 {v!r} は未対応（{sorted(set(VARIABLE_ALIASES))}）"
                )
            if name == "res_*":
                wanted.extend(a for a in available if a.startswith("res_") and a not in wanted)
            elif name in available and name not in wanted:
                wanted.append(name)
    if not wanted:
        return list(available)
    return wanted


def _messi_available() -> bool:
    """messi が import できるか（自動 HTML 出力の判定）."""
    try:
        import messi  # noqa: F401
    except ImportError:
        return False
    return True


def _formats(requests: tuple[OutputRequest, ...]) -> set[OutputFormat]:
    fmts: set[OutputFormat] = {OutputFormat.NPZ}
    for req in requests:
        fmts.update(req.formats)
    # FORMAT= をどこにも書いていなければ、messi がある環境では HTML も自動で出す。
    if not any(req.formats_explicit for req in requests) and _messi_available():
        fmts.add(OutputFormat.HTML)
    return fmts


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if v is None:
        return "null"
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*?|<>=!%@`'\"\n") or s.strip() != s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return s


def dump_yaml(data: Any, indent: int = 0) -> str:
    """依存なしの簡易 YAML 出力（dict / list / スカラーのみ）."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(data, Mapping):
        if not data:
            return pad + "{}"
        for k, v in data.items():
            if isinstance(v, (Mapping, list, tuple)) and len(v) > 0:
                lines.append(f"{pad}{k}:")
                lines.append(dump_yaml(v, indent + 1))
            else:
                if isinstance(v, (Mapping, list, tuple)):
                    lines.append(f"{pad}{k}: " + ("{}" if isinstance(v, Mapping) else "[]"))
                else:
                    lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(data, (list, tuple)):
        if not data:
            return pad + "[]"
        for v in data:
            if isinstance(v, (Mapping, list, tuple)) and len(v) > 0:
                body = dump_yaml(v, indent + 1)
                first, _, rest = body.partition("\n")
                lines.append(f"{pad}- {first.strip()}")
                if rest:
                    lines.append(rest)
            else:
                lines.append(f"{pad}- {_yaml_scalar(v)}")
        return "\n".join(lines)
    return pad + _yaml_scalar(data)


def git_commit_hash() -> str | None:
    """作業ツリーの git コミットハッシュ（取得できなければ None）."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_vtk_rectilinear(
    path: Path, grid: StructuredGridMap, fields: Mapping[str, np.ndarray], names: list[str]
) -> None:
    """VTK legacy ASCII の RECTILINEAR_GRID + CELL_DATA を書く."""
    nx, ny, nz = grid.dimensions
    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("ykep .inp field output\nASCII\nDATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")
        for label, lines in (("X", grid.x_lines), ("Y", grid.y_lines), ("Z", grid.z_lines)):
            f.write(f"{label}_COORDINATES {len(lines)} double\n")
            f.write(" ".join(repr(float(v)) for v in lines) + "\n")
        f.write(f"CELL_DATA {nx * ny * nz}\n")
        for name in names:
            arr = np.asarray(fields[name], dtype=float)
            if arr.ndim == 4:
                f.write(f"VECTORS {name} double\n")
                flat = arr.reshape(-1, arr.shape[-1], order="F")
                for row in flat:
                    f.write(" ".join(repr(float(v)) for v in row) + "\n")
            else:
                f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                for v in arr.ravel(order="F"):
                    f.write(repr(float(v)) + "\n")


_VTK_CELL_TYPE_DEFAULT = 12  # VTK_HEXAHEDRON


def write_vtk_unstructured(
    path: Path, mesh: MeshData, fields: Mapping[str, np.ndarray], names: list[str]
) -> None:
    """VTK legacy ASCII の UNSTRUCTURED_GRID + CELL_DATA を書く（六面体など固定節点数のセル）."""
    coords = np.asarray(mesh.node_coords, dtype=float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    conn = np.asarray(mesh.connectivity, dtype=np.int64)
    n_cells = conn.shape[0]
    types = (
        np.asarray(mesh.cell_types, dtype=np.int64)
        if mesh.cell_types is not None
        else np.full(n_cells, _VTK_CELL_TYPE_DEFAULT)
    )
    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("ykep .inp field output\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {coords.shape[0]} double\n")
        for row in coords:
            f.write(" ".join(repr(float(v)) for v in row) + "\n")
        counts = np.sum(conn >= 0, axis=1)
        f.write(f"CELLS {n_cells} {int(np.sum(counts + 1))}\n")
        for row, c in zip(conn, counts, strict=True):
            f.write(f"{int(c)} " + " ".join(str(int(v)) for v in row[: int(c)]) + "\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for t in types:
            f.write(f"{int(t)}\n")
        f.write(f"CELL_DATA {n_cells}\n")
        for name in names:
            arr = np.asarray(fields[name], dtype=float)
            if arr.ndim == 2:
                f.write(f"VECTORS {name} double\n")
                for row in arr:
                    f.write(" ".join(repr(float(v)) for v in row) + "\n")
            else:
                f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                for v in arr.reshape(-1):
                    f.write(repr(float(v)) + "\n")


def _write_html(inp: FieldOutputInput, names: list[str], out_dir: Path) -> str | None:
    """``<job>.html``（messi mirador）を書く。messi が無ければ警告して None."""
    html_path = out_dir / f"{inp.job_name}.html"
    if inp.grid is not None:
        mirador_input = MiradorExportInput(
            x_lines=inp.grid.x_lines,
            y_lines=inp.grid.y_lines,
            z_lines=inp.grid.z_lines,
            fields={n: inp.fields[n] for n in names},
            output_path=str(html_path),
            title=inp.title or inp.job_name,
        )
    else:
        mirador_input = MiradorExportInput(
            x_lines=None,
            y_lines=None,
            z_lines=None,
            fields={n: inp.fields[n] for n in names},
            output_path=str(html_path),
            title=inp.title or inp.job_name,
            auto_slices=False,
            mesh=inp.mesh,
        )
    try:
        MiradorExportProcess().execute(mirador_input)
    except MiradorUnavailableError as exc:
        warnings.warn(f"FORMAT=HTML をスキップ: {exc}", RuntimeWarning, stacklevel=2)
        return None
    return str(html_path)


def write_field_output(inp: FieldOutputInput) -> FieldOutputResult:
    out_dir = Path(inp.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    available = list(inp.fields)
    names = _selected_variables(inp.requests, available)
    fmts = _formats(inp.requests)
    paths: list[str] = []

    if inp.grid is None and inp.mesh is None:
        raise ValueError("FieldOutputInput には grid か mesh のどちらかが必要です")
    npz_path = out_dir / f"{inp.job_name}.npz"
    arrays: dict[str, np.ndarray] = {}
    if inp.grid is not None:
        arrays["x_lines"] = inp.grid.x_lines
        arrays["y_lines"] = inp.grid.y_lines
        arrays["z_lines"] = inp.grid.z_lines
    else:
        assert inp.mesh is not None
        arrays["node_coords"] = np.asarray(inp.mesh.node_coords)
        arrays["connectivity"] = np.asarray(inp.mesh.connectivity)
        if inp.mesh.cell_types is not None:
            arrays["cell_types"] = np.asarray(inp.mesh.cell_types)
    for name in names:
        arrays[name] = np.asarray(inp.fields[name])
    np.savez(npz_path, **arrays)
    paths.append(str(npz_path))

    if OutputFormat.VTK in fmts:
        vtk_path = out_dir / f"{inp.job_name}.vtk"
        if inp.grid is not None:
            write_vtk_rectilinear(vtk_path, inp.grid, inp.fields, names)
        else:
            assert inp.mesh is not None
            write_vtk_unstructured(vtk_path, inp.mesh, inp.fields, names)
        paths.append(str(vtk_path))

    if OutputFormat.HTML in fmts:
        html_path = _write_html(inp, names, out_dir)
        if html_path is not None:
            paths.append(html_path)

    summary = dict(inp.summary)
    summary["output_files"] = [Path(p).name for p in paths] + [f"{inp.job_name}.yaml"]
    summary["variables"] = names
    yaml_path = out_dir / f"{inp.job_name}.yaml"
    yaml_path.write_text(dump_yaml(summary) + "\n", encoding="utf-8")
    paths.append(str(yaml_path))
    return FieldOutputResult(paths=tuple(paths))


class InpOutputWriterProcess(PostProcess["FieldOutputInput", "FieldOutputResult"]):
    """``*OUTPUT, FIELD`` に従って NPZ / YAML / VTK を書き出す PostProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpOutputWriter",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [MiradorExportProcess]

    def process(self, input_data: FieldOutputInput) -> FieldOutputResult:
        return write_field_output(input_data)
