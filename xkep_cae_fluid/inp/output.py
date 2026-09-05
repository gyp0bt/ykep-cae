"""``*OUTPUT, FIELD`` の書き出し（NPZ / YAML サマリ / VTK legacy）.

- ``<job>.npz``: 格子線と場（u, v, w, p, T …）。必ず出力
- ``<job>.yaml``: 収束情報・残差・実行条件（STA2 防止ルール: ログと照合できる形）
- ``<job>.vtk``: ``FORMAT=VTK`` 指定時。RECTILINEAR_GRID + CELL_DATA（依存ライブラリなし）
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.inp.case import OutputFormat, OutputRequest
from xkep_cae_fluid.inp.grid import StructuredGridMap

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
    grid : StructuredGridMap
    fields : Mapping[str, np.ndarray]
        内部変数名 → (nx, ny, nz) 配列。ベクトルは "U" → (nx, ny, nz, 3)
    summary : Mapping[str, Any]
        YAML に書く実行サマリ（収束・反復・残差など）
    requests : tuple[OutputRequest, ...]
        ``*OUTPUT`` 要求。空なら NPZ で全変数
    """

    job_name: str
    output_dir: str
    grid: StructuredGridMap
    fields: Mapping[str, np.ndarray]
    summary: Mapping[str, Any] = field(default_factory=dict)
    requests: tuple[OutputRequest, ...] = ()


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
            if name in available and name not in wanted:
                wanted.append(name)
    if not wanted:
        return list(available)
    return wanted


def _formats(requests: tuple[OutputRequest, ...]) -> set[OutputFormat]:
    fmts: set[OutputFormat] = {OutputFormat.NPZ}
    for req in requests:
        fmts.update(req.formats)
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


def write_field_output(inp: FieldOutputInput) -> FieldOutputResult:
    out_dir = Path(inp.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    available = list(inp.fields)
    names = _selected_variables(inp.requests, available)
    fmts = _formats(inp.requests)
    paths: list[str] = []

    npz_path = out_dir / f"{inp.job_name}.npz"
    arrays: dict[str, np.ndarray] = {
        "x_lines": inp.grid.x_lines,
        "y_lines": inp.grid.y_lines,
        "z_lines": inp.grid.z_lines,
    }
    for name in names:
        arrays[name] = np.asarray(inp.fields[name])
    np.savez(npz_path, **arrays)
    paths.append(str(npz_path))

    if OutputFormat.VTK in fmts:
        vtk_path = out_dir / f"{inp.job_name}.vtk"
        write_vtk_rectilinear(vtk_path, inp.grid, inp.fields, names)
        paths.append(str(vtk_path))

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
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: FieldOutputInput) -> FieldOutputResult:
        return write_field_output(input_data)
