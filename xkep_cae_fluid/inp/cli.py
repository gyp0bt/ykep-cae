"""``ykep`` コマンド（Abaqus 風の引数: ``ykep -j=case.inp int``）.

書式::

    ykep -j=<job>[.inp] [int|interactive] [-o=<dir>] [-p name=value ...] [--check]
                       [--mesh=auto|structured|unstructured]
    ykep job=<job> interactive
    ykep -j=<job> view [-o=<dir>] [--slice=<axis>=<pos> ...] [--no-slices] [--no-vectors] [--collapse-panel]
                       [--cut=<axis>=<pos> | --cut=<nx>,<ny>,<nz>,<d>]

- ``-j=`` / ``job=``: 入力ファイル（拡張子省略時は .inp を補う）
- ``int`` / ``interactive``: 反復ログを端末にも表示（無指定ならファイルのみ）
- ``-o=`` / ``out=``: 出力ディレクトリ（既定は .inp と同じ場所）
- ``-p name=value``: ``*PARAMETER`` の初期値を与える（.inp 内の定義が優先）
- ``--check``: 解析を実行せず読込・格子復元・マッピングのみ検証
- ``--mesh=``: メッシュ経路。``auto``（既定: 箱格子なら構造格子、そうでなければ非構造）、
  ``structured``（箱格子を強制）、``unstructured``（面ベース非構造を強制。``*HEAT TRANSFER`` / ``*DARCY``）
- ``view``: 解析を実行せず、既にある ``<out>/<job>.npz`` から messi mirador の
  3D ビューア ``<out>/<job>.html`` を書く（``--slice=x=0.05`` で断面スラブ、複数可。
  ``--cut=z=0.5`` / ``--cut=1,1,0,0.3`` で任意平面の断面（view cut）を有効にして開く。
  ``--cut`` を付けると中央断面スラブの自動挿入は行わない。明示した ``--slice`` は残る）

ログは常に ``<out>/<job>.log`` に残す（CLAUDE.md のログ出力ルール）。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from xkep_cae_fluid.inp.parameters import ParameterError, ParameterValue, safe_eval
from xkep_cae_fluid.inp.runner import InpCaseRunnerProcess, InpJobInput
from xkep_cae_fluid.post.mirador import (
    MiradorExportInput,
    MiradorExportProcess,
    SlicePlane,
    load_npz_fields,
    load_npz_mesh,
)

USAGE = (
    "usage: ykep -j=<job>[.inp] [int] [-o=<dir>] [-p name=value ...] [--check]"
    " [--mesh=auto|structured|unstructured]\n"
    "       ykep -j=<job> view [-o=<dir>] [--slice=<axis>=<pos> ...] [--no-slices] [--no-vectors]"
    " [--collapse-panel] [--cut=<axis>=<pos> | --cut=<nx>,<ny>,<nz>,<d>]"
)

CutPlaneSpec = tuple[tuple[float, float, float], float]


class CliError(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


@dataclass(frozen=True)
class CliArgs:
    """解釈済みのコマンドライン引数."""

    job: InpJobInput
    interactive: bool = False
    view: bool = False
    slices: tuple[SlicePlane, ...] = ()
    no_slices: bool = False
    no_vectors: bool = False
    collapse_panel: bool = False
    cut: CutPlaneSpec | None = None


def _parse_slice(text: str) -> SlicePlane:
    """``x=0.05`` / ``z=3i``（セル添字）を :class:`SlicePlane` にする."""
    axis, _, value = text.partition("=")
    axis = axis.strip().lower()
    value = value.strip()
    if axis not in ("x", "y", "z") or not value:
        raise CliError(f"--slice は <axis>=<座標> か <axis>=<添字>i の形式: {text!r}")
    try:
        if value.endswith("i"):
            return SlicePlane(axis=axis, index=int(value[:-1]))
        return SlicePlane(axis=axis, position=float(value))
    except ValueError as exc:
        raise CliError(f"--slice の値を解釈できません: {text!r}") from exc


_CUT_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _parse_cut(text: str) -> CutPlaneSpec:
    """``z=0.5`` / ``-x=0.1`` / ``nx,ny,nz,d`` を ``((nx, ny, nz), d)`` にする.

    ``axis=pos`` は軸に垂直な平面（``n·x ≤ pos`` 側を残す）。``-axis=pos`` は法線を
    反転して残す側を入れ替える。4 つの数は任意平面 ``n·x = d``。
    """
    t = text.strip()
    if "=" in t:
        axis, _, value = t.partition("=")
        axis = axis.strip().lower()
        sign = 1.0
        if axis[:1] in ("+", "-"):
            sign = -1.0 if axis[0] == "-" else 1.0
            axis = axis[1:]
        if axis not in _CUT_AXES or not value.strip():
            raise CliError(f"--cut は <axis>=<座標> か <nx>,<ny>,<nz>,<d> の形式: {text!r}")
        try:
            d = float(value)
        except ValueError as exc:
            raise CliError(f"--cut の値を解釈できません: {text!r}") from exc
        n = _CUT_AXES[axis]
        return (sign * n[0], sign * n[1], sign * n[2]), sign * d
    parts = [v.strip() for v in t.split(",")]
    if len(parts) != 4:
        raise CliError(f"--cut は <axis>=<座標> か <nx>,<ny>,<nz>,<d> の形式: {text!r}")
    try:
        nx, ny, nz, d = (float(v) for v in parts)
    except ValueError as exc:
        raise CliError(f"--cut の値を解釈できません: {text!r}") from exc
    if nx == 0.0 and ny == 0.0 and nz == 0.0:
        raise CliError(f"--cut の法線が零ベクトルです: {text!r}")
    return (nx, ny, nz), d


def parse_args(argv: list[str]) -> CliArgs | None:
    """引数を :class:`CliArgs` に変換する（``--help`` なら None）."""
    job: str | None = None
    interactive = False
    view = False
    out: str | None = None
    check = False
    mesh_mode = "auto"
    no_slices = False
    no_vectors = False
    collapse_panel = False
    cut: CutPlaneSpec | None = None
    slices: list[SlicePlane] = []
    params: dict[str, ParameterValue] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        low = arg.lower()
        if low in ("-h", "--help", "help"):
            return None
        if low.startswith("-j=") or low.startswith("job="):
            job = arg.split("=", 1)[1]
        elif low in ("-j", "job") and i + 1 < len(argv):
            i += 1
            job = argv[i]
        elif low.startswith("-o=") or low.startswith("out=") or low.startswith("output="):
            out = arg.split("=", 1)[1]
        elif low in ("-o", "out") and i + 1 < len(argv):
            i += 1
            out = argv[i]
        elif low == "-p" and i + 1 < len(argv):
            i += 1
            name, _, expr = argv[i].partition("=")
            if not name.isidentifier() or not expr:
                raise CliError(f"-p は name=value 形式: {argv[i]!r}")
            try:
                params[name] = safe_eval(expr, params)
            except ParameterError as exc:
                raise CliError(str(exc)) from exc
        elif low in ("int", "interactive"):
            interactive = True
        elif low == "view":
            view = True
        elif low.startswith("--slice="):
            slices.append(_parse_slice(arg.split("=", 1)[1]))
        elif low == "--slice" and i + 1 < len(argv):
            i += 1
            slices.append(_parse_slice(argv[i]))
        elif low == "--no-slices":
            no_slices = True
        elif low == "--no-vectors":
            no_vectors = True
        elif low == "--collapse-panel":
            collapse_panel = True
        elif low.startswith("--cut="):
            cut = _parse_cut(arg.split("=", 1)[1])
        elif low == "--cut" and i + 1 < len(argv):
            i += 1
            cut = _parse_cut(argv[i])
        elif low == "--check":
            check = True
        elif low.startswith("--mesh="):
            mesh_mode = arg.split("=", 1)[1].strip().lower()
            if mesh_mode not in ("auto", "structured", "unstructured"):
                raise CliError(f"--mesh は auto / structured / unstructured: {arg!r}")
        else:
            raise CliError(f"不明な引数: {arg!r}\n{USAGE}")
        i += 1
    if job is None:
        raise CliError(f"-j=<job> が必要です\n{USAGE}")
    path = Path(job)
    if path.suffix.lower() != ".inp":
        path = path.with_name(path.name + ".inp")
    if not path.is_file():
        raise CliError(f"入力ファイルが見つかりません: {path}")
    return CliArgs(
        job=InpJobInput(
            path=str(path), output_dir=out, parameters=params, check_only=check, mesh_mode=mesh_mode
        ),
        interactive=interactive,
        view=view,
        slices=tuple(slices),
        no_slices=no_slices,
        no_vectors=no_vectors,
        collapse_panel=collapse_panel,
        cut=cut,
    )


def run_view(parsed: CliArgs) -> Path:
    """``view`` モード: ``<out>/<job>.npz`` → ``<out>/<job>.html``（解析は走らない）."""
    path = Path(parsed.job.path)
    out_dir = Path(parsed.job.output_dir) if parsed.job.output_dir else path.parent
    npz = out_dir / f"{path.stem}.npz"
    if not npz.is_file():
        raise CliError(
            f"{npz} がありません。先に `ykep -j={path} int` で解析して NPZ を作ってください"
        )
    x, y, z, fields = load_npz_fields(npz)
    mesh = load_npz_mesh(npz) if x is None else None
    html = out_dir / f"{path.stem}.html"
    if mesh is not None and parsed.slices:
        raise CliError("非構造格子の結果では --slice は使えません（--cut を使ってください）")
    MiradorExportProcess().execute(
        MiradorExportInput(
            x_lines=x,
            y_lines=y,
            z_lines=z,
            fields=fields,
            output_path=str(html),
            title=path.stem,
            slices=parsed.slices,
            # --cut（任意平面の断面）を使うときは中央スラブの自動挿入はしない。
            auto_slices=mesh is None and not parsed.no_slices and parsed.cut is None,
            vector_field="" if parsed.no_vectors else None,
            panel_collapsed=parsed.collapse_panel,
            cut_plane=parsed.cut,
            mesh=mesh,
        )
    )
    return html


def _configure_logging(log_path: Path, interactive: bool) -> None:
    """ルートロガーに ykep 用ハンドラを付ける（前回の ykep ハンドラだけ外す）."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        if getattr(h, "_ykep_handler", False):
            root.removeHandler(h)
            h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh._ykep_handler = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    if interactive:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        sh._ykep_handler = True  # type: ignore[attr-defined]
        root.addHandler(sh)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_args(args)
    except CliError as exc:
        print(exc.message, file=sys.stderr)
        return 2
    if parsed is None:
        print(USAGE)
        print(__doc__)
        return 0
    job, interactive = parsed.job, parsed.interactive
    path = Path(job.path)
    out_dir = Path(job.output_dir) if job.output_dir else path.parent
    _configure_logging(out_dir / f"{path.stem}.log", interactive)
    logger = logging.getLogger("ykep")
    logger.info("ykep %s", " ".join(args))
    if parsed.view:
        try:
            html = run_view(parsed)
        except CliError as exc:
            print(exc.message, file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - CLI 境界で全例外をメッセージ化
            logger.error("%s: %s", type(exc).__name__, exc)
            print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        logger.info("view: %s", html)
        print(f"VIEW: {html}")
        return 0
    try:
        result = InpCaseRunnerProcess().execute(job)
    except Exception as exc:  # noqa: BLE001 - CLI 境界で全例外をメッセージ化
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if job.check_only:
        print(f"CHECK OK: {result.job_name} ({len(result.steps)} step(s))")
        for s in result.steps:
            print(
                f"  step {s.step_name}: {s.family.value} → {s.summary.get('solver', {}).get('process')}"
            )
        return 0
    for s in result.steps:
        state = "CONVERGED" if s.converged else "NOT CONVERGED"
        print(
            f"{state}: step {s.step_name} ({s.family.value}), {s.n_iterations} iterations, "
            f"{s.elapsed_seconds:.2f} s"
        )
        for p in s.output_paths:
            print(f"  -> {p}")
    return 0 if result.converged else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
