"""構造格子の解析結果を messi ``mirador``（three.js ビューア）で 3D レンダリングする.

ykep の解析結果は直交構造格子のセル値（``(nx, ny, nz)`` のスカラー、``(nx, ny, nz, 3)`` の
ベクトル）である。本モジュールはそれを messi の六面体メッシュ（C3D8）に写し、セル値を
要素場として載せて、ブラウザで回せる自己完結 HTML を書き出す。

- 外皮は ``domain`` elset。断面（スラブ）は 1 セル厚の別 elset（``x=0.05`` など）にし、
  ビューアの凡例で ``domain`` を隠すと断面の場が見える（既定で ``domain`` は非表示）
- ベクトル場（速度）はセル中心の矢印。スライダーで長さを変えられる
- messi は任意依存。未導入なら :class:`MiradorUnavailableError`

設計文書: ``docs/design/mirador-export.md``
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess

if TYPE_CHECKING:
    from xkep_cae_fluid.core.mesh import StructuredMeshResult
    from xkep_cae_fluid.heat_transfer.data import HeatTransferResult
    from xkep_cae_fluid.natural_convection.data import NaturalConvectionResult

_AXES = ("x", "y", "z")


class MiradorUnavailableError(ImportError):
    """messi が import できない（``pip install messi`` が必要）."""


@dataclass(frozen=True)
class SlicePlane:
    """断面スラブの指定（1 セル厚）.

    Parameters
    ----------
    axis : str
        ``"x"`` / ``"y"`` / ``"z"``
    position : float | None
        断面の座標。この座標を含むセル層を選ぶ（``index`` が無いとき）。
        両方 ``None`` なら中央のセル層
    index : int | None
        セル添字（0 始まり）。``position`` より優先
    name : str | None
        elset 名。``None`` なら ``"x=0.0458"`` のようにセル中心座標から作る
    """

    axis: str
    position: float | None = None
    index: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.axis not in _AXES:
            raise ValueError(f"axis は x/y/z のいずれか: {self.axis!r}")


@dataclass(frozen=True)
class MiradorExportInput:
    """:class:`MiradorExportProcess` の入力.

    Parameters
    ----------
    x_lines, y_lines, z_lines : np.ndarray
        格子線座標（昇順、長さ n+1）。``z_lines`` が 1 点だけの 2D 結果は
        面内セル幅の平均で 1 層に押し出す
    fields : Mapping[str, np.ndarray]
        名前 → セル値。スカラー ``(nx, ny, nz)``、ベクトル ``(nx, ny, nz, 3)``。
        2D の ``(nx, ny)`` / ``(nx, ny, 2)`` も可（z 方向 1 層、w=0 として扱う）
    output_path : str
        書き出す HTML のパス
    title : str
        ページタイトル
    slices : tuple[SlicePlane, ...]
        断面スラブ。空で ``auto_slices`` が真なら、セル数 3 以上の各軸の中央に 1 枚ずつ
    auto_slices : bool
        ``slices`` が空のとき中央断面を自動で入れる
    mask : np.ndarray | None
        ``(nx, ny, nz)`` の bool。False のセルは描かない（固体・ガラスなどを除く）
    vector_field : str | None
        矢印にするベクトル場の名前。``None`` なら最初のベクトル場、``""`` で矢印なし
    vector_scale : float | None
        単位大きさ当たりの矢印長 [m]。``None`` なら messi の自動（最大矢印 = セル代表長×1.5）
    init_mode : str | None
        最初に表示する場。``None`` なら ``T`` → ``P`` → 最初のスカラー → ``|U|`` の順
    hide_domain : bool
        開いた時点で ``domain``（外皮）を非表示にして断面を見せる（断面が無ければ表示）
    domain_name : str
        外皮 elset の名前
    panel_collapsed : bool
        ビューアの操作パネル（左上）を畳んだ状態で開く（「▾/▸」ボタンか ``h`` キーで開閉）
    cut_plane : tuple[tuple[float, float, float], float] | None
        開いた時点で有効にする**任意平面の断面**（messi の view cut）``((nx, ny, nz), d)``。
        平面 ``n·x = d`` で ``n·x ≤ d`` 側を残し、切り口をセル値で着色する（ビューアの
        「断面」チェック / ``c`` キーで on/off、法線・位置・反転はパネルで変更可）。
        指定時は外皮を隠さない（``hide_domain`` は無視。切った立体として見せる）
    verbose : bool
        messi の概要表示
    """

    x_lines: np.ndarray
    y_lines: np.ndarray
    z_lines: np.ndarray
    fields: Mapping[str, np.ndarray]
    output_path: str
    title: str = "ykep result"
    slices: tuple[SlicePlane, ...] = ()
    auto_slices: bool = True
    mask: np.ndarray | None = None
    vector_field: str | None = None
    vector_scale: float | None = None
    init_mode: str | None = None
    hide_domain: bool = True
    domain_name: str = "domain"
    panel_collapsed: bool = False
    cut_plane: tuple[tuple[float, float, float], float] | None = None
    verbose: bool = False


@dataclass(frozen=True)
class MiradorExportResult:
    """:class:`MiradorExportProcess` の出力（HTML に埋め込んだ統計を読み戻して照合可能）."""

    path: str
    n_cells: int
    n_nodes: int
    n_triangles: int
    field_names: tuple[str, ...]
    slice_names: tuple[str, ...]
    n_vectors: int
    init_mode: str
    n_section_cells: int = 0  # 断面（view cut）用に埋め込んだセル数（古い messi では 0）


@dataclass(frozen=True)
class StructuredHexMesh:
    """構造格子 → 六面体メッシュの中間表現（messi 非依存、テストで検証しやすい形）.

    Parameters
    ----------
    nodes : np.ndarray
        ``(n_nodes, 4)``: ラベル, x, y, z。節点添字は i が最速
    elements : np.ndarray
        ``(n_cells, 9)``: ラベル + 8 節点（Abaqus C3D8 順 = VTK_HEXAHEDRON 順）
    cell_labels : np.ndarray
        ``(nx, ny, nz)`` のセルラベル（mask で除いたセルは 0）
    elsets : Mapping[str, np.ndarray]
        elset 名 → 要素ラベル配列（``domain`` + 断面スラブ）
    """

    nodes: np.ndarray
    elements: np.ndarray
    cell_labels: np.ndarray
    elsets: Mapping[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 格子 → 六面体メッシュ
# ---------------------------------------------------------------------------


def _as_lines(lines: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(lines, dtype=float).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"{name} が空です")
    if arr.size >= 2 and np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} は昇順である必要があります")
    return arr


def _normalize_lines(
    x_lines: np.ndarray, y_lines: np.ndarray, z_lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """格子線を整え、z が 1 点だけ（2D 結果）なら面内セル幅の平均で 1 層押し出す."""
    x = _as_lines(x_lines, "x_lines")
    y = _as_lines(y_lines, "y_lines")
    z = _as_lines(z_lines, "z_lines")
    if x.size < 2 or y.size < 2:
        raise ValueError("x_lines / y_lines は 2 点以上（セル 1 個以上）が必要です")
    if z.size < 2:
        h = float(np.mean(np.concatenate([np.diff(x), np.diff(y)])))
        z = np.array([z[0], z[0] + h])
    return x, y, z


def _normalize_field(name: str, arr: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """場の配列を ``(nx, ny, nz)`` か ``(nx, ny, nz, 3)`` に揃える（2D 入力は 1 層へ）."""
    nx, ny, nz = dims
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        a = a[:, :, None]
    elif a.ndim == 3 and a.shape[:2] == (nx, ny) and nz == 1 and a.shape[2] in (2, 3):
        # (nx, ny, 2|3) はベクトルの 2D 版（ただし nz == a.shape[2] のときは曖昧なので
        # 3 軸目をセル添字として扱う: (nx, ny, nz) を優先）
        if a.shape[2] != nz:
            a = a[:, :, None, :]
    if a.ndim == 4 and a.shape[-1] == 2:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], axis=-1)
    if a.ndim == 3 and a.shape != (nx, ny, nz):
        raise ValueError(f"場 {name!r} の形状 {a.shape} が格子 {dims} と一致しません")
    if a.ndim == 4 and a.shape != (nx, ny, nz, 3):
        raise ValueError(f"ベクトル場 {name!r} の形状 {a.shape} は (nx, ny, nz, 3) が必要です")
    if a.ndim not in (3, 4):
        raise ValueError(f"場 {name!r} の次元 {a.ndim} は未対応（3 か 4）")
    return a


def _slice_index(plane: SlicePlane, lines: np.ndarray, n: int) -> int:
    if plane.index is not None:
        idx = int(plane.index)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise ValueError(f"断面 {plane.axis} の index {plane.index} が範囲外（0..{n - 1}）")
        return idx
    if plane.position is not None:
        pos = float(plane.position)
        if pos < lines[0] or pos > lines[-1]:
            raise ValueError(
                f"断面 {plane.axis}={pos} が格子範囲 [{lines[0]}, {lines[-1]}] の外です"
            )
        return int(min(max(np.searchsorted(lines, pos, side="right") - 1, 0), n - 1))
    return n // 2


def resolve_slices(
    slices: tuple[SlicePlane, ...],
    lines: tuple[np.ndarray, np.ndarray, np.ndarray],
    auto: bool,
) -> list[tuple[str, int, int]]:
    """断面指定を ``(elset 名, 軸番号, セル添字)`` に解決する（自動中央断面を含む）."""
    dims = tuple(len(ln) - 1 for ln in lines)
    planes = list(slices)
    if not planes and auto:
        planes = [SlicePlane(axis=ax) for ax, n in zip(_AXES, dims, strict=True) if n >= 3]
    out: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for pl in planes:
        ai = _AXES.index(pl.axis)
        idx = _slice_index(pl, lines[ai], dims[ai])
        center = 0.5 * (lines[ai][idx] + lines[ai][idx + 1])
        name = pl.name or f"{pl.axis}={center:.4g}"
        if name in seen:
            raise ValueError(f"断面 elset 名が重複しています: {name!r}")
        seen.add(name)
        out.append((name, ai, idx))
    return out


def build_structured_hex_mesh(
    x_lines: np.ndarray,
    y_lines: np.ndarray,
    z_lines: np.ndarray,
    mask: np.ndarray | None = None,
    slices: list[tuple[str, int, int]] | None = None,
    domain_name: str = "domain",
) -> StructuredHexMesh:
    """格子線から六面体メッシュを組む（i 最速のラベル、mask で間引き、断面を別 elset に）."""
    x, y, z = x_lines, y_lines, z_lines
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    px, py = nx + 1, ny + 1
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    n_nodes = px * py * (nz + 1)
    nodes = np.column_stack(
        [np.arange(1, n_nodes + 1, dtype=float), xx.ravel(), yy.ravel(), zz.ravel()]
    )

    def nid(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return 1 + i + px * j + px * py * k

    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    labels_flat = np.arange(1, i.size + 1)
    conn = np.column_stack(
        [
            labels_flat,
            nid(i, j, k),
            nid(i + 1, j, k),
            nid(i + 1, j + 1, k),
            nid(i, j + 1, k),
            nid(i, j, k + 1),
            nid(i + 1, j, k + 1),
            nid(i + 1, j + 1, k + 1),
            nid(i, j + 1, k + 1),
        ]
    ).astype(np.int64)

    keep = np.ones(i.size, dtype=bool)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2:
            m = m[:, :, None]
        if m.shape != (nx, ny, nz):
            raise ValueError(f"mask の形状 {m.shape} が格子 ({nx}, {ny}, {nz}) と一致しません")
        keep = m.ravel(order="F")

    # 断面スラブ: 先に指定したものが優先（重なるセルは最初の断面に入る）。
    owner = np.zeros(i.size, dtype=np.int64)  # 0 = domain, s+1 = slices[s]
    ijk = (i, j, k)
    for s, (_name, ai, idx) in enumerate(slices or []):
        sel = (ijk[ai] == idx) & keep & (owner == 0)
        owner[sel] = s + 1

    cell_labels = np.where(keep, labels_flat, 0).reshape(nx, ny, nz, order="F")
    elsets: dict[str, np.ndarray] = {domain_name: labels_flat[keep & (owner == 0)]}
    for s, (name, _ai, _idx) in enumerate(slices or []):
        elsets[name] = labels_flat[keep & (owner == s + 1)]
    return StructuredHexMesh(
        nodes=nodes, elements=conn[keep], cell_labels=cell_labels, elsets=elsets
    )


# ---------------------------------------------------------------------------
# 入力アダプタ（ソルバー結果 / メッシュ / NPZ → 格子線と場）
# ---------------------------------------------------------------------------


def fields_from_natural_convection(result: NaturalConvectionResult) -> dict[str, np.ndarray]:
    """:class:`NaturalConvectionResult` → ``{"U": (nx,ny,nz,3), "P", "T", 追加スカラー, 残差マップ res_*}``."""
    fields: dict[str, np.ndarray] = {
        "U": np.stack([result.u, result.v, result.w], axis=-1),
        "P": np.asarray(result.p),
        "T": np.asarray(result.T),
    }
    for name, arr in getattr(result, "extra_scalars", {}).items():
        fields[str(name)] = np.asarray(arr)
    for name, arr in getattr(result, "residual_fields", {}).items():  # 残差マップ
        fields[str(name)] = np.asarray(arr)
    return fields


def fields_from_heat_transfer(result: HeatTransferResult) -> dict[str, np.ndarray]:
    """:class:`HeatTransferResult` → ``{"T": (nx,ny,nz), "res_T": …}``."""
    fields = {"T": np.asarray(result.T)}
    for name, arr in getattr(result, "residual_fields", {}).items():
        fields[str(name)] = np.asarray(arr)
    return fields


def lines_from_structured_mesh(
    mesh: StructuredMeshResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:class:`StructuredMeshResult`（``dx, dy, dz`` + 節点座標）→ 格子線 3 本."""
    coords = np.asarray(mesh.mesh.node_coords, dtype=float)
    origin = coords.min(axis=0)
    x = origin[0] + np.concatenate([[0.0], np.cumsum(mesh.dx)])
    y = origin[1] + np.concatenate([[0.0], np.cumsum(mesh.dy)])
    z0 = origin[2] if coords.shape[1] >= 3 else 0.0
    z = z0 + np.concatenate([[0.0], np.cumsum(mesh.dz)])
    return x, y, z


def load_npz_fields(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """``ykep`` が書いた ``<job>.npz`` を読み ``(x_lines, y_lines, z_lines, fields)`` を返す."""
    with np.load(path) as data:
        try:
            x, y, z = data["x_lines"], data["y_lines"], data["z_lines"]
        except KeyError as exc:
            raise ValueError(f"{path}: x_lines / y_lines / z_lines がありません") from exc
        fields = {k: np.asarray(data[k]) for k in data.files if not k.endswith("_lines")}
    return x, y, z, fields


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------


def _import_messi() -> Any:
    try:
        import messi
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise MiradorUnavailableError(
            "messi が import できません。3D レンダリングには messi（gyp0bt/messi）が必要です: "
            "pip install -e <messi のパス>"
        ) from exc
    return messi


def _pick_init_mode(scalar_names: list[str], vector_names: list[str]) -> str | None:
    for pref in ("T", "P"):
        if pref in scalar_names:
            return pref
    if scalar_names:
        return scalar_names[0]
    if vector_names:
        return f"|{vector_names[0]}|"
    return None


def _embedded_data(html: str) -> dict[str, Any]:
    """messi が HTML に埋め込んだ ``const DATA = {...};`` を読み戻す（統計の照合用）."""
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def _normalize_cut_plane(
    cut_plane: tuple[tuple[float, float, float], float] | None,
) -> tuple[tuple[float, float, float], float] | None:
    """``cut_plane`` を検証して float 化する（messi 非依存。零法線・非数は ValueError）."""
    if cut_plane is None:
        return None
    normal, d = cut_plane
    n = tuple(float(v) for v in normal)
    if len(n) != 3 or not all(np.isfinite(n)) or not any(n):
        raise ValueError(f"cut_plane の法線が不正です（零ベクトル / 非数）: {normal!r}")
    if not np.isfinite(float(d)):
        raise ValueError(f"cut_plane の位置 d が非数です: {d!r}")
    return (n[0], n[1], n[2]), float(d)


def export_mirador(inp: MiradorExportInput) -> MiradorExportResult:
    """:class:`MiradorExportProcess` の本体（messi の Mesher を組んで ``export_html``）."""
    # 入力検証と六面体メッシュ構築は messi 非依存なので先に済ませる（messi 未導入でも
    # 形状不一致などは ValueError で返す。MiradorUnavailableError は HTML を書く段階だけ）。
    x, y, z = _normalize_lines(inp.x_lines, inp.y_lines, inp.z_lines)
    dims = (len(x) - 1, len(y) - 1, len(z) - 1)
    fields = {str(k): _normalize_field(str(k), v, dims) for k, v in inp.fields.items()}
    if not fields:
        raise ValueError("fields が空です（少なくとも 1 つの場が必要）")
    scalar_names = [k for k, v in fields.items() if v.ndim == 3]
    vector_names = [k for k, v in fields.items() if v.ndim == 4]

    cut_plane = _normalize_cut_plane(inp.cut_plane)
    slices = resolve_slices(inp.slices, (x, y, z), inp.auto_slices)
    hexmesh = build_structured_hex_mesh(
        x, y, z, mask=inp.mask, slices=slices, domain_name=inp.domain_name
    )
    if hexmesh.elements.shape[0] == 0:
        raise ValueError("描画するセルがありません（mask で全て除外されています）")

    messi = _import_messi()
    mesh = messi.Mesher.load(verbose=False)
    mesh.add_nodes(name="global", arr=hexmesh.nodes)
    for name, labels in hexmesh.elsets.items():
        if labels.size == 0:
            continue
        rows = hexmesh.elements[np.isin(hexmesh.elements[:, 0], labels)]
        mesh.add_elements(name=name, type="C3D8", arr=rows)

    keep = hexmesh.cell_labels.ravel(order="F") > 0
    labels = hexmesh.cell_labels.ravel(order="F")[keep]
    for name, arr in fields.items():
        if arr.ndim == 3:
            vals = arr.ravel(order="F")[keep]
            mesh.set_element_field(name, dict(zip(labels.tolist(), vals.tolist(), strict=True)))
        else:
            vals = arr.reshape(-1, 3, order="F")[keep]
            mesh.set_element_field(
                name, {int(lab): row.copy() for lab, row in zip(labels, vals, strict=True)}
            )
            mesh.element_fields[name].components = ["x", "y", "z"]

    vector_field = inp.vector_field
    if vector_field is None:
        vector_field = vector_names[0] if vector_names else ""
    elif vector_field and vector_field not in vector_names:
        raise ValueError(
            f"vector_field {vector_field!r} はベクトル場ではありません: {vector_names}"
        )
    init_mode = inp.init_mode or _pick_init_mode(scalar_names, vector_names)
    # 任意平面の断面（view cut）を使うときは外皮を隠さない（切った立体として見せる）。
    hidden = (inp.domain_name,) if (inp.hide_domain and slices and inp.cut_plane is None) else ()

    out = Path(inp.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # panel_collapsed / cut_plane は messi 0.10 で追加された引数。既定（False / None）の
    # ときは渡さず、引数を知らない古い messi でも動くようにしておく。
    extra: dict[str, Any] = {}
    if inp.panel_collapsed:
        extra["panel_collapsed"] = True
    if cut_plane is not None:
        extra["cut_plane"] = cut_plane
    mesh.export_html(
        str(out),
        title=inp.title,
        verbose=inp.verbose,
        vector_field=vector_field,
        vector_scale=inp.vector_scale,
        init_mode=init_mode,
        hidden_groups=hidden,
        **extra,
    )
    data = _embedded_data(out.read_text(encoding="utf-8"))
    return MiradorExportResult(
        path=str(out),
        n_cells=int(hexmesh.elements.shape[0]),
        n_nodes=int(hexmesh.nodes.shape[0]),
        n_triangles=int(data["nTriangles"]),
        field_names=tuple(str(s) for s in data.get("fieldNames", [])),
        slice_names=tuple(name for name, _ai, _idx in slices),
        n_vectors=int(data.get("vectors", {}).get("n", 0)),
        init_mode=str(data.get("initMode", "")),
        n_section_cells=int(data.get("nCells", 0)),
    )


class MiradorExportProcess(PostProcess["MiradorExportInput", "MiradorExportResult"]):
    """構造格子の解析結果を messi ``mirador`` の three.js HTML として書き出す PostProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="MiradorExport",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/mirador-export.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: MiradorExportInput) -> MiradorExportResult:
        return export_mirador(input_data)
