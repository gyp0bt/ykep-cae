"""非構造化メッシュ読み込み Process（OpenFOAM polyMesh 互換）.

OpenFOAM の constant/polyMesh/ ディレクトリから
points, faces, owner, neighbour, boundary を読み込み、
MeshData として返す。ASCII / バイナリ両形式に対応。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import MeshData

# ---------------------------------------------------------------------------
# 入出力データ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolyMeshInput:
    """polyMesh 読み込みの入力.

    Parameters
    ----------
    mesh_dir : str
        polyMesh ディレクトリのパス（points, faces, owner, neighbour を含む）
    """

    mesh_dir: str


@dataclass(frozen=True)
class PolyMeshResult:
    """polyMesh 読み込みの出力.

    Parameters
    ----------
    mesh : MeshData
        読み込まれたメッシュデータ
    boundary_patches : dict[str, dict]
        境界パッチ情報（パッチ名 → {type, nFaces, startFace}）。
        面インデックス配列は ``mesh.boundary_patches`` にもある
    """

    mesh: MeshData
    boundary_patches: dict[str, dict]


# ---------------------------------------------------------------------------
# OpenFOAM ファイルパーサ
# ---------------------------------------------------------------------------


def _is_binary_format(raw: bytes) -> bool:
    """FoamFile ヘッダの format フィールドが 'binary' かどうか判定."""
    # ヘッダ部分だけテキストとして解釈（最初の 512 バイト程度）
    header = raw[: min(len(raw), 1024)].decode("ascii", errors="replace")
    for line in header.splitlines():
        stripped = line.strip().rstrip(";")
        parts = stripped.split()
        if len(parts) == 2 and parts[0] == "format" and parts[1] == "binary":
            return True
    return False


def _find_binary_data_offset(raw: bytes) -> int:
    """バイナリファイルにおけるデータ本体の開始オフセットを返す.

    ヘッダの後の件数 + '(' の次のバイトを返す。
    """
    # テキスト部分を読んでヘッダ＋件数をスキップ
    # '(' のバイト位置を見つける（FoamFile ヘッダの }  の後にある）
    header_end = raw.find(b"}")
    if header_end < 0:
        header_end = 0
    # 件数の後の '(' を探す
    paren = raw.find(b"(", header_end)
    if paren < 0:
        msg = "バイナリ polyMesh ファイルに '(' が見つかりません"
        raise ValueError(msg)
    return paren + 1


def _read_count_from_bytes(raw: bytes) -> int:
    """バイナリファイルからデータ件数を読み取る."""
    header = raw[: min(len(raw), 1024)].decode("ascii", errors="replace")
    lines = header.splitlines()
    i = 0
    n = len(lines)
    # ヘッダスキップ
    while i < n:
        if lines[i].strip().startswith("FoamFile"):
            while i < n and "}" not in lines[i]:
                i += 1
            i += 1
            break
        i += 1
    # 件数を探す
    while i < n:
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("//"):
            try:
                return int(stripped)
            except ValueError:
                pass
        i += 1
    msg = "バイナリ polyMesh ファイルからデータ件数を読み取れません"
    raise ValueError(msg)


def parse_points_binary(raw: bytes) -> np.ndarray:
    """バイナリ形式の points ファイルを解析."""
    n_points = _read_count_from_bytes(raw)
    offset = _find_binary_data_offset(raw)
    data = np.frombuffer(raw, dtype=np.float64, count=n_points * 3, offset=offset)
    return data.reshape(n_points, 3).copy()


def parse_label_list_binary(raw: bytes) -> np.ndarray:
    """バイナリ形式の owner/neighbour ファイルを解析.

    OpenFOAM は 32bit int (label) をデフォルトで使用。
    """
    n_items = _read_count_from_bytes(raw)
    offset = _find_binary_data_offset(raw)
    # 32bit int を試行し、データ範囲が妥当か確認
    remaining = len(raw) - offset
    if remaining >= n_items * 4:
        data32 = np.frombuffer(raw, dtype=np.int32, count=n_items, offset=offset)
        if n_items == 0 or (data32.min() >= 0 and data32.max() < 10_000_000):
            return data32.astype(np.int64).copy()
    # 64bit int にフォールバック
    if remaining >= n_items * 8:
        data64 = np.frombuffer(raw, dtype=np.int64, count=n_items, offset=offset)
        return data64.copy()
    msg = f"バイナリ label リストのサイズ不整合: {remaining} bytes, {n_items} items"
    raise ValueError(msg)


def parse_faces_binary(raw: bytes) -> list[list[int]]:
    """バイナリ形式の faces ファイルを解析.

    OpenFOAM の compactListList 形式:
    n_faces 個のインデックスリスト + 合計ノード数分のラベル。
    形式: [n_faces]([indices of size n_faces+1])([labels])
    """
    n_faces = _read_count_from_bytes(raw)
    offset = _find_binary_data_offset(raw)
    remaining = raw[offset:]

    # compactListList: まず (n_faces+1) 個のオフセット配列、次にラベル配列
    # ただし OpenFOAM のバイナリ faces は単純に連続して
    # n_nodes_per_face, node0, node1, ... の繰り返しの場合もある

    # 方式1: n_face 個のオフセットテーブル + ラベルデータ（compactListList）
    # 最初のバイトを int32 として読んで妥当性を確認
    idx_size = (n_faces + 1) * 4
    if len(remaining) >= idx_size:
        offsets = np.frombuffer(remaining[:idx_size], dtype=np.int32)
        total_nodes = int(offsets[-1])
        label_start = idx_size
        label_bytes = total_nodes * 4
        if (
            len(remaining) >= label_start + label_bytes
            and offsets[0] == 0
            and np.all(np.diff(offsets) >= 0)
        ):
            labels = np.frombuffer(
                remaining[label_start : label_start + label_bytes],
                dtype=np.int32,
            )
            faces: list[list[int]] = []
            for i in range(n_faces):
                start = int(offsets[i])
                end = int(offsets[i + 1])
                faces.append(labels[start:end].tolist())
            return faces

    # 方式2: 各面が [n_nodes, node0, node1, ...] として連続格納
    faces2: list[list[int]] = []
    pos = 0
    for _ in range(n_faces):
        if pos + 4 > len(remaining):
            break
        (n_nodes,) = struct.unpack_from("<i", remaining, pos)
        pos += 4
        if pos + n_nodes * 4 > len(remaining):
            break
        nodes = struct.unpack_from(f"<{n_nodes}i", remaining, pos)
        pos += n_nodes * 4
        faces2.append(list(nodes))
    return faces2


def _skip_header(lines: list[str]) -> int:
    """FoamFile ヘッダをスキップして本体の開始行インデックスを返す."""
    i = 0
    n = len(lines)
    # FoamFile ヘッダブロックをスキップ
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("FoamFile"):
            # { ... } ブロックを飛ばす
            while i < n and "}" not in lines[i]:
                i += 1
            i += 1  # } の次の行
            break
        i += 1
    return i


def _read_count(lines: list[str], start: int) -> tuple[int, int]:
    """データ件数を読み取り、開き括弧 '(' の次の行インデックスを返す."""
    i = start
    n = len(lines)
    count = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("//"):
            try:
                count = int(stripped)
                i += 1
                break
            except ValueError:
                pass
        i += 1
    # '(' を見つける
    while i < n:
        if "(" in lines[i]:
            i += 1
            break
        i += 1
    return count, i


def parse_points(text: str) -> np.ndarray:
    """OpenFOAM points ファイルを解析して (n_points, 3) 配列を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_points, data_start = _read_count(lines, start)

    points = np.zeros((n_points, 3))
    idx = 0
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        # "(x y z)" 形式
        line = line.strip("()")
        parts = line.split()
        if len(parts) == 3:
            points[idx] = [float(p) for p in parts]
            idx += 1
    return points


def parse_faces(text: str) -> list[list[int]]:
    """OpenFOAM faces ファイルを解析して面のリストを返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_faces, data_start = _read_count(lines, start)

    faces: list[list[int]] = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        # "4(0 1 5 4)" 形式
        if "(" in line:
            inner = line[line.index("(") + 1 : line.index(")")]
            node_ids = [int(x) for x in inner.split()]
            faces.append(node_ids)
    return faces


def parse_label_list(text: str) -> np.ndarray:
    """OpenFOAM の owner/neighbour ファイルを解析して整数配列を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_items, data_start = _read_count(lines, start)

    labels = np.zeros(n_items, dtype=np.int64)
    idx = 0
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        if line:
            labels[idx] = int(line)
            idx += 1
    return labels


def parse_boundary(text: str) -> dict[str, dict]:
    """OpenFOAM boundary ファイルを解析してパッチ情報を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    _n_patches, data_start = _read_count(lines, start)

    patches: dict[str, dict] = {}
    i = data_start
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == ")":
            break
        # パッチ名
        if line and not line.startswith("//") and line != "{":
            patch_name = line
            i += 1
            # { を見つける
            while i < n and "{" not in lines[i]:
                i += 1
            i += 1
            # パッチの属性を読む
            patch_data: dict[str, str | int] = {}
            while i < n:
                pline = lines[i].strip().rstrip(";")
                if "}" in pline:
                    i += 1
                    break
                parts = pline.split()
                if len(parts) == 2:
                    key, val = parts
                    try:
                        patch_data[key] = int(val)
                    except ValueError:
                        patch_data[key] = val
                i += 1
            patches[patch_name] = patch_data
            continue
        i += 1
    return patches


# ---------------------------------------------------------------------------
# メッシュ計算ヘルパー
# ---------------------------------------------------------------------------


def _compute_face_geometry(
    points: np.ndarray,
    faces: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """面の面積、法線、中心を計算.

    三角形分割で面積と法線を計算する。
    """
    n_faces = len(faces)
    areas = np.zeros(n_faces)
    normals = np.zeros((n_faces, 3))
    centers = np.zeros((n_faces, 3))

    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        center = pts.mean(axis=0)
        centers[f_idx] = center

        # 三角形分割で面積と法線を計算
        area_vec = np.zeros(3)
        for j in range(1, len(face_nodes) - 1):
            v1 = pts[j] - pts[0]
            v2 = pts[j + 1] - pts[0]
            area_vec += 0.5 * np.cross(v1, v2)

        area = np.linalg.norm(area_vec)
        areas[f_idx] = area
        if area > 0:
            normals[f_idx] = area_vec / area

    return areas, normals, centers


def _compute_cell_geometry(
    points: np.ndarray,
    faces: list[list[int]],
    owner: np.ndarray,
    neighbour: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """セルの体積と中心を計算.

    面中心をもとにセル中心を近似し、発散定理で体積を計算する。
    """
    # セル中心の初期推定: 所属する面の中心の平均
    cell_centers = np.zeros((n_cells, 3))
    cell_face_count = np.zeros(n_cells)

    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        fc = pts.mean(axis=0)
        o = owner[f_idx]
        cell_centers[o] += fc
        cell_face_count[o] += 1
        if f_idx < len(neighbour):
            nb = neighbour[f_idx]
            cell_centers[nb] += fc
            cell_face_count[nb] += 1

    safe_count = np.maximum(cell_face_count, 1)
    cell_centers /= safe_count[:, np.newaxis]

    # セル体積: 発散定理 V = (1/3) * Σ (r_f · n_f * A_f)
    cell_volumes = np.zeros(n_cells)
    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        fc = pts.mean(axis=0)

        # 面面積ベクトル
        area_vec = np.zeros(3)
        for j in range(1, len(face_nodes) - 1):
            v1 = pts[j] - pts[0]
            v2 = pts[j + 1] - pts[0]
            area_vec += 0.5 * np.cross(v1, v2)

        vol_contrib = np.dot(fc, area_vec) / 3.0
        cell_volumes[owner[f_idx]] += vol_contrib
        if f_idx < len(neighbour):
            cell_volumes[neighbour[f_idx]] -= vol_contrib

    cell_volumes = np.abs(cell_volumes)
    return cell_volumes, cell_centers


# ---------------------------------------------------------------------------
# セル節点順序の復元（面リスト → 要素接続）
# ---------------------------------------------------------------------------


def _cell_face_lists(
    faces: list[list[int]], owner: np.ndarray, neighbour: np.ndarray, n_cells: int
) -> list[list[list[int]]]:
    """セルごとに「そのセルから見て外向き」に並べた面節点リストを集める.

    OpenFOAM の面は owner から見て外向きなので、neighbour 側では逆順にする。
    """
    cell_faces: list[list[list[int]]] = [[] for _ in range(n_cells)]
    n_internal = len(neighbour)
    for f_idx, face_nodes in enumerate(faces):
        cell_faces[int(owner[f_idx])].append(list(face_nodes))
        if f_idx < n_internal:
            cell_faces[int(neighbour[f_idx])].append(list(face_nodes)[::-1])
    return cell_faces


def _rotate_to_min(nodes: list[int], points: np.ndarray) -> list[int]:
    """節点列を (y, x, z) 辞書順で最小の節点から始まるように巡回させる."""
    keys = [(float(points[n][1]), float(points[n][0]), float(points[n][2])) for n in nodes]
    start = min(range(len(nodes)), key=lambda i: keys[i])
    return nodes[start:] + nodes[:start]


def _order_cell_nodes(cell_faces: list[list[int]], points: np.ndarray) -> tuple[list[int], int]:
    """1 セルの外向き面リストから、節点順序付きの接続と VTK セル種別を返す.

    - 六面体（4 節点面 × 6）と楔（三角 2 + 四角 3）: 最も下（外向き法線 z が最小）
      の面を底面とし、内向き（右手系で上向き）に並べ替える。上面の節点は、底面の
      各節点から側面の辺をたどって決める（Abaqus C3D8 / C3D6 の順序）。
    - 四面体（三角 × 4）と角錐（四角 1 + 三角 4）: 底面 + 残りの 1 節点（頂点）。
    - それ以外: 節点集合をソートしたものを返し、種別は多面体。
    """
    from xkep_cae_fluid.core.data import (
        CELL_TYPE_HEX,
        CELL_TYPE_POLYHEDRON,
        CELL_TYPE_PYRAMID,
        CELL_TYPE_TET,
        CELL_TYPE_WEDGE,
    )

    sizes = sorted(len(f) for f in cell_faces)
    all_nodes = sorted({n for f in cell_faces for n in f})

    def outward_z(face: list[int]) -> float:
        pts = points[face]
        area_vec = np.zeros(3)
        for j in range(1, len(face) - 1):
            area_vec += 0.5 * np.cross(pts[j] - pts[0], pts[j + 1] - pts[0])
        return float(area_vec[2])

    def top_via_edges(base: list[int]) -> list[int] | None:
        base_set = set(base)
        top: list[int] = []
        for n0 in base:
            found: int | None = None
            for f in cell_faces:
                if n0 not in f or set(f) == base_set:
                    continue
                i = f.index(n0)
                for cand in (f[(i + 1) % len(f)], f[i - 1]):
                    if cand not in base_set:
                        found = cand
                        break
                if found is not None:
                    break
            if found is None:
                return None
            top.append(found)
        return top

    kind: int | None = None
    if sizes == [4, 4, 4, 4, 4, 4] and len(all_nodes) == 8:
        kind = CELL_TYPE_HEX
    elif sizes == [3, 3, 4, 4, 4] and len(all_nodes) == 6:
        kind = CELL_TYPE_WEDGE
    elif sizes == [3, 3, 3, 3] and len(all_nodes) == 4:
        kind = CELL_TYPE_TET
    elif sizes == [3, 3, 3, 3, 4] and len(all_nodes) == 5:
        kind = CELL_TYPE_PYRAMID

    if kind in (CELL_TYPE_HEX, CELL_TYPE_WEDGE):
        n_base = 4 if kind == CELL_TYPE_HEX else 3
        candidates = [f for f in cell_faces if len(f) == n_base]
        base_out = min(candidates, key=outward_z)
        base = _rotate_to_min(base_out[::-1], points)  # 内向き（右手系で上向き）
        top = top_via_edges(base)
        if top is not None and len(set(top)) == n_base:
            return base + top, kind
    elif kind in (CELL_TYPE_TET, CELL_TYPE_PYRAMID):
        n_base = 3 if kind == CELL_TYPE_TET else 4
        candidates = [f for f in cell_faces if len(f) == n_base]
        base_out = min(candidates, key=outward_z)
        base = _rotate_to_min(base_out[::-1], points)
        apex = [n for n in all_nodes if n not in base]
        if len(apex) == 1:
            return base + apex, kind

    return all_nodes, CELL_TYPE_POLYHEDRON


def build_ordered_connectivity(
    points: np.ndarray,
    faces: list[list[int]],
    owner: np.ndarray,
    neighbour: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """面リスト（owner/neighbour）から節点順序付き connectivity と cell_types を作る.

    Returns
    -------
    connectivity : np.ndarray
        (n_cells, max_nodes) int64。短いセルは -1 詰め
    cell_types : np.ndarray
        (n_cells,) VTK セル種別 ID
    """
    cell_faces = _cell_face_lists(faces, owner, neighbour, n_cells)
    ordered: list[list[int]] = []
    types = np.zeros(n_cells, dtype=np.int64)
    for c in range(n_cells):
        nodes, kind = _order_cell_nodes(cell_faces[c], points)
        ordered.append(nodes)
        types[c] = kind
    max_nodes = max((len(n) for n in ordered), default=0)
    connectivity = np.full((n_cells, max_nodes), -1, dtype=np.int64)
    for c, nodes in enumerate(ordered):
        connectivity[c, : len(nodes)] = nodes
    return connectivity, types


# ---------------------------------------------------------------------------
# PolyMeshReaderProcess
# ---------------------------------------------------------------------------


class PolyMeshReaderProcess(PreProcess["PolyMeshInput", "PolyMeshResult"]):
    """OpenFOAM polyMesh 読み込み Process.

    constant/polyMesh/ ディレクトリから非構造化メッシュを読み込み、
    MeshData として返す。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="PolyMeshReader",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/polymesh-reader.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: PolyMeshInput) -> PolyMeshResult:
        """polyMesh ディレクトリからメッシュを読み込む（ASCII/バイナリ自動判定）."""
        mesh_dir = Path(input_data.mesh_dir)

        # ファイル読み込み（バイナリ/ASCII 自動判定）
        points_raw = (mesh_dir / "points").read_bytes()
        faces_raw = (mesh_dir / "faces").read_bytes()
        owner_raw = (mesh_dir / "owner").read_bytes()
        neighbour_raw = (mesh_dir / "neighbour").read_bytes()

        if _is_binary_format(points_raw):
            points = parse_points_binary(points_raw)
        else:
            points = parse_points(points_raw.decode())

        if _is_binary_format(faces_raw):
            faces_list = parse_faces_binary(faces_raw)
        else:
            faces_list = parse_faces(faces_raw.decode())

        if _is_binary_format(owner_raw):
            owner = parse_label_list_binary(owner_raw)
        else:
            owner = parse_label_list(owner_raw.decode())

        if _is_binary_format(neighbour_raw):
            neighbour = parse_label_list_binary(neighbour_raw)
        else:
            neighbour = parse_label_list(neighbour_raw.decode())

        # boundary は常に ASCII（OpenFOAM 仕様）
        boundary_patches = parse_boundary((mesh_dir / "boundary").read_text())

        n_cells = int(owner.max()) + 1
        n_internal_faces = len(neighbour)

        # 面の幾何情報
        face_areas, face_normals, face_centers = _compute_face_geometry(points, faces_list)

        # セルの幾何情報
        cell_volumes, cell_centers = _compute_cell_geometry(
            points, faces_list, owner, neighbour, n_cells
        )

        # connectivity: 面リストから節点順序付きの要素接続を復元（六面体・楔・
        # 四面体・角錐は Abaqus / VTK の順序、それ以外は節点集合を -1 詰め）
        connectivity, cell_types = build_ordered_connectivity(
            points, faces_list, owner, neighbour, n_cells
        )

        # 境界パッチ → 境界面インデックス
        patches: dict[str, np.ndarray] = {}
        for name, info in boundary_patches.items():
            start = int(info.get("startFace", n_internal_faces))
            count = int(info.get("nFaces", 0))
            patches[name] = np.arange(start, start + count, dtype=np.int64)

        mesh = MeshData(
            node_coords=points,
            connectivity=connectivity,
            cell_volumes=cell_volumes,
            face_areas=face_areas,
            face_normals=face_normals,
            face_centers=face_centers,
            cell_centers=cell_centers,
            cell_types=cell_types,
            face_owner=owner,
            face_neighbour=neighbour,
            boundary_patches=patches,
        )

        return PolyMeshResult(mesh=mesh, boundary_patches=boundary_patches)
