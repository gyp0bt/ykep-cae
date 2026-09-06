"""``CaseDefinition`` の節点・要素表から面ベースの ``MeshData`` を組む（非構造格子経路）.

:class:`StructuredGridRecoveryProcess` が「軸平行の完全な箱格子」だけを (i, j, k) に
復元するのに対し、:class:`InpMeshProcess` は任意の六面体（C3D8 系）/ 楔（C3D6 系）/ 四面体
（C3D4 系）/ 角錐（C3D5）、2D なら四辺形（CPS4 系）/ 三角形（CPS3 系、厚さ ``depth_2d`` で押し出して
六面体 / 楔）のメッシュを owner/neighbour の面リストに変換する。要素種別の混在も可。
2 次要素（C3D10 / C3D15 / C3D20、CPS6 / CPS8 系）は頂点節点だけを使う（中間節点は面の照合にも
幾何にも使わない。1 次の面ベース FVM なので中間節点の情報は捨てる）。
歪んだ要素・穴あき・非直交メッシュも受け付け、面ベース FVM 層
（:mod:`xkep_cae_fluid.fvm`）の方程式ファミリーがそのまま解ける。

- ``*ELSET`` → セル index 集合（``cell_sets``）、``*NSET`` → 節点 index 集合
- ``*SURFACE`` → 境界面インデックス（``mesh.boundary_patches`` と ``surface_faces``）。
  内部面を含む面はパッチにならない（``surface_faces`` には内部面 index のまま入る）。
  ``baffle_surfaces`` に挙げた面の内部面は **2 枚の境界面（両側）に分割**して同名のパッチにする
  （厚さゼロのバッフル・薄板・仕切り。両側とも同じ境界条件を受ける）
- 予約面名 ``XM/XP/YM/YP/ZM/ZP`` は境界面を外向き法線の主軸で分類して自動生成する
  （``*SURFACE`` に同名があればそちらが優先。バッフル面は分類しない）
- ``*BOUNDARY, TYPE=PERIODIC``（``CaseDefinition.periodic``）の 2 面は面中心を並進で照合し、
  **master 面を内部面に昇格**（neighbour = slave 側のセル、``MeshData.face_offset = −t``）して
  slave 面を消す。周期面は境界パッチにならず、fvm 層は通常の内部面として扱う（並進周期のみ）
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import (
    CELL_TYPE_HEX,
    CELL_TYPE_PYRAMID,
    CELL_TYPE_TET,
    CELL_TYPE_WEDGE,
    MeshData,
)
from xkep_cae_fluid.core.mesh_reader import compute_cell_geometry, compute_face_geometry
from xkep_cae_fluid.inp.case import CaseDefinition, PeriodicDefinition, SurfaceDefinition
from xkep_cae_fluid.inp.grid import _HEX_FACES, FACE_NAMES, UnsupportedMeshError

# Abaqus の面番号 S1.. → 局所節点 index（四面体 C3D4、角錐 C3D5、楔 C3D6。六面体は grid._HEX_FACES）
_TET_FACES: dict[str, tuple[int, ...]] = {
    "S1": (0, 1, 2),
    "S2": (0, 3, 1),
    "S3": (1, 3, 2),
    "S4": (2, 3, 0),
}
_PYRAMID_FACES: dict[str, tuple[int, ...]] = {
    "S1": (0, 1, 2, 3),
    "S2": (0, 1, 4),
    "S3": (1, 2, 4),
    "S4": (2, 3, 4),
    "S5": (3, 0, 4),
}
_WEDGE_FACES: dict[str, tuple[int, ...]] = {
    "S1": (0, 1, 2),
    "S2": (3, 5, 4),
    "S3": (0, 3, 4, 1),
    "S4": (1, 4, 5, 2),
    "S5": (2, 5, 3, 0),
}
_FACES_3D: dict[int, dict[str, tuple[int, ...]]] = {
    4: _TET_FACES,
    5: _PYRAMID_FACES,
    6: _WEDGE_FACES,
    8: _HEX_FACES,
}
_CELL_TYPE_3D: dict[int, int] = {
    4: CELL_TYPE_TET,
    5: CELL_TYPE_PYRAMID,
    6: CELL_TYPE_WEDGE,
    8: CELL_TYPE_HEX,
}
# 要素の節点数 → 頂点数（2 次要素は頂点だけ使う。Abaqus は頂点が先頭に並ぶ）
_CORNERS_3D: dict[int, int] = {4: 4, 10: 4, 5: 5, 6: 6, 15: 6, 8: 8, 20: 8}
_CORNERS_2D: dict[int, int] = {3: 3, 6: 3, 4: 4, 8: 4}
# 2D 要素の辺 S1.. → 押し出した楔 / 六面体の側面ラベル
_EDGE_TO_SIDE_2D: dict[int, dict[str, str]] = {
    3: {"S1": "S3", "S2": "S4", "S3": "S5"},
    4: {"S1": "S3", "S2": "S4", "S3": "S5", "S4": "S6"},
}


@dataclass(frozen=True)
class InpMeshInput:
    """:class:`InpMeshProcess` の入力.

    Parameters
    ----------
    case : CaseDefinition
    depth_2d : float
        2D 要素（4 節点四辺形）を押し出す z 方向の厚さ [m]
    reserved_patches : bool
        予約面名 XM..ZP を外向き法線から自動生成する
    baffle_surfaces : tuple[str, ...]
        内部面を 2 枚の境界面に分割してパッチにする ``*SURFACE`` 名（厚さゼロのバッフル）。
        境界面だけの面を挙げても何もしない。``.inp`` ランナーは境界条件の target になった
        ``*SURFACE`` を自動でここに渡す
    """

    case: CaseDefinition
    depth_2d: float = 1.0
    reserved_patches: bool = True
    baffle_surfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class InpMeshResult:
    """面ベース ``MeshData`` と .inp エンティティの対応.

    Parameters
    ----------
    mesh : MeshData
        面情報・境界パッチ付きメッシュ（``dimensions=None``）。``connectivity`` は要素の最大節点数幅で
        -1 詰め、``cell_types`` は VTK 種別（四面体 10 / 楔 13 / 六面体 12）
    element_ids : np.ndarray
        (n_cells,) セル順の要素 ID
    node_ids : np.ndarray
        (n_nodes,) 節点順の節点 ID（2D 押し出し時は上面の複製節点にも元の ID）
    cell_sets : Mapping[str, np.ndarray]
        elset 名 → セル index
    node_sets : Mapping[str, np.ndarray]
        nset 名 → 節点 index（2D 押し出しでは底面・上面の両方）
    surface_faces : Mapping[str, np.ndarray]
        ``*SURFACE`` 名 → 面 index（バッフルは両側の境界面、分割しなかった内部面はその index）
    ndim : int
        元の要素次元（2 または 3）
    baffle_surfaces : tuple[str, ...]
        実際に内部面を分割した ``*SURFACE`` 名
    baffle_faces : np.ndarray
        バッフルの境界面 index（両側、(n_baffle_faces,)）。無ければ空
    periodic_faces : np.ndarray
        周期対を併合した内部面 index（``*BOUNDARY, TYPE=PERIODIC``）。無ければ空
    periodic_surfaces : tuple[str, ...]
        周期境界に使った面名（``*SURFACE`` 名または予約面名。境界条件は置けない）
    """

    mesh: MeshData
    element_ids: np.ndarray
    node_ids: np.ndarray
    cell_sets: Mapping[str, np.ndarray]
    node_sets: Mapping[str, np.ndarray]
    surface_faces: Mapping[str, np.ndarray]
    ndim: int
    baffle_surfaces: tuple[str, ...] = ()
    baffle_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    periodic_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    periodic_surfaces: tuple[str, ...] = ()

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    def cell_index_of(self, element_ids: np.ndarray) -> np.ndarray:
        """要素 ID 配列 → セル index 配列."""
        lookup = {int(e): i for i, e in enumerate(self.element_ids.tolist())}
        try:
            return np.array(
                [lookup[int(e)] for e in np.asarray(element_ids).tolist()], dtype=np.int64
            )
        except KeyError as exc:
            raise KeyError(f"要素 {exc.args[0]} はメッシュに存在しません") from exc

    def mask_for_elements(self, element_ids: np.ndarray) -> np.ndarray:
        """要素 ID 配列 → (n_cells,) bool."""
        mask = np.zeros(self.n_cells, dtype=bool)
        mask[self.cell_index_of(element_ids)] = True
        return mask

    def node_values_to_cells(self, node_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        """節点値（指定節点のみ）→ 要素節点平均のセル値（未指定セルは NaN）."""
        lookup: dict[int, list[int]] = {}
        for idx, nid in enumerate(self.node_ids.tolist()):
            lookup.setdefault(int(nid), []).append(idx)
        node_val = np.full(self.mesh.n_nodes, np.nan)
        for n, v in zip(np.asarray(node_ids).tolist(), np.asarray(values).tolist(), strict=True):
            for idx in lookup.get(int(n), []):
                node_val[idx] = v
        conn = self.mesh.connectivity
        vals = np.where(conn >= 0, node_val[np.maximum(conn, 0)], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(vals, axis=1)


# ---------------------------------------------------------------------------
# 構築
# ---------------------------------------------------------------------------


def _element_connectivity(
    case: CaseDefinition, depth_2d: float
) -> tuple[np.ndarray, list[list[int]], list[list[int]], np.ndarray, np.ndarray, int, list[int]]:
    """節点座標と要素接続を返す.

    Returns
    -------
    (coords, cells_raw, cells_out, elem_ids, node_ids, ndim, n2d)
        ``cells_raw`` は Abaqus の節点順そのまま（面ラベル S1.. の基準）、``cells_out`` は右手系
        （底面 → 上面）に正規化した出力用の順序。2D 要素は押し出し済み（底面 + 上面）。
        ``n2d`` は 2D 要素の元の節点数（3 / 4、3D なら 0）
    """
    if not case.elements:
        raise UnsupportedMeshError("*ELEMENT がありません")
    dims = {"3D" in b.element_type.upper() for b in case.elements}
    if len(dims) != 1:
        raise UnsupportedMeshError("2D 要素と 3D 要素が混在しています")
    is3d = dims.pop()
    ndim = 3 if is3d else 2
    corners = _CORNERS_3D if is3d else _CORNERS_2D
    for b in case.elements:
        if b.nodes_per_element not in corners:
            raise UnsupportedMeshError(
                f"要素 {b.element_type}（{b.nodes_per_element} 節点）は未対応"
                f"（3D は四面体 4/10 / 角錐 5 / 楔 6/15 / 六面体 8/20 節点、"
                f"2D は三角形 3/6 / 四辺形 4/8 節点）"
            )

    node_index = {int(n): i for i, n in enumerate(case.nodes.ids.tolist())}
    coords = np.asarray(case.nodes.coords, dtype=np.float64)
    elem_ids = np.concatenate([b.ids for b in case.elements]).astype(np.int64)
    node_ids = np.asarray(case.nodes.ids, dtype=np.int64)
    cells: list[list[int]] = []
    for b in case.elements:
        n_corner = corners[b.nodes_per_element]
        for row in b.connectivity.tolist():
            try:
                cells.append([node_index[int(n)] for n in row[:n_corner]])
            except KeyError as exc:
                raise UnsupportedMeshError(
                    f"要素が未定義の節点 {exc.args[0]} を参照しています"
                ) from exc

    cells_raw: list[list[int]] = []
    cells_out: list[list[int]] = []
    n2d: list[int] = []
    if ndim == 2:
        if depth_2d <= 0:
            raise UnsupportedMeshError("2D 要素には depth_2d > 0 が必要です")
        n_nodes = coords.shape[0]
        top = coords.copy()
        top[:, 2] = coords[:, 2] + depth_2d
        coords = np.vstack([coords, top])
        node_ids = np.concatenate([node_ids, node_ids])
        for poly in cells:
            p = coords[poly]
            area2 = 0.0
            for a, b_ in zip(range(len(poly)), [*range(1, len(poly)), 0], strict=True):
                area2 += p[a, 0] * p[b_, 1] - p[b_, 0] * p[a, 1]
            bottom = poly
            upper = [n + n_nodes for n in poly]
            # 面ラベル（辺 S1.. → 側面）は元の並びで定義するので、時計回りの多角形は
            # 上面を底面にして右手系にする（側面の節点集合は変わらない）
            cells_raw.append(bottom + upper)
            cells_out.append(bottom + upper if area2 >= 0 else upper + bottom)
            n2d.append(len(poly))
    else:
        for nodes in cells:
            p = coords[nodes]
            k = len(nodes)
            if k == 4:
                triple = float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))
                out = nodes if triple >= 0 else [nodes[0], nodes[2], nodes[1], nodes[3]]
            elif k == 5:
                triple = float(np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[4] - p[0]))
                out = nodes if triple >= 0 else [nodes[0], nodes[3], nodes[2], nodes[1], nodes[4]]
            elif k == 6:
                triple = float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))
                out = nodes if triple >= 0 else nodes[3:] + nodes[:3]
            else:
                triple = float(np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[4] - p[0]))
                out = nodes if triple >= 0 else nodes[4:] + nodes[:4]
            cells_raw.append(list(nodes))
            cells_out.append(list(out))
            n2d.append(0)
    return coords, cells_raw, cells_out, elem_ids, node_ids, ndim, n2d


def build_inp_mesh(
    case: CaseDefinition,
    depth_2d: float = 1.0,
    reserved_patches: bool = True,
    baffle_surfaces: tuple[str, ...] = (),
) -> InpMeshResult:
    """``CaseDefinition`` から面ベースの :class:`MeshData` を組み立てる.

    ``baffle_surfaces`` の ``*SURFACE`` が内部面を含めば、その面を owner 側・neighbour 側の
    2 枚の境界面に分割し（面の節点列は owner から外向き / その逆順）、同名のパッチにする。
    """
    coords, cells_raw, cells_out, elem_ids, node_ids, ndim, n2d = _element_connectivity(
        case, depth_2d
    )
    n_cells = len(cells_raw)
    kinds = [len(c) for c in cells_raw]

    # 面の照合: 節点集合をキーに owner/neighbour を決める
    seen: dict[tuple[int, ...], tuple[int, str, list[int]]] = {}
    internal: list[tuple[list[int], int, int]] = []  # (nodes(outward from owner), owner, neighbour)
    internal_key: list[tuple[int, str, int, str]] = []
    boundary: list[tuple[list[int], int, str]] = []
    n_faces_expected = 0
    for c in range(n_cells):
        cell = cells_raw[c]
        center_est = coords[cell].mean(axis=0)
        table = _FACES_3D[kinds[c]]
        n_faces_expected += len(table)
        for label, local in table.items():
            nodes = [int(cell[i]) for i in local]
            # owner から見て外向きに並べ替える
            pts = coords[nodes]
            area_vec = np.zeros(3)
            for j in range(1, len(nodes) - 1):
                area_vec += 0.5 * np.cross(pts[j] - pts[0], pts[j + 1] - pts[0])
            if np.dot(area_vec, pts.mean(axis=0) - center_est) < 0:
                nodes = nodes[::-1]
            key = tuple(sorted(nodes))
            if key in seen:
                c0, label0, nodes0 = seen.pop(key)
                internal.append((nodes0, c0, c))
                internal_key.append((c0, label0, c, label))
            else:
                seen[key] = (c, label, nodes)
    # 3 つ以上のセルで共有された面は seen から pop された後に再登録される → 検出
    for _key, (c, label, nodes) in seen.items():
        boundary.append((nodes, c, label))
    n_int = len(internal)

    def _index_faces() -> tuple[int, dict[tuple[int, str], int]]:
        """(セル, 面ラベル) → 面 index。内部面が先、境界面が後."""
        n_i = len(internal)
        ef: dict[tuple[int, str], int] = {}
        for fi, (c0, l0, c1, l1) in enumerate(internal_key):
            ef[(c0, l0)] = fi
            ef[(c1, l1)] = fi
        for bi, (_nodes, c, label) in enumerate(boundary):
            ef[(c, label)] = n_i + bi
        return n_i, ef

    n_int, elem_face = _index_faces()
    if len(elem_face) != n_faces_expected:
        raise UnsupportedMeshError("同じ面を 3 つ以上の要素が共有しています（メッシュが不正）")

    # バッフル: 指定 *SURFACE の内部面を 2 枚の境界面に分割する
    lookup = {int(e): i for i, e in enumerate(elem_ids.tolist())}
    split: set[int] = set()
    used_baffles: list[str] = []
    for name in baffle_surfaces:
        key = name.strip().upper()
        if key not in case.surfaces:
            raise UnsupportedMeshError(f"baffle_surfaces の {name!r} は *SURFACE にありません")
        idx = _surface_to_faces(case.surfaces[key], case, lookup, elem_face, kinds, n2d, ndim)
        inner = {int(i) for i in idx.tolist() if i < n_int}
        if inner:
            split |= inner
            used_baffles.append(key)
    n_boundary_orig = len(boundary)
    baffle_partner: dict[int, int] = {}
    if split:
        kept = [i for i in range(n_int) if i not in split]
        pairs: list[tuple[list[int], int, str]] = []
        for i in sorted(split):
            nodes0, c0, c1 = internal[i]
            _c0, l0, _c1, l1 = internal_key[i]
            pairs.append((list(nodes0), c0, l0))
            pairs.append((list(nodes0)[::-1], c1, l1))
        internal = [internal[i] for i in kept]
        internal_key = [internal_key[i] for i in kept]
        boundary = boundary + pairs
        n_int, elem_face = _index_faces()
        for k in range(len(pairs) // 2):
            a = n_int + n_boundary_orig + 2 * k
            baffle_partner[a] = a + 1
            baffle_partner[a + 1] = a
    baffle_faces_old = np.array(sorted(baffle_partner), dtype=np.int64)

    faces_list = [f[0] for f in internal] + [b[0] for b in boundary]
    owner = np.array([f[1] for f in internal] + [b[1] for b in boundary], dtype=np.int64)
    neighbour = np.array([f[2] for f in internal], dtype=np.int64)

    # 幾何は周期面を併合する前（両面とも境界面のまま）に計算する。併合後の内部面は master 側の
    # 幾何を持ち、neighbour（slave 側のセル）は face_offset で並進して戻した位置に置く
    face_areas, face_normals, face_centers = compute_face_geometry(coords, faces_list)
    cell_volumes, cell_centers = compute_cell_geometry(
        coords, faces_list, owner, neighbour, n_cells
    )
    if np.any(face_areas <= 0):
        raise UnsupportedMeshError("面積ゼロの面があります（縮退要素）")
    if np.any(cell_volumes <= 0):
        raise UnsupportedMeshError("体積ゼロの要素があります")

    # 集合
    cell_sets: dict[str, np.ndarray] = {}
    for name, sd in case.elsets.items():
        cell_sets[name] = np.array([lookup[int(e)] for e in sd.ids.tolist()], dtype=np.int64)
    node_lookup: dict[int, list[int]] = {}
    for idx, nid in enumerate(node_ids.tolist()):
        node_lookup.setdefault(int(nid), []).append(idx)
    node_sets: dict[str, np.ndarray] = {}
    for name, sd in case.nsets.items():
        node_sets[name] = np.array(
            sorted(i for n in sd.ids.tolist() for i in node_lookup.get(int(n), [])), dtype=np.int64
        )

    # *SURFACE → 面（併合前の index。バッフルの面はどちら側を指していても両側を含める）
    surface_raw: dict[str, np.ndarray] = {}
    for name, surf in case.surfaces.items():
        idx = _surface_to_faces(surf, case, lookup, elem_face, kinds, n2d, ndim)
        if baffle_partner:
            both = set(idx.tolist()) | {
                baffle_partner[i] for i in idx.tolist() if i in baffle_partner
            }
            idx = np.array(sorted(both), dtype=np.int64)
        surface_raw[name] = idx
    is_baffle_old = np.zeros(len(faces_list), dtype=bool)
    is_baffle_old[baffle_faces_old] = True
    reserved_raw = _reserved_face_names(face_normals, n_int, is_baffle_old)

    # 周期面の併合（*BOUNDARY, TYPE=PERIODIC）: master 面を内部面に昇格し slave 面を消す
    face_map = np.arange(len(faces_list), dtype=np.int64)
    face_offset = np.zeros((n_int, 3))
    periodic_faces = np.zeros(0, dtype=np.int64)
    if case.periodic:

        def _resolve(name: str) -> np.ndarray:
            key = name.strip().upper()
            if key in surface_raw:
                return surface_raw[key]
            if key in reserved_raw:
                return reserved_raw[key]
            raise UnsupportedMeshError(
                f"周期境界の面 {name!r} は *SURFACE でも予約面名（外皮に存在する面）でもありません"
            )

        merged = _merge_periodic(
            case.periodic,
            coords,
            _resolve,
            n_int,
            faces_list,
            owner,
            neighbour,
            face_areas,
            face_normals,
            face_centers,
        )
        (
            faces_list,
            owner,
            neighbour,
            face_areas,
            face_normals,
            face_centers,
            face_offset,
            face_map,
            periodic_faces,
        ) = merged
        n_int = int(neighbour.shape[0])

    def _remap(idx: np.ndarray) -> np.ndarray:
        return np.unique(face_map[np.asarray(idx, dtype=np.int64)]).astype(np.int64)

    surface_faces: dict[str, np.ndarray] = {}
    patches: dict[str, np.ndarray] = {}
    for name, idx in surface_raw.items():
        new_idx = _remap(idx)
        surface_faces[name] = new_idx
        if new_idx.size and np.all(new_idx >= n_int):
            patches[name] = new_idx
    baffle_faces = _remap(baffle_faces_old) if baffle_faces_old.size else baffle_faces_old
    periodic_surfaces = tuple(sorted({n for p in case.periodic for n in (p.master, p.slave)}))

    # 予約面名（外向き法線の主軸。バッフル面は外皮ではないので分類しない。周期面は内部面）。
    # 同じ面が ``*SURFACE`` のパッチと予約名の両方に入りうる（両方に境界条件を書くと後勝ち）
    if reserved_patches:
        is_baffle = np.zeros(len(faces_list), dtype=bool)
        is_baffle[baffle_faces] = True
        for name, idx in _reserved_face_names(face_normals, n_int, is_baffle).items():
            if name not in patches:
                patches[name] = idx

    width = max(kinds)
    conn = np.full((n_cells, width), -1, dtype=np.int64)
    for c, nodes in enumerate(cells_out):
        conn[c, : len(nodes)] = nodes
    cell_types = np.array([_CELL_TYPE_3D[k] for k in kinds], dtype=np.int64)

    mesh = MeshData(
        node_coords=coords,
        connectivity=conn,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        face_normals=face_normals,
        face_centers=face_centers,
        cell_centers=cell_centers,
        cell_types=cell_types,
        face_owner=owner,
        face_neighbour=neighbour,
        boundary_patches=patches,
        face_offset=face_offset if periodic_faces.size else None,
    )
    return InpMeshResult(
        mesh=mesh,
        element_ids=elem_ids,
        node_ids=node_ids,
        cell_sets=cell_sets,
        node_sets=node_sets,
        surface_faces=surface_faces,
        ndim=ndim,
        baffle_surfaces=tuple(used_baffles),
        baffle_faces=baffle_faces,
        periodic_faces=periodic_faces,
        periodic_surfaces=periodic_surfaces,
    )


def _reserved_face_names(
    face_normals: np.ndarray, n_int: int, exclude: np.ndarray
) -> dict[str, np.ndarray]:
    """境界面を外向き法線の主軸で XM..ZP に分類する（``exclude`` の面は分類しない）."""
    bn = face_normals[n_int:]
    if bn.shape[0] == 0:
        return {}
    axis = np.argmax(np.abs(bn), axis=1)
    keep = ~np.asarray(exclude, dtype=bool)[n_int:]
    out: dict[str, np.ndarray] = {}
    for name in FACE_NAMES:
        ax = "XYZ".index(name[0])
        sign = 1.0 if name[1] == "P" else -1.0
        sel = keep & (axis == ax) & (np.sign(bn[np.arange(len(bn)), axis]) == sign)
        if np.any(sel):
            out[name] = n_int + np.nonzero(sel)[0].astype(np.int64)
    return out


def _merge_periodic(
    periodic: tuple[PeriodicDefinition, ...],
    coords: np.ndarray,
    resolve: Callable[[str], np.ndarray],
    n_int: int,
    faces_list: list[list[int]],
    owner: np.ndarray,
    neighbour: np.ndarray,
    face_areas: np.ndarray,
    face_normals: np.ndarray,
    face_centers: np.ndarray,
) -> tuple[
    list[list[int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """周期対の境界面を照合し、master 面を内部面（neighbour = slave 側のセル）に併合する.

    slave 面の中心が master 面の中心を並進 t だけ動かした位置に一致することを要求する
    （許容は座標の対角長 × 1e-6）。併合した内部面は master の幾何を持ち、
    ``face_offset = −t`` で neighbour セル中心を owner 側に戻す。slave 面は消え、
    面 index の対応は ``face_map``（旧 → 新。slave 面は併合先の内部面）で返す。

    Returns
    -------
    (faces_list, owner, neighbour, areas, normals, centers, face_offset, face_map, periodic_faces)
    """
    from scipy.spatial import cKDTree

    n_faces = len(faces_list)
    span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    tol = 1e-6 * max(span, 1e-300)
    used = np.zeros(n_faces, dtype=bool)
    masters: list[int] = []
    slaves: list[int] = []
    offsets: list[np.ndarray] = []
    for per in periodic:
        m = np.asarray(resolve(per.master), dtype=np.int64)
        s = np.asarray(resolve(per.slave), dtype=np.int64)
        label = f"{per.master} ↔ {per.slave}"
        if m.size == 0 or s.size == 0:
            raise UnsupportedMeshError(f"周期境界 {label} の面が空です")
        if np.any(m < n_int) or np.any(s < n_int):
            raise UnsupportedMeshError(f"周期境界 {label} の面に内部面が含まれています")
        if np.any(used[m]) or np.any(used[s]):
            raise UnsupportedMeshError(f"周期境界 {label} の面が他の周期境界と重なっています")
        if m.size != s.size:
            raise UnsupportedMeshError(
                f"周期境界 {label} の面数が一致しません（{m.size} 対 {s.size}）"
            )
        if per.translation is None:
            t = face_centers[s].mean(axis=0) - face_centers[m].mean(axis=0)
        else:
            t = np.asarray(per.translation, dtype=np.float64)
        target = face_centers[m] + t[None, :]
        dist, j = cKDTree(face_centers[s]).query(target)
        if float(np.max(dist)) > tol or np.unique(j).size != m.size:
            raise UnsupportedMeshError(
                f"周期境界 {label} の面が並進 t={tuple(float(v) for v in t)} で一致しません"
                f"（最大ずれ {float(np.max(dist)):.3e} m、許容 {tol:.3e} m。"
                f"両面のメッシュ分割が同じか、並進ベクトルが正しいか確認）"
            )
        s_matched = s[j]
        dots = np.sum(face_normals[m] * face_normals[s_matched], axis=1)
        if np.any(dots > -1.0 + 1e-6):
            raise UnsupportedMeshError(
                f"周期境界 {label} の面法線が反平行ではありません（並進周期は平行な 2 面のみ）"
            )
        masters.extend(m.tolist())
        slaves.extend(s_matched.tolist())
        offsets.extend([-t] * m.size)
        used[m] = True
        used[s] = True

    m_arr = np.asarray(masters, dtype=np.int64)
    s_arr = np.asarray(slaves, dtype=np.int64)
    keep_b = [i for i in range(n_int, n_faces) if not used[i]]
    new_order = np.concatenate(
        [np.arange(n_int, dtype=np.int64), m_arr, np.asarray(keep_b, dtype=np.int64)]
    )
    face_map = np.full(n_faces, -1, dtype=np.int64)
    face_map[new_order] = np.arange(new_order.size, dtype=np.int64)
    face_map[s_arr] = face_map[m_arr]
    n_int_new = n_int + m_arr.size
    faces_new = [faces_list[int(i)] for i in new_order.tolist()]
    owner_new = owner[new_order]
    neighbour_new = np.concatenate([neighbour, owner[s_arr]]).astype(np.int64)
    face_offset = np.zeros((n_int_new, 3))
    if offsets:
        face_offset[n_int:n_int_new] = np.asarray(offsets, dtype=np.float64)
    periodic_faces = np.arange(n_int, n_int_new, dtype=np.int64)
    return (
        faces_new,
        owner_new,
        neighbour_new,
        face_areas[new_order],
        face_normals[new_order],
        face_centers[new_order],
        face_offset,
        face_map,
        periodic_faces,
    )


def _surface_to_faces(
    surface: SurfaceDefinition,
    case: CaseDefinition,
    lookup: dict[int, int],
    elem_face: dict[tuple[int, str], int],
    kinds: list[int],
    n2d: list[int],
    ndim: int,
) -> np.ndarray:
    out: list[int] = []
    for entry in surface.entries:
        for e in case.element_ids_of(entry.target).tolist():
            c = lookup[int(e)]
            label = entry.face
            if ndim == 2:
                # 多角形の辺 S1.. は押し出した楔 / 六面体の側面 S3.. に対応
                edges = _EDGE_TO_SIDE_2D[n2d[c]]
                if label not in edges:
                    raise UnsupportedMeshError(
                        f"面ラベル {label} は {n2d[c]} 節点の 2D 要素 {e} では使えません"
                    )
                label = edges[label]
            elif label not in _FACES_3D[kinds[c]]:
                raise UnsupportedMeshError(
                    f"面ラベル {label} は {kinds[c]} 節点の 3D 要素 {e} では使えません"
                )
            out.append(elem_face[(c, label)])
    return np.array(sorted(set(out)), dtype=np.int64)


class InpMeshProcess(PreProcess["InpMeshInput", "InpMeshResult"]):
    """``*NODE`` / ``*ELEMENT``（六面体 / 楔 / 四面体 / 角錐、2D 四辺形 / 三角形、2 次要素は頂点のみ）を面ベースの非構造 ``MeshData`` にする PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpMesh",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/unstructured-inp-mesh.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMeshInput) -> InpMeshResult:
        return build_inp_mesh(
            input_data.case,
            input_data.depth_2d,
            input_data.reserved_patches,
            tuple(input_data.baffle_surfaces),
        )
