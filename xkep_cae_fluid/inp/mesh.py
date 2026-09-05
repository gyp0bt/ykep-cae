"""``CaseDefinition`` の節点・要素表から面ベースの ``MeshData`` を組む（非構造格子経路）.

:class:`StructuredGridRecoveryProcess` が「軸平行の完全な箱格子」だけを (i, j, k) に
復元するのに対し、:class:`InpMeshProcess` は任意の六面体（C3D8 系）/ 四辺形（CPS4 系、
厚さ ``depth_2d`` で押し出し）メッシュを owner/neighbour の面リストに変換する。
歪んだ要素・穴あき・非直交メッシュも受け付け、面ベース FVM 層
（:mod:`xkep_cae_fluid.fvm`）の方程式ファミリーがそのまま解ける。

- ``*ELSET`` → セル index 集合（``cell_sets``）、``*NSET`` → 節点 index 集合
- ``*SURFACE`` → 境界面インデックス（``mesh.boundary_patches`` と ``surface_faces``）。
  内部面を含む面は :class:`UnsupportedMeshError`
- 予約面名 ``XM/XP/YM/YP/ZM/ZP`` は境界面を外向き法線の主軸で分類して自動生成する
  （``*SURFACE`` に同名があればそちらが優先）
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import CELL_TYPE_HEX, MeshData
from xkep_cae_fluid.core.mesh_reader import compute_cell_geometry, compute_face_geometry
from xkep_cae_fluid.inp.case import CaseDefinition, SurfaceDefinition
from xkep_cae_fluid.inp.grid import _HEX_FACES, _QUAD_FACES, FACE_NAMES, UnsupportedMeshError


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
    """

    case: CaseDefinition
    depth_2d: float = 1.0
    reserved_patches: bool = True


@dataclass(frozen=True)
class InpMeshResult:
    """面ベース ``MeshData`` と .inp エンティティの対応.

    Parameters
    ----------
    mesh : MeshData
        面情報・境界パッチ付きメッシュ（``dimensions=None``）
    element_ids : np.ndarray
        (n_cells,) セル順の要素 ID
    node_ids : np.ndarray
        (n_nodes,) 節点順の節点 ID（2D 押し出し時は上面の複製節点にも元の ID）
    cell_sets : Mapping[str, np.ndarray]
        elset 名 → セル index
    node_sets : Mapping[str, np.ndarray]
        nset 名 → 節点 index（2D 押し出しでは底面・上面の両方）
    surface_faces : Mapping[str, np.ndarray]
        ``*SURFACE`` 名 → 面 index
    ndim : int
        元の要素次元（2 または 3）
    """

    mesh: MeshData
    element_ids: np.ndarray
    node_ids: np.ndarray
    cell_sets: Mapping[str, np.ndarray]
    node_sets: Mapping[str, np.ndarray]
    surface_faces: Mapping[str, np.ndarray]
    ndim: int

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


def _hex_connectivity(
    case: CaseDefinition, depth_2d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """節点座標・六面体接続（右手系に正規化）・要素 ID・節点 ID・次元を返す."""
    if not case.elements:
        raise UnsupportedMeshError("*ELEMENT がありません")
    n_per = {b.nodes_per_element for b in case.elements}
    if len(n_per) != 1:
        raise UnsupportedMeshError(f"要素の節点数が混在しています: {sorted(n_per)}")
    nodes_per = n_per.pop()
    if nodes_per not in (4, 8):
        raise UnsupportedMeshError(
            f"{nodes_per} 節点要素は未対応（4 節点四辺形 / 8 節点六面体のみ）"
        )
    ndim = 3 if nodes_per == 8 else 2

    node_index = {int(n): i for i, n in enumerate(case.nodes.ids.tolist())}
    coords = np.asarray(case.nodes.coords, dtype=np.float64)
    conn_raw = np.concatenate([b.connectivity for b in case.elements]).astype(np.int64)
    elem_ids = np.concatenate([b.ids for b in case.elements]).astype(np.int64)
    try:
        conn = np.vectorize(node_index.__getitem__)(conn_raw)
    except KeyError as exc:
        raise UnsupportedMeshError(f"要素が未定義の節点 {exc.args[0]} を参照しています") from exc
    node_ids = np.asarray(case.nodes.ids, dtype=np.int64)

    if ndim == 2:
        if depth_2d <= 0:
            raise UnsupportedMeshError("2D 要素には depth_2d > 0 が必要です")
        n_nodes = coords.shape[0]
        top = coords.copy()
        top[:, 2] = coords[:, 2] + depth_2d
        coords = np.vstack([coords, top])
        node_ids = np.concatenate([node_ids, node_ids])
        # 四辺形の向き（xy 面の符号付き面積）で反時計回りに揃える
        p = coords[conn]
        area2 = np.zeros(conn.shape[0])
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            area2 += p[:, a, 0] * p[:, b, 1] - p[:, b, 0] * p[:, a, 1]
        cw = area2 < 0
        conn[cw] = conn[cw][:, ::-1]
        conn = np.hstack([conn, conn + n_nodes])
    else:
        # 右手系（底面 → 上面）に揃える: (n1−n0)×(n3−n0)·(n4−n0) が負なら底面と上面を入れ替える
        p = coords[conn]
        triple = np.einsum(
            "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]), p[:, 4] - p[:, 0]
        )
        flipped = triple < 0
        conn[flipped] = conn[flipped][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return coords, conn, elem_ids, node_ids, ndim


def build_inp_mesh(
    case: CaseDefinition, depth_2d: float = 1.0, reserved_patches: bool = True
) -> InpMeshResult:
    """``CaseDefinition`` から面ベースの :class:`MeshData` を組み立てる."""
    coords, conn, elem_ids, node_ids, ndim = _hex_connectivity(case, depth_2d)
    n_cells = conn.shape[0]
    centers_est = coords[conn].mean(axis=1)

    # 面の照合: 節点集合をキーに owner/neighbour を決める
    face_labels = list(_HEX_FACES)  # S1..S6
    seen: dict[tuple[int, ...], tuple[int, str, list[int]]] = {}
    internal: list[tuple[list[int], int, int]] = []  # (nodes(outward from owner), owner, neighbour)
    internal_key: list[tuple[int, str, int, str]] = []
    boundary: list[tuple[list[int], int, str]] = []
    for c in range(n_cells):
        for label in face_labels:
            local = _HEX_FACES[label]
            nodes = [int(conn[c, i]) for i in local]
            # owner から見て外向きに並べ替える
            pts = coords[nodes]
            area_vec = np.zeros(3)
            for j in range(1, 3):
                area_vec += 0.5 * np.cross(pts[j] - pts[0], pts[j + 1] - pts[0])
            if np.dot(area_vec, pts.mean(axis=0) - centers_est[c]) < 0:
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
    faces_list = [f[0] for f in internal] + [b[0] for b in boundary]
    owner = np.array([f[1] for f in internal] + [b[1] for b in boundary], dtype=np.int64)
    neighbour = np.array([f[2] for f in internal], dtype=np.int64)

    # (セル, 面ラベル) → 面 index
    elem_face: dict[tuple[int, str], int] = {}
    for fi, (c0, l0, c1, l1) in enumerate(internal_key):
        elem_face[(c0, l0)] = fi
        elem_face[(c1, l1)] = fi
    for bi, (_nodes, c, label) in enumerate(boundary):
        elem_face[(c, label)] = n_int + bi
    if len(elem_face) != 6 * n_cells:
        raise UnsupportedMeshError("同じ面を 3 つ以上の要素が共有しています（メッシュが不正）")

    face_areas, face_normals, face_centers = compute_face_geometry(coords, faces_list)
    cell_volumes, cell_centers = compute_cell_geometry(
        coords, faces_list, owner, neighbour, n_cells
    )
    if np.any(face_areas <= 0):
        raise UnsupportedMeshError("面積ゼロの面があります（縮退要素）")
    if np.any(cell_volumes <= 0):
        raise UnsupportedMeshError("体積ゼロの要素があります")

    # 集合
    lookup = {int(e): i for i, e in enumerate(elem_ids.tolist())}
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

    # *SURFACE → 面
    faces_table = _HEX_FACES if ndim == 3 else _QUAD_FACES
    surface_faces: dict[str, np.ndarray] = {}
    patches: dict[str, np.ndarray] = {}
    for name, surf in case.surfaces.items():
        idx = _surface_to_faces(surf, case, lookup, elem_face, faces_table, ndim)
        surface_faces[name] = idx
        if np.any(idx < n_int):
            raise UnsupportedMeshError(
                f"*SURFACE {name}: 内部面を含んでいます（境界条件のパッチは境界面のみ）"
            )
        patches[name] = idx

    # 予約面名（外向き法線の主軸）
    if reserved_patches:
        bn = face_normals[n_int:]
        axis = np.argmax(np.abs(bn), axis=1)
        for name in FACE_NAMES:
            if name in patches:
                continue
            ax = "XYZ".index(name[0])
            sign = 1.0 if name[1] == "P" else -1.0
            sel = (axis == ax) & (np.sign(bn[np.arange(len(bn)), axis]) == sign)
            if np.any(sel):
                patches[name] = n_int + np.nonzero(sel)[0].astype(np.int64)

    mesh = MeshData(
        node_coords=coords,
        connectivity=conn,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        face_normals=face_normals,
        face_centers=face_centers,
        cell_centers=cell_centers,
        cell_types=np.full(n_cells, CELL_TYPE_HEX, dtype=np.int64),
        face_owner=owner,
        face_neighbour=neighbour,
        boundary_patches=patches,
    )
    return InpMeshResult(
        mesh=mesh,
        element_ids=elem_ids,
        node_ids=node_ids,
        cell_sets=cell_sets,
        node_sets=node_sets,
        surface_faces=surface_faces,
        ndim=ndim,
    )


def _surface_to_faces(
    surface: SurfaceDefinition,
    case: CaseDefinition,
    lookup: dict[int, int],
    elem_face: dict[tuple[int, str], int],
    faces_table: dict[str, tuple[int, ...]],
    ndim: int,
) -> np.ndarray:
    out: list[int] = []
    for entry in surface.entries:
        if entry.face not in faces_table:
            raise UnsupportedMeshError(f"面ラベル {entry.face} は {ndim}D 要素では使えません")
        label = entry.face
        if ndim == 2:
            # 四辺形の辺 S1..S4 は押し出し六面体の側面 S3..S6 に対応
            label = {"S1": "S3", "S2": "S4", "S3": "S5", "S4": "S6"}[label]
        for e in case.element_ids_of(entry.target).tolist():
            out.append(elem_face[(lookup[int(e)], label)])
    return np.array(sorted(set(out)), dtype=np.int64)


class InpMeshProcess(PreProcess["InpMeshInput", "InpMeshResult"]):
    """``*NODE`` / ``*ELEMENT`` を面ベースの非構造 ``MeshData`` にする PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpMesh",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/unstructured-inp-mesh.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMeshInput) -> InpMeshResult:
        return build_inp_mesh(input_data.case, input_data.depth_2d, input_data.reserved_patches)
