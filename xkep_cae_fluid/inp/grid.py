"""節点・要素表から直交構造格子 (i, j, k) を復元する.

ykep の全ソルバーは構造直交格子を前提とするため、``*NODE`` / ``*ELEMENT`` で
与えられた六面体（または四辺形）メッシュが軸平行の完全な箱格子であることを検証し、
要素 ID → セル (i, j, k)、``*SURFACE`` → 領域 6 面（XM..ZP）の対応を作る。
非直交・非構造メッシュは :class:`UnsupportedMeshError` で拒否する。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.inp.case import CaseDefinition, SurfaceDefinition

FACE_NAMES: tuple[str, ...] = ("XM", "XP", "YM", "YP", "ZM", "ZP")

# Abaqus の面ラベル → 要素内ローカル節点（0 始まり）
_HEX_FACES: dict[str, tuple[int, ...]] = {
    "S1": (0, 1, 2, 3),
    "S2": (4, 7, 6, 5),
    "S3": (0, 4, 5, 1),
    "S4": (1, 5, 6, 2),
    "S5": (2, 6, 7, 3),
    "S6": (3, 7, 4, 0),
}
_QUAD_FACES: dict[str, tuple[int, ...]] = {
    "S1": (0, 1),
    "S2": (1, 2),
    "S3": (2, 3),
    "S4": (3, 0),
}


class UnsupportedMeshError(ValueError):
    """構造直交格子として解釈できないメッシュ."""


@dataclass(frozen=True)
class StructuredGridInput:
    """:class:`StructuredGridRecoveryProcess` の入力.

    Parameters
    ----------
    case : CaseDefinition
    rel_tol : float
        格子線の同一判定に使う相対許容（領域寸法基準）
    depth_2d : float
        2D 要素（4 節点四辺形）のときに補う z 方向の厚さ [m]
    """

    case: CaseDefinition
    rel_tol: float = 1.0e-8
    depth_2d: float = 1.0


@dataclass(frozen=True)
class StructuredGridMap:
    """復元した構造格子と .inp エンティティの対応.

    Parameters
    ----------
    x_lines, y_lines, z_lines : np.ndarray
        格子線座標（昇順、長さ n+1）
    element_ids : np.ndarray
        (n_elem,) 要素 ID
    element_ijk : np.ndarray
        (n_elem, 3) 各要素のセル添字
    node_ids : np.ndarray
        (n_nodes,) 節点 ID
    node_ijk : np.ndarray
        (n_nodes, 3) 各節点の格子線添字
    ndim : int
        2 または 3（元の要素次元）
    """

    x_lines: np.ndarray
    y_lines: np.ndarray
    z_lines: np.ndarray
    element_ids: np.ndarray
    element_ijk: np.ndarray
    node_ids: np.ndarray
    node_ijk: np.ndarray
    ndim: int

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return (len(self.x_lines) - 1, len(self.y_lines) - 1, len(self.z_lines) - 1)

    @property
    def origin(self) -> tuple[float, float, float]:
        return (float(self.x_lines[0]), float(self.y_lines[0]), float(self.z_lines[0]))

    @property
    def lengths(self) -> tuple[float, float, float]:
        return (
            float(self.x_lines[-1] - self.x_lines[0]),
            float(self.y_lines[-1] - self.y_lines[0]),
            float(self.z_lines[-1] - self.z_lines[0]),
        )

    @property
    def spacings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.diff(self.x_lines), np.diff(self.y_lines), np.diff(self.z_lines))

    @property
    def is_uniform(self) -> bool:
        return all(
            d.size == 0 or np.allclose(d, d[0], rtol=1.0e-6, atol=0.0) for d in self.spacings
        )

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.dimensions
        return nx * ny * nz

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """セル中心座標 (nx,), (ny,), (nz,)."""
        return (
            0.5 * (self.x_lines[:-1] + self.x_lines[1:]),
            0.5 * (self.y_lines[:-1] + self.y_lines[1:]),
            0.5 * (self.z_lines[:-1] + self.z_lines[1:]),
        )

    def _elem_lookup(self) -> dict[int, int]:
        return {int(e): idx for idx, e in enumerate(self.element_ids.tolist())}

    def mask_for_elements(self, element_ids: np.ndarray) -> np.ndarray:
        """要素 ID 配列 → (nx, ny, nz) の bool マスク."""
        lookup = self._elem_lookup()
        mask = np.zeros(self.dimensions, dtype=bool)
        for e in np.asarray(element_ids).tolist():
            try:
                i, j, k = self.element_ijk[lookup[int(e)]]
            except KeyError as exc:
                raise KeyError(f"要素 {e} は格子に存在しません") from exc
            mask[i, j, k] = True
        return mask

    def node_values_to_cells(
        self, node_ids: np.ndarray, values: np.ndarray, case: CaseDefinition
    ) -> np.ndarray:
        """節点値（NaN=未指定）を要素節点平均でセル値に変換する（未指定セルは NaN）."""
        node_lookup = {int(n): idx for idx, n in enumerate(case.nodes.ids.tolist())}
        node_val = np.full(case.nodes.n_nodes, np.nan)
        for n, v in zip(np.asarray(node_ids).tolist(), np.asarray(values).tolist(), strict=True):
            node_val[node_lookup[int(n)]] = v
        out = np.full(self.dimensions, np.nan)
        lookup = self._elem_lookup()
        for block in case.elements:
            idx = np.vectorize(node_lookup.__getitem__)(block.connectivity)
            vals = node_val[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(vals, axis=1)
            for e, m in zip(block.ids.tolist(), mean.tolist(), strict=True):
                if not np.isnan(m):
                    i, j, k = self.element_ijk[lookup[int(e)]]
                    out[i, j, k] = m
        return out

    def resolve_surface_face(self, surface: SurfaceDefinition, case: CaseDefinition) -> str:
        """``*SURFACE`` が領域 6 面のどれか 1 面全体に一致するかを判定して面名を返す.

        Raises
        ------
        UnsupportedMeshError
            面が内部にある、複数面にまたがる、または面の一部しか覆っていない場合
        """
        nx, ny, nz = self.dimensions
        lookup = self._elem_lookup()
        node_lookup = {int(n): idx for idx, n in enumerate(case.nodes.ids.tolist())}
        faces_table = _HEX_FACES if self.ndim == 3 else _QUAD_FACES
        block_of: dict[int, tuple[int, int]] = {}
        for bi, block in enumerate(case.elements):
            for ri, e in enumerate(block.ids.tolist()):
                block_of[int(e)] = (bi, ri)

        counts: dict[str, set[int]] = {name: set() for name in FACE_NAMES}
        for entry in surface.entries:
            elem_ids = case.element_ids_of(entry.target)
            if entry.face not in faces_table:
                raise UnsupportedMeshError(
                    f"面ラベル {entry.face} は {self.ndim}D 要素では使えません"
                )
            local = faces_table[entry.face]
            for e in elem_ids.tolist():
                bi, ri = block_of[int(e)]
                conn = case.elements[bi].connectivity[ri]
                coords = case.nodes.coords[[node_lookup[int(n)] for n in conn]]
                center = coords.mean(axis=0)
                face_center = coords[list(local)].mean(axis=0)
                d = face_center - center
                axis = int(np.argmax(np.abs(d)))
                sign = "P" if d[axis] > 0 else "M"
                name = "XYZ"[axis] + sign
                i, j, k = self.element_ijk[lookup[int(e)]]
                on_boundary = {
                    "XM": i == 0,
                    "XP": i == nx - 1,
                    "YM": j == 0,
                    "YP": j == ny - 1,
                    "ZM": k == 0,
                    "ZP": k == nz - 1,
                }[name]
                if not on_boundary:
                    raise UnsupportedMeshError(
                        f"*SURFACE {surface.name}: 要素 {e} の面 {entry.face} は領域境界にありません"
                        "（内部面は未対応）"
                    )
                counts[name].add(int(e))

        hit = [name for name, s in counts.items() if s]
        if len(hit) != 1:
            raise UnsupportedMeshError(
                f"*SURFACE {surface.name} は 1 つの領域面に収まっていません: {hit}"
            )
        name = hit[0]
        required = {
            "XM": ny * nz,
            "XP": ny * nz,
            "YM": nx * nz,
            "YP": nx * nz,
            "ZM": nx * ny,
            "ZP": nx * ny,
        }[name]
        if len(counts[name]) != required:
            raise UnsupportedMeshError(
                f"*SURFACE {surface.name} は面 {name} の一部（{len(counts[name])}/{required} セル）"
                "しか覆っていません。部分面は未対応です"
            )
        return name


def _unique_lines(values: np.ndarray, tol: float) -> np.ndarray:
    s = np.sort(values)
    lines = [s[0]]
    for v in s[1:]:
        if v - lines[-1] > tol:
            lines.append(v)
    return np.asarray(lines, dtype=float)


def recover_structured_grid(
    case: CaseDefinition, rel_tol: float = 1.0e-8, depth_2d: float = 1.0
) -> StructuredGridMap:
    """節点・要素表から構造格子を復元する（:class:`StructuredGridRecoveryProcess` の本体）."""
    if not case.elements:
        raise UnsupportedMeshError("要素がありません")
    widths = {b.nodes_per_element for b in case.elements}
    if len(widths) != 1:
        raise UnsupportedMeshError("2D 要素と 3D 要素が混在しています")
    ndim = 3 if widths.pop() == 8 else 2

    coords = case.nodes.coords
    extent = np.ptp(coords, axis=0)
    scale = float(np.max(extent)) if np.max(extent) > 0 else 1.0
    tol = rel_tol * scale

    x_lines = _unique_lines(coords[:, 0], tol)
    y_lines = _unique_lines(coords[:, 1], tol)
    if ndim == 3:
        z_lines = _unique_lines(coords[:, 2], tol)
    else:
        if np.ptp(coords[:, 2]) > tol:
            raise UnsupportedMeshError("2D 要素なのに節点の z 座標が一様ではありません")
        z0 = float(coords[0, 2])
        z_lines = np.array([z0, z0 + depth_2d])

    nx, ny = len(x_lines) - 1, len(y_lines) - 1
    nz = len(z_lines) - 1
    if nx < 1 or ny < 1 or nz < 1:
        raise UnsupportedMeshError("格子線が不足しています（各方向 2 本以上必要）")

    expected_nodes = (nx + 1) * (ny + 1) * ((nz + 1) if ndim == 3 else 1)
    if case.nodes.n_nodes != expected_nodes:
        raise UnsupportedMeshError(
            f"節点数 {case.nodes.n_nodes} が箱格子の {expected_nodes} と一致しません"
            "（非構造・欠損・非直交メッシュは未対応）"
        )
    expected_cells = nx * ny * (nz if ndim == 3 else 1)
    if case.n_elements != expected_cells:
        raise UnsupportedMeshError(
            f"要素数 {case.n_elements} が箱格子の {expected_cells} と一致しません"
        )

    # 節点 → 格子線添字
    def _index(lines: np.ndarray, v: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(lines, v - tol)
        idx = np.clip(idx, 0, len(lines) - 1)
        if np.any(np.abs(lines[idx] - v) > tol):
            raise UnsupportedMeshError("節点が格子線上にありません（非直交メッシュ）")
        return idx

    ni = _index(x_lines, coords[:, 0])
    nj = _index(y_lines, coords[:, 1])
    nk = _index(z_lines, coords[:, 2]) if ndim == 3 else np.zeros(case.nodes.n_nodes, dtype=int)
    node_ijk = np.stack([ni, nj, nk], axis=1)
    seen = set(map(tuple, node_ijk.tolist()))
    if len(seen) != case.nodes.n_nodes:
        raise UnsupportedMeshError("同じ格子点に複数の節点があります")

    node_lookup = {int(n): idx for idx, n in enumerate(case.nodes.ids.tolist())}
    xc, yc, zc = (
        0.5 * (x_lines[:-1] + x_lines[1:]),
        0.5 * (y_lines[:-1] + y_lines[1:]),
        0.5 * (z_lines[:-1] + z_lines[1:]),
    )
    elem_ids: list[int] = []
    elem_ijk: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int, int]] = set()
    for block in case.elements:
        idx = np.vectorize(node_lookup.__getitem__)(block.connectivity)
        centers = coords[idx].mean(axis=1)
        i = np.argmin(np.abs(xc[None, :] - centers[:, [0]]), axis=1)
        j = np.argmin(np.abs(yc[None, :] - centers[:, [1]]), axis=1)
        k = (
            np.argmin(np.abs(zc[None, :] - centers[:, [2]]), axis=1)
            if ndim == 3
            else np.zeros(block.n_elements, dtype=int)
        )
        # 要素の各節点がそのセルの 8（4）隅であることを検証
        for row in range(block.n_elements):
            corner = node_ijk[idx[row]]
            lo = corner.min(axis=0)
            hi = corner.max(axis=0)
            expect_hi = lo + (np.array([1, 1, 1]) if ndim == 3 else np.array([1, 1, 0]))
            if not (
                np.array_equal(hi, expect_hi)
                and lo[0] == i[row]
                and lo[1] == j[row]
                and lo[2] == k[row]
            ):
                raise UnsupportedMeshError(
                    f"要素 {int(block.ids[row])} は軸平行の単一セルになっていません"
                )
            key = (int(i[row]), int(j[row]), int(k[row]))
            if key in occupied:
                raise UnsupportedMeshError(f"セル {key} を複数の要素が占めています")
            occupied.add(key)
            elem_ids.append(int(block.ids[row]))
            elem_ijk.append(key)

    return StructuredGridMap(
        x_lines=x_lines,
        y_lines=y_lines,
        z_lines=z_lines,
        element_ids=np.asarray(elem_ids, dtype=int),
        element_ijk=np.asarray(elem_ijk, dtype=int),
        node_ids=case.nodes.ids.copy(),
        node_ijk=node_ijk,
        ndim=ndim,
    )


class StructuredGridRecoveryProcess(PreProcess["StructuredGridInput", "StructuredGridMap"]):
    """``*NODE`` / ``*ELEMENT`` を構造直交格子として解釈する PreProcess.

    箱格子でない場合は :class:`UnsupportedMeshError` を送出する。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="StructuredGridRecovery",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: StructuredGridInput) -> StructuredGridMap:
        return recover_structured_grid(input_data.case, input_data.rel_tol, input_data.depth_2d)
