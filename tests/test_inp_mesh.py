"""InpMeshProcess（.inp → 面ベース非構造 MeshData）のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import kuhn_tet_text

from xkep_cae_fluid.core.data import (
    CELL_TYPE_HEX,
    CELL_TYPE_PYRAMID,
    CELL_TYPE_TET,
    CELL_TYPE_WEDGE,
)
from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.fvm import PatchBC, diffusive_face_flux, resolve_boundary
from xkep_cae_fluid.heat_transfer import HeatTransferFVMInput, HeatTransferFVMProcess
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.grid import (
    StructuredGridInput,
    StructuredGridRecoveryProcess,
    UnsupportedMeshError,
)
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess, InpMeshResult, build_inp_mesh
from xkep_cae_fluid.inp.parser import InpSyntaxError, parse_inp_file, parse_inp_text

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "inp"

MIXED_TEXT = """\
*NODE
1,0,0,0
2,1,0,0
3,2,0,0
4,0,1,0
5,1,1,0
6,2,1,0
7,0,0,1
8,1,0,1
9,2,0,1
10,0,1,1
11,1,1,1
12,2,1,1
*ELEMENT, TYPE=C3D8, ELSET=HEX
1, 1,2,5,4,7,8,11,10
*ELEMENT, TYPE=C3D6, ELSET=WEDGE
2, 2,3,6,8,9,12
3, 2,6,5,8,12,11
*SURFACE, NAME=RIGHT, TYPE=ELEMENT
2, S4
*SURFACE, NAME=BOTTOM, TYPE=ELEMENT
HEX, S1
WEDGE, S1
"""


def _closure(m) -> float:
    av = m.face_normals * m.face_areas[:, None]
    closure = np.zeros((m.n_cells, 3))
    np.add.at(closure, m.face_owner, av)
    np.add.at(closure, m.face_neighbour, -av[: m.n_internal_faces])
    return float(np.abs(closure).max())


def _tri_sheet_text(nx: int, ny: int) -> str:
    """矩形を三角形 2 個 / 升目で割った CPS3 メッシュ + 左辺の *SURFACE LEFT."""

    def nid(i: int, j: int) -> int:
        return 1 + i + (nx + 1) * j

    lines = ["*NODE"]
    for j in range(ny + 1):
        for i in range(nx + 1):
            lines.append(f" {nid(i, j)}, {i / nx}, {j / ny}")
    lines.append("*ELEMENT, TYPE=CPS3, ELSET=TRI")
    e = 0
    for j in range(ny):
        for i in range(nx):
            e += 1
            lines.append(f" {e}, {nid(i, j)}, {nid(i + 1, j)}, {nid(i + 1, j + 1)}")
            e += 1
            lines.append(f" {e}, {nid(i, j)}, {nid(i + 1, j + 1)}, {nid(i, j + 1)}")
    lines.append("*SURFACE, NAME=LEFT, TYPE=ELEMENT")
    for j in range(ny):
        lines.append(f" {2 * (j * nx) + 2}, S3")  # 升目 i=0 の 2 個目: 辺 (i, j+1) → (i, j)
    return "\n".join(lines) + "\n"


def _hex_mesh_text(
    nx: int, ny: int, nz: int, spacing=(1.0, 1.0, 1.0), shuffle: bool = False, rotate: bool = False
) -> str:
    """Abaqus 風 *NODE/*ELEMENT テキスト（test_inp_grid と同じ規則。ID 100+ / 要素 11+）."""
    rng = np.random.default_rng(0)
    dx, dy, dz = spacing
    xs = np.arange(nx + 1) * dx
    ys = np.arange(ny + 1) * dy
    zs = np.arange(nz + 1) * dz

    def nid(i, j, k):
        return 100 + i + (nx + 1) * (j + (ny + 1) * k)

    nodes = [
        (nid(i, j, k), xs[i], ys[j], zs[k])
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]
    elems = []
    e = 10
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e += 1
                c = [
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]
                if rotate:
                    c = c[1:4] + c[:1] + c[5:8] + c[4:5]
                elems.append((e, c))
    if shuffle:
        rng.shuffle(nodes)
        rng.shuffle(elems)
    lines = ["*NODE"] + [f" {n}, {x}, {y}, {z}" for n, x, y, z in nodes]
    lines += ["*ELEMENT, TYPE=C3D8, ELSET=BOX"] + [
        f" {e}, " + ", ".join(map(str, c)) for e, c in elems
    ]
    return "\n".join(lines) + "\n"


def _quad_sheet_text(nx: int, ny: int) -> str:
    """CPS4 の nx×ny シート（節点 1.., 要素 1 + i + nx j、辺 S2 が +x 側）."""
    lines = ["*NODE"]
    for j in range(ny + 1):
        for i in range(nx + 1):
            lines.append(f" {1 + i + (nx + 1) * j}, {i}, {j}, 0")
    lines.append("*ELEMENT, TYPE=CPS4, ELSET=ALL")
    for j in range(ny):
        for i in range(nx):
            n0 = 1 + i + (nx + 1) * j
            lines.append(f" {1 + i + nx * j}, {n0}, {n0 + 1}, {n0 + nx + 2}, {n0 + nx + 1}")
    return "\n".join(lines) + "\n"


def _pyramid_cube_text() -> str:
    """単位立方体を中心 (0.5,0.5,0.5) を頂点とする角錐 6 個に分ける（C3D5、S1 が底面）.

    要素 1（−x 面）だけ底面を裏向き（頂点が底面の法線の負側）に書き、向きの正規化を通す。
    """
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    lines = ["*NODE"] + [f" {i + 1}, {x}, {y}, {z}" for i, (x, y, z) in enumerate(corners)]
    lines.append(" 9, 0.5, 0.5, 0.5")
    # 底面は外向き法線が外皮を向く並び（頂点 9 は内側 = 負側）→ 要素 1 は敢えて逆順
    bases = {
        1: (1, 4, 8, 5),  # −x（外向き −x: 逆順で書く）
        2: (2, 3, 7, 6),  # +x
        3: (1, 5, 6, 2),  # −y
        4: (4, 3, 7, 8),  # +y（順序は面ラベル検査のため内向きでも可）
        5: (1, 2, 3, 4),  # −z
        6: (5, 8, 7, 6),  # +z
    }
    lines.append("*ELEMENT, TYPE=C3D5, ELSET=PYR")
    for e, b in bases.items():
        lines.append(f" {e}, " + ", ".join(map(str, b)) + ", 9")
    lines.append("*SURFACE, NAME=BASES, TYPE=ELEMENT")
    lines.append(" PYR, S1")
    return "\n".join(lines) + "\n"


_C3D20_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _c3d20_from_linear(text: str) -> str:
    """C3D8 の *NODE/*ELEMENT テキストから、辺の中点を足した C3D20 テキストを作る."""
    case = build_case(parse_inp_text(text))
    ids = case.nodes.ids.tolist()
    coords = {int(n): case.nodes.coords[k] for k, n in enumerate(ids)}
    next_id = max(ids) + 1
    mid: dict[tuple[int, int], int] = {}
    lines = ["*NODE"] + [f" {n}, {coords[n][0]}, {coords[n][1]}, {coords[n][2]}" for n in ids]
    elems = []
    for b in case.elements:
        for eid, row in zip(b.ids.tolist(), b.connectivity.tolist(), strict=True):
            extra = []
            for a, c in _C3D20_EDGES:
                key = (min(row[a], row[c]), max(row[a], row[c]))
                if key not in mid:
                    mid[key] = next_id
                    xyz = 0.5 * (coords[row[a]] + coords[row[c]])
                    lines.append(f" {next_id}, {xyz[0]}, {xyz[1]}, {xyz[2]}")
                    next_id += 1
                extra.append(mid[key])
            elems.append(f" {eid}, " + ", ".join(map(str, list(row) + extra)))
    lines += ["*ELEMENT, TYPE=C3D20, ELSET=BOX"] + elems
    return "\n".join(lines) + "\n"


def _cps8_sheet_text(nx: int, ny: int) -> str:
    """CPS8（8 節点四辺形）の nx×ny シート。中間節点は辺の中点."""
    lines = ["*NODE"]
    nid = 1
    corner = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            corner[(i, j)] = nid
            lines.append(f" {nid}, {i}, {j}, 0")
            nid += 1
    mid = {}

    def mid_id(a, b):
        key = (min(a, b), max(a, b))
        nonlocal nid
        if key not in mid:
            mid[key] = nid
            xa, ya = a
            xb, yb = b
            lines.append(f" {nid}, {(xa + xb) / 2}, {(ya + yb) / 2}, 0")
            nid += 1
        return mid[key]

    elems = []
    for j in range(ny):
        for i in range(nx):
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            ids = [corner[q] for q in c] + [mid_id(c[k], c[(k + 1) % 4]) for k in range(4)]
            elems.append(f" {1 + i + nx * j}, " + ", ".join(map(str, ids)))
    lines += ["*ELEMENT, TYPE=CPS8, ELSET=ALL"] + elems
    return "\n".join(lines) + "\n"


def _structured_index(grid, result: InpMeshResult) -> np.ndarray:
    """InpMesh のセル順 → 構造格子のセル index."""
    nx, ny, nz = grid.dimensions
    lookup = {int(e): idx for idx, e in enumerate(grid.element_ids.tolist())}
    ijk = np.array([grid.element_ijk[lookup[int(e)]] for e in result.element_ids.tolist()])
    return ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]


@binds_to(InpMeshProcess)
class TestInpMeshAPI:
    def test_meta(self):
        assert InpMeshProcess.meta.name == "InpMesh"
        assert InpMeshProcess.meta.module == "pre"

    def test_returns_result(self):
        case = build_case(parse_inp_text(_hex_mesh_text(2, 2, 1)))
        res = InpMeshProcess().execute(InpMeshInput(case=case))
        assert isinstance(res, InpMeshResult)
        assert res.ndim == 3
        assert res.mesh.n_cells == 4
        assert res.mesh.dimensions is None
        assert res.mesh.n_internal_faces == 4
        assert res.mesh.n_boundary_faces == 16
        assert set(res.mesh.boundary_patches) == {"XM", "XP", "YM", "YP", "ZM", "ZP"}

    def test_rejects_mixed_elements(self):
        text = (
            _hex_mesh_text(1, 1, 1)
            + "*NODE\n900,5,0,0\n901,6,0,0\n902,6,1,0\n903,5,1,0\n"
            + "*ELEMENT, TYPE=CPS4\n500,900,901,902,903\n"
        )
        with pytest.raises(UnsupportedMeshError, match="混在"):
            build_inp_mesh(build_case(parse_inp_text(text)))

    def test_rejects_quadratic_and_unknown_face_labels(self):
        text = MIXED_TEXT.replace("2, S4", "2, S6")  # 楔に S6 は無い
        with pytest.raises(UnsupportedMeshError, match="S6"):
            build_inp_mesh(build_case(parse_inp_text(text)))
        tri = _tri_sheet_text(2, 1).replace("S3\n", "S4\n")  # 三角形の辺は S1..S3
        with pytest.raises(UnsupportedMeshError, match="S4"):
            build_inp_mesh(build_case(parse_inp_text(tri)))

    def test_surface_with_internal_face_is_not_a_patch_unless_baffle(self):
        """内部面を含む *SURFACE はパッチにならない。baffle_surfaces に挙げると両側の境界面に分割される."""
        text = _hex_mesh_text(2, 1, 1) + "*SURFACE, NAME=MID, TYPE=ELEMENT\n11, S4\n"
        case = build_case(parse_inp_text(text))
        res = build_inp_mesh(case)
        assert "MID" not in (res.mesh.boundary_patches or {})
        assert res.surface_faces["MID"].tolist() == [0] and res.mesh.n_internal_faces == 1
        assert res.baffle_surfaces == () and len(res.baffle_faces) == 0
        # 境界面だけの *SURFACE を挙げても何も起きない
        same = build_inp_mesh(case, baffle_surfaces=("MID",))
        assert same.baffle_surfaces == ("MID",) and len(same.baffle_faces) == 2
        m = same.mesh
        assert m.n_internal_faces == 0 and m.n_boundary_faces == 12
        assert m.patch_faces("MID").tolist() == [10, 11]
        # 分割した 2 面: 片方は要素 11 が owner で +x 向き、もう片方は要素 12 が owner で −x 向き
        assert m.face_owner[10] == 0 and m.face_owner[11] == 1
        np.testing.assert_allclose(m.face_normals[10], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(m.face_normals[11], [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(m.face_centers[10], m.face_centers[11])
        # 予約面名 XM/XP はバッフル面を拾わない。体積・閉包は変わらない
        assert len(m.patch_faces("XM")) == 1 and len(m.patch_faces("XP")) == 1
        np.testing.assert_allclose(m.cell_volumes, res.mesh.cell_volumes)
        assert _closure(m) < 1e-12
        with pytest.raises(UnsupportedMeshError, match="SURFACE"):
            build_inp_mesh(case, baffle_surfaces=("NOPE",))

    def test_baffle_in_2d_sheet_and_surface_from_other_side(self):
        """2D 四辺形の辺をバッフルにする。反対側の要素から指定しても両側が入る."""
        text = _quad_sheet_text(4, 2) + "*SURFACE, NAME=MID, TYPE=ELEMENT\n2, S2\n6, S2\n"
        case = build_case(parse_inp_text(text))
        res = build_inp_mesh(case, baffle_surfaces=("MID",))
        m = res.mesh
        assert res.baffle_surfaces == ("MID",) and len(res.baffle_faces) == 4
        assert m.n_internal_faces == 10 - 2 and set(m.patch_faces("MID").tolist()) == set(
            res.baffle_faces.tolist()
        )
        np.testing.assert_allclose(np.abs(m.face_normals[res.baffle_faces][:, 0]), 1.0)
        # 反対側（要素 3, 7 の S4）で指定しても同じ 4 面
        other = text.replace("2, S2\n6, S2\n", "3, S4\n7, S4\n")
        res2 = build_inp_mesh(build_case(parse_inp_text(other)), baffle_surfaces=("MID",))
        assert set(res2.surface_faces["MID"].tolist()) == set(res.surface_faces["MID"].tolist())

    def test_pyramids_fill_cube(self):
        """立方体を中心頂点の角錐 6 個に分けたメッシュ: 体積・閉包・面数・種別・面ラベル・向きの正規化."""
        res = build_inp_mesh(build_case(parse_inp_text(_pyramid_cube_text())))
        m = res.mesh
        assert m.n_cells == 6 and np.all(m.cell_types == CELL_TYPE_PYRAMID)
        assert m.connectivity.shape == (6, 5) and np.all(m.connectivity >= 0)
        assert m.cell_volumes.sum() == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(m.cell_volumes, 1.0 / 6.0, rtol=1e-12)
        assert m.n_internal_faces == 12 and m.n_boundary_faces == 6
        assert _closure(m) < 1e-12
        for name in ("XM", "XP", "YM", "YP", "ZM", "ZP"):
            assert len(m.patch_faces(name)) == 1
        assert m.patch_faces("BASES").tolist() == m.patch_faces("XM").tolist() + [
            int(i) for i in sorted(set(range(12, 18)) - set(m.patch_faces("XM").tolist()))
        ]
        # 頂点が底面の裏側にある（負の体積）順序の角錐も右手系に正規化される
        p = m.node_coords[m.connectivity]
        triple = np.einsum(
            "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]), p[:, 4] - p[:, 0]
        )
        assert np.all(triple > 0)

    def test_quadratic_elements_use_corner_nodes(self):
        """C3D20 / CPS8 は頂点だけを使い、1 次要素と同じ面リスト・体積・面ラベルになる."""
        lin = build_inp_mesh(
            build_case(
                parse_inp_text(_hex_mesh_text(2, 1, 1) + "*SURFACE, NAME=R, TYPE=ELEMENT\n12, S4\n")
            )
        )
        quad = build_inp_mesh(
            build_case(
                parse_inp_text(
                    _c3d20_from_linear(_hex_mesh_text(2, 1, 1))
                    + "*SURFACE, NAME=R, TYPE=ELEMENT\n12, S4\n"
                )
            )
        )
        assert quad.mesh.n_cells == 2 and quad.mesh.connectivity.shape == (2, 8)
        assert quad.mesh.n_internal_faces == lin.mesh.n_internal_faces
        np.testing.assert_allclose(quad.mesh.cell_volumes, lin.mesh.cell_volumes)
        np.testing.assert_allclose(quad.mesh.face_areas, lin.mesh.face_areas)
        assert quad.surface_faces["R"].tolist() == lin.surface_faces["R"].tolist()
        assert quad.mesh.n_nodes == lin.mesh.n_nodes + 20  # 中間節点は座標表に残る（未使用）
        # 2D の 8 節点四辺形
        sheet = build_inp_mesh(build_case(parse_inp_text(_cps8_sheet_text(3, 2))))
        assert sheet.mesh.n_cells == 6 and sheet.mesh.connectivity.shape == (6, 8)
        assert sheet.mesh.cell_volumes.sum() == pytest.approx(6.0, rel=1e-12)
        assert _closure(sheet.mesh) < 1e-12
        with pytest.raises(InpSyntaxError, match="未対応"):
            build_case(
                parse_inp_text("*NODE\n1,0,0,0\n2,1,0,0\n3,0,1,0\n*ELEMENT, TYPE=C3D3\n1,1,2,3\n")
            )


class TestInpMeshPhysics:
    @pytest.mark.parametrize("shuffle,rotate", [(False, False), (True, True)])
    def test_box_lattice_matches_structured_mesh(self, shuffle: bool, rotate: bool):
        """同じ箱格子を構造経路と面経路で作ると体積・面積・隣接・パッチが一致する."""
        nx, ny, nz = 4, 3, 2
        case = build_case(
            parse_inp_text(
                _hex_mesh_text(nx, ny, nz, spacing=(0.5, 0.25, 1.0), shuffle=shuffle, rotate=rotate)
            )
        )
        grid = StructuredGridRecoveryProcess().execute(StructuredGridInput(case=case))
        res = InpMeshProcess().execute(InpMeshInput(case=case))
        sm = (
            StructuredMeshProcess()
            .execute(StructuredMeshInput(Lx=2.0, Ly=0.75, Lz=2.0, nx=nx, ny=ny, nz=nz))
            .mesh
        )
        um = res.mesh
        perm = _structured_index(grid, res)

        np.testing.assert_allclose(um.cell_volumes, sm.cell_volumes[perm], rtol=1e-12)
        np.testing.assert_allclose(um.cell_centers, sm.cell_centers[perm], atol=1e-12)
        assert um.n_internal_faces == sm.n_internal_faces
        assert um.n_boundary_faces == sm.n_boundary_faces
        # 隣接グラフ（構造 index に写像した owner/neighbour の対）
        n_int = um.n_internal_faces
        pairs_u = {
            tuple(sorted((int(perm[o]), int(perm[n]))))
            for o, n in zip(um.face_owner[:n_int], um.face_neighbour, strict=True)
        }
        pairs_s = {
            tuple(sorted((int(o), int(n))))
            for o, n in zip(sm.face_owner[: sm.n_internal_faces], sm.face_neighbour, strict=True)
        }
        assert pairs_u == pairs_s
        # 面積の多重集合
        np.testing.assert_allclose(np.sort(um.face_areas), np.sort(sm.face_areas), rtol=1e-12)
        # 各パッチの面中心集合
        for name in ("XM", "XP", "YM", "YP", "ZM", "ZP"):
            cu = np.sort(um.face_centers[um.patch_faces(name)], axis=0)
            cs = np.sort(sm.face_centers[sm.patch_faces(name)], axis=0)
            np.testing.assert_allclose(cu, cs, atol=1e-12)
        # 内部面法線は owner → neighbour 向き
        d = um.cell_centers[um.face_neighbour] - um.cell_centers[um.face_owner[:n_int]]
        assert np.all(np.sum(d * um.face_normals[:n_int], axis=1) > 0)
        # 境界面法線は外向き
        b = um.boundary_faces
        d_b = um.face_centers[b] - um.cell_centers[um.face_owner[b]]
        assert np.all(np.sum(d_b * um.face_normals[b], axis=1) > 0)

    def test_sheared_hex_mesh_is_accepted(self):
        """箱格子でない（せん断で歪んだ）六面体メッシュも面リストになる."""
        text = _hex_mesh_text(3, 2, 2)
        case = build_case(parse_inp_text(text))
        # 節点をせん断変形: x += 0.3 y、z += 0.2 x
        coords = case.nodes.coords.copy()
        coords[:, 0] += 0.3 * coords[:, 1]
        coords[:, 2] += 0.2 * coords[:, 0]
        from dataclasses import replace

        case2 = replace(case, nodes=replace(case.nodes, coords=coords))
        with pytest.raises(UnsupportedMeshError):
            StructuredGridRecoveryProcess().execute(StructuredGridInput(case=case2))
        res = InpMeshProcess().execute(InpMeshInput(case=case2))
        m = res.mesh
        assert m.n_cells == 12
        # せん断は体積を変えない（各セル 1.0）
        np.testing.assert_allclose(m.cell_volumes, 1.0, rtol=1e-12)
        # 各セルの面積ベクトル和はゼロ（閉じている）
        av = m.face_normals * m.face_areas[:, None]
        closure = np.zeros((m.n_cells, 3))
        np.add.at(closure, m.face_owner, av)
        np.add.at(closure, m.face_neighbour, -av[: m.n_internal_faces])
        np.testing.assert_allclose(closure, 0.0, atol=1e-12)

    def test_quad_mesh_is_extruded(self):
        """4 節点四辺形は depth_2d で押し出され、ZM/ZP パッチを持つ."""
        text = (
            "*NODE\n1,0,0\n2,1,0\n3,2,0\n4,0,1\n5,1,1\n6,2,1\n"
            "*ELEMENT, TYPE=CPS4, ELSET=SHEET\n1,1,2,5,4\n2,2,3,6,5\n"
            "*SURFACE, NAME=LEFT, TYPE=ELEMENT\n1, S4\n"
        )
        case = build_case(parse_inp_text(text))
        res = InpMeshProcess().execute(InpMeshInput(case=case, depth_2d=0.5))
        m = res.mesh
        assert res.ndim == 2
        assert m.n_cells == 2 and m.n_nodes == 12
        np.testing.assert_allclose(m.cell_volumes, 0.5)
        assert m.n_internal_faces == 1
        assert len(m.patch_faces("ZM")) == 2 and len(m.patch_faces("ZP")) == 2
        left = m.patch_faces("LEFT")
        assert len(left) == 1
        np.testing.assert_allclose(m.face_normals[left[0]], [-1.0, 0.0, 0.0])
        assert res.node_ids.tolist() == [1, 2, 3, 4, 5, 6] * 2

    def test_example_plate_surfaces_match_structured(self):
        """plate-mesh.inp の *SURFACE（LEFT/RIGHT/TOP）が構造格子側の面判定と一致する."""
        case = build_case(parse_inp_file(str(EXAMPLES / "plate-mesh.inp")))
        grid = StructuredGridRecoveryProcess().execute(StructuredGridInput(case=case))
        res = InpMeshProcess().execute(InpMeshInput(case=case))
        m = res.mesh
        for name in ("LEFT", "RIGHT", "TOP"):
            face = grid.resolve_surface_face(case.surfaces[name], case)
            np.testing.assert_array_equal(
                np.sort(m.patch_faces(name)), np.sort(m.patch_faces(face))
            )
        # elset → セル集合
        assert len(res.cell_sets["HEATER"]) == len(case.elsets["HEATER"].ids)
        mask = res.mask_for_elements(case.elsets["HEATER"].ids)
        assert mask.sum() == len(case.elsets["HEATER"].ids)
        # 節点値 → セル値（全節点同じ値なら全セルその値）
        vals = res.node_values_to_cells(case.nodes.ids, np.full(case.nodes.n_nodes, 3.0))
        np.testing.assert_allclose(vals, 3.0)

    def test_tet_mesh_kuhn_cube(self):
        """Kuhn 分割の四面体（4×4×4 の立方体 → 384 個）: 体積・閉包・予約パッチ・セル種別."""
        res = build_inp_mesh(build_case(parse_inp_text(kuhn_tet_text(4, 4, 4))))
        m = res.mesh
        assert m.n_cells == 384 and m.connectivity.shape == (384, 4)
        assert np.all(m.cell_types == CELL_TYPE_TET) and np.all(m.connectivity >= 0)
        assert m.cell_volumes.sum() == pytest.approx(1.0, rel=1e-12)
        assert m.n_internal_faces == 672 and m.n_boundary_faces == 192
        assert _closure(m) < 1e-12
        for name in ("XM", "XP", "YM", "YP", "ZM", "ZP"):
            assert len(m.patch_faces(name)) == 32
        # 四面体は右手系（節点 0 から見て 1,2,3 が正の体積）に正規化されている
        p = m.node_coords[m.connectivity]
        triple = np.einsum(
            "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
        )
        assert np.all(triple > 0)
        assert len(res.cell_sets["TETS"]) == 384

    def test_tet_mesh_heat_conduction_linear(self):
        """四面体メッシュの熱伝導（両端 Dirichlet、側面断熱）: 線形分布と熱流束 kA ΔT/L."""
        m = build_inp_mesh(build_case(parse_inp_text(kuhn_tet_text(4, 4, 4)))).mesh
        bcs = {"XM": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(1.0)}
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=m,
                conductivity=2.0,
                T0=np.zeros(m.n_cells),
                bcs=bcs,
                linear_solver="direct",
                max_nonorthogonal_iter=50,
            )
        )
        assert res.converged
        np.testing.assert_allclose(res.T, m.cell_centers[:, 0], atol=1e-5)
        flux = diffusive_face_flux(m, res.T, 2.0, resolve_boundary(m, bcs))
        q_in = float(flux[m.patch_faces("XP")].sum())  # T_b > T_P なので owner へ流入（負）
        assert -q_in == pytest.approx(2.0, rel=1e-5)
        # 流入 = 流出（遅延補正の収束判定 1e-8 のオーダーで一致）
        assert float(flux[m.patch_faces("XM")].sum()) == pytest.approx(-q_in, rel=1e-6)

    def test_tet_mesh_all_dirichlet_linear_field_exact(self):
        """全面 Dirichlet の線形場は四面体でも 1e-7 で再現（スキュー補正 + 非直交補正）."""
        m = build_inp_mesh(build_case(parse_inp_text(kuhn_tet_text(3, 3, 3)))).mesh

        def lin(pts):
            return 1.0 + 2.0 * pts[:, 0] - 0.5 * pts[:, 1] + 3.0 * pts[:, 2]

        bcs = {
            nm: PatchBC.dirichlet(lin(m.face_centers[m.patch_faces(nm)]))
            for nm in ("XM", "XP", "YM", "YP", "ZM", "ZP")
        }
        res = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=m,
                conductivity=1.0,
                T0=np.zeros(m.n_cells),
                bcs=bcs,
                linear_solver="direct",
                max_nonorthogonal_iter=50,
            )
        )
        np.testing.assert_allclose(res.T, lin(m.cell_centers), atol=1e-7)

    def test_wedge_mesh_from_triangles(self):
        """三角形（CPS3）は楔に押し出され、辺の *SURFACE が側面になる。熱伝導の線形分布は厳密."""
        res = build_inp_mesh(build_case(parse_inp_text(_tri_sheet_text(4, 3))), depth_2d=0.5)
        m = res.mesh
        assert res.ndim == 2 and m.n_cells == 24 and m.connectivity.shape == (24, 6)
        assert np.all(m.cell_types == CELL_TYPE_WEDGE)
        assert m.cell_volumes.sum() == pytest.approx(0.5, rel=1e-12)
        assert _closure(m) < 1e-12
        assert len(m.patch_faces("ZM")) == 24 and len(m.patch_faces("ZP")) == 24
        left = m.patch_faces("LEFT")
        assert len(left) == 3
        np.testing.assert_allclose(m.face_normals[left], [[-1.0, 0.0, 0.0]] * 3, atol=1e-12)
        bcs = {"LEFT": PatchBC.dirichlet(0.0), "XP": PatchBC.dirichlet(1.0)}
        out = HeatTransferFVMProcess().execute(
            HeatTransferFVMInput(
                mesh=m, conductivity=1.0, T0=np.zeros(24), bcs=bcs, linear_solver="direct"
            )
        )
        np.testing.assert_allclose(out.T, m.cell_centers[:, 0], atol=1e-8)

    def test_mixed_hex_and_wedges(self):
        """六面体 1 個 + 楔 2 個（2 個目の升目を対角で割る）: 種別ごとの体積・-1 詰め・面ラベル."""
        res = build_inp_mesh(build_case(parse_inp_text(MIXED_TEXT)))
        m = res.mesh
        np.testing.assert_allclose(m.cell_volumes, [1.0, 0.5, 0.5])
        assert m.cell_types.tolist() == [CELL_TYPE_HEX, CELL_TYPE_WEDGE, CELL_TYPE_WEDGE]
        assert m.connectivity.shape == (3, 8)
        assert np.all(m.connectivity[0] >= 0) and np.all(m.connectivity[1:, 6:] == -1)
        assert m.n_internal_faces == 2  # 六面体–楔 1 面、楔–楔（対角面）1 面
        right = m.patch_faces("RIGHT")
        np.testing.assert_allclose(m.face_normals[right], [[1.0, 0.0, 0.0]])
        assert len(m.patch_faces("BOTTOM")) == 3 and len(m.patch_faces("XP")) == 1
        assert res.cell_sets["WEDGE"].tolist() == [1, 2]
        assert _closure(m) < 1e-12

    def test_inverted_elements_keep_face_labels(self):
        """節点順が左手系の要素（底面と上面が逆）でも S1 は元の節点順で定義した面を指す."""
        text = MIXED_TEXT.replace("1, 1,2,5,4,7,8,11,10", "1, 7,8,11,10,1,2,5,4").replace(
            "2, 2,3,6,8,9,12", "2, 8,9,12,2,3,6"
        )
        res = build_inp_mesh(build_case(parse_inp_text(text)))
        m = res.mesh
        np.testing.assert_allclose(m.cell_volumes, [1.0, 0.5, 0.5])
        bottom = m.patch_faces("BOTTOM")
        assert len(bottom) == 3
        # 反転した要素 1・2 の S1 は z = 1 の面（元の節点順の「底面」）
        z = m.face_centers[bottom, 2]
        assert sorted(np.round(z, 12).tolist()) == [0.0, 1.0, 1.0]
        # 出力の接続は右手系に正規化されている（六面体: (n1−n0)×(n3−n0)·(n4−n0) > 0）
        p = m.node_coords[m.connectivity[0]]
        assert np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[4] - p[0]) > 0
