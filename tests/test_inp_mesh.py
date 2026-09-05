"""InpMeshProcess（.inp → 面ベース非構造 MeshData）のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.grid import (
    StructuredGridInput,
    StructuredGridRecoveryProcess,
    UnsupportedMeshError,
)
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess, InpMeshResult, build_inp_mesh
from xkep_cae_fluid.inp.parser import parse_inp_file, parse_inp_text

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "inp"


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

    def test_surface_with_internal_face_rejected(self):
        text = _hex_mesh_text(2, 1, 1) + "*SURFACE, NAME=MID, TYPE=ELEMENT\n11, S4\n"
        case = build_case(parse_inp_text(text))
        with pytest.raises(UnsupportedMeshError, match="内部面"):
            build_inp_mesh(case)


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
