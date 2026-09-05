"""ykep .inp フォーマット: 構造格子復元（StructuredGridRecoveryProcess）のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.grid import (
    StructuredGridInput,
    StructuredGridRecoveryProcess,
    UnsupportedMeshError,
    recover_structured_grid,
)
from xkep_cae_fluid.inp.parser import parse_inp_text


def _hex_mesh_text(
    nx: int, ny: int, nz: int, spacing=(1.0, 1.0, 1.0), shuffle: bool = False, rotate: bool = False
) -> str:
    """Abaqus 風 *NODE/*ELEMENT テキストを生成（ID の順序や節点順序をかき混ぜられる）."""
    rng = np.random.default_rng(0)
    dx, dy, dz = spacing
    xs = (
        np.cumsum([0.0] + [dx * (1 + 0.5 * i) for i in range(nx)])
        if isinstance(dx, float) and dx < 0
        else np.arange(nx + 1) * dx
    )
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
                    # 底面を 1 つ回転（1-2-3-4 → 2-3-4-1）、上面も同様: 面ラベルの幾何判定を試す
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


@binds_to(StructuredGridRecoveryProcess)
class TestStructuredGridRecoveryAPI:
    def test_grid_keyword_recovery(self):
        case = build_case(
            parse_inp_text("*GRID, NX=3, NY=2, NZ=4, LX=3, LY=1, LZ=2, ORIGIN=1 2 3\n")
        )
        grid = StructuredGridRecoveryProcess().execute(StructuredGridInput(case=case))
        assert grid.dimensions == (3, 2, 4)
        assert grid.origin == (1.0, 2.0, 3.0)
        assert grid.lengths == pytest.approx((3.0, 1.0, 2.0))
        assert grid.is_uniform and grid.ndim == 3 and grid.n_cells == 24
        # 要素 1 は (0,0,0)、要素 2 は (1,0,0)（i が最内）
        assert grid.element_ijk[0].tolist() == [0, 0, 0]
        assert grid.element_ijk[1].tolist() == [1, 0, 0]

    def test_node_element_recovery_with_shuffled_ids_and_rotated_connectivity(self):
        case = build_case(parse_inp_text(_hex_mesh_text(3, 2, 2, shuffle=True, rotate=True)))
        grid = recover_structured_grid(case)
        assert grid.dimensions == (3, 2, 2)
        mask = grid.mask_for_elements(np.array([11]))  # 要素 11 = (0,0,0)
        assert mask[0, 0, 0] and mask.sum() == 1

    def test_2d_quads_get_unit_depth(self):
        text = "*NODE\n 1,0,0\n 2,1,0\n 3,2,0\n 4,0,1\n 5,1,1\n 6,2,1\n*ELEMENT, TYPE=CPS4\n 1,1,2,5,4\n 2,2,3,6,5\n"
        grid = recover_structured_grid(build_case(parse_inp_text(text)), depth_2d=0.25)
        assert grid.dimensions == (2, 1, 1) and grid.ndim == 2
        assert grid.lengths == pytest.approx((2.0, 1.0, 0.25))

    def test_nonuniform_spacing_detected(self):
        text = "*NODE\n 1,0,0,0\n 2,1,0,0\n 3,3,0,0\n 4,0,1,0\n 5,1,1,0\n 6,3,1,0\n 7,0,0,1\n 8,1,0,1\n 9,3,0,1\n 10,0,1,1\n 11,1,1,1\n 12,3,1,1\n*ELEMENT, TYPE=C3D8\n 1,1,2,5,4,7,8,11,10\n 2,2,3,6,5,8,9,12,11\n"
        grid = recover_structured_grid(build_case(parse_inp_text(text)))
        assert not grid.is_uniform
        assert grid.spacings[0].tolist() == [1.0, 2.0]

    def test_surface_face_resolution(self):
        text = _hex_mesh_text(2, 2, 1) + (
            "*ELSET, ELSET=LEFT\n 11, 13\n*ELSET, ELSET=PART\n 11\n"
            "*SURFACE, NAME=FULL_LEFT\n LEFT, S6\n"
            "*SURFACE, NAME=PARTIAL\n PART, S6\n"
            "*SURFACE, NAME=INTERIOR\n LEFT, S4\n"
            "*SURFACE, NAME=TOP\n BOX, S2\n"
            "*SURFACE, NAME=MIXED\n LEFT, S6\n BOX, S2\n"
        )
        case = build_case(parse_inp_text(text))
        grid = recover_structured_grid(case)
        assert grid.resolve_surface_face(case.surfaces["FULL_LEFT"], case) == "XM"
        assert grid.resolve_surface_face(case.surfaces["TOP"], case) == "ZP"
        with pytest.raises(UnsupportedMeshError, match="一部"):
            grid.resolve_surface_face(case.surfaces["PARTIAL"], case)
        with pytest.raises(UnsupportedMeshError, match="境界にありません"):
            grid.resolve_surface_face(case.surfaces["INTERIOR"], case)
        with pytest.raises(UnsupportedMeshError, match="収まって"):
            grid.resolve_surface_face(case.surfaces["MIXED"], case)

    def test_node_values_to_cells(self):
        case = build_case(parse_inp_text("*GRID, NX=2, NY=1, NZ=1, LX=2, LY=1, LZ=1\n"))
        grid = recover_structured_grid(case)
        # 左端の 4 節点（x=0）に 10、他は未指定 → 要素 1 は平均 10、要素 2 は NaN
        left = case.nodes.ids[case.nodes.coords[:, 0] == 0.0]
        cell = grid.node_values_to_cells(left, np.full(left.size, 10.0), case)
        assert cell[0, 0, 0] == 10.0 and np.isnan(cell[1, 0, 0])

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            # 要素が 1 つ欠けている（箱格子でない）
            (
                "*NODE\n 1,0,0,0\n 2,1,0,0\n 3,2,0,0\n 4,0,1,0\n 5,1,1,0\n 6,2,1,0\n 7,0,0,1\n 8,1,0,1\n 9,2,0,1\n 10,0,1,1\n 11,1,1,1\n 12,2,1,1\n*ELEMENT, TYPE=C3D8\n 1,1,2,5,4,7,8,11,10\n",
                "要素数",
            ),
            # 節点が格子線上にない（歪んだ要素）
            (
                "*NODE\n 1,0,0,0\n 2,1,0,0\n 3,1,1,0\n 4,0,1.2,0\n 5,0,0,1\n 6,1,0,1\n 7,1,1,1\n 8,0,1,1\n*ELEMENT, TYPE=C3D8\n 1,1,2,3,4,5,6,7,8\n",
                "節点数|格子線",
            ),
            # 2D と 3D の混在
            (
                "*NODE\n 1,0,0,0\n 2,1,0,0\n 3,1,1,0\n 4,0,1,0\n 5,0,0,1\n 6,1,0,1\n 7,1,1,1\n 8,0,1,1\n*ELEMENT, TYPE=C3D8\n 1,1,2,3,4,5,6,7,8\n*ELEMENT, TYPE=CPS4\n 2,1,2,3,4\n",
                "混在",
            ),
        ],
    )
    def test_unsupported_meshes(self, text: str, match: str):
        with pytest.raises(UnsupportedMeshError, match=match):
            recover_structured_grid(build_case(parse_inp_text(text)))
