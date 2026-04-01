"""PolyMeshReaderProcess テスト.

OpenFOAM polyMesh 形式のメッシュ読み込みをテストする。
テスト用に小さな 2x1x1 セルのメッシュデータを作成して検証する。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from xkep_cae_fluid.core.mesh_reader import (
    PolyMeshInput,
    PolyMeshReaderProcess,
    PolyMeshResult,
    _is_binary_format,
    parse_boundary,
    parse_faces,
    parse_faces_binary,
    parse_label_list,
    parse_label_list_binary,
    parse_points,
    parse_points_binary,
)
from xkep_cae_fluid.core.testing import binds_to


def _create_test_polymesh(mesh_dir: Path) -> None:
    """2x1x1 セルの polyMesh テストデータを作成.

    セル0: (0,0,0)-(0.5,1,1)
    セル1: (0.5,0,0)-(1,1,1)
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # 12 ノード (2+1) x (1+1) x (1+1)
    points_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    object      points;
}
12
(
(0 0 0)
(0.5 0 0)
(1 0 0)
(0 1 0)
(0.5 1 0)
(1 1 0)
(0 0 1)
(0.5 0 1)
(1 0 1)
(0 1 1)
(0.5 1 1)
(1 1 1)
)
"""

    # 11 面: 1内部面 + 10境界面
    # 内部面 (面0): x=0.5 の面 (1,4,10,7)
    # 境界面:
    #   face1: x=0 の面 (セル0) (0,6,9,3)
    #   face2: x=1 の面 (セル1) (2,5,11,8)
    #   face3: y=0 の面 (セル0) (0,1,7,6)
    #   face4: y=0 の面 (セル1) (1,2,8,7)
    #   face5: y=1 の面 (セル0) (3,9,10,4)
    #   face6: y=1 の面 (セル1) (4,10,11,5)
    #   face7: z=0 の面 (セル0) (0,3,4,1)
    #   face8: z=0 の面 (セル1) (1,4,5,2)
    #   face9: z=1 の面 (セル0) (6,7,10,9)
    #   face10: z=1 の面 (セル1) (7,8,11,10)
    faces_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    object      faces;
}
11
(
4(1 4 10 7)
4(0 6 9 3)
4(2 5 11 8)
4(0 1 7 6)
4(1 2 8 7)
4(3 9 10 4)
4(4 10 11 5)
4(0 3 4 1)
4(1 4 5 2)
4(6 7 10 9)
4(7 8 11 10)
)
"""

    # owner: 面0はセル0, 面1はセル0, 面2はセル1, ...
    owner_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      owner;
}
11
(
0
0
1
0
1
0
1
0
1
0
1
)
"""

    # neighbour: 内部面（面0）の neighbour はセル1
    neighbour_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      neighbour;
}
1
(
1
)
"""

    boundary_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}
3
(
inlet
{
    type            patch;
    nFaces          1;
    startFace       1;
}
outlet
{
    type            patch;
    nFaces          1;
    startFace       2;
}
walls
{
    type            wall;
    nFaces          8;
    startFace       3;
}
)
"""

    (mesh_dir / "points").write_text(points_text)
    (mesh_dir / "faces").write_text(faces_text)
    (mesh_dir / "owner").write_text(owner_text)
    (mesh_dir / "neighbour").write_text(neighbour_text)
    (mesh_dir / "boundary").write_text(boundary_text)


@binds_to(PolyMeshReaderProcess)
class TestPolyMeshReaderAPI:
    """PolyMeshReaderProcess の API テスト."""

    def test_meta_exists(self):
        """ProcessMeta が定義されていること."""
        assert PolyMeshReaderProcess.meta.name == "PolyMeshReader"
        assert PolyMeshReaderProcess.meta.module == "pre"

    def test_process_returns_result(self):
        """process() が PolyMeshResult を返すこと."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert isinstance(result, PolyMeshResult)

    def test_mesh_data_fields(self):
        """MeshData の必須フィールドが設定されていること."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            mesh = result.mesh
            assert mesh.node_coords is not None
            assert mesh.connectivity is not None
            assert mesh.cell_volumes is not None
            assert mesh.face_areas is not None
            assert mesh.face_normals is not None
            assert mesh.face_centers is not None
            assert mesh.cell_centers is not None
            assert mesh.face_owner is not None
            assert mesh.face_neighbour is not None


class TestPolyMeshReaderPhysics:
    """PolyMeshReaderProcess の物理テスト."""

    def test_cell_count(self):
        """セル数が正しいこと（2セル）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert result.mesh.n_cells == 2

    def test_node_count(self):
        """ノード数が正しいこと（12ノード）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert result.mesh.n_nodes == 12

    def test_cell_volumes(self):
        """セル体積が正しいこと（各セル 0.5）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            np.testing.assert_allclose(sorted(result.mesh.cell_volumes), [0.5, 0.5], rtol=0.1)

    def test_total_volume(self):
        """総体積がドメイン体積と一致（1.0）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            np.testing.assert_allclose(sum(result.mesh.cell_volumes), 1.0, rtol=0.1)

    def test_boundary_patches(self):
        """境界パッチが正しく読み込まれること."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert "inlet" in result.boundary_patches
            assert "outlet" in result.boundary_patches
            assert "walls" in result.boundary_patches
            assert result.boundary_patches["inlet"]["type"] == "patch"
            assert result.boundary_patches["walls"]["type"] == "wall"

    def test_internal_face_count(self):
        """内部面数が正しいこと（1面）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert len(result.mesh.face_neighbour) == 1

    def test_face_normals_unit(self):
        """面法線が単位ベクトルであること."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_test_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            norms = np.linalg.norm(result.mesh.face_normals, axis=1)
            # 面積ゼロでない面は単位ベクトル
            nonzero = norms > 1e-12
            np.testing.assert_allclose(norms[nonzero], 1.0, rtol=1e-10)


class TestParserFunctions:
    """個別パーサ関数のテスト."""

    def test_parse_points(self):
        text = """\
FoamFile
{
    version     2.0;
}
3
(
(0 0 0)
(1 0 0)
(0 1 0)
)
"""
        pts = parse_points(text)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[1], [1, 0, 0])

    def test_parse_faces(self):
        text = """\
FoamFile
{
    version     2.0;
}
2
(
4(0 1 2 3)
3(0 1 4)
)
"""
        faces = parse_faces(text)
        assert len(faces) == 2
        assert faces[0] == [0, 1, 2, 3]
        assert faces[1] == [0, 1, 4]

    def test_parse_label_list(self):
        text = """\
FoamFile
{
    version     2.0;
}
4
(
0
1
0
1
)
"""
        labels = parse_label_list(text)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])

    def test_parse_boundary(self):
        text = """\
FoamFile
{
    version     2.0;
}
2
(
inlet
{
    type            patch;
    nFaces          10;
    startFace       100;
}
walls
{
    type            wall;
    nFaces          40;
    startFace       110;
}
)
"""
        patches = parse_boundary(text)
        assert "inlet" in patches
        assert patches["inlet"]["type"] == "patch"
        assert patches["inlet"]["nFaces"] == 10
        assert patches["walls"]["type"] == "wall"


# ---------------------------------------------------------------------------
# バイナリ形式テスト
# ---------------------------------------------------------------------------


def _make_binary_header(class_name: str, object_name: str) -> bytes:
    """OpenFOAM バイナリ形式のヘッダを生成."""
    return (
        f"FoamFile\n"
        f"{{\n"
        f"    version     2.0;\n"
        f"    format      binary;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        f"}}\n"
    ).encode("ascii")


def _create_binary_polymesh(mesh_dir: Path) -> None:
    """2x1x1 セルのバイナリ polyMesh テストデータを作成."""
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --- points (12 ノード) ---
    pts = np.array(
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0.5, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [0.5, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    # OpenFOAM は '(' の直後にバイナリデータ
    points_raw = _make_binary_header("vectorField", "points") + b"12\n("
    points_raw += pts.tobytes()
    points_raw += b")\n"
    (mesh_dir / "points").write_bytes(points_raw)

    # --- owner (11面) ---
    owner_arr = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    owner_raw = _make_binary_header("labelList", "owner") + b"11\n("
    owner_raw += owner_arr.tobytes()
    owner_raw += b")\n"
    (mesh_dir / "owner").write_bytes(owner_raw)

    # --- neighbour (1内部面) ---
    neigh_arr = np.array([1], dtype=np.int32)
    neigh_raw = _make_binary_header("labelList", "neighbour") + b"1\n("
    neigh_raw += neigh_arr.tobytes()
    neigh_raw += b")\n"
    (mesh_dir / "neighbour").write_bytes(neigh_raw)

    # --- faces (11 面, compactListList) ---
    # 各面4ノード → offsets = [0, 4, 8, 12, ..., 44]
    offsets = np.arange(12, dtype=np.int32) * 4
    face_nodes_list = [
        [1, 4, 10, 7],  # 内部面
        [0, 6, 9, 3],
        [2, 5, 11, 8],
        [0, 1, 7, 6],
        [1, 2, 8, 7],
        [3, 9, 10, 4],
        [4, 10, 11, 5],
        [0, 3, 4, 1],
        [1, 4, 5, 2],
        [6, 7, 10, 9],
        [7, 8, 11, 10],
    ]
    labels = np.array([n for face in face_nodes_list for n in face], dtype=np.int32)
    faces_raw = _make_binary_header("faceCompactList", "faces") + b"11\n("
    faces_raw += offsets.tobytes()
    faces_raw += labels.tobytes()
    faces_raw += b")\n"
    (mesh_dir / "faces").write_bytes(faces_raw)

    # --- boundary (ASCII, 常にASCII) ---
    boundary_text = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}
3
(
inlet
{
    type            patch;
    nFaces          1;
    startFace       1;
}
outlet
{
    type            patch;
    nFaces          1;
    startFace       2;
}
walls
{
    type            wall;
    nFaces          8;
    startFace       3;
}
)
"""
    (mesh_dir / "boundary").write_text(boundary_text)


class TestBinaryFormat:
    """バイナリ形式の検出・パーステスト."""

    def test_is_binary_format_detection(self):
        """バイナリ形式判定が正しく動作."""
        binary_header = _make_binary_header("vectorField", "points")
        assert _is_binary_format(binary_header) is True

        ascii_header = b"FoamFile\n{\n    format      ascii;\n}\n"
        assert _is_binary_format(ascii_header) is False

    def test_parse_points_binary(self):
        """バイナリ points の解析."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        raw = _make_binary_header("vectorField", "points") + b"2\n("
        raw += pts.tobytes()
        raw += b")\n"
        result = parse_points_binary(raw)
        np.testing.assert_allclose(result, pts)

    def test_parse_label_list_binary(self):
        """バイナリ label リストの解析."""
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        raw = _make_binary_header("labelList", "owner") + b"4\n("
        raw += labels.tobytes()
        raw += b")\n"
        result = parse_label_list_binary(raw)
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_parse_faces_binary_compact(self):
        """バイナリ faces (compactListList) の解析."""
        faces_list = [[0, 1, 2, 3], [4, 5, 6]]
        offsets = np.array([0, 4, 7], dtype=np.int32)
        labels = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
        raw = _make_binary_header("faceCompactList", "faces") + b"2\n("
        raw += offsets.tobytes()
        raw += labels.tobytes()
        raw += b")\n"
        result = parse_faces_binary(raw)
        assert result == faces_list


class TestBinaryPolyMeshReader:
    """バイナリ形式 polyMesh の統合テスト."""

    def test_binary_cell_count(self):
        """バイナリ: セル数が正しいこと."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_binary_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            assert result.mesh.n_cells == 2

    def test_binary_total_volume(self):
        """バイナリ: 総体積がドメイン体積と一致（1.0）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_dir = Path(tmpdir) / "polyMesh"
            _create_binary_polymesh(mesh_dir)
            result = PolyMeshReaderProcess().process(PolyMeshInput(str(mesh_dir)))
            np.testing.assert_allclose(sum(result.mesh.cell_volumes), 1.0, rtol=0.1)

    def test_binary_ascii_consistency(self):
        """バイナリとASCIIの読み込み結果が一致."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ascii_dir = Path(tmpdir) / "ascii"
            binary_dir = Path(tmpdir) / "binary"
            _create_test_polymesh(ascii_dir)
            _create_binary_polymesh(binary_dir)
            r_ascii = PolyMeshReaderProcess().process(PolyMeshInput(str(ascii_dir)))
            r_bin = PolyMeshReaderProcess().process(PolyMeshInput(str(binary_dir)))
            assert r_ascii.mesh.n_cells == r_bin.mesh.n_cells
            np.testing.assert_allclose(r_ascii.mesh.node_coords, r_bin.mesh.node_coords, atol=1e-10)
            np.testing.assert_allclose(
                sum(r_ascii.mesh.cell_volumes),
                sum(r_bin.mesh.cell_volumes),
                rtol=0.1,
            )
