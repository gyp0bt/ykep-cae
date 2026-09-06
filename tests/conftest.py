"""pytest 設定."""

import pytest


def kuhn_tet_text(
    nx: int, ny: int, nz: int, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0
) -> str:
    """箱を Kuhn 分割（立方体 1 個 → 四面体 6 個）した C3D4 メッシュの .inp テキスト（ID は 1 始まり）."""
    import itertools

    def nid(i: int, j: int, k: int) -> int:
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    lines = ["*NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f" {nid(i, j, k)}, {i / nx * lx}, {j / ny * ly}, {k / nz * lz}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=TETS")
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for perm in itertools.permutations(range(3)):
                    p = [0, 0, 0]
                    nodes = [nid(i, j, k)]
                    for ax in perm:
                        p[ax] = 1
                        nodes.append(nid(i + p[0], j + p[1], k + p[2]))
                    e += 1
                    lines.append(f" {e}, " + ", ".join(map(str, nodes)))
    return "\n".join(lines) + "\n"


@pytest.fixture
def kuhn_tets():
    """``kuhn_tets(nx, ny, nz) -> MeshData``（四面体の面ベース非構造メッシュ）."""
    from xkep_cae_fluid.inp.builder import build_case
    from xkep_cae_fluid.inp.mesh import build_inp_mesh
    from xkep_cae_fluid.inp.parser import parse_inp_text

    def make(nx: int, ny: int, nz: int, **kw):
        return build_inp_mesh(build_case(parse_inp_text(kuhn_tet_text(nx, ny, nz, **kw)))).mesh

    return make
