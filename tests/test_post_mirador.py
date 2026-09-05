"""3D レンダリング（messi mirador 連携）PostProcess のテスト.

messi 未導入の環境では本モジュールのテストは skip する（``binds_to`` の紐付けは
skip でも評価されるので、契約 C3 は満たす）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.post.mirador import (
    MiradorExportInput,
    MiradorExportProcess,
    SlicePlane,
    build_structured_hex_mesh,
    fields_from_heat_transfer,
    fields_from_natural_convection,
    lines_from_structured_mesh,
    load_npz_fields,
    resolve_slices,
)

try:
    import messi  # noqa: F401

    _HAS_MESSI = True
except ImportError:  # pragma: no cover - 環境依存
    _HAS_MESSI = False

needs_messi = pytest.mark.skipif(not _HAS_MESSI, reason="messi 未導入")


def _embedded(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    return json.loads(html.split("const DATA = ")[1].split(";\n")[0])


def _lines(nx: int, ny: int, nz: int, L: float = 1.0):
    return np.linspace(0, L, nx + 1), np.linspace(0, L, ny + 1), np.linspace(0, L, nz + 1)


@binds_to(MiradorExportProcess)
class TestMiradorExportAPI:
    # ---- messi 非依存: 六面体メッシュ構築と断面解決 ----
    def test_hex_mesh_labels_and_connectivity(self):
        x, y, z = _lines(2, 1, 1)
        hm = build_structured_hex_mesh(x, y, z)
        assert hm.nodes.shape == (12, 4) and hm.elements.shape == (2, 9)
        # i が最速、節点順は C3D8（下面 4 → 上面 4、反時計回り）
        assert hm.elements[1].tolist() == [2, 2, 3, 6, 5, 8, 9, 12, 11]
        assert hm.cell_labels.shape == (2, 1, 1) and hm.cell_labels[1, 0, 0] == 2
        assert np.allclose(hm.nodes[11, 1:], [1.0, 1.0, 1.0])
        assert list(hm.elsets) == ["domain"] and hm.elsets["domain"].tolist() == [1, 2]

    def test_mask_and_slices_partition_cells(self):
        x, y, z = _lines(3, 3, 3)
        mask = np.ones((3, 3, 3), dtype=bool)
        mask[0, 0, 0] = False
        sl = resolve_slices((SlicePlane("x"), SlicePlane("y", position=0.5)), (x, y, z), True)
        assert [(n, a, i) for n, a, i in sl] == [("x=0.5", 0, 1), ("y=0.5", 1, 1)]
        hm = build_structured_hex_mesh(x, y, z, mask=mask, slices=sl)
        assert hm.elements.shape[0] == 26 and hm.cell_labels[0, 0, 0] == 0
        sizes = {k: len(v) for k, v in hm.elsets.items()}
        # x スラブ 9、y スラブは x スラブと重なる 3 セルを除いて 6、残りが domain
        assert sizes == {"domain": 11, "x=0.5": 9, "y=0.5": 6}
        assert set(hm.elsets["domain"]) | set(hm.elsets["x=0.5"]) | set(hm.elsets["y=0.5"]) == set(
            hm.elements[:, 0]
        )

    def test_auto_slices_only_for_axes_with_3_or_more_cells(self):
        x, y, z = _lines(4, 4, 2)
        names = [n for n, _, _ in resolve_slices((), (x, y, z), True)]
        assert names == ["x=0.625", "y=0.625"]
        assert resolve_slices((), (x, y, z), False) == []
        # index 指定（負も可）と名前指定
        sl = resolve_slices((SlicePlane("z", index=-1, name="top"),), (x, y, z), True)
        assert sl == [("top", 2, 1)]

    def test_slice_errors(self):
        x, y, z = _lines(2, 2, 2)
        with pytest.raises(ValueError, match="axis"):
            SlicePlane("w")
        with pytest.raises(ValueError, match="範囲外"):
            resolve_slices((SlicePlane("x", index=5),), (x, y, z), False)
        with pytest.raises(ValueError, match="格子範囲"):
            resolve_slices((SlicePlane("y", position=2.0),), (x, y, z), False)
        with pytest.raises(ValueError, match="重複"):
            resolve_slices((SlicePlane("x"), SlicePlane("x")), (x, y, z), False)

    def test_adapters(self, tmp_path: Path):
        from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess

        mesh = StructuredMeshProcess().execute(
            StructuredMeshInput(Lx=1.0, Ly=0.5, Lz=0.25, nx=2, ny=1, nz=1, origin=(1.0, 2.0, 3.0))
        )
        x, y, z = lines_from_structured_mesh(mesh)
        assert np.allclose(x, [1.0, 1.5, 2.0]) and np.allclose(y, [2.0, 2.5])
        assert np.allclose(z, [3.0, 3.25])

        class _NC:
            u = np.ones((2, 1, 1))
            v = np.zeros((2, 1, 1))
            w = np.zeros((2, 1, 1))
            p = np.zeros((2, 1, 1))
            T = np.full((2, 1, 1), 300.0)
            extra_scalars = {"C": np.ones((2, 1, 1))}

        f = fields_from_natural_convection(_NC())
        assert f["U"].shape == (2, 1, 1, 3) and set(f) == {"U", "P", "T", "C"}

        class _HT:
            T = np.zeros((2, 1, 1))

        assert list(fields_from_heat_transfer(_HT())) == ["T"]

        npz = tmp_path / "r.npz"
        np.savez(npz, x_lines=x, y_lines=y, z_lines=z, T=f["T"], U=f["U"])
        xx, _yy, _zz, fields = load_npz_fields(npz)
        assert np.allclose(xx, x) and set(fields) == {"T", "U"}
        np.savez(tmp_path / "bad.npz", T=f["T"])
        with pytest.raises(ValueError, match="x_lines"):
            load_npz_fields(tmp_path / "bad.npz")

    # ---- messi 依存: HTML 出力と埋め込みデータの整合 ----
    @needs_messi
    def test_export_html_embeds_cell_values(self, tmp_path: Path):
        x, y, z = _lines(3, 2, 1)
        nx, ny, nz = 3, 2, 1
        T = np.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz, order="F") + 300.0
        U = np.zeros((nx, ny, nz, 3))
        U[..., 0] = 1.0
        U[2, 1, 0] = [0.0, 0.0, 2.0]
        out = tmp_path / "v.html"
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"T": T, "U": U}, str(out), title="t", auto_slices=False)
        )
        assert res.path == str(out) and out.exists()
        assert res.n_cells == 6 and res.n_nodes == 24 and res.slice_names == ()
        # 外皮: 2(nx ny + ny nz + nz nx) = 2(6 + 2 + 3) = 22 面 = 44 三角形
        assert res.n_triangles == 44
        assert res.field_names == ("T", "|U|", "U_x", "U_y", "U_z")
        assert res.n_vectors == 6 and res.init_mode == "T"
        d = _embedded(out)
        assert d["nTriangles"] == 44 and d["groupNames"] == ["domain"]
        assert d["hiddenGroups"] == []  # 断面が無いので外皮は表示
        # 三角形 → owner セルの T がセル値と一致（ラベルは i 最速: label = 1 + i + nx*j）
        for t, owner in enumerate(d["triElement"]):
            i, j = (owner - 1) % nx, (owner - 1) // nx
            assert d["metrics"]["T"][t] == T[i, j, 0]
        assert d["ranges"]["T"] == [300.0, 305.0]
        assert d["vectors"]["maxMag"] == 2.0 and len(d["vectors"]["origins"]) == 18
        # セル 1 の中心は (1/6, 1/4, 1/2)
        assert np.allclose(d["vectors"]["origins"][:3], [1 / 6, 0.25, 0.5])
        assert d["panelCollapsed"] is False  # 既定は操作パネルを開いた状態

    @needs_messi
    def test_panel_collapsed_is_passed_to_messi(self, tmp_path: Path):
        x, y, z = _lines(2, 2, 1)
        T = np.zeros((2, 2, 1))
        out = tmp_path / "pc.html"
        MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"T": T}, str(out), auto_slices=False, panel_collapsed=True)
        )
        assert _embedded(out)["panelCollapsed"] is True

    @needs_messi
    def test_cut_plane_is_passed_to_messi(self, tmp_path: Path):
        x, y, z = _lines(4, 4, 4)
        T = np.zeros((4, 4, 4))
        out = tmp_path / "cut.html"
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"T": T}, str(out), cut_plane=((0.0, 0.0, 2.0), 0.5))
        )
        d = _embedded(out)
        assert d["cut"] == {"normal": [0.0, 0.0, 1.0], "d": 0.5}
        assert res.n_section_cells == 64 and d["nCells"] == 64  # 全セルが切り口用に載る
        assert res.slice_names  # 自動スラブは残るが…
        assert d["hiddenGroups"] == []  # …外皮は隠さない（切った立体として見せる）
        assert d["cells"]["offset"][-1] == 64 * 8

    def test_cut_plane_zero_normal_rejected(self, tmp_path: Path):
        x, y, z = _lines(2, 2, 2)
        T = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="cut_plane"):
            MiradorExportProcess().execute(
                MiradorExportInput(
                    x, y, z, {"T": T}, str(tmp_path / "z.html"), cut_plane=((0.0, 0.0, 0.0), 0.5)
                )
            )

    @needs_messi
    def test_slices_hide_domain_and_add_interface_faces(self, tmp_path: Path):
        x, y, z = _lines(4, 4, 4)
        T = np.zeros((4, 4, 4))
        out = tmp_path / "s.html"
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"T": T}, str(out), slices=(SlicePlane("z"),))
        )
        assert res.slice_names == ("z=0.625",)
        d = _embedded(out)
        assert d["groupNames"] == ["domain", "z=0.625"]
        assert d["hiddenGroups"] == ["domain"]
        # 外皮 96 面 = 192 三角形 + スラブと domain の共有面（上下 16 面 × 両向き × 2 三角形 = 128）
        assert res.n_triangles == 192 + 128
        assert res.n_vectors == 0 and res.init_mode == "T"
        res2 = MiradorExportProcess().execute(
            MiradorExportInput(
                x, y, z, {"T": T}, str(out), slices=(SlicePlane("z"),), hide_domain=False
            )
        )
        assert _embedded(Path(res2.path))["hiddenGroups"] == []

    @needs_messi
    def test_2d_fields_are_extruded_and_mask_applies(self, tmp_path: Path):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        z = np.array([0.0])
        P = np.arange(4.0).reshape(2, 2)
        U2 = np.ones((2, 2, 2))
        mask = np.array([[True, True], [True, False]])
        out = tmp_path / "p.html"
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"P": P, "U": U2}, str(out), mask=mask, auto_slices=False)
        )
        assert res.n_cells == 3 and res.n_vectors == 3
        assert res.field_names == ("P", "|U|", "U_x", "U_y", "U_z")
        d = _embedded(out)
        assert d["ranges"]["U_z"] == [0.0, 0.0] and d["ranges"]["P"] == [0.0, 2.0]
        zs = np.array(d["positions"]).reshape(-1, 3)[:, 2]
        assert zs.min() == 0.0 and abs(zs.max() - 0.5) < 1e-12  # 面内セル幅で 1 層押し出し

    @needs_messi
    def test_init_mode_and_vector_options(self, tmp_path: Path):
        x, y, z = _lines(2, 2, 2)
        U = np.ones((2, 2, 2, 3))
        P = np.zeros((2, 2, 2))
        out = tmp_path / "o.html"
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"U": U, "P": P}, str(out), auto_slices=False)
        )
        assert res.init_mode == "P"  # T が無ければ P
        res = MiradorExportProcess().execute(
            MiradorExportInput(x, y, z, {"U": U}, str(out), auto_slices=False)
        )
        assert res.init_mode == "|U|" and res.n_vectors == 8
        res = MiradorExportProcess().execute(
            MiradorExportInput(
                x,
                y,
                z,
                {"U": U, "P": P},
                str(out),
                vector_field="",
                init_mode="U_x",
                vector_scale=0.1,
                auto_slices=False,
            )
        )
        assert res.n_vectors == 0 and res.init_mode == "U_x"
        with pytest.raises(ValueError, match="ベクトル場ではありません"):
            MiradorExportProcess().execute(
                MiradorExportInput(x, y, z, {"U": U, "P": P}, str(out), vector_field="P")
            )

    def test_input_validation(self, tmp_path: Path):
        x, y, z = _lines(2, 2, 2)
        out = str(tmp_path / "e.html")
        with pytest.raises(ValueError, match="一致しません"):
            MiradorExportProcess().execute(
                MiradorExportInput(x, y, z, {"T": np.zeros((3, 2, 2))}, out)
            )
        with pytest.raises(ValueError, match="空"):
            MiradorExportProcess().execute(MiradorExportInput(x, y, z, {}, out))
        with pytest.raises(ValueError, match="昇順"):
            MiradorExportProcess().execute(
                MiradorExportInput(x[::-1], y, z, {"T": np.zeros((2, 2, 2))}, out)
            )
        with pytest.raises(ValueError, match="mask"):
            MiradorExportProcess().execute(
                MiradorExportInput(
                    x, y, z, {"T": np.zeros((2, 2, 2))}, out, mask=np.ones((2, 2), dtype=bool)
                )
            )
