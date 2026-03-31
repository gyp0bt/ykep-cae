"""TemperatureMapProcess テスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.heat_transfer import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferFDMProcess,
    HeatTransferInput,
    HeatTransferResult,
    TemperatureMapInput,
    TemperatureMapOutput,
    TemperatureMapProcess,
    setup_cjk_font,
)


def _make_simple_result() -> tuple[HeatTransferResult, float, float, float]:
    """テスト用の簡易解析結果を生成する."""
    nx, ny, nz = 10, 10, 5
    Lx, Ly, Lz = 1e-3, 1e-3, 0.5e-3

    k = np.full((nx, ny, nz), 50.0)
    C = np.full((nx, ny, nz), 3.9e6)
    q = np.zeros((nx, ny, nz))
    q[:5, :5, :] = 1e9

    inp = HeatTransferInput(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        k=k,
        C=C,
        q=q,
        T0=np.full((nx, ny, nz), 300.0),
        bc_xm=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_ym=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_zm=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_xp=BoundarySpec(BoundaryCondition.DIRICHLET, 300.0),
        bc_yp=BoundarySpec(BoundaryCondition.DIRICHLET, 300.0),
        bc_zp=BoundarySpec(BoundaryCondition.DIRICHLET, 300.0),
        max_iter=5000,
        tol=1e-7,
    )
    solver = HeatTransferFDMProcess(vectorized=True)
    result = solver.process(inp)
    return result, Lx, Ly, Lz


@binds_to(TemperatureMapProcess)
class TestTemperatureMapAPI:
    """TemperatureMapProcess の API テスト."""

    def test_meta_exists(self) -> None:
        assert TemperatureMapProcess.meta.name == "TemperatureMap"

    def test_process_returns_output(self) -> None:
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out = viz.process(TemperatureMapInput(result=result, Lx=Lx, Ly=Ly, Lz=Lz))
        assert isinstance(out, TemperatureMapOutput)
        assert out.T_min <= out.T_max

    def test_save_to_file(self, tmp_path: Path) -> None:
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out_path = tmp_path / "test.png"
        out = viz.process(
            TemperatureMapInput(
                result=result,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                output_path=out_path,
            )
        )
        assert out.saved_path is not None
        assert out.saved_path.exists()

    def test_slice_axes(self) -> None:
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        for axis in ("x", "y", "z"):
            out = viz.process(
                TemperatureMapInput(
                    result=result,
                    Lx=Lx,
                    Ly=Ly,
                    Lz=Lz,
                    slice_axis=axis,
                )
            )
            assert out.T_min <= out.T_max

    def test_invalid_axis_raises(self) -> None:
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        with pytest.raises(ValueError):
            viz.process(
                TemperatureMapInput(
                    result=result,
                    Lx=Lx,
                    Ly=Ly,
                    Lz=Lz,
                    slice_axis="w",
                )
            )

    def test_layer_boundaries(self, tmp_path: Path) -> None:
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out = viz.process(
            TemperatureMapInput(
                result=result,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                layer_boundaries=(0.25e-3,),
                layer_labels=("Layer 1", "Layer 2"),
                output_path=tmp_path / "layers.png",
            )
        )
        assert out.saved_path is not None
        assert out.saved_path.exists()

    def test_mirror_axes(self, tmp_path: Path) -> None:
        """ミラーリング表示が正しく動作すること."""
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out = viz.process(
            TemperatureMapInput(
                result=result,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                mirror_axes=("x", "y"),
                output_path=tmp_path / "mirrored.png",
            )
        )
        assert out.saved_path is not None
        assert out.saved_path.exists()

    def test_mirror_z_axis(self) -> None:
        """z方向ミラーリングで温度範囲が維持されること."""
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out = viz.process(
            TemperatureMapInput(
                result=result,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                mirror_axes=("z",),
                slice_axis="y",
            )
        )
        # ミラーしても温度範囲は変わらない
        assert out.T_min >= 299.0
        assert out.T_max <= 400.0

    def test_cjk_font_setup(self) -> None:
        """CJKフォント設定が正常に動作すること."""
        font_name = setup_cjk_font()
        # フォントが見つかった場合は文字列、見つからない場合は None
        assert font_name is None or isinstance(font_name, str)

    def test_cjk_font_in_process(self, tmp_path: Path) -> None:
        """use_cjk_font=True でプロセスが正常動作すること."""
        result, Lx, Ly, Lz = _make_simple_result()
        viz = TemperatureMapProcess()
        out = viz.process(
            TemperatureMapInput(
                result=result,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                use_cjk_font=True,
                title="温度分布 [K]",
                output_path=tmp_path / "cjk_test.png",
            )
        )
        assert out.saved_path is not None
        assert out.saved_path.exists()
