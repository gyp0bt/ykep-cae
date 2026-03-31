"""MultilayerBuilderProcess テスト.

API テスト（契約準拠）と物理テスト（物性値配列の正確性）を含む。
"""

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.heat_transfer.multilayer import (
    LayerSpec,
    MultilayerBuilderProcess,
    MultilayerInput,
    MultilayerOutput,
)

# ---------------------------------------------------------------------------
# API テスト
# ---------------------------------------------------------------------------


@binds_to(MultilayerBuilderProcess)
class TestMultilayerBuilderAPI:
    """MultilayerBuilderProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        """ProcessMeta が定義されていること."""
        assert MultilayerBuilderProcess.meta.name == "MultilayerBuilder"
        assert MultilayerBuilderProcess.meta.module == "pre"

    def test_process_returns_output(self):
        """process() が MultilayerOutput を返すこと."""
        inp = MultilayerInput(
            layers=(
                LayerSpec(thickness=0.001, k=10.0, C=1000.0, name="Layer1"),
                LayerSpec(thickness=0.001, k=20.0, C=2000.0, name="Layer2"),
            ),
            nx=5,
            ny=5,
            Lx=0.01,
            Ly=0.01,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)
        assert isinstance(result, MultilayerOutput)

    def test_empty_layers_raises(self):
        """空の層定義でエラーになること."""
        inp = MultilayerInput(
            layers=(),
            nx=5,
            ny=5,
            Lx=0.01,
            Ly=0.01,
        )
        builder = MultilayerBuilderProcess()
        with pytest.raises(ValueError, match="1層以上"):
            builder.process(inp)


# ---------------------------------------------------------------------------
# 物理テスト
# ---------------------------------------------------------------------------


class TestMultilayerBuilderPhysics:
    """多層ビルダーの物性値配列の正確性テスト."""

    def test_two_layer_conductivity(self):
        """2層構造で k が正しく配列に反映されること."""
        k1, k2 = 25.0, 50.0
        inp = MultilayerInput(
            layers=(
                LayerSpec(thickness=0.002, k=k1, C=1000.0, name="Ceramic"),
                LayerSpec(thickness=0.002, k=k2, C=2000.0, name="Steel"),
            ),
            nx=3,
            ny=3,
            Lx=0.01,
            Ly=0.01,
            nz_per_meter=1000.0,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)

        # 各層2セル（0.002m * 1000 cells/m = 2）
        assert result.nz == 4
        assert result.k.shape == (3, 3, 4)
        np.testing.assert_array_equal(result.k[:, :, :2], k1)
        np.testing.assert_array_equal(result.k[:, :, 2:], k2)

    def test_layer_boundaries(self):
        """層境界位置が正しいこと."""
        inp = MultilayerInput(
            layers=(
                LayerSpec(thickness=0.001, k=10.0, name="A"),
                LayerSpec(thickness=0.002, k=20.0, name="B"),
                LayerSpec(thickness=0.001, k=30.0, name="C"),
            ),
            nx=2,
            ny=2,
            Lx=0.01,
            Ly=0.01,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)

        assert len(result.layer_boundaries) == 2
        np.testing.assert_allclose(result.layer_boundaries[0], 0.001)
        np.testing.assert_allclose(result.layer_boundaries[1], 0.003)
        assert result.Lz == pytest.approx(0.004)
        assert result.layer_names == ("A", "B", "C")

    def test_heat_generation_per_layer(self):
        """層ごとの発熱量が正しく設定されること."""
        inp = MultilayerInput(
            layers=(
                LayerSpec(thickness=0.001, k=10.0, q=0.0, name="Insulator"),
                LayerSpec(thickness=0.001, k=50.0, q=1e9, name="Heater"),
            ),
            nx=4,
            ny=4,
            Lx=0.01,
            Ly=0.01,
            nz_per_meter=1000.0,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)

        # 下層: q=0, 上層: q=1e9
        np.testing.assert_array_equal(result.q[:, :, :1], 0.0)
        np.testing.assert_array_equal(result.q[:, :, 1:], 1e9)

    def test_default_initial_temperature(self):
        """デフォルト初期温度が全セルに設定されること."""
        inp = MultilayerInput(
            layers=(LayerSpec(thickness=0.001, k=10.0),),
            nx=3,
            ny=3,
            Lx=0.01,
            Ly=0.01,
            T0_default=350.0,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)

        np.testing.assert_array_equal(result.T0, 350.0)

    def test_default_layer_names(self):
        """名前未指定の場合に自動命名されること."""
        inp = MultilayerInput(
            layers=(
                LayerSpec(thickness=0.001, k=10.0),
                LayerSpec(thickness=0.001, k=20.0),
            ),
            nx=2,
            ny=2,
            Lx=0.01,
            Ly=0.01,
        )
        builder = MultilayerBuilderProcess()
        result = builder.process(inp)

        assert result.layer_names == ("Layer1", "Layer2")
