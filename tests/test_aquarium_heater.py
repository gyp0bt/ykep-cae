"""HeaterProcess テスト (Phase 6.2b).

API テスト（Process 契約）と物理テスト（熱量保存・ヒステリシス）を含む。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.aquarium import (
    AquariumGeometryInput,
    AquariumGeometryProcess,
    HeaterGeometry,
    HeaterInput,
    HeaterMode,
    HeaterProcess,
    HeaterResult,
)
from xkep_cae_fluid.core.testing import binds_to


def _make_heater_inputs(
    mode: HeaterMode = HeaterMode.CONSTANT_FLUX,
    power_watts: float = 200.0,
    setpoint_K: float = 300.0,
    hysteresis_band_K: float = 1.0,
    measured_T_K: float = 298.0,
    prev_on: bool = True,
    heater_range: tuple = ((0.75, 0.9), (0.1, 0.2), (0.1, 0.3)),
) -> HeaterInput:
    """共通入力ファクトリ."""
    geom = AquariumGeometryProcess().process(
        AquariumGeometryInput(Lx=0.9, Ly=0.3, Lz=0.45, nx=18, ny=6, nz=12, substrate_depth=0.0)
    )
    return HeaterInput(
        x_centers=geom.x_centers,
        y_centers=geom.y_centers,
        z_centers=geom.z_centers,
        dx=geom.dx,
        dy=geom.dy,
        dz=geom.dz,
        geometry=HeaterGeometry(
            x_range=heater_range[0],
            y_range=heater_range[1],
            z_range=heater_range[2],
        ),
        mode=mode,
        power_watts=power_watts,
        setpoint_K=setpoint_K,
        hysteresis_band_K=hysteresis_band_K,
        measured_T_K=measured_T_K,
        prev_on=prev_on,
    )


@binds_to(HeaterProcess)
class TestHeaterAPI:
    """HeaterProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        assert HeaterProcess.meta.name == "HeaterProcess"
        assert HeaterProcess.meta.module == "pre"

    def test_process_returns_heater_result(self):
        res = HeaterProcess().process(_make_heater_inputs())
        assert isinstance(res, HeaterResult)
        assert res.q_vol.ndim == 3
        assert res.mask.shape == res.q_vol.shape

    def test_invalid_power_raises(self):
        with pytest.raises(ValueError):
            HeaterProcess().process(_make_heater_inputs(power_watts=-10.0))

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError):
            HeaterProcess().process(_make_heater_inputs(hysteresis_band_K=-1.0))

    def test_invalid_range_raises(self):
        """min >= max の範囲指定で ValueError."""
        with pytest.raises(ValueError):
            HeaterProcess().process(
                _make_heater_inputs(heater_range=((0.9, 0.8), (0.1, 0.2), (0.1, 0.3)))
            )

    def test_empty_heater_region_raises(self):
        """格子範囲外の範囲指定で ValueError."""
        with pytest.raises(ValueError):
            HeaterProcess().process(
                _make_heater_inputs(heater_range=((-0.5, -0.4), (0.1, 0.2), (0.1, 0.3)))
            )


class TestHeaterPhysics:
    """物理的妥当性テスト."""

    def test_constant_flux_always_on(self):
        """CONSTANT_FLUX は常に ON."""
        res = HeaterProcess().process(
            _make_heater_inputs(mode=HeaterMode.CONSTANT_FLUX, measured_T_K=350.0)
        )
        assert res.on is True
        assert res.q_vol.sum() > 0

    def test_constant_flux_power_conserved(self):
        """∫q_vol dV ≈ power_watts."""
        power = 200.0
        inp = _make_heater_inputs(mode=HeaterMode.CONSTANT_FLUX, power_watts=power)
        res = HeaterProcess().process(inp)
        cell_vol = np.einsum("i,j,k->ijk", inp.dx, inp.dy, inp.dz)
        total_power = float((res.q_vol * cell_vol).sum())
        assert abs(total_power - power) / power < 1e-10

    def test_constant_flux_q_vol_zero_outside_mask(self):
        """マスク外のセルは q_vol=0."""
        inp = _make_heater_inputs(mode=HeaterMode.CONSTANT_FLUX)
        res = HeaterProcess().process(inp)
        assert (res.q_vol[~res.mask] == 0.0).all()

    def test_hysteresis_on_below_setpoint(self):
        """measured_T <= setpoint - band/2 で ON."""
        inp = _make_heater_inputs(
            mode=HeaterMode.CONSTANT_TEMPERATURE,
            setpoint_K=300.0,
            hysteresis_band_K=1.0,
            measured_T_K=299.0,  # 設定-0.5 以下
            prev_on=False,
        )
        res = HeaterProcess().process(inp)
        assert res.on is True

    def test_hysteresis_off_above_setpoint(self):
        """measured_T >= setpoint + band/2 で OFF（q_vol 全ゼロ）."""
        inp = _make_heater_inputs(
            mode=HeaterMode.CONSTANT_TEMPERATURE,
            setpoint_K=300.0,
            hysteresis_band_K=1.0,
            measured_T_K=301.0,  # 設定+0.5 以上
            prev_on=True,
        )
        res = HeaterProcess().process(inp)
        assert res.on is False
        assert (res.q_vol == 0.0).all()

    def test_hysteresis_midband_keeps_prev_state(self):
        """中間帯では prev_on を維持."""
        common = dict(
            mode=HeaterMode.CONSTANT_TEMPERATURE,
            setpoint_K=300.0,
            hysteresis_band_K=1.0,
            measured_T_K=300.1,  # 中間帯
        )
        res_on = HeaterProcess().process(_make_heater_inputs(**common, prev_on=True))
        res_off = HeaterProcess().process(_make_heater_inputs(**common, prev_on=False))
        assert res_on.on is True
        assert res_off.on is False
        # 状態に応じて q_vol も対応
        assert res_on.q_vol.sum() > 0
        assert res_off.q_vol.sum() == 0.0


class TestHeaterIntegration:
    """NaturalConvectionInput との互換性."""

    def test_q_vol_shape_matches_nc_input(self):
        """q_vol の shape が (nx, ny, nz) で NaturalConvectionInput.q_vol 互換."""
        inp = _make_heater_inputs()
        res = HeaterProcess().process(inp)
        assert res.q_vol.dtype == np.float64
        assert res.q_vol.shape == (len(inp.x_centers), len(inp.y_centers), len(inp.z_centers))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
