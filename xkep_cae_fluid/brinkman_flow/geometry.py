"""厚さ場ビルダー（平板 / U ターン流路）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.brinkman_flow.data import BrinkmanGeometry, ThicknessModel, ThicknessSpec
from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess


@dataclass(frozen=True)
class ThicknessInput:
    """UTurnThicknessProcess の入力."""

    nx: int
    ny: int
    spec: ThicknessSpec
    geometry: BrinkmanGeometry


@dataclass(frozen=True)
class ThicknessResult:
    """厚さ場 h (nx, ny) と流路マスク."""

    thickness: np.ndarray
    channel_mask: np.ndarray


class UTurnThicknessProcess(PreProcess["ThicknessInput", "ThicknessResult"]):
    """厚さ場を生成する PreProcess.

    - FLAT: 全域 h_channel
    - UTURN: inlet 帯（x 方向往路）→ 右端の折返し帯 → outlet 帯（復路）のみ h_channel、
      他は h_blocked。往路/復路の幅は channel_width、中心は inlet/outlet 中心に一致させる。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="UTurnThickness",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/brinkman-flow-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ThicknessInput) -> ThicknessResult:
        inp = input_data
        g = inp.geometry
        spec = inp.spec
        xc = (np.arange(inp.nx) + 0.5) * (g.lx / inp.nx)
        yc = (np.arange(inp.ny) + 0.5) * (g.ly / inp.ny)
        X, Y = np.meshgrid(xc, yc, indexing="ij")

        if spec.model is ThicknessModel.FLAT:
            mask = np.ones((inp.nx, inp.ny), dtype=bool)
        else:
            w = spec.channel_width
            y_in = 0.5 * (g.inlet_y0 + g.inlet_y1)
            y_out = 0.5 * (g.outlet_y0 + g.outlet_y1)
            turn_x0 = g.lx - w if spec.turn_x0 is None else spec.turn_x0
            leg_in = (Y > y_in - 0.5 * w) & (Y < y_in + 0.5 * w)
            leg_out = (Y > y_out - 0.5 * w) & (Y < y_out + 0.5 * w)
            turn = (X > turn_x0) & (Y > y_out - 0.5 * w) & (Y < y_in + 0.5 * w)
            mask = leg_in | leg_out | turn

        h = np.where(mask, spec.h_channel, spec.h_blocked)
        return ThicknessResult(thickness=h, channel_mask=mask)
