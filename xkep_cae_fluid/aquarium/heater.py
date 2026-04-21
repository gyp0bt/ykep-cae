"""水槽ヒーター体積熱源 Process (Phase 6.2b).

水槽用棒状ヒーター相当の体積熱源を計算する PreProcess。
`NaturalConvectionInput.q_vol` に直接渡せる 3 次元配列を返す。

CONSTANT_FLUX（定熱流束）と CONSTANT_TEMPERATURE（定温ヒステリシス制御）の
2 モードに対応。状態遷移は外部ループで `prev_on` を保持して渡す方針（Process を純粋関数として保つ）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess


class HeaterMode(Enum):
    """ヒーター運転モード."""

    CONSTANT_FLUX = "constant_flux"  # 定熱流束（常時 ON）
    CONSTANT_TEMPERATURE = "constant_temperature"  # 定温ヒステリシス制御


@dataclass(frozen=True)
class HeaterGeometry:
    """ヒーター形状（バウンディングボックス指定）.

    Parameters
    ----------
    x_range, y_range, z_range : tuple[float, float]
        各方向の (min, max) 座標 [m]。
        この範囲に中心が含まれるセルがヒーターセルとなる。
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]


@dataclass(frozen=True)
class HeaterInput:
    """ヒーター Process 入力.

    `AquariumGeometryResult` の座標/セル幅配列を受け取り、ヒーター形状から
    `q_vol` [W/m³] を組み立てる。

    Parameters
    ----------
    x_centers, y_centers, z_centers : np.ndarray
        各方向のセル中心座標 [m]
    dx, dy, dz : np.ndarray
        各方向のセル幅 [m]
    geometry : HeaterGeometry
        ヒーター形状
    mode : HeaterMode
        運転モード
    power_watts : float
        定格電力 [W]
    setpoint_K : float
        設定温度 [K]（CONSTANT_TEMPERATURE モードのみ）
    hysteresis_band_K : float
        ヒステリシス幅 [K]（CONSTANT_TEMPERATURE モードのみ）
    measured_T_K : float
        温度センサ位置の測定温度 [K]（CONSTANT_TEMPERATURE モードのみ）
    prev_on : bool
        直前ステップの ON/OFF 状態（中間帯判定用）
    """

    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    geometry: HeaterGeometry
    mode: HeaterMode = HeaterMode.CONSTANT_FLUX
    power_watts: float = 200.0
    setpoint_K: float = 299.15
    hysteresis_band_K: float = 1.0
    measured_T_K: float = 298.15
    prev_on: bool = True


@dataclass(frozen=True)
class HeaterResult:
    """ヒーター Process 出力.

    Parameters
    ----------
    q_vol : np.ndarray (nx, ny, nz)
        体積熱源 [W/m³]。OFF 時は全ゼロ。
    mask : np.ndarray (nx, ny, nz) bool
        ヒーター占有セル
    on : bool
        現在の ON/OFF 状態（次ステップに渡す）
    volume_m3 : float
        ヒーター領域の体積合計 [m³]
    """

    q_vol: np.ndarray
    mask: np.ndarray
    on: bool
    volume_m3: float


def _decide_on_state(
    mode: HeaterMode,
    measured_T: float,
    setpoint: float,
    band: float,
    prev_on: bool,
) -> bool:
    """ヒステリシス制御で ON/OFF 状態を決定."""
    if mode == HeaterMode.CONSTANT_FLUX:
        return True
    half = band / 2.0
    if measured_T <= setpoint - half:
        return True
    if measured_T >= setpoint + half:
        return False
    return prev_on


def _build_heater_mask(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    geometry: HeaterGeometry,
) -> np.ndarray:
    """バウンディングボックス内セルを True にするマスクを作成."""
    xm = (x_centers >= geometry.x_range[0]) & (x_centers <= geometry.x_range[1])
    ym = (y_centers >= geometry.y_range[0]) & (y_centers <= geometry.y_range[1])
    zm = (z_centers >= geometry.z_range[0]) & (z_centers <= geometry.z_range[1])

    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
    mask = np.zeros((nx, ny, nz), dtype=bool)
    # outer product of 1D masks
    mask[np.ix_(xm, ym, zm)] = True
    return mask


class HeaterProcess(PreProcess["HeaterInput", "HeaterResult"]):
    """水槽ヒーター体積熱源計算 Process.

    `AquariumGeometryResult` からの座標/セル幅配列と `HeaterGeometry` に基づき、
    `NaturalConvectionInput.q_vol` 形状の配列を構築する。定温モードでは
    ヒステリシス制御に従って ON/OFF を判定し、OFF 時は全ゼロを返す。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="HeaterProcess",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/aquarium-heater.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: HeaterInput) -> HeaterResult:
        """ヒーター体積熱源を計算."""
        inp = input_data

        if inp.power_watts < 0:
            msg = f"power_watts は非負: {inp.power_watts}"
            raise ValueError(msg)
        if inp.hysteresis_band_K < 0:
            msg = f"hysteresis_band_K は非負: {inp.hysteresis_band_K}"
            raise ValueError(msg)
        if inp.geometry.x_range[0] >= inp.geometry.x_range[1]:
            msg = f"x_range の min<max が必要: {inp.geometry.x_range}"
            raise ValueError(msg)
        if inp.geometry.y_range[0] >= inp.geometry.y_range[1]:
            msg = f"y_range の min<max が必要: {inp.geometry.y_range}"
            raise ValueError(msg)
        if inp.geometry.z_range[0] >= inp.geometry.z_range[1]:
            msg = f"z_range の min<max が必要: {inp.geometry.z_range}"
            raise ValueError(msg)

        mask = _build_heater_mask(inp.x_centers, inp.y_centers, inp.z_centers, inp.geometry)

        # セル体積（(nx, ny, nz) tensor product）
        dxyz = np.einsum("i,j,k->ijk", inp.dx, inp.dy, inp.dz)
        volume_m3 = float(dxyz[mask].sum())

        if volume_m3 <= 0.0:
            msg = (
                f"ヒーター領域にセルが含まれない。geometry={inp.geometry}。"
                "バウンディングボックスが格子範囲外か、領域が細すぎる可能性。"
            )
            raise ValueError(msg)

        on = _decide_on_state(
            inp.mode,
            inp.measured_T_K,
            inp.setpoint_K,
            inp.hysteresis_band_K,
            inp.prev_on,
        )

        q_vol = np.zeros(mask.shape, dtype=np.float64)
        if on:
            q_per_m3 = inp.power_watts / volume_m3
            q_vol[mask] = q_per_m3

        return HeaterResult(q_vol=q_vol, mask=mask, on=on, volume_m3=volume_m3)
