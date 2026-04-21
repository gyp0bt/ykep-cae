"""水槽用外部フィルター Process (Phase 6.3a).

エーハイム相当の外部フィルターのリターンパイプ（水槽へ吐出）/
インテークパイプ（水槽から吸入）を水槽内部セルに配置し、
`NaturalConvectionInput.internal_face_bcs` に流し込める
`InternalFaceBC` のペアを返却する PreProcess。

用語の混乱を避けるため、水槽（CFD領域）視点の `inflow`（吐出 = 水槽へ流入）/
`outflow`（吸入 = 水槽から流出）で統一する。CFD 的には:

- `inflow_geometry` → InternalFaceBC(INLET) — 領域への速度流入
- `outflow_geometry` → InternalFaceBC(OUTLET) — 領域からの圧力基準点

Process は純粋関数として設計し、格子座標（`AquariumGeometryResult`）と
バウンディングボックス + 流量 Q [L/h] + 吐出方向ベクトルを受け取って、
流量を満たす速度ベクトルを自動計算する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.natural_convection.data import InternalFaceBC, InternalFaceBCKind


@dataclass(frozen=True)
class NozzleGeometry:
    """ノズル（吐出/吸入）のバウンディングボックス.

    Parameters
    ----------
    x_range, y_range, z_range : tuple[float, float]
        各方向の (min, max) 座標 [m]。
        この範囲に中心が含まれるセルがノズル占有セル。
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]


@dataclass(frozen=True)
class AquariumFilterInput:
    """外部フィルター Process 入力.

    `AquariumGeometryResult` の座標/セル幅配列と、水槽への吐出/吸入ノズルの
    バウンディングボックス + 流量 Q [L/h] + 吐出方向を受け取る。

    Parameters
    ----------
    x_centers, y_centers, z_centers : np.ndarray
        セル中心座標 [m]
    dx, dy, dz : np.ndarray
        セル幅 [m]
    inflow_geometry : NozzleGeometry
        水槽への吐出ノズル形状（フィルターのリターン側）
    outflow_geometry : NozzleGeometry
        水槽からの吸入ノズル形状（フィルターのインテーク側）
    flow_rate_lph : float
        流量 Q [L/h]。エーハイム 2213 相当で 440 L/h, 2217 で 1000 L/h。
    inflow_direction : tuple[float, float, float]
        吐出方向ベクトル（単位ベクトルに正規化される）。
        例: `(1, 0, 0)` は +x、`(0, 0, 1)` は +z。
    inflow_temperature_K : float | None
        吐出水温 [K]。ヒーター・フィルター通過後の水温を指定する場合に設定。
        None なら温度拘束なし（INLET ではあるがエネルギー方程式は自由）。
    label : str
        識別子（ログ・デバッグ用）
    """

    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    inflow_geometry: NozzleGeometry
    outflow_geometry: NozzleGeometry
    flow_rate_lph: float = 440.0
    inflow_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    inflow_temperature_K: float | None = None
    label: str = "filter"


@dataclass(frozen=True)
class AquariumFilterResult:
    """外部フィルター Process 出力.

    Parameters
    ----------
    inflow_bc : InternalFaceBC
        水槽への吐出 BC（INLET 種別）。NaturalConvectionInput.internal_face_bcs に渡す。
    outflow_bc : InternalFaceBC
        水槽からの吸入 BC（OUTLET 種別）。NaturalConvectionInput.internal_face_bcs に渡す。
    inflow_mask : np.ndarray
        吐出ノズル占有セル (nx, ny, nz) bool
    outflow_mask : np.ndarray
        吸入ノズル占有セル (nx, ny, nz) bool
    inflow_velocity : tuple[float, float, float]
        計算された吐出速度ベクトル [m/s]
    flow_rate_m3s : float
        流量 [m³/s]（L/h から換算）
    inflow_area_m2 : float
        吐出ノズル投影面積（吐出方向に直交する面積） [m²]
    """

    inflow_bc: InternalFaceBC
    outflow_bc: InternalFaceBC
    inflow_mask: np.ndarray
    outflow_mask: np.ndarray
    inflow_velocity: tuple[float, float, float]
    flow_rate_m3s: float
    inflow_area_m2: float


def _build_nozzle_mask(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    geom: NozzleGeometry,
) -> np.ndarray:
    """バウンディングボックス内セルを True にするマスクを作成."""
    xm = (x_centers >= geom.x_range[0]) & (x_centers <= geom.x_range[1])
    ym = (y_centers >= geom.y_range[0]) & (y_centers <= geom.y_range[1])
    zm = (z_centers >= geom.z_range[0]) & (z_centers <= geom.z_range[1])
    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[np.ix_(xm, ym, zm)] = True
    return mask


def _projected_area(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    mask: np.ndarray,
    direction: tuple[float, float, float],
) -> float:
    """吐出方向に直交する投影面積を計算.

    各セルの面積 `dy[j]*dz[k]|dir_x| + dz[k]*dx[i]|dir_y| + dx[i]*dy[j]|dir_z|`
    を `mask=True` のセルで合計する。軸に沿った吐出では素朴な直交面積になる。
    斜め方向でも投影面積として妥当な値を返す（断面ではないので厳密には
    `|dot(normal, dir)| * face_area` の和）。
    """
    area_x = np.einsum("j,k->jk", dy, dz) * abs(direction[0])
    ax = np.broadcast_to(area_x[np.newaxis, :, :], mask.shape)
    area_y = np.einsum("i,k->ik", dx, dz) * abs(direction[1])
    ay = np.broadcast_to(area_y[:, np.newaxis, :], mask.shape)
    area_z = np.einsum("i,j->ij", dx, dy) * abs(direction[2])
    az = np.broadcast_to(area_z[:, :, np.newaxis], mask.shape)
    per_cell_area = ax + ay + az
    return float(per_cell_area[mask].sum())


class AquariumFilterProcess(
    PreProcess["AquariumFilterInput", "AquariumFilterResult"],
):
    """水槽用外部フィルター `InternalFaceBC` 生成 Process.

    バウンディングボックス指定の吐出/吸入ノズルを `InternalFaceBC`
    (INLET = 水槽への吐出、OUTLET = 水槽からの吸入) に変換する。
    吐出速度は `Q / A_projected` で自動計算され、吐出方向ベクトルを
    掛けた速度ベクトルが INLET の velocity となる。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="AquariumFilterProcess",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/aquarium-filter.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: AquariumFilterInput) -> AquariumFilterResult:
        inp = input_data

        if inp.flow_rate_lph <= 0:
            msg = f"flow_rate_lph は正の値: {inp.flow_rate_lph}"
            raise ValueError(msg)
        for name, rng in (
            ("inflow_geometry.x_range", inp.inflow_geometry.x_range),
            ("inflow_geometry.y_range", inp.inflow_geometry.y_range),
            ("inflow_geometry.z_range", inp.inflow_geometry.z_range),
            ("outflow_geometry.x_range", inp.outflow_geometry.x_range),
            ("outflow_geometry.y_range", inp.outflow_geometry.y_range),
            ("outflow_geometry.z_range", inp.outflow_geometry.z_range),
        ):
            if rng[0] >= rng[1]:
                msg = f"{name} は min<max が必要: {rng}"
                raise ValueError(msg)
        dir_vec = np.asarray(inp.inflow_direction, dtype=np.float64)
        dir_norm = float(np.linalg.norm(dir_vec))
        if dir_norm < 1e-12:
            msg = f"inflow_direction はゼロベクトル不可: {inp.inflow_direction}"
            raise ValueError(msg)
        dir_unit = dir_vec / dir_norm

        inflow_mask = _build_nozzle_mask(
            inp.x_centers, inp.y_centers, inp.z_centers, inp.inflow_geometry
        )
        outflow_mask = _build_nozzle_mask(
            inp.x_centers, inp.y_centers, inp.z_centers, inp.outflow_geometry
        )
        if not np.any(inflow_mask):
            msg = (
                f"inflow_geometry={inp.inflow_geometry} にセルが含まれない。"
                "バウンディングボックスが格子範囲外か、領域が細すぎる可能性。"
            )
            raise ValueError(msg)
        if not np.any(outflow_mask):
            msg = (
                f"outflow_geometry={inp.outflow_geometry} にセルが含まれない。"
                "バウンディングボックスが格子範囲外か、領域が細すぎる可能性。"
            )
            raise ValueError(msg)
        if np.any(inflow_mask & outflow_mask):
            msg = (
                "inflow_geometry と outflow_geometry のセルが重なっています。"
                "吐出ノズルと吸入ノズルは別領域に配置してください。"
            )
            raise ValueError(msg)

        # 流量 [m³/s]
        flow_rate_m3s = inp.flow_rate_lph * 1e-3 / 3600.0  # L/h → m³/s

        # 投影面積 [m²]（吐出方向に直交する断面）
        inflow_area_m2 = _projected_area(inp.dx, inp.dy, inp.dz, inflow_mask, tuple(dir_unit))
        if inflow_area_m2 <= 0:
            msg = (
                f"inflow ノズルの投影面積がゼロ: direction={inp.inflow_direction}。"
                "吐出方向が軸と直交するなど面積計算が成立しない可能性。"
            )
            raise ValueError(msg)

        speed = flow_rate_m3s / inflow_area_m2
        velocity = (
            float(speed * dir_unit[0]),
            float(speed * dir_unit[1]),
            float(speed * dir_unit[2]),
        )

        inflow_bc = InternalFaceBC(
            kind=InternalFaceBCKind.INLET,
            mask=inflow_mask,
            velocity=velocity,
            temperature=inp.inflow_temperature_K,
            label=f"{inp.label}:inflow",
        )
        outflow_bc = InternalFaceBC(
            kind=InternalFaceBCKind.OUTLET,
            mask=outflow_mask,
            label=f"{inp.label}:outflow",
        )

        return AquariumFilterResult(
            inflow_bc=inflow_bc,
            outflow_bc=outflow_bc,
            inflow_mask=inflow_mask,
            outflow_mask=outflow_mask,
            inflow_velocity=velocity,
            flow_rate_m3s=flow_rate_m3s,
            inflow_area_m2=inflow_area_m2,
        )
