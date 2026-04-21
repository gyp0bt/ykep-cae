"""水槽ジオメトリ生成 Process (Phase 6.2a).

90×30×45 cm (W×D×H) 水槽を想定し、構造化メッシュ上に底床マスク・ガラス壁マスク・
水領域マスクを構築する PreProcess。既存の StructuredMeshProcess を再利用する。

ロードマップ `docs/roadmap-aquarium.md` に基づき、x=幅、y=奥行き、z=高さ（鉛直方向）
の座標系を採用する。推奨重力ベクトルは (0, 0, -9.81) m/s²。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.core.mesh import (
    StructuredMeshInput,
    StructuredMeshProcess,
)


@dataclass(frozen=True)
class AquariumGeometryInput:
    """水槽ジオメトリ入力.

    座標系は x=幅 (W)、y=奥行き (D)、z=高さ (H)、鉛直方向 = z。
    デフォルトは 90×30×45 cm 水槽（Lx=0.9, Ly=0.3, Lz=0.45 m）。

    Parameters
    ----------
    Lx, Ly, Lz : float
        水槽内部領域サイズ [m]
    nx, ny, nz : int
        各方向のセル数
    substrate_depth : float
        底床（砂層）厚さ [m]。0 で底床なし。
    substrate_refinement_ratio : float
        z 方向ストレッチ比率（最大幅/最小幅）。1.0 で等間隔、
        >1.0 で底床（z 下端）側を細かく、水面（z 上端）側を粗くする。
    glass_thickness : float
        ガラス壁厚さ [m]。0 で厚みなし（純水領域のみ）。
    origin : tuple[float, float, float]
        原点座標 (x0, y0, z0)
    """

    Lx: float = 0.9
    Ly: float = 0.3
    Lz: float = 0.45
    nx: int = 36
    ny: int = 12
    nz: int = 30
    substrate_depth: float = 0.05
    substrate_refinement_ratio: float = 1.0
    glass_thickness: float = 0.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AquariumGeometryResult:
    """水槽ジオメトリ結果.

    NaturalConvectionInput に直接渡せるよう、`solid_mask` と推奨 `gravity` を返却する。

    Parameters
    ----------
    mesh : MeshData
        構造化メッシュデータ
    dx, dy, dz : np.ndarray
        各方向のセル幅配列
    x_centers, y_centers, z_centers : np.ndarray
        各方向のセル中心座標
    substrate_mask : np.ndarray (nx, ny, nz) bool
        底床（砂層）セル
    glass_mask : np.ndarray (nx, ny, nz) bool
        ガラス壁セル（`glass_thickness=0` で全 False）
    water_mask : np.ndarray (nx, ny, nz) bool
        水セル（非固体セル）
    solid_mask : np.ndarray (nx, ny, nz) bool
        `substrate_mask | glass_mask`。NaturalConvectionInput.solid_mask 互換。
    gravity : tuple[float, float, float]
        推奨重力ベクトル [m/s²]。デフォルトは (0, 0, -9.81)。
    """

    mesh: MeshData
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray
    substrate_mask: np.ndarray
    glass_mask: np.ndarray
    water_mask: np.ndarray
    solid_mask: np.ndarray
    gravity: tuple[float, float, float]


def _build_substrate_mask(
    shape: tuple[int, int, int],
    z_centers: np.ndarray,
    origin_z: float,
    substrate_depth: float,
) -> np.ndarray:
    """底床マスクを構築する.

    z 方向下端から `substrate_depth` までの範囲に含まれるセルを True にする。
    """
    mask = np.zeros(shape, dtype=bool)
    if substrate_depth <= 0.0:
        return mask
    substrate_top = origin_z + substrate_depth
    substrate_cells_z = z_centers <= substrate_top + 1e-12
    mask[:, :, substrate_cells_z] = True
    return mask


def _build_glass_mask(
    shape: tuple[int, int, int],
    dx: np.ndarray,
    dy: np.ndarray,
    glass_thickness: float,
) -> np.ndarray:
    """ガラス壁マスクを構築する.

    x/y 各方向の両端から `glass_thickness` 以内のセルをガラスとする。
    z 方向（上下）はガラスなし（上面=水面、下面=底床）。
    """
    mask = np.zeros(shape, dtype=bool)
    if glass_thickness <= 0.0:
        return mask

    # x 方向（左壁/右壁）
    cumsum_xm = np.cumsum(dx)
    cumsum_xp = np.cumsum(dx[::-1])[::-1]
    x_left_cells = cumsum_xm <= glass_thickness + 1e-12
    x_right_cells = cumsum_xp <= glass_thickness + 1e-12
    mask[x_left_cells, :, :] = True
    mask[x_right_cells, :, :] = True

    # y 方向（前壁/後壁）
    cumsum_ym = np.cumsum(dy)
    cumsum_yp = np.cumsum(dy[::-1])[::-1]
    y_front_cells = cumsum_ym <= glass_thickness + 1e-12
    y_back_cells = cumsum_yp <= glass_thickness + 1e-12
    mask[:, y_front_cells, :] = True
    mask[:, y_back_cells, :] = True

    return mask


class AquariumGeometryProcess(PreProcess["AquariumGeometryInput", "AquariumGeometryResult"]):
    """水槽ジオメトリ生成 Process.

    StructuredMeshProcess をラップし、水槽の底床・ガラス壁・水領域を
    マスク配列として返す。Phase 6.2b（ヒーター）以降の solid_mask / q_vol 入力の
    ベースとなる。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="AquariumGeometryProcess",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/aquarium-geometry.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [StructuredMeshProcess]

    def process(self, input_data: AquariumGeometryInput) -> AquariumGeometryResult:
        """水槽ジオメトリを生成する."""
        inp = input_data

        if inp.substrate_depth < 0:
            msg = f"substrate_depth は非負: {inp.substrate_depth}"
            raise ValueError(msg)
        if inp.substrate_depth >= inp.Lz:
            msg = (
                f"substrate_depth={inp.substrate_depth} が水槽高さ Lz={inp.Lz} 以上。"
                "底床が水槽を満たしてしまうため不正。"
            )
            raise ValueError(msg)
        if inp.substrate_refinement_ratio <= 0:
            msg = f"substrate_refinement_ratio は正の値が必要: {inp.substrate_refinement_ratio}"
            raise ValueError(msg)

        # ストレッチング指定（z 方向のみ refinement 対応）
        if inp.substrate_refinement_ratio > 1.0:
            stretch_z: tuple[float, ...] = (inp.substrate_refinement_ratio, 1.0)
        else:
            stretch_z = (1.0,)

        mesh_inp = StructuredMeshInput(
            Lx=inp.Lx,
            Ly=inp.Ly,
            Lz=inp.Lz,
            nx=inp.nx,
            ny=inp.ny,
            nz=inp.nz,
            stretch_x=(1.0,),
            stretch_y=(1.0,),
            stretch_z=stretch_z,
            origin=inp.origin,
        )
        mesh_res = StructuredMeshProcess().process(mesh_inp)

        x_centers = inp.origin[0] + np.cumsum(mesh_res.dx) - mesh_res.dx / 2
        y_centers = inp.origin[1] + np.cumsum(mesh_res.dy) - mesh_res.dy / 2
        z_centers = inp.origin[2] + np.cumsum(mesh_res.dz) - mesh_res.dz / 2

        shape = (inp.nx, inp.ny, inp.nz)

        substrate_mask = _build_substrate_mask(shape, z_centers, inp.origin[2], inp.substrate_depth)
        glass_mask = _build_glass_mask(shape, mesh_res.dx, mesh_res.dy, inp.glass_thickness)
        solid_mask = substrate_mask | glass_mask
        water_mask = ~solid_mask

        return AquariumGeometryResult(
            mesh=mesh_res.mesh,
            dx=mesh_res.dx,
            dy=mesh_res.dy,
            dz=mesh_res.dz,
            x_centers=x_centers,
            y_centers=y_centers,
            z_centers=z_centers,
            substrate_mask=substrate_mask,
            glass_mask=glass_mask,
            water_mask=water_mask,
            solid_mask=solid_mask,
            gravity=(0.0, 0.0, -9.81),
        )
