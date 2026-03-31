"""多層シート物性値ビルダー PreProcess.

積層構造の物性値配列（k, C, q, T0）を層定義リストから自動構築する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess


@dataclass(frozen=True)
class LayerSpec:
    """1層の物性定義.

    Parameters
    ----------
    thickness : float
        層厚 [m]
    k : float
        熱伝導率 [W/(m·K)]
    C : float
        体積熱容量 ρCp [J/(m³·K)]
    q : float
        体積発熱量 [W/m³]
    name : str
        層名（可視化ラベル用）
    """

    thickness: float
    k: float
    C: float = 1.0
    q: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class MultilayerInput:
    """多層シートビルダーの入力.

    Parameters
    ----------
    layers : tuple[LayerSpec, ...]
        層定義（z=0 側から積層順）
    nx, ny : int
        x, y 方向のセル数
    Lx, Ly : float
        x, y 方向の領域サイズ [m]
    T0_default : float
        デフォルト初期温度 [K]
    nz_per_meter : float
        z方向のセル密度 [cells/m]（デフォルト 1000 = 1mm あたり 1セル）
    """

    layers: tuple[LayerSpec, ...]
    nx: int
    ny: int
    Lx: float
    Ly: float
    T0_default: float = 300.0
    nz_per_meter: float = 1000.0


@dataclass(frozen=True)
class MultilayerOutput:
    """多層シートビルダーの出力.

    Parameters
    ----------
    k : np.ndarray
        熱伝導率配列 (nx, ny, nz)
    C : np.ndarray
        体積熱容量配列 (nx, ny, nz)
    q : np.ndarray
        体積発熱量配列 (nx, ny, nz)
    T0 : np.ndarray
        初期温度配列 (nx, ny, nz)
    Lz : float
        z方向の合計厚み [m]
    nz : int
        z方向の合計セル数
    layer_boundaries : tuple[float, ...]
        層境界の z 座標 [m]（最下端=0, 最上端=Lz を除く中間境界）
    layer_names : tuple[str, ...]
        層名一覧
    """

    k: np.ndarray
    C: np.ndarray
    q: np.ndarray
    T0: np.ndarray
    Lz: float
    nz: int
    layer_boundaries: tuple[float, ...]
    layer_names: tuple[str, ...]


class MultilayerBuilderProcess(PreProcess["MultilayerInput", "MultilayerOutput"]):
    """多層シートの物性値配列を構築する PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="MultilayerBuilder",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/multilayer-builder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: MultilayerInput) -> MultilayerOutput:
        """層定義から物性値配列を構築する."""
        inp = input_data
        if len(inp.layers) == 0:
            raise ValueError("layers は1層以上必要です")

        # 各層のセル数を決定
        Lz_total = sum(layer.thickness for layer in inp.layers)
        nz_cells = []
        for layer in inp.layers:
            n = max(1, round(layer.thickness * inp.nz_per_meter))
            nz_cells.append(n)
        nz_total = sum(nz_cells)

        # 3D 配列の構築
        k_arr = np.empty((inp.nx, inp.ny, nz_total))
        C_arr = np.empty((inp.nx, inp.ny, nz_total))
        q_arr = np.empty((inp.nx, inp.ny, nz_total))
        T0_arr = np.full((inp.nx, inp.ny, nz_total), inp.T0_default)

        z_offset = 0
        boundaries: list[float] = []
        names: list[str] = []
        z_pos = 0.0

        for i, (layer, nz_layer) in enumerate(zip(inp.layers, nz_cells, strict=True)):
            k_arr[:, :, z_offset : z_offset + nz_layer] = layer.k
            C_arr[:, :, z_offset : z_offset + nz_layer] = layer.C
            q_arr[:, :, z_offset : z_offset + nz_layer] = layer.q
            z_offset += nz_layer
            z_pos += layer.thickness
            if i < len(inp.layers) - 1:
                boundaries.append(z_pos)
            names.append(layer.name if layer.name else f"Layer{i + 1}")

        return MultilayerOutput(
            k=k_arr,
            C=C_arr,
            q=q_arr,
            T0=T0_arr,
            Lz=Lz_total,
            nz=nz_total,
            layer_boundaries=tuple(boundaries),
            layer_names=tuple(names),
        )
