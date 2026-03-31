"""伝熱解析結果の可視化 PostProcess.

温度マップ（2Dスライス）の描画・保存機能を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.heat_transfer.data import HeatTransferResult


@dataclass(frozen=True)
class TemperatureMapInput:
    """温度マップ可視化の入力.

    Parameters
    ----------
    result : HeatTransferResult
        伝熱解析の結果
    Lx, Ly, Lz : float
        領域サイズ [m]
    slice_axis : str
        スライス軸 ("x", "y", "z")
    slice_index : int | None
        スライス位置のインデックス（None = 中央）
    title : str
        図のタイトル
    output_path : Path | None
        保存先パス（None = 保存しない）
    cmap : str
        カラーマップ名
    show_colorbar : bool
        カラーバーを表示するか
    figsize : tuple[float, float]
        図のサイズ (width, height) [inch]
    dpi : int
        解像度
    layer_boundaries : tuple[float, ...] | None
        層境界位置 [m]（描画用の水平線）
    layer_labels : tuple[str, ...] | None
        各層のラベル
    vmin : float | None
        カラーマップの最小値
    vmax : float | None
        カラーマップの最大値
    """

    result: HeatTransferResult
    Lx: float
    Ly: float
    Lz: float
    slice_axis: str = "y"
    slice_index: int | None = None
    title: str = "Temperature [K]"
    output_path: Path | None = None
    cmap: str = "hot"
    show_colorbar: bool = True
    figsize: tuple[float, float] = (10, 6)
    dpi: int = 150
    layer_boundaries: tuple[float, ...] | None = None
    layer_labels: tuple[str, ...] | None = None
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class TemperatureMapOutput:
    """温度マップ可視化の出力.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        生成された図オブジェクト
    saved_path : Path | None
        保存先パス（保存しなかった場合は None）
    T_min : float
        スライス内の最小温度
    T_max : float
        スライス内の最大温度
    """

    fig: object  # matplotlib.figure.Figure
    saved_path: Path | None = None
    T_min: float = 0.0
    T_max: float = 0.0


class TemperatureMapProcess(PostProcess["TemperatureMapInput", "TemperatureMapOutput"]):
    """温度マップ（2Dスライス）を描画する PostProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="TemperatureMap",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/temperature-map.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: TemperatureMapInput) -> TemperatureMapOutput:
        """温度マップを描画する."""
        inp = input_data
        T = inp.result.T
        nx, ny, nz = T.shape

        # スライス位置の決定
        axis = inp.slice_axis.lower()
        if axis == "x":
            idx = inp.slice_index if inp.slice_index is not None else nx // 2
            T_slice = T[idx, :, :].T  # (nz, ny)
            extent_h = inp.Ly
            extent_v = inp.Lz
            xlabel = "y [m]"
            ylabel = "z [m]"
            nh, nv = ny, nz
        elif axis == "y":
            idx = inp.slice_index if inp.slice_index is not None else ny // 2
            T_slice = T[:, idx, :].T  # (nz, nx)
            extent_h = inp.Lx
            extent_v = inp.Lz
            xlabel = "x [m]"
            ylabel = "z [m]"
            nh, nv = nx, nz
        elif axis == "z":
            idx = inp.slice_index if inp.slice_index is not None else nz // 2
            T_slice = T[:, :, idx].T  # (ny, nx)
            extent_h = inp.Lx
            extent_v = inp.Ly
            xlabel = "x [m]"
            ylabel = "y [m]"
            nh, nv = nx, ny
        else:
            raise ValueError(f"不正なスライス軸: {axis}")

        # セル中心座標
        dh = extent_h / nh
        dv = extent_v / nv
        h_coords = np.linspace(dh / 2, extent_h - dh / 2, nh)
        v_coords = np.linspace(dv / 2, extent_v - dv / 2, nv)

        fig, ax = plt.subplots(1, 1, figsize=inp.figsize)

        vmin = inp.vmin if inp.vmin is not None else float(T_slice.min())
        vmax = inp.vmax if inp.vmax is not None else float(T_slice.max())

        im = ax.pcolormesh(
            h_coords,
            v_coords,
            T_slice,
            cmap=inp.cmap,
            shading="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # 層境界線の描画
        if inp.layer_boundaries is not None:
            for boundary_pos in inp.layer_boundaries:
                ax.axhline(y=boundary_pos, color="white", linewidth=0.8, linestyle="--")

        # 層ラベルの描画
        if inp.layer_labels is not None and inp.layer_boundaries is not None:
            boundaries = list(inp.layer_boundaries)
            # 層の中心位置にラベルを配置
            all_bounds = [0.0] + boundaries + [extent_v]
            for i, label in enumerate(inp.layer_labels):
                if i < len(all_bounds) - 1:
                    y_center = (all_bounds[i] + all_bounds[i + 1]) / 2
                    ax.text(
                        extent_h * 0.02,
                        y_center,
                        label,
                        color="white",
                        fontsize=8,
                        va="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(inp.title)
        ax.set_aspect("equal")

        if inp.show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Temperature [K]")

        fig.tight_layout()

        saved_path = None
        if inp.output_path is not None:
            inp.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(inp.output_path, dpi=inp.dpi, bbox_inches="tight")
            saved_path = inp.output_path

        return TemperatureMapOutput(
            fig=fig,
            saved_path=saved_path,
            T_min=float(T_slice.min()),
            T_max=float(T_slice.max()),
        )
