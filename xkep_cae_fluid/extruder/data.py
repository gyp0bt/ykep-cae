"""単軸押出 展開チャネル 2.5D のデータ契約.

座標系:
  x  横断方向（フライトに直交）。x=0 と x=W_t は周期。+x が下流側の隣チャネル
  y  深さ。y=0 スクリュー根元、y=H バレル
  z  下流方向（フライトに沿う）。完全発達を仮定し ∂/∂z = 0

幾何恒等式（docs/design/single-screw-extruder.md §2.1.1）:
  W_t    = πD·sinφ      チャネル 1 ピッチのフライト直交幅
  L_turn = πD·cosφ      隣チャネルまでの下流距離
  W_t / L_turn = tanφ           （D にもリードにも依らない）
  β = G·L_turn/W_t = G·cotφ     横断方向の一様圧力勾配（同上）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae_fluid.core.data import MeshData


@dataclass(frozen=True)
class ScrewSpec:
    """スクリュー諸元と格子解像度.

    Parameters
    ----------
    D : float
        バレル内径 [m]
    lead : float
        リード（1 回転あたりの軸方向前進量）[m]
    H : float
        計量部チャネル深さ [m]
    e : float
        フライト幅（フライト直交方向）[m]
    delta : float
        フライト隙間 [m]。0.0 で閉チャネル（G1/G2 用）
    N : float
        回転数 [1/s]。rpm ではないことに注意
    nx_channel : int
        チャネル部（フライト以外）の x 方向セル数
    nx_land : int
        フライト頂部（ランド）の x 方向セル数
    ny_bulk : int
        隙間より下のバルク部の y 方向セル数
    n_gap : int
        隙間 delta の中に入れる y 方向セル数。delta=0 なら無視される
    """

    D: float
    lead: float
    H: float
    e: float
    delta: float
    N: float
    nx_channel: int = 200
    nx_land: int = 48
    ny_bulk: int = 60
    n_gap: int = 20

    @property
    def phi(self) -> float:
        """リード角 [rad]. tanφ = lead / (πD)."""
        return math.atan(self.lead / (math.pi * self.D))

    @property
    def W_t(self) -> float:
        """チャネル 1 ピッチのフライト直交幅 W_t = πD·sinφ [m]."""
        return math.pi * self.D * math.sin(self.phi)

    @property
    def W(self) -> float:
        """チャネル幅（フライトを除く）[m]."""
        return self.W_t - self.e

    @property
    def L_turn(self) -> float:
        """隣チャネルまでの下流距離 L_turn = πD·cosφ [m]."""
        return math.pi * self.D * math.cos(self.phi)

    @property
    def V(self) -> float:
        """バレルの相対周速 V = πDN [m/s]."""
        return math.pi * self.D * self.N

    @property
    def u_barrel(self) -> float:
        """バレルの横断方向速度 [m/s]. 負（-x = 上流側）."""
        return -self.V * math.sin(self.phi)

    @property
    def w_barrel(self) -> float:
        """バレルの下流方向速度 [m/s]. 正（下流向き）."""
        return self.V * math.cos(self.phi)

    def beta(self, G: float) -> float:
        """横断方向の一様圧力勾配 β = dP/dx = G·cotφ [Pa/m].

        断面内運動量には体積力 f_x = -β として入る。
        """
        return G / math.tan(self.phi)


@dataclass(frozen=True)
class ChannelGrid:
    """展開チャネル断面の不等間隔格子.

    Parameters
    ----------
    dx, dy : np.ndarray
        セル幅 (nx,), (ny,) [m]
    xc, yc : np.ndarray
        セル中心座標 (nx,), (ny,) [m]
    solid : np.ndarray
        (nx, ny) bool。True = フライト（固体）
    spec : ScrewSpec
        元の諸元
    mesh : MeshData
        StructuredMeshProcess が生成した MeshData（来歴保持用）
    """

    dx: np.ndarray
    dy: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    solid: np.ndarray
    spec: ScrewSpec
    mesh: MeshData

    @property
    def nx(self) -> int:
        return int(self.dx.shape[0])

    @property
    def ny(self) -> int:
        return int(self.dy.shape[0])

    @property
    def area_free(self) -> float:
        """流体セルの断面積和 [m²]."""
        cell = self.dx[:, None] * self.dy[None, :]
        return float(cell[~self.solid].sum())


@dataclass(frozen=True)
class DownChannelInput:
    """下流方向流れ w の入力.

    Parameters
    ----------
    grid : ChannelGrid
        断面格子
    mu : np.ndarray
        (nx, ny) 粘度場 [Pa·s]。ニュートンなら定数配列
    G : float
        下流方向圧力勾配 dp/dz [Pa/m]。押出（背圧あり）は正
    """

    grid: ChannelGrid
    mu: np.ndarray
    G: float


@dataclass(frozen=True)
class DownChannelResult:
    """下流方向流れ w の結果.

    Parameters
    ----------
    w : np.ndarray
        (nx, ny) 下流方向速度 [m/s]。固体セルは 0
    Q : float
        体積流量 [m³/s]（断面積分 ∫∫ w dx dy）
    """

    w: np.ndarray
    Q: float
