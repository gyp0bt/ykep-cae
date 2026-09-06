"""展開チャネル（構造格子）のせん断速度と混合指数.

粘度モデル Strategy そのものは方程式ファミリー非依存なので
:mod:`xkep_cae_fluid.fvm.viscosity` にある（``NewtonianViscosity`` /
``PowerLawViscosity`` / ``CarreauViscosity``）。ここには構造格子 (nx, ny) 専用の
γ̇ と λ の評価だけを置く。

せん断速度（設計文書 §2）:

    γ̇² = 2[(∂u/∂x)² + (∂v/∂y)²] + (∂u/∂y + ∂v/∂x)² + (∂w/∂x)² + (∂w/∂y)²

勾配はセル中心での Green-Gauss（面値の差 ÷ セル幅）で評価する。単なる中心差分だと
境界行で 1 次精度に落ちるが、**せん断が最大になるのはバレル直下**なのでそこが
粗いのは筋が悪い。面値に壁・バレルの境界値をそのまま入れる Green-Gauss なら、
単純せん断を境界セルも含めて厳密に再現する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae_fluid.extruder.data import ChannelGrid


__all__ = ["mixing_index", "strain_rate"]


def _grad_x(f: np.ndarray, grid: ChannelGrid, wall: float = 0.0) -> np.ndarray:
    """x 方向のセル中心勾配（周期・Green-Gauss）.

    固体に接する面は壁値 `wall` を使う。
    """
    dx = grid.dx
    solid = grid.solid
    f_next = np.roll(f, -1, axis=0)
    dx_next = np.roll(dx, -1)
    solid_next = np.roll(solid, -1, axis=0)

    w0 = (dx_next / (dx + dx_next))[:, None]
    face = w0 * f + (1.0 - w0) * f_next
    face = np.where((~solid) & (~solid_next), face, wall)

    face_east = face
    face_west = np.roll(face, 1, axis=0)
    return (face_east - face_west) / dx[:, None]


def _grad_y(f: np.ndarray, grid: ChannelGrid, f_bottom: float, f_top: float) -> np.ndarray:
    """y 方向のセル中心勾配（Green-Gauss）.

    y=0 の面に `f_bottom`（スクリュー根元）、y=H の面に `f_top`（バレル）を置く。
    固体に接する内部面は壁値 0。
    """
    dy = grid.dy
    solid = grid.solid
    face = np.zeros((grid.nx, grid.ny + 1))

    w0 = (dy[1:] / (dy[:-1] + dy[1:]))[None, :]
    interior = w0 * f[:, :-1] + (1.0 - w0) * f[:, 1:]
    both = (~solid[:, :-1]) & (~solid[:, 1:])
    face[:, 1:-1] = np.where(both, interior, 0.0)
    face[:, 0] = f_bottom
    face[:, -1] = f_top

    return (face[:, 1:] - face[:, :-1]) / dy[None, :]


def strain_rate(u: np.ndarray, v: np.ndarray, w: np.ndarray, grid: ChannelGrid) -> np.ndarray:
    """セル中心のせん断速度 γ̇ [1/s].

        γ̇² = 2[(∂u/∂x)² + (∂v/∂y)²] + (∂u/∂y + ∂v/∂x)² + (∂w/∂x)² + (∂w/∂y)²

    境界値は展開チャネルの境界条件を使う。
    y=0（スクリュー根元）で u=v=w=0、y=H（バレル）で u=u_barrel, v=0, w=w_barrel。
    固体セルの γ̇ は 0 を返す（そこの粘度は使われない）。
    """
    s = grid.spec
    ux = _grad_x(u, grid)
    vx = _grad_x(v, grid)
    wx = _grad_x(w, grid)
    uy = _grad_y(u, grid, 0.0, s.u_barrel)
    vy = _grad_y(v, grid, 0.0, 0.0)
    wy = _grad_y(w, grid, 0.0, s.w_barrel)

    g2 = 2.0 * (ux**2 + vy**2) + (uy + vx) ** 2 + wx**2 + wy**2
    gamma = np.sqrt(np.maximum(g2, 0.0))
    gamma[grid.solid] = 0.0
    return gamma


def mixing_index(u: np.ndarray, v: np.ndarray, w: np.ndarray, grid: ChannelGrid) -> np.ndarray:
    """セル中心の混合指数 λ = |D| / (|D| + |Ω|).

    D はひずみ速度テンソル、Ω は渦度テンソル、|·| は Frobenius ノルム
    （Manas-Zloczower の混合指数）。

        λ = 0    純回転（変形が無く混ざらない）
        λ = 0.5  単純せん断
        λ = 1    純伸長（分散混合に最も効く）

    完全発達（∂/∂z = 0）なので速度勾配は
    ∇V = [[ux, uy, 0], [vx, vy, 0], [wx, wy, 0]]。これから

        D:D = ux² + vy² + (uy+vx)²/2 + wx²/2 + wy²/2
        Ω:Ω = (uy−vx)²/2 + wx²/2 + wy²/2

    なお γ̇ = sqrt(2·D:D) が strain_rate() の定義と一致する。
    """
    s = grid.spec
    ux = _grad_x(u, grid)
    vx = _grad_x(v, grid)
    wx = _grad_x(w, grid)
    uy = _grad_y(u, grid, 0.0, s.u_barrel)
    vy = _grad_y(v, grid, 0.0, 0.0)
    wy = _grad_y(w, grid, 0.0, s.w_barrel)

    dd = ux**2 + vy**2 + 0.5 * (uy + vx) ** 2 + 0.5 * wx**2 + 0.5 * wy**2
    oo = 0.5 * (uy - vx) ** 2 + 0.5 * wx**2 + 0.5 * wy**2
    norm_d = np.sqrt(np.maximum(dd, 0.0))
    norm_o = np.sqrt(np.maximum(oo, 0.0))
    total = norm_d + norm_o
    lam = np.where(total > 0.0, norm_d / np.where(total > 0.0, total, 1.0), 0.0)
    lam[grid.solid] = 0.0
    return lam
