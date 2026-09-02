"""粘度モデル Strategy とせん断速度の評価.

FluidProperties.power_law_n / power_law_k は宣言済みで未使用だった。
非ニュートンの席は既に用意されていたので、ここに具象を座らせる。

粘度は「直交する振る舞い軸」なので Process ではなく Strategy Protocol とし、
ソルバー側は StrategySlot で受ける（core/strategies と同じ流儀）。

せん断速度（設計文書 §2）:

    γ̇² = 2[(∂u/∂x)² + (∂v/∂y)²] + (∂u/∂y + ∂v/∂x)² + (∂w/∂x)² + (∂w/∂y)²

勾配はセル中心での Green-Gauss（面値の差 ÷ セル幅）で評価する。単なる中心差分だと
境界行で 1 次精度に落ちるが、**せん断が最大になるのはバレル直下**なのでそこが
粗いのは筋が悪い。面値に壁・バレルの境界値をそのまま入れる Green-Gauss なら、
単純せん断を境界セルも含めて厳密に再現する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from xkep_cae_fluid.extruder.data import ChannelGrid


@runtime_checkable
class ViscosityModelStrategy(Protocol):
    """せん断速度 → 粘度の対応を規定する."""

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """せん断速度 γ̇ [1/s] から粘度 μ [Pa·s] を返す（形状は入力と同じ）."""
        ...


@dataclass(frozen=True)
class NewtonianViscosity:
    """ニュートン流体 μ = const."""

    mu: float

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(gamma_dot, dtype=np.float64), self.mu)


@dataclass(frozen=True)
class PowerLawViscosity:
    """べき乗則 μ = K·γ̇^(n−1).

    n < 1 では γ̇ → 0 で発散するので gamma_min でクランプし、さらに mu_max で
    頭を押さえる。**この 2 つは数値上の安全弁であって物理ではない**ので、
    結果がこれらに依存しないことをテストで確認すること。

    既定の gamma_min = 1e-2 s⁻¹ は 40mm 機の代表せん断速度 V/H ≈ 52 s⁻¹ の
    2×10⁻⁴ 倍にあたる。
    """

    K: float
    n: float
    gamma_min: float = 1.0e-2
    mu_max: float = 1.0e8

    def __post_init__(self) -> None:
        if self.n <= 0.0:
            msg = f"べき乗則指数 n は正が必要: {self.n}"
            raise ValueError(msg)
        if self.K <= 0.0:
            msg = f"べき乗則定数 K は正が必要: {self.K}"
            raise ValueError(msg)
        if self.gamma_min <= 0.0:
            msg = f"gamma_min は正が必要: {self.gamma_min}"
            raise ValueError(msg)

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        g = np.maximum(np.asarray(gamma_dot, dtype=np.float64), self.gamma_min)
        return np.minimum(self.K * g ** (self.n - 1.0), self.mu_max)


@dataclass(frozen=True)
class CarreauViscosity:
    """Carreau モデル μ = μ_∞ + (μ_0 − μ_∞)[1 + (λγ̇)²]^((n−1)/2).

    低せん断で μ_0、高せん断で μ_∞ に漸近するのでクランプが要らない。
    n = 1 かつ μ_0 = μ_∞ でニュートンに退化する。
    """

    mu_0: float
    mu_inf: float
    lam: float
    n: float

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        g = np.asarray(gamma_dot, dtype=np.float64)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * (1.0 + (self.lam * g) ** 2) ** (
            (self.n - 1.0) / 2.0
        )


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
