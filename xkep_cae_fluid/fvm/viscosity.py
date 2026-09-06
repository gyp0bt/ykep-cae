"""粘度モデル Strategy とセル勾配からのせん断速度（方程式ファミリー非依存）.

粘度は「直交する振る舞い軸」なので Process ではなく Strategy Protocol とし、ソルバー側は
StrategySlot / 入力フィールドで受ける（``core/strategies`` と同じ流儀）。構造格子専用の
γ̇ 評価（:mod:`xkep_cae_fluid.extruder.viscosity`）に対し、ここは ``MeshData`` の面リスト上で
最小二乗のセル勾配テンソル ∇u から評価する。

    γ̇ = sqrt(2 D:D)、D = ½(∇u + ∇uᵀ)

壁・流入面は Dirichlet 値を点集合に入れるので、単純せん断は境界セルでも厳密に再現する
（Green–Gauss は非 Dirichlet 境界で法線勾配を過小評価するが、速度は壁で Dirichlet なので問題ない）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.geometry import cell_gradient_lsq
from xkep_cae_fluid.fvm.momentum import VelocityBoundaryFaces, component_boundary


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


def velocity_gradient_cells(mesh: MeshData, u: np.ndarray, vb: VelocityBoundaryFaces) -> np.ndarray:
    """セル中心の速度勾配テンソル L_ij = ∂u_j/∂x_i (n_cells, 3, 3)（最小二乗、Dirichlet 面を含む）.

    ``u`` は (n_cells, 3)。メッシュが 2 次元（面法線が 2 成分）なら z 行はゼロ。
    """
    n = mesh.n_cells
    nd = mesh.face_normals.shape[1]
    u3 = np.asarray(u, dtype=np.float64)
    if u3.shape[1] < 3:
        u3 = np.hstack([u3, np.zeros((n, 3 - u3.shape[1]))])
    L = np.zeros((n, 3, 3))
    for j in range(3):
        bf = component_boundary(vb, u3, j)
        g = cell_gradient_lsq(mesh, u3[:, j], bf)
        L[:, :nd, j] = g[:, :nd]
    return L


def strain_rate_from_gradient(L: np.ndarray) -> np.ndarray:
    """速度勾配テンソルからせん断速度 γ̇ = sqrt(2 D:D) (n_cells,)."""
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    return np.sqrt(np.maximum(2.0 * np.einsum("nij,nij->n", D, D), 0.0))


def mixing_index_from_gradient(L: np.ndarray) -> np.ndarray:
    """混合指数 λ = |D| / (|D| + |Ω|)（0: 純回転、0.5: 単純せん断、1: 純伸長）(n_cells,)."""
    Lt = np.transpose(L, (0, 2, 1))
    D = 0.5 * (L + Lt)
    W = 0.5 * (L - Lt)
    nd_ = np.sqrt(np.maximum(np.einsum("nij,nij->n", D, D), 0.0))
    nw = np.sqrt(np.maximum(np.einsum("nij,nij->n", W, W), 0.0))
    total = nd_ + nw
    return np.where(total > 0.0, nd_ / np.where(total > 0.0, total, 1.0), 0.0)


def viscous_stress_transpose_source(L: np.ndarray, grad_mu: np.ndarray) -> np.ndarray:
    """変粘度の応力 ∇·(μ ∇uᵀ) のうち ∇·(μ∇u) に含まれない部分 (n_cells, 3).

    非圧縮なら ∂_j(μ ∂_i u_j) = Σ_j (∂_i u_j)(∂_j μ)。μ が一様ならゼロ。
    運動量方程式の拡散項を ∇·(μ∇u) で組むときの陽的ソースとして足す。
    """
    return np.einsum("nij,nj->ni", L, np.asarray(grad_mu, dtype=np.float64))


__all__ = [
    "ViscosityModelStrategy",
    "NewtonianViscosity",
    "PowerLawViscosity",
    "CarreauViscosity",
    "velocity_gradient_cells",
    "strain_rate_from_gradient",
    "mixing_index_from_gradient",
    "viscous_stress_transpose_source",
]
