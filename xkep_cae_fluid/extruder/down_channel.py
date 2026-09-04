"""下流方向流れ w の可変係数 Poisson ソルバー.

完全発達を仮定すると下流方向運動量は

    0 = −G + ∂/∂x(μ ∂w/∂x) + ∂/∂y(μ ∂w/∂y)        すなわち  ∇·(μ∇w) = G

という 2 次元の可変係数 Poisson になる。慣性項は Re ~ 10⁻³ で落としてある。
ニュートン流体では断面内流れ (u,v) と完全に分離し、この式だけで流量が決まる。
だから古典的な形状係数 Fd/Fp と直接比較でき、ゲート G1/G2 が成立する。

離散化（有限体積・セル中心・単位 z 厚さあたり）:

    Σ_faces μ_f A_f (w_P − w_N)/d_PN = −G · dx_i · dy_j

    x 面: A_f = dy_j,  d_PN = (dx_i + dx_{i±1})/2   （i=0 と i=nx-1 は周期で接続）
    y 面: A_f = dx_i,  d_PN = (dy_j + dy_{j±1})/2
    壁面: 面上 Dirichlet。d = dx_i/2 または dy_j/2
      - スクリュー根元・フライト表面: w = 0
      - バレル y=H:                   w = spec.w_barrel = V cosφ
    μ_f は隣接セルの調和平均（拡散型作用素の面値として物理的に正しい平均）

右辺の符号は `∇·(μ∇w) = G` を上の形に移項して出る `−G·dV`。G>0（背圧）で
w が負向きに引かれ、押出量が減る向きになる。

w は周期で**跳びが無い**（跳ぶのは圧力だけ）。行列は対称正定値で、
クリープ流れゆえ反復不要。splu 一発で機械精度の解が出る。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.extruder.data import DownChannelInput, DownChannelResult


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """調和平均 2ab/(a+b). 拡散型作用素の面値として物理的に正しい平均."""
    return 2.0 * a * b / (a + b)


class DownChannelFlowProcess(SolverProcess["DownChannelInput", "DownChannelResult"]):
    """下流方向速度 w の可変係数 Poisson を疎直接解で解く.

    行列は対称正定値。クリープ流れなので反復は不要で、splu 一発で
    機械精度の解が出る。これにより解析解との比較が反復の収束残差に汚されない。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="DownChannelFlow",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: DownChannelInput) -> DownChannelResult:
        """可変係数 Poisson を組んで解き、速度場と流量を返す."""
        g = input_data.grid
        mu = np.asarray(input_data.mu, dtype=np.float64)
        nx, ny = g.nx, g.ny
        if mu.shape != (nx, ny):
            msg = f"mu の形状が格子と不一致: {mu.shape} != {(nx, ny)}"
            raise ValueError(msg)
        if np.any(mu <= 0.0):
            msg = "粘度に 0 以下の値が含まれる"
            raise ValueError(msg)

        fluid = ~g.solid
        n_unknown = int(fluid.sum())
        if n_unknown == 0:
            msg = "流体セルが 1 つも無い（固体マスクが領域全体を覆っている）"
            raise ValueError(msg)

        idx = np.full((nx, ny), -1, dtype=np.int64)
        idx[fluid] = np.arange(n_unknown, dtype=np.int64)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        diag = np.zeros(n_unknown)
        rhs = np.zeros(n_unknown)

        dx, dy = g.dx, g.dy

        def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
            rows.append(r)
            cols.append(c)
            vals.append(v)

        # --- x 方向の面（周期） ---
        for i in range(nx):
            ip = (i + 1) % nx
            dist = 0.5 * (dx[i] + dx[ip])
            coef = harmonic_mean(mu[i, :], mu[ip, :]) * dy / dist  # (ny,)

            both = fluid[i, :] & fluid[ip, :]
            if both.any():
                a = idx[i, both]
                b = idx[ip, both]
                c = coef[both]
                diag[a] += c
                diag[b] += c
                add(a, b, -c)
                add(b, a, -c)

            # 固体との境界: 面上 w=0 の Dirichlet（右辺への寄与は 0）
            wall_a = fluid[i, :] & ~fluid[ip, :]
            if wall_a.any():
                diag[idx[i, wall_a]] += mu[i, wall_a] * dy[wall_a] / (0.5 * dx[i])
            wall_b = fluid[ip, :] & ~fluid[i, :]
            if wall_b.any():
                diag[idx[ip, wall_b]] += mu[ip, wall_b] * dy[wall_b] / (0.5 * dx[ip])

        # --- y 方向の内部面 ---
        for j in range(ny - 1):
            dist = 0.5 * (dy[j] + dy[j + 1])
            coef = harmonic_mean(mu[:, j], mu[:, j + 1]) * dx / dist  # (nx,)

            both = fluid[:, j] & fluid[:, j + 1]
            if both.any():
                a = idx[both, j]
                b = idx[both, j + 1]
                c = coef[both]
                diag[a] += c
                diag[b] += c
                add(a, b, -c)
                add(b, a, -c)

            wall_a = fluid[:, j] & ~fluid[:, j + 1]
            if wall_a.any():
                diag[idx[wall_a, j]] += mu[wall_a, j] * dx[wall_a] / (0.5 * dy[j])
            wall_b = fluid[:, j + 1] & ~fluid[:, j]
            if wall_b.any():
                diag[idx[wall_b, j + 1]] += mu[wall_b, j + 1] * dx[wall_b] / (0.5 * dy[j + 1])

        # --- y=0 スクリュー根元（w = 0） ---
        bot = fluid[:, 0]
        if bot.any():
            diag[idx[bot, 0]] += mu[bot, 0] * dx[bot] / (0.5 * dy[0])

        # --- y=H バレル（w = w_barrel） ---
        top = fluid[:, ny - 1]
        if top.any():
            coef_top = mu[top, ny - 1] * dx[top] / (0.5 * dy[ny - 1])
            diag[idx[top, ny - 1]] += coef_top
            rhs[idx[top, ny - 1]] += coef_top * g.spec.w_barrel

        # --- ソース項 −G·dV ---
        cell_area = (dx[:, None] * dy[None, :])[fluid]
        rhs -= input_data.G * cell_area

        add(np.arange(n_unknown, dtype=np.int64), np.arange(n_unknown, dtype=np.int64), diag)
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_unknown, n_unknown),
        ).tocsc()

        try:
            lu = spla.splu(A)
        except RuntimeError as exc:
            msg = (
                "下流方向 Poisson の LU 分解に失敗した"
                f"（未知数 {n_unknown}。格子または粘度場が不正の可能性）"
            )
            raise RuntimeError(msg) from exc
        x = lu.solve(rhs)

        w = np.zeros((nx, ny))
        w[fluid] = x
        return DownChannelResult(w=w, Q=float(np.sum(x * cell_area)))
