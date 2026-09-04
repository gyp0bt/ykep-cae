"""断面内 (u, v, p̃) の 2 次元 Stokes ソルバー（MAC 千鳥格子・疎直接解）.

Re ~ 10⁻³ なので慣性項が無く、系は線形。SIMPLE のような圧力-速度連成反復は
不要で、鞍点系 [A −Bᵀ; B 0] を LU 一発で解ける。

可変粘度に対応するため μ∇²u ではなく完全な ∇·(μ(∇u + ∇uᵀ)) を離散化する。
法線応力の μ はセル中心値、せん断応力の μ は**節点**値（隣接する流体セルの調和平均）。

x 周期の圧力跳びは P = βx + p̃ の分解により横断方向の一様体積力
f_x = −β = −G·cotφ として入る（docs/design/single-screw-extruder.md §2.1.1）。

千鳥配置:
    u  x 面 (nx, ny)      面 i はセル i の西面。周期なので nx 枚
    v  y 面 (nx, ny+1)    面 j はセル j の南面。j=0 が根元、j=ny がバレル
    p̃ セル中心 (nx, ny)

境界条件:
    y=0 根元      u=v=0（v は未知数にせず 0 固定、u はせん断項に壁値 0 で入る）
    y=H バレル    v=0、u はせん断項に u_barrel が入る（右辺へ）
    フライト表面  u=v=0
    x 両端        周期

**固体に隣接するせん断項の距離は半セルにする。** フライト頂面の no-slip は
y 面の上にあるので、フルセル距離を使うと壁が固体セル中心まで半セル分深く置かれ、
隙間が実効的に広くなる。隙間は 0.1mm しかなく 1% 精度を狙う場所なので許容できない。
代償として粘性作用素の対称性が崩れる（対称性は「せん断の微分距離＝CV 寸法」の
ときだけ成立する）が、splu は非対称でも解けるので精度を取る。
フライト無しのケースでは全距離が自然なので対称性が復活し、テストで確認している。

粒子追跡のために、面流束を積分した節点流れ関数 ψ も返す。ψ の双一次補間から
作った速度はセル内で厳密に発散ゼロなので、粒子が湧いたり消えたりしない。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.extruder.data import (
    ChannelGrid,
    CrossChannelInput,
    CrossChannelResult,
)


class StokesLayout:
    """MAC 千鳥格子の自由度番号づけ.

    u は両隣のセルが流体な x 面、v は上下のセルが流体な内部 y 面、
    p̃ は流体セルにだけ自由度を置く。それ以外の速度は壁値 0 の既知量。
    """

    def __init__(self, grid: ChannelGrid) -> None:
        nx, ny = grid.nx, grid.ny
        fluid = ~grid.solid

        self.grid = grid
        self.nx = nx
        self.ny = ny
        self.fluid = fluid

        u_ok = np.roll(fluid, 1, axis=0) & fluid  # 面 i はセル i-1 と i の間
        v_ok = np.zeros((nx, ny + 1), dtype=bool)
        v_ok[:, 1:ny] = fluid[:, :-1] & fluid[:, 1:]

        self.iu = np.full((nx, ny), -1, dtype=np.int64)
        self.iv = np.full((nx, ny + 1), -1, dtype=np.int64)
        self.ip = np.full((nx, ny), -1, dtype=np.int64)

        self.nu = int(u_ok.sum())
        self.nv = int(v_ok.sum())
        self.npr = int(fluid.sum())

        self.iu[u_ok] = np.arange(self.nu, dtype=np.int64)
        self.iv[v_ok] = self.nu + np.arange(self.nv, dtype=np.int64)
        self.ip[fluid] = self.nu + self.nv + np.arange(self.npr, dtype=np.int64)

        self.u_ok = u_ok
        self.v_ok = v_ok
        self.n_total = self.nu + self.nv + self.npr
        self.p_offset = self.nu + self.nv


def _node_viscosity(grid: ChannelGrid, mu: np.ndarray) -> np.ndarray:
    """節点 (i, jn) の粘度: 周囲 4 セルのうち流体のものの調和平均.

    せん断応力の面値としては調和平均が正しい（強い粘度差でも層の直列抵抗を再現する）。
    周囲が全て固体の節点は 0 を返すが、そこのせん断項は使われない。
    """
    nx, ny = grid.nx, grid.ny
    inv_sum = np.zeros((nx, ny + 1))
    count = np.zeros((nx, ny + 1))
    jn = np.arange(ny + 1)

    for di in (0, -1):
        ii = (np.arange(nx) + di) % nx
        for dj in (0, -1):
            jj = jn + dj
            ok = (jj >= 0) & (jj <= ny - 1)
            sub_mu = mu[np.ix_(ii, jj[ok])]
            sub_fluid = (~grid.solid)[np.ix_(ii, jj[ok])]
            inv_sum[:, ok] += np.where(sub_fluid, 1.0 / np.where(sub_fluid, sub_mu, 1.0), 0.0)
            count[:, ok] += sub_fluid

    return np.where(count > 0, count / np.where(inv_sum > 0, inv_sum, 1.0), 0.0)


class _NodeShear:
    """節点 (i, jn) における τ = μ(∂u/∂y + ∂v/∂x) の線形形.

    τ / μ_node = cu_above·U(iu_above) + cu_below·U(iu_below)
               + cv_right·V(iv_right) + cv_left·V(iv_left) + const

    const はバレル速度（既知量）から来る項で、右辺へ回す。
    索引が -1 の項は壁値 0 なので寄与しない。

    固体に隣接する側は距離を半セルにする（壁が y 面・x 面の上にあるため）。
    """

    def __init__(self, layout: StokesLayout) -> None:
        grid = layout.grid
        nx, ny = layout.nx, layout.ny
        dx, dy = grid.dx, grid.dy
        shape = (nx, ny + 1)

        # ---- ∂u/∂y ----
        iu_below = np.full(shape, -1, dtype=np.int64)
        iu_above = np.full(shape, -1, dtype=np.int64)
        iu_below[:, 1:] = layout.iu
        iu_above[:, :ny] = layout.iu

        dy_below = np.zeros(ny + 1)
        dy_below[1:] = dy
        dy_above = np.zeros(ny + 1)
        dy_above[:ny] = dy

        below_ok = iu_below >= 0
        above_ok = iu_above >= 0
        # jn=0 は根元壁（値 0）、jn=ny はバレル（値 u_barrel）が「外側」に来る
        is_bottom = np.zeros(shape, dtype=bool)
        is_bottom[:, 0] = True
        is_top = np.zeros(shape, dtype=bool)
        is_top[:, ny] = True

        d = np.ones(shape)
        cu_above = np.zeros(shape)
        cu_below = np.zeros(shape)
        const = np.zeros(shape)

        # 内部: 両側有効 → 自然距離
        both = below_ok & above_ok & ~is_bottom & ~is_top
        d_both = 0.5 * (dy_below + dy_above)[None, :]
        d = np.where(both, np.broadcast_to(d_both, shape), d)

        # 上だけ有効（下が固体、または jn=0 の根元壁）→ 半セル
        only_above = above_ok & (~below_ok | is_bottom)
        d = np.where(only_above, np.broadcast_to(0.5 * dy_above[None, :], shape), d)

        # 下だけ有効（上が固体）→ 半セル
        only_below = below_ok & ~above_ok & ~is_top
        d = np.where(only_below, np.broadcast_to(0.5 * dy_below[None, :], shape), d)

        # バレル面: 上はバレル（既知）、下はセル値（無効なら壁値 0）→ 半セル
        d = np.where(is_top, np.broadcast_to(0.5 * dy_below[None, :], shape), d)

        d = np.where(d > 0, d, 1.0)
        cu_above = np.where(both | only_above, 1.0 / d, 0.0)
        cu_below = np.where(both | only_below, -1.0 / d, 0.0)
        # is_top では above がバレル定数、below はセル自由度（有効なら）
        cu_above = np.where(is_top, 0.0, cu_above)
        cu_below = np.where(is_top & below_ok, -1.0 / d, cu_below)
        const = np.where(is_top, grid.spec.u_barrel / d, 0.0)

        self.iu_above = np.where(is_top, -1, iu_above)
        self.iu_below = iu_below
        self.cu_above = cu_above
        self.cu_below = cu_below
        self.const = const

        # ---- ∂v/∂x ----
        iv_right = layout.iv
        iv_left = np.roll(layout.iv, 1, axis=0)
        dx_right = np.broadcast_to(dx[:, None], shape)
        dx_left = np.broadcast_to(np.roll(dx, 1)[:, None], shape)

        right_ok = iv_right >= 0
        left_ok = iv_left >= 0
        dxu = 0.5 * (np.roll(dx, 1) + dx)
        dist = np.ones(shape)
        dist = np.where(right_ok & left_ok, np.broadcast_to(dxu[:, None], shape), dist)
        dist = np.where(right_ok & ~left_ok, 0.5 * dx_right, dist)
        dist = np.where(left_ok & ~right_ok, 0.5 * dx_left, dist)

        self.iv_right = iv_right
        self.iv_left = iv_left
        self.cv_right = np.where(right_ok, 1.0 / dist, 0.0)
        self.cv_left = np.where(left_ok, -1.0 / dist, 0.0)

    def pairs(self):
        """(係数配列, 索引配列) の 4 組を返す."""
        return (
            (self.cu_above, self.iu_above),
            (self.cu_below, self.iu_below),
            (self.cv_right, self.iv_right),
            (self.cv_left, self.iv_left),
        )


def build_stokes_system(
    grid: ChannelGrid, mu: np.ndarray, G: float, p_pin_value: float = 0.0
) -> tuple[sp.csc_matrix, np.ndarray, StokesLayout, float]:
    """断面内 Stokes の鞍点系 [A −αBᵀ; αB 0] と右辺を組む.

    勾配ブロックは発散ブロック B の厳密な転置に −1 を掛けたものを使う。
    別々に組むと離散的な発散ゼロが機械精度で成立しなくなる。

    **圧力スケーリング α**: 粘性ブロックの要素は μ·dy/dx 〜 μ·dx/dy のオーダー
    （40mm 機で 4×10¹ 〜 5×10⁴）、発散ブロックは dx, dy のオーダー（10⁻⁴）で、
    そのまま組むと 8 桁の開きが条件数に直結し、速度の相対精度が 10⁻¹⁰ 程度まで
    落ちる。未知数を p̃ = α·p' と置き直し連続式を α 倍すれば両ブロックの大きさが
    揃う。α は実測値 max|A_vel| / max|B| から決める（無次元化の自動版）。
    戻り値の α で p' を掛け戻すこと。

    圧力の定数自由度は、最初の流体セルの連続式を p' = p_pin_value/α で置き換えて
    消す（連続式は 1 本が冗長なので、置き換えても情報は失われない）。
    """
    layout = StokesLayout(grid)
    nx, ny = layout.nx, layout.ny
    dx, dy = grid.dx, grid.dy
    dxu = 0.5 * (np.roll(dx, 1) + dx)
    dyv = np.zeros(ny + 1)
    dyv[1:ny] = 0.5 * (dy[:-1] + dy[1:])

    mu_node = _node_viscosity(grid, mu)
    node = _NodeShear(layout)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros(layout.n_total)

    def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
        rows.append(np.asarray(r, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(np.asarray(v, dtype=np.float64))

    def emit_node(row: np.ndarray, ni: np.ndarray, nj: np.ndarray, factor: np.ndarray) -> None:
        """節点せん断項 factor·τ/μ_node を row に加算する（μ_node は factor に含める）."""
        for coef, idx in node.pairs():
            c = coef[ni, nj] * factor
            k = idx[ni, nj]
            m = k >= 0
            if m.any():
                add(row[m], k[m], c[m])
        np.add.at(rhs, row, -node.const[ni, nj] * factor)

    # ---------------- x 運動量 ----------------
    ui, uj = np.nonzero(layout.u_ok)
    r_u = layout.iu[ui, uj]
    ie = ui  # 東側セル = i
    iw = (ui - 1) % nx  # 西側セル = i-1

    c_east = 2.0 * mu[ie, uj] * dy[uj] / dx[ie]
    c_west = 2.0 * mu[iw, uj] * dy[uj] / dx[iw]
    add(r_u, r_u, c_east + c_west)
    k_e = layout.iu[(ui + 1) % nx, uj]
    m = k_e >= 0
    add(r_u[m], k_e[m], -c_east[m])
    k_w = layout.iu[(ui - 1) % nx, uj]
    m = k_w >= 0
    add(r_u[m], k_w[m], -c_west[m])

    emit_node(r_u, ui, uj + 1, -dxu[ui] * mu_node[ui, uj + 1])
    emit_node(r_u, ui, uj, +dxu[ui] * mu_node[ui, uj])

    # 体積力 f_x = −β（右辺へ）
    beta = grid.spec.beta(G)
    np.add.at(rhs, r_u, -beta * dxu[ui] * dy[uj])

    # ---------------- y 運動量 ----------------
    vi, vj = np.nonzero(layout.v_ok)
    r_v = layout.iv[vi, vj]
    j_above = vj  # 上のセル行
    j_below = vj - 1

    c_north = 2.0 * mu[vi, j_above] * dx[vi] / dy[j_above]
    c_south = 2.0 * mu[vi, j_below] * dx[vi] / dy[j_below]
    add(r_v, r_v, c_north + c_south)
    k_n = layout.iv[vi, vj + 1]
    m = k_n >= 0
    add(r_v[m], k_n[m], -c_north[m])
    k_s = layout.iv[vi, vj - 1]
    m = k_s >= 0
    add(r_v[m], k_s[m], -c_south[m])

    ie_node = (vi + 1) % nx
    emit_node(r_v, ie_node, vj, -dyv[vj] * mu_node[ie_node, vj])
    emit_node(r_v, vi, vj, +dyv[vj] * mu_node[vi, vj])

    # ---------------- 連続式 B と勾配 −Bᵀ ----------------
    pi_, pj = np.nonzero(layout.fluid)
    r_p_local = layout.ip[pi_, pj] - layout.p_offset

    b_rows: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    b_vals: list[np.ndarray] = []

    def add_b(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
        b_rows.append(np.asarray(r, dtype=np.int64))
        b_cols.append(np.asarray(c, dtype=np.int64))
        b_vals.append(np.asarray(v, dtype=np.float64))

    k = layout.iu[pi_, pj]  # 西面
    m = k >= 0
    add_b(r_p_local[m], k[m], -dy[pj][m])
    k = layout.iu[(pi_ + 1) % nx, pj]  # 東面
    m = k >= 0
    add_b(r_p_local[m], k[m], dy[pj][m])
    k = layout.iv[pi_, pj]  # 南面
    m = k >= 0
    add_b(r_p_local[m], k[m], -dx[pi_][m])
    k = layout.iv[pi_, pj + 1]  # 北面
    m = k >= 0
    add_b(r_p_local[m], k[m], dx[pi_][m])

    br = np.concatenate(b_rows)
    bc = np.concatenate(b_cols)
    bv = np.concatenate(b_vals)

    # 圧力スケーリング: 粘性ブロックと発散ブロックの大きさを揃える
    a_max = max(float(np.max(np.abs(np.concatenate(vals)))), 1e-300)
    b_max = max(float(np.max(np.abs(bv))), 1e-300)
    alpha = a_max / b_max

    # 勾配ブロックは B の厳密な転置 × (−α)
    add(bc, layout.p_offset + br, -alpha * bv)
    # 発散ブロック × α（ピン留めする行だけ除く）
    keep = br != 0
    add(layout.p_offset + br[keep], bc[keep], alpha * bv[keep])
    # ピン行: p' = p_pin_value / α
    add(np.array([layout.p_offset]), np.array([layout.p_offset]), np.array([a_max]))
    rhs[layout.p_offset] = a_max * p_pin_value / alpha

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(layout.n_total, layout.n_total),
    ).tocsc()
    return A, rhs, layout, alpha


class CrossChannelStokesProcess(SolverProcess["CrossChannelInput", "CrossChannelResult"]):
    """断面内 2D Stokes を MAC 千鳥格子 + 鞍点系の疎直接解で解く."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="CrossChannelStokes",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: CrossChannelInput) -> CrossChannelResult:
        """鞍点系を組んで直接解き、速度・圧力・流れ関数を返す."""
        grid = input_data.grid
        mu = np.asarray(input_data.mu, dtype=np.float64)
        nx, ny = grid.nx, grid.ny
        if mu.shape != (nx, ny):
            msg = f"mu の形状が格子と不一致: {mu.shape} != {(nx, ny)}"
            raise ValueError(msg)
        if np.any(mu[~grid.solid] <= 0.0):
            msg = "流体セルの粘度に 0 以下の値が含まれる"
            raise ValueError(msg)

        A, rhs, layout, alpha = build_stokes_system(grid, mu, input_data.G, input_data.p_pin_value)
        try:
            lu = spla.splu(A)
        except RuntimeError as exc:
            msg = (
                "断面内 Stokes の LU 分解に失敗した"
                f"（未知数 {layout.n_total}: u={layout.nu}, v={layout.nv}, p={layout.npr}）"
            )
            raise RuntimeError(msg) from exc
        sol = lu.solve(rhs)

        u_face = np.zeros((nx, ny))
        u_face[layout.u_ok] = sol[layout.iu[layout.u_ok]]
        v_face = np.zeros((nx, ny + 1))
        v_face[layout.v_ok] = sol[layout.iv[layout.v_ok]]
        p = np.zeros((nx, ny))
        p[layout.fluid] = alpha * sol[layout.ip[layout.fluid]]

        u = 0.5 * (u_face + np.roll(u_face, -1, axis=0))
        v = 0.5 * (v_face[:, :-1] + v_face[:, 1:])
        u[grid.solid] = 0.0
        v[grid.solid] = 0.0

        flux = (np.roll(u_face, -1, axis=0) - u_face) * grid.dy[None, :] + (
            v_face[:, 1:] - v_face[:, :-1]
        ) * grid.dx[:, None]
        cell_area = grid.dx[:, None] * grid.dy[None, :]
        ref_rate = abs(grid.spec.u_barrel) / grid.spec.H
        div = np.abs(flux / cell_area)[~grid.solid]
        div_max = float(div.max() / ref_rate) if div.size else 0.0

        psi = self._stream_function(u_face, v_face, grid)
        ref_psi = abs(grid.spec.u_barrel) * grid.spec.H
        psi_periodicity = float(np.max(np.abs(psi[nx, :] - psi[0, :])) / ref_psi)

        return CrossChannelResult(
            u=u,
            v=v,
            u_face=u_face,
            v_face=v_face,
            p=p,
            psi=psi,
            div_max=div_max,
            psi_periodicity=psi_periodicity,
        )

    @staticmethod
    def _stream_function(u_face: np.ndarray, v_face: np.ndarray, grid: ChannelGrid) -> np.ndarray:
        """面流束を積分して節点流れ関数 ψ (nx+1, ny+1) を作る.

        ψ[i,j+1] − ψ[i,j] = u_face[i,j]·dy[j]
        ψ[i+1,j] − ψ[i,j] = −v_face[i,j]·dx[i]

        この構成なら双一次補間した (∂ψ/∂y, −∂ψ/∂x) がセル内で厳密に発散ゼロになる。
        単方向にしか積分していないが、離散的な発散ゼロが成り立っていれば
        もう一方の経路と一致する（psi_periodicity で確認する）。
        """
        col0 = np.concatenate([[0.0], np.cumsum(u_face[0, :] * grid.dy)])
        shift = np.concatenate(
            [
                np.zeros((1, grid.ny + 1)),
                np.cumsum(v_face * grid.dx[:, None], axis=0),
            ]
        )
        return col0[None, :] - shift
