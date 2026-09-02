"""粒子追跡 Process（流れ関数補間 + 適応 RK4）.

**速度を直接補間しない。** 節点流れ関数 ψ を双一次補間して

    u = ∂ψ/∂y,   v = −∂ψ/∂x

とすれば、セル内で ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y − ∂²ψ/∂x∂y ≡ 0 が**恒等的に**成り立つ。
速度を直接双一次補間すると離散的な発散ゼロが壊れ、粒子が渦の中心に落ち込んだり
壁に貼り付いたりして RTD の裾が偽物になる。

**追跡座標は z（下流方向）ではなく ζ（軸方向）を使う。**
x 周期の同一視は (x, z) ~ (x − W_t, z + L_turn) なので z は跳ぶが、
ζ = x cosφ + z sinφ は W_t cosφ = L_turn sinφ の相殺で**不変**になる。

    dx/dt = u,   dy/dt = v,   dζ/dt = u cosφ + w sinφ

計量部の長さは実機でも軸方向で測るので、ζ >= z_axial を脱出条件にすれば
跳びの記帳ミスが入り込む余地が無い。x は W_t で折り返すだけ（ζ は変えない）。

**種まきは決定論的な流束重み。** 流体セル 1 個につき 1 粒子を中心に置き、
ζ=0 面を通る体積流束を重みとして持たせる。ζ=const 面の面積要素は
dA_plane = dx·dy/sinφ、流束密度は u cosφ + w sinφ なので

    weight = max(u·cotφ + w, 0) · dx · dy

重みの総和は Q_axial に一致する（負の流束を落とした分だけ僅かに上振れする）。
乱数配置だと 20000 個でも相対標準誤差 0.7% が乗るが、決定論的な求積なら
モンテカルロ誤差がゼロになり、残るのは O(h²) の空間求積誤差だけになる。
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.extruder.data import (
    ChannelGrid,
    ExtruderFlowResult,
    ParticleTrackInput,
    ParticleTrackResult,
)
from xkep_cae_fluid.extruder.viscosity import mixing_index

_EPS = 1e-30

_DT_MAX_FRACTION = 0.02
"""1 ステップの時間刻みの上限（理論平均滞留時間に対する比）.

淀み点（u ≈ v ≈ 0）では dt = cfl·dx/|u| が発散し、1 ステップで 10^13 s といった
時間を踏んでしまう。位置は CFL で守られるが、経過時間が壊れて ⟨t⟩ が 10 桁単位で
飛ぶ（実測）。理論平均滞留時間 t_ref = z_axial·A_free/(sinφ·Σweight) は種まき時点で
分かるので、これを基準に上限を置く。平均的な粒子は最低 50 ステップ、
max_steps=50000 なら 1000·t_ref まで追える。
"""

_EXTRAPOLATION_MIN_PROGRESS = 0.1
"""外挿を許す最小の軸方向進行率（z_axial に対する比）.

これ未満しか進んでいない粒子は「定常周回で ζ が t に比例する」という外挿の
前提を満たしていないので外挿せず、未解決として報告する。外挿係数は最大 10 倍。
"""


def _node_field(
    cell: np.ndarray,
    grid: ChannelGrid,
    *,
    zero_at_wall: bool,
    bottom: float | None = None,
    top: float | None = None,
) -> np.ndarray:
    """セル中心場を節点 (nx+1, ny+1) へ移す.

    zero_at_wall=True: 固体に接する節点を 0 にする（速度のように壁で 0 になる量）。
    zero_at_wall=False: 流体セルだけの平均を取る（せん断速度のように壁でも 0 でない量）。
    bottom / top: y=0（スクリュー根元）と y=H（バレル）の節点に入れる既知値。

    **境界値を入れないと流束が系統的にずれる。** 根元の節点に隣接セルの w を
    そのまま置くと、壁で 0 になるべきところが正の値になり、双一次補間した w の
    面積分が真値より大きくなる。滞留時間が系統的に短く出る原因になった。
    """
    nx, ny = grid.nx, grid.ny
    acc = np.zeros((nx, ny + 1))
    cnt = np.zeros((nx, ny + 1))
    touches_solid = np.zeros((nx, ny + 1), dtype=bool)
    fluid = ~grid.solid

    for di in (0, -1):
        ii = (np.arange(nx) + di) % nx
        for dj in (0, -1):
            jj = np.arange(ny + 1) + dj
            ok = (jj >= 0) & (jj <= ny - 1)
            sub = cell[np.ix_(ii, jj[ok])]
            sub_fluid = fluid[np.ix_(ii, jj[ok])]
            acc[:, ok] += np.where(sub_fluid, sub, 0.0)
            cnt[:, ok] += sub_fluid
            touches_solid[:, ok] |= ~sub_fluid

    node = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), 0.0)
    if zero_at_wall:
        node = np.where(touches_solid, 0.0, node)
    if bottom is not None:
        node[:, 0] = bottom
    if top is not None:
        node[:, ny] = top
    return np.concatenate([node, node[:1, :]], axis=0)


def _locate(coord: np.ndarray, nodes: np.ndarray, n: int) -> np.ndarray:
    """座標からセル添字を返す（節点配列 nodes は長さ n+1、単調増加）."""
    idx = np.searchsorted(nodes, coord, side="right") - 1
    return np.clip(idx, 0, n - 1)


class _Interpolator:
    """ψ から発散ゼロな (u, v) を、節点場から w・γ̇・λ を双一次補間する."""

    def __init__(self, flow: ExtruderFlowResult) -> None:
        grid = flow.grid
        self.grid = grid
        self.dx = grid.dx
        self.dy = grid.dy
        self.x_node = np.concatenate([[0.0], np.cumsum(grid.dx)])
        self.y_node = np.concatenate([[0.0], np.cumsum(grid.dy)])
        self.psi = flow.psi
        self.w_node = _node_field(
            flow.w, grid, zero_at_wall=True, bottom=0.0, top=grid.spec.w_barrel
        )
        self.g_node = _node_field(flow.gamma_dot, grid, zero_at_wall=False)
        lam = mixing_index(flow.u, flow.v, flow.w, grid)
        self.lam_node = _node_field(lam, grid, zero_at_wall=False)

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = _locate(x, self.x_node, self.grid.nx)
        j = _locate(y, self.y_node, self.grid.ny)
        return i, j

    def velocity(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, w) を返す。u, v は ψ の双一次補間から作るので発散ゼロ."""
        i, j = self.cell_of(x, y)
        dxi = self.dx[i]
        dyj = self.dy[j]
        s = (x - self.x_node[i]) / dxi
        t = (y - self.y_node[j]) / dyj

        p00 = self.psi[i, j]
        p10 = self.psi[i + 1, j]
        p01 = self.psi[i, j + 1]
        p11 = self.psi[i + 1, j + 1]

        u = ((p01 - p00) * (1.0 - s) + (p11 - p10) * s) / dyj
        v = -((p10 - p00) * (1.0 - t) + (p11 - p01) * t) / dxi
        w = self._bilinear(self.w_node, i, j, s, t)
        return u, v, w

    def scalars(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(γ̇, λ) を返す."""
        i, j = self.cell_of(x, y)
        s = (x - self.x_node[i]) / self.dx[i]
        t = (y - self.y_node[j]) / self.dy[j]
        return (
            self._bilinear(self.g_node, i, j, s, t),
            self._bilinear(self.lam_node, i, j, s, t),
        )

    @staticmethod
    def _bilinear(
        node: np.ndarray, i: np.ndarray, j: np.ndarray, s: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        return (
            node[i, j] * (1.0 - s) * (1.0 - t)
            + node[i + 1, j] * s * (1.0 - t)
            + node[i, j + 1] * (1.0 - s) * t
            + node[i + 1, j + 1] * s * t
        )


class ParticleTrackerProcess(PostProcess["ParticleTrackInput", "ParticleTrackResult"]):
    """流れ関数補間 + 適応 RK4 による粒子追跡.

    全粒子を同時に（それぞれ固有の dt で）進める。粒子ごとに時刻がずれるが、
    定常流なので各粒子の経過時間だけが意味を持ち、同期は不要。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ParticleTracker",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ParticleTrackInput) -> ParticleTrackResult:
        """流束重み付きで種をまき、ζ >= z_axial まで追跡する."""
        inp = input_data
        flow = inp.flow
        grid = flow.grid
        spec = grid.spec
        if inp.z_axial <= 0.0:
            msg = f"軸方向長さ z_axial は正が必要: {inp.z_axial}"
            raise ValueError(msg)
        if inp.stride < 1:
            msg = f"stride は 1 以上が必要: {inp.stride}"
            raise ValueError(msg)

        interp = _Interpolator(flow)
        cos_phi = math.cos(spec.phi)
        sin_phi = math.sin(spec.phi)
        cot_phi = cos_phi / sin_phi
        w_t = spec.W_t

        x0, y0, weight = self._seed(interp, flow, cot_phi, inp.stride)
        n = x0.shape[0]
        if n == 0:
            msg = "軸方向流束が正の流体セルが 1 つも無い"
            raise ValueError(msg)

        x = x0.copy()
        y = y0.copy()
        zeta = np.zeros(n)
        t = np.zeros(n)
        gamma = np.zeros(n)
        lam_int = np.zeros(n)
        wraps = np.zeros(n, dtype=np.int64)
        steps = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        escaped = np.zeros(n, dtype=bool)

        y_lo = 1e-12 * spec.H
        y_hi = spec.H - y_lo

        # 理論平均滞留時間から 1 ステップの時間刻み上限を決める
        t_ref = inp.z_axial * grid.area_free / (sin_phi * max(weight.sum(), _EPS))
        dt_max = _DT_MAX_FRACTION * t_ref

        for _ in range(inp.max_steps):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            xa, ya = x[idx], y[idx]

            u1, v1, w1 = interp.velocity(xa, ya)
            i, j = interp.cell_of(xa, ya)
            speed_x = np.abs(u1) + _EPS
            speed_y = np.abs(v1) + _EPS
            dt = inp.cfl * np.minimum(grid.dx[i] / speed_x, grid.dy[j] / speed_y)
            # ζ の行き過ぎを抑える（脱出時刻の線形内挿を効かせるため）
            dz1 = u1 * cos_phi + w1 * sin_phi
            remain = inp.z_axial - zeta[idx]
            forward = dz1 > 0.0
            dt = np.where(forward, np.minimum(dt, 2.0 * remain / np.maximum(dz1, _EPS)), dt)
            dt = np.minimum(dt, dt_max)

            xk, yk, zk, gk, lk = self._rk4(interp, xa, ya, dt, cos_phi, sin_phi)

            # 固体に入ったら刻みを半分にして退避（ψ 補間なら本来入らない）
            for _retry in range(6):
                ii, jj = interp.cell_of(np.mod(xk, w_t), np.clip(yk, y_lo, y_hi))
                bad = grid.solid[ii, jj]
                if not bad.any():
                    break
                dt = np.where(bad, dt * 0.5, dt)
                xk, yk, zk, gk, lk = self._rk4(interp, xa, ya, dt, cos_phi, sin_phi)

            zeta_new = zeta[idx] + zk
            crossed = zeta_new >= inp.z_axial
            frac = np.ones_like(dt)
            moving = crossed & (zk > 0.0)
            frac[moving] = (inp.z_axial - zeta[idx][moving]) / zk[moving]
            frac = np.clip(frac, 0.0, 1.0)

            x_new = xa + (xk - xa) * frac
            y_new = ya + (yk - ya) * frac
            wrap_delta = np.floor(x_new / w_t).astype(np.int64)
            x[idx] = np.mod(x_new, w_t)
            y[idx] = np.clip(y_new, y_lo, y_hi)
            wraps[idx] += wrap_delta
            zeta[idx] = np.where(crossed, inp.z_axial, zeta_new)
            t[idx] += dt * frac
            gamma[idx] += gk * frac
            lam_int[idx] += lk * frac
            steps[idx] += 1

            done = idx[crossed]
            escaped[done] = True
            alive[done] = False

        # ステップ上限に達した粒子を軸方向の進行率から外挿して閉じる。
        # バレル面では u_barrel·cosφ + w_barrel·sinφ = −V sinφcosφ + V cosφsinφ = 0
        # なので、その直下の材料は周回するだけで軸方向にほとんど進まず、
        # 滞留時間が y→H で発散する（根元側も同様）。重みは (H−y) で消えるので
        # 積分は収束するが、有限ステップでは閉じない。定常な周回軌道では
        # ζ が t に比例するので、t_res = t·z_axial/ζ が良い近似になる。
        #
        # **外挿には歯止めが要る。** ζ ≈ 0 の粒子（フライト隅の淀みや二次渦に
        # 捕まったもの）は「ζ が t に比例する定常周回」という前提を満たしておらず、
        # 係数 z_axial/ζ が発散して ⟨t⟩ を 10 桁単位で壊す（実測）。
        # 進行率が信用できる粒子だけ外挿し、残りは未解決として正直に報告する。
        # 重み×滞留時間の寄与は淀み近傍で O(h²) で消えるので、除外しても収束する。
        extrapolated = np.zeros(n, dtype=bool)
        zeta_min = _EXTRAPOLATION_MIN_PROGRESS * inp.z_axial
        stuck = alive & (zeta >= zeta_min)
        if stuck.any():
            factor = inp.z_axial / zeta[stuck]
            t[stuck] *= factor
            gamma[stuck] *= factor
            lam_int[stuck] *= factor
            extrapolated[stuck] = True
            escaped[stuck] = True

        lam_mean = np.where(t > 0.0, lam_int / np.where(t > 0.0, t, 1.0), 0.0)
        return ParticleTrackResult(
            weight=weight,
            t_res=t,
            gamma_total=gamma,
            lambda_mean=lam_mean,
            n_wraps=wraps,
            x0=x0,
            y0=y0,
            x=x,
            y=y,
            escaped=escaped,
            extrapolated=extrapolated,
            n_steps=steps,
        )

    @staticmethod
    def _seed(
        interp: _Interpolator, flow: ExtruderFlowResult, cot_phi: float, stride: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """流体セル中心に種をまき、ζ=0 面を通る体積流束を重みにする.

        速度は**追跡に使うのと同じ補間場**から取る。双一次場はセル中心値が
        セル平均に厳密に一致する（s=t=0.5 で 4 節点の平均になる）ので、
        1 点求積でもセル積分が厳密になり、⟨t⟩ = V/Q の関係が補間場に対して
        厳密に成立する。生のセル中心値で重みを作ると場が食い違って数 % ずれる。
        """
        grid = flow.grid
        xc = np.broadcast_to(grid.xc[:, None], grid.solid.shape)
        yc = np.broadcast_to(grid.yc[None, :], grid.solid.shape)
        u_i, _v_i, w_i = interp.velocity(xc.ravel(), yc.ravel())
        u_i = u_i.reshape(grid.solid.shape)
        w_i = w_i.reshape(grid.solid.shape)

        area = grid.dx[:, None] * grid.dy[None, :]
        flux = (cot_phi * u_i + w_i) * area
        keep = np.zeros(grid.solid.shape, dtype=bool)
        keep[::stride, ::stride] = True
        mask = (~grid.solid) & (flux > 0.0) & keep

        ii, jj = np.nonzero(mask)
        # 間引いた分は重みを stride² 倍して総流束を保つ
        weight = flux[ii, jj] * float(stride) ** 2
        return grid.xc[ii].copy(), grid.yc[jj].copy(), weight

    @staticmethod
    def _rk4(
        interp: _Interpolator,
        x: np.ndarray,
        y: np.ndarray,
        dt: np.ndarray,
        cos_phi: float,
        sin_phi: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """RK4 で 1 ステップ進め、(x, y, Δζ, Δγ, Δ∫λdt) を返す."""
        grid = interp.grid
        w_t = grid.spec.W_t
        h = grid.spec.H
        y_lo, y_hi = 0.0, h

        def sample(xx: np.ndarray, yy: np.ndarray):
            xw = np.mod(xx, w_t)
            yw = np.clip(yy, y_lo, y_hi)
            u, v, w = interp.velocity(xw, yw)
            g, lam = interp.scalars(xw, yw)
            return u, v, u * cos_phi + w * sin_phi, g, lam

        k1 = sample(x, y)
        k2 = sample(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1])
        k3 = sample(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1])
        k4 = sample(x + dt * k3[0], y + dt * k3[1])

        def combine(m: int) -> np.ndarray:
            return (k1[m] + 2.0 * k2[m] + 2.0 * k3[m] + k4[m]) / 6.0

        return (
            x + dt * combine(0),
            y + dt * combine(1),
            dt * combine(2),
            dt * combine(3),
            dt * combine(4),
        )
