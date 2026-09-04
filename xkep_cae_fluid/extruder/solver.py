"""押出流れ解析の統合 Process（Picard で粘度結合）.

ニュートン流体では下流方向 w と断面内 (u,v) が完全に分離する。
非ニュートンでは粘度 μ(γ̇) だけを介して結合するので、
「線形解 2 本を回して粘度場を更新する」不動点反復に落ちる。

    μ⁰ = model.viscosity(V/H)                    代表せん断速度で初期化
    繰り返し k:
        w  ← DownChannelFlowProcess(grid, μ^k, G)        （線形・厳密）
        uv ← CrossChannelStokesProcess(grid, μ^k, G)      （線形・厳密）
        γ̇ ← strain_rate(u, v, w, grid)
        μ^{k+1} = (1−ω)·μ^k + ω·model.viscosity(γ̇)
    収束判定: max|μ^{k+1} − μ^k| / max|μ^k| < tol

**Newton を採らない理由**: 見かけ粘度項のヤコビアンは ∂μ/∂γ̇ · ∂γ̇/∂(∇u) を
通じて全成分に密に絡み、組み立てコストが線形解 1 回分を大きく超える。一方
Picard は線形解が毎回厳密なので、反復が粘度場の不動点探索だけに集約される。
せん断減粘（n<1）は粘度の自己安定化が効くので ω=0.5 で十分収束する。
収束しなければ ω を下げ、それでも駄目なら「収束しなかった」と報告すること
（CLAUDE.md の STA2 防止ルール）。

**押出量は Q = ∫∫w dA ではない。** 隙間があると Q はむしろ増える（バレル直下の
w が最大の場所に e×δ の流路が足されるため）。軸方向の一定面 ζ=const を通る流束を
取ると、面は展開平面上で ξ̂ 方向に円周 πD 分伸びるので

    Q_axial = ∫∫(u cosφ + w sinφ) dl dy = cotφ·∫∫u dA + ∫∫w dA

断面内は 2D 非圧縮なのでどの x 面を通る流束も同じ Q_leak であり ∫∫u dA = Q_leak·W_t、
さらに恒等式 cotφ·W_t = L_turn（§2.1.1）から

    Q_axial = Q + L_turn·Q_leak

漏れ 1 単位が材料を L_turn だけ下流方向に戻す、という描像そのもの。
40mm 機の実測で Q は閉チャネル比 +0.3% だが Q_axial は −3.0% になる。
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.slots import StrategySlot
from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
from xkep_cae_fluid.extruder.data import (
    CrossChannelInput,
    DownChannelInput,
    ExtruderFlowInput,
    ExtruderFlowResult,
)
from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.viscosity import ViscosityModelStrategy, strain_rate


class ExtruderFlowProcess(SolverProcess["ExtruderFlowInput", "ExtruderFlowResult"]):
    """幾何生成 → 粘度 Picard → w / (u,v) 求解 → 流量と漏れ量の算出.

    粘度モデルは StrategySlot で注入する（未設定なら AttributeError）。
    ニュートンの場合は 1 回目で粘度が動かないので n_iter=1 で収束する。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ExtruderFlow",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [
        ScrewGeometryProcess,
        DownChannelFlowProcess,
        CrossChannelStokesProcess,
    ]

    viscosity = StrategySlot(ViscosityModelStrategy)

    def process(self, input_data: ExtruderFlowInput) -> ExtruderFlowResult:
        """Picard 反復で粘度場の不動点を求め、速度場・流量・漏れ量を返す."""
        t0 = time.perf_counter()
        inp = input_data
        if not 0.0 < inp.relax_mu <= 1.0:
            msg = f"緩和係数 relax_mu は 0 < ω <= 1 が必要: {inp.relax_mu}"
            raise ValueError(msg)
        model = self.viscosity

        grid = ScrewGeometryProcess().process(inp.spec)
        gamma_ref = inp.spec.V / inp.spec.H
        mu = model.viscosity(np.full((grid.nx, grid.ny), gamma_ref))

        down = DownChannelFlowProcess()
        cross = CrossChannelStokesProcess()
        mu_history: list[float] = []
        converged = False
        n_iter = 0
        w_res = None
        uv = None
        gamma = np.zeros((grid.nx, grid.ny))

        for k in range(1, inp.max_iter + 1):
            n_iter = k
            w_res = down.process(DownChannelInput(grid=grid, mu=mu, G=inp.G))
            uv = cross.process(CrossChannelInput(grid=grid, mu=mu, G=inp.G))
            gamma = strain_rate(uv.u, uv.v, w_res.w, grid)
            mu_star = model.viscosity(gamma)
            mu_new = (1.0 - inp.relax_mu) * mu + inp.relax_mu * mu_star
            scale = max(float(np.max(np.abs(mu))), 1e-30)
            change = float(np.max(np.abs(mu_new - mu)) / scale)
            mu_history.append(change)
            mu = mu_new
            if change < inp.tol:
                converged = True
                break

        assert w_res is not None and uv is not None  # max_iter >= 1 が保証する

        # 漏れ量: フライトランド中央の x 面を通る正味横断流束 [m²/s]
        i_land = grid.nx // 2
        q_leak = float(np.sum(uv.u_face[i_land, :] * grid.dy))
        q_axial = w_res.Q + grid.spec.L_turn * q_leak

        return ExtruderFlowResult(
            grid=grid,
            u=uv.u,
            v=uv.v,
            w=w_res.w,
            u_face=uv.u_face,
            v_face=uv.v_face,
            psi=uv.psi,
            p=uv.p,
            mu=mu,
            gamma_dot=gamma,
            Q=w_res.Q,
            Q_leak=q_leak,
            Q_axial=q_axial,
            converged=converged,
            n_iter=n_iter,
            mu_history=tuple(mu_history),
            div_max=uv.div_max,
            elapsed_seconds=time.perf_counter() - t0,
        )
