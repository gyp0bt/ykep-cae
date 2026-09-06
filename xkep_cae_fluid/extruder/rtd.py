"""滞留時間分布（RTD）と混練性指標の後処理 Process.

E(t) は滞留時間の確率密度、F(t) はその累積。押出では「どれだけ揃った熱・
せん断履歴を与えられるか」が品質を決めるので、分布の広がり t_p90/t_p10 と
累積せん断ひずみ γ = ∫γ̇ dt を主指標にする。

**厳密関係 ⟨t⟩ = z_axial·A_free / (sinφ·Q)。**
(x, y, ζ) 空間で流れは体積保存（∂u/∂x + ∂v/∂y + ∂(dζ/dt)/∂ζ = 0）なので、
ζ=0 から ζ=z_axial までの領域について「体積 ÷ 流束」が流束重み付き平均滞留時間に
一致する。体積は A_free·z_axial、流束は ∫∫(u cosφ + w sinφ)dA = sinφ·Q。
補間誤差・脱出時刻の内挿ミス・種まき重みの誤りを**同時に**捕まえる強い検査で、
文献の RTD 曲線との目視比較よりはるかに鋭い。

分布統計は全て**流束重み付き**で取る。粒子は流体セル 1 個につき 1 個しか
置いていないが、それぞれが ζ=0 面を通る体積流束を重みとして持っているため。
重み付き分位点・経験分布そのものは :mod:`xkep_cae_fluid.post.statistics` にあり、
非構造メッシュ版（:class:`~xkep_cae_fluid.post.rtd.ResidenceTimeProcess`）と共有する。
ここは展開チャネル 2.5D 専用で、``⟨t⟩ = z_axial·A_free/(sinφ·Q)`` の形に特殊化してある。
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.extruder.data import RTDInput, RTDResult
from xkep_cae_fluid.post.statistics import weighted_quantile

__all__ = ["RTDProcess"]


class RTDProcess(PostProcess["RTDInput", "RTDResult"]):
    """粒子追跡結果から滞留時間分布と混練性指標を集計する."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="RTD",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: RTDInput) -> RTDResult:
        """流束重み付きで E(t), F(t) と混練性指標を作る."""
        inp = input_data
        tr = inp.track
        grid = inp.flow.grid
        spec = grid.spec

        ok = tr.escaped
        if not ok.any():
            msg = "脱出した粒子が 1 つも無い（max_steps か z_axial を見直すこと）"
            raise ValueError(msg)

        w = tr.weight[ok]
        t = tr.t_res[ok]
        gamma = tr.gamma_total[ok]
        lam = tr.lambda_mean[ok]
        total_w = float(tr.weight.sum())

        t_mean = float(np.sum(w * t) / w.sum())
        flux = math.sin(spec.phi) * total_w
        t_theory = inp.z_axial * grid.area_free / flux

        t_edges = np.linspace(float(t.min()), float(t.max()), inp.n_bins + 1)
        hist, _ = np.histogram(t, bins=t_edges, weights=w)
        widths = np.diff(t_edges)
        density = hist / (w.sum() * widths)
        cumulative = np.concatenate([[0.0], np.cumsum(hist) / w.sum()])

        t_q = weighted_quantile(t, w, np.array([0.1, 0.5, 0.9]))
        g_q = weighted_quantile(gamma, w, np.array([0.1, 0.5, 0.9]))

        return RTDResult(
            t_edges=t_edges,
            E=density,
            F=cumulative,
            t_mean=t_mean,
            t_mean_theory=float(t_theory),
            t_min=float(t.min()),
            t_p10=float(t_q[0]),
            t_p50=float(t_q[1]),
            t_p90=float(t_q[2]),
            spread=float(t_q[2] / t_q[0]),
            gamma_mean=float(np.sum(w * gamma) / w.sum()),
            gamma_p10=float(g_q[0]),
            gamma_p50=float(g_q[1]),
            gamma_p90=float(g_q[2]),
            lambda_mean=float(np.sum(w * lam) / w.sum()),
            extrapolated_weight_fraction=float(tr.weight[tr.extrapolated].sum() / total_w),
            unresolved_weight_fraction=float(tr.weight[~ok].sum() / total_w),
        )
