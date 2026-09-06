"""非構造メッシュの滞留時間分布（RTD）を集計する PostProcess.

:class:`~xkep_cae_fluid.post.tracking.ParticleTrackFVMProcess` の結果から
E(t)（滞留時間の確率密度）と F(t)（その累積）、および経路積分スカラー
（累積せん断ひずみ ∫γ̇ dt など）の分布を作る。押出では「どれだけ揃った熱・
せん断履歴を与えられるか」が品質を決めるので、分布の広がり t_p90/t_p10 と
累積せん断ひずみが主指標になる。

**厳密関係 ⟨t⟩ = 体積 ÷ 流束 が最も鋭い検査。**
定常・非圧縮なら、追跡した領域の体積を通過流束で割った値が流束重み付き平均
滞留時間に一致する。再構成場の誤り・脱出時刻の内挿ミス・種まき重みの誤りを
**同時に**捕まえるので、文献の RTD 曲線との目視比較よりはるかに強い。
理論値は :attr:`~xkep_cae_fluid.post.tracking.ParticleTrackFVMResult.t_mean_theory`
に入っている。

構造格子の展開チャネル専用版は
:class:`~xkep_cae_fluid.extruder.rtd.RTDProcess`。こちらは非構造メッシュ用で、
経路積分するスカラーを呼び出し側が自由に決められる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.post.statistics import weighted_ecdf, weighted_quantile
from xkep_cae_fluid.post.tracking import ParticleTrackFVMResult

__all__ = [
    "ResidenceTimeInput",
    "ResidenceTimeProcess",
    "ResidenceTimeResult",
]

_QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ResidenceTimeInput:
    """RTD 集計の入力.

    Parameters
    ----------
    track : ParticleTrackFVMResult
        粒子追跡の結果
    n_bins : int
        E(t) のヒストグラム区間数
    rate_scalars : tuple[str, ...]
        時間平均（∫s dt / t）で見るスカラー名。混合指数のように「経路に沿った
        平均値」に意味がある量に使う。ここに入れない量は積算値のまま扱う
    """

    track: ParticleTrackFVMResult
    n_bins: int = 200
    rate_scalars: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResidenceTimeResult:
    """滞留時間分布と経路積分スカラーの統計.

    Parameters
    ----------
    t_edges : np.ndarray
        (n_bins+1,) ヒストグラムの区間端 [s]
    E : np.ndarray
        (n_bins,) 滞留時間の確率密度 [1/s]。∫E dt = 1
    F : np.ndarray
        (n_bins+1,) 累積分布 [-]
    t_ecdf, F_ecdf : np.ndarray
        ビン幅に依存しない重み付き経験分布（昇順の滞留時間と累積割合）
    t_mean, t_mean_theory : float
        流束重み付き平均滞留時間と理論値 [s]
    t_min, t_p10, t_p50, t_p90 : float
        最短・10/50/90 パーセンタイル [s]
    spread : float
        t_p90 / t_p10。1 に近いほど揃った履歴
    scalar_mean : dict[str, float]
        経路積分スカラーの流束重み付き平均
    scalar_quantiles : dict[str, tuple[float, float, float]]
        同じく 10/50/90 パーセンタイル
    escaped_fraction : float
        脱出した粒子の重み割合
    extrapolated_weight_fraction : float
        進行率から外挿して閉じた粒子の重み割合
    unresolved_weight_fraction : float
        脱出も外挿もできなかった粒子の重み割合。**ここが大きい結果は信用できない**
    exit_weight_fraction : dict[str, float]
        流出した境界パッチごとの重み割合
    """

    t_edges: np.ndarray
    E: np.ndarray
    F: np.ndarray
    t_ecdf: np.ndarray
    F_ecdf: np.ndarray
    t_mean: float
    t_mean_theory: float
    t_min: float
    t_p10: float
    t_p50: float
    t_p90: float
    spread: float
    scalar_mean: dict[str, float] = field(default_factory=dict)
    scalar_quantiles: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    escaped_fraction: float = 0.0
    extrapolated_weight_fraction: float = 0.0
    unresolved_weight_fraction: float = 0.0
    exit_weight_fraction: dict[str, float] = field(default_factory=dict)


class ResidenceTimeProcess(PostProcess["ResidenceTimeInput", "ResidenceTimeResult"]):
    """粒子追跡結果から滞留時間分布と経路積分スカラーの統計を作る."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ResidenceTime",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/particle-tracking-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ResidenceTimeInput) -> ResidenceTimeResult:
        """流束重み付きで E(t), F(t) と各スカラーの分位点を出す."""
        inp = input_data
        tr = inp.track
        if inp.n_bins < 1:
            raise ValueError(f"n_bins は 1 以上が必要: {inp.n_bins}")
        unknown = set(inp.rate_scalars) - set(tr.integrals)
        if unknown:
            raise ValueError(f"rate_scalars に未知の名前: {sorted(unknown)}")

        ok = tr.escaped
        if not ok.any():
            raise ValueError("脱出した粒子が 1 つも無い（max_steps か length を見直すこと）")
        total_w = float(tr.weight.sum())
        w = tr.weight[ok]
        t = tr.t_res[ok]
        w_sum = float(w.sum())

        t_edges = np.linspace(float(t.min()), float(t.max()), inp.n_bins + 1)
        hist, _ = np.histogram(t, bins=t_edges, weights=w)
        density = hist / (w_sum * np.maximum(np.diff(t_edges), 1e-300))
        cumulative = np.concatenate([[0.0], np.cumsum(hist) / w_sum])
        t_ecdf, f_ecdf = weighted_ecdf(t, w)
        t_q = weighted_quantile(t, w, np.array(_QUANTILES))

        scalar_mean: dict[str, float] = {}
        scalar_q: dict[str, tuple[float, float, float]] = {}
        for name, values in tr.integrals.items():
            v = values[ok]
            if name in inp.rate_scalars:
                v = np.where(t > 0.0, v / np.where(t > 0.0, t, 1.0), 0.0)
            scalar_mean[name] = float(np.sum(w * v) / w_sum)
            q = weighted_quantile(v, w, np.array(_QUANTILES))
            scalar_q[name] = (float(q[0]), float(q[1]), float(q[2]))

        exit_fraction: dict[str, float] = {}
        for k, name in enumerate(tr.patch_names):
            share = float(tr.weight[tr.exit_patch == k].sum() / total_w)
            if share > 0.0:
                exit_fraction[name] = share

        return ResidenceTimeResult(
            t_edges=t_edges,
            E=density,
            F=cumulative,
            t_ecdf=t_ecdf,
            F_ecdf=f_ecdf,
            t_mean=float(np.sum(w * t) / w_sum),
            t_mean_theory=float(tr.t_mean_theory),
            t_min=float(t.min()),
            t_p10=float(t_q[0]),
            t_p50=float(t_q[1]),
            t_p90=float(t_q[2]),
            spread=float(t_q[2] / t_q[0]) if t_q[0] > 0.0 else float("inf"),
            scalar_mean=scalar_mean,
            scalar_quantiles=scalar_q,
            escaped_fraction=float(w_sum / total_w),
            extrapolated_weight_fraction=float(tr.weight[tr.extrapolated].sum() / total_w),
            unresolved_weight_fraction=float(tr.weight[~ok].sum() / total_w),
            exit_weight_fraction=exit_fraction,
        )
