"""重み付き分布統計（流束重み付きの分位点・経験分布）.

粒子追跡は「流体セル 1 個につき 1 粒子」しか置かないが、粒子ごとに通過流束を
重みとして持つ。分布統計は全て**流束重み付き**で取らないと、遅い領域
（壁際の薄い層）を過大評価した偽の滞留時間分布になる。
"""

from __future__ import annotations

import numpy as np

__all__ = ["weighted_ecdf", "weighted_quantile"]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """重み付き分位点（線形内挿）.

    values を昇順に並べ、累積重みが q に達する位置を線形内挿で求める。
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0.0:
        msg = "重みの総和が 0 以下"
        raise ValueError(msg)
    # 区間中点の累積割合（重み付き経験分布の標準的な定義）
    cum = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(q, cum, v)


def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """重み付き経験分布（昇順の値と区間中点の累積割合）.

    ヒストグラムの F と違ってビン幅に依存しないので、文献曲線との max|ΔF| の
    比較に使う。:func:`weighted_quantile` と同じ中点流儀なので分位点が逆算で一致する。
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0.0:
        msg = "重みの総和が 0 以下"
        raise ValueError(msg)
    return v, (np.cumsum(w) - 0.5 * w) / total
