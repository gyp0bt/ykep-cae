"""緩和係数の適応的調整（構造格子版 / 非構造版で共有）.

SIMPLE 系の外部反復で、収束判定に使う最大残差の推移から速度・圧力の緩和係数を
動かす。構造格子の :class:`~xkep_cae_fluid.natural_convection.solver.NaturalConvectionFDMProcess`
（status-16 の ``adaptive_relaxation``）と非構造の
:class:`~xkep_cae_fluid.incompressible.solver.NavierStokesFVMProcess` が同じ規則を使う。

- 残差が前回の ``improve_ratio`` 倍未満に減った → 緩和を積極化（α_u, α_p を ``grow`` 倍、上限あり）
- 残差が前回の ``worsen_ratio`` 倍を超えて増えた → 緩和を保守化（``shrink`` 倍、下限あり）
- 残差がこれまでの最小値 ``min_res`` の ``stall_ratio`` 倍を超えた（1 反復あたりの増え方は小さいが
  じわじわ発散している）→ 保守化。status-16 の規則（前回比だけ）では捕まえられなかった型
- それ以外・前回残差が無い → 変えない
- ``simple_cap``: SIMPLE では Patankar の目安 α_p ≤ 1 − α_u を超えない
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelaxationBounds:
    """適応緩和の上下限と増減率."""

    alpha_u_max: float = 0.9
    alpha_u_min: float = 0.1
    alpha_p_max: float = 0.5
    alpha_p_min: float = 0.05
    grow: float = 1.1
    shrink: float = 0.8
    improve_ratio: float = 0.8
    worsen_ratio: float = 1.2
    stall_ratio: float = 5.0


DEFAULT_RELAXATION_BOUNDS = RelaxationBounds()


def adapt_relaxation_factors(
    alpha_u: float,
    alpha_p: float,
    max_res: float,
    prev_max_res: float,
    bounds: RelaxationBounds = DEFAULT_RELAXATION_BOUNDS,
    *,
    min_res: float | None = None,
    simple_cap: bool = False,
) -> tuple[float, float]:
    """残差の推移から (α_u, α_p) を返す。変更が無ければ入力をそのまま返す.

    Parameters
    ----------
    alpha_u, alpha_p : float
        現在の速度・圧力の緩和係数
    max_res : float
        今回の収束判定残差（運動量・質量の最大）
    prev_max_res : float
        前回の同残差（初回は 0 → 変更しない）
    min_res : float | None
        これまでの最小残差（前回の保守化以降）。``max_res > stall_ratio × min_res`` なら
        前回比が小さくても保守化する。None なら判定しない
    simple_cap : bool
        True なら SIMPLE の目安 α_p ≤ 1 − α_u（新しい α_u で評価）を上限に加える
    """
    if not (prev_max_res > 0.0 and max_res > 0.0 and np.isfinite(max_res)):
        return alpha_u, alpha_p
    ratio = max_res / prev_max_res
    stalled = min_res is not None and min_res > 0.0 and max_res > bounds.stall_ratio * min_res
    if ratio > bounds.worsen_ratio or stalled:
        new_u = max(alpha_u * bounds.shrink, bounds.alpha_u_min)
        new_p = max(alpha_p * bounds.shrink, bounds.alpha_p_min)
    elif ratio < bounds.improve_ratio:
        new_u = min(alpha_u * bounds.grow, bounds.alpha_u_max)
        new_p = min(alpha_p * bounds.grow, bounds.alpha_p_max)
    else:
        return alpha_u, alpha_p
    if simple_cap:
        new_p = min(new_p, max(1.0 - new_u, bounds.alpha_p_min))
    return float(new_u), float(new_p)


__all__ = ["DEFAULT_RELAXATION_BOUNDS", "RelaxationBounds", "adapt_relaxation_factors"]
