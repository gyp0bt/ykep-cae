"""計量部（矩形チャネル）の形状係数 Fd / Fp の級数解.

古典解（Tadmor & Gogos）:

    Q = (V_z W H / 2)·F_d − (W H³ / 12μ)·G·F_p

    F_d(h) = (16/(π³h)) Σ_{i:odd} tanh(a i)/i³        a = πh/2,  h = H/W
    F_p(h) = (192/(π⁵h)) Σ_{i:odd} tanh(a i)/i⁵

そのままでは tanh → 1 の尾が 1/i³ でしか減衰せず、倍精度に届かせるのに
i ~ 10⁵ 項を要する。tanh(x) = 1 − 2/(e^{2x}+1) と分解し、定数部を ζ の
閉形式に置き換えると残る級数が指数減衰する。

    Σ_{i odd} 1/i³ = (7/8)ζ(3)        Σ_{i odd} 1/i⁵ = (31/32)ζ(5)

    F_d = (16/(π³h)) [ (7/8)ζ(3)   − Σ_{i odd} t(a i)/i³ ]     t(x) = 2/(e^{2x}+1)
    F_p = (192/(π⁵h)) [ (31/32)ζ(5) − Σ_{i odd} t(a i)/i⁵ ]

**1 − tanh(x) を引き算で作らず 2/(e^{2x}+1) で直接評価するのが桁落ち対策の核心。**

浅溝側（h → 0）では括弧内が π³h/16 に縮むため相対的な桁落ちが残る。損失は h の
対数でしか増えないので h = 10⁻³ でも 13 桁が残るが、h < 10⁻⁴ では漸近式の方が
むしろ正確になるので切り替える（実測: h=10⁻⁵ で直接和は 7 桁目が崩れる）。

このモジュールは「真値の供給源」であり、ソルバーからは参照されない。
検証テスト（ゲート G1/G2）だけが使う。
"""

from __future__ import annotations

import math

import numpy as np

# Σ_{i odd} 1/i³ = (7/8)ζ(3),  Σ_{i odd} 1/i⁵ = (31/32)ζ(5)
_ODD_ZETA3 = 7.0 / 8.0 * 1.2020569031595942854
_ODD_ZETA5 = 31.0 / 32.0 * 1.0369277551433699263

_CUTOFF = 25.0
"""級数の打ち切り: a·i > 25 で t(a·i) < 2e-22（倍精度の下に沈む）."""

_H_ASYMPTOTIC = 1.0e-4
"""これ未満の h では漸近式に切り替える（直接和は桁落ちで劣化し、かつ項数が増える）."""

_SHALLOW_DRAG_SLOPE = 16.0 / math.pi**3 * _ODD_ZETA3
"""1 − F_d ≃ この係数 × h。実測値 0.5427545144 と 8 桁一致（h ≲ 0.2）."""


def _tanh_tail(x: np.ndarray) -> np.ndarray:
    """1 − tanh(x) = 2/(e^{2x}+1). 引き算を経由しないので桁落ちしない."""
    return 2.0 / (np.exp(2.0 * np.minimum(x, 350.0)) + 1.0)


def _odd_indices(a: float) -> np.ndarray:
    """a·i < _CUTOFF を満たす奇数 i の配列."""
    i_max = max(3, int(math.ceil(_CUTOFF / a)) + 1)
    return np.arange(1, i_max + 1, 2, dtype=np.float64)


def shape_factor_drag(h: float) -> float:
    """引きずり流れの形状係数 F_d(h), h = H/W.

    h → 0 で 1（無限幅平板の極限）、h が大きいほど側壁抵抗で小さくなる。
    h = 1（正方形断面）でちょうど 1/2 になる。
    """
    if h <= 0.0:
        msg = f"h = H/W は正の値が必要: {h}"
        raise ValueError(msg)
    if h < _H_ASYMPTOTIC:
        return 1.0 - _SHALLOW_DRAG_SLOPE * h
    a = math.pi * h / 2.0
    i = _odd_indices(a)
    tail = float(np.sum(_tanh_tail(a * i) / i**3))
    return 16.0 / (math.pi**3 * h) * (_ODD_ZETA3 - tail)


def shape_factor_pressure(h: float) -> float:
    """圧力流れの形状係数 F_p(h), h = H/W.

    F_d より 1 に近く保たれる（圧力流れは側壁の影響を受けにくい）。
    浅溝の漸近形は 1 − F_p ≃ h²（実測で係数 → 1）。
    """
    if h <= 0.0:
        msg = f"h = H/W は正の値が必要: {h}"
        raise ValueError(msg)
    if h < _H_ASYMPTOTIC:
        return 1.0 - h * h
    a = math.pi * h / 2.0
    i = _odd_indices(a)
    tail = float(np.sum(_tanh_tail(a * i) / i**5))
    return 192.0 / (math.pi**5 * h) * (_ODD_ZETA5 - tail)


def metering_flow_rate(
    V_z: float,
    W: float,
    H: float,
    mu: float,
    G: float,
    *,
    F_d: float,
    F_p: float,
) -> float:
    """計量部の体積流量 [m³/s].

    Q = (V_z W H / 2)·F_d − (W H³ / 12μ)·G·F_p

    第 1 項が引きずり流れ、第 2 項が圧力流れ（G>0 = 背圧で押出量を減らす）。
    G に対して厳密な一次関数であり、Q=0 となる G が閉塞点。
    """
    return V_z * W * H * F_d / 2.0 - W * H**3 * G * F_p / (12.0 * mu)
