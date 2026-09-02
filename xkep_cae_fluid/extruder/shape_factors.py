"""計量部（矩形チャネル）の形状係数 Fd / Fp の級数解.

古典解（Tadmor & Gogos）:

    Q = (V_z W H / 2)·F_d − (W H³ / 12μ)·G·F_p

**F_d と F_p では H と W の役割が入れ替わる。** 同じ形だと思い込むと 6% ずれる
（実際にゲート G2 で捕まえた）。h = H/W として

    F_d(h) = (16/(π³h)) Σ_{i:odd} tanh(iπh/2)/i³
    F_p(h) = 1 − (192h/π⁵) Σ_{i:odd} tanh(iπ/(2h))/i⁵

導出（F_p）: w = (G/2μ)(y²−Hy) は側壁条件を満たさないので、x=0,W で
これを打ち消す調和補正を sin(iπy/H)·cosh(iπ(x−W/2)/H)/cosh(iπW/2H) で足す。
y(H−y) = Σ_odd (8H²/(i³π³)) sin(iπy/H) を使って積分すると上式になる。
F_d 側は w=V_z を y=H に課したラプラス解 Σ (4V/iπ) sin(iπx/W) sinh(iπy/W)/sinh(iπH/W)
の積分から出る（cosh c − 1 )/sinh c = tanh(c/2) で tanh になる）。

数値面。F_d は tanh → 1 の尾が 1/i³ でしか減衰せず、倍精度に届かせるのに
i ~ 10⁵ 項を要する。tanh(x) = 1 − 2/(e^{2x}+1) と分解し、定数部を ζ の
閉形式に置き換えると残る級数が指数減衰する。

    Σ_{i odd} 1/i³ = (7/8)ζ(3)        Σ_{i odd} 1/i⁵ = (31/32)ζ(5)

    F_d = (16/(π³h)) [ (7/8)ζ(3)   − Σ_{i odd} t(iπh/2)/i³ ]     t(x) = 2/(e^{2x}+1)
    F_p = 1 − (192h/π⁵) [ (31/32)ζ(5) − Σ_{i odd} t(iπ/(2h))/i⁵ ]

**1 − tanh(x) を引き算で作らず 2/(e^{2x}+1) で直接評価するのが桁落ち対策の核心。**

桁落ちの出方は 2 つで逆向きになる。
- F_d は浅溝側（h → 0）で括弧内が π³h/16 に縮み、相対的な桁落ちが起きる。
  損失は h の対数でしか増えず h = 10⁻³ でも 13 桁残るが、h < 10⁻⁴ では
  漸近式の方がむしろ正確なので切り替える（h=10⁻⁵ で直接和は 7 桁目が崩れる）。
- F_p は浅溝側では前置係数 192h/π⁵ が 0 に縮むだけで桁落ちしない。逆に深溝側
  （h → ∞、F_p → 0）で 1 − 1 の形になるが、h=5 でも損失は 1.5 桁。分岐は不要。

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
"""F_d はこれ未満の h で漸近式に切り替える（直接和は桁落ちで劣化し、項数も増える）.

F_p は浅溝側で桁落ちしないので分岐しない。
"""

_SHALLOW_DRAG_SLOPE = 16.0 / math.pi**3 * _ODD_ZETA3
"""1 − F_d ≃ この係数 × h = 0.5427545144。h を 4 桁振って 8 桁一致を実測確認."""

SHALLOW_PRESSURE_SLOPE = 192.0 / math.pi**5 * _ODD_ZETA5
"""1 − F_p ≃ この係数 × h = 0.6302488763。F_p の方が側壁の影響を強く受ける."""


def _tanh_tail(x: np.ndarray) -> np.ndarray:
    """1 − tanh(x) = 2/(e^{2x}+1). 引き算を経由しないので桁落ちしない."""
    return 2.0 / (np.exp(2.0 * np.minimum(x, 350.0)) + 1.0)


def _odd_indices(scale: float) -> np.ndarray:
    """scale·i < _CUTOFF を満たす奇数 i の配列.

    scale は級数の tanh に入る 1 項あたりの引数（F_d は πh/2、F_p は π/(2h)）。
    """
    i_max = max(3, int(math.ceil(_CUTOFF / scale)) + 1)
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

    F_d とは H と W の役割が入れ替わり、tanh の引数が iπ/(2h) になる。
    浅溝の漸近形は 1 − F_p ≃ 0.63025·h で、F_d の傾き 0.54275 より大きい。
    つまり**圧力流れの方が側壁の影響を強く受け、常に F_p < F_d** になる。
    """
    if h <= 0.0:
        msg = f"h = H/W は正の値が必要: {h}"
        raise ValueError(msg)
    b = math.pi / (2.0 * h)
    i = _odd_indices(b)
    tail = float(np.sum(_tanh_tail(b * i) / i**5))
    return 1.0 - 192.0 * h / math.pi**5 * (_ODD_ZETA5 - tail)


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
