"""形状係数 Fd / Fp の級数解のテスト（ゲート G1/G2 の真値）.

文献値をハードコードせず級数を自前で実装し、素朴な多項打ち切りとの一致で
固定する（1a/a02 で矩形管の f·Re 級数に対して行ったのと同じ方針）。
"""

from __future__ import annotations

import math

import pytest

from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)


class TestShapeFactorAPI:
    """入力検証."""

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="正の値"):
            shape_factor_drag(0.0)
        with pytest.raises(ValueError, match="正の値"):
            shape_factor_pressure(-1.0)


class TestShapeFactorPhysics:
    """級数解の物理的・数学的性質."""

    def test_shallow_limit_is_one(self):
        """H/W → 0 で無限幅平板に退化し Fd, Fp → 1."""
        assert shape_factor_drag(1e-6) == pytest.approx(1.0, abs=1e-5)
        assert shape_factor_pressure(1e-6) == pytest.approx(1.0, abs=1e-5)

    def test_shallow_slopes(self):
        """1−F ≃ 係数×h の傾きが閉形式と一致すること.

        Fd: 16(7/8)ζ(3)/π³ = 0.5427545  /  Fp: 192(31/32)ζ(5)/π⁵ = 0.6302489
        Fp の傾きの方が大きい = 圧力流れの方が側壁の影響を強く受ける。
        """
        from xkep_cae_fluid.extruder.shape_factors import (
            _SHALLOW_DRAG_SLOPE,
            SHALLOW_PRESSURE_SLOPE,
        )

        assert _SHALLOW_DRAG_SLOPE == pytest.approx(0.5427545144, rel=1e-9)
        assert SHALLOW_PRESSURE_SLOPE == pytest.approx(0.6302488763, rel=1e-9)
        assert SHALLOW_PRESSURE_SLOPE > _SHALLOW_DRAG_SLOPE
        h = 1e-3
        assert (1.0 - shape_factor_drag(h)) / h == pytest.approx(_SHALLOW_DRAG_SLOPE, rel=1e-6)
        assert (1.0 - shape_factor_pressure(h)) / h == pytest.approx(
            SHALLOW_PRESSURE_SLOPE, rel=1e-5
        )

    def test_monotone_decreasing(self):
        """側壁の抵抗が効くので H/W が増えるほど小さくなる."""
        hs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        fd = [shape_factor_drag(h) for h in hs]
        fp = [shape_factor_pressure(h) for h in hs]
        assert all(a > b for a, b in zip(fd, fd[1:], strict=False))
        assert all(a > b for a, b in zip(fp, fp[1:], strict=False))

    def test_pressure_factor_below_drag_factor(self):
        """圧力流れは 4 面全部が抵抗になるので常に Fp < Fd.

        Fd と Fp を同じ形の級数だと思い込むとここが逆転する（実際に間違えた）。
        """
        for h in (0.01, 0.05, 0.117248, 0.5, 1.0, 2.0, 5.0):
            assert shape_factor_pressure(h) < shape_factor_drag(h), f"h={h}"

    def test_square_channel_drag_factor_is_half(self):
        """H/W = 1（正方形断面）で Fd = 1/2 になる（級数の非自明な性質）."""
        assert shape_factor_drag(1.0) == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize(
        ("h", "fd", "fp"),
        [
            (0.010000, 0.994572454856, 0.993697511237),
            (0.050000, 0.972862274278, 0.968487556186),
            (0.117248, 0.936363118691, 0.926104579754),
            (0.200000, 0.891449128218, 0.873950262564),
            (0.500000, 0.729584593001, 0.686045031359),
            (1.000000, 0.500000000000, 0.421731044865),
            (2.000000, 0.270415406999, 0.171511257840),
        ],
    )
    def test_reference_values(self, h, fd, fp):
        """参照値と一致すること（回帰固定）.

        参照値は mpmath 40 桁で素朴級数を 4×10⁵ 項まで足した独立計算と
        相対 1e-10 以内で一致することを確認済み。ここでは回帰の固定が目的。
        独立性そのものは test_agrees_with_naive_series が担保する。
        """
        assert shape_factor_drag(h) == pytest.approx(fd, rel=1e-8)
        assert shape_factor_pressure(h) == pytest.approx(fp, rel=1e-8)

    def test_agrees_with_naive_series(self):
        """打ち切り対策を入れた式が、素朴な多項打ち切りと一致すること.

        素朴形は tanh→1 の尾が 1/i³ でしか減衰しないので 2×10⁵ 項要る。
        ここが一致すれば ζ 閉形式への置換が正しい。
        """
        for h in (0.05, 0.117248, 0.2, 0.5, 1.0, 2.0):
            a = math.pi * h / 2.0
            naive_d = sum(math.tanh(a * i) / i**3 for i in range(1, 200001, 2))
            naive_d *= 16.0 / (math.pi**3 * h)
            b = math.pi / (2.0 * h)
            naive_p = sum(math.tanh(b * i) / i**5 for i in range(1, 20001, 2))
            naive_p = 1.0 - 192.0 * h / math.pi**5 * naive_p
            assert shape_factor_drag(h) == pytest.approx(naive_d, rel=1e-9)
            assert shape_factor_pressure(h) == pytest.approx(naive_p, rel=1e-12)

    def test_asymptotic_branch_is_continuous(self):
        """Fd の漸近式への切り替え点で不連続にならないこと（Fp に分岐は無い）."""
        from xkep_cae_fluid.extruder.shape_factors import _H_ASYMPTOTIC

        eps = _H_ASYMPTOTIC * 1e-6
        assert shape_factor_drag(_H_ASYMPTOTIC - eps) == pytest.approx(
            shape_factor_drag(_H_ASYMPTOTIC + eps), rel=1e-9
        )


class TestMeteringFlowRate:
    """流量式の重ね合わせ構造."""

    def test_superposition(self):
        """Q(G) が G の一次関数で、G=0 で純引きずり、Q=0 で閉塞点になること."""
        V_z, W, H, mu = 0.199573, 0.0341156, 0.004, 1000.0  # noqa: N806
        h = H / W
        fd, fp = shape_factor_drag(h), shape_factor_pressure(h)

        q0 = metering_flow_rate(V_z, W, H, mu, 0.0, F_d=fd, F_p=fp)
        assert q0 == pytest.approx(V_z * W * H * fd / 2.0, rel=1e-12)

        g_closed = q0 * 12.0 * mu / (W * H**3 * fp)
        q_closed = metering_flow_rate(V_z, W, H, mu, g_closed, F_d=fd, F_p=fp)
        assert q_closed == pytest.approx(0.0, abs=1e-15)

        q1 = metering_flow_rate(V_z, W, H, mu, 1.0e6, F_d=fd, F_p=fp)
        q2 = metering_flow_rate(V_z, W, H, mu, 2.0e6, F_d=fd, F_p=fp)
        assert q0 - q1 == pytest.approx(q1 - q2, rel=1e-12)
