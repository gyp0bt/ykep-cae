"""ゲート G5: Pinto–Tadmor 文献 RTD との照合（浅溝極限への収束）.

設計: docs/design/extruder-g5-literature-rtd.md §3.2

文献モデル（無限幅・折返し瞬時・周回無限回）の仮定を ykep 側で近づけるための三つの条件:

1. **閉チャネル（δ = 0）。** 隙間があると隙間を突っ切る速い経路が生まれ、その割合
   ≈ 2 tanφ (δ/H) L/W は δ/H を固定すると H/W → 0 でも消えない。
2. **周回を重ねる長さ（z_axial = 0.5 m、周回 ≈ 5 回）。** 周回が 1〜2 回では滞留時間が
   種の高さで決まり（軸方向速度 3ξ(1−ξ) は ξ = 1/2 で最大 → t_min/t̄ = 2/3）、
   文献の「周回平均」の極限に届かない。周回時間 ≈ W/V_x は H に依らないので、
   z_axial を固定すれば H 系列の周回数は揃う。
3. **小さい時間刻み（cfl = 0.1）。** 既定 cfl = 1.0 では ψ 双一次場のセル境界で
   RK4 が流線を横切り、周回を重ねると壁際へ流れ込む（|Δψ|/ψ_max 中央値 8%）。

規格化は t̄_∞ = t̄_theory·F_d（側壁の無い平均滞留時間）。側壁は流量を F_d 倍に
減らして t̄_theory を延ばすが、分位点を担う溝中央の流線は側壁を知らない。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    RTDInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.pinto_tadmor import RTDComparison, compare_rtd, pinto_tadmor_rtd
from xkep_cae_fluid.extruder.rtd import RTDProcess
from xkep_cae_fluid.extruder.shape_factors import shape_factor_drag
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
Z_AXIAL = 0.5
"""軸方向長さ [m]。周回時間 ≈ W/V_x ≈ 1.6 s、t̄ ≈ 6.6 s（H=1 mm）で周回 ≈ 5 回."""
CFL = 0.1
SHALLOW_SERIES = (0.004, 0.002, 0.001)
"""H [m]。W ≈ 34.1 mm なので H/W = 0.117, 0.059, 0.029."""
TOL_P10 = TOL_P50 = 0.03
TOL_P90 = 0.05
TOL_CURVE = 0.05
"""分位関数の相対偏差の F ∈ [0.05, 0.9] 平均に対する閾値."""
TOL_MEAN = 0.02
"""⟨t⟩/t̄_theory − 1 の閾値。流線ドリフトが残っていれば裾が伸びてここに出る."""

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def shallow_spec(H: float, ny: int = 32, nx_channel: int = 80) -> ScrewSpec:
    """閉チャネル。nx_land は固体なので粗くてよい."""
    return replace(_BASE, H=H, delta=0.0, nx_channel=nx_channel, nx_land=8, ny_bulk=ny, n_gap=0)


def pipeline(spec: ScrewSpec, G: float = 0.0, z_axial: float = Z_AXIAL, cfl: float = CFL):
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    track = ParticleTrackerProcess().process(
        ParticleTrackInput(flow=flow, z_axial=z_axial, cfl=cfl)
    )
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=z_axial, n_bins=100))
    return flow, track, rtd


def compare(track, rtd, spec: ScrewSpec, pt) -> RTDComparison:
    """t̄_∞ = t̄_theory·F_d で規格化して文献曲線と比べる."""
    ok = track.escaped
    f_d = shape_factor_drag(spec.H / spec.W)
    return compare_rtd(track.t_res[ok], track.weight[ok], rtd.t_mean_theory * f_d, pt)


@pytest.fixture(scope="module")
def pt():
    return pinto_tadmor_rtd(0.0)


@pytest.fixture(scope="module")
def series(pt):
    """H 系列（細格子 32×80）。合計 2〜3 分."""
    out = []
    for H in SHALLOW_SERIES:
        spec = shallow_spec(H)
        _, track, rtd = pipeline(spec)
        out.append((H, rtd, compare(track, rtd, spec, pt)))
    return out


@pytest.mark.slow
class TestGateG5:
    """浅溝極限で文献曲線に収束する（すべて閾値規格化比 < 1.00 で合格）."""

    def test_shallowest_quantiles_match_literature(self, series):
        _, _, cmp = series[-1]
        assert cmp.dev_p10 / TOL_P10 < 1.0
        assert cmp.dev_p50 / TOL_P50 < 1.0
        assert cmp.dev_p90 / TOL_P90 < 1.0

    def test_shallowest_curve_matches_literature(self, series):
        _, _, cmp = series[-1]
        assert cmp.curve_l1 / TOL_CURVE < 1.0

    def test_deviation_shrinks_monotonically_with_depth(self, series):
        """側壁の影響（p50・p90・曲線）が H/W とともに単調に減ること.

        p10 は最浅以外でも 1% 内に入り差が種まきの粗さに埋もれるので単調性は課さない。
        """
        p50 = [c.dev_p50 for _, _, c in series]
        p90 = [c.dev_p90 for _, _, c in series]
        l1 = [c.curve_l1 for _, _, c in series]
        assert p50 == sorted(p50, reverse=True)
        assert p90 == sorted(p90, reverse=True)
        assert l1 == sorted(l1, reverse=True)

    def test_mean_is_conserved_without_streamline_drift(self, series):
        """cfl = 0.1 なら閉チャネルでも ⟨t⟩ = V/Q が 2% で成り立つ."""
        for _, rtd, _ in series:
            assert abs(rtd.t_mean / rtd.t_mean_theory - 1.0) / TOL_MEAN < 1.0

    def test_minimum_residence_time_is_three_quarters(self, series):
        """t_min/t̄_∞ → 3/4（停留高さ ξ=2/3 の粒子）。種はセル中心なので 5% 以内."""
        for _, rtd, cmp in series:
            t_min = rtd.t_min / rtd.t_mean_theory
            f_d = cmp.p50_ratio / (rtd.t_p50 / rtd.t_mean_theory)  # = 1/F_d
            assert abs(t_min * f_d / 0.75 - 1.0) / 0.05 < 1.0


class TestGateG5Quick:
    """粗格子 16×40・最浅のみ（≈ 5 s）。p10・p50 は粗格子でも 3% に入る.

    p90 は粗格子だと 10% 超えるので細格子（TestGateG5, slow）で見る。
    """

    def test_shallowest_p10_p50_match_literature(self, pt):
        spec = shallow_spec(0.001, ny=16, nx_channel=40)
        _, track, rtd = pipeline(spec)
        cmp = compare(track, rtd, spec, pt)
        assert cmp.dev_p10 / TOL_P10 < 1.0
        assert cmp.dev_p50 / TOL_P50 < 1.0
