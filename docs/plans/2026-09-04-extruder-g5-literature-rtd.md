# ゲート G5 文献 RTD 照合 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pinto–Tadmor 1970 の計量部 RTD を自前で再導出し、ykep 2.5D の RTD が浅溝極限でそれに収束することをゲート G5 として固定し、レポートを公開して Phase 2 の前提を差し替える。

**Architecture:** `shape_factors.py` と同じ「真値の供給源」モジュール `pinto_tadmor.py`（numpy のみ、ソルバーから参照しない）を足し、既存の `ExtruderFlowProcess → ParticleTrackerProcess → RTDProcess` の出力を重み付き ECDF にして比較する。ゲートは pytest（粗格子）、フル解像度の系列は `experiments/extruder/g5_literature.py` が JSON に落とし、`g5_report.py` が図解レポートを組む。

**Tech Stack:** Python 3.12, numpy, pytest, `.venv/bin/python`, `~/work/tb/bin/mdview`, Artifact 公開。

**Spec:** `docs/design/extruder-g5-literature-rtd.md`

## 実行結果（2026-09-04、全タスク完了）

計画どおりに Task 1–6 を実施し、G5 は全項目 比 < 1.00 で通過（[status-29](../status/status-29.md)、
[レポート](../reports/extruder/g5-literature-rtd.md)）。実行中に**計画から変えた点**は
次の 3 つで、いずれも文献モデルの仮定を ykep 側で満たすための条件（設計 §3.2 に反映済み）。

| 計画 | 実際 | 理由（メカニズム） |
|---|---|---|
| 隙間あり δ/H 固定、z = 0.05 m、既定 cfl | **閉チャネル δ = 0、z = 0.5 m、cfl = 0.1** | 隙間の速い経路の流量比 ≈ 2tanφ(δ/H)L/W は H/W → 0 で消えない。周回 1〜2 回では t_min が 3/4 でなく 2/3 に落ちる。既定 cfl = 1.0 は流線ドリフトで裾を汚す |
| t/t̄_theory で規格化 | **t/(t̄_theory·F_d)** | 側壁は流量を F_d 倍に減らすが溝中央の流線は側壁を知らない |
| 曲線は max で判定 | **L1 平均で判定、max は観察** | 1 粒子/セルの種まきで F > 0.6 が階段になり max が汚れる |

Task 4 の `result.json` スキーマも上記に合わせて変わっている（`t_ref`, `*_over_ref`,
`n_loops`, `extrapolated_weight_fraction` 等。実体は `experiments/extruder/g5_literature.py`）。
以下は着手時の計画のまま残す。

---

## Global Constraints

- 実行は `cd ~/work/ykep-cae && PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python`。メモリ 4 GB 以内。
- ソルバー・追跡・RTD Process のロジックには触れない（spec §2）。
- 判定は閾値規格化比（比 < 1.00 で合格）で書く。
- 例外は `raise ... from e`。裸の再 raise で真因を消さない。
- 粗格子で 30 s を超えるテストは `@pytest.mark.slow`。
- ログは `logs/`（gitignored）。作業ディレクトリは `/tmp/of-g5`（レポートにはこの名で書く）。
- 文書は日本語。ユーザーは `gyp さん`。

---

### Task 1: 文献モデル `pinto_tadmor.py`

**Files:**
- Create: `xkep_cae_fluid/extruder/pinto_tadmor.py`
- Modify: `xkep_cae_fluid/extruder/__init__.py`（export 追加）
- Test: `tests/test_extruder_pinto_tadmor.py`

**Interfaces:**
- Produces: `pinto_tadmor_rtd(r: float = 0.0, n_xi: int = 4000) -> PintoTadmorRTD`、
  `PintoTadmorRTD(t_over_tbar: np.ndarray, F: np.ndarray, t_min_ratio: float, t_p10_ratio: float, t_p50_ratio: float, t_p90_ratio: float, tbar_over_L_Vz: float, r: float)`

- [x] **Step 1: 失敗するテストを書く**

```python
"""Pinto–Tadmor 型 RTD モデル（真値の供給源）のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.extruder.pinto_tadmor import PintoTadmorRTD, pinto_tadmor_rtd


class TestExactRelations:
    @pytest.mark.parametrize("r", [0.0, -0.3, -0.7])
    def test_mean_is_volume_over_flow(self, r):
        """t̄·V_z/L = 2/(1+r)（流管の体積÷流束の厳密値）に 1e-3 で一致."""
        res = pinto_tadmor_rtd(r)
        assert abs(res.tbar_over_L_Vz / (2.0 / (1.0 + r)) - 1.0) / 1.0e-3 < 1.0

    def test_minimum_is_three_quarters(self):
        """t_min/t̄ = 3/4。最速は再循環の停留高さ ξ=2/3 に居て折り返さない粒子."""
        res = pinto_tadmor_rtd(0.0)
        assert abs(res.t_min_ratio - 0.75) / 1.0e-4 < 1.0

    def test_reduced_curve_is_independent_of_back_pressure(self):
        """F(t/t̄) は r によらない（3ξ(1−ξ) = ξ − (3ξ²−2ξ) と 1 周の横断変位 0）."""
        a, b, c = (pinto_tadmor_rtd(r) for r in (0.0, -0.3, -0.7))
        for key in ("t_p10_ratio", "t_p50_ratio", "t_p90_ratio"):
            assert abs(getattr(a, key) - getattr(b, key)) / 1.0e-5 < 1.0
            assert abs(getattr(a, key) - getattr(c, key)) / 1.0e-5 < 1.0

    def test_reference_quantiles(self):
        """設計時に確認した分位点（p10 0.7524, p50 0.8225, p90 1.3247）."""
        res = pinto_tadmor_rtd(0.0)
        assert res.t_p10_ratio == pytest.approx(0.7524, abs=5e-4)
        assert res.t_p50_ratio == pytest.approx(0.8225, abs=5e-4)
        assert res.t_p90_ratio == pytest.approx(1.3247, abs=5e-4)


class TestCurveShape:
    def test_cumulative_is_monotone_from_zero_to_one(self):
        res = pinto_tadmor_rtd(0.0)
        assert isinstance(res, PintoTadmorRTD)
        assert np.all(np.diff(res.t_over_tbar) >= 0.0)
        assert np.all(np.diff(res.F) >= 0.0)
        assert res.F[0] < 1.0e-3 and res.F[-1] > 1.0 - 1.0e-3
        assert res.t_over_tbar[0] == pytest.approx(res.t_min_ratio)

    def test_converged_in_n_xi(self):
        """n_ξ を 4 倍にしても分位点が 1e-4 で動かない."""
        a, b = pinto_tadmor_rtd(0.0, n_xi=1000), pinto_tadmor_rtd(0.0, n_xi=4000)
        assert abs(a.t_p50_ratio - b.t_p50_ratio) / 1.0e-4 < 1.0


class TestArguments:
    @pytest.mark.parametrize("r", [-1.0, -1.5, 0.1])
    def test_rejects_back_pressure_ratio_out_of_range(self, r):
        with pytest.raises(ValueError, match="Q_p/Q_d"):
            pinto_tadmor_rtd(r)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="n_xi"):
            pinto_tadmor_rtd(0.0, n_xi=8)
```

- [x] **Step 2: 失敗を確認**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_extruder_pinto_tadmor.py -q`
Expected: `ModuleNotFoundError: xkep_cae_fluid.extruder.pinto_tadmor`

- [x] **Step 3: 実装**

```python
"""Pinto–Tadmor 1970 型の計量部 RTD（文献モデル、真値の供給源）.

Pinto, G. & Tadmor, Z. (1970) Polym. Eng. Sci. 10, 279. Tadmor & Gogos
*Principles of Polymer Processing* §7 に再掲。仮定: 無限幅平板（側壁無し）、
等温ニュートン、漏れ無し、フライトでの折返しは瞬時。ξ = y/H（0 根元、1 バレル）、
r = Q_p/Q_d ∈ (−1, 0]。

    下流  ŵ(ξ) = w/V_z = ξ + 3r·ξ(1−ξ)
    横断  û(ξ) = u/V_x = 3ξ² − 2ξ      （正味流量 0、ξ = 2/3 で符号反転）

上層 ξ ∈ (2/3, 1) の粒子はフライトで折り返し、横断流束の保存
∫_ξ^1 û = −∫_0^{ξ_c} û ⇔ g(ξ) = g(ξ_c), g(s) = s²(1−s) で決まる下層 ξ_c に移る。
1 周の時間重み（横断 1 回 ∝ 1/|û|）で平均した下流速度と滞留時間は

    w̄/V_z = (ŵ/û + ŵ_c/|û_c|) / (1/û + 1/|û_c|),   t = L/w̄

**普遍性の機構。** 3ξ(1−ξ) = ξ − (3ξ²−2ξ)、つまり圧力流れ分布は「引きずり分布 −
横断分布」に恒等的に等しい。横断速度は閉じた流線 1 周で変位ゼロなので周平均が
消え、どの流線でも w̄ = (1+r)·w̄_drag、流線対の流量も (1+r) 倍。t も t̄ も同じ因子で
割られて F(t/t̄) は r に依らない。t̄ = HWL/Q は流管の体積÷流束として厳密。

**数値評価は下層 ξ_c で標本化する。** 上層 ξ で標本化すると ξ→1 の粒子が根元
ξ_c ~ √(1−ξ) に張り付いて t ~ 1/√(1−ξ) と発散し、中点則が O(1/√n) にしか
収束しない（実測 1000→16000 点で 0.37%→0.09%）。下層 ξ_c ∈ (0, 2/3) の中点で
標本化し、流線対の流量 dQ = [ŵ(ξ_c) + ŵ(ξ)·|û_c|/û] dξ_c を重みにすると
被積分関数が滑らかになり、1000 点で t̄ が 1e-7、普遍性が 1e-6 で成立する。
r < −1/3 では根元付近の ŵ が負（逆流）になるが、流線対の正味流量は
(1+r)×引きずり対流量 > 0 なので重みは常に正。

t_min/t̄ = 3/4 は厳密値。最速の粒子はバレル面ではなく再循環の停留高さ ξ = 2/3 に
居て一度も折り返さない粒子で、t_min = L/(⅔V_z) = ¾·(2L/V_z)。

このモジュールは「真値の供給源」であり、ソルバーからは参照されない。
検証テスト（ゲート G5）とレポートだけが使う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

XI_SPLIT = 2.0 / 3.0
"""横断速度が符号反転する高さ。再循環の停留線."""

_BISECT_ITERS = 60
"""二分法の反復数。区間幅 1/3 × 2⁻⁶⁰ で倍精度に届く."""


@dataclass(frozen=True)
class PintoTadmorRTD:
    """縮約 RTD 曲線と代表値（すべて t̄ = HWL/Q で規格化）.

    Parameters
    ----------
    t_over_tbar : np.ndarray
        滞留時間 / 平均滞留時間、昇順
    F : np.ndarray
        累積分布（区間中点の累積割合、`weighted_quantile` と同じ流儀）
    t_min_ratio, t_p10_ratio, t_p50_ratio, t_p90_ratio : float
        最短・10/50/90 パーセンタイル（t̄ 規格化）
    tbar_over_L_Vz : float
        t̄·V_z/L。厳密値は 2/(1+r)
    r : float
        背圧比 Q_p/Q_d
    """

    t_over_tbar: np.ndarray
    F: np.ndarray
    t_min_ratio: float
    t_p10_ratio: float
    t_p50_ratio: float
    t_p90_ratio: float
    tbar_over_L_Vz: float
    r: float


def _g(s: np.ndarray) -> np.ndarray:
    """横断流束の積分 g(s) = s²(1−s)。(0, 2/3) で増加、(2/3, 1) で減少."""
    return s * s * (1.0 - s)


def _upper_partner(xi_lower: np.ndarray) -> np.ndarray:
    """g(ξ) = g(ξ_c) を満たす上層の高さ ξ ∈ (2/3, 1) を二分法で解く（ベクトル化）."""
    target = _g(xi_lower)
    lo = np.full_like(xi_lower, XI_SPLIT)
    hi = np.ones_like(xi_lower)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        above = _g(mid) > target  # g は (2/3, 1) で単調減少
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def pinto_tadmor_rtd(r: float = 0.0, n_xi: int = 4000) -> PintoTadmorRTD:
    """縮約 RTD 曲線 F(t/t̄) を返す.

    Parameters
    ----------
    r : float
        背圧比 Q_p/Q_d ∈ (−1, 0]。0 = 純引きずり。−1 は閉塞（Q = 0、t̄ 発散）で不可
    n_xi : int
        下層 ξ_c の標本数（中点則）
    """
    if not (-1.0 < r <= 0.0):
        msg = f"背圧比 Q_p/Q_d は (−1, 0] が必要: {r}"
        raise ValueError(msg)
    if n_xi < 16:
        msg = f"n_xi は 16 以上が必要: {n_xi}"
        raise ValueError(msg)

    edges = np.linspace(0.0, XI_SPLIT, n_xi + 1)
    xi_c = 0.5 * (edges[:-1] + edges[1:])
    d_xi = edges[1] - edges[0]
    xi = _upper_partner(xi_c)

    def w_hat(s: np.ndarray) -> np.ndarray:
        return s + 3.0 * r * s * (1.0 - s)

    u_up = 3.0 * xi * xi - 2.0 * xi  # û(ξ) > 0
    u_lo = 2.0 * xi_c - 3.0 * xi_c * xi_c  # |û(ξ_c)| > 0
    a, b = 1.0 / u_up, 1.0 / u_lo
    t = (a + b) / (w_hat(xi) * a + w_hat(xi_c) * b)  # t·V_z/L
    dq = (w_hat(xi_c) + w_hat(xi) * u_lo / u_up) * d_xi  # 流線対の流量（|dξ/dξ_c| = |û_c|/û）

    tbar = float(np.sum(t * dq) / np.sum(dq))
    order = np.argsort(t)
    t_red = t[order] / tbar
    q = dq[order]
    F = (np.cumsum(q) - 0.5 * q) / q.sum()
    p10, p50, p90 = np.interp([0.1, 0.5, 0.9], F, t_red)
    return PintoTadmorRTD(
        t_over_tbar=t_red,
        F=F,
        t_min_ratio=float(t_red[0]),
        t_p10_ratio=float(p10),
        t_p50_ratio=float(p50),
        t_p90_ratio=float(p90),
        tbar_over_L_Vz=tbar,
        r=float(r),
    )
```

`__init__.py` に `from xkep_cae_fluid.extruder.pinto_tadmor import PintoTadmorRTD, pinto_tadmor_rtd` と `__all__` の 2 項目を追加。

- [x] **Step 4: 合格を確認**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_extruder_pinto_tadmor.py -q`
Expected: 全件 PASS

- [x] **Step 5: コミット**

```bash
git add xkep_cae_fluid/extruder/pinto_tadmor.py xkep_cae_fluid/extruder/__init__.py tests/test_extruder_pinto_tadmor.py
git commit -m "feat(extruder): Pinto–Tadmor 型 RTD の再導出 — t_min/t̄ = 3/4 と背圧比普遍性を固定"
```

---

### Task 2: 重み付き ECDF `weighted_ecdf`

**Files:**
- Modify: `xkep_cae_fluid/extruder/rtd.py`（`weighted_quantile` の直後）
- Modify: `xkep_cae_fluid/extruder/__init__.py`
- Test: `tests/test_extruder_rtd.py`（`TestWeightedQuantile` の後に追加）

**Interfaces:**
- Produces: `weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]`（昇順の値と、区間中点の累積割合 F）

- [x] **Step 1: 失敗するテストを書く**

```python
class TestWeightedEcdf:
    """重み付き経験分布。`weighted_quantile` と同じ中点流儀なので分位点が逆算で一致する."""

    def test_matches_weighted_quantile(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(1.0, 3.0, 500)
        w = rng.uniform(0.1, 1.0, 500)
        t, F = weighted_ecdf(v, w)
        assert np.all(np.diff(t) >= 0.0)
        assert 0.0 < F[0] < F[-1] < 1.0
        for q in (0.1, 0.5, 0.9):
            assert np.interp(q, F, t) == pytest.approx(float(weighted_quantile(v, w, q)))

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="重み"):
            weighted_ecdf(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
```

import 行に `weighted_ecdf` を足す。

- [x] **Step 2: 失敗を確認**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_extruder_rtd.py -q -k Ecdf`
Expected: `ImportError: cannot import name 'weighted_ecdf'`

- [x] **Step 3: 実装**

```python
def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """重み付き経験分布（昇順の値と区間中点の累積割合）.

    ヒストグラムの F と違ってビン幅に依存しないので、文献曲線との max|ΔF| の
    比較に使う。`weighted_quantile` と同じ中点流儀なので分位点が逆算で一致する。
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0.0:
        msg = "重みの総和が 0 以下"
        raise ValueError(msg)
    return v, (np.cumsum(w) - 0.5 * w) / total
```

- [x] **Step 4: 合格を確認**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_extruder_rtd.py -q -k "Ecdf or Quantile"`
Expected: PASS

- [x] **Step 5: コミット**

```bash
git add xkep_cae_fluid/extruder/rtd.py xkep_cae_fluid/extruder/__init__.py tests/test_extruder_rtd.py
git commit -m "feat(extruder): weighted_ecdf — ビン幅に依らない F 曲線"
```

---

### Task 3: ゲート G5（pytest）

**Files:**
- Create: `tests/test_extruder_literature_rtd.py`

**Interfaces:**
- Consumes: `pinto_tadmor_rtd`, `weighted_ecdf`, 既存 `ExtruderFlowProcess / ParticleTrackerProcess / RTDProcess`
- Produces: モジュール定数 `SHALLOW_SERIES`（H の列）と関数 `reduced_curve(track, rtd) -> (t/t̄_theory, F)`。Task 4 が同じ定義を使う

- [x] **Step 1: テストを書く（最初は閾値で落ちるかもしれない = 実測して判定する）**

```python
"""ゲート G5: Pinto–Tadmor 文献 RTD との照合（浅溝極限への収束）.

設計: docs/design/extruder-g5-literature-rtd.md §3.2
40 mm 機の D, lead, e を保ったまま H を 4 → 2 → 1 mm にして H/W を 0.117 → 0.029 まで
下げる。δ/H = 0.025 固定、G = 0。文献モデルは無限幅なので、ykep の縮約分位点が
単調に近づき、最浅で許容内に入ることを見る。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    RTDInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.pinto_tadmor import pinto_tadmor_rtd
from xkep_cae_fluid.extruder.rtd import RTDProcess, weighted_ecdf
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
Z_AXIAL = 0.050
SHALLOW_SERIES = (0.004, 0.002, 0.001)
"""H [m]。W ≈ 34.1 mm なので H/W = 0.117, 0.059, 0.029."""
DELTA_OVER_H = 0.025
TOL_P10 = TOL_P50 = 0.03
TOL_P90 = 0.05
TOL_CURVE = 0.05


def shallow_spec(H: float, ny: int = 16, n_gap: int = 6, nx_channel: int = 40) -> ScrewSpec:
    return ScrewSpec(
        D=0.040, lead=0.040, H=H, e=0.004, delta=DELTA_OVER_H * H, N=100.0 / 60.0,
        nx_channel=nx_channel, nx_land=12, ny_bulk=ny, n_gap=n_gap,
    )


def pipeline(spec: ScrewSpec, G: float = 0.0, z_axial: float = Z_AXIAL):
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    track = ParticleTrackerProcess().process(ParticleTrackInput(flow=flow, z_axial=z_axial))
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=z_axial, n_bins=100))
    return flow, track, rtd


def reduced_curve(track, rtd) -> tuple[np.ndarray, np.ndarray]:
    """脱出粒子の t/t̄_theory と重み付き ECDF."""
    ok = track.escaped
    t, F = weighted_ecdf(track.t_res[ok] / rtd.t_mean_theory, track.weight[ok])
    return t, F


def quantile_ratios(rtd, pt) -> dict[str, float]:
    """ykep 分位点（t̄_theory 規格化）÷ 文献分位点 − 1 の絶対値."""
    return {
        "p10": abs(rtd.t_p10 / rtd.t_mean_theory / pt.t_p10_ratio - 1.0),
        "p50": abs(rtd.t_p50 / rtd.t_mean_theory / pt.t_p50_ratio - 1.0),
        "p90": abs(rtd.t_p90 / rtd.t_mean_theory / pt.t_p90_ratio - 1.0),
    }


@pytest.fixture(scope="module")
def series():
    pt = pinto_tadmor_rtd(0.0)
    out = []
    for H in SHALLOW_SERIES:
        _, track, rtd = pipeline(shallow_spec(H))
        out.append((H, track, rtd, quantile_ratios(rtd, pt)))
    return pt, out


class TestGateG5:
    def test_shallowest_quantiles_match_literature(self, series):
        _, out = series
        dev = out[-1][3]
        ratios = {"p10": dev["p10"] / TOL_P10, "p50": dev["p50"] / TOL_P50, "p90": dev["p90"] / TOL_P90}
        assert max(ratios.values()) < 1.0, f"閾値規格化比: {ratios}"

    def test_deviation_shrinks_toward_shallow_limit(self, series):
        _, out = series
        for key in ("p10", "p50"):
            devs = [o[3][key] for o in out]
            assert devs[0] > devs[1] > devs[2], f"{key} が単調に近づいていない: {devs}"

    def test_shallowest_curve_matches_up_to_p90(self, series):
        pt, out = series
        _, track, rtd, _ = out[-1]
        t, F = reduced_curve(track, rtd)
        keep = F <= 0.9
        F_pt = np.interp(t[keep], pt.t_over_tbar, pt.F)
        assert float(np.max(np.abs(F[keep] - F_pt))) / TOL_CURVE < 1.0
```

- [x] **Step 2: 実行して実測を見る**

Run: `PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_literature_rtd.py -q --durations=5 2>&1 | tee logs/g5-pytest.log`

判断:
- 3 件 PASS → Step 3 へ。
- `test_shallowest_quantiles_match_literature` だけ落ちて偏差が単調に縮んでいる → 格子を 1 段細かく（ny=32, n_gap=10, nx=80）して再実行。それでも届かなければ Task 4 の H/W=0.015 追加で側壁効果の残りかを切り分け、閾値変更はレポートに理由を書いてから。
- 単調性が落ちる → 追跡の `t_mean_theory` と `Q_axial` を疑い（隙間比 δ/H を固定しているか）、`spec_gap` と同じ流儀で `delta` を確認。

30 s を超えたら `TestGateG5` に `@pytest.mark.slow`。

- [x] **Step 3: コミット**

```bash
git add tests/test_extruder_literature_rtd.py
git commit -m "test(extruder): ゲート G5 — 浅溝極限で Pinto–Tadmor 曲線に収束"
```

---

### Task 4: 実験スクリプト `g5_literature.py`（フル解像度）

**Files:**
- Create: `experiments/extruder/g5_literature.py`

**Interfaces:**
- Consumes: Task 1–3 の関数（テストモジュールからは import せず、同じ定義を本ファイルに持つ）
- Produces: `<out>/result.json` — `{"pt": {...}, "cases": [{"label", "H", "H_over_W", "delta", "G", "r", "n_particles", "seconds", "t_mean_ratio", "quantiles": {"p10","p50","p90"}, "dev": {...}, "curve": {"t": [...], "F": [...]}}], "meta": {...}}`。Task 5 が読む

- [x] **Step 1: 実装**

```python
"""ゲート G5 のフル解像度系列: Pinto–Tadmor 文献 RTD との照合.

    OMP_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python experiments/extruder/g5_literature.py --out /tmp/of-g5

40 mm 機の D, lead, e を保って H = 4, 2, 1 mm（H/W = 0.117 → 0.029）、δ/H = 0.025、G = 0。
加えて観察用に (a) 最浅で r = Q_p/Q_d = −0.3 相当の背圧、(b) 最浅・閉チャネル δ = 0。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ParticleTrackInput, RTDInput, ScrewSpec
from xkep_cae_fluid.extruder.pinto_tadmor import pinto_tadmor_rtd
from xkep_cae_fluid.extruder.rtd import RTDProcess, weighted_ecdf
from xkep_cae_fluid.extruder.shape_factors import shape_factor_drag, shape_factor_pressure
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
Z_AXIAL = 0.200
"""計量部 5D。テストの 50 mm より長い実寸."""
SHALLOW_SERIES = (0.004, 0.002, 0.001)
DELTA_OVER_H = 0.025
R_OBSERVE = -0.3
TOL = {"p10": 0.03, "p50": 0.03, "p90": 0.05, "curve": 0.05}
CURVE_POINTS = 400


def spec_for(H: float, delta: float) -> ScrewSpec:
    return ScrewSpec(
        D=0.040, lead=0.040, H=H, e=0.004, delta=delta, N=100.0 / 60.0,
        nx_channel=200, nx_land=48, ny_bulk=60, n_gap=20 if delta > 0.0 else 0,
    )


def pressure_gradient_for_ratio(spec: ScrewSpec, r: float, mu: float) -> float:
    """Q_p/Q_d = −H²·G·F_p / (6μ·w_barrel·F_d) を G について解く."""
    h = spec.H / spec.W
    return -r * 6.0 * mu * spec.w_barrel * shape_factor_drag(h) / (spec.H**2 * shape_factor_pressure(h))


def run_case(label: str, spec: ScrewSpec, G: float, r: float, pt) -> dict:
    t0 = time.perf_counter()
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    track = ParticleTrackerProcess().process(ParticleTrackInput(flow=flow, z_axial=Z_AXIAL))
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=Z_AXIAL, n_bins=100))
    sec = time.perf_counter() - t0

    ok = track.escaped
    t, F = weighted_ecdf(track.t_res[ok] / rtd.t_mean_theory, track.weight[ok])
    keep = F <= 0.9
    curve_dev = float(np.max(np.abs(F[keep] - np.interp(t[keep], pt.t_over_tbar, pt.F))))
    q = {k: getattr(rtd, f"t_{k}") / rtd.t_mean_theory for k in ("p10", "p50", "p90")}
    dev = {k: abs(q[k] / getattr(pt, f"t_{k}_ratio") - 1.0) for k in q}
    # 曲線は F の等間隔標本で間引く（JSON を小さく保つ）
    Fs = np.linspace(float(F[0]), min(float(F[-1]), 0.995), CURVE_POINTS)
    ts = np.interp(Fs, F, t)
    print(
        f"{label:14s} H/W={spec.H / spec.W:.4f} n={int(ok.sum())} {sec:6.1f}s "
        f"t_mean/theory={rtd.t_mean / rtd.t_mean_theory:.4f} "
        f"p10/p50/p90={q['p10']:.4f}/{q['p50']:.4f}/{q['p90']:.4f} "
        f"dev={dev['p10']:.4f}/{dev['p50']:.4f}/{dev['p90']:.4f} curve={curve_dev:.4f}",
        flush=True,
    )
    return {
        "label": label,
        "H": spec.H,
        "H_over_W": spec.H / spec.W,
        "delta": spec.delta,
        "G": G,
        "r": r,
        "n_particles": int(ok.sum()),
        "seconds": sec,
        "t_mean_ratio": rtd.t_mean / rtd.t_mean_theory,
        "t_min_ratio": rtd.t_min / rtd.t_mean_theory,
        "quantiles": q,
        "dev": dev,
        "curve_dev": curve_dev,
        "ratios": {
            "p10": dev["p10"] / TOL["p10"],
            "p50": dev["p50"] / TOL["p50"],
            "p90": dev["p90"] / TOL["p90"],
            "curve": curve_dev / TOL["curve"],
        },
        "unresolved_weight_fraction": rtd.unresolved_weight_fraction,
        "curve": {"t": ts.tolist(), "F": Fs.tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pt = pinto_tadmor_rtd(0.0)
    Fs = np.linspace(float(pt.F[0]), 0.995, CURVE_POINTS)
    cases = []
    for H in SHALLOW_SERIES:
        spec = spec_for(H, DELTA_OVER_H * H)
        cases.append(run_case(f"gap H={H * 1e3:g}mm", spec, 0.0, 0.0, pt))
    H = SHALLOW_SERIES[-1]
    spec = spec_for(H, DELTA_OVER_H * H)
    G = pressure_gradient_for_ratio(spec, R_OBSERVE, MU)
    cases.append(run_case(f"gap r={R_OBSERVE}", spec, G, R_OBSERVE, pt))
    cases.append(run_case("closed H=1mm", spec_for(H, 0.0), 0.0, 0.0, pt))

    result = {
        "pt": {
            "t_min_ratio": pt.t_min_ratio,
            "t_p10_ratio": pt.t_p10_ratio,
            "t_p50_ratio": pt.t_p50_ratio,
            "t_p90_ratio": pt.t_p90_ratio,
            "curve": {"t": np.interp(Fs, pt.F, pt.t_over_tbar).tolist(), "F": Fs.tolist()},
        },
        "cases": cases,
        "meta": {"mu": MU, "z_axial": Z_AXIAL, "delta_over_H": DELTA_OVER_H, "tol": TOL},
    }
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: 実行（バックグラウンド、ログは logs/）**

Run: `PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/g5_literature.py --out /tmp/of-g5 > logs/g5-literature.log 2>&1`
Expected: 5 ケースが各 1–3 分で終わり、系列 3 本の `ratios` がすべて < 1.0（最浅）。r=−0.3 の分位点が G=0 の最浅と数 % 以内なら「普遍性は有限幅でも保たれる」と書ける。

- [x] **Step 3: コミット**

```bash
git add experiments/extruder/g5_literature.py
git commit -m "feat(extruder): G5 フル解像度系列スクリプト"
```

---

### Task 5: レポート生成 `g5_report.py` と公開

**Files:**
- Create: `experiments/extruder/g5_report.py`
- Create: `docs/reports/extruder/g5-literature-rtd.md`（生成物、コミットする）
- Modify: `docs/reports/extruder/README.md`（行追加）

**Interfaces:**
- Consumes: `/tmp/of-g5/result.json`（Task 4 のスキーマ）

- [x] **Step 1: 実装**

`g3_report.py` の骨格（`_git`, `_load`, `_mark`, `_fmt_ratio`, argparse `--work --out`）を再利用し、以下を組む。

```python
def polyline(t: list[float], F: list[float], x0, y0, w, h, tmax, color, dash="") -> str:
    pts = " ".join(
        f"{x0 + w * min(ti, tmax) / tmax:.1f},{y0 + h * (1.0 - Fi):.1f}" for ti, Fi in zip(t, F)
    )
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"{extra}/>'


def curve_svg(res: dict) -> str:
    x0, y0, w, h, tmax = 60, 20, 640, 300, 3.0
    parts = [
        '<svg viewBox="0 0 760 380" role="img" aria-label="縮約 RTD 曲線 F(t/t̄): 文献モデルと ykep の H/W 系列" style="max-width:100%;height:auto;font-family:sans-serif;font-size:12px">',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="currentColor" opacity="0.5"/>',
    ]
    for k in range(1, 4):  # 縦グリッド t/t̄ = 1, 2, 3
        x = x0 + w * k / tmax
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + h}" stroke="currentColor" opacity="0.15"/>')
        parts.append(f'<text x="{x:.1f}" y="{y0 + h + 16}" text-anchor="middle">{k}</text>')
    for Fk in (0.1, 0.5, 0.9):
        y = y0 + h * (1 - Fk)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" stroke="currentColor" opacity="0.15"/>')
        parts.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{Fk}</text>')
    parts.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">t / t̄</text>')
    parts.append(f'<text x="14" y="{y0 + h / 2}" text-anchor="middle" transform="rotate(-90 14 {y0 + h / 2})">F</text>')
    parts.append(polyline(res["pt"]["curve"]["t"], res["pt"]["curve"]["F"], x0, y0, w, h, tmax, "#B24714"))
    colors = ["#5B7DB1", "#3E8E7E", "#1F1F1F", "#8A6BBE", "#999999"]
    legend_y = y0 + 20
    parts.append(f'<line x1="{x0 + 20}" y1="{legend_y}" x2="{x0 + 50}" y2="{legend_y}" stroke="#B24714" stroke-width="1.8"/>')
    parts.append(f'<text x="{x0 + 56}" y="{legend_y + 4}">Pinto–Tadmor（無限幅）</text>')
    for i, c in enumerate(res["cases"]):
        dash = "6,4" if c["r"] != 0.0 else ("2,3" if c["delta"] == 0.0 else "")
        parts.append(polyline(c["curve"]["t"], c["curve"]["F"], x0, y0, w, h, tmax, colors[i], dash))
        ly = legend_y + 18 * (i + 1)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x0 + 20}" y1="{ly}" x2="{x0 + 50}" y2="{ly}" stroke="{colors[i]}" stroke-width="1.8"{extra}/>')
        parts.append(f'<text x="{x0 + 56}" y="{ly + 4}">ykep {c["label"]}（H/W = {c["H_over_W"]:.3f}）</text>')
    parts.append("</svg>")
    return "\n".join(parts)
```

レポート本文の章立て（spec §4）:
1. 全体像（SVG: 文献モデル ↔ ykep の比較の箱と矢印）、結果表 — 系列 3 本 × {p10, p50, p90, curve} の閾値規格化比と ✅/❌、生の分位点を括弧で添える
2. 縮約曲線の重ね描き（`curve_svg`）
3. メカニズム: (a) なぜ r によらないか（恒等式 3ξ(1−ξ) = ξ − (3ξ²−2ξ) と 1 周の横断変位 0）、(b) 側壁が裾を伸ばす（Bigg & Middleman 1974: 側壁近傍の遅い流線が長時間側に足される。系列で偏差が H/W にほぼ比例して縮むことを表で示す）、(c) 隙間が短時間側を広げる（closed ケースとの差）、(d) 実測との鎖（Wolf & White 1976 の放射性トレーサ実験が計量部について Pinto–Tadmor 曲線を確認、と記す。数値の転記はしない）、(e) t_min/t̄ = 3/4 の意味（最速は停留高さ ξ = 2/3 の粒子）
4. 再現手順: `g5_literature.py --out /tmp/of-g5`（所要時間を実測で書く）→ `g5_report.py --work /tmp/of-g5 --out docs/reports/extruder/g5-literature-rtd.md`

- [x] **Step 2: 生成 → mdview → Artifact**

```bash
PYTHONPATH=. .venv/bin/python experiments/extruder/g5_report.py --work /tmp/of-g5 --out docs/reports/extruder/g5-literature-rtd.md
~/work/tb/bin/mdview docs/reports/extruder/g5-literature-rtd.md
```

`/tmp/mdview/g5-literature-rtd.html` の `<title>` `<style>` と `<body>` の中身だけを `docs/reports/extruder/g5-literature-rtd.html` に取り出し、`Artifact` で公開（favicon 📚）。URL を `docs/reports/extruder/README.md` の表に追加（G3 行と同じ書式）。

- [x] **Step 3: コミット**

```bash
git add experiments/extruder/g5_report.py docs/reports/extruder/g5-literature-rtd.md docs/reports/extruder/g5-literature-rtd.html docs/reports/extruder/README.md
git commit -m "docs(extruder): G5 文献 RTD 照合レポート"
```

---

### Task 6: 文書更新と PR

**Files:**
- Modify: `docs/design/single-screw-extruder.md`（§3 表に G5 行、§7 Phase 2 前提）
- Modify: `docs/roadmap.md`（Phase 7 の `[ ] 実機データとの突き合わせ` 行）
- Modify: `docs/plans/2026-09-02-single-screw-extruder-impl.md`（§D 前提文）
- Create: `docs/status/status-29.md`、Modify: `docs/status/status-index.md`, `README.md`

- [x] **Step 1: 設計文書 §3 表に行を足す**

```
| **G5** | 文献 RTD（Pinto & Tadmor 1970）との照合 | 浅溝極限 H/W → 0 で縮約曲線 F(t/t̄) が収束。詳細 `extruder-g5-literature-rtd.md` |
```

§7 の Phase 2 行の前提を「実機データ突き合わせ」→「G5 文献照合 ✅（実機・想定機が無いため差し替え、2026-09-04）」に。

- [x] **Step 2: roadmap / plan §D**

roadmap: `- [x] 実機データとの突き合わせ（Phase 2 の前提）` → `- [x] 文献 RTD 照合 G5（Phase 2 の前提。実機データ突き合わせを差し替え）`。
plan §D: 「Phase 2 に入る前に…実機データと突き合わせること」の段落を「実機・想定機が無いため、文献照合 G5（`docs/design/extruder-g5-literature-rtd.md`）に差し替えた（2026-09-04）」に。

- [x] **Step 3: status-29**

`status-28.md` の型で: ヘッダ（日付・ブランチ・テスト数）、G5 の結果表（閾値規格化比）、確定した設計上の論点（下層標本化、t_min/t̄ = 3/4、普遍性の恒等式、側壁効果の H/W 比例）、資源表（各ケースの秒数）、次にやること（Phase 2）。`status-index.md` に行追加、`README.md` のテスト数・日付更新。

- [x] **Step 4: 全テスト → ruff → コミット → push → PR**

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q -x --ignore=tests/test_benchmark_runner.py 2>&1 | tail -5
.venv/bin/ruff check xkep_cae_fluid/extruder experiments/extruder tests/test_extruder_pinto_tadmor.py tests/test_extruder_literature_rtd.py && .venv/bin/ruff format --check xkep_cae_fluid/extruder experiments/extruder tests
git add -A docs README.md && git commit -m "docs: status-29 — ゲート G5 で Phase 2 の前提を文献照合に差し替え"
git push -u origin claude/single-screw-extruder-impl
gh pr create --title "feat(extruder): ゲート G5 — Pinto–Tadmor 文献 RTD との照合" --body "..."
```

（PR #28 が同ブランチで open なら push だけで PR に載る。その場合は `gh pr comment` で追記。）
