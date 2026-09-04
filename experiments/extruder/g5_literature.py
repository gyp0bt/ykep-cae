"""ゲート G5 の実験: 文献 RTD（Pinto–Tadmor）との照合を条件ごとに走らせて result.json に落とす.

    PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/g5_literature.py --out /tmp/of-g5

設計: docs/design/extruder-g5-literature-rtd.md §3.3。ゲートと同じ格子（32×80）・cfl = 0.1 を
基準に、文献モデルの仮定を 1 つずつ外した系列を並べる。

- series   : H = 4, 2, 1 mm、閉チャネル、z = 0.5 m（ゲートそのもの）
- length   : H = 1 mm、z = 0.05, 0.2, 1.0 m（周回数の効果。0.5 は series と共有）
- gap      : H = 1 mm、δ/H = 0.025（隙間の速い経路）
- cfl      : H = 1 mm、cfl = 1.0（流線ドリフト）
- pressure : H = 1 mm、r = Q_p/Q_d = −0.3 相当の背圧（普遍性）

所要 6〜8 分（2 スレッド）。ログは logs/ に。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from dataclasses import replace

import numpy as np

from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    RTDInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.pinto_tadmor import (
    XI_SPLIT,
    _upper_partner,
    compare_rtd,
    pinto_tadmor_rtd,
)
from xkep_cae_fluid.extruder.rtd import RTDProcess
from xkep_cae_fluid.extruder.shape_factors import shape_factor_drag, shape_factor_pressure
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

MU = 1000.0
Z_AXIAL = 0.5
CFL = 0.1
NY, NX = 32, 80
F_CURVE = np.linspace(0.0025, 0.9975, 400)
"""曲線を書き出す F の格子（分位関数 t(F) として保存する）."""

_BASE = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0)


def closed_spec(H: float) -> ScrewSpec:
    return replace(_BASE, H=H, delta=0.0, nx_channel=NX, nx_land=8, ny_bulk=NY, n_gap=0)


def gap_spec(H: float, delta_over_h: float) -> ScrewSpec:
    return replace(
        _BASE, H=H, delta=delta_over_h * H, nx_channel=NX, nx_land=12, ny_bulk=NY, n_gap=10
    )


def pressure_gradient_for_ratio(spec: ScrewSpec, r: float, mu: float) -> float:
    """Q_p/Q_d = r になる下流方向圧力勾配 G [Pa/m].

    Q_d = V_z H W F_d / 2、Q_p = −H³ W G F_p / (12 μ) より
    G = −r · 6 μ V_z F_d / (H² F_p)。r < 0 で G > 0（背圧）。
    """
    h = spec.H / spec.W
    return (
        -r
        * 6.0
        * mu
        * spec.w_barrel
        * shape_factor_drag(h)
        / (spec.H**2 * shape_factor_pressure(h))
    )


def loop_period_median(spec: ScrewSpec) -> float:
    """文献モデルで流量の中央値を担う流線対の周回時間 [s].

    流線対 (ξ_c, ξ) の周回時間は W/|u(ξ_c)| + W/u(ξ)。流量の累積が 1/2 になる
    ξ_c を取る。H に依らないので z_axial を固定すれば周回数は H 系列で揃う。
    """
    xi_c = np.linspace(0.0, XI_SPLIT, 4001)[1:-1]
    xi = _upper_partner(xi_c)
    u_lo = 2.0 * xi_c - 3.0 * xi_c * xi_c
    u_up = 3.0 * xi * xi - 2.0 * xi
    dq = (xi_c + xi * u_lo / u_up) * (xi_c[1] - xi_c[0])  # r = 0 の流線対流量（ŵ = ξ）
    cum = np.cumsum(dq) / dq.sum()
    k = int(np.searchsorted(cum, 0.5))
    v_x = abs(spec.u_barrel)
    return spec.W / v_x * (1.0 / u_lo[k] + 1.0 / u_up[k])


def run_case(
    label: str,
    spec: ScrewSpec,
    *,
    z_axial: float = Z_AXIAL,
    cfl: float = CFL,
    r: float = 0.0,
    pt,
) -> dict:
    G = pressure_gradient_for_ratio(spec, r, MU) if r != 0.0 else 0.0
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=MU)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    t0 = time.perf_counter()
    track = ParticleTrackerProcess().process(
        ParticleTrackInput(flow=flow, z_axial=z_axial, cfl=cfl)
    )
    wall = time.perf_counter() - t0
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=z_axial, n_bins=100))

    h = spec.H / spec.W
    f_d = shape_factor_drag(h)
    f_p = shape_factor_pressure(h)
    # 側壁の無い平均滞留時間 t̄_∞ = t̄_theory · Q/Q_∞、Q = Q_d∞(F_d + r F_p)、Q_∞ = Q_d∞(1 + r)
    t_ref = rtd.t_mean_theory * (f_d + r * f_p) / (1.0 + r)
    ok = track.escaped
    cmp = compare_rtd(track.t_res[ok], track.weight[ok], t_ref, pt)
    t_loop = loop_period_median(spec)
    n = rtd.t_mean_theory
    print(
        f"[{label}] H={spec.H * 1e3:g}mm delta/H={spec.delta / spec.H:.3f} z={z_axial} cfl={cfl} "
        f"r={r} G={G:.3g} {wall:.0f}s  loops={t_ref / t_loop:.1f}  mean/th={rtd.t_mean / n:.3f} "
        f"[/t_ref] tmin={rtd.t_min / t_ref:.4f} p10={cmp.p10_ratio:.4f} p50={cmp.p50_ratio:.4f} "
        f"p90={cmp.p90_ratio:.4f} dev={cmp.dev_p10:.4f}/{cmp.dev_p50:.4f}/{cmp.dev_p90:.4f} "
        f"L1={cmp.curve_l1:.4f} max={cmp.curve_max:.4f} ext={track.weight[track.extrapolated].sum() / track.weight.sum():.4f}",
        flush=True,
    )
    return {
        "label": label,
        "H": spec.H,
        "W": spec.W,
        "H_over_W": h,
        "F_d": f_d,
        "F_p": f_p,
        "delta_over_H": spec.delta / spec.H,
        "z_axial": z_axial,
        "cfl": cfl,
        "r": r,
        "G": G,
        "grid": [spec.ny_bulk, spec.nx_channel, spec.n_gap],
        "n_particles": int(track.weight.size),
        "wall_s": wall,
        "t_loop_median": t_loop,
        "n_loops": t_ref / t_loop,
        "t_mean_theory": rtd.t_mean_theory,
        "t_ref": t_ref,
        "t_mean_over_theory": rtd.t_mean / n,
        "t_min_over_ref": rtd.t_min / t_ref,
        "p10_over_ref": cmp.p10_ratio,
        "p50_over_ref": cmp.p50_ratio,
        "p90_over_ref": cmp.p90_ratio,
        "p10_over_theory": rtd.t_p10 / n,
        "p50_over_theory": rtd.t_p50 / n,
        "p90_over_theory": rtd.t_p90 / n,
        "dev_p10": cmp.dev_p10,
        "dev_p50": cmp.dev_p50,
        "dev_p90": cmp.dev_p90,
        "curve_l1": cmp.curve_l1,
        "curve_max": cmp.curve_max,
        "f_range": list(cmp.f_range),
        "unresolved_weight_fraction": rtd.unresolved_weight_fraction,
        "extrapolated_weight_fraction": float(
            track.weight[track.extrapolated].sum() / track.weight.sum()
        ),
        "curve_F": F_CURVE.tolist(),
        "curve_t": np.interp(F_CURVE, cmp.F, cmp.t_over_tbar).tolist(),
    }


def _git(*args: str) -> str:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="粗格子 16×40 で全ケース（動作確認用）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    global NY, NX
    if args.quick:
        NY, NX = 16, 40

    pt = pinto_tadmor_rtd(0.0)
    cases: list[dict] = []

    def add(label, spec, **kw):
        cases.append(run_case(label, spec, pt=pt, **kw))

    for H in (0.004, 0.002, 0.001):
        add("series", closed_spec(H))
    for z in (0.05, 0.2, 1.0):
        add("length", closed_spec(0.001), z_axial=z)
    add("gap", gap_spec(0.001, 0.025))
    add("cfl", closed_spec(0.001), cfl=1.0)
    add("pressure", closed_spec(0.001), r=-0.3)

    result = {
        "meta": {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "mu": MU,
            "grid": [NY, NX],
            "cfl": CFL,
            "z_axial": Z_AXIAL,
            "quick": args.quick,
            "base_spec": {
                "D": _BASE.D,
                "lead": _BASE.lead,
                "e": _BASE.e,
                "N": _BASE.N,
                "phi_deg": math.degrees(_BASE.phi),
                "W": _BASE.W,
                "V": _BASE.V,
                "u_barrel": _BASE.u_barrel,
                "w_barrel": _BASE.w_barrel,
            },
        },
        "pt": {
            "r": 0.0,
            "t_min_ratio": pt.t_min_ratio,
            "t_p10_ratio": pt.t_p10_ratio,
            "t_p50_ratio": pt.t_p50_ratio,
            "t_p90_ratio": pt.t_p90_ratio,
            "curve_F": F_CURVE.tolist(),
            "curve_t": np.interp(F_CURVE, pt.F, pt.t_over_tbar).tolist(),
        },
        "cases": cases,
    }
    path = os.path.join(args.out, "result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
