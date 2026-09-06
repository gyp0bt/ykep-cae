"""汎用記法の .inp（展開チャネル 2.5D）から滞留時間分布と混練性指標を出す例.

``examples/inp/extruder-channel-1.inp`` を解き、面流束ベースの粒子追跡
（:class:`~xkep_cae_fluid.post.tracking.ParticleTrackFVMProcess`）で ζ（軸方向座標）が
``z_axial`` に達するまで追い、:class:`~xkep_cae_fluid.post.rtd.ResidenceTimeProcess` で
E(t) / F(t) と累積せん断ひずみ γ = ∫γ̇ dt・混合指数 λ を集計する。

構造格子専用の 2.5D ソルバー + 流れ関数トラッカー（ゲート G4a/G4b/G5 通過済み）を
リファレンスとして同じ量を出し、並べて表示する。**原理も実装も別物**なので、
両者が揃えば汎用経路の妥当性が言える。

実行::

    python examples/extruder_generic_rtd.py 2>&1 | tee examples/inp/results/extruder-channel-1-rtd.log
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ParticleTrackInput, RTDInput, ScrewSpec
from xkep_cae_fluid.extruder.rtd import RTDProcess
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.fvm.viscosity import NewtonianViscosity
from xkep_cae_fluid.incompressible import NavierStokesFVMProcess
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mapping import InpMeshMappingInput, InpToNavierStokesFVMProcess
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess
from xkep_cae_fluid.inp.parser import parse_inp_text
from xkep_cae_fluid.post.rtd import ResidenceTimeInput, ResidenceTimeProcess
from xkep_cae_fluid.post.tracking import ParticleTrackFVMInput, ParticleTrackFVMProcess

INP = Path(__file__).with_name("inp") / "extruder-channel-1.inp"
Z_AXIAL = 0.05
"""計量部の軸方向長さ [m]。粒子は ζ >= Z_AXIAL で脱出したとみなす."""

# examples/inp/extruder-channel-1.inp を書いたときの諸元（2016 セル）。
# 参照側の構造格子を**同じ解像度**にしないと、比較が離散化差に埋もれる。
SPEC = ScrewSpec(
    D=0.04,
    lead=0.04,
    H=0.004,
    e=0.004,
    delta=2.0e-4,
    N=1.0,
    nx_channel=60,
    nx_land=12,
    ny_bulk=24,
    n_gap=8,
)
N_CELLS = 2016
MU = 1000.0
G = 1.0e5


def run_generic() -> tuple[object, object]:
    """.inp をそのまま解いて追跡する（汎用経路）."""
    case = build_case(parse_inp_text(INP.read_text(encoding="utf-8")))
    mesh = InpMeshProcess().execute(InpMeshInput(case=case))
    ns = InpToNavierStokesFVMProcess().execute(
        InpMeshMappingInput(case=case, mesh=mesh, step_index=0)
    )
    t0 = time.perf_counter()
    flow = NavierStokesFVMProcess().execute(ns)
    print(
        f"[汎用] 解 {time.perf_counter() - t0:.2f} s  セル {mesh.mesh.n_cells}  収束 {flow.converged}"
    )
    if not flow.converged:
        raise RuntimeError("汎用経路が収束しませんでした（この結果は使えません）")

    t0 = time.perf_counter()
    track = ParticleTrackFVMProcess().execute(
        ParticleTrackFVMInput(
            mesh=mesh.mesh,
            face_flux=flow.mass_flux,
            density=ns.rho,
            seed="axial",
            axis=(math.cos(SPEC.phi), 0.0, math.sin(SPEC.phi)),
            length=Z_AXIAL,
            max_steps=60_000,
            scalars={"gamma": flow.strain_rate, "lam": flow.mixing_index},
        )
    )
    print(
        f"[汎用] 追跡 {time.perf_counter() - t0:.1f} s  粒子 {track.n_particles}"
        f"  ステップ {track.n_steps.min()}..{track.n_steps.max()}"
        f"  x 巻き戻し最大 {np.abs(track.shift_total[:, 0]).max() / SPEC.W_t:.1f} 周"
    )
    return track, ResidenceTimeProcess().execute(
        ResidenceTimeInput(track=track, rate_scalars=("lam",))
    )


def run_reference() -> object:
    """構造格子専用の 2.5D ソルバー + 流れ関数トラッカー（リファレンス）."""
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(MU)
    t0 = time.perf_counter()
    flow = proc.execute(ExtruderFlowInput(spec=SPEC, G=G))
    if not flow.converged:
        raise RuntimeError("専用ソルバーが収束しませんでした")
    track = ParticleTrackerProcess().execute(
        ParticleTrackInput(flow=flow, z_axial=Z_AXIAL, cfl=0.25, max_steps=50_000)
    )
    print(f"[参照] 解 + 追跡 {time.perf_counter() - t0:.1f} s  粒子 {track.weight.size}")
    return RTDProcess().execute(RTDInput(track=track, flow=flow, z_axial=Z_AXIAL))


def main() -> None:
    print(
        f"諸元: D={SPEC.D * 1e3:.0f} mm, H={SPEC.H * 1e3:.1f} mm, δ={SPEC.delta * 1e3:.1f} mm, "
        f"N={SPEC.N} 1/s, φ={math.degrees(SPEC.phi):.2f}°, μ={MU} Pa·s, G={G:.0e} Pa/m, "
        f"z_axial={Z_AXIAL} m"
    )
    _, gen = run_generic()
    ref = run_reference()

    rows = [
        ("⟨t⟩ [s]", gen.t_mean, ref.t_mean),
        ("⟨t⟩ 理論 [s]", gen.t_mean_theory, ref.t_mean_theory),
        ("t_p10 [s]", gen.t_p10, ref.t_p10),
        ("t_p50 [s]", gen.t_p50, ref.t_p50),
        ("t_p90 [s]", gen.t_p90, ref.t_p90),
        ("広がり t_p90/t_p10", gen.spread, ref.spread),
        ("γ = ∫γ̇dt", gen.scalar_mean["gamma"], ref.gamma_mean),
        ("混合指数 λ", gen.scalar_mean["lam"], ref.lambda_mean),
    ]
    print(f"\n{'量':<22}{'汎用':>14}{'参照':>14}{'相対差':>12}")
    for name, a, b in rows:
        print(f"{name:<22}{a:>14.6g}{b:>14.6g}{abs(a / b - 1.0):>12.2e}")
    print(
        f"\n外挿で閉じた重み割合: 汎用 {gen.extrapolated_weight_fraction:.3g} / "
        f"参照 {ref.extrapolated_weight_fraction:.3g}"
    )
    print(
        f"未解決の重み割合:     汎用 {gen.unresolved_weight_fraction:.3g} / "
        f"参照 {ref.unresolved_weight_fraction:.3g}"
    )
    print(
        "\n⟨t⟩ と理論値 length·V/Σw の一致は、種まき重み・面の受け渡し・周期の巻き戻し・"
        "脱出時刻の内挿が\n全て整合しているときにしか成り立たない（RTD の最も鋭い検査）。"
    )


if __name__ == "__main__":
    main()
