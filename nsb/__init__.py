"""nsb: 2D Brinkman 補正 Navier-Stokes（FVM, Newton–Krylov）の比較実験用パッケージ.

手元コードの構成（core / solver / utils / geo）に合わせた薄いレイヤ。
離散化（残差・1 次風上ヤコビアン・Rhie–Chow）は
`xkep_cae_fluid.brinkman_flow.assembly.BrinkmanDiscretization` を共有し、
Newton + 擬似時間の制御則だけを `solver.solve_steady` に明示的に書き下している。
"""

from nsb.adjoint import (
    ImplicitSolve,
    Objective,
    colored_fd_jacobian,
    source_mean_pressure_objective,
)
from nsb.core import BC, FaceType, NSBInput, NSBResult, NSBSettings
from nsb.geo import make_case, make_flat_h, make_uturn_h, run_flat, run_uturn, uturn_bc_preset
from nsb.solver import solve_steady

__all__ = [
    "BC",
    "ImplicitSolve",
    "Objective",
    "colored_fd_jacobian",
    "source_mean_pressure_objective",
    "FaceType",
    "NSBInput",
    "NSBResult",
    "NSBSettings",
    "make_case",
    "make_flat_h",
    "make_uturn_h",
    "run_flat",
    "run_uturn",
    "solve_steady",
    "uturn_bc_preset",
]
