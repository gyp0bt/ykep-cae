"""nsb: 2D Brinkman 補正 Navier-Stokes（FVM, Newton–Krylov）の比較実験用パッケージ.

手元コードの構成（core / solver / utils / geo）に合わせた薄いレイヤ。
離散化（残差・1 次風上ヤコビアン・Rhie–Chow）は `nsb.assembly.BrinkmanDiscretization`、
境界条件・入力型は `nsb.data` に持ち、`xkep_cae_fluid` には依存しない（numpy / scipy のみ）。
`nsb/{data,assembly}.py` は `xkep_cae_fluid/brinkman_flow/{data,assembly}.py` の**コピー**
（import 行のみ差し替え）。同期は `python scripts/sync_nsb_from_xkep.py`、
乖離検出は `tests/test_nsb_standalone.py`。
Newton + 擬似時間の制御則だけを `solver.solve_steady` に明示的に書き下している。
"""

from nsb.adjoint import (
    ImplicitSolve,
    Objective,
    colored_fd_jacobian,
    source_mean_pressure_objective,
)
from nsb.assembly import BrinkmanDiscretization, StateArrays
from nsb.core import BC, FaceType, NSBInput, NSBResult, NSBSettings
from nsb.data import (
    BoundaryKind,
    BoundaryPatch,
    ConvectionSchemeType,
    MaskFn,
    WeightFn,
    disk_mask,
    east_span,
    north_span,
    rect_mask,
    smooth_disk,
    south_span,
    west_span,
)
from nsb.geo import make_case, make_flat_h, make_uturn_h, run_flat, run_uturn, uturn_bc_preset
from nsb.solver import solve_steady

__all__ = [
    "BC",
    "BoundaryKind",
    "BoundaryPatch",
    "BrinkmanDiscretization",
    "ConvectionSchemeType",
    "FaceType",
    "ImplicitSolve",
    "MaskFn",
    "NSBInput",
    "NSBResult",
    "NSBSettings",
    "Objective",
    "StateArrays",
    "WeightFn",
    "colored_fd_jacobian",
    "disk_mask",
    "east_span",
    "make_case",
    "make_flat_h",
    "make_uturn_h",
    "north_span",
    "rect_mask",
    "run_flat",
    "run_uturn",
    "smooth_disk",
    "solve_steady",
    "source_mean_pressure_objective",
    "south_span",
    "uturn_bc_preset",
    "west_span",
]
