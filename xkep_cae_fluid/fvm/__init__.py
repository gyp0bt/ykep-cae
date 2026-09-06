"""面ベース FVM の共通低レイヤー（方程式ファミリー非依存）.

``MeshData``（構造格子 / polyMesh / .inp 由来を問わない）の面リストの上で

- 境界パッチ条件の解決（:mod:`boundary`）
- 面補間・面物性・勾配などの幾何演算（:mod:`geometry`）
- 拡散・対流（1 次風上 + TVD 遅延補正）・時間項（Euler / BDF2）・ソース項の係数行列組み立てと非直交補正（:mod:`assembly`）
- 線形ソルバー Strategy（:mod:`linear`）

を提供する。各方程式ファミリー（スカラー輸送、Darcy、伝熱 …）の SolverProcess は
ここを組み合わせて書く薄い層になる。設計は
``docs/design/fvm-layer.md`` を参照。
"""

from xkep_cae_fluid.fvm.assembly import (
    CONVECTION_SCHEMES,
    TVD_LIMITERS,
    assemble_convection,
    assemble_diffusion,
    assemble_scalar_transport,
    boundary_tangent,
    convection_correction,
    diffusive_face_flux,
    nonorthogonal_correction,
    solve_corrected,
    time_derivative_terms,
    tvd_deferred_correction,
)
from xkep_cae_fluid.fvm.boundary import BCKind, BoundaryFaces, PatchBC, resolve_boundary
from xkep_cae_fluid.fvm.geometry import (
    boundary_face_values,
    cell_gradient,
    cell_gradient_lsq,
    face_decomposition,
    face_diffusivity,
    face_gradient,
    face_interpolation_weights,
    face_mass_flux,
    face_skewness,
    internal_face_values,
    is_orthogonal,
    max_nonorthogonality_deg,
)
from xkep_cae_fluid.fvm.linear import (
    AMGSolver,
    BiCGSTABSolver,
    DirectSolver,
    make_linear_solver,
    relative_residual,
)

__all__ = [
    "CONVECTION_SCHEMES",
    "TVD_LIMITERS",
    "BCKind",
    "PatchBC",
    "BoundaryFaces",
    "resolve_boundary",
    "face_interpolation_weights",
    "face_diffusivity",
    "face_mass_flux",
    "internal_face_values",
    "boundary_face_values",
    "cell_gradient",
    "cell_gradient_lsq",
    "face_gradient",
    "face_decomposition",
    "face_skewness",
    "is_orthogonal",
    "max_nonorthogonality_deg",
    "assemble_diffusion",
    "assemble_convection",
    "assemble_scalar_transport",
    "nonorthogonal_correction",
    "boundary_tangent",
    "diffusive_face_flux",
    "tvd_deferred_correction",
    "convection_correction",
    "time_derivative_terms",
    "solve_corrected",
    "DirectSolver",
    "BiCGSTABSolver",
    "AMGSolver",
    "make_linear_solver",
    "relative_residual",
]
