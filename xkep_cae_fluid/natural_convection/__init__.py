"""3次元自然対流解析モジュール (FDM + SIMPLE法).

Boussinesq近似による非圧縮性自然対流を、等間隔直交格子上の
FDM + SIMPLE法で解く。固体-流体練成伝熱にも対応。
"""

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    NaturalConvectionResult,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess

__all__ = [
    "FluidBoundaryCondition",
    "FluidBoundarySpec",
    "NaturalConvectionFDMProcess",
    "NaturalConvectionInput",
    "NaturalConvectionResult",
    "ThermalBoundaryCondition",
]
