"""3次元非定常伝熱解析モジュール (FDM)."""

from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
    HeatTransferResult,
)
from xkep_cae_fluid.heat_transfer.solver import HeatTransferFDMProcess

__all__ = [
    "BoundaryCondition",
    "BoundarySpec",
    "HeatTransferFDMProcess",
    "HeatTransferInput",
    "HeatTransferResult",
]
