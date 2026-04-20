"""汎用スカラー輸送モジュール.

任意のスカラー量（CO2/O2 濃度、トレーサー、塩分等）を
既存流体場の上で対流-拡散-ソース方程式として輸送する。

Phase 6.1a（水槽 CAE ロードマップ）で新設。
"""

from xkep_cae_fluid.scalar_transport.data import (
    ExtraScalarSpec,
    ScalarBoundaryCondition,
    ScalarBoundarySpec,
    ScalarFieldSpec,
    ScalarTransportInput,
    ScalarTransportResult,
)
from xkep_cae_fluid.scalar_transport.solver import ScalarTransportProcess

__all__ = [
    "ExtraScalarSpec",
    "ScalarBoundaryCondition",
    "ScalarBoundarySpec",
    "ScalarFieldSpec",
    "ScalarTransportInput",
    "ScalarTransportResult",
    "ScalarTransportProcess",
]
