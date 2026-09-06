"""Darcy 流れ（多孔質媒体の圧力ポアソン方程式）ソルバー（面ベース FVM）.

``*DARCY`` 方程式ファミリー。設計は ``docs/design/darcy-flow-fvm.md``。
"""

from xkep_cae_fluid.darcy.data import (
    DarcyBCKind,
    DarcyFlowInput,
    DarcyFlowResult,
    DarcyPatchBC,
)
from xkep_cae_fluid.darcy.solver import DarcyFlowProcess

__all__ = [
    "DarcyBCKind",
    "DarcyPatchBC",
    "DarcyFlowInput",
    "DarcyFlowResult",
    "DarcyFlowProcess",
]
