"""プロセスアーキテクチャ基盤（流体解析向け）.

AbstractProcess + Strategy Protocol によるソルバー契約化フレームワーク。
xkep-cae と共通の Process Architecture を FDM/FVM 向けに適応。
"""

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta, ProcessMetaclass
from xkep_cae_fluid.core.benchmark import (
    BenchmarkRunInput,
    BenchmarkRunnerProcess,
    BenchmarkRunResult,
    RunManifest,
    capture_environment,
    serialize_config,
)
from xkep_cae_fluid.core.categories import (
    BatchProcess,
    PostProcess,
    PreProcess,
    SolverProcess,
    VerifyProcess,
)
from xkep_cae_fluid.core.data import (
    BoundaryData,
    FlowFieldData,
    FluidProperties,
    MeshData,
    SolverInputData,
    SolverResultData,
    VerifyInput,
    VerifyResult,
)
from xkep_cae_fluid.core.diagnostics import (
    DeprecatedProcessError,
    NonDefaultStrategyWarning,
    ProcessExecutionLog,
)
from xkep_cae_fluid.core.mesh import (
    StructuredMeshInput,
    StructuredMeshProcess,
    StructuredMeshResult,
)
from xkep_cae_fluid.core.registry import ProcessRegistry
from xkep_cae_fluid.core.runner import ExecutionContext, ProcessRunner
from xkep_cae_fluid.core.slots import StrategySlot, collect_strategy_slots, collect_strategy_types
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.core.tree import NodeType, ProcessNode, ProcessTree

__all__ = [
    "AbstractProcess",
    "ProcessMeta",
    "ProcessMetaclass",
    "PreProcess",
    "SolverProcess",
    "PostProcess",
    "VerifyProcess",
    "BatchProcess",
    "binds_to",
    "ProcessTree",
    "ProcessNode",
    "NodeType",
    "MeshData",
    "BoundaryData",
    "FluidProperties",
    "FlowFieldData",
    "SolverInputData",
    "SolverResultData",
    "VerifyInput",
    "VerifyResult",
    "ProcessRegistry",
    "ProcessRunner",
    "ExecutionContext",
    "BenchmarkRunnerProcess",
    "BenchmarkRunInput",
    "BenchmarkRunResult",
    "RunManifest",
    "capture_environment",
    "serialize_config",
    "StrategySlot",
    "collect_strategy_slots",
    "collect_strategy_types",
    "ProcessExecutionLog",
    "NonDefaultStrategyWarning",
    "DeprecatedProcessError",
    "StructuredMeshProcess",
    "StructuredMeshInput",
    "StructuredMeshResult",
]
