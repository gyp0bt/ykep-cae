"""コアモジュールのインポートテスト."""

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta, ProcessMetaclass
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
    ProcessExecutionLog,
)
from xkep_cae_fluid.core.registry import ProcessRegistry
from xkep_cae_fluid.core.runner import ExecutionContext, ProcessRunner
from xkep_cae_fluid.core.slots import StrategySlot
from xkep_cae_fluid.core.tree import NodeType, ProcessNode, ProcessTree


class TestCoreImport:
    """core モジュールが正しくインポートできることを確認."""

    def test_abstract_process(self):
        assert AbstractProcess is not None
        assert ProcessMeta is not None
        assert ProcessMetaclass is not None

    def test_categories(self):
        assert PreProcess is not None
        assert SolverProcess is not None
        assert PostProcess is not None
        assert VerifyProcess is not None
        assert BatchProcess is not None

    def test_data_schemas(self):
        assert MeshData is not None
        assert BoundaryData is not None
        assert FluidProperties is not None
        assert FlowFieldData is not None
        assert SolverInputData is not None
        assert SolverResultData is not None
        assert VerifyInput is not None
        assert VerifyResult is not None

    def test_registry(self):
        reg = ProcessRegistry.default()
        assert reg is not None

    def test_runner(self):
        ctx = ExecutionContext()
        runner = ProcessRunner(ctx)
        assert runner is not None

    def test_slots(self):
        assert StrategySlot is not None

    def test_diagnostics(self):
        log = ProcessExecutionLog.instance()
        assert log is not None

    def test_tree(self):
        assert ProcessTree is not None
        assert ProcessNode is not None
        assert NodeType is not None


class TestStrategyProtocols:
    """Strategy Protocol が正しくインポートできることを確認."""

    def test_convection_scheme(self):
        from xkep_cae_fluid.core.strategies.protocols import ConvectionSchemeStrategy

        assert ConvectionSchemeStrategy is not None

    def test_diffusion_scheme(self):
        from xkep_cae_fluid.core.strategies.protocols import DiffusionSchemeStrategy

        assert DiffusionSchemeStrategy is not None

    def test_time_integration(self):
        from xkep_cae_fluid.core.strategies.protocols import TimeIntegrationStrategy

        assert TimeIntegrationStrategy is not None

    def test_turbulence_model(self):
        from xkep_cae_fluid.core.strategies.protocols import TurbulenceModelStrategy

        assert TurbulenceModelStrategy is not None

    def test_pv_coupling(self):
        from xkep_cae_fluid.core.strategies.protocols import PressureVelocityCouplingStrategy

        assert PressureVelocityCouplingStrategy is not None

    def test_linear_solver(self):
        from xkep_cae_fluid.core.strategies.protocols import LinearSolverStrategy

        assert LinearSolverStrategy is not None
