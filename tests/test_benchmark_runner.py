"""BenchmarkRunnerProcess テスト."""

from dataclasses import dataclass
from typing import ClassVar

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.benchmark import BenchmarkRunInput, BenchmarkRunnerProcess
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.testing import binds_to


@dataclass(frozen=True)
class _DummyInput:
    value: float = 1.0


@dataclass(frozen=True)
class _DummyOutput:
    result: float = 2.0


class _DummySolverProcess(SolverProcess["_DummyInput", "_DummyOutput"]):
    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummySolverProcess",
        module="solve",
        version="0.1.0",
        document_path="../xkep_cae_fluid/core/docs/benchmark_runner.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []
    _skip_registry = True

    def process(self, input_data: _DummyInput) -> _DummyOutput:
        return _DummyOutput(result=input_data.value * 2)


@binds_to(BenchmarkRunnerProcess)
class TestBenchmarkRunnerProcess:
    """BenchmarkRunnerProcess の契約テスト."""

    def test_basic_execution(self):
        runner = BenchmarkRunnerProcess()
        dummy = _DummySolverProcess()
        config = _DummyInput(value=3.0)

        result = runner.process(
            BenchmarkRunInput(
                process=dummy,
                config=config,
                result_extractors={"result": lambda r: r.result},
                output_dir="/tmp/xkep_fluid_benchmark_test",
            )
        )

        assert result.result.result == 6.0
        assert result.manifest.process_name == "_DummySolverProcess"
        assert result.manifest.results_summary["result"] == 6.0
        assert result.manifest.elapsed_seconds >= 0

    def test_manifest_yaml(self):
        runner = BenchmarkRunnerProcess()
        dummy = _DummySolverProcess()
        config = _DummyInput(value=1.0)

        result = runner.process(
            BenchmarkRunInput(
                process=dummy,
                config=config,
                output_dir="/tmp/xkep_fluid_benchmark_test",
            )
        )

        yaml_str = result.manifest.to_yaml()
        assert "process:" in yaml_str
        assert "_DummySolverProcess" in yaml_str
