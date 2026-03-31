"""BenchmarkRunner -- 実行マニフェスト自動記録.

STA2防止のため、プロセス実行時の全パラメータ・環境情報・結果サマリーを
自動でYAMLマニフェストに記録する。

設計仕様: docs/benchmark_runner.md
"""

from __future__ import annotations

import datetime
import hashlib
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import BatchProcess

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class EnvironmentInfo:
    """実行環境のスナップショット."""

    git_commit: str
    git_branch: str
    git_dirty: bool
    python_version: str
    numpy_version: str
    timestamp: str


def capture_environment() -> EnvironmentInfo:
    """現在の実行環境を取得."""
    git_commit = _git_cmd("rev-parse", "HEAD")
    git_branch = _git_cmd("rev-parse", "--abbrev-ref", "HEAD")
    git_dirty = bool(_git_cmd("status", "--porcelain").strip())
    return EnvironmentInfo(
        git_commit=git_commit.strip(),
        git_branch=git_branch.strip(),
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    )


def _git_cmd(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def serialize_config(obj: Any) -> Any:
    """frozen dataclass を再帰的に dict にシリアライズ."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        md5 = hashlib.md5(obj.tobytes(), usedforsecurity=False).hexdigest()[:12]
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype), "md5": md5}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = serialize_config(val)
        return result
    if isinstance(obj, (list, tuple)):
        if len(obj) > 100:
            return {"__sequence__": True, "length": len(obj), "type": type(obj).__name__}
        return [serialize_config(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_config(v) for k, v in obj.items()}
    if callable(obj):
        mod = getattr(obj, "__module__", "?")
        qn = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        return f"{mod}.{qn}"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"__sparse__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    r = repr(obj)
    if len(r) > 200:
        r = r[:200] + "..."
    return {"__repr__": r, "type": type(obj).__name__}


@dataclass(frozen=True)
class RunManifest:
    """1回のベンチマーク実行の全記録."""

    process_name: str
    process_version: str
    environment: EnvironmentInfo
    config_params: dict[str, Any]
    results_summary: dict[str, Any]
    elapsed_seconds: float
    status_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        env = self.environment
        return {
            "process": {"name": self.process_name, "version": self.process_version},
            "environment": {
                "git_commit": env.git_commit,
                "git_branch": env.git_branch,
                "git_dirty": env.git_dirty,
                "python_version": env.python_version,
                "numpy_version": env.numpy_version,
                "timestamp": env.timestamp,
            },
            "config": self.config_params,
            "results": self.results_summary,
            "elapsed_seconds": self.elapsed_seconds,
            "status_file": self.status_file,
        }

    def to_yaml(self) -> str:
        return _dict_to_yaml(self.to_dict())


def _dict_to_yaml(d: dict, indent: int = 0) -> str:
    """軽量YAML出力（PyYAML非依存）."""
    lines: list[str] = []
    prefix = "  " * indent
    for k, v in d.items():
        if v is None:
            lines.append(f"{prefix}{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{prefix}{k}: {v}")
        elif isinstance(v, str):
            if "\n" in v or ":" in v or "#" in v:
                lines.append(f'{prefix}{k}: "{v}"')
            else:
                lines.append(f"{prefix}{k}: {v}")
        elif isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_dict_to_yaml(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}:")
            for item in v:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.append(_dict_to_yaml(item, indent + 2))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


@dataclass(frozen=True)
class BenchmarkRunInput(Generic[TIn]):
    """BenchmarkRunnerProcess の入力."""

    process: Any
    config: TIn
    result_extractors: dict[str, Callable] = field(default_factory=dict)
    status_file: str | None = None
    output_dir: str | None = None


@dataclass(frozen=True)
class BenchmarkRunResult(Generic[TOut]):
    """BenchmarkRunnerProcess の出力."""

    result: TOut
    manifest: RunManifest
    manifest_path: str | None = None


class BenchmarkRunnerProcess(BatchProcess["BenchmarkRunInput", "BenchmarkRunResult"]):
    """プロセス実行 + マニフェスト自動記録."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="BenchmarkRunnerProcess",
        module="batch",
        version="0.1.0",
        document_path="docs/benchmark_runner.md",
        stability="experimental",
        support_tier="ci-required",
    )

    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: BenchmarkRunInput) -> BenchmarkRunResult:
        proc = input_data.process
        config = input_data.config

        env = capture_environment()
        config_params = serialize_config(config)
        if not isinstance(config_params, dict):
            config_params = {"__config__": config_params}

        t0 = time.perf_counter()
        result = proc.process(config)
        elapsed = time.perf_counter() - t0

        results_summary: dict[str, Any] = {}
        for key, extractor in input_data.result_extractors.items():
            try:
                val = extractor(result)
                results_summary[key] = _sanitize_value(val)
            except Exception as e:
                results_summary[key] = f"ERROR: {e}"

        proc_meta = getattr(type(proc), "meta", None)
        manifest = RunManifest(
            process_name=type(proc).__name__,
            process_version=proc_meta.version if proc_meta else "unknown",
            environment=env,
            config_params=config_params,
            results_summary=results_summary,
            elapsed_seconds=round(elapsed, 3),
            status_file=input_data.status_file,
        )

        manifest_path = self._save_manifest(manifest, input_data.output_dir)

        return BenchmarkRunResult(
            result=result,
            manifest=manifest,
            manifest_path=manifest_path,
        )

    def _save_manifest(self, manifest: RunManifest, output_dir: str | None) -> str | None:
        try:
            if output_dir is None:
                base = Path("docs/benchmarks")
            else:
                base = Path(output_dir)
            base.mkdir(parents=True, exist_ok=True)

            ts = manifest.environment.timestamp.replace(":", "").replace("-", "")[:15]
            name = manifest.process_name
            filename = f"{name}_{ts}.yaml"
            path = base / filename
            path.write_text(manifest.to_yaml(), encoding="utf-8")
            return str(path)
        except Exception:
            return None


def _sanitize_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        return {"shape": list(val.shape), "dtype": str(val.dtype)}
    return repr(val)
