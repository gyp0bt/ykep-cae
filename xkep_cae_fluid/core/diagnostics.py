"""Process 実行診断 -- 実行ログ・呼び出し元追跡・レポート生成.

全 AbstractProcess.process() 呼び出しを自動記録し、
どのプロセスがどこから呼ばれたかをレポートする。

設計仕様: docs/process_diagnostics.md
"""

from __future__ import annotations

import atexit
import inspect
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class ProcessExecutionEntry:
    """1回のプロセス実行記録."""

    process_class: str
    process_module: str
    caller_file: str
    caller_function: str
    caller_line: int
    parent_process: str | None
    elapsed_seconds: float
    timestamp: float
    warning_type: str | None = None


class ProcessExecutionLog:
    """プロセス実行ログのシングルトン."""

    _instance: ClassVar[ProcessExecutionLog | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._entries: list[ProcessExecutionEntry] = []
        self._call_stack: list[str] = []
        self._enabled: bool = True
        self._report_on_exit: bool = True
        self._report_path: Path | None = None

    @classmethod
    def instance(cls) -> ProcessExecutionLog:
        """シングルトンインスタンスを返す."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """テスト用: インスタンスをリセットする."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._entries.clear()
                cls._instance._call_stack.clear()

    @property
    def entries(self) -> list[ProcessExecutionEntry]:
        return list(self._entries)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def record_start(self, process_class_name: str) -> _CallContext:
        """プロセス実行開始を記録."""
        parent = self._call_stack[-1] if self._call_stack else None
        self._call_stack.append(process_class_name)
        caller_file, caller_function, caller_line = _find_caller()
        return _CallContext(
            process_class=process_class_name,
            parent_process=parent,
            caller_file=caller_file,
            caller_function=caller_function,
            caller_line=caller_line,
            t0=time.perf_counter(),
        )

    def record_end(
        self,
        ctx: _CallContext,
        *,
        warning_type: str | None = None,
    ) -> None:
        """プロセス実行完了を記録する."""
        elapsed = time.perf_counter() - ctx.t0
        if self._call_stack and self._call_stack[-1] == ctx.process_class:
            self._call_stack.pop()

        entry = ProcessExecutionEntry(
            process_class=ctx.process_class,
            process_module=ctx.process_class,
            caller_file=ctx.caller_file,
            caller_function=ctx.caller_function,
            caller_line=ctx.caller_line,
            parent_process=ctx.parent_process,
            elapsed_seconds=elapsed,
            timestamp=time.time(),
            warning_type=warning_type,
        )
        self._entries.append(entry)

    def generate_report(self) -> str:
        """Markdown 形式のプロセス使用レポートを生成する."""
        lines = [
            "# Process Usage Report",
            "",
            f"> 生成時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 総実行数: {len(self._entries)}",
            "",
        ]

        if not self._entries:
            lines.append("*実行記録なし*")
            return "\n".join(lines)

        from collections import Counter, defaultdict

        lines.append("## プロセス別サマリー")
        lines.append("")
        lines.append("| Process | 呼出回数 | 合計時間(s) | 平均時間(s) | 警告 |")
        lines.append("|---------|---------|------------|------------|------|")

        call_counts: Counter[str] = Counter()
        total_times: dict[str, float] = defaultdict(float)
        warning_counts: Counter[str] = Counter()

        for entry in self._entries:
            call_counts[entry.process_class] += 1
            total_times[entry.process_class] += entry.elapsed_seconds
            if entry.warning_type:
                warning_counts[entry.process_class] += 1

        for proc_name in sorted(call_counts.keys()):
            n = call_counts[proc_name]
            total = total_times[proc_name]
            avg = total / n
            warn = warning_counts.get(proc_name, 0)
            warn_str = str(warn) if warn > 0 else ""
            lines.append(f"| {proc_name} | {n} | {total:.3f} | {avg:.3f} | {warn_str} |")

        return "\n".join(lines)

    def write_report(self, path: Path | None = None) -> Path:
        """レポートをファイルに書き出す."""
        if path is None:
            path = _find_repo_root() / "docs" / "generated" / "process_usage_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_report()
        path.write_text(report, encoding="utf-8")
        return path


@dataclass
class _CallContext:
    """record_start() が返す呼び出しコンテキスト."""

    process_class: str
    parent_process: str | None
    caller_file: str
    caller_function: str
    caller_line: int
    t0: float


def _find_caller() -> tuple[str, str, int]:
    """inspect.stack() から AbstractProcess 外の最初の呼び出し元を返す."""
    skip_files = {"base.py", "diagnostics.py", "runner.py"}
    skip_modules = {
        "xkep_cae_fluid.core.base",
        "xkep_cae_fluid.core.diagnostics",
        "xkep_cae_fluid.core.runner",
    }

    try:
        stack = inspect.stack()
    except Exception:
        return ("<unknown>", "<unknown>", 0)

    for frame_info in stack[2:]:
        filename = frame_info.filename
        basename = os.path.basename(filename)

        if basename in skip_files:
            continue

        module = frame_info.frame.f_globals.get("__name__", "")
        if module in skip_modules:
            continue

        try:
            repo_root = _find_repo_root()
            rel_path = str(Path(filename).resolve().relative_to(repo_root))
        except (ValueError, RuntimeError):
            rel_path = filename

        return (rel_path, frame_info.function, frame_info.lineno)

    return ("<unknown>", "<unknown>", 0)


def _find_repo_root() -> Path:
    """リポジトリルートを検索する."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current


class DeprecatedProcessError(RuntimeError):
    """deprecated プロセスの実行時エラー."""

    pass


class NonDefaultStrategyWarning(UserWarning):
    """デフォルトではない Strategy 構成が使用された場合の警告."""

    pass


def _atexit_report() -> None:
    """セッション終了時にプロセス使用レポートを書き出す."""
    log = ProcessExecutionLog.instance()
    if not log.entries:
        return
    if not log._report_on_exit:
        return
    try:
        path = log.write_report()
        import sys

        print(
            f"[ProcessDiagnostics] レポート出力: {path}",
            file=sys.stderr,
        )
    except Exception:
        pass


atexit.register(_atexit_report)
