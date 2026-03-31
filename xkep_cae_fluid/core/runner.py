"""ProcessRunner -- プロセス実行管理.

プロセス実行時に依存チェック・プロファイリング・ログ出力を一元管理する。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from xkep_cae_fluid.core.base import AbstractProcess
from xkep_cae_fluid.core.registry import ProcessRegistry

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionContext:
    """プロセス実行のコンテキスト."""

    dry_run: bool = False
    profile: bool = True
    log_file: Path | None = None
    validate_deps: bool = __debug__
    checksum_inputs: bool = __debug__


@dataclass(frozen=True)
class ExecutionRecord:
    """1回の実行記録."""

    process_name: str
    elapsed_seconds: float
    checksum_before: str | None
    checksum_after: str | None
    checksum_ok: bool
    dry_run: bool


class ProcessRunner:
    """プロセスの実行管理.

    責務:
    1. 実行前: uses 依存チェック
    2. 実行前: 入力 checksum 計算
    3. 実行: process() 呼び出し (dry_run 時はスキップ)
    4. 実行後: checksum 検証 + プロファイル記録 + ログ出力
    """

    def __init__(self, context: ExecutionContext | None = None) -> None:
        self.context = context or ExecutionContext()
        self._execution_log: list[ExecutionRecord] = []

    def run(self, process: AbstractProcess, input_data: Any) -> Any:
        """プロセスを実行する."""
        proc_name = type(process).__name__

        if self.context.validate_deps:
            self._validate_deps(process)

        checksum_before: str | None = None
        if self.context.checksum_inputs:
            checksum_before = AbstractProcess._compute_checksum(input_data)

        if self.context.dry_run:
            self._record(proc_name, 0.0, checksum_before, None, True, dry_run=True)
            self._log(f"[dry_run] {proc_name} -- スキップ")
            return None

        t0 = time.perf_counter()
        result = process.process(input_data)
        elapsed = time.perf_counter() - t0

        checksum_after: str | None = None
        checksum_ok = True
        if self.context.checksum_inputs and checksum_before is not None:
            checksum_after = AbstractProcess._compute_checksum(input_data)
            checksum_ok = checksum_before == checksum_after
            if not checksum_ok:
                msg = f"{proc_name}.process() が入力データを変更しました"
                logger.warning(msg)
                raise AssertionError(msg)

        self._record(proc_name, elapsed, checksum_before, checksum_after, checksum_ok)
        self._log(f"{proc_name}: {elapsed:.4f}s")

        return result

    def run_pipeline(
        self,
        steps: list[tuple[AbstractProcess, Any]],
    ) -> list[Any]:
        """複数プロセスを順次実行."""
        results = []
        for process, input_data in steps:
            result = self.run(process, input_data)
            results.append(result)
        return results

    def get_report(self) -> str:
        """実行ログのサマリーレポート."""
        lines = ["ProcessRunner Report", "=" * 40]
        total = 0.0
        for rec in self._execution_log:
            status = "dry_run" if rec.dry_run else f"{rec.elapsed_seconds:.4f}s"
            cksum = "OK" if rec.checksum_ok else "NG"
            lines.append(f"  {rec.process_name}: {status} (checksum: {cksum})")
            total += rec.elapsed_seconds
        lines.append(f"  --- total: {total:.4f}s ({len(self._execution_log)} steps)")
        return "\n".join(lines)

    def _validate_deps(self, process: AbstractProcess) -> None:
        effective = process.effective_uses()
        registry = ProcessRegistry.default()
        for dep in effective:
            dep_name = dep.__name__
            if dep_name not in registry:
                logger.warning(f"{type(process).__name__} の依存 {dep_name} がレジストリに未登録")

    def _record(
        self,
        proc_name: str,
        elapsed: float,
        checksum_before: str | None,
        checksum_after: str | None,
        checksum_ok: bool,
        *,
        dry_run: bool = False,
    ) -> None:
        self._execution_log.append(
            ExecutionRecord(
                process_name=proc_name,
                elapsed_seconds=elapsed,
                checksum_before=checksum_before,
                checksum_after=checksum_after,
                checksum_ok=checksum_ok,
                dry_run=dry_run,
            )
        )

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.context.log_file is not None:
            with open(self.context.log_file, "a") as f:
                f.write(message + "\n")
