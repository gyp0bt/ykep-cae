"""AbstractProcess 基底クラスとメタクラス.

全プロセスの基底クラス。メタクラスにより process() を自動ラップし、
実行トレース・プロファイリングを透過的に実現する。

xkep-cae と共通の Process Architecture 基盤。
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from xkep_cae_fluid.core.registry import ProcessRegistry, RegistryProxy

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class ProcessMeta:
    """プロセスのメタ情報.

    document_path: ソースファイルからの相対パスで設計文書を指定。
    """

    name: str
    module: str  # "pre", "solve", "post", "verify", "batch" 等
    version: str = "0.1.0"
    deprecated: bool = False
    deprecated_by: str | None = None
    document_path: str = ""
    stability: str = "stable"  # experimental / stable / frozen / deprecated
    support_tier: str = "ci-required"  # ci-required / compat-only / dev-only


class ProcessMetaclass(type(ABC)):
    """AbstractProcess のメタクラス.

    process() メソッドを自動ラップし、以下を実現する:
    - 実行トレース: どの process() が呼ばれたかを記録
    - プロファイリング: process() 単位の実行時間を自動計測
    """

    _call_stack: ClassVar[list[str]] = []
    _profile_data: ClassVar[dict[str, list[float]]] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if "process" in namespace and callable(namespace["process"]):
            original = namespace["process"]

            @functools.wraps(original)
            def traced_process(self, input_data):  # noqa: ANN001
                cls_name = type(self).__name__

                meta = getattr(type(self), "meta", None)
                if meta is not None and getattr(meta, "deprecated", False):
                    from xkep_cae_fluid.core.diagnostics import DeprecatedProcessError

                    successor = getattr(meta, "deprecated_by", None) or "不明"
                    raise DeprecatedProcessError(
                        f"{cls_name} は deprecated です。 後継プロセス: {successor}"
                    )

                from xkep_cae_fluid.core.diagnostics import ProcessExecutionLog

                log = ProcessExecutionLog.instance()
                ctx = None
                if log.enabled:
                    ctx = log.record_start(cls_name)

                ProcessMetaclass._call_stack.append(cls_name)
                t0 = time.perf_counter()
                warning_type = None
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = original(self, input_data)
                    for w in caught:
                        warnings.warn_explicit(
                            w.message,
                            w.category,
                            w.filename,
                            w.lineno,
                        )
                        wname = type(w.message).__name__
                        if "Deprecated" in wname:
                            warning_type = "deprecated"
                finally:
                    elapsed = time.perf_counter() - t0
                    ProcessMetaclass._call_stack.pop()
                    if cls_name not in ProcessMetaclass._profile_data:
                        ProcessMetaclass._profile_data[cls_name] = []
                    ProcessMetaclass._profile_data[cls_name].append(elapsed)

                    if ctx is not None and log.enabled:
                        log.record_end(ctx, warning_type=warning_type)

                return result

            cls.process = traced_process

        return cls

    @classmethod
    def get_trace(mcs) -> list[str]:
        """現在の実行スタック（デバッグ用）."""
        return list(mcs._call_stack)

    @classmethod
    def get_profile_report(mcs) -> str:
        """全プロセスのプロファイルレポート."""
        lines = ["Process Profile Report", "=" * 40]
        for name, times in sorted(mcs._profile_data.items()):
            n = len(times)
            total = sum(times)
            avg = total / n if n > 0 else 0
            lines.append(f"  {name}: {n} calls, total={total:.3f}s, avg={avg:.3f}s")
        return "\n".join(lines)

    @classmethod
    def reset_profile(mcs) -> None:
        """プロファイルデータをリセット."""
        mcs._profile_data.clear()
        mcs._call_stack.clear()


class AbstractProcess(ABC, Generic[TIn, TOut], metaclass=ProcessMetaclass):
    """全プロセスの基底クラス.

    契約:
    - uses に宣言したプロセスのみを process() 内で使用可能
    - Input/Output型はジェネリックパラメータで明示
    - __init_subclass__ でクラス定義時に制約違反を検出
    - メタクラスが process() を自動ラップし、実行トレース + プロファイリング
    """

    meta: ClassVar[ProcessMeta]
    uses: ClassVar[list[type[AbstractProcess]]] = []

    _registry: ClassVar[dict[str, type[AbstractProcess]]] = RegistryProxy(ProcessRegistry.default)  # type: ignore[assignment]
    _used_by: ClassVar[list[type[AbstractProcess]]] = []
    _test_class: ClassVar[str | None] = None
    _verify_scripts: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        abstract_methods = getattr(cls, "__abstractmethods__", frozenset())
        if abstract_methods or ABC in cls.__bases__:
            return

        if not hasattr(cls, "meta") or not isinstance(cls.meta, ProcessMeta):
            raise TypeError(f"{cls.__name__} は ProcessMeta を定義してください")

        _doc_path = cls.meta.document_path
        if not _doc_path:
            _doc_path = getattr(cls, "document_path", "")
        if not _doc_path:
            raise TypeError(f"{cls.__name__} は meta.document_path を定義してください")

        src_file = inspect.getfile(cls)
        src_dir = Path(src_file).parent
        doc_full = (src_dir / _doc_path).resolve()
        if not doc_full.is_file():
            raise FileNotFoundError(f"{cls.__name__}: ドキュメントが見つかりません: {doc_full}")

        for dep in cls.uses:
            if hasattr(dep, "meta") and dep.meta.deprecated:
                warnings.warn(
                    f"{cls.__name__} は deprecated な {dep.__name__} を使用。"
                    f" 後継: {dep.meta.deprecated_by}",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if getattr(cls, "_skip_registry", False):
            cls._used_by = []
            return

        ProcessRegistry.default().register(cls)
        cls._used_by = []

        for dep in cls.uses:
            if not hasattr(dep, "_used_by") or dep._used_by is AbstractProcess._used_by:
                dep._used_by = []
            dep._used_by.append(cls)

    @abstractmethod
    def process(self, input_data: TIn) -> TOut:
        """メイン処理. サブクラスで実装."""
        ...

    def effective_uses(self) -> list[type[AbstractProcess]]:
        """静的 uses + StrategySlot の動的依存を統合した実効依存リストを返す."""
        from xkep_cae_fluid.core.slots import collect_strategy_types

        static = list(self.__class__.uses)
        runtime: list[type] = collect_strategy_types(self)
        seen = {id(c) for c in static}
        for dep in runtime:
            if id(dep) not in seen:
                static.append(dep)
                seen.add(id(dep))
        return static

    @staticmethod
    def _compute_checksum(data: Any) -> str | None:
        """入力データの numpy 配列チェックサムを計算（C9: 不変性検証用）."""
        import numpy as np

        if not hasattr(data, "__dataclass_fields__"):
            return None
        h = hashlib.md5(usedforsecurity=False)
        for f in fields(data):
            val = getattr(data, f.name, None)
            if isinstance(val, np.ndarray):
                h.update(val.tobytes())
        return h.hexdigest()

    def execute(self, input_data: TIn) -> TOut:
        """process() の公開エントリポイント（C9: 入力不変性チェック付き）."""
        if __debug__:
            checksum_before = self._compute_checksum(input_data)
        result = self.process(input_data)
        if __debug__ and checksum_before is not None:
            checksum_after = self._compute_checksum(input_data)
            assert checksum_before == checksum_after, (
                f"{type(self).__name__}.process() が入力データの numpy 配列を変更しました。"
                "frozen dataclass の numpy 配列は in-place 変更可能ですが、"
                "process() は入力を不変に保つ契約です。"
            )
        return result

    @classmethod
    def get_dependency_tree(cls) -> dict:
        """再帰的に依存ツリーを返す."""
        return {
            "name": cls.__name__,
            "module": cls.meta.module if hasattr(cls, "meta") else "?",
            "uses": [dep.get_dependency_tree() for dep in cls.uses],
        }

    @classmethod
    def _resolve_document_path(cls) -> str:
        """document_path を解決."""
        doc_path = cls.meta.document_path
        if not doc_path:
            doc_path = getattr(cls, "document_path", "")
        return doc_path

    @classmethod
    def _resolve_document_fullpath(cls) -> Path | None:
        """ドキュメントの絶対パスを返す."""
        doc_path = cls._resolve_document_path()
        if not doc_path:
            return None
        src_file = inspect.getfile(cls)
        src_dir = Path(src_file).parent
        full = (src_dir / doc_path).resolve()
        return full if full.is_file() else None

    @classmethod
    def document_markdown(cls) -> str:
        """Markdownドキュメント自動生成."""
        lines = [
            f"## {cls.__name__}",
            f"- **モジュール**: {cls.meta.module}",
            f"- **バージョン**: {cls.meta.version}",
        ]
        doc_path = cls._resolve_document_path()
        if doc_path:
            lines.append(f"- **設計文書**: `{doc_path}`")
        if cls.meta.deprecated:
            lines.append(f"- **DEPRECATED** → {cls.meta.deprecated_by}")
        if cls.uses:
            lines.append(f"- **依存**: {', '.join(d.__name__ for d in cls.uses)}")
        if cls._used_by:
            lines.append(f"- **被依存**: {', '.join(d.__name__ for d in cls._used_by)}")
        if cls._test_class:
            lines.append(f"- **テスト**: `{cls._test_class}`")
        if cls._verify_scripts:
            lines.append("- **検証スクリプト**:")
            for vs in cls._verify_scripts:
                lines.append(f"  - `{vs}`")
        return "\n".join(lines)

    @classmethod
    def get_document(cls, *, include_deps: bool = True, depth: int = 0) -> str:
        """設計文書の内容 + ランタイム依存関係ドキュメントを返す."""
        indent = "#" * min(depth + 2, 6)
        sections: list[str] = []

        sections.append(f"{indent} {cls.__name__} (v{cls.meta.version})")
        if cls.meta.deprecated:
            sections.append(f"> **DEPRECATED** → {cls.meta.deprecated_by}")
        sections.append("")

        doc_full = cls._resolve_document_fullpath()
        if doc_full is not None:
            content = doc_full.read_text(encoding="utf-8").strip()
            sections.append(content)
            sections.append("")
        else:
            doc_path = cls._resolve_document_path()
            if doc_path:
                sections.append(f"*設計文書 `{doc_path}` が見つかりません*")
                sections.append("")

        if include_deps and cls.uses:
            sections.append(f"{indent}# 依存プロセス")
            sections.append("")
            for dep in cls.uses:
                dep_doc = dep.get_document(include_deps=True, depth=depth + 1)
                sections.append(dep_doc)

        return "\n".join(sections)
