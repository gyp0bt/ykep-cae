"""ProcessTree（実行グラフ）.

プロセスの実行順序と依存関係をグラフとして表現し、バリデーションを行う。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xkep_cae_fluid.core.base import AbstractProcess


class NodeType(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    CONDITIONAL = auto()


@dataclass(frozen=True)
class ProcessNode:
    """実行グラフのノード."""

    process_class: type[AbstractProcess]
    process_instance: AbstractProcess | None = None
    children: tuple[ProcessNode, ...] = ()
    node_type: NodeType = NodeType.SEQUENTIAL
    condition: Any | None = None

    def to_mermaid(self, indent: int = 0) -> str:
        """Mermaid フローチャート形式で出力."""
        name = self.process_class.__name__
        prefix = "  " * indent
        lines = [f"{prefix}{name}"]
        for child in self.children:
            lines.append(f"{prefix}{name} --> {child.process_class.__name__}")
            lines.extend(child.to_mermaid(indent + 1).split("\n"))
        return "\n".join(lines)


@dataclass(frozen=True)
class ProcessTree:
    """プロセス実行グラフ全体."""

    root: ProcessNode
    name: str = ""

    def validate(self) -> list[str]:
        """依存関係の整合性チェック."""
        errors: list[str] = []
        visited: set[str] = set()
        self._validate_node(self.root, visited, errors)
        return errors

    def _validate_node(self, node: ProcessNode, visited: set[str], errors: list[str]) -> None:
        cls = node.process_class
        name = cls.__name__

        if name in visited:
            errors.append(f"循環依存検出: {name}")
            return
        visited.add(name)

        all_classes = self._collect_all_classes(self.root)

        if node.process_instance is not None:
            deps = node.process_instance.effective_uses()
        else:
            deps = list(cls.uses)

        static_uses = set(cls.uses)
        for dep in deps:
            if dep not in all_classes:
                source = "uses" if dep in static_uses else "StrategySlot"
                errors.append(
                    f"{name} は {dep.__name__} を {source} で参照しているが、ツリーに含まれていない"
                )

        for child in node.children:
            self._validate_node(child, visited.copy(), errors)

    def _collect_all_classes(self, node: ProcessNode, seen: set[int] | None = None) -> set[type]:
        if seen is None:
            seen = set()
        node_id = id(node)
        if node_id in seen:
            return set()
        seen.add(node_id)
        classes = {node.process_class}
        for child in node.children:
            classes |= self._collect_all_classes(child, seen)
        return classes

    def to_markdown(self) -> str:
        """実行フロー図をMarkdownで出力."""
        lines = [f"# {self.name}" if self.name else "# ProcessTree"]
        self._node_to_md(self.root, lines, 0)
        return "\n".join(lines)

    def _node_to_md(self, node: ProcessNode, lines: list[str], depth: int) -> None:
        prefix = "  " * depth + "- "
        meta_info = ""
        if hasattr(node.process_class, "meta"):
            meta_info = f" ({node.process_class.meta.module})"
        lines.append(f"{prefix}{node.process_class.__name__}{meta_info}")
        for child in node.children:
            self._node_to_md(child, lines, depth + 1)

    def to_mermaid(self) -> str:
        """Mermaid フローチャート形式で出力."""
        lines = ["graph TD"]
        self._node_to_mermaid(self.root, lines)
        return "\n".join(lines)

    def _node_to_mermaid(self, node: ProcessNode, lines: list[str]) -> None:
        name = node.process_class.__name__
        for child in node.children:
            child_name = child.process_class.__name__
            lines.append(f"  {name} --> {child_name}")
            self._node_to_mermaid(child, lines)
