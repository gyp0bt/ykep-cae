"""ProcessRegistry -- プロセスレジストリの一元管理.

AbstractProcess._registry (dict) を ProcessRegistry クラスに昇格させる。

責務:
1. プロセスの登録・検索・列挙
2. 依存グラフクエリ（uses / used_by の横断検索）
3. テスト時のレジストリ隔離（isolate()）
4. カテゴリ・stability によるフィルタリング

後方互換:
- AbstractProcess._registry は ProcessRegistry.default()._store への
  プロキシとして維持される（RegistryProxy クラス経由）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xkep_cae_fluid.core.base import AbstractProcess

logger = logging.getLogger(__name__)


class ProcessRegistry:
    """プロセスレジストリ.

    シングルトン: ProcessRegistry.default() でグローバルインスタンスにアクセス。
    テスト時は isolate() でスナップショットコピーを作成して隔離する。
    """

    _default_instance: ProcessRegistry | None = None

    def __init__(self) -> None:
        self._store: dict[str, type[AbstractProcess]] = {}

    # --- シングルトンアクセス ---

    @classmethod
    def default(cls) -> ProcessRegistry:
        """グローバルレジストリを返す（遅延初期化）."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def _set_default(cls, instance: ProcessRegistry) -> None:
        """テスト用: デフォルトインスタンスを差し替える."""
        cls._default_instance = instance

    # --- 登録 ---

    def register(self, cls: type[AbstractProcess]) -> None:
        """プロセスクラスを登録."""
        name = cls.__name__
        if name in self._store:
            existing = self._store[name]
            if existing is not cls:
                logger.warning(
                    f"ProcessRegistry: {name} が再登録されました "
                    f"(旧: {existing.__module__}, 新: {cls.__module__})"
                )
        self._store[name] = cls

    # --- 検索・列挙 ---

    def get(self, name: str) -> type[AbstractProcess] | None:
        """名前でプロセスを取得."""
        return self._store.get(name)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def items(self) -> Any:
        """dict.items() 互換."""
        return self._store.items()

    def keys(self) -> Any:
        """dict.keys() 互換."""
        return self._store.keys()

    def values(self) -> Any:
        """dict.values() 互換."""
        return self._store.values()

    # --- フィルタリング ---

    def filter_by_category(self, category_name: str) -> list[tuple[str, type[AbstractProcess]]]:
        """カテゴリ名（MRO内のクラス名）でフィルタリング."""
        return [
            (name, cls)
            for name, cls in sorted(self._store.items())
            if any(base.__name__ == category_name for base in cls.__mro__)
        ]

    def filter_by_stability(self, stability: str) -> list[tuple[str, type[AbstractProcess]]]:
        """stability フィールドでフィルタリング."""
        return [
            (name, cls)
            for name, cls in sorted(self._store.items())
            if hasattr(cls, "meta") and cls.meta.stability == stability
        ]

    def non_deprecated(self) -> list[tuple[str, type[AbstractProcess]]]:
        """deprecated でないプロセスを返す."""
        return [
            (name, cls)
            for name, cls in sorted(self._store.items())
            if not (hasattr(cls, "meta") and cls.meta.deprecated)
        ]

    def exclude_test_fixtures(self) -> list[tuple[str, type[AbstractProcess]]]:
        """テスト用フィクスチャを除外したプロセスを返す."""
        return [
            (name, cls) for name, cls in sorted(self._store.items()) if not _is_test_fixture(cls)
        ]

    def concrete_processes(self) -> list[tuple[str, type[AbstractProcess]]]:
        """テスト用フィクスチャを除外した全具象プロセス."""
        return self.exclude_test_fixtures()

    # --- 依存グラフクエリ ---

    def dependants_of(self, process_name: str) -> list[str]:
        """指定プロセスに依存するプロセスのリスト（used_by 逆引き）."""
        cls = self._store.get(process_name)
        if cls is None:
            return []
        return [dep.__name__ for dep in getattr(cls, "_used_by", [])]

    def dependencies_of(self, process_name: str) -> list[str]:
        """指定プロセスが依存するプロセスのリスト（uses 正引き）."""
        cls = self._store.get(process_name)
        if cls is None:
            return []
        return [dep.__name__ for dep in cls.uses]

    # --- テスト用隔離 ---

    def isolate(self) -> ProcessRegistry:
        """現在のレジストリのスナップショットコピーを返す."""
        isolated = ProcessRegistry()
        isolated._store = dict(self._store)
        return isolated

    # --- dict 互換 ---

    def __getitem__(self, key: str) -> type[AbstractProcess]:
        return self._store[key]

    def __setitem__(self, key: str, value: type[AbstractProcess]) -> None:
        self._store[key] = value

    def __repr__(self) -> str:
        return f"ProcessRegistry({len(self._store)} processes)"


class RegistryProxy(dict):
    """AbstractProcess._registry の後方互換プロキシ.

    AbstractProcess._registry への直接アクセスを ProcessRegistry.default() に転送する。
    """

    def __init__(self, registry_fn: Any) -> None:
        super().__init__()
        self._registry_fn = registry_fn

    @property
    def _store(self) -> dict:
        return self._registry_fn()._store

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def items(self) -> Any:
        return self._store.items()

    def keys(self) -> Any:
        return self._store.keys()

    def values(self) -> Any:
        return self._store.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def __repr__(self) -> str:
        return f"RegistryProxy({len(self._store)} processes)"


def _is_test_fixture(cls: type) -> bool:
    """テスト用フィクスチャ（tests/ 配下で定義されたプロセス）かどうか."""
    module = getattr(cls, "__module__", "")
    return ".tests." in module or module.startswith("tests.")
