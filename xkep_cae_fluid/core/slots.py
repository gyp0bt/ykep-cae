"""StrategySlot -- Strategy slot の型付きディスクリプタ.

クラス変数として宣言し、インスタンスで具象 Strategy を設定する。
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")

_SENTINEL = object()


class StrategySlot(Generic[T]):
    """Strategy slot の型付きディスクリプタ.

    Usage:
        class MySolver(SolverProcess[...]):
            convection = StrategySlot(ConvectionSchemeStrategy)
            turbulence = StrategySlot(TurbulenceModelStrategy, required=False)
    """

    def __init__(self, protocol: type[T], *, required: bool = True) -> None:
        self.protocol = protocol
        self.required = required
        self._attr_name: str = ""
        self._public_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_slot_{name}"
        self._public_name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        val = getattr(obj, self._attr_name, _SENTINEL)
        if val is _SENTINEL:
            if self.required:
                raise AttributeError(
                    f"{type(obj).__name__}.{self._public_name} は未設定 (required StrategySlot)"
                )
            return None
        return val

    def __set__(self, obj: Any, value: T) -> None:
        if value is None:
            if self.required:
                raise TypeError(
                    f"{type(obj).__name__}.{self._public_name}: "
                    f"required StrategySlot に None は設定不可"
                )
            setattr(obj, self._attr_name, None)
            return
        if not isinstance(value, self.protocol):
            raise TypeError(
                f"{type(obj).__name__}.{self._public_name}: "
                f"{type(value).__name__} は {self.protocol.__name__} を満たしていない"
            )
        setattr(obj, self._attr_name, value)


def collect_strategy_slots(cls: type) -> dict[str, StrategySlot]:
    """クラスの全 StrategySlot を name -> StrategySlot で返す."""
    result = {}
    for klass in reversed(cls.__mro__):
        for name, attr in vars(klass).items():
            if isinstance(attr, StrategySlot):
                result[name] = attr
    return result


def collect_strategy_types(instance: Any) -> list[type]:
    """インスタンスの全 StrategySlot に設定された具象クラスの型リストを返す."""
    slots = collect_strategy_slots(type(instance))
    types = []
    for _name, slot in slots.items():
        val = getattr(instance, slot._attr_name, _SENTINEL)
        if val is not _SENTINEL and val is not None:
            types.append(type(val))
    return types
