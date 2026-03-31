"""プロセス-テスト 1:1 紐付けツール.

binds_to デコレータにより、テストクラスとプロセスを1:1で対応付ける。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xkep_cae_fluid.core.base import AbstractProcess


def binds_to(process_class: type[AbstractProcess]):
    """テストクラスをプロセスに1:1紐付けするデコレータ.

    Usage::

        @binds_to(MeshGenerationProcess)
        class TestMeshGenerationProcess:
            ...
    """

    def decorator(test_cls: type) -> type:
        if process_class._test_class is not None:
            raise ValueError(
                f"{process_class.__name__} には既に "
                f"{process_class._test_class} が紐付けられています。"
                "1:1 対応を維持してください。"
            )
        path = f"{test_cls.__module__}::{test_cls.__qualname__}"
        process_class._test_class = path
        return test_cls

    return decorator
