"""プロセスカテゴリ分類.

AbstractProcess の中間抽象クラスを定義する。
具象プロセスはこれらのいずれかを継承する。
"""

from __future__ import annotations

from abc import ABC
from typing import TypeVar

from xkep_cae_fluid.core.base import AbstractProcess

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class PreProcess(AbstractProcess[TIn, TOut], ABC):
    """前処理: メッシュ生成、境界条件設定、初期条件設定等."""


class SolverProcess(AbstractProcess[TIn, TOut], ABC):
    """求解: 時間進行、反復収束、圧力-速度連成等."""


class PostProcess(AbstractProcess[TIn, TOut], ABC):
    """後処理: 結果抽出、出力、可視化."""


class VerifyProcess(AbstractProcess[TIn, TOut], ABC):
    """検証: 物理量チェック、解析解比較、収束検証."""


class BatchProcess(AbstractProcess[TIn, TOut], ABC):
    """バッチ: 複数プロセスの直列/並列実行."""
