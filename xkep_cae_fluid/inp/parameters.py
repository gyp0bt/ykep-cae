"""``*PARAMETER`` 用の安全な式評価と ``<name>`` 置換.

Abaqus の ``*PARAMETER`` は Python 式で変数を定義し、以降の行で ``<name>`` を
値に置き換える。本モジュールは ``ast`` のホワイトリスト評価で同等の機能を提供する
（``eval`` は使わない）。``<...>`` の中身は名前だけでなく式も許容する。
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

ParameterValue = int | float | str | bool | tuple

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "round": round,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "ceil": math.ceil,
    "floor": math.floor,
    "degrees": math.degrees,
    "radians": math.radians,
}

SAFE_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_SUBST_RE = re.compile(r"<([^<>]+)>")


class ParameterError(ValueError):
    """``*PARAMETER`` の定義・置換エラー."""


def safe_eval(expression: str, namespace: Mapping[str, ParameterValue]) -> ParameterValue:
    """ホワイトリスト方式で Python 式を評価する.

    Parameters
    ----------
    expression : str
        評価する式（例: ``"length / nx"``, ``"2 * pi * r"``）
    namespace : Mapping[str, ParameterValue]
        参照可能な変数

    Returns
    -------
    ParameterValue
        評価結果

    Raises
    ------
    ParameterError
        構文エラー、未定義名、許可されない構文（属性参照、import、lambda 等）
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ParameterError(f"式の構文エラー: {expression!r} ({exc.msg})") from exc

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)):
                return node.value
            raise ParameterError(f"許可されない定数: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in namespace:
                return namespace[node.id]
            if node.id in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[node.id]
            raise ParameterError(f"未定義のパラメータ: {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if type(op) not in _CMP_OPS:
                    raise ParameterError("許可されない比較演算子")
                right = _eval(comparator)
                if not _CMP_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return _eval(node.body) if _eval(node.test) else _eval(node.orelse)
        if isinstance(node, ast.Tuple):
            return tuple(_eval(e) for e in node.elts)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                raise ParameterError("許可されない関数呼び出し")
            if node.keywords:
                raise ParameterError("キーワード引数は使用できません")
            return SAFE_FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
        raise ParameterError(f"許可されない構文: {type(node).__name__}")

    return _eval(tree)


def parse_assignment(
    line: str, namespace: Mapping[str, ParameterValue]
) -> tuple[str, ParameterValue]:
    """``name = expr`` 形式の 1 行を評価し (name, value) を返す."""
    if "=" not in line:
        raise ParameterError(f"*PARAMETER の行は 'name = expr' 形式が必要: {line!r}")
    name, expr = line.split("=", 1)
    name = name.strip()
    if not name.isidentifier():
        raise ParameterError(f"パラメータ名が不正: {name!r}")
    value = safe_eval(expr, namespace)
    return name, value


def format_value(value: ParameterValue) -> str:
    """置換時の文字列表現（int はそのまま、float は repr、tuple はカンマ区切り）."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return ", ".join(format_value(v) for v in value)
    return str(value)


def substitute(line: str, namespace: Mapping[str, ParameterValue]) -> str:
    """行中の ``<expr>`` を評価結果で置き換える."""
    if "<" not in line:
        return line

    def _repl(match: re.Match[str]) -> str:
        return format_value(safe_eval(match.group(1), namespace))

    return _SUBST_RE.sub(_repl, line)
