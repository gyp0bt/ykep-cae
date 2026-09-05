"""Abaqus 風キーワード構文のトークナイザ（``*KEYWORD, PARAM=VALUE`` + データ行）.

処理の流れ:

1. ``*INCLUDE, INPUT=file`` を再帰的に展開（呼び出し元ファイルからの相対パス）
2. ``**`` で始まる行（コメント）と空行を除去
3. ``*PARAMETER`` ブロックを評価し、以降の行の ``<expr>`` を置換
4. 行末カンマによる継続行を連結
5. キーワード行を (キーワード名, パラメータ辞書) に、データ行をカンマ区切りトークンに分解

構文木は :class:`KeywordBlock` の列で表現し、意味付け（節点・要素・ステップ等）は
:mod:`xkep_cae_fluid.inp.builder` が担当する。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.inp.parameters import (
    ParameterError,
    ParameterValue,
    parse_assignment,
    substitute,
)

_WS_RE = re.compile(r"\s+")

# データ行をカンマ分割せず生のまま保持するキーワード
RAW_DATA_KEYWORDS: frozenset[str] = frozenset({"HEADING"})


class InpSyntaxError(ValueError):
    """キーワード構文エラー（ファイル名と行番号を含む）."""

    def __init__(self, message: str, source: str = "", line_no: int = 0) -> None:
        location = f"{source}:{line_no}: " if source else ""
        super().__init__(f"{location}{message}")
        self.source = source
        self.line_no = line_no


@dataclass(frozen=True)
class SourceLine:
    """前処理済みの 1 行（継続行連結・パラメータ置換済み）."""

    text: str
    source: str
    line_no: int

    @property
    def is_keyword(self) -> bool:
        return self.text.startswith("*")


@dataclass(frozen=True)
class KeywordBlock:
    """1 つのキーワード行とそれに続くデータ行.

    Parameters
    ----------
    keyword : str
        正規化したキーワード名（大文字、``*`` なし、連続空白は 1 つ。例: ``"SOLID SECTION"``）
    params : Mapping[str, str]
        ``KEY=VALUE`` パラメータ（キーは大文字）。フラグ形式（``STEADY STATE``）は値 ``""``
    data : tuple[tuple[str, ...], ...]
        データ行（カンマ分割・前後空白除去済み）
    raw_lines : tuple[str, ...]
        データ行の生テキスト（``*HEADING`` 等の自由文用）
    source : str
        由来ファイル
    line_no : int
        キーワード行の行番号（1 始まり）
    """

    keyword: str
    params: Mapping[str, str] = field(default_factory=dict)
    data: tuple[tuple[str, ...], ...] = ()
    raw_lines: tuple[str, ...] = ()
    source: str = ""
    line_no: int = 0

    def has(self, name: str) -> bool:
        """パラメータ（フラグ含む）の有無."""
        return name.upper() in self.params

    def get(self, name: str, default: str | None = None) -> str | None:
        """パラメータ値（大文字小文字を区別しない）."""
        return self.params.get(name.upper(), default)

    def require(self, name: str) -> str:
        """必須パラメータ値。無ければ :class:`InpSyntaxError`."""
        value = self.get(name)
        if value is None or value == "":
            raise InpSyntaxError(
                f"*{self.keyword} には {name.upper()}= が必要です", self.source, self.line_no
            )
        return value

    def location(self) -> str:
        return f"{self.source}:{self.line_no}"


@dataclass(frozen=True)
class InpParseInput:
    """:class:`InpKeywordParseProcess` の入力.

    Parameters
    ----------
    path : str
        .inp ファイルのパス。``text`` を与える場合は ``*INCLUDE`` の基準ディレクトリ兼ラベル
    text : str | None
        ファイルを読まずに直接テキストを解析する場合に指定
    parameters : Mapping[str, ParameterValue]
        外部から与える初期パラメータ（``*PARAMETER`` より優先度は低い。CLI の上書き用）
    """

    path: str
    text: str | None = None
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)


@dataclass(frozen=True)
class InpParseResult:
    """トークナイズ結果."""

    blocks: tuple[KeywordBlock, ...]
    parameters: Mapping[str, ParameterValue]
    source: str


# ---------------------------------------------------------------------------
# 前処理（インクルード展開・コメント除去・パラメータ置換・継続行）
# ---------------------------------------------------------------------------


def _normalize_keyword(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lstrip("*").strip()).upper()


def parse_keyword_line(text: str, source: str = "", line_no: int = 0) -> tuple[str, dict[str, str]]:
    """``*KEY, A=1, FLAG`` を (``"KEY"``, ``{"A": "1", "FLAG": ""}``) に分解する."""
    parts = [p.strip() for p in text.split(",")]
    keyword = _normalize_keyword(parts[0])
    if not keyword:
        raise InpSyntaxError("キーワード名が空です", source, line_no)
    params: dict[str, str] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            params[_WS_RE.sub(" ", key.strip()).upper()] = value.strip()
        else:
            params[_WS_RE.sub(" ", part).upper()] = ""
    return keyword, params


def split_data_line(text: str) -> tuple[str, ...]:
    """データ行をカンマで分割し前後空白を除去する（末尾の空要素は落とす）."""
    tokens = [t.strip() for t in text.split(",")]
    while tokens and tokens[-1] == "":
        tokens.pop()
    return tuple(tokens)


def _read_lines(path: Path, depth: int, seen: set[Path]) -> list[SourceLine]:
    """ファイルを読み ``*INCLUDE`` を展開した生の行列を返す（パラメータ置換前）."""
    if depth > 16:
        raise InpSyntaxError("*INCLUDE の入れ子が深すぎます（上限 16）", str(path))
    resolved = path.resolve()
    if resolved in seen:
        raise InpSyntaxError("*INCLUDE が循環しています", str(path))
    seen = seen | {resolved}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InpSyntaxError(f"ファイルを読めません: {exc}", str(path)) from exc
    return _expand_text(text, path, depth, seen)


def _expand_text(text: str, path: Path, depth: int, seen: set[Path]) -> list[SourceLine]:
    lines: list[SourceLine] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        if stripped.startswith("*") and _normalize_keyword(stripped.split(",")[0]) == "INCLUDE":
            _, params = parse_keyword_line(stripped, str(path), idx)
            target = params.get("INPUT", "")
            if not target:
                raise InpSyntaxError("*INCLUDE には INPUT= が必要です", str(path), idx)
            include_path = Path(target)
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            lines.extend(_read_lines(include_path, depth + 1, seen))
            continue
        lines.append(SourceLine(stripped, str(path), idx))
    return lines


def _apply_parameters(
    lines: list[SourceLine], initial: Mapping[str, ParameterValue]
) -> tuple[list[SourceLine], dict[str, ParameterValue]]:
    """``*PARAMETER`` ブロックを評価し、他の行の ``<expr>`` を置換する."""
    namespace: dict[str, ParameterValue] = dict(initial)
    out: list[SourceLine] = []
    in_parameter = False
    for line in lines:
        if line.is_keyword:
            keyword, _ = parse_keyword_line(line.text, line.source, line.line_no)
            in_parameter = keyword == "PARAMETER"
            if in_parameter:
                continue
        elif in_parameter:
            try:
                name, value = parse_assignment(substitute(line.text, namespace), namespace)
            except ParameterError as exc:
                raise InpSyntaxError(str(exc), line.source, line.line_no) from exc
            namespace[name] = value
            continue
        try:
            text = substitute(line.text, namespace)
        except ParameterError as exc:
            raise InpSyntaxError(str(exc), line.source, line.line_no) from exc
        out.append(SourceLine(text, line.source, line.line_no))
    return out, namespace


def _join_continuations(lines: list[SourceLine]) -> list[SourceLine]:
    """行末カンマの継続行を連結する（キーワード行・データ行とも）."""
    out: list[SourceLine] = []
    pending: SourceLine | None = None
    for line in lines:
        if pending is not None and not line.is_keyword:
            pending = SourceLine(pending.text + " " + line.text, pending.source, pending.line_no)
            if not pending.text.endswith(","):
                out.append(pending)
                pending = None
            continue
        if pending is not None:
            # 継続中にキーワードが来た: 末尾カンマは単なる区切りとみなし打ち切る
            out.append(pending)
            pending = None
        if line.text.endswith(","):
            pending = line
        else:
            out.append(line)
    if pending is not None:
        out.append(pending)
    return out


def _group_blocks(lines: list[SourceLine]) -> tuple[KeywordBlock, ...]:
    blocks: list[KeywordBlock] = []
    current_kw: tuple[str, dict[str, str], SourceLine] | None = None
    data: list[tuple[str, ...]] = []
    raw: list[str] = []

    def _flush() -> None:
        if current_kw is None:
            return
        keyword, params, head = current_kw
        blocks.append(
            KeywordBlock(
                keyword=keyword,
                params=params,
                data=tuple(data),
                raw_lines=tuple(raw),
                source=head.source,
                line_no=head.line_no,
            )
        )

    for line in lines:
        if line.is_keyword:
            _flush()
            keyword, params = parse_keyword_line(line.text, line.source, line.line_no)
            current_kw = (keyword, params, line)
            data = []
            raw = []
        else:
            if current_kw is None:
                raise InpSyntaxError(
                    "キーワード行の前にデータ行があります", line.source, line.line_no
                )
            raw.append(line.text)
            if current_kw[0] not in RAW_DATA_KEYWORDS:
                data.append(split_data_line(line.text))
    _flush()
    return tuple(blocks)


def parse_inp_text(
    text: str, path: str = "<string>", parameters: Mapping[str, ParameterValue] | None = None
) -> InpParseResult:
    """テキストを直接トークナイズする（``*INCLUDE`` は ``path`` のディレクトリ基準）."""
    base = Path(path)
    lines = _expand_text(text, base, 0, {base.resolve()} if base.exists() else set())
    lines, namespace = _apply_parameters(lines, parameters or {})
    lines = _join_continuations(lines)
    return InpParseResult(blocks=_group_blocks(lines), parameters=namespace, source=path)


def parse_inp_file(
    path: str | Path, parameters: Mapping[str, ParameterValue] | None = None
) -> InpParseResult:
    """ファイルをトークナイズする."""
    p = Path(path)
    lines = _read_lines(p, 0, set())
    lines, namespace = _apply_parameters(lines, parameters or {})
    lines = _join_continuations(lines)
    return InpParseResult(blocks=_group_blocks(lines), parameters=namespace, source=str(p))


class InpKeywordParseProcess(PreProcess["InpParseInput", "InpParseResult"]):
    """.inp テキストを :class:`KeywordBlock` 列に分解する PreProcess.

    ``*INCLUDE`` 展開・コメント除去・``*PARAMETER`` 評価/置換・継続行連結を行う。
    意味解釈はしない（未知のキーワードもそのまま通す）。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpKeywordParse",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpParseInput) -> InpParseResult:
        if input_data.text is not None:
            return parse_inp_text(input_data.text, input_data.path, input_data.parameters)
        return parse_inp_file(input_data.path, input_data.parameters)
