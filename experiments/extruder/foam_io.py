"""OpenFOAM ascii フィールドの最小リーダと Docker ラッパ呼び出し."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

OF_WRAPPER = os.path.expanduser("~/work/1a/a02/tools/of")
OF_IMAGE = os.environ.get("OF_IMG", "opencfd/openfoam-run:2312")


def run_of(case: str, *args: str, log: str | None = None) -> None:
    """`~/work/1a/a02/tools/of` でケースディレクトリ内のコマンドを回す.

    失敗したら stdout/stderr の末尾を添えて RuntimeError を投げる
    （CalledProcessError を __cause__ に残す）。
    """
    env = dict(
        os.environ, OF_CPUS=os.environ.get("OF_CPUS", "1"), OF_MEM=os.environ.get("OF_MEM", "1200m")
    )
    cmd = [OF_WRAPPER, *args]
    try:
        res = subprocess.run(cmd, cwd=case, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stdout or "")[-2000:] + (e.stderr or "")[-2000:]
        msg = f"OpenFOAM コマンド失敗: {' '.join(args)} (case={case})\n{tail}"
        raise RuntimeError(msg) from e
    if log:
        Path(log).write_text(res.stdout + res.stderr, encoding="utf-8")


def latest_time(case: str) -> str:
    times = []
    for d in os.listdir(case):
        try:
            times.append((float(d), d))
        except ValueError:
            continue
    if not times:
        msg = f"時刻ディレクトリが無い: {case}"
        raise FileNotFoundError(msg)
    return max(times)[1]


_LIST_RE = re.compile(r"nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\n?\s*\(", re.S)


def _parse_list(text: str, start: int, kind: str, count: int) -> np.ndarray:
    """`(` の直後から count 個の要素を読む."""
    if kind == "scalar":
        m = re.compile(r"\(([^()]*)\)", re.S).match(text, start - 1)
        if m is None:
            msg = "scalar リストの閉じ括弧が見つからない"
            raise ValueError(msg)
        vals = np.fromstring(m.group(1), sep=" ")
    elif kind == "vector":
        # (x y z) を count 個
        body_end = text.index("\n)", start)
        body = text[start:body_end]
        vals = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ").reshape(-1, 3)
    else:
        msg = f"未対応の List 型: {kind}"
        raise ValueError(msg)
    if vals.shape[0] != count:
        msg = f"要素数不一致: 期待 {count}, 実際 {vals.shape[0]}"
        raise ValueError(msg)
    return vals


def read_internal_field(path: str) -> np.ndarray:
    """internalField を (N,) か (N,3) で返す."""
    text = Path(path).read_text(encoding="utf-8")
    i = text.index("internalField")
    seg = text[i : i + 200]
    if "uniform" in seg and "nonuniform" not in seg:
        m = re.search(r"uniform\s+(\([^)]*\)|[-+0-9.eE]+)", seg)
        if m is None:
            msg = f"uniform 値を解釈できない: {path}"
            raise ValueError(msg)
        return np.fromstring(m.group(1).strip("()"), sep=" ")
    m = _LIST_RE.search(text, i)
    if m is None:
        msg = f"internalField のリストが見つからない: {path}"
        raise ValueError(msg)
    return _parse_list(text, m.end(), m.group(1), int(m.group(2)))


def read_patch_field(path: str, patch: str) -> np.ndarray:
    """boundaryField の指定パッチの value を返す."""
    text = Path(path).read_text(encoding="utf-8")
    i = text.index("boundaryField")
    m = re.search(rf"\n\s*{re.escape(patch)}\s*\{{", text[i:])
    if m is None:
        msg = f"パッチ {patch} が無い: {path}"
        raise FileNotFoundError(msg)
    j = i + m.end()
    seg = text[j:]
    mv = re.search(r"value\s+uniform\s+(\([^)]*\)|[-+0-9.eE]+)\s*;", seg[:300])
    if mv is not None:
        return np.fromstring(mv.group(1).strip("()"), sep=" ")
    ml = _LIST_RE.search(seg)
    if ml is None:
        msg = f"パッチ {patch} に value が無い: {path}"
        raise ValueError(msg)
    return _parse_list(seg, ml.end(), ml.group(1), int(ml.group(2)))


def read_cell_centres(case: str, time: str) -> np.ndarray:
    """`postProcess -func writeCellCentres` が書く C を (N,3) で返す."""
    return read_internal_field(os.path.join(case, time, "C"))


def continuity_converged(log_path: str) -> tuple[int, bool]:
    """simpleFoam ログから最終反復回数と residualControl 到達の有無を返す."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    its = re.findall(r"^Time = (\d+)", text, re.M)
    n = int(its[-1]) if its else 0
    return n, "SIMPLE solution converged" in text
