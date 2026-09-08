"""入れ子反復（nested iteration）: 粗格子の収束解を補間して細格子の初期場にする駆動部.

[粗→細の補間] セル中心データの 2× 細分。区分定数注入（4 セルに同じ値）か双一次補間
  （細セル中心は粗セル中心から ±1/4 セルずれるので、重み 9/16・3/16・3/16・1/16、境界は最近傍で外挿）。
  双一次のほうが初期残差は小さい（288×192 で |R|/|R_ref| 0.50 → 0.08）が、SER の古典形出発
  CFL = cfl_init·|R_ref|/|R_init| がその分大きく出るので、SER の制御則と組で決める（status-41）。
[段の深さ] 各段は newton_tol を coarse_tol まで緩めて解き、最終段だけ settings.newton_tol で解く。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from nsb.core import NSBInput, NSBResult, NSBSettings
from nsb.solver import LogFn, solve_steady


def prolong_inject(a: np.ndarray) -> np.ndarray:
    """(nx, ny) → (2nx, 2ny) の区分定数注入."""
    return np.repeat(np.repeat(np.asarray(a), 2, axis=0), 2, axis=1)


def prolong_bilinear(a: np.ndarray) -> np.ndarray:
    """(nx, ny) → (2nx, 2ny) のセル中心双一次補間（境界は最近傍で外挿）."""
    a = np.asarray(a, dtype=np.float64)
    nx, ny = a.shape
    out = np.empty((2 * nx, 2 * ny))
    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    for da, sa in ((0, -1), (1, 1)):
        ia = np.clip(i + sa, 0, nx - 1)
        for db, sb in ((0, -1), (1, 1)):
            jb = np.clip(j + sb, 0, ny - 1)
            out[2 * i + da, 2 * j + db] = (
                9.0 * a[i, j] + 3.0 * a[ia, j] + 3.0 * a[i, jb] + a[ia, jb]
            ) / 16.0
    return out


PROLONGATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "inject": prolong_inject,
    "bilinear": prolong_bilinear,
}


@dataclass(frozen=True)
class NestedLevel:
    """1 段の記録: 格子と Newton 反復・GMRES 反復・所要時間."""

    nx: int
    ny: int
    newton_tol: float
    result: NSBResult
    elapsed: float


@dataclass(frozen=True)
class NestedResult:
    """入れ子反復の結果（最終段の NSBResult と段ごとの記録）."""

    levels: tuple[NestedLevel, ...]

    @property
    def result(self) -> NSBResult:
        return self.levels[-1].result

    @property
    def elapsed_total(self) -> float:
        return float(sum(lv.elapsed for lv in self.levels))

    @property
    def n_iter_total(self) -> int:
        return int(sum(lv.result.n_iter for lv in self.levels))


def prolong_fields(
    res: NSBResult, nx: int, ny: int, prolongation: str = "bilinear"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """粗格子の解 (res.u/v/p) を (nx, ny) まで 2× ずつ補間する（整数倍の 2 冪のみ）."""
    P = PROLONGATIONS[prolongation]
    u, v, p = res.u, res.v, res.p
    while u.shape != (nx, ny):
        if nx % u.shape[0] or ny % u.shape[1] or u.shape[0] > nx:
            raise ValueError(f"{u.shape} から ({nx}, {ny}) へは 2× の繰返しで届かない")
        u, v, p = P(u), P(v), P(p)
    return u, v, p


def solve_nested(
    make_input: Callable[[int], NSBInput],
    refines: Sequence[int],
    coarse_tol: float = 1e-4,
    prolongation: str = "bilinear",
    settings: NSBSettings | None = None,
    log: LogFn | None = print,
) -> NestedResult:
    """refines（粗い順、最後が目的の細格子）を順に解き、各段の解を次段の初期場にする.

    make_input(refine) が NSBInput を返す（settings は上書きするので make_input 側で与えなくてよい）。
    coarse_tol は最終段以外の newton_tol。
    """
    s = settings or NSBSettings()
    levels: list[NestedLevel] = []
    prev: NSBResult | None = None
    for k, r in enumerate(refines):
        tol = s.newton_tol if k == len(refines) - 1 else coarse_tol
        inp = replace(make_input(r), settings=replace(s, newton_tol=tol))
        if prev is not None:
            u0, v0, p0 = prolong_fields(prev, inp.nx, inp.ny, prolongation)
            inp = replace(inp, u0=u0, v0=v0, p0=p0)
        if log is not None:
            log(f"[nested] level {k + 1}/{len(refines)}: {inp.nx}x{inp.ny} tol={tol:.1e}")
        t0 = time.perf_counter()
        res = solve_steady(inp, log)
        levels.append(NestedLevel(inp.nx, inp.ny, tol, res, time.perf_counter() - t0))
        if not res.converged:
            break
        prev = res
    return NestedResult(tuple(levels))
