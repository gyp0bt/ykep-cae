"""nsbp: nsb（2D Brinkman-NS）の離散化を PETSc（petsc4py）で解く高速化実験パッケージ.

離散化は nsb と同一（`nsb.assembly.BrinkmanDiscretization` の係数と `nsb.fastres` 相当のカーネル）、
違うのはソルバー: DMDA の MPI 分割、FD カラーリングの厳密ヤコビアン、FGMRES + PCFIELDSPLIT（Schur、hypre）。

`nsbp.solver` は import 時に PETSc（MPI_Init）を初期化するので、ここでは読み込まない。
petsc4py が要る側は `from nsbp.solver import NSBPSolver, NSBPSettings, solve_steady` を明示する
（`nsbp.HAVE_PETSC` で有無を判定できる）。
"""

from __future__ import annotations

from importlib.util import find_spec

from nsbp.problem import Patch, PatchCoefficients, make_discretization

HAVE_PETSC = find_spec("petsc4py") is not None

__all__ = ["HAVE_PETSC", "Patch", "PatchCoefficients", "make_discretization"]
