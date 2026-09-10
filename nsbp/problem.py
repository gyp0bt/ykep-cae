"""問題定義: nsb の入力・係数配列を DMDA の局所パッチへ切り出す.

物理・離散化は `nsb.assembly.BrinkmanDiscretization` のものをそのまま使う（各ランクが全体配列を冗長に組み、
自分のゴースト付きパッチだけ切り出す。配列は O(N) で 1152×768 でも数十 MB なので分散させる価値がない）。
ソルバー側（PETSc）だけが nsb と違う、という切り分けを保つための層。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nsb.assembly import BrinkmanDiscretization
from nsb.core import NSBInput


@dataclass(frozen=True)
class Patch:
    """1 ランクのゴースト付きパッチ [gxs, gxe) × [gys, gye) と所有範囲 [xs, xe) × [ys, ye)."""

    nx: int
    ny: int
    xs: int
    xe: int
    ys: int
    ye: int
    gxs: int
    gxe: int
    gys: int
    gye: int

    @property
    def gnx(self) -> int:
        return self.gxe - self.gxs

    @property
    def gny(self) -> int:
        return self.gye - self.gys

    @property
    def own(self) -> tuple[slice, slice]:
        """パッチ配列上の所有セルのスライス."""
        return slice(self.xs - self.gxs, self.xe - self.gxs), slice(
            self.ys - self.gys, self.ye - self.gys
        )

    @property
    def sides_physical(self) -> tuple[bool, bool, bool, bool]:
        """(W, E, S, N) が物理境界か."""
        return self.gxs == 0, self.gxe == self.nx, self.gys == 0, self.gye == self.ny


class PatchCoefficients:
    """残差カーネル `nsbp.kernels.residual_patch` に渡す、パッチ上の係数配列一式."""

    def __init__(self, disc: BrinkmanDiscretization, patch: Patch) -> None:
        self.disc = disc
        self.patch = patch
        gx = slice(patch.gxs, patch.gxe)
        gy = slice(patch.gys, patch.gye)
        W, E, S, N = disc.sides["W"], disc.sides["E"], disc.sides["S"], disc.sides["N"]
        w_phys, e_phys, s_phys, n_phys = patch.sides_physical

        def c(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(a, dtype=np.float64)

        def cb(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(a, dtype=np.bool_)

        # W/E は j 方向（gy）、S/N は i 方向（gx）の配列
        self.args_bc = (
            bool(w_phys),
            bool(e_phys),
            bool(s_phys),
            bool(n_phys),
            cb(W.is_outlet[gy]),
            cb(E.is_outlet[gy]),
            cb(S.is_outlet[gx]),
            cb(N.is_outlet[gx]),
            c(W.p[gy]),
            c(E.p[gy]),
            c(S.p[gx]),
            c(N.p[gx]),
            c(disc.u_w[gy]),
            c(disc.v_w[gy]),
            c(disc.u_e[gy]),
            c(disc.v_e[gy]),
            c(disc.u_s[gx]),
            c(disc.v_s[gx]),
            c(disc.u_n[gx]),
            c(disc.v_n[gx]),
            c(disc.diff_diag[gx, gy]),
            c(disc._drag_vol[gx, gy]),
            c(disc.q_src[gx, gy]),
            c(disc.q_sink[gx, gy]),
            c(disc.c_sink[gx, gy]),
            c(disc.cp_sink[gx, gy]),
        )
        # [壁セル] パッチ上の面マスクと流体マスク（ゴースト込み）
        self.wall_x = np.ascontiguousarray(
            disc.wall_x[patch.gxs : patch.gxe + 1, gy], dtype=np.int8
        )
        self.wall_y = np.ascontiguousarray(
            disc.wall_y[gx, patch.gys : patch.gye + 1], dtype=np.int8
        )
        self.active = np.ascontiguousarray(disc.active[gx, gy], dtype=np.float64)
        self.has_solid = bool(disc.has_solid)
        self.rho = float(disc.rho)
        self.mu = float(disc.mu)
        self.dx = float(disc.dx)
        self.dy = float(disc.dy)
        self.vol = float(disc.vol)


def make_discretization(inp: NSBInput) -> BrinkmanDiscretization:
    return BrinkmanDiscretization(inp.to_flow_input())
