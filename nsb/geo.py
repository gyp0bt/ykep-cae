"""uturn / flat の厚さ場と境界条件プリセット、およびそれらを実行する run_* 関数."""

from __future__ import annotations

import numpy as np

from nsb.core import BC, FaceType, NSBInput, NSBResult, NSBSettings
from nsb.solver import LogFn, solve_steady

LX, LY = 0.7, 0.4
INLET_Y = (0.25, 0.35)  # 左壁、上から 0.05 空けて高さ 0.1
OUTLET_Y = (0.05, 0.15)  # 左壁、下から 0.05 空けて高さ 0.1
H_CHANNEL, H_BLOCKED = 1.0e-3, 1.0e-5
CHANNEL_WIDTH = 0.1
BASE_NX, BASE_NY = 72, 48


def make_flat_h(nx: int, ny: int, h: float = H_CHANNEL) -> np.ndarray:
    """全域一様な厚さ場."""
    return np.full((nx, ny), h)


def make_uturn_h(
    nx: int,
    ny: int,
    h_channel: float = H_CHANNEL,
    h_blocked: float = H_BLOCKED,
    width: float = CHANNEL_WIDTH,
) -> np.ndarray:
    """U ターン経路（往路: inlet 高さ、折返し: 右端 width、復路: outlet 高さ）のみ h_channel."""
    xc = (np.arange(nx) + 0.5) * LX / nx
    yc = (np.arange(ny) + 0.5) * LY / ny
    x, y = np.meshgrid(xc, yc, indexing="ij")
    leg_in = (y > INLET_Y[0]) & (y < INLET_Y[1])
    leg_out = (y > OUTLET_Y[0]) & (y < OUTLET_Y[1])
    turn = (x > LX - width) & (y > OUTLET_Y[0]) & (y < INLET_Y[1])
    mask = leg_in | leg_out | turn
    return np.where(mask, h_channel, h_blocked)


def uturn_bc_preset(ny: int, u_in: float) -> BC:
    """左壁: 上部 inlet、下部 outlet、それ以外 WALL."""
    yc = (np.arange(ny) + 0.5) * LY / ny
    west = np.array([FaceType.WALL] * ny, dtype=object)
    west[(yc > INLET_Y[0]) & (yc < INLET_Y[1])] = FaceType.VELOCITY_INLET
    west[(yc > OUTLET_Y[0]) & (yc < OUTLET_Y[1])] = FaceType.PRESSURE_OUTLET
    return BC(west=west, u_inlet=u_in)


def make_case(
    model: str,
    refine: int,
    u_in: float,
    settings: NSBSettings | None = None,
    init: NSBResult | None = None,
) -> NSBInput:
    """model="uturn"/"flat"、refine=1,2,4 で NSBInput を組む."""
    nx, ny = BASE_NX * refine, BASE_NY * refine
    h = make_uturn_h(nx, ny) if model == "uturn" else make_flat_h(nx, ny)
    kw = {} if init is None else {"u0": init.u, "v0": init.v, "p0": init.p}
    return NSBInput(
        nx=nx,
        ny=ny,
        lx=LX,
        ly=LY,
        h=h,
        bc=uturn_bc_preset(ny, u_in),
        settings=settings or NSBSettings(),
        **kw,
    )


def run_uturn(
    refine: int = 1,
    u_in: float = 0.1,
    settings: NSBSettings | None = None,
    log: LogFn | None = print,
) -> tuple[NSBInput, NSBResult]:
    inp = make_case("uturn", refine, u_in, settings)
    return inp, solve_steady(inp, log)


def run_flat(
    refine: int = 1,
    u_in: float = 0.1,
    settings: NSBSettings | None = None,
    log: LogFn | None = print,
) -> tuple[NSBInput, NSBResult]:
    inp = make_case("flat", refine, u_in, settings)
    return inp, solve_steady(inp, log)
