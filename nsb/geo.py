"""uturn / flat の厚さ場と境界条件プリセット、およびそれらを実行する run_* 関数."""

from __future__ import annotations

import numpy as np

from nsb.core import BC, NSBInput, NSBResult, NSBSettings
from nsb.data import MaskFn, west_span
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
    inlet_y: tuple[float, float] = INLET_Y,
    outlet_y: tuple[float, float] = OUTLET_Y,
) -> np.ndarray:
    """U ターン経路（往路: inlet 高さ、折返し: 右端 width、復路: outlet 高さ）のみ h_channel.

    inlet_y / outlet_y を変えると往路・復路の位置も追従する（左壁 inlet/outlet 前提）。
    """
    xc = (np.arange(nx) + 0.5) * LX / nx
    yc = (np.arange(ny) + 0.5) * LY / ny
    x, y = np.meshgrid(xc, yc, indexing="ij")
    leg_in = (y > inlet_y[0]) & (y < inlet_y[1])
    leg_out = (y > outlet_y[0]) & (y < outlet_y[1])
    lo, hi = min(inlet_y[0], outlet_y[0]), max(inlet_y[1], outlet_y[1])
    turn = (x > LX - width) & (y > lo) & (y < hi)
    mask = leg_in | leg_out | turn
    return np.where(mask, h_channel, h_blocked)


def uturn_bc_preset(
    ny: int,
    u_in: float | None = None,
    mass_flow: float | None = None,
    inlet_y: tuple[float, float] = INLET_Y,
    outlet_y: tuple[float, float] = OUTLET_Y,
) -> BC:
    """左壁: 上部 inlet（速度 u_in または質量流量 mass_flow [kg/s]）、下部 outlet、それ以外 WALL."""
    if (u_in is None) == (mass_flow is None):
        raise ValueError("u_in か mass_flow のどちらか一方を指定してください")
    inlet_mask: MaskFn = west_span(*inlet_y)
    inlet = (
        BC.velocity_inlet(inlet_mask, u_in)
        if u_in is not None
        else BC.mass_flow_inlet(inlet_mask, mass_flow)
    )
    return BC(patches=(inlet, BC.pressure_outlet(west_span(*outlet_y))))


def make_case(
    model: str,
    refine: int,
    u_in: float | None = None,
    settings: NSBSettings | None = None,
    init: NSBResult | None = None,
    mass_flow: float | None = None,
    inlet_y: tuple[float, float] = INLET_Y,
    outlet_y: tuple[float, float] = OUTLET_Y,
    bc: BC | None = None,
) -> NSBInput:
    """model="uturn"/"flat"、refine=1,2,4 で NSBInput を組む.

    inlet は u_in（速度）か mass_flow（質量流量 [kg/s]、厚さ込み）で指定し、
    inlet_y / outlet_y で左壁上の位置・サイズを変えられる（uturn では厚さ場も追従）。
    任意の壁に置きたい場合は bc に BC を直接渡す（その場合 inlet_y/outlet_y は uturn 厚さ場にのみ使う）。
    """
    nx, ny = BASE_NX * refine, BASE_NY * refine
    h = (
        make_uturn_h(nx, ny, inlet_y=inlet_y, outlet_y=outlet_y)
        if model == "uturn"
        else make_flat_h(nx, ny)
    )
    kw = {} if init is None else {"u0": init.u, "v0": init.v, "p0": init.p}
    if bc is None:
        bc = uturn_bc_preset(ny, u_in, mass_flow, inlet_y, outlet_y)
    return NSBInput(
        nx=nx, ny=ny, lx=LX, ly=LY, h=h, bc=bc, settings=settings or NSBSettings(), **kw
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
