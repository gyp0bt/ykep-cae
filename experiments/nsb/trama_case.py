"""messi trama のパターン JSON から nsb の蛇行流路ケースを組む.

パターン（nodes / edges[via, width]）の中心線を格子に落として、中心線から幅/2 以内を
h_channel、それ以外を h_blocked にする。パターンの左端 2 ノード（edge の from / to）を
領域内（inner cell）の inlet（interior_source, 質量流量指定）/ outlet
（interior_pressure_sink, 圧力基準）にする。

座標規約（仮定）: パターン単位を一様に scale [mm/unit] 倍し、領域 lx×ly [mm] の中央に置く。
「横 600, 縦 350」は領域サイズと解釈した（パターンの bbox 62×57 unit を非等方に引き伸ばすと
流路幅が方向で変わるので採らない）。

使用例::

    python experiments/nsb/trama_case.py ../tmp/pattern.json --mass 0.15 \
        2>&1 | tee experiments/nsb/logs/trama-m0.15-$(date +%s).log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nsb import (  # noqa: E402
    BC,
    NSBInput,
    NSBResult,
    NSBSettings,
    disk_mask,
    smooth_disk,
    solve_steady,
    west_span,
)
from nsb.utils import save_fields, summary  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_PATTERN = HERE.parents[2] / "tmp" / "pattern.json"


@dataclass(frozen=True)
class TramaGeometry:
    """パターンを物理座標 [m] に写したもの."""

    polyline: np.ndarray  # (n, 2) [m]、from → via… → to
    width: float  # 流路幅 [m]
    inlet: tuple[float, float]  # from ノード [m]
    outlet: tuple[float, float]  # to ノード [m]
    lx: float
    ly: float

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.polyline, axis=0), axis=1).sum())


def load_trama(
    path: Path,
    lx_mm: float = 600.0,
    ly_mm: float = 350.0,
    scale_mm: float | None = None,
    margin_widths: float = 1.0,
    variant: str = "orig",
) -> TramaGeometry:
    """パターン JSON → TramaGeometry（scale_mm=None なら余白 margin_widths×幅 を残して収める）.

    variant="ortho" は斜め区間（隣接点で x, y が同時に変わる区間）を、格子に沿う 2 区間
    （まず x、次に y）に置き換える（階段状不連続の有無だけを変えた対照ケース用）。
    """
    d = json.loads(Path(path).read_text())
    nodes = {k: np.asarray(v, float) for k, v in d["nodes"].items()}
    if len(d["edges"]) != 1:
        raise ValueError(f"エッジ 1 本のパターンを想定: {len(d['edges'])} 本")
    e = d["edges"][0]
    pts = np.vstack([nodes[e["from"]], np.asarray(e.get("via", []), float), nodes[e["to"]]])
    if variant == "ortho":
        out = [pts[0]]
        for q in pts[1:]:
            prev = out[-1]
            if prev[0] != q[0] and prev[1] != q[1]:
                out.append(np.array([q[0], prev[1]]))
            out.append(q)
        pts = np.vstack(out)
    elif variant != "orig":
        raise ValueError(f"variant は orig / ortho: {variant!r}")
    w = float(e["width"])
    lo = pts.min(axis=0) - w / 2
    hi = pts.max(axis=0) + w / 2
    if scale_mm is None:
        ext = hi - lo + 2 * margin_widths * w
        scale_mm = float(min(lx_mm / ext[0], ly_mm / ext[1]))
    center = (lo + hi) / 2
    dom = np.array([lx_mm, ly_mm]) / 2
    phys = ((pts - center) * scale_mm + dom) * 1e-3
    return TramaGeometry(
        polyline=phys,
        width=w * scale_mm * 1e-3,
        inlet=tuple(phys[0]),
        outlet=tuple(phys[-1]),
        lx=lx_mm * 1e-3,
        ly=ly_mm * 1e-3,
    )


def make_straight_geometry(
    angle_deg: float,
    width_mm: float = 34.5,
    length_mm: float = 240.0,
    lx_mm: float = 320.0,
    ly_mm: float = 320.0,
) -> TramaGeometry:
    """領域中央に置いた直線流路（角度 angle_deg）: 階段状不連続の効果だけを見る対照ケース."""
    t = np.deg2rad(angle_deg)
    c = np.array([lx_mm, ly_mm]) / 2
    d = np.array([np.cos(t), np.sin(t)]) * length_mm / 2
    pts = np.vstack([c - d, c + d]) * 1e-3
    return TramaGeometry(
        polyline=pts,
        width=width_mm * 1e-3,
        inlet=tuple(pts[0]),
        outlet=tuple(pts[1]),
        lx=lx_mm * 1e-3,
        ly=ly_mm * 1e-3,
    )


def distance_to_polyline(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """各点から折れ線までの最短距離."""
    P = np.stack([x, y], axis=-1)
    d = np.full(x.shape, np.inf)
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        proj = a + t[..., None] * ab
        d = np.minimum(d, np.linalg.norm(P - proj, axis=-1))
    return d


def make_trama_h(
    geo: TramaGeometry,
    nx: int,
    ny: int,
    h_channel: float,
    h_blocked: float,
) -> np.ndarray:
    xc = (np.arange(nx) + 0.5) * geo.lx / nx
    yc = (np.arange(ny) + 0.5) * geo.ly / ny
    x, y = np.meshgrid(xc, yc, indexing="ij")
    inside = distance_to_polyline(x, y, geo.polyline) <= geo.width / 2
    return np.where(inside, h_channel, h_blocked)


def make_trama_input(
    geo: TramaGeometry,
    mass_flow: float,
    dx_mm: float = 1.5,
    rho: float = 1000.0,
    mu: float = 3.0e-3,
    h_channel: float = 3.8e-3,
    h_blocked: float = 1.0e-5,
    sink_conductance: float = 1.0e-3,
    settings: NSBSettings | None = None,
    port_radius: float | None = None,
    port: str = "interior",
    sink_smooth_cells: float = 0.0,
    port_radius_factor: float = 1.0,
) -> NSBInput:
    """蛇行流路 NSBInput.

    port="interior": inlet = interior_source（円板）、outlet = interior_pressure_sink（円板、圧力基準）。
    port="wall": 両端ノードから左壁 x=0 まで流路を延長し、左壁の mass_flow_inlet / pressure_outlet
    （高さ = 流路幅）にする。
    """
    nx = int(round(geo.lx / (dx_mm * 1e-3)))
    ny = int(round(geo.ly / (dx_mm * 1e-3)))
    w2 = geo.width / 2
    if port == "interior":
        h = make_trama_h(geo, nx, ny, h_channel, h_blocked)
        r = (w2 if port_radius is None else port_radius) * port_radius_factor
        dxm = dx_mm * 1e-3
        if sink_smooth_cells > 0:
            eps = sink_smooth_cells * dxm
            patches = (
                BC.interior_source(None, mass_flow, weight=smooth_disk(*geo.inlet, r, eps)),
                BC.interior_pressure_sink(
                    None, sink_conductance, p=0.0, weight=smooth_disk(*geo.outlet, r, eps)
                ),
            )
        else:
            patches = (
                BC.interior_source(disk_mask(*geo.inlet, r), mass_flow),
                BC.interior_pressure_sink(disk_mask(*geo.outlet, r), sink_conductance, p=0.0),
            )
        bc = BC(patches=patches)
    elif port == "wall":
        ya, yb = geo.inlet[1], geo.outlet[1]
        poly = np.vstack([[[-w2, ya]], geo.polyline, [[-w2, yb]]])  # 壁の外まで伸ばして端を平らに
        geo_w = TramaGeometry(poly, geo.width, geo.inlet, geo.outlet, geo.lx, geo.ly)
        h = make_trama_h(geo_w, nx, ny, h_channel, h_blocked)
        bc = BC(
            patches=(
                BC.mass_flow_inlet(west_span(ya - w2, ya + w2), mass_flow),
                BC.pressure_outlet(west_span(yb - w2, yb + w2)),
            )
        )
    else:
        raise ValueError(f"port は interior / wall: {port!r}")
    return NSBInput(
        nx=nx,
        ny=ny,
        lx=geo.lx,
        ly=geo.ly,
        h=h,
        bc=bc,
        rho=rho,
        mu=mu,
        mu_b=mu,
        settings=settings or NSBSettings(),
    )


def describe(geo: TramaGeometry, inp: NSBInput, mass_flow: float) -> dict[str, float]:
    """代表スケール（平均流速・Re）."""
    h = float(inp.h.max())
    u_mean = mass_flow / (inp.rho * geo.width * h)
    return {
        "nx": inp.nx,
        "ny": inp.ny,
        "dx_mm": inp.dx * 1e3,
        "width_mm": geo.width * 1e3,
        "cells_across": geo.width / inp.dx,
        "path_length_m": geo.path_length,
        "u_mean": u_mean,
        "Re_h": inp.rho * u_mean * h / inp.mu,
        "Re_2h": inp.rho * u_mean * 2 * h / inp.mu,
        "Re_w": inp.rho * u_mean * geo.width / inp.mu,
        "Re_cell": inp.rho * u_mean * inp.dx / inp.mu,
        "brinkman_drag_per_u": 12 * inp.mu / h**2,
        "inertia_per_u2": inp.rho / geo.width,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pattern", nargs="?", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--mass", type=float, default=0.15, help="質量流量 [kg/s]")
    ap.add_argument("--dx", type=float, default=1.5, help="格子幅 [mm]")
    ap.add_argument("--cfl-init", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--convection", default="sou")
    ap.add_argument("--linear-solver", default="jfnk_simple")
    ap.add_argument("--jacobian", default="fou", choices=["fou", "fd"])
    ap.add_argument("--h-blocked", type=float, default=1.0e-5)
    ap.add_argument("--cfl-max", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.0, help="圧力の擬似時間項（人工圧縮性）β")
    ap.add_argument("--steady-ser", action="store_true", help="pseudo_time_in_residual=False")
    ap.add_argument("--ls", type=int, default=0, help="line_search_halvings")
    ap.add_argument(
        "--sink-smooth", type=float, default=0.0, help="ポート窓の遷移幅 [セル]（0 で階段）"
    )
    ap.add_argument("--port-radius-factor", type=float, default=1.0)
    ap.add_argument("--init-from", type=Path, default=None, help="初期場 (u, v, p) を読む npz")
    ap.add_argument(
        "--continuation",
        default="",
        help="質量流量の継続法: カンマ区切り（例 0.0015,0.005,0.015,0.05,0.15）。前段の解を流量比で拡大して初期場にする",
    )
    ap.add_argument("--port", default="interior", choices=["interior", "wall"])
    ap.add_argument("--variant", default="orig", choices=["orig", "ortho"])
    ap.add_argument(
        "--straight", type=float, default=None, help="直線流路の角度 [deg]（パターンの代わり）"
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args(argv)

    geo = (
        load_trama(a.pattern, variant=a.variant)
        if a.straight is None
        else make_straight_geometry(a.straight)
    )
    kw = {
        "newton_max_iter": a.max_iter,
        "convection": a.convection,
        "linear_solver": a.linear_solver,
        "jacobian": a.jacobian,
        "pseudo_compressibility": a.beta,
        "pseudo_time_in_residual": not a.steady_ser,
        "line_search_halvings": a.ls,
    }
    if a.cfl_max is not None:
        kw["cfl_max"] = a.cfl_max
    if a.cfl_init is not None:
        kw["cfl_init"] = a.cfl_init
    inp = make_trama_input(
        geo,
        a.mass,
        dx_mm=a.dx,
        settings=NSBSettings(**kw),
        port=a.port,
        h_blocked=a.h_blocked,
        sink_smooth_cells=a.sink_smooth,
        port_radius_factor=a.port_radius_factor,
    )
    info = describe(geo, inp, a.mass)
    print(
        "[trama] "
        + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()),
        flush=True,
    )
    print(f"[trama] inlet={geo.inlet} outlet={geo.outlet} settings={kw}", flush=True)
    t0 = time.perf_counter()
    stages: list[dict[str, Any]] = []
    if a.continuation:
        masses = [float(v) for v in a.continuation.split(",")]
        prev: NSBResult | None = None
        prev_m = masses[0]
        for m_k in masses:
            inp_k = make_trama_input(
                geo,
                m_k,
                dx_mm=a.dx,
                settings=inp.settings,
                port=a.port,
                h_blocked=a.h_blocked,
                sink_smooth_cells=a.sink_smooth,
                port_radius_factor=a.port_radius_factor,
            )
            if prev is not None:
                ratio = m_k / prev_m
                inp_k = NSBInput(
                    **{k: v for k, v in inp_k.__dict__.items() if k not in ("u0", "v0", "p0")},
                    u0=prev.u * ratio,
                    v0=prev.v * ratio,
                    p0=prev.p * ratio,
                )
            print(
                f"[trama] === continuation stage mass={m_k:g} (init from {prev_m:g}) ===",
                flush=True,
            )
            res = solve_steady(inp_k, log=lambda m: print(m, flush=True))
            stages.append(
                {
                    "mass": m_k,
                    "converged": bool(res.converged),
                    "reason": res.failure_reason,
                    "n_iter": int(res.n_iter),
                    "rel_steady_final": float(res.rel_steady_residual),
                    "elapsed": float(res.elapsed),
                }
            )
            print(
                f"[trama] stage mass={m_k:g}: converged={res.converged} it={res.n_iter} rel={res.rel_steady_residual:.2e}",
                flush=True,
            )
            if not res.converged:
                break
            prev, prev_m = res, m_k
        inp = inp_k
    else:
        if a.init_from is not None:
            z = np.load(a.init_from)
            inp = NSBInput(
                **{k: v for k, v in inp.__dict__.items() if k not in ("u0", "v0", "p0")},
                u0=z["u"],
                v0=z["v"],
                p0=z["p"],
            )
            print(f"[trama] init from {a.init_from}", flush=True)
        res = solve_steady(inp, log=lambda m: print(m, flush=True))
    geo_tag = f"straight{a.straight:g}" if a.straight is not None else a.variant
    tag = a.tag or (
        f"{geo_tag}-{a.port}-m{a.mass:g}-dx{a.dx:g}-{a.convection}-{a.linear_solver}-{a.jacobian}"
        + (f"-hb{a.h_blocked:g}" if a.h_blocked != 1.0e-5 else "")
        + (f"-beta{a.beta:g}" if a.beta > 0 else "")
        + (f"-cfl{a.cfl_init:g}" if a.cfl_init is not None else "")
        + ("-sser" if a.steady_ser else "")
        + (f"-ls{a.ls}" if a.ls > 0 else "")
        + (f"-smooth{a.sink_smooth:g}" if a.sink_smooth > 0 else "")
        + (f"-prf{a.port_radius_factor:g}" if a.port_radius_factor != 1.0 else "")
        + ("-cont" if a.continuation else "")
        + ("-init" if a.init_from is not None else "")
    )
    out: dict[str, Any] = dict(summary(inp, res))
    out.update(
        {f"case_{k}": (float(v) if isinstance(v, float) else int(v)) for k, v in info.items()}
    )
    out["residual_history"] = [float(v) for v in res.residual_history]
    out["steady_residual_history"] = [float(v) for v in res.steady_residual_history]
    out["cfl_history"] = [float(v) for v in res.cfl_history]
    out["elapsed_total"] = time.perf_counter() - t0
    if stages:
        out["continuation"] = stages
        out["n_iter_total"] = int(sum(st["n_iter"] for st in stages))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"trama_{tag}.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    save_fields(a.out / f"trama_{tag}_fields.npz", inp, res)
    print(
        f"[trama] saved {a.out / f'trama_{tag}.yaml'} converged={res.converged} it={res.n_iter}",
        flush=True,
    )
    return 0 if res.converged else 1


if __name__ == "__main__":
    sys.exit(main())
