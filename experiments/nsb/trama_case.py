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
    variant: str = (
        "orig"  # 中心線の作り方（orig / ortho / lead）。OF 側と揃えるため case.json に載せる
    )

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.polyline, axis=0), axis=1).sum())


def prepend_lead(
    pts: np.ndarray, w: float, back_widths: float = 3.0, stub_widths: float = 5.0
) -> np.ndarray:
    """入口側に「助走 + 90 度の曲がり」を継ぎ足した中心線を返す.

    幾何: 元の中心線の第 1 区間の向きを d、その右手法線を n = (d_y, -d_x) とする。
    入口ノード a から d の**逆向き**に back_widths×w 戻った点を曲がり角 corner に置き、
    そこから n 方向に stub_widths×w 伸ばした先を新しい入口にする。

        新入口 ●──stub──> corner ─back─> a ──(元のパターン)──>
                          └ 90 度

    この向きなら（trama の第 1 区間 d = +x に対して）助走脚は入口ノードの左下に伸び、
    流れは「上向きに助走 → 右へ 90 度曲がる → 元の蛇行に入る」になる。ポートの噴流が
    発達しきってから曲がりに入るので、ポート境界条件と曲がりの影響を分けて見られる。
    """
    d = pts[1] - pts[0]
    n_d = float(np.linalg.norm(d))
    if n_d == 0.0:
        raise ValueError("中心線の第 1 区間の長さが 0 です")
    d = d / n_d
    n = np.array([d[1], -d[0]])
    corner = pts[0] - d * back_widths * w
    start = corner + n * stub_widths * w
    return np.vstack([start[None, :], corner[None, :], pts])


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
    variant="lead" は入口側に助走脚と 90 度の曲がりを継ぎ足す（`prepend_lead`）。
    """
    d = json.loads(Path(path).read_text())
    nodes = {k: np.asarray(v, float) for k, v in d["nodes"].items()}
    if len(d["edges"]) != 1:
        raise ValueError(f"エッジ 1 本のパターンを想定: {len(d['edges'])} 本")
    e = d["edges"][0]
    pts = np.vstack([nodes[e["from"]], np.asarray(e.get("via", []), float), nodes[e["to"]]])
    w = float(e["width"])
    if variant == "ortho":
        out = [pts[0]]
        for q in pts[1:]:
            prev = out[-1]
            if prev[0] != q[0] and prev[1] != q[1]:
                out.append(np.array([q[0], prev[1]]))
            out.append(q)
        pts = np.vstack(out)
    elif variant == "lead":
        pts = prepend_lead(pts, w)
    elif variant != "orig":
        raise ValueError(f"variant は orig / ortho / lead: {variant!r}")
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
        variant=variant,
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
    friction_re_crit: float = 0.0,
    h_solid: float = 0.0,
) -> NSBInput:
    """蛇行流路 NSBInput.

    port="interior": inlet = interior_source（円板）、outlet = interior_pressure_sink（円板、圧力基準）。
    port="carve": 円板セルを刳り抜き、露出したリング面を inlet（法線流入速度）/ outlet（p 固定）に
    する。OpenFOAM の `cylinderToCell` + `subsetMesh` と 1 対 1（`trama_of_case.py` 参照）。
    OF 側は円周が閉塞域に接すると圧力が跳ねるので `port_shrink_cells=2` で 2 セル縮めている。
    ここでも `port_radius_factor` ではなく **2 セルぶん縮めた半径**を既定にして揃える。
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
    elif port == "carve":
        h = make_trama_h(geo, nx, ny, h_channel, h_blocked)
        dxm = dx_mm * 1e-3
        # OpenFOAM の port_shrink_cells=2 と同じ縮め方（円周を流路の内側に置く）
        r = (w2 - 2.0 * dxm if port_radius is None else port_radius) * port_radius_factor
        if r <= 2.0 * dxm:
            raise ValueError(f"ポート半径が小さすぎる: {r:g} m（dx={dxm:g} m）")
        bc = BC(
            patches=(
                BC.port_inlet(disk_mask(*geo.inlet, r), mass_flow),
                BC.port_outlet(disk_mask(*geo.outlet, r), p=0.0),
            )
        )
    elif port == "wall":
        ya, yb = geo.inlet[1], geo.outlet[1]
        poly = np.vstack([[[-w2, ya]], geo.polyline, [[-w2, yb]]])  # 壁の外まで伸ばして端を平らに
        geo_w = TramaGeometry(poly, geo.width, geo.inlet, geo.outlet, geo.lx, geo.ly, geo.variant)
        h = make_trama_h(geo_w, nx, ny, h_channel, h_blocked)
        bc = BC(
            patches=(
                BC.mass_flow_inlet(west_span(ya - w2, ya + w2), mass_flow),
                BC.pressure_outlet(west_span(yb - w2, yb + w2)),
            )
        )
    else:
        raise ValueError(f"port は interior / carve / wall: {port!r}")
    return NSBInput(
        nx=nx,
        ny=ny,
        lx=geo.lx,
        ly=geo.ly,
        h=h,
        bc=bc,
        h_solid=h_solid,
        rho=rho,
        mu=mu,
        mu_b=mu,
        settings=settings or NSBSettings(),
        friction_re_crit=friction_re_crit,
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


def make_tag(a: argparse.Namespace) -> str:
    geo_tag = f"straight{a.straight:g}" if a.straight is not None else a.variant
    return a.tag or (
        f"{geo_tag}-{a.port}-m{a.mass:g}-dx{a.dx:g}-{a.convection}-{a.linear_solver}-{a.jacobian}"
        + (f"-hb{a.h_blocked:g}" if a.h_blocked != 1.0e-5 else "")
        + (f"-hs{a.h_solid:g}" if a.h_solid > 0 else "")
        + (f"-beta{a.beta:g}" if a.beta > 0 else "")
        + (f"-cfl{a.cfl_init:g}" if a.cfl_init is not None else "")
        + ("-sser" if a.steady_ser else "")
        + (f"-ls{a.ls}" if a.ls > 0 else "")
        + (f"-smooth{a.sink_smooth:g}" if a.sink_smooth > 0 else "")
        + (f"-prf{a.port_radius_factor:g}" if a.port_radius_factor != 1.0 else "")
        + ("-cont" if a.continuation else "")
        + ("-init" if a.init_from is not None else "")
        + (f"-frz{a.freeze:g}" if a.freeze > 0 else "")
        + (f"-rf{a.refreeze}" if a.refreeze > 0 else "")
        + (f"-fric{a.friction:g}" if a.friction > 0 else "")
        + (f"-dt{a.unsteady:g}" if a.unsteady > 0 else "")
    )


def run_unsteady(
    a: argparse.Namespace, geo: TramaGeometry, inp: NSBInput, info: dict[str, float]
) -> int:
    """物理時間の陰的非定常計算（`nsb.unsteady.solve_unsteady`）を走らせ、時系列とスナップショットを保存する."""
    from nsb.unsteady import solve_unsteady

    if a.init_from is not None:
        z = np.load(a.init_from)
        inp = NSBInput(
            **{k: v for k, v in inp.__dict__.items() if k not in ("u0", "v0", "p0")},
            u0=z["u"],
            v0=z["v"],
            p0=z["p"],
        )
        print(f"[trama] init from {a.init_from}", flush=True)
    t0 = time.perf_counter()
    res = solve_unsteady(
        inp,
        dt=a.unsteady,
        n_steps=a.n_steps,
        log=lambda m: print(m, flush=True),
        newton_max=a.newton_per_step,
        step_tol=a.step_tol,
        save_every=a.save_every,
        avg_start=a.avg_start,
        stop_at_steady=not a.no_stop_at_steady,
    )
    tag = make_tag(a)
    out: dict[str, Any] = {
        "reached_steady": bool(res.reached_steady),
        "reason": res.failure_reason,
        "dt": float(res.dt),
        "n_steps": len(res.times),
        "t_final": float(res.times[-1]) if res.times else 0.0,
        "rel_steady_final": float(res.steady_residual[-1]) if res.steady_residual else float("nan"),
        "rel_steady_min": float(min(res.steady_residual)) if res.steady_residual else float("nan"),
        "steps_hit_max_newton": int(res.n_steps_hit_max_newton),
        "dt_backoffs": int(res.n_dt_backoffs),
        "avg_window": [float(v) for v in res.avg_window],
        "newton_total": int(sum(res.newton_iters)),
        "gmres_total": int(sum(res.gmres_iters)),
        "residual_ref": float(res.residual_ref),
        "elapsed": float(res.elapsed),
        "elapsed_total": time.perf_counter() - t0,
    }
    out.update(
        {f"case_{k}": (float(v) if isinstance(v, float) else int(v)) for k, v in info.items()}
    )
    out["times"] = [float(v) for v in res.times]
    out["steady_residual"] = [float(v) for v in res.steady_residual]
    out["step_residual"] = [float(v) for v in res.step_residual]
    out["newton_iters"] = [int(v) for v in res.newton_iters]
    out["gmres_iters"] = [int(v) for v in res.gmres_iters]
    out["dt_history"] = [float(v) for v in res.dt_history]
    out["probes"] = {k: [float(x) for x in v] for k, v in res.probes.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"trama_{tag}.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    avg = (
        {}
        if res.mean_u is None
        else {
            "mean_u": res.mean_u,
            "mean_v": res.mean_v,
            "mean_p": res.mean_p,
            "rms_u": res.rms_u,
            "rms_v": res.rms_v,
            "avg_window": np.array(res.avg_window),
        }
    )
    np.savez_compressed(
        a.out / f"trama_{tag}_fields.npz",
        u=res.u,
        v=res.v,
        p=res.p,
        h=inp.h,
        **avg,
        snap_t=np.array([sn[0] for sn in res.snapshots]),
        snap_u=np.array([sn[1] for sn in res.snapshots]),
        snap_v=np.array([sn[2] for sn in res.snapshots]),
        snap_p=np.array([sn[3] for sn in res.snapshots]),
    )
    print(
        f"[trama] saved {a.out / f'trama_{tag}.yaml'} reached_steady={res.reached_steady} "
        f"steps={len(res.times)} rel_final={out['rel_steady_final']:.2e}",
        flush=True,
    )
    return 0 if res.reached_steady else 1


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
    ap.add_argument(
        "--venkat-k", type=float, default=5.0, help="Venkatakrishnan 定数 K（大きいほど緩い）"
    )
    ap.add_argument(
        "--rc-dt",
        action="store_true",
        help="[非定常] Rhie–Chow 係数に時間微分の対角を入れる d_f = V/(a_P + ρV/Δt)"
        "（OpenFOAM の 1/A と同じ形）",
    )
    ap.add_argument("--linear-solver", default="jfnk_simple")
    ap.add_argument("--jacobian", default="fou", choices=["fou", "fd"])
    ap.add_argument("--h-blocked", type=float, default=1.0e-5)
    ap.add_argument(
        "--h-solid",
        type=float,
        default=0.0,
        help="[壁セル] この厚さ以下のセルを壁として解かない [m]（0 で無効。閉塞域なら 1e-4 程度）",
    )
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
    ap.add_argument("--port", default="interior", choices=["interior", "carve", "wall"])
    ap.add_argument("--variant", default="orig", choices=["orig", "ortho", "lead"])
    ap.add_argument("--lx", type=float, default=600.0, help="領域の横 [mm]")
    ap.add_argument("--ly", type=float, default=350.0, help="領域の縦 [mm]")
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="パターン単位 → mm の倍率。既定 None は領域に収まるよう自動。縮小ジオメトリで"
        "流路幅と格子を本番と揃えたいときに明示する（trama は 4.928149）",
    )
    ap.add_argument(
        "--straight", type=float, default=None, help="直線流路の角度 [deg]（パターンの代わり）"
    )
    ap.add_argument(
        "--freeze", type=float, default=0.0, help="limiter_freeze_rel（0 で無効。nsbp は 1e-3）"
    )
    ap.add_argument(
        "--refreeze", type=int, default=0, help="limiter_refreeze_max（ψ の Picard 反復）"
    )
    ap.add_argument(
        "--friction",
        type=float,
        default=0.0,
        help="隙間の摩擦則 Re_c（Blasius 型なら 2040。0 で層流 12μ/h² のみ）",
    )
    ap.add_argument(
        "--unsteady", type=float, default=0.0, help="物理時間刻み Δt [s]（>0 で非定常計算）"
    )
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--newton-per-step", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--step-tol", type=float, default=1e-3)
    ap.add_argument(
        "--avg-start",
        type=float,
        default=-1.0,
        help="この時刻 [s] 以降を時間平均する（OpenFOAM の fieldAverage と同じ量。負で無効）",
    )
    ap.add_argument(
        "--no-stop-at-steady",
        action="store_true",
        help="定常残差が閾値を割っても止めずに最後まで進める（時間平均を取り切るため）",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args(argv)

    geo = (
        load_trama(a.pattern, lx_mm=a.lx, ly_mm=a.ly, scale_mm=a.scale, variant=a.variant)
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
        "venkat_k": a.venkat_k,
        "rc_with_pseudo_time": a.rc_dt,
        "limiter_freeze_rel": a.freeze,
        "limiter_refreeze_max": a.refreeze,
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
        friction_re_crit=a.friction,
        h_solid=a.h_solid,
    )
    info = describe(geo, inp, a.mass)
    print(
        "[trama] "
        + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()),
        flush=True,
    )
    print(f"[trama] inlet={geo.inlet} outlet={geo.outlet} settings={kw}", flush=True)
    t0 = time.perf_counter()
    if a.unsteady > 0:
        return run_unsteady(a, geo, inp, info)
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
                friction_re_crit=a.friction,
                h_solid=a.h_solid,
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
                    "limiter_frozen_at": int(res.limiter_frozen_at),
                    "rel_unfrozen": float(res.residual_unfrozen / res.residual_ref),
                    "steady_residual_history": [
                        float(v / res.residual_ref) for v in res.steady_residual_history
                    ],
                    "cfl_history": [float(v) for v in res.cfl_history],
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
    tag = make_tag(a)
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
