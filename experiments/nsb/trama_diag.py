"""trama ケースの最初の線形系の GMRES 診断（前処理の弱点の切り分け）と場の診断マップ.

同じ右辺 b = −R(x_stokes) に対して
  matvec ∈ {fd: JFNK の有限差分（solve_linear と同じ）, j1: J1+τ の厳密な積}
  precond ∈ {simple: SIMPLE 型ブロック前処理, pardiso: LU(J1+τ)}
の 4 組で Arnoldi（再出発なし、m 本）を回し、
  - 反復ごとの GMRES 残差（Givens 推定）と最後の真の残差 |b − A x|/|b|
  - 前処理付き作用素 A M⁻¹ の Ritz 値（H_m の固有値）
を記録する。j1×pardiso は恒等作用素なので 1 反復で落ちるはず（検算）。
j1×simple: SIMPLE 近似だけの効き。fd×pardiso: J1 と真の作用素の食い違いだけの効き。
fd×simple: 本番と同じ。

場のマップ（Stokes 場の速さ、セル Re、a_P の対流割合、RC 係数、階段セル、
SIMPLE で失敗した後の残差の空間分布、J1τ⁻¹b の空間分布）も同じ npz に保存する。

使用例::

    python experiments/nsb/trama_diag.py --mass 0.15 2>&1 | tee experiments/nsb/logs/trama-diag-$(date +%s).log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.nsb.trama_case import (  # noqa: E402
    DEFAULT_PATTERN,
    load_trama,
    make_straight_geometry,
    make_trama_input,
)
from experiments.nsb.trama_lin import FirstStep, first_step  # noqa: E402
from nsb import NSBSettings  # noqa: E402
from nsb.linalg import PardisoLU  # noqa: E402
from nsb.precond import SimpleBlockPreconditioner  # noqa: E402

HERE = Path(__file__).resolve().parent
Op = Callable[[np.ndarray], np.ndarray]


def arnoldi_gmres(
    matvec: Op, precond: Op, b: np.ndarray, m: int
) -> tuple[np.ndarray, list[float], float, np.ndarray]:
    """右前処理 Arnoldi（再出発なし）。戻り値 (Ritz 値, Givens 残差比の履歴, 真の残差比, x)."""
    n = b.size
    beta = float(np.linalg.norm(b))
    V = np.zeros((m + 1, n))
    Z = np.zeros((m, n))
    H = np.zeros((m + 1, m))
    V[0] = b / beta
    hist: list[float] = []
    e1 = np.zeros(m + 1)
    e1[0] = beta
    j_done = 0
    for j in range(m):
        Z[j] = precond(V[j])
        w = matvec(Z[j])
        # CGS2
        for _ in range(2):
            h = V[: j + 1] @ w
            w = w - V[: j + 1].T @ h
            H[: j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        j_done = j + 1
        y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], e1[: j + 2], rcond=None)
        hist.append(float(np.linalg.norm(H[: j + 2, : j + 1] @ y - e1[: j + 2]) / beta))
        if H[j + 1, j] < 1e-14 * beta:
            break
        V[j + 1] = w / H[j + 1, j]
    Hm = H[:j_done, :j_done]
    ritz = np.linalg.eigvals(Hm)
    y, *_ = np.linalg.lstsq(H[: j_done + 1, :j_done], e1[: j_done + 1], rcond=None)
    x = Z[:j_done].T @ y
    true_ratio = float(np.linalg.norm(b - matvec(x)) / beta)
    return ritz, hist, true_ratio, x


def gmres_matrix(fs: FirstStep, m: int, s: NSBSettings) -> tuple[dict, dict[str, np.ndarray]]:
    n = fs.n
    J = fs.J1_tau
    b = fs.rhs
    out: dict = {}
    fields: dict[str, np.ndarray] = {}
    pc_simple = SimpleBlockPreconditioner(
        n, s.simple_schur_cycles, s.simple_ilu_drop_tol, s.simple_ilu_fill_factor
    ).factorize(J)
    pc_lu = PardisoLU().factorize(J)
    mats = {"j1": lambda v: J @ v, "fd": fs.fd_matvec}
    pcs = {"simple": pc_simple.solve, "pardiso": pc_lu.solve}
    for mk, mv in mats.items():
        for pk, pv in pcs.items():
            t0 = time.perf_counter()
            ritz, hist, true_ratio, x = arnoldi_gmres(mv, pv, b, m)
            key = f"{mk}-{pk}"
            out[key] = {
                "hist": hist,
                "true_ratio": true_ratio,
                "ritz_re": ritz.real.tolist(),
                "ritz_im": ritz.imag.tolist(),
                "norm_x": float(np.linalg.norm(x)),
                "elapsed": time.perf_counter() - t0,
            }
            r = b - mv(x)
            fields[f"resid_{key}"] = r
            fields[f"x_{key}"] = x
            print(
                f"[diag] {key}: it={len(hist)} givens={hist[-1]:.3e} true={true_ratio:.3e} "
                f"|x|={np.linalg.norm(x):.3e} ritz |λ| in [{abs(ritz).min():.3e},{abs(ritz).max():.3e}] "
                f"re<0: {(ritz.real < 0).sum()} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    pc_simple.free()
    pc_lu.free()
    return out, fields


def field_maps(fs: FirstStep) -> dict[str, np.ndarray]:
    disc = fs.disc
    shape = (disc.nx, disc.ny)
    u, v, p = disc.split(fs.x)
    st = fs.st
    rho, dx, dy, vol = disc.rho, disc.dx, disc.dy, disc.vol
    fx_lin = rho * dy * st.ufx
    fy_lin = rho * dx * st.vfy
    a_conv = (
        np.maximum(fx_lin[1:], 0.0)
        + np.maximum(-fx_lin[:-1], 0.0)
        + np.maximum(fy_lin[:, 1:], 0.0)
        + np.maximum(-fy_lin[:, :-1], 0.0)
    )
    a_diff = disc.diff_diag
    a_drag = disc.drag * vol
    tau = fs.tau.reshape(shape)
    h = disc.inp.thickness
    chan = h > 1e-4
    # 階段セル: 流路セルで、x 方向と y 方向の両方に閉塞セルの隣接を持つ（凸角）
    blocked = ~chan
    nb_x = np.zeros(shape, bool)
    nb_y = np.zeros(shape, bool)
    nb_x[1:] |= blocked[:-1]
    nb_x[:-1] |= blocked[1:]
    nb_y[:, 1:] |= blocked[:, :-1]
    nb_y[:, :-1] |= blocked[:, 1:]
    stair = chan & nb_x & nb_y
    wall_adj = chan & (nb_x | nb_y)
    speed = np.hypot(u, v)
    return {
        "u": u,
        "v": v,
        "p": p,
        "speed": speed,
        "h": h,
        "re_cell": rho * speed * dx / disc.mu,
        "a_conv": a_conv,
        "a_diff": a_diff,
        "a_drag": a_drag,
        "a_p": st.a_p,
        "tau": tau,
        "conv_frac": a_conv / (a_conv + a_diff + a_drag),
        "tau_over_ap": tau / st.a_p,
        "d_cell": vol / st.a_p,
        "stair": stair,
        "wall_adj": wall_adj,
        "q_src": disc.q_src,
        "c_sink": disc.c_sink,
        "R_u": fs.steady_resid(fs.x)[: disc.n].reshape(shape),
        "R_v": fs.steady_resid(fs.x)[disc.n : 2 * disc.n].reshape(shape),
        "R_p": fs.steady_resid(fs.x)[2 * disc.n :].reshape(shape),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--port", default="interior")
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--cfl", type=float, default=None)
    ap.add_argument("--m", type=int, default=150, help="Arnoldi 次元")
    ap.add_argument("--straight", type=float, default=None)
    ap.add_argument("--variant", default="orig")
    ap.add_argument("--convection", default="sou")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args(argv)
    geo = (
        load_trama(DEFAULT_PATTERN, variant=a.variant)
        if a.straight is None
        else make_straight_geometry(a.straight)
    )
    s = NSBSettings(convection=a.convection)
    inp = make_trama_input(geo, a.mass, dx_mm=a.dx, settings=s, port=a.port)
    fs = first_step(inp, cfl=a.cfl)
    gtag = f"straight{a.straight:g}" if a.straight is not None else a.variant
    label = f"{gtag}-{a.port}-m{a.mass:g}-dx{a.dx:g}-cfl{fs.cfl:g}-{a.convection}"
    print(
        f"[diag] {label}: n={fs.n} |R_ref|={fs.r_ref:.4e} |x|={np.linalg.norm(fs.x):.3e} "
        f"tau[min,max]=[{fs.tau.min():.3e},{fs.tau.max():.3e}]",
        flush=True,
    )
    res, fields = gmres_matrix(fs, a.m, s)
    maps = field_maps(fs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"trama_diag_{label}.json").write_text(
        json.dumps(
            {"label": label, "n": fs.n, "cfl": fs.cfl, "r_ref": fs.r_ref, "gmres": res}, indent=1
        )
    )
    np.savez_compressed(
        a.out / f"trama_diag_{label}.npz", nx=fs.disc.nx, ny=fs.disc.ny, b=fs.rhs, **fields, **maps
    )
    print(f"[diag] saved {a.out / f'trama_diag_{label}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
