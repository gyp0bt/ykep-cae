"""吸出マニホールドの位置・径を連続設計変数にして圧力損失を勾配法で下げるデモ（flat 72×48）.

θ = (cx, cy, r)。注入マニホールドは固定、目的関数は注入部の平均圧力（= 圧損）。
勾配は nsb.adjoint（彩色 FD ヤコビアン + 随伴）で求め、初回は全体解き直しの中心差分と照合する。
径には上限 r ≤ 0.08 m の射影を入れる（径を大きくすれば圧損は単調に下がるため）。

使用例::

    python experiments/nsb/manifold_optimize.py 2>&1 | tee experiments/nsb/logs/manifold-opt-$(date +%s).log
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from nsb import BC, ImplicitSolve, NSBInput, NSBSettings, source_mean_pressure_objective
from nsb.geo import LX, LY
from nsb.utils import save_fields
from xkep_cae_fluid.brinkman_flow import disk_mask, smooth_disk

HERE = Path(__file__).resolve().parent
NX, NY = 72, 48
MDOT, COND = 0.1, 1e-4
R_MAX = 0.08
BOUNDS = np.array([[0.1, LX - 0.1], [0.1, LY - 0.1], [0.02, R_MAX]])


def build(theta: np.ndarray) -> NSBInput:
    cx, cy, r = theta
    eps = LX / NX
    bc = BC(
        patches=(
            BC.interior_source(disk_mask(0.15, 0.2, 0.05), MDOT),
            BC.interior_pressure_sink(None, COND, p=0.0, weight=smooth_disk(cx, cy, r, eps)),
        )
    )
    return NSBInput(
        nx=NX,
        ny=NY,
        lx=LX,
        ly=LY,
        h=np.full((NX, NY), 1e-3),
        bc=bc,
        settings=NSBSettings(
            velocity_floor=0.05,
            pseudo_time_in_residual=False,
            alpha_u=1.0,
            init_field="stokes",
            newton_tol=1e-9,
            newton_max_iter=120,
        ),
    )


def main() -> None:
    prob = ImplicitSolve(build)
    obj = source_mean_pressure_objective()
    theta = np.array([0.55, 0.30, 0.04])
    hist: list[dict] = []

    print("===== 勾配の照合（θ0）=====", flush=True)
    t0 = time.perf_counter()
    res, x = prob.forward(theta)
    f, g = prob.gradient(theta, x, obj)
    t_grad = time.perf_counter() - t0
    g_fd = np.zeros(3)
    for k, h in enumerate([2e-3, 2e-3, 1e-3]):
        e = np.zeros(3)
        e[k] = h
        _, xp = prob.forward(theta + e, init=res)
        _, xm = prob.forward(theta - e, init=res)
        g_fd[k] = (obj.value(xp, build(theta + e)) - obj.value(xm, build(theta - e))) / (2 * h)
    print(
        f"f={f:.2f} Pa  adjoint grad={g}  fd grad={g_fd}  rel.err={np.abs(g - g_fd) / np.abs(g_fd)}"
    )
    print(f"forward+gradient {t_grad:.1f}s（forward {res.n_iter} 反復）", flush=True)

    print("===== 射影勾配法 =====", flush=True)
    step = np.array([0.02, 0.02, 0.01]) / np.abs(g)  # 初回勾配で正規化したステップ
    init = res
    for it in range(15):
        theta_new = np.clip(theta - step * g, BOUNDS[:, 0], BOUNDS[:, 1])
        res_new, x_new = prob.forward(theta_new, init=init)
        f_new, g_new = prob.gradient(theta_new, x_new, obj)
        if f_new > f:  # 改善しなければステップ半減
            step *= 0.5
            print(f"  it={it} f={f_new:.2f} > {f:.2f}: step 半減", flush=True)
            continue
        theta, f, g, init = theta_new, f_new, g_new, res_new
        hist.append({"it": it, "theta": theta.tolist(), "f": float(f), "grad": g.tolist()})
        print(
            f"  it={it} θ=({theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.4f}) f={f:.2f} Pa |g|={np.linalg.norm(g):.3g}",
            flush=True,
        )
        if np.linalg.norm(step * g) < 1e-4:
            break

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "manifold_optimize.yaml").write_text(
        yaml.safe_dump(
            {"grad_check": {"adjoint": g.tolist(), "fd": g_fd.tolist()}, "history": hist},
            sort_keys=False,
        )
    )
    save_fields(HERE / "results" / "manifold_opt_final_fields.npz", build(theta), init)
    print(f"final θ={theta} f={f:.2f} Pa")


if __name__ == "__main__":
    main()
