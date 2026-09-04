"""保存した場 (*_fields.npz) から速度・圧力分布図を描く.

使用例::

    python experiments/brinkman_uturn/plot_fields.py experiments/brinkman_uturn/results_cfl0.5/uturn_r1_U2_second_order_upwind_jfnk_fields.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]

LX, LY = 0.7, 0.4
HERE = Path(__file__).resolve().parent


def plot(npz_path: Path, out_dir: Path) -> Path:
    d = np.load(npz_path)
    u, v, p, h = d["u"], d["v"], d["p"], d["h"]
    nx, ny = u.shape
    x = (np.arange(nx) + 0.5) * LX / nx
    y = (np.arange(ny) + 0.5) * LY / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    speed = np.hypot(u, v)
    blocked = h < 1e-4
    tag = npz_path.stem.replace("_fields", "")
    u_in = float(tag.split("_U")[1].split("_")[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    fig.suptitle(f"{tag}  (nx={nx}, ny={ny}, U_in={u_in:g} m/s)", fontsize=12)

    ax = axes[0, 0]
    im = ax.pcolormesh(X, Y, speed, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|u| [m/s]")
    step = max(1, nx // 36)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="w",
        scale=u_in * 25,
        width=0.002,
    )
    ax.set_title("速度の大きさ + ベクトル")

    ax = axes[0, 1]
    im = ax.pcolormesh(X, Y, p, cmap="coolwarm", shading="auto")
    fig.colorbar(im, ax=ax, label="p [Pa]")
    ax.contour(X, Y, p, levels=20, colors="k", linewidths=0.4)
    ax.set_title("圧力")

    ax = axes[1, 0]
    ax.pcolormesh(X, Y, speed, cmap="viridis", shading="auto")
    us = np.where(blocked, np.nan, u)
    vs = np.where(blocked, np.nan, v)
    ax.streamplot(x, y, us.T, vs.T, color="w", density=1.4, linewidth=0.6, arrowsize=0.8)
    ax.set_title("流線")

    ax = axes[1, 1]
    im = ax.pcolormesh(X, Y, u, cmap="RdBu_r", shading="auto", vmin=-u_in, vmax=u_in)
    fig.colorbar(im, ax=ax, label="u [m/s]")
    ax.set_title("x 方向速度 u")

    for ax in axes.ravel():
        if blocked.any():
            ax.contour(X, Y, blocked.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    out = out_dir / f"{tag}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    for arg in sys.argv[1:]:
        print(plot(Path(arg), out_dir))


if __name__ == "__main__":
    main()
