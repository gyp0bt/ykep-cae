"""status-46 の図: リミター凍結・摩擦則の継続法の残差履歴（f8）と物理時間非定常の時系列（f9）.

走行中でも描けるようにログを直接読む（yaml は走行終了時にしか書かれない）。

使用例::

    python experiments/nsb/trama_plots2.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"font.family": ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"], "font.size": 10}
)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
FIGS = HERE / "results" / "trama_figs"
# dataviz の既定カテゴリ順（固定順で割当て、循環させない）
CAT = ["#1f5fbf", "#d1495b", "#3a9d5d", "#e8a33d", "#7b4fa6", "#3aa6b9", "#8c6d31"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dddddd"

RE_STAGE = re.compile(r"=== continuation stage mass=([0-9.e+-]+)")
RE_IT = re.compile(
    r"\[nsb\] it=(\d+) \|R_tau\|=.*\|R_steady\|/\|R_ref\|=([0-9.e+-]+) cfl=([0-9.e+-]+)"
)
RE_FRZ = re.compile(r"\[nsb\] it=(\d+) limiter frozen")
RE_UNF = re.compile(r"\[nsb\] it=(\d+) limiter unfrozen")
RE_STAGE_DONE = re.compile(r"stage mass=([0-9.e+-]+): converged=(\w+) it=(\d+) rel=([0-9.e+-]+)")
RE_STEP = re.compile(
    r"\[nsb-t\] step=(\d+) t=([0-9.e+-]+) dt=([0-9.e+-]+) \|R_steady\|/\|R_ref\|=([0-9.e+-]+) "
    r"\|R_tau\|/\|R_ref\|=([0-9.e+-]+) \(start [0-9.e+-]+\) newton=(\d+) gmres=(\d+) "
    r"speed_max=([0-9.e+-]+) KE=([0-9.e+-]+) dp=([0-9.e+-]+) m_out=([0-9.e+-]+)"
)


def parse_continuation(path: Path) -> list[dict]:
    """段ごとの (反復, 定常残差比, cfl) と凍結/解凍イベント."""
    stages: list[dict] = []
    cur: dict | None = None
    for line in path.read_text().splitlines():
        m = RE_STAGE.search(line)
        if m:
            cur = {
                "mass": float(m.group(1)),
                "it": [],
                "rel": [],
                "cfl": [],
                "frozen": [],
                "unfrozen": [],
            }
            stages.append(cur)
            continue
        if cur is None:
            continue
        m = RE_IT.search(line)
        if m:
            cur["it"].append(int(m.group(1)))
            cur["rel"].append(float(m.group(2)))
            cur["cfl"].append(float(m.group(3)))
            continue
        m = RE_FRZ.search(line)
        if m:
            cur["frozen"].append(int(m.group(1)))
            continue
        m = RE_UNF.search(line)
        if m:
            cur["unfrozen"].append(int(m.group(1)))
            continue
        m = RE_STAGE_DONE.search(line)
        if m:
            cur["converged"] = m.group(2) == "True"
            cur["n_iter"] = int(m.group(3))
            cur["rel_final"] = float(m.group(4))
    return stages


def parse_unsteady(path: Path) -> dict[str, np.ndarray]:
    rows = [RE_STEP.search(line) for line in path.read_text().splitlines()]
    rows = [m.groups() for m in rows if m]
    if not rows:
        return {}
    a = np.array(rows, dtype=float)
    keys = [
        "step",
        "t",
        "dt",
        "rel",
        "rel_tau",
        "newton",
        "gmres",
        "speed_max",
        "ke",
        "dp",
        "m_out",
    ]
    return {k: a[:, i] for i, k in enumerate(keys)}


def latest(pattern: str) -> Path | None:
    files = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def fig_continuation(runs: list[tuple[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, pattern) in zip(axes, runs, strict=True):
        f = latest(pattern)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.5)
        ax.axhline(1e-6, color=MUTED, lw=0.8, ls="--")
        if f is None:
            ax.text(0.5, 0.5, "no log", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        offset = 0
        for k, st in enumerate(parse_continuation(f)):
            if not st["it"]:
                continue
            it = np.array(st["it"]) + offset
            col = CAT[k % len(CAT)]
            ax.plot(it, st["rel"], color=col, lw=1.6, label=f"ṁ={st['mass']:g}")
            for i0 in st["frozen"]:
                ax.axvline(i0 + offset, color=col, lw=0.8, ls=":")
            for i0 in st["unfrozen"]:
                ax.plot([i0 + offset], [np.interp(i0, st["it"], st["rel"])], "x", color=col, ms=6)
            offset += max(st["it"])
        ax.set_xlabel("累積 Newton 反復", color=INK)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("|R_steady| / |R_ref|", color=INK)
    fig.suptitle("継続法の定常残差（点線: リミター凍結、×: 解凍）", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_unsteady(runs: list[tuple[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    ax_r, ax_s, ax_k, ax_d = axes.ravel()
    for k, (label, pattern) in enumerate(runs):
        f = latest(pattern)
        if f is None:
            continue
        d = parse_unsteady(f)
        if not d:
            continue
        col = CAT[k % len(CAT)]
        ax_r.plot(d["t"], d["rel"], color=col, lw=1.6, label=label)
        ax_s.plot(d["t"], d["speed_max"], color=col, lw=1.6, label=label)
        ax_k.plot(d["t"], d["ke"], color=col, lw=1.6, label=label)
        ax_d.plot(d["t"], d["dt"], color=col, lw=1.2, label=label)
    ax_r.set_yscale("log")
    ax_r.set_ylabel("|R_steady| / |R_ref|")
    ax_r.axhline(1e-6, color=MUTED, lw=0.8, ls="--")
    ax_s.set_ylabel("最大流速 [m/s]")
    ax_k.set_ylabel("運動エネルギー [J/m]")
    ax_d.set_ylabel("Δt [s]")
    ax_d.set_yscale("log")
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("物理時間の陰的非定常（後退 Euler、Δt 後退あり）", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_continuation(
        [
            ("内部ポート（T, 凍結なし）", "trama-T-cont-sser-*.log"),
            ("内部 + 凍結（A2, 即解凍）", "trama-A2-*.log"),
            ("内部 + 凍結（A3, patience 3）", "trama-A3-*.log"),
            ("内部 + 凍結 + ラインサーチ（A4）", "trama-A4-*.log"),
            ("内部 + 摩擦則 + 凍結（E2）", "trama-E2-*.log"),
            ("壁ポート（W, 凍結なし）", "trama-W-wall-cont-sser-*.log"),
            ("壁 + 凍結（B2）", "trama-B2-*.log"),
        ],
        FIGS / "f8_continuation_freeze.png",
    )
    fig_unsteady(
        [("Δx 1.5 mm, Δt ≤ 5 ms, J1 LU + ラインサーチ（G4）", "trama-G4-*.log")],
        FIGS / "f9_unsteady.png",
    )
    print(f"[plots2] saved {FIGS / 'f8_continuation_freeze.png'}, {FIGS / 'f9_unsteady.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
