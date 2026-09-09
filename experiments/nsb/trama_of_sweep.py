"""流量掃引の結果をまとめる: 定常解が存在しなくなる流量を挟む.

深さ平均の運動量式で、横向きの渦を隙間の抗力 12μ/h² が殺せるかどうかは
1 つの無次元数で決まる:

    N = ρ u h² / (12 μ w) = L_drag / w,      L_drag = (Re_h / 12) · h

L_drag は「渦の速度差が 1/e に減る移動距離」。N はそれを流路幅で測ったもの。
N ≫ 1 なら渦は次のターンまで生き延び、場は時間的に振動する（＝定常解が無い）。

    python experiments/nsb/trama_of_sweep.py --work /tmp/of-trama --figs .../trama_figs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import load_trama  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402


def drag_number(mass_flow: float, *, w: float, h: float = 3.8e-3, mu: float = 3.0e-3) -> float:
    """N = ṁ h / (12 μ w²)（＝ ρ u h²/(12 μ w)、u = ṁ/(ρ w h)）."""
    return mass_flow * h / (12.0 * mu * w**2)


def collect(work: Path, geo) -> list[dict]:
    """各ケースの result.json / case.json から代表数を組む.

    case.json には ν, ρ, d_channel = 12/h² が入っているので、h と μ は逆算できる。
    u = ṁ/(ρ w h) から Re_h = ρ u h/μ、Re_w = ρ u w/μ、N = ρ u h²/(12 μ w)。
    """
    rows = []
    for case in sorted(work.iterdir()):
        rj, cj = case / "result.json", case / "case.json"
        if not (rj.exists() and cj.exists()):
            continue
        r, c = json.loads(rj.read_text()), json.loads(cj.read_text())
        rho, nu = c["rho"], c["nu"]
        mu = rho * nu
        h = (12.0 / c["d_channel"]) ** 0.5
        mass = c["flow_rate"] * rho
        u = mass / (rho * geo.width * h)
        res_p = r["final_residual"].get("p")
        rows.append(
            {
                "case": case.name,
                "variant": c["variant"],
                "mass_flow": mass,
                "h": h,
                "mu": mu,
                "u": u,
                "Re_h": rho * u * h / mu,
                "Re_w": rho * u * geo.width / mu,
                "N": rho * u * h**2 / (12.0 * mu * geo.width),
                "iterations": r["iterations"],
                "converged": bool(r["converged"]),
                # 判定 1e-6 を切らなくても 1e-5 未満なら「定常解に着いている」と読む
                # （階段状の壁の折れ点で 2〜3e-6 の小さな床が残るケースがある）
                "steady": bool(r["converged"]) or (res_p is not None and res_p < 1.0e-5),
                "res_p": res_p,
                "res_p_min": r["min_residual"].get("p"),
            }
        )
    rows.sort(key=lambda d: (d["variant"], d["N"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_of_sweep.json")
    a = ap.parse_args()

    geo = load_trama(a.pattern)
    rows = collect(Path(a.work), geo)
    if not rows:
        raise SystemExit("結果が無い")
    hdr = f"{'case':20s} {'mass':>7s} {'h[mm]':>6s} {'mu':>8s} {'Re_h':>7s} {'Re_w':>8s} {'N':>7s} {'it':>6s} {'収束':>5s} {'res p':>9s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['case']:20s} {r['mass_flow']:7.4g} {r['h'] * 1e3:6.3g} {r['mu']:8.3g} "
            f"{r['Re_h']:7.0f} {r['Re_w']:8.0f} {r['N']:7.3f} {r['iterations']:6d} "
            f"{str(r['converged']):>5s} {(r['res_p'] if r['res_p'] is not None else float('nan')):9.2e}"
        )
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    ww = [r for r in rows if r["variant"] == "walls" and "nolim" not in r["case"]]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xs = [r["N"] for r in ww]
    ys = [r["res_p"] for r in ww]
    ax.loglog(xs, ys, "-", color="#1F6FB4", lw=1.2, zorder=1)
    for r in ww:
        ok = r["steady"]
        ax.loglog(
            [r["N"]],
            [r["res_p"]],
            "o" if ok else "s",
            ms=9,
            mfc="#1F6FB4" if ok else "none",
            mec="#1F6FB4",
            mew=1.6,
            zorder=3,
        )
        ax.annotate(
            f"{r['mass_flow']:g} kg/s\n{r['iterations']} 反復",
            (r["N"], r["res_p"]),
            textcoords="offset points",
            xytext=(8, -18 if ok else 6),
            fontsize=8,
        )
    ax.axhline(1e-6, color="0.55", ls=":", lw=1)
    ax.axvspan(1.33, 2.22, color="#B24714", alpha=0.14, zorder=0)
    ax.text(
        1.5,
        max(ys) * 0.35,
        "定常解が\n消える境目",
        fontsize=9,
        color="#B24714",
        ha="center",
        va="top",
    )
    ax.set_xlabel("N = L_drag / w　（横渦が抗力で消える移動距離 ÷ 流路幅）")
    ax.set_ylabel("最終反復の p 初期残差")
    ax.set_title("定常解が存在しなくなる流量（OpenFOAM simpleFoam、walls 変種）")
    ax.grid(alpha=0.3, which="both")
    ax.plot([], [], "o", color="#1F6FB4", label="収束（残差が判定を切る）")
    ax.plot([], [], "s", mfc="none", mec="#1F6FB4", mew=1.6, label="床に張り付く（定常解が無い）")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = Path(a.figs) / "f12_of_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # --- 2 パラメータの地図: 面内慣性 Re_w × 抗力 N -------------------
    pts = [r for r in rows if r["variant"] == "walls" and "nolim" not in r["case"]]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for r in pts:
        ok = r["steady"]
        ax.loglog(
            [r["N"]],
            [r["Re_w"]],
            "o" if ok else "s",
            ms=11,
            mfc="#1F6FB4" if ok else "none",
            mec="#B24714" if not ok else "#1F6FB4",
            mew=2.0,
        )
        ax.annotate(
            f"{r['mass_flow']:g} kg/s, h={r['h'] * 1e3:.3g} mm",
            (r["N"], r["Re_w"]),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=8,
        )
    ax.axvline(2.0, color="#B24714", ls="--", lw=1.2)
    ax.axhline(1000.0, color="#555", ls="--", lw=1.2)
    ax.text(
        2.15, ax.get_ylim()[0] * 1.4, "N ≈ 2\n抗力が渦を殺せなくなる", fontsize=9, color="#B24714"
    )
    ax.text(
        ax.get_xlim()[0] * 1.3, 1150, "Re_w ≈ 1000（面内の剥離が立つ）", fontsize=9, color="#555"
    )
    ax.set_xlabel("N = L_drag / w（隙間の抗力の弱さ）")
    ax.set_ylabel("Re_w = ρ u w / μ（面内の慣性）")
    ax.set_title("定常解があるのはどちらか一方が足りないとき（●収束 / □床に張り付く）")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out2 = Path(a.figs) / "f13_of_map.png"
    fig.savefig(out2, dpi=110)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
