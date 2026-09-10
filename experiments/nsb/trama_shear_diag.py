"""剥離開始位置のずれ（§14.6b の +77〜84 mm）を「種の振幅」と「空間増幅率」に分解する.

§14.6b は「nsb の方が下流で崩れる」という**位置**だけを測った。位置は

    x_onset(A_th) = x_bend + ln(A_th / A_0) / σ

で決まるので、ずれの原因は種の振幅 A_0（曲がりを出た時点の擾乱）か、空間増幅率 σ
（剪断層の不安定性 − 数値減衰）のどちらか。この 2 つを分けて測る。

測るもの:

1. **助走脚の横断プロファイルと内側剪断層の運動量厚み θ**（変動 RMS ≈ 0 の決定論区間）。
   σ は剪断層の厚みでほぼ決まる（σθ ≈ 一定）ので、θ が揃っていれば「元の場が厚い」説は消える。
2. **上側水平区間の断面最大 RMS の指数回帰** → σ と、曲がり位置に外挿した A_0。
   閾値 5/10/15/20% u_mean を横切る位置も出す（§14.6b と同じ量）。
3. nsb 側の数値減衰の出どころ 2 つを場として:
   - Venkatakrishnan リミター ψ（0 で 1 次風上に退化 ＝ 数値粘性 ½ρ|u|Δ(1−ψ)）
   - Rhie–Chow 係数 d_f = V/a_P に時間微分の対角 ρV/Δt を入れた場合に縮む倍率

使用例::

    # 本番ジオメトリ（400×233）
    python experiments/nsb/trama_shear_diag.py \
        --nsb 本番=experiments/nsb/results/trama_LEAD-carve-m01-B_fields.npz \
        --of /tmp/.../of-trama/walls-lead2
    # 縮小ジオメトリの掃引（274×162）
    python experiments/nsb/trama_shear_diag.py \
        --pattern experiments/nsb/patterns/lead_stub.json --lx 411 --ly 243 --scale 4.928149 \
        --nsb base=...npz --nsb k100=...npz --of /tmp/.../of-stub --tag stub

[README](../../README.md)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "extruder"))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import load_trama, make_trama_input  # noqa: E402
from trama_of_case import DEFAULT_PATTERN  # noqa: E402
from trama_of_compare import to_grid  # noqa: E402
from trama_solid_compare import load_of_mean  # noqa: E402

FRACS = (0.05, 0.10, 0.15, 0.20)


def momentum_thickness(q: np.ndarray, s: np.ndarray) -> float:
    """混合層の運動量厚み θ = ∫ f(1−f) ds（f は両端で 0→1 に規格化した速度）."""
    if q.size < 3:
        return float("nan")
    hi, lo = float(q[0]), float(q[-1])
    if not np.isfinite(hi) or not np.isfinite(lo) or hi - lo <= 0:
        return float("nan")
    f = np.clip((q - lo) / (hi - lo), 0.0, 1.0)
    return float(np.trapezoid(f * (1.0 - f), s))


def leg_profiles(
    speed: np.ndarray,
    keep: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    stations: list[float],
    x_window: tuple[float, float],
) -> list[dict]:
    """縦の助走脚（x が流路幅方向、y が流れ方向）の横断プロファイルと内側剪断層の厚み.

    x_window は助走脚だけを切り出す範囲 [mm]。水平線は他の脚とも交わるので、これが無いと
    プロファイルが飛び地になって厚みの積分が意味を失う。
    """
    out = []
    inwin = (x_mm >= x_window[0]) & (x_mm <= x_window[1])
    for y0 in stations:
        j = int(np.argmin(np.abs(y_mm - y0)))
        col = speed[:, j].copy()
        col[~(keep[:, j] & inwin)] = np.nan
        ok = np.isfinite(col)
        if ok.sum() < 8:
            continue
        idx = np.flatnonzero(ok)
        xs, q = x_mm[idx], col[idx]
        mid = len(q) // 2
        il = int(np.argmax(q[:mid]))  # 左の噴流ピーク
        ir = mid + int(np.argmax(q[mid:]))  # 右の噴流ピーク
        ic = il + int(np.argmin(q[il : ir + 1]))  # 中央の後流極小
        th = []
        for peak in (il, ir):
            lo, hi = min(peak, ic), max(peak, ic)
            seg, ss = q[lo : hi + 1], xs[lo : hi + 1]
            if peak > ic:  # 「ピーク → 極小」の向きに揃える
                seg, ss = seg[::-1], ss[::-1]
            th.append(momentum_thickness(seg, np.abs(ss - ss[0])))
        out.append(
            {
                "y_mm": float(y_mm[j]),
                "x_mm": xs.tolist(),
                "q": q.tolist(),
                "jet": 0.5 * (float(q[il]) + float(q[ir])),
                "wake_min": float(q[ic]),
                "theta_mm": float(np.nanmean(th)),
            }
        )
    return out


def envelope(rms: np.ndarray, keep: np.ndarray, xs: np.ndarray, band: np.ndarray) -> np.ndarray:
    """上側水平区間の各 x での断面最大 RMS."""
    return np.nanmax(np.where(keep, rms, np.nan)[np.ix_(xs, band)], axis=1)


def growth_fit(x: np.ndarray, r: np.ndarray, x0: float, lo: float, hi: float) -> dict:
    """RMS(x) の指数増幅域を ln 線形回帰して σ [1/mm] と x0 に外挿した種 A_0 を返す.

    飽和後は包絡が寝る（下がることもある）ので、**最初の極大より上流だけ**を使う。
    """
    top = int(np.nanargmax(np.where(np.isfinite(r), r, -np.inf)))
    m = np.isfinite(r) & (r > lo) & (r < hi)
    m[top + 1 :] = False
    if m.sum() < 5:
        return {"sigma_per_mm": float("nan"), "amp_at_bend": float("nan"), "n": int(m.sum())}
    c = np.polyfit(x[m], np.log(r[m]), 1)
    return {
        "sigma_per_mm": float(c[0]),
        "growth_length_mm": float(1.0 / c[0]),
        "amp_at_bend": float(np.exp(np.polyval(c, x0))),
        "x_from_mm": float(x[m][0]),
        "x_to_mm": float(x[m][-1]),
        "n": int(m.sum()),
    }


def crossings(x: np.ndarray, r: np.ndarray, u_mean: float) -> dict[str, float]:
    """RMS が閾値 frac·u_mean を最初に超える x [mm]（超えなければ nan）."""
    out = {}
    for f in FRACS:
        idx = np.flatnonzero(np.isfinite(r) & (r >= f * u_mean))
        out[f"{f:.2f}"] = float(x[idx[0]]) if idx.size else float("nan")
    return out


def nsb_internals(
    npz, geo, mass: float, dx_mm: float, dt: float, reg: np.ndarray, venkat_k: float = 5.0
) -> tuple[dict, np.ndarray, np.ndarray]:
    """リミター ψ・リミター由来の数値粘性・Rhie–Chow 係数の過大倍率（保存場から再構成）."""
    from nsb.assembly import BrinkmanDiscretization

    inp = make_trama_input(geo, mass, dx_mm=dx_mm, port="carve", h_solid=1.0e-4)
    disc = BrinkmanDiscretization(inp.to_flow_input())
    x = disc.mask_state(np.concatenate([npz["u"].ravel(), npz["v"].ravel(), npz["p"].ravel()]))
    psi_u, psi_v = disc.limiter(x, venkat_k)
    st = disc.compute_state(x, inp.settings.scheme, venkat_k)
    psi = np.minimum(psi_u, psi_v)
    rc_ratio = (st.a_p + inp.rho * disc.vol / dt) / st.a_p
    nu_num = 0.5 * np.hypot(npz["u"], npz["v"]) * (dx_mm * 1e-3) * (1.0 - psi) / (inp.mu / inp.rho)
    stats = {
        "psi_median": float(np.median(psi[reg])),
        "psi_p10": float(np.percentile(psi[reg], 10)),
        "frac_psi_lt_half": float((psi[reg] < 0.5).mean()),
        "nu_num_over_nu_median": float(np.median(nu_num[reg])),
        "nu_num_over_nu_p90": float(np.percentile(nu_num[reg], 90)),
        "rc_ratio_median": float(np.median(rc_ratio[reg])),
        "rc_ratio_p90": float(np.percentile(rc_ratio[reg], 90)),
    }
    return stats, psi, rc_ratio


def main() -> None:  # noqa: PLR0915, PLR0912
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nsb",
        action="append",
        required=True,
        help="ラベル=fields.npz[@venkat_k]（複数可。@K はリミター統計をその走行の K で出すため）",
    )
    ap.add_argument("--of", default=None, help="OpenFOAM ケース（省略可）")
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--lx", type=float, default=600.0)
    ap.add_argument("--ly", type=float, default=350.0)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument(
        "--x-from",
        type=float,
        default=18.0,
        help="増幅の回帰・閾値判定を始める、曲がりからの距離 [mm]（曲がりの幾何そのものを避ける）",
    )
    ap.add_argument("--tag", default="lead")
    ap.add_argument("--figs", default="experiments/nsb/results/trama_figs")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    geo = load_trama(a.pattern, lx_mm=a.lx, ly_mm=a.ly, scale_mm=a.scale, variant="lead")
    poly = geo.polyline * 1e3
    x_bend, y_leg = float(poly[1][0]), float(poly[1][1])
    # 上側水平区間の終わり: 曲がりから y が変わらない間だけ辿る（本番は蛇行の続きがあるので
    # polyline の最後の点ではない）
    k = 1
    while k + 1 < len(poly) and abs(poly[k + 1][1] - y_leg) < 1.0e-9:
        k += 1
    x_leg_end = float(poly[k][0])
    w2 = geo.width * 1e3 / 2
    h_ch = 3.8e-3
    u_mean = a.mass / (1000.0 * geo.width * h_ch)

    cases, kcase = {}, {}
    for spec in a.nsb:
        lab, _, path = spec.partition("=")
        path, _, kstr = path.partition("@")
        lab = lab or Path(path).stem
        cases[lab] = np.load(path)
        kcase[lab] = float(kstr) if kstr else 5.0
    first = next(iter(cases.values()))
    nx, ny = first["u"].shape
    x_mm = (np.arange(nx) + 0.5) * a.lx / nx
    y_mm = (np.arange(ny) + 0.5) * a.ly / ny
    keep = first["h"] > 1.0e-4

    of = None
    if a.of:
        f = load_of_mean(Path(a.of))
        of = to_grid(f, nx, ny, geo.lx, geo.ly)
        c = f["C"]
        i = np.rint(c[:, 0] / (geo.lx / nx) - 0.5).astype(int)
        j = np.rint(c[:, 1] / (geo.ly / ny) - 0.5).astype(int)
        r = np.full((nx, ny), np.nan)
        r[i, j] = f["R"][:, 0] + f["R"][:, 3]
        of["rms"] = np.sqrt(np.clip(r, 0.0, None))
        keep = keep & of["mask"]

    print(
        f"[diag] {a.tag}: 格子 {nx}x{ny} dx={a.dx} mm  曲がり x={x_bend:.1f} mm  "
        f"上側水平区間 y={y_leg:.1f} mm  水平区間の終わり x={x_leg_end:.1f} mm  u_mean={u_mean:.4f} m/s"
    )

    # ---- 1. 助走脚の剪断層厚み ----
    y0, y1 = float(poly[0][1]), y_leg
    stations = list(np.linspace(y0 + 20.0, y1 - 15.0, 7))
    win = (x_bend - w2 - 1.0, x_bend + w2 + 1.0)
    prof = {
        lab: leg_profiles(np.hypot(d["mean_u"], d["mean_v"]), keep, x_mm, y_mm, stations, win)
        for lab, d in cases.items()
    }
    if of is not None:
        prof["OpenFOAM"] = leg_profiles(np.hypot(of["u"], of["v"]), keep, x_mm, y_mm, stations, win)
    print(
        "\n[diag] 助走脚 内側剪断層の運動量厚み θ [mm]（噴流ピーク → 後流極小。Δx =", a.dx, "mm）"
    )
    print("  " + "y [mm]".rjust(8) + "".join(f"{lab:>12}" for lab in prof))
    for k in range(len(next(iter(prof.values())))):
        row = "".join(f"{p[k]['theta_mm']:12.3f}" for p in prof.values())
        print(f"  {next(iter(prof.values()))[k]['y_mm']:8.1f}{row}")

    # ---- 2. 上側水平区間の増幅 ----
    band = (y_mm > y_leg - w2) & (y_mm < y_leg + w2)
    xs = (x_mm > x_bend + a.x_from) & (x_mm < min(x_bend + 200.0, x_leg_end - 30.0))
    xg = x_mm[xs]
    env, fits, cross = {}, {}, {}
    for lab, d in cases.items():
        env[lab] = envelope(np.hypot(d["rms_u"], d["rms_v"]), keep, xs, band)
    if of is not None:
        env["OpenFOAM"] = envelope(of["rms"], keep, xs, band)
    for lab, e in env.items():
        fits[lab] = growth_fit(xg, e, x_bend, 0.003 * u_mean, 0.35 * u_mean)
        cross[lab] = crossings(xg, e, u_mean)

    print("\n[diag] 上側水平区間 断面最大 RMS = A_0 e^{σ(x−x_bend)}")
    hdr = f"  {'':>12} {'σ [1/mm]':>10} {'e 倍長':>8} {'A_0':>10} {'A_0/u_mean':>11}"
    hdr += "".join(f"{f'x@{f:.0%}':>9}" for f in FRACS) + f"{'5→20%':>8}{'RMS 最大':>9}"
    print(hdr)
    for lab, f in fits.items():
        row = (
            f"  {lab:>12} {f['sigma_per_mm']:10.4f} {f.get('growth_length_mm', np.nan):8.1f} "
            f"{f['amp_at_bend']:10.2e} {f['amp_at_bend'] / u_mean:11.4f}"
        )
        row += "".join(f"{cross[lab][f'{fr:.2f}'] - x_bend:9.1f}" for fr in FRACS)
        span = cross[lab]["0.20"] - cross[lab]["0.05"]
        row += f"{span:8.1f}{np.nanmax(env[lab]) / u_mean:9.3f}"
        print(row)
    print(
        "  （x@ は曲がりからの距離 [mm]、5→20% はその 2 閾値の間隔 ＝ 育つのに要る距離、"
        "RMS 最大は u_mean 比。nan は区間内で閾値に達しない）"
    )

    if of is not None and len(cases) >= 1:
        ref = fits["OpenFOAM"]
        print("\n[diag] OpenFOAM を基準にしたずれの内訳（10% 閾値）")
        ath = 0.10 * u_mean
        for lab, f in fits.items():
            if lab == "OpenFOAM" or not np.isfinite(f["sigma_per_mm"]):
                continue
            d_seed = np.log(ref["amp_at_bend"] / f["amp_at_bend"]) / f["sigma_per_mm"]
            d_sig = np.log(ath / ref["amp_at_bend"]) * (
                1.0 / f["sigma_per_mm"] - 1.0 / ref["sigma_per_mm"]
            )
            print(
                f"  {lab:>12}: 種 {d_seed:+7.1f} mm + 増幅率 {d_sig:+7.1f} mm "
                f"= {d_seed + d_sig:+7.1f} mm （σ 比 {ref['sigma_per_mm'] / f['sigma_per_mm']:.2f}、"
                f"A_0 比 {ref['amp_at_bend'] / f['amp_at_bend']:.2f}）"
            )

    # ---- 3. nsb の内部量 ----
    reg = np.zeros((nx, ny), dtype=bool)
    reg[np.ix_((x_mm > x_bend) & (x_mm < min(x_bend + 200.0, x_leg_end - 30.0)), band)] = True
    reg &= keep
    internals, psi0, rc0 = {}, None, None
    print("\n[diag] 剪断層領域（曲がり〜+200 mm）の nsb 内部量")
    print(
        f"  {'':>12} {'ψ 中央値':>9} {'ψ 下位10%':>10} {'ψ<0.5':>7} "
        f"{'ν_num/ν 中央':>12} {'同 上位10%':>11} {'RC 過大 中央':>12} {'同 上位10%':>11}"
    )
    for lab, d in cases.items():
        st, psi, rc = nsb_internals(d, geo, a.mass, a.dx, a.dt, reg, kcase[lab])
        internals[lab] = st
        if psi0 is None:
            psi0, rc0 = psi, rc
        print(
            f"  {lab:>12} {st['psi_median']:9.3f} {st['psi_p10']:10.3f} "
            f"{st['frac_psi_lt_half']:7.1%} {st['nu_num_over_nu_median']:12.1f} "
            f"{st['nu_num_over_nu_p90']:11.1f} {st['rc_ratio_median']:12.2f} "
            f"{st['rc_ratio_p90']:11.2f}"
        )

    out_json = a.out_json or f"experiments/nsb/results/trama_shear_diag_{a.tag}.json"
    Path(out_json).write_text(
        json.dumps(
            {
                "u_mean": u_mean,
                "x_bend_mm": x_bend,
                "growth": fits,
                "crossings": cross,
                "internals": internals,
                "theta": {
                    k: [{"y_mm": r["y_mm"], "theta_mm": r["theta_mm"]} for r in v]
                    for k, v in prof.items()
                },
                "x_mm": xg.tolist(),
                "envelope": {k: np.where(np.isfinite(v), v, None).tolist() for k, v in env.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n[diag] {out_json}")
    _figures(a, prof, xg, env, fits, x_mm, y_mm, psi0, rc0, keep, x_bend, y_leg, w2, u_mean)


def _figures(a, prof, xg, env, fits, x_mm, y_mm, psi, rc, keep, x_bend, y_leg, w2, u_mean) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = {lab: ("k" if lab == "OpenFOAM" else cols[i % len(cols)]) for i, lab in enumerate(prof)}

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for lab, p in prof.items():
        ax[0].plot(p[0]["x_mm"], p[0]["q"], color=cmap[lab], label=f"{lab} y={p[0]['y_mm']:.0f}")
        ax[0].plot(p[-1]["x_mm"], p[-1]["q"], color=cmap[lab], ls="--")
        ax[1].plot(
            [r["y_mm"] for r in p], [r["theta_mm"] for r in p], "o-", color=cmap[lab], label=lab
        )
    ax[0].set(
        xlabel="x [mm]（流路幅方向）",
        ylabel="平均速さ [m/s]",
        title="助走脚の横断プロファイル（実線 上流 / 破線 下流）",
    )
    ax[0].legend(fontsize=8)
    ax[1].axhline(a.dx, color="0.5", ls=":", label=f"格子 Δx = {a.dx} mm")
    ax[1].set(
        xlabel="y [mm]（助走脚に沿って）", ylabel="運動量厚み θ [mm]", title="内側剪断層の厚み"
    )
    ax[1].legend(fontsize=8)
    for lab, e in env.items():
        ax[2].semilogy(
            xg - x_bend,
            e,
            color=cmap.get(lab, "0.4"),
            label=f"{lab} σ={fits[lab]['sigma_per_mm']:.3f}/mm",
        )
    for fr in (0.05, 0.10, 0.15):
        ax[2].axhline(fr * u_mean, color="0.7", lw=0.7)
    ax[2].set(
        xlabel="曲がりからの距離 [mm]",
        ylabel="断面最大 変動 RMS [m/s]",
        title="上側水平区間の指数増幅",
    )
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"f23_{a.tag}_shear.png", dpi=130)

    if psi is None:
        return
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.4))
    sel = (x_mm > x_bend - 30) & (x_mm < x_bend + 230)
    selj = (y_mm > y_leg - w2 - 8) & (y_mm < y_leg + w2 + 8)
    ext = [x_mm[sel][0], x_mm[sel][-1], y_mm[selj][0], y_mm[selj][-1]]
    for k, (arr, ttl, vmin, vmax, cm) in enumerate(
        (
            (psi, "Venkatakrishnan リミター ψ（0 = 1 次風上に退化）", 0.0, 1.0, "viridis"),
            (rc, "Rhie–Chow 係数の過大倍率 (a_P+ρV/Δt)/a_P", 1.0, 10.0, "magma"),
        )
    ):
        z = np.where(keep, arr, np.nan)[np.ix_(sel, selj)]
        im = ax[k].imshow(
            z.T, origin="lower", extent=ext, aspect="auto", vmin=vmin, vmax=vmax, cmap=cm
        )
        ax[k].set(title=ttl, ylabel="y [mm]")
        fig.colorbar(im, ax=ax[k])
    ax[1].set_xlabel("x [mm]")
    fig.tight_layout()
    fig.savefig(figs / f"f24_{a.tag}_limiter_rc.png", dpi=130)
    print(f"[diag] {figs}/f23_{a.tag}_shear.png, f24_{a.tag}_limiter_rc.png")


if __name__ == "__main__":
    main()
