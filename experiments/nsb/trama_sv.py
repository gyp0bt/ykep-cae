"""trama ケースの最初の線形系の特異値解析（条件数・弱い特異方向・J1 と真の作用素の食い違い）.

  - σ_max(J1+τ): svds（k=1）
  - σ_min(J1+τ) と右特異ベクトル: (J1+τ)^{-1} の最大特異値を PARDISO 分解（J と Jᵀ）の
    LinearOperator に svds をかけて求める
  - その弱い方向 v_min に沿った真の作用素（有限差分）と J1+τ の食い違い |A v − J1τ v| / |J1τ v|
  - 擬似時間対角 τ が有限差分 matvec に 2 重に入っていることの数値確認

使用例::

    python experiments/nsb/trama_sv.py --mass 0.15 --port interior 2>&1 | tee experiments/nsb/logs/trama-sv-$(date +%s).log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

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

HERE = Path(__file__).resolve().parent


def block_norms(v: np.ndarray, n: int) -> list[float]:
    return [float(np.linalg.norm(v[i * n : (i + 1) * n])) for i in range(3)]


def singular_extremes(J: sparse.csr_matrix, k: int = 3) -> dict:
    """σ_max と σ_min（k 本）および対応する右特異ベクトル."""
    t0 = time.perf_counter()
    s_max = float(spla.svds(J, k=1, return_singular_vectors=False)[0])
    t1 = time.perf_counter()
    lu = PardisoLU().factorize(J.tocsr())
    lut = PardisoLU().factorize(J.T.tocsr())
    op = spla.LinearOperator(J.shape, matvec=lu.solve, rmatvec=lut.solve, dtype=float)
    # J^{-1} の最大特異値 = 1/σ_min(J)。右特異ベクトル（u 側）は J の左特異ベクトル…
    # J^{-1} = V Σ^{-1} Uᵀ なので J^{-1} の左特異ベクトルが J の右特異ベクトル
    U, S, Vt = spla.svds(op, k=k, which="LM")
    lu.free()
    lut.free()
    order = np.argsort(-S)
    return {
        "sigma_max": s_max,
        "sigma_min": [float(1.0 / S[i]) for i in order],
        "v_min": [U[:, i] for i in order],  # J の右特異ベクトル（弱い方向）
        "t_svds_max": t1 - t0,
        "t_svds_min": time.perf_counter() - t1,
    }


def analyze(fs: FirstStep, label: str, out_dir: Path, k: int = 3) -> dict:
    n = fs.n
    J = fs.J1_tau
    res: dict = {"label": label, "n": n, "cfl": fs.cfl, "r_ref": fs.r_ref}
    ex = singular_extremes(J, k=k)
    res["sigma_max"] = ex["sigma_max"]
    res["sigma_min"] = ex["sigma_min"]
    res["kappa"] = ex["sigma_max"] / ex["sigma_min"][0]
    print(
        f"[sv:{label}] sigma_max={ex['sigma_max']:.4e} sigma_min={ex['sigma_min']} "
        f"kappa={res['kappa']:.4e} (t={ex['t_svds_max']:.1f}+{ex['t_svds_min']:.1f}s)",
        flush=True,
    )
    vs = []
    for i, v in enumerate(ex["v_min"]):
        Jv = J @ v
        Av = fs.fd_matvec(v)
        mis = float(np.linalg.norm(Av - Jv) / max(np.linalg.norm(Jv), 1e-300))
        vs.append(
            {
                "sigma": ex["sigma_min"][i],
                "blocks_v": block_norms(v, n),
                "blocks_Jv": block_norms(Jv, n),
                "blocks_Av_minus_Jv": block_norms(Av - Jv, n),
                "mismatch": mis,
                "p_mean_over_norm": float(
                    abs(v[2 * n :].mean()) * np.sqrt(n) / max(np.linalg.norm(v[2 * n :]), 1e-300)
                ),
            }
        )
        print(
            f"[sv:{label}]  v_min[{i}] sigma={ex['sigma_min'][i]:.3e} blocks(u,v,p)={np.round(vs[-1]['blocks_v'], 3).tolist()} "
            f"|Av-Jv|/|Jv|={mis:.3e} p-const share={vs[-1]['p_mean_over_norm']:.3f}",
            flush=True,
        )
    res["v_min"] = vs
    # J1τ^{-1} b 方向（E1 の再現）と τ 2 重カウントの確認
    b = fs.rhs
    lu = PardisoLU().factorize(J)
    d = lu.solve(b)
    lu.free()
    Ad = fs.fd_matvec(d)
    res["step"] = {
        "norm_d": float(np.linalg.norm(d)),
        "norm_x": float(np.linalg.norm(fs.x)),
        "blocks_d": block_norms(d, n),
        "true_resid_ratio": float(np.linalg.norm(b - Ad) / np.linalg.norm(b)),
        "blocks_b_minus_Ad": block_norms(b - Ad, n),
    }
    print(f"[sv:{label}] step: {res['step']}", flush=True)
    # τ の 2 重カウント: 流路セルの u だけを持つ v で (A_fd v − J1 v) / v ≈ 2τ か τ か
    u_idx = np.arange(n)[fs.disc.inp.thickness.ravel() > 1e-4]
    v = np.zeros(3 * n)
    v[u_idx] = 1.0
    diff = (fs.fd_matvec(v) - fs.J1 @ v)[u_idx] / fs.tau[u_idx]
    res["tau_multiplier_fd_minus_J1"] = {
        "median": float(np.median(diff)),
        "q10": float(np.quantile(diff, 0.1)),
        "q90": float(np.quantile(diff, 0.9)),
    }
    print(
        f"[sv:{label}] (A_fd v - J1 v)/tau on channel u: {res['tau_multiplier_fd_minus_J1']}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"trama_sv_{label}.npz",
        v_min=np.stack(ex["v_min"]),
        sigma_min=np.array(ex["sigma_min"]),
        d=d,
        b=b,
        Ad=Ad,
        x=fs.x,
        h=fs.disc.inp.thickness,
        tau=fs.tau,
    )
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mass", type=float, nargs="+", default=[0.0015, 0.15])
    ap.add_argument("--port", nargs="+", default=["interior"])
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--cfl", type=float, default=None, help="既定 cfl_init")
    ap.add_argument("--straight", type=float, default=None)
    ap.add_argument("--variant", default="orig")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args(argv)
    geo = (
        load_trama(DEFAULT_PATTERN, variant=a.variant)
        if a.straight is None
        else make_straight_geometry(a.straight)
    )
    results = []
    for port in a.port:
        for m in a.mass:
            inp = make_trama_input(geo, m, dx_mm=a.dx, settings=NSBSettings(), port=port)
            fs = first_step(inp, cfl=a.cfl)
            gtag = f"straight{a.straight:g}" if a.straight is not None else a.variant
            label = f"{gtag}-{port}-m{m:g}-dx{a.dx:g}-cfl{fs.cfl:g}"
            results.append(analyze(fs, label, a.out, k=a.k))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "-".join(a.port) + "-" + "-".join(f"{m:g}" for m in a.mass) + f"-dx{a.dx:g}"
    if a.straight is not None:
        tag = f"straight{a.straight:g}-" + tag
    (a.out / f"trama_sv_{tag}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"[sv] saved {a.out / f'trama_sv_{tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
