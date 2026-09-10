"""定常解の線形安定性: 収束した場のヤコビアンの固有値を見る.

蛇行流路は流量を上げると定常計算が収束しなくなる（§11）。原因を「定常解が無い」と書いたが、
正確には **定常解（R(x) = 0 の根）は在るが線形不安定になる**（超臨界 Hopf）ことがある。
Newton は根を探すので不安定な根にも乗れるが、SIMPLE や時間前進は乗れない。その区別を数で出す。

非定常方程式は M dx/dt + R(x) = 0（M = diag(ρV, ρV, 0)、圧力に時間微分は無い）。
定常解まわりの摂動は M dδ/dt = −J δ なので、束 (−J, M) の固有値 λ の実部が正なら不安定。
M が特異（圧力行がゼロ）なので shift-invert（σ 近傍）で解く。固体セルは未知数から外す。

**使う前に確かめること 2 つ**（どちらも満たさないと出た数に意味が無い）:

1. 与えた場が**本当に根か**。リミター凍結（`--freeze`）付きで収束させた場は「凍結した問題」の根で、
   凍結を解いた真の残差は 1e-4〜1e-3 残る（§10.2）。このスクリプトは |R(x)|/|R_ref| を表示して
   `--root-tol` を超えたら警告する。根でない点のヤコビアンの固有値は安定性を語らない。
2. **shift の置き場所**。`sigma=0` の shift-invert が返すのは |λ| の小さい方から k 個で、
   実部が正でも |λ| が大きい固有値は拾えない。不安定を探すなら虚軸沿いに σ を振る
   （`--sigma` / `--sigma-imag`）。想定周期 T の Hopf なら σ ≈ 2πi/T の近く。

    python experiments/nsb/trama_stability.py --npz experiments/nsb/results/trama_X_fields.npz \
        --mass 0.025 --port wall --h-solid 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import DEFAULT_PATTERN, load_trama, make_trama_input  # noqa: E402

from nsb.assembly import BrinkmanDiscretization  # noqa: E402
from nsb.fdjac import colored_fd_jacobian  # noqa: E402
from nsb.linalg import pardiso_solve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="収束した場の *_fields.npz")
    ap.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--port", default="wall", choices=["interior", "wall"])
    ap.add_argument("--h-solid", type=float, default=1.0e-4)
    ap.add_argument("--k", type=int, default=12, help="求める固有値の数")
    ap.add_argument("--sigma", type=float, default=0.0, help="shift の実部")
    ap.add_argument("--sigma-imag", type=float, default=0.0, help="shift の虚部 [rad/s]")
    ap.add_argument(
        "--root-tol",
        type=float,
        default=1.0e-5,
        help="|R(x)|/|R_ref| がこれを超えたら『根ではない』と警告する",
    )
    ap.add_argument("--jacobian", default="fd", choices=["fd", "fou"])
    ap.add_argument("--out-json", default="experiments/nsb/results/trama_stability.json")
    a = ap.parse_args()

    geo = load_trama(a.pattern)
    inp = make_trama_input(geo, a.mass, dx_mm=a.dx, port=a.port, h_solid=a.h_solid)
    disc = BrinkmanDiscretization(inp.to_flow_input())
    z = np.load(a.npz)
    x = disc.mask_state(np.concatenate([z["u"].ravel(), z["v"].ravel(), z["p"].ravel()]))
    s = inp.settings
    r_norm = float(np.linalg.norm(disc.residual_fast(x, s.scheme, s.venkat_k)))
    # 参照残差: Stokes–Brinkman 場での完全 NS 定常残差（`nsb.solver.stokes_reference` と同じ基準）
    x0 = np.zeros(3 * disc.n)
    st0 = disc.compute_state(x0, s.scheme, s.venkat_k)
    j0 = disc.jacobian_first_order(st0, convection=False, x=x0).tocsr()
    x_st = pardiso_solve(j0, -disc.residual_from_state(x0, st0, convection=False))
    r_ref = max(float(np.linalg.norm(disc.residual_fast(x_st, s.scheme, s.venkat_k))), 1e-300)
    rel = r_norm / r_ref
    print(f"[stab] |R(x)|/|R_ref| = {rel:.3e}  (active {disc.n_active}/{disc.n})")
    is_root = rel <= a.root_tol
    if not is_root:
        print(
            f"[stab] 警告: この場は根ではない（{rel:.2e} > {a.root_tol:g}）。"
            "リミター凍結付きで収束させた場は『凍結した問題』の根で、真の残差はこの通り残る。"
            "以下の固有値は定常解の安定性を語らない。"
        )

    if a.jacobian == "fd":
        j = colored_fd_jacobian(
            lambda xx: disc.residual_fast(xx, s.scheme, s.venkat_k),
            x,
            disc.nx,
            disc.ny,
            radius=s.fd_jacobian_radius,
        )
        j = disc.apply_solid_rows(j)
    else:
        j = disc.jacobian_first_order(disc.compute_state(x, s.scheme, s.venkat_k), x=x)
    n = disc.n
    live = disc.live3 if disc.has_solid else np.ones(3 * n, dtype=bool)
    j = sparse.csr_matrix(j)[live, :][:, live]
    m_diag = np.concatenate(
        [np.full(n, inp.rho * disc.vol), np.full(n, inp.rho * disc.vol), np.zeros(n)]
    )[live]
    m = sparse.diags(m_diag)

    # M dδ/dt = −J δ → 束 (−J, M) の固有値。M が特異なので shift-invert
    sigma = complex(a.sigma, a.sigma_imag)
    vals = eigs(
        -j.tocsc(),
        k=a.k,
        M=m.tocsc(),
        sigma=sigma,
        which="LM",
        return_eigenvectors=False,
        tol=1e-8,
    )
    vals = np.asarray(vals)
    order = np.argsort(-vals.real)
    vals = vals[order]
    out = {
        "mass_flow": a.mass,
        "residual_norm": r_norm,
        "residual_rel": rel,
        "is_root": bool(is_root),
        "sigma": [a.sigma, a.sigma_imag],
        "n_active": int(disc.n_active),
        "eigenvalues": [[float(v.real), float(v.imag)] for v in vals],
        "max_real": float(vals.real.max()),
        "unstable": bool(vals.real.max() > 0.0),
    }
    print("[stab] 実部の大きい順（λ [1/s]、正なら不安定）:")
    for v in vals:
        f = abs(v.imag) / (2 * np.pi)
        print(
            f"   {v.real:+.4f} {v.imag:+.4f}j   周期 {1 / f:.4f} s"
            if f > 0
            else f"   {v.real:+.4f}"
        )
    verdict = "不安定" if out["unstable"] else f"σ={sigma} の近傍には不安定な固有値なし"
    print(f"[stab] max Re(λ) = {out['max_real']:+.4f} 1/s → {verdict}")
    if not is_root:
        print("[stab] （ただし上の警告のとおり、この場は根ではないので判定にはならない）")
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
