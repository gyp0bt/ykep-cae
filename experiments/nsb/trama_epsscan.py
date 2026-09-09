"""弱い特異方向 v_min（閉塞セルの圧力モード）に沿った残差の差分商を ε で掃引する.

差分商 (R(x+εv) − R(x))/ε が ε→0 で一定に収束すれば本物の導関数、1/ε で伸びるなら
丸めノイズ、途中で飛べば折れ点（風上切替・リミター）。J1 v と比べる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.nsb.trama_case import DEFAULT_PATTERN, load_trama, make_trama_input  # noqa: E402
from experiments.nsb.trama_lin import first_step  # noqa: E402
from nsb import NSBSettings  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    mass = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    geo = load_trama(DEFAULT_PATTERN)
    inp = make_trama_input(geo, mass, settings=NSBSettings())
    fs = first_step(inp)
    n = fs.n
    sv = np.load(HERE / "results" / f"trama_sv_orig-interior-m{mass:g}-dx1.5-cfl0.25.npz")
    chan = (fs.disc.inp.thickness > 1e-4).ravel()
    rng = np.random.default_rng(1)
    dirs = {"v_min[0] (p, blocked)": sv["v_min"][0]}
    for name, mask in [("random p in blocked", ~chan), ("random p in channel", chan)]:
        v = np.zeros(3 * n)
        v[2 * n :][mask] = rng.standard_normal(mask.sum())
        dirs[name] = v / np.linalg.norm(v)
    v = np.zeros(3 * n)
    v[:n][chan] = rng.standard_normal(chan.sum())
    dirs["random u in channel"] = v / np.linalg.norm(v)
    r0 = fs.resid_tau(fs.x)
    x_norm = float(np.linalg.norm(fs.x))
    eps_solver = float(np.sqrt(np.finfo(float).eps)) * np.sqrt(1.0 + x_norm)
    print(f"[eps] mass={mass} |x|={x_norm:.3e} solver eps (|v|=1) = {eps_solver:.3e}")
    for name, v in dirs.items():
        Jv = fs.J1_tau @ v
        print(
            f"[eps] --- {name}: |J1tau v|={np.linalg.norm(Jv):.3e} blocks={[float(np.linalg.norm(Jv[i * n : (i + 1) * n])) for i in range(3)]}"
        )
        prev = None
        for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, eps_solver, 1e-6, 1e-7, 1e-8, 1e-9]:
            q = (fs.resid_tau(fs.x + e * v) - r0) / e
            blk = [float(np.linalg.norm(q[i * n : (i + 1) * n])) for i in range(3)]
            dq = (
                ""
                if prev is None
                else f" |Δq/q|={np.linalg.norm(q - prev) / max(np.linalg.norm(q), 1e-300):.2e}"
            )
            print(
                f"[eps]   eps={e:.2e}: |q|={np.linalg.norm(q):.3e} blocks(u,v,p)={np.array(blk).round(12).tolist()} |q-Jv|/|Jv|={np.linalg.norm(q - Jv) / np.linalg.norm(Jv):.2e}{dq}"
            )
            prev = q


if __name__ == "__main__":
    main()
