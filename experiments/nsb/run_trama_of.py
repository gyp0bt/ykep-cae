"""trama の OpenFOAM ケースを生成 → メッシュ → simpleFoam まで一気に回す.

OpenFOAM は Docker（`~/work/1a/a02/tools/of`、メモリ・CPU 上限つき）で回す。
ゲート G3（押し出し機）で作ったラッパと同じ段取り。

    python experiments/nsb/run_trama_of.py --variant porous --out /tmp/of-trama/porous \
        2>&1 | tee experiments/nsb/logs/of-trama-porous-$(date +%s).log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "extruder"))

from foam_io import continuity_converged, latest_time, run_of  # noqa: E402
from trama_case import load_trama  # noqa: E402
from trama_of_case import DEFAULT_PATTERN, retype_patches, write_case  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def residual_history(log_path: Path) -> dict[str, list[float]]:
    """simpleFoam ログから反復ごとの初期残差（Ux, Uy, p）と流量を拾う."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hist: dict[str, list[float]] = {"Ux": [], "Uy": [], "p": [], "time": []}
    for m in re.finditer(r"^Time = (\d+)$(.*?)(?=^Time = |\Ztime)", text, re.M | re.S):
        step, body = int(m.group(1)), m.group(2)
        got = {}
        for f in ("Ux", "Uy", "p"):
            mm = re.search(rf"Solving for {f}, Initial residual = ([-+0-9.eE]+)", body)
            if mm:
                got[f] = float(mm.group(1))
        if len(got) == 3:
            hist["time"].append(step)
            for f in ("Ux", "Uy", "p"):
                hist[f].append(got[f])
    return hist


def flow_balance(log_path: Path) -> dict[str, float]:
    """最後に記録された inlet/outlet の phi 総和 [m³/s]."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for name, key in (("massIn", "inlet"), ("massOut", "outlet")):
        ms = re.findall(rf"{name}\s+sum\(\w+\)\s*\(phi\)\s*=\s*([-+0-9.eE]+)", text)
        if not ms:
            ms = re.findall(rf"sum\({key}\)\s*of\s*phi\s*=\s*([-+0-9.eE]+)", text)
        if ms:
            out[key] = float(ms[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", nargs="?", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["porous", "walls"], default="porous")
    ap.add_argument(
        "--geo-variant",
        choices=["orig", "ortho", "lead"],
        default="orig",
        help="中心線の作り方（nsb 側の trama_case.py --variant と揃える）",
    )
    ap.add_argument("--dx", type=float, default=1.5)
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--end-time", type=int, default=5000)
    ap.add_argument("--write-interval", type=int, default=500)
    ap.add_argument("--scheme", choices=["sou", "upwind", "linear"], default="sou")
    ap.add_argument("--no-limited-grad", action="store_true")
    ap.add_argument("--relax-u", type=float, default=0.7)
    ap.add_argument("--relax-p", type=float, default=0.3)
    ap.add_argument("--simplec", action="store_true")
    ap.add_argument("--h-blocked", type=float, default=1.0e-5)
    ap.add_argument("--mu", type=float, default=3.0e-3, help="粘度 [Pa·s]。N を振る対照用")
    ap.add_argument(
        "--h-channel",
        type=float,
        default=3.8e-3,
        help="流路の隙間 [m]。h を変えると Re_h は不変のまま N だけ動く（u ∝ 1/h）",
    )
    ap.add_argument("--transient", action="store_true", help="pimpleFoam で物理時間を進める")
    ap.add_argument("--end-time-s", type=float, default=4.0)
    ap.add_argument("--write-interval-s", type=float, default=0.05)
    ap.add_argument("--max-co", type=float, default=5.0)
    ap.add_argument("--avg-start", type=float, default=2.0)
    ap.add_argument("--init-from", default=None, help="この ケースの最新時刻を初期場にする")
    ap.add_argument("--mesh-only", action="store_true")
    ap.add_argument("--skip-mesh", action="store_true")
    ap.add_argument("--of-mem", default="8g")
    ap.add_argument("--of-cpus", default="4")
    a = ap.parse_args()

    os.environ["OF_MEM"] = a.of_mem
    os.environ["OF_CPUS"] = a.of_cpus
    case = Path(a.out)
    geo = load_trama(a.pattern, variant=a.geo_variant)
    spec = write_case(
        case,
        geo,
        variant=a.variant,
        dx_mm=a.dx,
        mass_flow=a.mass,
        end_time=a.end_time,
        write_interval=a.write_interval,
        scheme=a.scheme,
        limited_grad=not a.no_limited_grad,
        relax_u=a.relax_u,
        relax_p=a.relax_p,
        consistent=a.simplec,
        h_blocked=a.h_blocked,
        mu=a.mu,
        h_channel=a.h_channel,
        transient=a.transient,
        end_time_s=a.end_time_s,
        write_interval_s=a.write_interval_s,
        max_co=a.max_co,
        avg_start=a.avg_start,
    )
    _log(f"[of] case {case} variant={spec.variant} {spec.nx}x{spec.ny} dx={spec.dx * 1e3:.3g} mm")
    _log(
        f"[of] nu={spec.nu:g}  Q={spec.flow_rate:g} m^3/s  nu*d channel={spec.nu * spec.d_channel:.4g} blocked={spec.nu * spec.d_blocked:.4g} 1/s"
    )

    if not a.skip_mesh:
        t0 = time.perf_counter()
        # 古い時刻ディレクトリが残っていると subsetMesh が旧格子のフィールドを
        # 写そうとして「Size N is not equal to the expected length M」で落ちる
        for d in case.iterdir():
            if d.is_dir() and re.fullmatch(r"[0-9]+(\.[0-9]+)?", d.name):
                shutil.rmtree(d)
        shutil.rmtree(case / "constant" / "polyMesh", ignore_errors=True)
        run_of(str(case), "./mesh.sh", log=str(case / "log.mesh"))
        kinds = {"inlet": "patch", "outlet": "patch"}
        if spec.variant == "walls":
            kinds["wallChannel"] = "wall"
        retype_patches(case, kinds)
        shutil.rmtree(case / "0", ignore_errors=True)
        shutil.copytree(case / "0.orig", case / "0")
        run_of(str(case), "checkMesh", "-constant", log=str(case / "log.checkMesh"))
        txt = (case / "log.checkMesh").read_text(errors="replace")
        dirs = re.search(r"Mesh has (\d+) geometric.*directions \(([^)]*)\)", txt)
        ncell = re.search(r"cells:\s+(\d+)", txt)
        _log(
            f"[of] mesh: cells={ncell.group(1) if ncell else '?'} "
            f"geometricD={dirs.group(2) if dirs else '?'} ({time.perf_counter() - t0:.1f} s)"
        )
        if dirs and dirs.group(1) != "2":
            raise RuntimeError(f"2 次元格子になっていない: geometricD={dirs.group(2)}")
    else:
        shutil.rmtree(case / "0", ignore_errors=True)
        shutil.copytree(case / "0.orig", case / "0")
    if a.mesh_only:
        return

    if a.init_from is not None:
        src = Path(a.init_from)
        st = latest_time(str(src))
        for f in ("U", "p", "phi"):
            if (src / st / f).exists():
                shutil.copy(src / st / f, case / "0" / f)
        _log(f"[of] 初期場を {src}/{st} から取った")

    t0 = time.perf_counter()
    solver = "pimpleFoam" if a.transient else "simpleFoam"
    log_path = case / f"log.{solver}"
    try:
        # ログは tee でコンテナ内から逐次書かせる（途中経過を見るため）
        # pipefail が無いと tee の終了状態で成功に見えてしまう
        run_of(str(case), "bash", "-c", f"set -o pipefail; {solver} 2>&1 | tee log.{solver}")
        failed = False
    except RuntimeError as e:
        _log(f"[of] simpleFoam 異常終了: {e}")
        failed = True
    elapsed = time.perf_counter() - t0
    n_it, converged = (0, False)
    if log_path.exists():
        if a.transient:
            times = re.findall(
                r"^Time = ([-+0-9.eE]+)$", log_path.read_text(errors="replace"), re.M
            )
            n_it = len(times)
            converged = bool(times) and float(times[-1]) >= a.end_time_s - 1e-9
        else:
            n_it, converged = continuity_converged(str(log_path))
    hist = residual_history(log_path) if log_path.exists() else {}
    bal = flow_balance(log_path) if log_path.exists() else {}
    res = {
        "variant": spec.variant,
        "iterations": n_it,
        "converged": bool(converged),
        "failed": failed,
        "elapsed": elapsed,
        "flow_rate_target": spec.flow_rate,
        "flow_balance": bal,
        "final_residual": {k: (v[-1] if v else None) for k, v in hist.items() if k != "time"},
        "min_residual": {k: (min(v) if v else None) for k, v in hist.items() if k != "time"},
    }
    (case / "result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    _log(f"[of] {solver} {n_it} 反復/ステップ 完走={converged} ({elapsed:.1f} s)")
    _log(f"[of] 最終初期残差 {res['final_residual']}")
    _log(f"[of] 流量 target={spec.flow_rate:.6g} balance={bal}")


if __name__ == "__main__":
    main()
