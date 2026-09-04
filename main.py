"""nsb パラメータスタディ: 手元構成（踏んでいる線込み）と修正構成を同じケースで比較する.

使用例::

    python main.py --models uturn flat --refine 1 --u 0.1 1 2 --configs mine fixed \
        2>&1 | tee experiments/nsb/logs/main-$(date +%s).log

configs:
  mine   : 局所 Δτ、速度下限なし、cfl_init=0.5、擬似時間項を残差にも含める（手元構成の推定）
  mine_nores : mine から擬似時間項を残差から外す
  floor  : mine + 速度下限 0.1 U_in
  fixed  : 局所 Δτ、速度下限 0.1 U_in、擬似時間項は対角のみ（本リポジトリの基準）
  global : 大域 Δτ、速度下限 0.1 U_in、擬似時間項は対角のみ
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from nsb import NSBSettings, make_case, solve_steady
from nsb.utils import save_fields, summary

OUT = Path("experiments/nsb")


def make_settings(config: str, u_in: float, max_iter: int) -> NSBSettings:
    common = dict(cfl_init=0.5, newton_max_iter=max_iter)
    if config == "mine":
        return NSBSettings(
            local_dtau=True, velocity_floor=0.0, pseudo_time_in_residual=True, **common
        )
    if config == "mine_nores":
        return NSBSettings(
            local_dtau=True, velocity_floor=0.0, pseudo_time_in_residual=False, **common
        )
    if config == "floor":
        return NSBSettings(
            local_dtau=True, velocity_floor=0.1 * u_in, pseudo_time_in_residual=True, **common
        )
    if config == "fixed":
        return NSBSettings(
            local_dtau=True, velocity_floor=0.1 * u_in, pseudo_time_in_residual=False, **common
        )
    if config == "global":
        return NSBSettings(
            local_dtau=False, velocity_floor=0.1 * u_in, pseudo_time_in_residual=False, **common
        )
    raise ValueError(f"unknown config: {config}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["uturn", "flat"])
    ap.add_argument("--refine", nargs="+", type=int, default=[1])
    ap.add_argument("--u", nargs="+", type=float, default=[0.1, 1.0, 2.0])
    ap.add_argument("--configs", nargs="+", default=["mine", "fixed"])
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for model in args.models:
        for refine in args.refine:
            for u_in in args.u:
                for config in args.configs:
                    name = f"{model}_r{refine}_U{u_in:g}_{config}"
                    print(f"===== {name} =====", flush=True)
                    inp = make_case(model, refine, u_in, make_settings(config, u_in, args.max_iter))
                    res = solve_steady(inp, log=lambda m: print(m, flush=True))
                    results[name] = summary(inp, res)
                    print(f"--> {name}: {results[name]}", flush=True)
                    if args.save_fields and res.converged:
                        save_fields(OUT / "results" / f"{name}_fields.npz", inp, res)

    (OUT / "results").mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    out = OUT / "results" / f"main{tag}.yaml"
    out.write_text(yaml.safe_dump(results, sort_keys=False))
    print("\n==== summary ====")
    for k, v in results.items():
        print(
            f"{k:28s} conv={v['converged']!s:5s} reason={v['reason']:10s} it={v['n_iter']:3d} "
            f"step1={v['first_step_ratio']:.2e} rel={v['rel_final']:.2e} "
            f"steady={v['rel_steady_final']:.2e} m_out/m_in={v['mass_ratio']:.4f}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
