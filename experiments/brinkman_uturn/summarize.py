"""results*/ の YAML を集計して Markdown 表を出力する.

使用例::

    python experiments/brinkman_uturn/summarize.py experiments/brinkman_uturn/results
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def load_cases(result_dir: Path) -> list[dict]:
    cases = []
    for f in sorted(result_dir.glob("*_jfnk.yaml")) + sorted(
        result_dir.glob("*_defect_correction.yaml")
    ):
        cases.append(yaml.safe_load(f.read_text()))
    return cases


def table(cases: list[dict]) -> str:
    lines = [
        "| model | mesh | U [m/s] | Re_in | 収束 | 理由 | Newton 反復 | 最終相対残差 | 最小相対残差 | 初回残差増幅 | 時間 [s] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    key = lambda c: (c["model"], c["refine"], c["u_inlet"])  # noqa: E731
    for c in sorted(cases, key=key):
        h = c["residual_history"]
        amp = h[1] / h[0] if len(h) > 1 else float("nan")
        lines.append(
            f"| {c['model']} | {c['nx']}×{c['ny']} | {c['u_inlet']:g} | {c['re_inlet']:.0f} | "
            f"{'○' if c['converged'] else '✗'} | {c['failure_reason'] or '-'} | {c['n_newton']} | "
            f"{c['final_relative_residual']:.1e} | {c['min_relative_residual']:.1e} | {amp:.1f}× | {c['elapsed_seconds']:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    for d in sys.argv[1:]:
        p = Path(d)
        cases = load_cases(p)
        print(f"### {p.name}（{len(cases)} ケース）\n")
        print(table(cases))
        print()


if __name__ == "__main__":
    main()
