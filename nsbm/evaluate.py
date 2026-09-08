"""評価: テスト θ ごとに複数の方式（初期場 × cfl_init）で nsb を回し、Newton 反復数を比べる.

[方式] "base[_nK][@cfl]"。base は stokes / knn / unet、_nK は Newton 射影 K 歩、@cfl は SER の出発係数 cfl_init を
  数値（"@4"）か UNet の予測値（"@pred"）にする（無指定は settings の既定 0.25）。
  例: "stokes", "stokes@4", "stokes@pred"（Stokes 場 + 予測 cfl: cfl ヘッドだけの効果）, "unet@pred"（場 + cfl）,
  "unet"（場だけ、既定 cfl）。
[kNN] 入力画像 x をチャネルごとに標準化して平坦化し、train 集合との L2 距離で近傍 k 件、距離逆数重みで y を平均。
  学習なしでデータセットの近さだけを使う基準。UNet がこれに勝たなければ学習の価値はない。
[後処理] kNN / UNet の初期場は `mask_blocked` で閉塞セルの速度を 0 にしてから渡す（残差比 1000 → 数倍）。
[Stokes] 生成時と同じ（u0/v0/p0 なし）。反復数は再走行して揃える（r0_ratio と cfl0 も取る）。
[指標] n_iter、converged、r0_ratio = |R(x0)|/|R_ref|（初期残差の比、Stokes は 1.0 のはず）、cfl0（SER の出発 CFL）。
[場の精度] `field_metrics`: チャネルごとの R²（全セル・全サンプルをまとめた 1 − SSE/SST とサンプル別の中央値）、
  場の最大値・最小値の誤差（正規化単位の絶対値と、正解の値域で割った比の中央値 / 90 パーセンタイル）。
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from nsb.core import NSBSettings
from nsb.solver import solve_steady
from nsbm.dataset import Sample
from nsbm.families import Theta, build_input
from nsbm.features import denormalize_y, mask_blocked
from nsbm.project import newton_project

LogFn = Callable[[str], None]
METHODS = ("stokes", "knn", "unet")


def parse_method(name: str) -> tuple[str, int]:
    """ "unet_n2@pred" → ("unet", 2): 基底の初期場と Newton 射影の歩数（cfl 指定は `parse_cfl`）."""
    name = name.partition("@")[0]
    base, _, suffix = name.partition("_")
    return base, int(suffix[1:]) if suffix.startswith("n") else 0


def parse_cfl(name: str) -> str | float | None:
    """ "unet@pred" → "pred"、"stokes@4" → 4.0、"stokes" → None（既定の cfl_init）."""
    _, sep, spec = name.partition("@")
    if not sep:
        return None
    return "pred" if spec == "pred" else float(spec)


def field_metrics(
    yhat: np.ndarray, y: np.ndarray, names: Sequence[str] = ("u", "v", "p")
) -> dict[str, Any]:
    """予測 (N,3,H,W) と正解の R² と場の最大値・最小値の誤差（正規化単位）."""
    out: dict[str, Any] = {}
    for c, nm in enumerate(names):
        a, b = (
            yhat[:, c].reshape(len(y), -1).astype(float),
            y[:, c].reshape(len(y), -1).astype(float),
        )
        sse = ((a - b) ** 2).sum()
        sst = ((b - b.mean()) ** 2).sum()
        r2_each = 1.0 - ((a - b) ** 2).sum(1) / np.maximum(
            ((b - b.mean(1, keepdims=True)) ** 2).sum(1), 1e-30
        )
        rng = np.maximum(b.max(1) - b.min(1), 1e-12)
        e_max = a.max(1) - b.max(1)
        e_min = a.min(1) - b.min(1)
        out[nm] = {
            "r2_pooled": float(1.0 - sse / max(sst, 1e-30)),
            "r2_median": float(np.median(r2_each)),
            "r2_p10": float(np.percentile(r2_each, 10)),
            "r2_p90": float(np.percentile(r2_each, 90)),
            "r2_min": float(r2_each.min()),
            "rmse": float(np.sqrt(((a - b) ** 2).mean())),
            "max_err_abs_median": float(np.median(np.abs(e_max))),
            "max_err_abs_p90": float(np.percentile(np.abs(e_max), 90)),
            "max_err_rel_median": float(np.median(np.abs(e_max) / rng)),
            "max_err_rel_p90": float(np.percentile(np.abs(e_max) / rng, 90)),
            "max_err_signed_mean": float(e_max.mean()),
            "min_err_abs_median": float(np.median(np.abs(e_min))),
            "min_err_abs_p90": float(np.percentile(np.abs(e_min), 90)),
            "min_err_rel_median": float(np.median(np.abs(e_min) / rng)),
            "min_err_rel_p90": float(np.percentile(np.abs(e_min) / rng, 90)),
            "min_err_signed_mean": float(e_min.mean()),
        }
    return out


def channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return mean, std


def knn_predict(
    x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, k: int = 4, stats=None
) -> np.ndarray:
    """(N, C, H, W) の train に対し x (C, H, W) の近傍 k 件の y を距離逆数重みで平均."""
    mean, std = stats or channel_stats(x_train)
    a = ((x_train - mean) / std).reshape(len(x_train), -1)
    b = ((x[None] - mean) / std).reshape(1, -1)
    d = np.sqrt(((a - b) ** 2).sum(axis=1))
    nn = np.argsort(d)[:k]
    w = 1.0 / (d[nn] + 1e-9)
    w /= w.sum()
    return np.tensordot(w, y_train[nn], axes=1)


def run_with_init(
    theta: Theta,
    init: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    settings: NSBSettings | None = None,
) -> dict[str, Any]:
    inp = build_input(theta, settings, init)
    t0 = time.perf_counter()
    res = solve_steady(inp, log=None)
    return {
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        "r0_ratio": float(res.steady_residual_history[0] / res.residual_ref),
        "cfl0": float(res.cfl_history[0]) if res.cfl_history else float("nan"),
        "elapsed": time.perf_counter() - t0,
    }


def _job(
    args: tuple[int, str, dict, Any, NSBSettings | None, np.ndarray | None, float | None],
) -> tuple[int, str, dict[str, Any]]:
    k, method, theta_d, init, settings, x_img, cfl_init = args
    theta = Theta.from_dict(theta_d)
    _, steps = parse_method(method)
    if cfl_init is not None:
        settings = dataclasses.replace(settings or NSBSettings(), cfl_init=float(cfl_init))
    extra: dict[str, Any] = {}
    if steps > 0 and init is not None:
        init, info = newton_project(build_input(theta, settings), init, x_img, steps)
        extra = {
            "proj_r_before": info["r_before"],
            "proj_r_after": info["r_after"],
            "proj_steps": info["steps_taken"],
        }
    extra["cfl_init"] = (settings or NSBSettings()).cfl_init
    return k, method, {**run_with_init(theta, init, settings), **extra}


def _init_worker() -> None:
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass


def evaluate(
    predict: Callable[[Sample], tuple[np.ndarray, float]],
    samples: Sequence[Sample],
    split: dict[str, list[int]],
    k: int = 4,
    n_workers: int = 8,
    settings: NSBSettings | None = None,
    methods: Sequence[str] = METHODS,
    log: LogFn | None = print,
) -> list[dict[str, Any]]:
    """テスト集合の各 θ を methods の各方式で解く。predict は Sample → (正規化 y (3,72,48), 予測 cfl_init) の UNet 推論."""
    x_tr = np.stack([samples[i].x for i in split["train"]])
    y_tr = np.stack([samples[i].y for i in split["train"]])
    stats = channel_stats(x_tr)
    jobs = []
    rows: dict[int, dict[str, Any]] = {}
    for k_idx in list(split["test"]) + list(split.get("hard", [])):
        s = samples[k_idx]
        rows[k_idx] = {
            "group": "hard" if k_idx in set(split.get("hard", [])) else "test",
            "seed": s.theta.seed,
            "family": s.theta.family,
            "u_in": s.theta.u_in,
            "h0": s.theta.h0,
            "inlet": s.theta.inlet.wall,
            "outlet": s.theta.outlet.wall,
            "n_iter_dataset": s.n_iter,
        }
        bases = {parse_method(m)[0] for m in methods}
        inits: dict[str, Any] = {"stokes": None}
        if "knn" in bases:
            inits["knn"] = mask_blocked(
                s.x, denormalize_y(s.theta, knn_predict(x_tr, y_tr, s.x, k, stats))
            )
        cfl_pred: float | None = None
        if "unet" in bases or any(parse_cfl(m) == "pred" for m in methods):
            y_pred, cfl_pred = predict(s)
            inits["unet"] = mask_blocked(s.x, denormalize_y(s.theta, y_pred))
            rows[k_idx]["cfl_pred"] = float(cfl_pred)
        for m in methods:
            spec = parse_cfl(m)
            cfl_m = cfl_pred if spec == "pred" else spec
            jobs.append(
                (k_idx, m, s.theta.to_dict(), inits[parse_method(m)[0]], settings, s.x, cfl_m)
            )
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for n, (k_idx, m, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            for key, val in r.items():
                rows[k_idx][f"{m}_{key}"] = val
            if log is not None:
                log(
                    f"[{n + 1}/{len(jobs)}] seed={rows[k_idx]['seed']} {rows[k_idx]['family']:10s} {m:12s} "
                    f"conv={r['converged']!s:5s} newton={r['n_iter']:3d} r0={r['r0_ratio']:.2e} "
                    f"cfl_init={r['cfl_init']:.3g} cfl0={r['cfl0']:.2e} ({time.perf_counter() - t0:6.1f}s)"
                )
    return [rows[i] for i in list(split["test"]) + list(split.get("hard", []))]


def summarize(rows: Sequence[dict[str, Any]], methods: Sequence[str] = METHODS) -> dict[str, Any]:
    """全体・ファミリ別の Newton 反復数の中央値/四分位、Stokes 比、勝敗数."""

    def block(sub: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(sub)}
        for m in methods:
            it = np.array([r[f"{m}_n_iter"] for r in sub], dtype=float)
            conv = np.array([r[f"{m}_converged"] for r in sub])
            r0 = np.array([r[f"{m}_r0_ratio"] for r in sub])
            out[m] = {
                "converged": int(conv.sum()),
                "cfl_init_median": float(
                    np.median([r.get(f"{m}_cfl_init", float("nan")) for r in sub])
                ),
                "n_iter_median": float(np.median(it)),
                "n_iter_q1": float(np.percentile(it, 25)),
                "n_iter_q3": float(np.percentile(it, 75)),
                "n_iter_mean": float(it.mean()),
                "r0_ratio_median": float(np.median(r0)),
            }
        base_m = "stokes" if "stokes" in methods else methods[0]
        st = np.array([r[f"{base_m}_n_iter"] for r in sub], dtype=float)
        out["baseline"] = base_m
        for m in methods:
            if m == base_m:
                continue
            it = np.array([r[f"{m}_n_iter"] for r in sub], dtype=float)
            out[m]["ratio_to_stokes_median"] = float(np.median(it / st))
            out[m]["wins"] = int((it < st).sum())
            out[m]["ties"] = int((it == st).sum())
            out[m]["losses"] = int((it > st).sum())
        if "knn" in methods and "unet" in methods:
            a = np.array([r["unet_n_iter"] for r in sub])
            b = np.array([r["knn_n_iter"] for r in sub])
            out["unet_vs_knn"] = {
                "wins": int((a < b).sum()),
                "ties": int((a == b).sum()),
                "losses": int((a > b).sum()),
            }
        return out

    test = [r for r in rows if r.get("group", "test") == "test"]
    hard = [r for r in rows if r.get("group") == "hard"]
    fams = sorted({r["family"] for r in test})
    out = {
        "all": block(test),
        "by_family": {f: block([r for r in test if r["family"] == f]) for f in fams},
    }
    if hard:
        out["hard"] = block(hard)
    return out
