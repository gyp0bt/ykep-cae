"""データ生成: θ → nsb（Stokes 発進、既定設定）の収束解を並列に集めて npz シャードに保存する.

[並列] 1 件は 72×48 で数秒〜十数秒。ワーカーは 1 スレッドに絞り（BLAS / numba のスレッド同士の奪い合いを避ける）
  空きコア数だけ並べる。環境変数はワーカー起動前（numpy import 前）に設定する必要があるので
  `experiments/nsbm/gen.py` の先頭で設定し、spawn した子に継承させる。`_init_worker` は numba だけ念押し。
[保存] `Sample` は入力画像 x・正規化した正解 y・θ・収束情報。未収束も保存する（学習ラベルからは除外）。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nsb.core import NSBSettings
from nsb.solver import solve_steady
from nsbm.families import FAMILIES, Theta, build_h, build_input, sample_theta
from nsbm.features import denormalize_y, make_x, normalize_y

LogFn = Callable[[str], None]


@dataclass
class Sample:
    theta: Theta
    x: np.ndarray  # (8, 72, 48) float32
    y: np.ndarray  # (3, 72, 48) float32、正規化済み
    n_iter: int
    converged: bool
    n_gmres_total: int
    residual_ref: float
    elapsed: float

    def fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """物理量 (u, v, p) に戻す."""
        return denormalize_y(self.theta, self.y)


def solve_sample(
    seed: int,
    families: Sequence[str] = FAMILIES,
    settings: NSBSettings | None = None,
) -> Sample:
    theta = sample_theta(seed, families)
    h = build_h(theta)
    inp = build_input(theta, settings)
    t0 = time.perf_counter()
    res = solve_steady(inp, log=None)
    return Sample(
        theta=theta,
        x=make_x(theta, h),
        y=normalize_y(theta, res.u, res.v, res.p),
        n_iter=int(res.n_iter),
        converged=bool(res.converged),
        n_gmres_total=int(res.n_gmres_total),
        residual_ref=float(res.residual_ref),
        elapsed=time.perf_counter() - t0,
    )


def save_shard(path: Path, samples: Sequence[Sample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.stack([s.x for s in samples]),
        y=np.stack([s.y for s in samples]),
        theta=np.array([json.dumps(s.theta.to_dict()) for s in samples]),
        n_iter=np.array([s.n_iter for s in samples], dtype=np.int32),
        converged=np.array([s.converged for s in samples], dtype=bool),
        n_gmres_total=np.array([s.n_gmres_total for s in samples], dtype=np.int32),
        residual_ref=np.array([s.residual_ref for s in samples], dtype=np.float64),
        elapsed=np.array([s.elapsed for s in samples], dtype=np.float64),
    )
    return path


def load_shards(directory: Path) -> list[Sample]:
    out: list[Sample] = []
    for path in sorted(Path(directory).glob("shard-*.npz")):
        with np.load(path) as z:
            for k in range(len(z["theta"])):
                out.append(
                    Sample(
                        theta=Theta.from_dict(json.loads(str(z["theta"][k]))),
                        x=z["x"][k],
                        y=z["y"][k],
                        n_iter=int(z["n_iter"][k]),
                        converged=bool(z["converged"][k]),
                        n_gmres_total=int(z["n_gmres_total"][k]),
                        residual_ref=float(z["residual_ref"][k]),
                        elapsed=float(z["elapsed"][k]),
                    )
                )
    return out


def _init_worker() -> None:
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass


def _solve_seed(args: tuple[int, NSBSettings | None]) -> Sample:
    seed, settings = args
    return solve_sample(seed, settings=settings)


def generate(
    seeds: Sequence[int],
    out_dir: Path,
    n_workers: int,
    shard_size: int = 256,
    log: LogFn | None = print,
    settings: NSBSettings | None = None,
) -> list[Path]:
    """seeds を並列に解いて shard_size 件ごとに npz へ。既存のシャード番号の続きから書く."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(out_dir.glob("shard-*.npz")))
    paths: list[Path] = []
    buf: list[Sample] = []
    t0 = time.perf_counter()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for s in pool.imap_unordered(_solve_seed, [(int(s), settings) for s in seeds], chunksize=1):
            done += 1
            buf.append(s)
            if log is not None:
                log(
                    f"[{done}/{len(seeds)}] seed={s.theta.seed} {s.theta.family:10s} "
                    f"h0={s.theta.h0:.2e} u_in={s.theta.u_in:.2f} {s.theta.inlet.wall}->{s.theta.outlet.wall} "
                    f"conv={s.converged!s:5s} newton={s.n_iter:3d} gmres={s.n_gmres_total:5d} "
                    f"{s.elapsed:6.1f}s  (wall {time.perf_counter() - t0:7.1f}s)"
                )
            if len(buf) >= shard_size:
                paths.append(save_shard(out_dir / f"shard-{n_existing + len(paths):04d}.npz", buf))
                buf = []
    if buf:
        paths.append(save_shard(out_dir / f"shard-{n_existing + len(paths):04d}.npz", buf))
    return paths
