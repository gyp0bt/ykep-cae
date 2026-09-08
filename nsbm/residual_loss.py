"""残差駆動の損失: 予測場 x_0（と cfl_init）から nsb と同じ擬似時間 Newton を K 歩展開し、各歩の定常残差比の和と
その x_0・log cfl_init に関する勾配（凍結ヤコビアン随伴）を返す.

[舞台] 未知数空間 R^{3n}（u, v, p）。真の解は R(x) = 0 の点。予測場 x_0 はその近くの点で、Newton は
  x_{k+1} = x_k − (J_k + D_k)^{-1} R(x_k) で解へ引き寄せる（D_k = ρV/Δτ_k の対角、u・v ブロックのみ）。
[損失] L = Σ_{k=0}^{K} w(ρ_k)、ρ_k = |R(x_k)| / |R_ref|（定常残差、Stokes 参照場で規格化）。
  w は "ratio"（ρ そのまま、gyp さん指定）か "log"（log ρ、反復ごとの減衰率を等重みに見る）。
[勾配] 1 歩の写像を J_k・D_k を凍結した線形写像とみなすと ∂x_{k+1}/∂x_k = (J_k+D_k)^{-1} D_k。
  随伴は x̄_k = g_k + D_k (J_k+D_k)^{-T} x̄_{k+1}、g_k = w'(ρ_k)·J_kᵀ R_k / (|R_k| |R_ref|)。
  D_k = c/CFL_k なので同じ転置解 w = (J+D)^{-T} x̄_{k+1} から CFL̄_k += (D_k δ_k)·w / CFL_k が無料で出る。
  SER の連鎖 CFL_{k+1} = CFL_k·ratio_k は ratio_k を凍結、CFL_0 = cfl_init·|R_ref|/|R(x_0)| は x_0 依存も含める。
  落としている項: Δτ_k の速度依存（∂D/∂x）、SER 比の残差依存、J の x 依存（2 階微分）。k=0 の項は厳密。
[コスト] 1 サンプル: 彩色 FD ヤコビアン (K+1) 回 + 疎 LU 2K 回（順・転置）。72×48 で 1〜2 s。
[整合] 順方向の 1 歩は nsb 自身の `solve_linear`（JFNK + 遅延前処理、GMRES 許容 1e-3）で進めるので、
  状態 x_k と残差履歴は `solve_steady` と同じ経路をたどる（厳密 J の直接解だと残差が 2〜3 倍速く落ち、
  実機より甘い損失になる: パイロット 27→51→8.8 vs nsb 27→106→42）。随伴側だけ厳密 J（彩色 FD）を使い、
  (J+D)^{-T} は疎 LU で解く。設定は sub_iters=1, pseudo_time_in_residual=True, rc_with_pseudo_time=False,
  alpha_u=1（nsb 既定）を前提にし、それ以外は NotImplementedError。線形解が実質失敗（reject_lin_ratio 超）
  なら nsb と同様に歩を捨てて CFL を ser_shrink 倍にする（その歩は展開に数えず、勾配経路にも入れない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from nsb.adjoint import colored_fd_jacobian
from nsb.assembly import BrinkmanDiscretization
from nsb.core import NSBInput
from nsb.linalg import pardiso_solve
from nsb.solver import LaggedPreconditioner, compute_dtau, solve_linear

TRANSFORMS = ("ratio", "log")


def _w(rho: float, transform: str) -> tuple[float, float]:
    """w(ρ) と w'(ρ)."""
    if transform == "ratio":
        return rho, 1.0
    if transform == "log":
        eps = 1e-12
        return float(np.log(rho + eps)), 1.0 / (rho + eps)
    raise ValueError(f"transform は {TRANSFORMS} のいずれか: {transform!r}")


@dataclass
class Unrolled:
    """展開した K 歩の記録（逆伝播に必要な量）."""

    xs: list[np.ndarray] = field(default_factory=list)  # x_0 .. x_K
    rs: list[np.ndarray] = field(default_factory=list)  # R(x_k) 定常残差
    rhos: list[float] = field(default_factory=list)  # |R(x_k)|/|R_ref|
    Js: list[sparse.csr_matrix] = field(default_factory=list)  # J_k (k=0..K)
    As: list[sparse.csc_matrix] = field(default_factory=list)  # J_k + D_k (k<K)
    Ds: list[np.ndarray] = field(default_factory=list)  # 対角 (3n,) (k<K)
    deltas: list[np.ndarray] = field(default_factory=list)  # δ_k (k<K)
    cfls: list[float] = field(default_factory=list)  # CFL_k (k<K)
    ratios: list[float] = field(default_factory=list)  # CFL_{k+1}/CFL_k (k<K)
    lin_ratios: list[float] = field(default_factory=list)  # 線形解の真の残差比（診断用）
    cfl0_capped: bool = False
    n_rejected: int = 0
    failure: str = ""


def _check_settings(inp: NSBInput) -> None:
    s = inp.settings
    if not (
        s.sub_iters == 1
        and s.pseudo_time_in_residual
        and not s.rc_with_pseudo_time
        and s.alpha_u == 1.0
    ):
        raise NotImplementedError(
            "residual_loss は nsb 既定の制御則（sub_iters=1, pseudo_time_in_residual, "
            "rc_with_pseudo_time=False, alpha_u=1）だけを再現する"
        )


def unroll(
    inp: NSBInput,
    x0: np.ndarray,
    cfl_init: float,
    r_ref: float,
    steps: int = 5,
    need_jacobians: bool = True,
    linear: str = "nsb",
) -> Unrolled:
    """x0 から擬似時間 Newton を steps 歩展開する（nsb の 1 反復 = 1 擬似時間ステップ）.

    linear="nsb": nsb の `solve_linear`（JFNK + 遅延前処理）。実機と同じ状態列になる。
    linear="direct": 厳密 J の疎 LU 直接解。損失が x0 の滑らかな関数になるので随伴の FD 照合に使う。
    """
    _check_settings(inp)
    s = inp.settings
    disc = BrinkmanDiscretization(inp.to_flow_input())
    n = disc.n
    u_floor = s.velocity_floor_ratio * disc.u_scale

    def resid(xx: np.ndarray) -> np.ndarray:
        return disc.residual_fast(xx, s.scheme, s.venkat_k, None)

    def resid_tau(x_prev: np.ndarray):
        """[残差] 擬似時間項込み R_τ = R + D(x − x_prev)（solve_linear の JFNK matvec 用）."""

        def f(xx: np.ndarray) -> np.ndarray:
            return resid(xx) + D * (xx - x_prev)

        return f

    pc = LaggedPreconditioner(s, n)
    rec = Unrolled()
    x = np.asarray(x0, dtype=float).copy()
    r = resid(x)
    rn = float(np.linalg.norm(r))
    rec.xs.append(x)
    rec.rs.append(r)
    rec.rhos.append(rn / r_ref)
    if not np.isfinite(rn) or rn == 0.0:
        rec.failure = "bad_init"
        return rec
    cfl_raw = cfl_init * r_ref / rn
    rec.cfl0_capped = cfl_raw > s.cfl_max
    cfl = float(min(s.cfl_max, cfl_raw))
    rn_tau = rn  # SER が見る擬似時間項込みの残差ノルム（初回は定常残差と同じ）
    for _k in range(steps):
        if rn / r_ref > s.divergence_ratio:
            rec.failure = "diverged"
            break
        uu, vv, _ = disc.split(x)
        dtau = compute_dtau(uu, vv, disc.dx, disc.dy, cfl, s, u_floor)
        tau = (inp.rho * disc.vol / dtau).ravel()
        D = np.concatenate([tau, tau, np.zeros(n)])
        pc.cfl = cfl
        J: sparse.csr_matrix | None = None
        if linear == "direct":
            J = colored_fd_jacobian(disc, resid, x)
            delta = pardiso_solve((J + sparse.diags(D)).tocsc(), -r)
            lin_ratio = 0.0
        else:
            st = disc.compute_state(x, s.scheme, s.venkat_k, None)
            try:
                delta, _n_g, _ok, lin_ratio = solve_linear(disc, st, x, r, D, resid_tau(x), s, pc)
            except (RuntimeError, ValueError) as exc:
                rec.failure = f"lu_failed: {exc}"
                break
        if not np.all(np.isfinite(delta)):
            rec.failure = "gmres_breakdown"
            break
        if s.reject_lin_ratio > 0.0 and not (lin_ratio <= s.reject_lin_ratio):
            cfl = float(cfl * s.ser_shrink)
            rec.n_rejected += 1
            if rec.n_rejected > steps:
                rec.failure = "rejected"
                break
            continue
        x_new = x + delta
        r_new = resid(x_new)
        rn_new = float(np.linalg.norm(r_new))
        rn_tau_new = float(np.linalg.norm(r_new + D * delta))
        if not np.isfinite(rn_new):
            rec.failure = "nan"
            break
        ratio_ser = rn_tau / rn_tau_new if rn_tau_new > 0.0 else s.ser_shrink
        cfl_new = float(min(s.cfl_max, cfl * float(np.clip(ratio_ser, s.ser_shrink, s.ser_growth))))
        if need_jacobians:
            if J is None:
                J = colored_fd_jacobian(disc, resid, x)
            rec.Js.append(J)
            rec.As.append((J + sparse.diags(D)).tocsc())
        rec.Ds.append(D)
        rec.deltas.append(delta)
        rec.cfls.append(cfl)
        rec.ratios.append(cfl_new / cfl)
        rec.lin_ratios.append(float(lin_ratio))
        rec.xs.append(x_new)
        rec.rs.append(r_new)
        rec.rhos.append(rn_new / r_ref)
        x, r, rn, rn_tau, cfl = x_new, r_new, rn_new, rn_tau_new, cfl_new
    pc.free()
    if need_jacobians and not rec.failure:
        rec.Js.append(colored_fd_jacobian(disc, resid, x))  # 最終点の直接項用
    return rec


def residual_loss(
    inp: NSBInput,
    x0: np.ndarray,
    cfl_init: float,
    r_ref: float,
    steps: int = 5,
    transform: str = "ratio",
    with_grad: bool = True,
    linear: str = "nsb",
) -> dict[str, Any]:
    """損失 L と勾配 (∂L/∂x_0, ∂L/∂log cfl_init) を返す.

    戻り値: loss, rhos (K+1 個), grad_x0 (3n,), grad_logcfl (float), n_steps, failure, cfl0。
    発散・失敗した歩以降は損失に入れない（n_steps で分かる）。
    """
    rec = unroll(inp, x0, cfl_init, r_ref, steps, need_jacobians=with_grad, linear=linear)
    K = len(rec.deltas)
    loss = 0.0
    dws: list[float] = []
    for rho in rec.rhos:
        w, dw = _w(rho, transform)
        loss += w
        dws.append(dw)
    out: dict[str, Any] = {
        "loss": float(loss),
        "rhos": [float(r) for r in rec.rhos],
        "n_steps": K,
        "failure": rec.failure,
        "cfl0": rec.cfls[0] if rec.cfls else float("nan"),
    }
    if not with_grad:
        return out
    n3 = x0.size
    if rec.failure and len(rec.Js) < len(rec.rhos):
        # 最終点のヤコビアンが無い（失敗で打ち切り）: 最終点の直接項は落とす
        rec.Js.append(None)  # type: ignore[arg-type]

    def direct(k: int) -> np.ndarray:
        J = rec.Js[k]
        if J is None:
            return np.zeros(n3)
        rk = rec.rs[k]
        nk = float(np.linalg.norm(rk))
        return dws[k] * (J.T @ rk) / (nk * r_ref)

    xbar = direct(K)
    cflbar = 0.0
    for k in range(K - 1, -1, -1):
        wv = pardiso_solve(rec.As[k].T.tocsc(), xbar)
        cflbar_k = (
            float(np.dot(rec.Ds[k] * rec.deltas[k], wv)) / rec.cfls[k] + cflbar * rec.ratios[k]
        )
        xbar = direct(k) + rec.Ds[k] * wv
        cflbar = cflbar_k
    grad_logcfl = 0.0
    if K > 0 and not rec.cfl0_capped:
        # CFL_0 = cfl_init·r_ref/|R_0|: ∂/∂log cfl_init = CFL_0、∂/∂x_0 = −CFL_0/|R_0|² · J_0ᵀ R_0
        cfl0 = rec.cfls[0]
        grad_logcfl = cflbar * cfl0
        r0 = rec.rs[0]
        n0 = float(np.linalg.norm(r0))
        xbar = xbar - cflbar * cfl0 / n0**2 * (rec.Js[0].T @ r0)
    out["grad_x0"] = xbar
    out["grad_logcfl"] = float(grad_logcfl)
    return out


# ----------------------------------------------------------------------------------------------
# 学習側との橋渡し: ワーカープールで残差損失と勾配を並列に評価し、torch には値と勾配を「直通」で渡す
# ----------------------------------------------------------------------------------------------


def _init_worker() -> None:
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass


def _res_job(args: tuple) -> dict[str, Any]:
    theta_d, x0, cfl_init, r_ref, steps, transform, with_grad, settings = args
    from nsbm.families import Theta, build_input

    inp = build_input(Theta.from_dict(theta_d), settings)
    try:
        out = residual_loss(
            inp,
            np.asarray(x0, dtype=float).ravel(),
            float(cfl_init),
            float(r_ref),
            steps,
            transform,
            with_grad,
        )
    except Exception as exc:  # 1 件の失敗で学習全体を止めない（原因は failure に残す）
        out = {
            "loss": float("nan"),
            "rhos": [],
            "n_steps": 0,
            "failure": f"{type(exc).__name__}: {exc}",
            "cfl0": float("nan"),
        }
        if with_grad:
            out["grad_x0"] = np.zeros(np.asarray(x0).size)
            out["grad_logcfl"] = 0.0
    if with_grad:
        out["grad_x0"] = np.asarray(out["grad_x0"], dtype=np.float32).reshape(np.shape(x0))
    return out


class ResidualLossPool:
    """spawn プールで `residual_loss` をサンプル並列に評価する（ワーカーは 1 スレッド）."""

    def __init__(
        self, n_workers: int = 8, steps: int = 5, transform: str = "ratio", settings=None
    ) -> None:
        import multiprocessing as mp

        self.steps, self.transform, self.settings = steps, transform, settings
        self.pool = mp.get_context("spawn").Pool(n_workers, initializer=_init_worker)

    def compute(
        self,
        thetas: list[dict],
        x0: np.ndarray,
        cfl_init: np.ndarray,
        r_ref: np.ndarray,
        with_grad: bool = True,
    ) -> list[dict[str, Any]]:
        jobs = [
            (
                thetas[b],
                x0[b],
                cfl_init[b],
                r_ref[b],
                self.steps,
                self.transform,
                with_grad,
                self.settings,
            )
            for b in range(len(thetas))
        ]
        return self.pool.map(_res_job, jobs, chunksize=1)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def __enter__(self) -> ResidualLossPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def straight_through(fields, logcfl, outs: list[dict[str, Any]]):
    """torch の損失テンソルを作る: 値は Σ_b L_b / B、勾配はワーカーが返した (∂L/∂x_0, ∂L/∂log cfl) をそのまま流す.

    surrogate = mean_b [ L_b + (x_b − x_b.detach())·ḡ_x + (c_b − c_b.detach())·ḡ_c ] は値が L の平均で、
    x_b・c_b に関する勾配がちょうど ḡ になる（custom autograd Function を書かずに済む）。失敗したサンプル（nan）は除く。
    """
    import torch

    ok = [b for b, o in enumerate(outs) if np.isfinite(o["loss"])]
    if not ok:
        return fields.sum() * 0.0 + logcfl.sum() * 0.0, 0
    g_x = torch.from_numpy(np.stack([outs[b]["grad_x0"] for b in ok])).to(fields.dtype)
    g_c = torch.tensor([outs[b]["grad_logcfl"] for b in ok], dtype=logcfl.dtype)
    vals = torch.tensor([outs[b]["loss"] for b in ok], dtype=fields.dtype)
    idx = torch.tensor(ok)
    f, c = fields[idx], logcfl[idx]
    sur = vals + ((f - f.detach()) * g_x).sum(dim=(1, 2, 3)) + (c - c.detach()) * g_c
    return sur.mean(), len(ok)
