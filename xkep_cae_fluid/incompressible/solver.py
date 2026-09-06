"""非圧縮 Navier–Stokes ソルバー（面ベース FVM、同位置格子 SIMPLE + Rhie–Chow）.

:class:`NavierStokesFVMProcess` は :mod:`xkep_cae_fluid.fvm`（:mod:`~xkep_cae_fluid.fvm.momentum` の
運動量・圧力連成カーネル、:mod:`~xkep_cae_fluid.fvm.assembly` のスカラー輸送）を組み合わせた
薄い方程式ファミリー層。構造格子 / polyMesh / .inp の ``MeshData`` を同じ経路で解く。

1 反復（SIMPLE / SIMPLEC / PISO）:

1. 圧力勾配 ∇p（最小二乗、OUTLET は Dirichlet、他はゼロ勾配）
2. 運動量 3 成分（対流 1 次風上 + TVD 遅延補正、拡散 + 非直交補正、浮力・抵抗、時間項 Euler / BDF2、
   陰的緩和 α_u）→ u*
3. Rhie–Chow の面質量流束 ṁ*、圧力補正 Σ a_f (p'_P − p'_N) = −Σ ṁ*
4. p += α_p p'、u −= (V/a_P) ∇p'、ṁ −= a_f Δp'（PISO は α_p = 1 で 3–4 を ``n_piso_correctors`` 回）
5. エネルギー（対流 ṁ + 拡散 k、固体は k_solid、陰的緩和 α_T）
6. 追加スカラー（同じ ṁ、``ScalarSpec``）

Brinkman 抵抗（透過率 K）は運動量の対角に μ/K V、Boussinesq 浮力は −ρ β (T − T_ref) g V。
領域内部の吐出・吸入（``InternalCellBC``）は速度固定セル + 圧力補正のピン留め。
設計は ``docs/design/navier-stokes-fvm.md``。
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np
from scipy import sparse

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm import (
    CONVECTION_SCHEMES,
    TVD_LIMITERS,
    PatchBC,
    assemble_scalar_transport,
    cell_gradient_lsq,
    make_linear_solver,
    resolve_boundary,
)
from xkep_cae_fluid.fvm.momentum import (
    VelocityBoundaryFaces,
    assemble_momentum,
    assemble_pressure_correction,
    correct_mass_flux,
    fix_rows,
    pressure_boundary,
    pressure_correction_coefficients,
    resolve_velocity_boundary,
    rhie_chow_mass_flux,
)
from xkep_cae_fluid.incompressible.data import (
    InternalCellBCKind,
    NavierStokesFVMInput,
    NavierStokesFVMResult,
)

logger = logging.getLogger(__name__)

_COMPONENTS = ("u", "v", "w")
_COUPLINGS = ("simple", "simplec", "piso")
_TIME_SCHEMES = ("euler", "bdf2")


def _momentum_scale(A, x: np.ndarray, b: np.ndarray, a_p: np.ndarray, u_ref: float) -> float:
    """運動量残差の正規化スケール max(‖b‖, ‖A x‖, ‖a_P‖ U_ref).

    成分が恒等的にゼロ（右辺が丸め誤差だけ）でも残差が 1 に張り付かないよう、
    速度スケール U_ref（全成分の最大絶対値）を使う。
    """
    finite = np.isfinite(a_p)
    return max(
        float(np.linalg.norm(b)),
        float(np.linalg.norm(A @ x)),
        float(np.linalg.norm(a_p[finite])) * u_ref,
        1e-300,
    )


def _residual_field(A, x: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    return np.abs(b - A @ x) / scale


def _solver(name: str, tol: float, maxiter: int):
    return make_linear_solver(
        name, **({} if name.lower() == "direct" else {"tol": tol, "maxiter": maxiter})
    )


class NavierStokesFVMProcess(SolverProcess["NavierStokesFVMInput", "NavierStokesFVMResult"]):
    """非圧縮 NS（SIMPLE / SIMPLEC + Rhie–Chow、Boussinesq、Brinkman、エネルギー）を ``MeshData`` 上で解く."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="NavierStokesFVM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/navier-stokes-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: NavierStokesFVMInput) -> NavierStokesFVMResult:
        t0 = time.perf_counter()
        inp = input_data
        if inp.rho <= 0.0 or inp.mu <= 0.0:
            raise ValueError("rho / mu は正の値が必要")
        if inp.coupling.lower() not in _COUPLINGS:
            raise ValueError(f"coupling は {_COUPLINGS} のみ: {inp.coupling!r}")
        if inp.convection.lower() not in CONVECTION_SCHEMES:
            raise ValueError(f"convection は {CONVECTION_SCHEMES} のみ: {inp.convection!r}")
        if inp.limiter.lower() not in TVD_LIMITERS:
            raise ValueError(f"limiter は {sorted(TVD_LIMITERS)} のみ: {inp.limiter!r}")
        if inp.time_scheme.lower() not in _TIME_SCHEMES:
            raise ValueError(f"time_scheme は {_TIME_SCHEMES} のみ: {inp.time_scheme!r}")
        if inp.coupling.lower() == "piso" and inp.n_piso_correctors < 1:
            raise ValueError("n_piso_correctors は 1 以上")
        names = [sp.name for sp in inp.scalars]
        if len(set(names)) != len(names):
            raise ValueError(f"scalars の名前が重複しています: {names}")
        if any(nm in (*_COMPONENTS, "T", "mass") for nm in names):
            raise ValueError(f"scalars の名前 {names} は u/v/w/T/mass と重ねられません")
        state = _State(inp)

        residual_history: dict[str, list[float]] = {
            k: [] for k in (*_COMPONENTS, "T", "mass", *names)
        }
        residual_fields: dict[str, np.ndarray] = {}
        n_outer = 0
        converged = False
        times: list[float] = []

        keys = (*_COMPONENTS, "mass", "T") if inp.solve_energy else (*_COMPONENTS, "mass")

        if not inp.is_transient:
            for it in range(inp.max_outer_iter):
                res, residual_fields = state.iterate(None, None)
                n_outer += 1
                for k, v in res.items():
                    residual_history[k].append(v)
                # 1 反復目は初期場（静止・一様温度）の残差がゼロになり得るので判定しない
                if it >= 1 and max(res[k] for k in keys) < inp.tol:
                    converged = True
                    break
            n_steps = 0
        else:
            n_steps = int(np.ceil(inp.t_end / inp.dt))
            converged = True
            t = 0.0
            old2: dict[str, np.ndarray] | None = None
            bdf2 = inp.time_scheme.lower() == "bdf2"
            for step in range(n_steps):
                t += inp.dt
                old = state.snapshot()
                step_ok = False
                for it in range(inp.max_outer_iter):
                    res, residual_fields = state.iterate(old, old2 if bdf2 else None)
                    n_outer += 1
                    for k, v in res.items():
                        residual_history[k].append(v)
                    if it >= 1 and max(res[k] for k in keys) < inp.tol:
                        step_ok = True
                        break
                converged &= step_ok
                old2 = old
                if (step + 1) % max(inp.output_interval, 1) == 0 or step == n_steps - 1:
                    times.append(t)

        return NavierStokesFVMResult(
            velocity=state.u,
            p=state.p,
            T=state.T,
            mass_flux=state.mass_flux,
            converged=bool(converged),
            n_outer_iterations=n_outer,
            n_timesteps=n_steps,
            residual_history=residual_history,
            residual_fields=residual_fields,
            elapsed_seconds=time.perf_counter() - t0,
            time_history=tuple(times),
            scalars={k: v.copy() for k, v in state.phi.items()},
        )


class _State:
    """SIMPLE 反復の状態（速度・圧力・温度・面流束）と 1 反復の実装."""

    def __init__(self, inp: NavierStokesFVMInput) -> None:
        self.inp = inp
        mesh = inp.mesh
        n = mesh.n_cells
        self.mesh = mesh
        self.u = (
            np.zeros((n, 3))
            if inp.u0 is None
            else np.asarray(inp.u0, dtype=np.float64).reshape(n, -1).copy()
        )
        if self.u.shape[1] < 3:
            self.u = np.hstack([self.u, np.zeros((n, 3 - self.u.shape[1]))])
        self.p = np.zeros(n) if inp.p0 is None else np.asarray(inp.p0, dtype=np.float64).reshape(n)
        self.T = (
            np.full(n, float(inp.T_ref))
            if inp.T0 is None
            else np.asarray(inp.T0, dtype=np.float64).reshape(n).copy()
        )
        self.blocked = (
            None if inp.solid_mask is None else np.asarray(inp.solid_mask, dtype=bool).reshape(n)
        )
        if self.blocked is not None:
            self.u[self.blocked] = 0.0
        vbcs = {k: v.velocity for k, v in inp.bcs.items()}
        self.vb: VelocityBoundaryFaces = resolve_velocity_boundary(mesh, vbcs)
        thermal = {k: v.thermal for k, v in inp.bcs.items() if v.thermal is not None}
        self.tb = resolve_boundary(mesh, thermal)
        self.bp = pressure_boundary(self.vb)
        self.bpc = pressure_boundary(self.vb, correction=True)
        self.k = np.full(n, float(inp.k_fluid))
        if inp.k_solid is not None and self.blocked is not None:
            ks = np.asarray(inp.k_solid, dtype=np.float64).reshape(n)
            self.k[self.blocked] = ks[self.blocked]
        self.drag: np.ndarray | None = None
        if inp.permeability is not None:
            K = np.asarray(inp.permeability, dtype=np.float64).reshape(n)
            if np.any(K <= 0.0):
                raise ValueError("permeability は正の値が必要（抵抗なしは inf）")
            self.drag = inp.mu / K
        self.gravity = np.asarray(inp.gravity, dtype=np.float64)
        self.mom_solver = _solver(inp.linear_solver, inp.tol_inner, inp.max_inner_iter)
        self.p_solver = _solver(inp.pressure_solver, inp.tol_inner, inp.max_inner_iter)
        self.T_solver = _solver(inp.linear_solver, inp.tol_inner, inp.max_inner_iter)
        # 内部セル境界（吐出: 速度・温度固定 + p' ピン留め、吸入: p' ピン留め）
        self.fixed_mask: np.ndarray | None = None
        self.fixed_u = np.zeros((n, 3))
        self.pinned: np.ndarray | None = None
        self.fixed_T_mask = np.zeros(n, dtype=bool)
        self.fixed_T = np.zeros(n)
        if inp.internal_bcs:
            fm = np.zeros(n, dtype=bool)
            pin = np.zeros(n, dtype=bool)
            for bc in inp.internal_bcs:
                m = np.asarray(bc.mask, dtype=bool).reshape(-1)
                if m.shape != (n,):
                    raise ValueError(f"InternalCellBC {bc.label!r} の mask は (n_cells,) が必要")
                if not np.any(m):
                    continue
                pin |= m
                if bc.kind == InternalCellBCKind.INLET:
                    fm |= m
                    self.fixed_u[m] = np.asarray(bc.velocity, dtype=np.float64)
                    if bc.temperature is not None:
                        self.fixed_T_mask |= m
                        self.fixed_T[m] = float(bc.temperature)
            if np.any(fm):
                self.fixed_mask = fm
                self.u[fm] = self.fixed_u[fm]
            if np.any(pin):
                self.pinned = pin
            if np.any(self.fixed_T_mask):
                self.T[self.fixed_T_mask] = self.fixed_T[self.fixed_T_mask]
        # 追加スカラー
        self.phi: dict[str, np.ndarray] = {}
        self.phi_b: dict[str, object] = {}
        self.phi_gamma: dict[str, float | np.ndarray] = {}
        for spec in inp.scalars:
            self.phi[spec.name] = (
                np.full(n, float(spec.phi0))
                if np.isscalar(spec.phi0)
                else np.asarray(spec.phi0, dtype=np.float64).reshape(n).copy()
            )
            self.phi_b[spec.name] = resolve_boundary(mesh, dict(spec.bcs))
            self.phi_gamma[spec.name] = (
                float(spec.diffusivity)
                if np.isscalar(spec.diffusivity)
                else np.asarray(spec.diffusivity, dtype=np.float64).reshape(n)
            )
        self.mass_flux = rhie_chow_mass_flux(
            mesh, self.u, self.p, np.zeros(n), np.zeros((n, 3)), self.vb, inp.rho, self.blocked
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        out = {"u": self.u.copy(), "T": self.T.copy()}
        for k, v in self.phi.items():
            out[k] = v.copy()
        return out

    def _buoyancy(self) -> np.ndarray | None:
        inp = self.inp
        if inp.beta == 0.0 or not np.any(self.gravity):
            return None
        return -inp.rho * inp.beta * (self.T - inp.T_ref)[:, None] * self.gravity[None, :]

    def iterate(
        self, old: dict[str, np.ndarray] | None, old2: dict[str, np.ndarray] | None = None
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """SIMPLE 系の 1 反復。``old`` は前ステップ（定常なら None）、``old2`` は前々ステップ（BDF2）."""
        inp = self.inp
        mesh = self.mesh
        n = mesh.n_cells
        nd = mesh.face_normals.shape[1]
        dt = inp.dt if old is not None else 0.0
        if old is None:
            old2 = None
        coupling = inp.coupling.lower()
        residuals: dict[str, float] = {}
        fields: dict[str, np.ndarray] = {}

        grad_p = cell_gradient_lsq(mesh, self.p, self.bp)
        if grad_p.shape[1] < 3:
            grad_p = np.hstack([grad_p, np.zeros((n, 3 - grad_p.shape[1]))])
        buoy = self._buoyancy()

        u_star = self.u.copy()
        a_p = np.zeros((n, 3))
        sum_nb = np.zeros((n, 3))
        systems: list[tuple[object, np.ndarray]] = []
        u_ref = max(float(np.max(np.abs(self.u))), float(np.max(np.abs(self.vb.velocity))), 1e-300)
        for c in range(nd):
            A, b, ap, off = assemble_momentum(
                mesh,
                component=c,
                u=self.u,
                mass_flux=self.mass_flux,
                mu=inp.mu,
                vb=self.vb,
                grad_p=grad_p,
                rho=inp.rho,
                alpha=inp.alpha_u,
                source=None if buoy is None else buoy[:, c],
                drag=self.drag,
                dt=dt,
                u_old=None if old is None else old["u"],
                blocked=self.blocked,
                u_old2=None if old2 is None else old2["u"],
                convection=inp.convection,
                limiter=inp.limiter,
                fixed_mask=self.fixed_mask,
                fixed_velocity=self.fixed_u,
            )
            scale = _momentum_scale(A, self.u[:, c], b, ap, u_ref)
            residuals[_COMPONENTS[c]] = float(np.linalg.norm(b - A @ self.u[:, c])) / scale
            fields[f"res_{_COMPONENTS[c]}"] = _residual_field(A, self.u[:, c], b, scale)
            u_star[:, c] = self.mom_solver.solve(A, b, x0=self.u[:, c])
            a_p[:, c] = ap
            sum_nb[:, c] = off
            systems.append((A, b))
        for c in range(nd, 3):
            residuals[_COMPONENTS[c]] = 0.0
            fields[f"res_{_COMPONENTS[c]}"] = np.zeros(n)
        u_star = self._enforce_fixed(u_star)

        # 圧力補正の d 係数（成分平均）
        ap_mean = np.mean(a_p[:, :nd], axis=1)
        if coupling == "simplec":
            denom = ap_mean - np.mean(sum_nb[:, :nd], axis=1)
            denom = np.where(denom > 1e-30, denom, ap_mean)
            d_cells = mesh.cell_volumes / denom
        else:
            d_cells = mesh.cell_volumes / ap_mean
        d_cells = np.where(np.isfinite(d_cells), d_cells, 0.0)

        # Rhie–Chow 面流束と圧力補正（PISO は α_p = 1 で複数回）
        a_int, a_b = pressure_correction_coefficients(mesh, d_cells, self.vb, inp.rho, self.blocked)
        alpha_p = 1.0 if coupling == "piso" else inp.alpha_p
        n_corr = inp.n_piso_correctors if coupling == "piso" else 1
        u_new = u_star
        imbalance = np.zeros(n)
        grad_p0 = grad_p
        for corr in range(n_corr):
            if corr > 0:
                # PISO 第 2 補正以降（Issa 1986）: 修正済み速度で隣接項 H(u) を再評価し、
                # 新しい圧力勾配で u** = (b − A_off u)/a_P を作ってから再度圧力補正する
                grad_p = cell_gradient_lsq(mesh, self.p, self.bp)
                if grad_p.shape[1] < 3:
                    grad_p = np.hstack([grad_p, np.zeros((n, 3 - grad_p.shape[1]))])
                u_hat = u_new.copy()
                for c in range(nd):
                    A_c, b_c = systems[c]
                    diag = np.asarray(A_c.diagonal(), dtype=np.float64)  # type: ignore[attr-defined]
                    off_u = A_c @ u_new[:, c] - diag * u_new[:, c]  # type: ignore[operator]
                    rhs = b_c + (grad_p0[:, c] - grad_p[:, c]) * mesh.cell_volumes - off_u
                    finite = np.isfinite(a_p[:, c])
                    u_hat[finite, c] = rhs[finite] / diag[finite]
                u_new = self._enforce_fixed(u_hat)
            m_star = rhie_chow_mass_flux(
                mesh, u_new, self.p, d_cells, grad_p, self.vb, inp.rho, self.blocked
            )
            A_pc, b_pc, imb = assemble_pressure_correction(
                mesh, m_star, a_int, a_b, self.vb, pinned=self.pinned
            )
            if corr == 0:
                imbalance = imb
            p_prime = self.p_solver.solve(A_pc, b_pc, x0=np.zeros(n))
            if self.blocked is not None:
                p_prime[self.blocked] = 0.0
            self.p = self.p + alpha_p * p_prime
            grad_pp = cell_gradient_lsq(mesh, p_prime, self.bpc)
            u_new = u_new.copy()
            u_new[:, :nd] -= d_cells[:, None] * grad_pp[:, :nd]
            u_new = self._enforce_fixed(u_new)
            self.mass_flux = correct_mass_flux(mesh, m_star, p_prime, a_int, a_b, self.vb)
        self.u = u_new

        # 質量残差: Σ|Σ_f ṁ_f| / (Σ_f |ṁ_f| / 2)（内部吐出・吸入セルは湧き出しなので除く）
        total = float(np.sum(np.abs(self.mass_flux))) / 2.0
        imb_free = imbalance if self.pinned is None else np.where(self.pinned, 0.0, imbalance)
        residuals["mass"] = float(np.sum(np.abs(imb_free))) / max(total, 1e-30)
        fields["res_mass"] = imb_free / mesh.cell_volumes

        # エネルギー
        if inp.solve_energy:
            rhoC = inp.rho * inp.Cp
            A_T, b_T = assemble_scalar_transport(
                mesh,
                gamma=self.k,
                bfaces=self.tb,
                mass_flux=self.mass_flux,
                source=inp.heat_source,
                rho=rhoC,
                dt=dt,
                phi_old=None if old is None else old["T"],
                phi_correction=self.T,
                phi_old2=None if old2 is None else old2["T"],
                convection=inp.convection,
                limiter=inp.limiter,
                bounded=True,
            )
            A_T, b_T = _relax(A_T, b_T, self.T, inp.alpha_T)
            if np.any(self.fixed_T_mask):
                idx = np.flatnonzero(self.fixed_T_mask)
                A_T = fix_rows(A_T, idx)
                b_T[idx] = self.fixed_T[idx]
            scale_T = max(float(np.linalg.norm(b_T)), float(np.linalg.norm(A_T @ self.T)), 1e-300)
            residuals["T"] = float(np.linalg.norm(b_T - A_T @ self.T)) / scale_T
            fields["res_T"] = _residual_field(A_T, self.T, b_T, scale_T)
            self.T = self.T_solver.solve(A_T, b_T, x0=self.T)
        else:
            residuals["T"] = 0.0
            fields["res_T"] = np.zeros(n)

        # 追加スカラー（同じ面質量流束、収束判定には含めない）
        for spec in inp.scalars:
            phi = self.phi[spec.name]
            A_s, b_s = assemble_scalar_transport(
                mesh,
                gamma=self.phi_gamma[spec.name],
                bfaces=self.phi_b[spec.name],  # type: ignore[arg-type]
                mass_flux=self.mass_flux,
                source=spec.source,
                rho=1.0,
                dt=dt,
                phi_old=None if old is None else old[spec.name],
                phi_correction=phi,
                phi_old2=None if old2 is None else old2[spec.name],
                convection=inp.convection,
                limiter=inp.limiter,
                bounded=True,
            )
            A_s, b_s = _relax(A_s, b_s, phi, spec.alpha)
            scale_s = max(float(np.linalg.norm(b_s)), float(np.linalg.norm(A_s @ phi)), 1e-300)
            residuals[spec.name] = float(np.linalg.norm(b_s - A_s @ phi)) / scale_s
            fields[f"res_{spec.name}"] = _residual_field(A_s, phi, b_s, scale_s)
            self.phi[spec.name] = self.T_solver.solve(A_s, b_s, x0=phi)
        return residuals, fields

    def _enforce_fixed(self, u: np.ndarray) -> np.ndarray:
        """固体セルは 0、内部吐出セルは指定速度に戻す."""
        if self.blocked is not None:
            u[self.blocked] = 0.0
        if self.fixed_mask is not None:
            u[self.fixed_mask] = self.fixed_u[self.fixed_mask]
        return u


def _relax(A, b: np.ndarray, x: np.ndarray, alpha: float):
    """陰的緩和: 対角を α で割り、(1−α)/α a_P x_prev を右辺へ."""
    if alpha >= 1.0:
        return A, b
    diag = np.asarray(A.diagonal(), dtype=np.float64)
    A = (A + sparse.diags(diag * (1.0 - alpha) / alpha)).tocsr()
    return A, b + diag * (1.0 - alpha) / alpha * x


__all__ = ["NavierStokesFVMProcess", "PatchBC", "MeshData"]
