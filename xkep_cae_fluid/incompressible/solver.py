"""非圧縮 Navier–Stokes ソルバー（面ベース FVM、同位置格子 SIMPLE + Rhie–Chow）.

:class:`NavierStokesFVMProcess` は :mod:`xkep_cae_fluid.fvm`（:mod:`~xkep_cae_fluid.fvm.momentum` の
運動量・圧力連成カーネル、:mod:`~xkep_cae_fluid.fvm.assembly` のスカラー輸送）を組み合わせた
薄い方程式ファミリー層。構造格子 / polyMesh / .inp の ``MeshData`` を同じ経路で解く。

1 反復（SIMPLE）:

1. 圧力勾配 ∇p（Green–Gauss、OUTLET は Dirichlet、他はゼロ勾配）
2. 運動量 3 成分（対流 1 次風上、拡散 + 非直交補正、浮力・抵抗、陰的緩和 α_u）→ u*
3. Rhie–Chow の面質量流束 ṁ*、圧力補正 Σ a_f (p'_P − p'_N) = −Σ ṁ*
4. p += α_p p'、u −= (V/a_P) ∇p'、ṁ −= a_f Δp'
5. エネルギー（対流 ṁ + 拡散 k、固体は k_solid、陰的緩和 α_T）

Brinkman 抵抗（透過率 K）は運動量の対角に μ/K V、Boussinesq 浮力は −ρ β (T − T_ref) g V。
設計は ``docs/design/navier-stokes-fvm.md``。
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm import (
    PatchBC,
    assemble_scalar_transport,
    cell_gradient_lsq,
    make_linear_solver,
    relative_residual,
    resolve_boundary,
)
from xkep_cae_fluid.fvm.momentum import (
    VelocityBoundaryFaces,
    assemble_momentum,
    assemble_pressure_correction,
    correct_mass_flux,
    pressure_boundary,
    pressure_correction_coefficients,
    resolve_velocity_boundary,
    rhie_chow_mass_flux,
)
from xkep_cae_fluid.incompressible.data import NavierStokesFVMInput, NavierStokesFVMResult

logger = logging.getLogger(__name__)

_COMPONENTS = ("u", "v", "w")


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
        if inp.coupling.lower() not in ("simple", "simplec"):
            raise ValueError(f"coupling は simple / simplec のみ: {inp.coupling!r}")
        state = _State(inp)

        residual_history: dict[str, list[float]] = {k: [] for k in (*_COMPONENTS, "T", "mass")}
        residual_fields: dict[str, np.ndarray] = {}
        n_outer = 0
        converged = False
        times: list[float] = []

        keys = (*_COMPONENTS, "mass", "T") if inp.solve_energy else (*_COMPONENTS, "mass")

        if not inp.is_transient:
            for it in range(inp.max_outer_iter):
                res, residual_fields = state.iterate(None)
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
            for step in range(n_steps):
                t += inp.dt
                old = state.snapshot()
                step_ok = False
                for it in range(inp.max_outer_iter):
                    res, residual_fields = state.iterate(old)
                    n_outer += 1
                    for k, v in res.items():
                        residual_history[k].append(v)
                    if it >= 1 and max(res[k] for k in keys) < inp.tol:
                        step_ok = True
                        break
                converged &= step_ok
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
        self.mass_flux = rhie_chow_mass_flux(
            mesh, self.u, self.p, np.zeros(n), np.zeros((n, 3)), self.vb, inp.rho, self.blocked
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        return {"u": self.u.copy(), "T": self.T.copy()}

    def _buoyancy(self) -> np.ndarray | None:
        inp = self.inp
        if inp.beta == 0.0 or not np.any(self.gravity):
            return None
        return -inp.rho * inp.beta * (self.T - inp.T_ref)[:, None] * self.gravity[None, :]

    def iterate(
        self, old: dict[str, np.ndarray] | None
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        inp = self.inp
        mesh = self.mesh
        n = mesh.n_cells
        nd = mesh.face_normals.shape[1]
        dt = inp.dt if old is not None else 0.0
        residuals: dict[str, float] = {}
        fields: dict[str, np.ndarray] = {}

        grad_p = cell_gradient_lsq(mesh, self.p, self.bp)
        if grad_p.shape[1] < 3:
            grad_p = np.hstack([grad_p, np.zeros((n, 3 - grad_p.shape[1]))])
        buoy = self._buoyancy()

        u_star = self.u.copy()
        a_p = np.zeros((n, 3))
        sum_nb = np.zeros((n, 3))
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
            )
            scale = _momentum_scale(A, self.u[:, c], b, ap, u_ref)
            residuals[_COMPONENTS[c]] = float(np.linalg.norm(b - A @ self.u[:, c])) / scale
            fields[f"res_{_COMPONENTS[c]}"] = _residual_field(A, self.u[:, c], b, scale)
            u_star[:, c] = self.mom_solver.solve(A, b, x0=self.u[:, c])
            a_p[:, c] = ap
            sum_nb[:, c] = off
        for c in range(nd, 3):
            residuals[_COMPONENTS[c]] = 0.0
            fields[f"res_{_COMPONENTS[c]}"] = np.zeros(n)
        if self.blocked is not None:
            u_star[self.blocked] = 0.0

        # 圧力補正の d 係数（成分平均）
        ap_mean = np.mean(a_p[:, :nd], axis=1)
        if inp.coupling.lower() == "simplec":
            denom = ap_mean - np.mean(sum_nb[:, :nd], axis=1)
            denom = np.where(denom > 1e-30, denom, ap_mean)
            d_cells = mesh.cell_volumes / denom
        else:
            d_cells = mesh.cell_volumes / ap_mean
        d_cells = np.where(np.isfinite(d_cells), d_cells, 0.0)

        # Rhie–Chow 面流束と圧力補正
        m_star = rhie_chow_mass_flux(
            mesh, u_star, self.p, d_cells, grad_p, self.vb, inp.rho, self.blocked
        )
        a_int, a_b = pressure_correction_coefficients(mesh, d_cells, self.vb, inp.rho, self.blocked)
        A_pc, b_pc, imbalance = assemble_pressure_correction(mesh, m_star, a_int, a_b, self.vb)
        p_prime = self.p_solver.solve(A_pc, b_pc, x0=np.zeros(n))
        if self.blocked is not None:
            p_prime[self.blocked] = 0.0

        # 修正
        self.p = self.p + inp.alpha_p * p_prime
        grad_pp = cell_gradient_lsq(mesh, p_prime, self.bpc)
        u_new = u_star.copy()
        u_new[:, :nd] -= d_cells[:, None] * grad_pp[:, :nd]
        if self.blocked is not None:
            u_new[self.blocked] = 0.0
        self.u = u_new
        self.mass_flux = correct_mass_flux(mesh, m_star, p_prime, a_int, a_b, self.vb)

        # 質量残差: Σ|Σ_f ṁ_f| / (Σ_f |ṁ_f| / 2)
        total = float(np.sum(np.abs(self.mass_flux))) / 2.0
        residuals["mass"] = float(np.sum(np.abs(imbalance))) / max(total, 1e-30)
        fields["res_mass"] = imbalance / mesh.cell_volumes

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
            )
            if inp.alpha_T < 1.0:
                from scipy import sparse

                diag = np.asarray(A_T.diagonal(), dtype=np.float64)
                A_T = (A_T + sparse.diags(diag * (1.0 - inp.alpha_T) / inp.alpha_T)).tocsr()
                b_T = b_T + diag * (1.0 - inp.alpha_T) / inp.alpha_T * self.T
            residuals["T"] = relative_residual(A_T, self.T, b_T)
            fields["res_T"] = _residual_field(
                A_T, self.T, b_T, max(float(np.linalg.norm(b_T)), 1e-300)
            )
            self.T = self.T_solver.solve(A_T, b_T, x0=self.T)
        else:
            residuals["T"] = 0.0
            fields["res_T"] = np.zeros(n)
        return residuals, fields


__all__ = ["NavierStokesFVMProcess", "PatchBC", "MeshData"]
