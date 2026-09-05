"""伝熱ソルバー（面ベース FVM、``MeshData`` 上）.

:class:`HeatTransferFVMProcess` は :mod:`xkep_cae_fluid.fvm` の低レイヤーを組み合わせて
熱伝導方程式 ρC ∂T/∂t − ∇·(k∇T) = q を解く薄い方程式ファミリー層で、
構造格子 / polyMesh / .inp 由来の ``MeshData`` を同じ経路で扱う。
構造格子上では既存の :class:`~xkep_cae_fluid.heat_transfer.solver.HeatTransferFDMProcess`
と同じ解を返す（``tests/test_heat_transfer_fvm.py`` で回帰）。

境界条件はパッチ名 → :class:`~xkep_cae_fluid.fvm.PatchBC`:

- ``PatchBC.dirichlet(T_wall)`` 温度固定
- ``PatchBC.neumann(q)`` 熱流束 [W/m²]（正 = 流入）
- ``PatchBC.robin(h, T_inf)`` 対流熱伝達
- 未指定パッチは断熱
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm import (
    PatchBC,
    assemble_scalar_transport,
    make_linear_solver,
    resolve_boundary,
    solve_corrected,
)


def _cell_array(value: float | np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    arr = arr.reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} は長さ n_cells={n} の配列かスカラーが必要: {arr.shape}")
    return arr


@dataclass(frozen=True)
class HeatTransferFVMInput:
    """面ベース FVM 伝熱の入力.

    Parameters
    ----------
    mesh : MeshData
        面情報と ``boundary_patches`` を持つメッシュ
    conductivity : float | np.ndarray
        熱伝導率 k [W/(m·K)]（スカラーかセル配列）
    T0 : np.ndarray
        初期温度 (n_cells,) [K]
    heat_capacity : float | np.ndarray
        体積熱容量 ρC [J/(m³·K)]（非定常のみ使う）
    heat_source : np.ndarray | None
        体積発熱 q (n_cells,) [W/m³]
    bcs : Mapping[str, PatchBC]
        パッチ名 → 境界条件（未指定は断熱）
    dt, t_end : float
        非定常なら dt > 0 [s]
    output_interval : int
        ``T_history`` に残す間隔（タイムステップ数）
    linear_solver : str
        ``direct`` / ``bicgstab`` / ``amg``
    tol, max_iter : 反復解法の設定
    max_nonorthogonal_iter : int
        非直交メッシュでの遅延補正の最大反復回数（直交メッシュでは 1 回）
    """

    mesh: MeshData
    conductivity: float | np.ndarray
    T0: np.ndarray
    heat_capacity: float | np.ndarray = 1.0
    heat_source: np.ndarray | None = None
    bcs: Mapping[str, PatchBC] = field(default_factory=dict)
    dt: float = 0.0
    t_end: float = 0.0
    output_interval: int = 1
    linear_solver: str = "bicgstab"
    tol: float = 1e-8
    max_iter: int = 500
    max_nonorthogonal_iter: int = 20

    @property
    def is_transient(self) -> bool:
        return self.dt > 0.0


@dataclass(frozen=True)
class HeatTransferFVMResult:
    """面ベース FVM 伝熱の出力.

    Parameters
    ----------
    T : np.ndarray
        最終温度 (n_cells,) [K]
    converged : bool
        線形系の相対残差が許容値内か（非定常は全ステップ）
    n_timesteps : int
        実行タイムステップ数（定常は 0）
    residual_history : tuple[float, ...]
        各ソルブの相対残差 ‖b − A T‖/‖b‖
    time_history, T_history :
        ``output_interval`` ごとの時刻と温度のスナップショット
    elapsed_seconds : float
    residual_fields : dict[str, np.ndarray]
        定常解の残差マップ ``res_T`` = |b − A T| / ‖b‖（非定常は空）
    """

    T: np.ndarray
    converged: bool
    n_timesteps: int = 0
    residual_history: tuple[float, ...] = ()
    time_history: tuple[float, ...] = ()
    T_history: tuple[np.ndarray, ...] = ()
    elapsed_seconds: float = 0.0
    residual_fields: dict[str, np.ndarray] = field(default_factory=dict)


class HeatTransferFVMProcess(SolverProcess["HeatTransferFVMInput", "HeatTransferFVMResult"]):
    """熱伝導（拡散中心差分 + 陰的 Euler）を ``MeshData`` 上で解く SolverProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="HeatTransferFVM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/heat-transfer-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: HeatTransferFVMInput) -> HeatTransferFVMResult:
        t0 = time.perf_counter()
        inp = input_data
        mesh = inp.mesh
        n = mesh.n_cells
        T = _cell_array(inp.T0, n, "T0").copy()
        k = _cell_array(inp.conductivity, n, "conductivity")
        if np.any(k <= 0.0):
            raise ValueError("conductivity は正の値が必要")
        C = _cell_array(inp.heat_capacity, n, "heat_capacity")
        q = None if inp.heat_source is None else _cell_array(inp.heat_source, n, "heat_source")
        if inp.is_transient and np.any(C <= 0.0):
            raise ValueError("非定常解析では heat_capacity が正の値である必要があります")

        bfaces = resolve_boundary(mesh, inp.bcs)
        solver = make_linear_solver(
            inp.linear_solver,
            **(
                {}
                if inp.linear_solver.lower() == "direct"
                else {"tol": inp.tol, "maxiter": inp.max_iter}
            ),
        )
        tol_ok = max(inp.tol * 10.0, 1e-6)

        def solve_step(T_old: np.ndarray | None) -> tuple[np.ndarray, float]:
            def build(T_corr: np.ndarray | None):
                return assemble_scalar_transport(
                    mesh,
                    gamma=k,
                    bfaces=bfaces,
                    source=q,
                    rho=C,
                    dt=inp.dt if T_old is not None else 0.0,
                    phi_old=T_old,
                    phi_correction=T_corr,
                )

            x, resid, _n = solve_corrected(
                mesh, build, solver, T, max_iter=inp.max_nonorthogonal_iter, tol=inp.tol
            )
            return x, resid

        if not inp.is_transient:
            T, resid = solve_step(None)
            A, b = assemble_scalar_transport(
                mesh, gamma=k, bfaces=bfaces, source=q, phi_correction=T
            )
            r = np.abs(b - A @ T)
            b_norm = float(np.linalg.norm(b))
            if b_norm >= 1e-30:
                r = r / b_norm
            return HeatTransferFVMResult(
                T=T,
                converged=bool(resid < tol_ok),
                n_timesteps=0,
                residual_history=(resid,),
                elapsed_seconds=time.perf_counter() - t0,
                residual_fields={"res_T": np.asarray(r)},
            )

        n_steps = int(np.ceil(inp.t_end / inp.dt))
        interval = max(int(inp.output_interval), 1)
        residuals: list[float] = []
        times: list[float] = []
        history: list[np.ndarray] = []
        all_ok = True
        t = 0.0
        for step in range(n_steps):
            t += inp.dt
            T, resid = solve_step(T)
            residuals.append(resid)
            all_ok &= resid < tol_ok
            if (step + 1) % interval == 0 or step == n_steps - 1:
                times.append(t)
                history.append(T.copy())
        return HeatTransferFVMResult(
            T=T,
            converged=bool(all_ok),
            n_timesteps=n_steps,
            residual_history=tuple(residuals),
            time_history=tuple(times),
            T_history=tuple(history),
            elapsed_seconds=time.perf_counter() - t0,
        )
