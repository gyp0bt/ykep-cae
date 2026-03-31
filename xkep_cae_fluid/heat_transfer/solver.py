"""3次元非定常伝熱解析ソルバー (FDM).

等間隔直交格子上で陰的オイラー法 + 反復法により3次元伝熱方程式を解く。
ベクトル化版（ヤコビ法）とスカラー版（ガウスザイデル法）を選択可能。
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
    HeatTransferResult,
)
from xkep_cae_fluid.heat_transfer.solver_vectorized import (
    solve_jacobi_step_vectorized,
)


def _harmonic_mean(a: float, b: float) -> float:
    """調和平均（面間熱伝導率の計算用）."""
    if a + b == 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _solve_gauss_seidel_step(
    T: np.ndarray,
    T_old: np.ndarray,
    inp: HeatTransferInput,
    is_transient: bool,
) -> float:
    """ガウスザイデル法の1反復を実行し残差を返す.

    Parameters
    ----------
    T : np.ndarray
        現在の温度場 (nx, ny, nz)。in-place で更新される。
    T_old : np.ndarray
        前タイムステップの温度場（非定常時）。定常時は無視。
    inp : HeatTransferInput
        入力パラメータ
    is_transient : bool
        非定常解析かどうか

    Returns
    -------
    float
        残差の L2 ノルム
    """
    nx, ny, nz = T.shape
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz

    residual_sum = 0.0
    n_cells = nx * ny * nz

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                k_c = inp.k[i, j, k]  # 中心セルの熱伝導率

                # 各方向の隣接セルからの寄与を計算
                a_sum = 0.0  # 係数の合計
                flux_sum = 0.0  # 隣接セルからの熱流束合計

                # --- x方向 ---
                # x- 面
                if i > 0:
                    k_face = _harmonic_mean(k_c, inp.k[i - 1, j, k])
                    coeff = k_face / dx2
                    a_sum += coeff
                    flux_sum += coeff * T[i - 1, j, k]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_xm, k_c, dx, dx2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # x+ 面
                if i < nx - 1:
                    k_face = _harmonic_mean(k_c, inp.k[i + 1, j, k])
                    coeff = k_face / dx2
                    a_sum += coeff
                    flux_sum += coeff * T[i + 1, j, k]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_xp, k_c, dx, dx2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # --- y方向 ---
                # y- 面
                if j > 0:
                    k_face = _harmonic_mean(k_c, inp.k[i, j - 1, k])
                    coeff = k_face / dy2
                    a_sum += coeff
                    flux_sum += coeff * T[i, j - 1, k]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_ym, k_c, dy, dy2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # y+ 面
                if j < ny - 1:
                    k_face = _harmonic_mean(k_c, inp.k[i, j + 1, k])
                    coeff = k_face / dy2
                    a_sum += coeff
                    flux_sum += coeff * T[i, j + 1, k]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_yp, k_c, dy, dy2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # --- z方向 ---
                # z- 面
                if k > 0:
                    k_face = _harmonic_mean(k_c, inp.k[i, j, k - 1])
                    coeff = k_face / dz2
                    a_sum += coeff
                    flux_sum += coeff * T[i, j, k - 1]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_zm, k_c, dz, dz2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # z+ 面
                if k < nz - 1:
                    k_face = _harmonic_mean(k_c, inp.k[i, j, k + 1])
                    coeff = k_face / dz2
                    a_sum += coeff
                    flux_sum += coeff * T[i, j, k + 1]
                else:
                    bc_a, bc_f = _bc_coefficients(inp.bc_zp, k_c, dz, dz2)
                    a_sum += bc_a
                    flux_sum += bc_f

                # 時間項
                time_coeff = 0.0
                time_source = 0.0
                if is_transient:
                    time_coeff = inp.C[i, j, k] / inp.dt
                    time_source = time_coeff * T_old[i, j, k]

                # 発熱量
                source = inp.q[i, j, k]

                # 対角係数
                a_p = a_sum + time_coeff

                if a_p == 0.0:
                    continue

                # ガウスザイデル更新
                T_new = (flux_sum + time_source + source) / a_p
                residual_sum += (T_new - T[i, j, k]) ** 2
                T[i, j, k] = T_new

    return np.sqrt(residual_sum / n_cells) if n_cells > 0 else 0.0


def _bc_coefficients(
    bc: BoundarySpec,
    k_c: float,
    d: float,
    d2: float,
) -> tuple[float, float]:
    """境界条件の係数を返す.

    Parameters
    ----------
    bc : BoundarySpec
        境界条件仕様
    k_c : float
        境界セルの熱伝導率
    d : float
        格子幅 (dx, dy, or dz)
    d2 : float
        格子幅の2乗

    Returns
    -------
    tuple[float, float]
        (対角係数への寄与, ソース項への寄与)
    """
    if bc.condition == BoundaryCondition.DIRICHLET:
        # 温度固定: セル中心から壁面まで d/2 なので
        # q = k_c * (T_wall - T_c) / (d/2) = 2*k_c/d * (T_wall - T_c)
        # → a_bc = 2*k_c/d / d = 2*k_c/d², flux = 2*k_c/d² * T_wall
        coeff = 2.0 * k_c / d2
        return coeff, coeff * bc.value
    elif bc.condition == BoundaryCondition.NEUMANN:
        # 熱流束指定: q_flux [W/m²] がセルに流入
        # ソース項に q_flux / d を加算（体積あたり）
        return 0.0, bc.value / d
    elif bc.condition == BoundaryCondition.ROBIN:
        # 対流熱伝達: q = h(T_inf - T_surface)
        # 熱抵抗の合成: R = d/(2*k_c) + 1/h
        # 有効熱伝達率: U = 2*k_c*h / (2*k_c + h*d)
        # 体積あたり: a_bc = U/d, flux = a_bc * T_inf
        h = bc.h_conv
        if h <= 0.0:
            return 0.0, 0.0
        u_eff = 2.0 * k_c * h / (2.0 * k_c + h * d)
        coeff = u_eff / d
        return coeff, coeff * bc.T_inf
    else:
        # 断熱: ∂T/∂n = 0 → 隣接なし（ゼロ勾配）
        return 0.0, 0.0


class HeatTransferFDMProcess(SolverProcess["HeatTransferInput", "HeatTransferResult"]):
    """3次元非定常伝熱解析ソルバー (FDM, ガウスザイデル法).

    等間隔直交格子上で陰的オイラー法とガウスザイデル反復により
    3次元伝熱方程式を解く。

    対応境界条件:
    - Dirichlet（温度固定）
    - Neumann（熱流束指定）
    - Adiabatic（断熱）
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="HeatTransferFDM",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/heat-transfer-fdm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def __init__(self, *, vectorized: bool = True) -> None:
        """初期化.

        Parameters
        ----------
        vectorized : bool
            True: NumPy ベクトル化ヤコビ法（高速）
            False: スカラー版ガウスザイデル法（低速・参照実装）
        """
        super().__init__()
        self._vectorized = vectorized

    def process(self, input_data: HeatTransferInput) -> HeatTransferResult:
        """伝熱解析を実行する."""
        t_start = time.perf_counter()
        inp = input_data

        # 温度場の初期化（書き込み用コピー）
        T = inp.T0.copy().astype(np.float64)
        nx, ny, nz = T.shape

        if inp.is_transient:
            return self._solve_transient(T, inp, t_start)
        else:
            return self._solve_steady(T, inp, t_start)

    def _solve_steady(
        self,
        T: np.ndarray,
        inp: HeatTransferInput,
        t_start: float,
    ) -> HeatTransferResult:
        """定常解析."""
        T_dummy = np.zeros_like(T)
        residuals: list[float] = []

        converged = False
        for _iteration in range(inp.max_iter):
            if self._vectorized:
                T_new, res = solve_jacobi_step_vectorized(T, T_dummy, inp, is_transient=False)
                T[:] = T_new
            else:
                res = _solve_gauss_seidel_step(T, T_dummy, inp, is_transient=False)
            residuals.append(res)
            if res < inp.tol:
                converged = True
                break

        elapsed = time.perf_counter() - t_start
        return HeatTransferResult(
            T=T,
            converged=converged,
            n_timesteps=0,
            iteration_counts=(len(residuals),),
            residual_history=(tuple(residuals),),
            time_history=(),
            T_history=(),
            elapsed_seconds=elapsed,
        )

    def _solve_transient(
        self,
        T: np.ndarray,
        inp: HeatTransferInput,
        t_start: float,
    ) -> HeatTransferResult:
        """非定常解析."""
        current_time = 0.0
        n_steps = int(np.ceil(inp.t_end / inp.dt))
        iteration_counts: list[int] = []
        residual_history: list[tuple[float, ...]] = []
        time_history: list[float] = []
        T_history: list[np.ndarray] = []
        converged = True

        for step in range(n_steps):
            current_time += inp.dt
            T_old = T.copy()

            step_residuals: list[float] = []
            step_converged = False
            for _iteration in range(inp.max_iter):
                if self._vectorized:
                    T_new, res = solve_jacobi_step_vectorized(T, T_old, inp, is_transient=True)
                    T[:] = T_new
                else:
                    res = _solve_gauss_seidel_step(T, T_old, inp, is_transient=True)
                step_residuals.append(res)
                if res < inp.tol:
                    step_converged = True
                    break

            iteration_counts.append(len(step_residuals))
            residual_history.append(tuple(step_residuals))

            if not step_converged:
                converged = False

            # 出力間隔に基づくスナップショット
            if (step + 1) % inp.output_interval == 0 or step == n_steps - 1:
                time_history.append(current_time)
                T_history.append(T.copy())

        elapsed = time.perf_counter() - t_start
        return HeatTransferResult(
            T=T,
            converged=converged,
            n_timesteps=n_steps,
            iteration_counts=tuple(iteration_counts),
            residual_history=tuple(residual_history),
            time_history=tuple(time_history),
            T_history=tuple(T_history),
            elapsed_seconds=elapsed,
        )
