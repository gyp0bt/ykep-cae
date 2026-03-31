"""Strategy Protocol 定義（流体解析向け）.

ソルバー内部の直交する振る舞い軸を Protocol で規定する。
具象実装は各 strategy モジュールに配置する。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp


@runtime_checkable
class ConvectionSchemeStrategy(Protocol):
    """対流項の離散化スキームを規定する.

    実装候補:
    - Upwind: 1次風上差分
    - CentralDifference: 中心差分
    - QUICK: 2次風上
    - TVD: Total Variation Diminishing (van Leer, Superbee 等)
    - WENO: Weighted Essentially Non-Oscillatory
    """

    def flux(
        self,
        phi: np.ndarray,
        velocity: np.ndarray,
        mesh: object,
    ) -> np.ndarray:
        """対流フラックスを計算.

        Args:
            phi: スカラー場 (n_cells,)
            velocity: 速度場 (n_cells, ndim)
            mesh: MeshData

        Returns:
            convective_flux: (n_cells,)
        """
        ...

    def matrix_coefficients(
        self,
        velocity: np.ndarray,
        mesh: object,
    ) -> sp.csr_matrix:
        """対流項の係数行列を構築.

        Returns:
            A_conv: (n_cells, n_cells) CSR
        """
        ...


@runtime_checkable
class DiffusionSchemeStrategy(Protocol):
    """拡散項の離散化スキームを規定する.

    実装候補:
    - CentralDiffusion: 中心差分拡散
    - CorrectedDiffusion: 非直交補正付き
    """

    def flux(
        self,
        phi: np.ndarray,
        diffusivity: float | np.ndarray,
        mesh: object,
    ) -> np.ndarray:
        """拡散フラックスを計算.

        Returns:
            diffusive_flux: (n_cells,)
        """
        ...

    def matrix_coefficients(
        self,
        diffusivity: float | np.ndarray,
        mesh: object,
    ) -> sp.csr_matrix:
        """拡散項の係数行列を構築.

        Returns:
            A_diff: (n_cells, n_cells) CSR
        """
        ...


@runtime_checkable
class TimeIntegrationStrategy(Protocol):
    """時間積分方法を規定する.

    実装候補:
    - Steady: 定常（擬似時間進行）
    - EulerImplicit: 1次陰的オイラー
    - CrankNicolson: Crank-Nicolson（2次精度）
    - BDF2: 2次後退差分
    """

    def temporal_contribution(
        self,
        phi_old: np.ndarray,
        cell_volumes: np.ndarray,
        dt: float,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """時間項の係数行列と右辺ベクトルを返す.

        Returns:
            (A_time, b_time)
        """
        ...


@runtime_checkable
class TurbulenceModelStrategy(Protocol):
    """乱流モデルを規定する.

    実装候補:
    - Laminar: 層流（乱流モデルなし）
    - KepsilonStandard: 標準 k-epsilon
    - KepsilonRNG: RNG k-epsilon
    - KomegaSST: k-omega SST
    - SpalartAllmaras: SA 1方程式モデル
    - LES_Smagorinsky: Smagorinsky LES
    """

    def effective_viscosity(
        self,
        velocity: np.ndarray,
        turbulent_ke: np.ndarray | None,
        turbulent_epsilon: np.ndarray | None,
        fluid_viscosity: float,
    ) -> np.ndarray:
        """有効粘性係数（分子粘性 + 乱流粘性）を計算.

        Returns:
            mu_eff: (n_cells,)
        """
        ...

    def transport_equations(
        self,
        velocity: np.ndarray,
        turbulent_ke: np.ndarray,
        turbulent_epsilon: np.ndarray,
        mesh: object,
        fluid: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """乱流輸送方程式の右辺を計算.

        Returns:
            (rhs_k, rhs_epsilon)
        """
        ...


@runtime_checkable
class PressureVelocityCouplingStrategy(Protocol):
    """圧力-速度連成法を規定する.

    実装候補:
    - SIMPLE: Semi-Implicit Method for Pressure-Linked Equations
    - SIMPLEC: SIMPLE-Consistent
    - PISO: Pressure Implicit with Splitting of Operators
    - CoupledSolver: 完全連成法
    """

    def pressure_equation(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        A_momentum: sp.csr_matrix,
        mesh: object,
        boundary: object,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """圧力方程式（ポアソン方程式）の係数行列と右辺を構築.

        Returns:
            (A_pressure, b_pressure)
        """
        ...

    def correct_velocity(
        self,
        velocity: np.ndarray,
        pressure_correction: np.ndarray,
        A_momentum: sp.csr_matrix,
        mesh: object,
    ) -> np.ndarray:
        """圧力補正に基づく速度補正.

        Returns:
            velocity_corrected: (n_cells, ndim)
        """
        ...


@runtime_checkable
class LinearSolverStrategy(Protocol):
    """線形連立方程式の解法を規定する.

    実装候補:
    - DirectSolver: spsolve 直接法
    - GMRES: 前処理付き GMRES
    - BiCGSTAB: 前処理付き BiCGSTAB
    - AMG: 代数的マルチグリッド
    """

    def solve(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Ax = b を解く.

        Args:
            A: 係数行列 (CSR)
            b: 右辺ベクトル

        Returns:
            解ベクトル x
        """
        ...
