"""Numba JIT 高速化ガウスザイデル法ソルバー.

Python の3重ループをネイティブコードにコンパイルし、
スカラーガウスザイデル法を大幅に高速化する。

使用には numba パッケージが必要:
    pip install 'xkep-cae-fluid[fast]'
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# 境界条件タイプの定数（Enum はNumbaで使えないため数値化）
BC_DIRICHLET = 0
BC_NEUMANN = 1
BC_ADIABATIC = 2
BC_ROBIN = 3


def _bc_to_int(condition_value: str) -> int:
    """BoundaryCondition enum の値を整数に変換."""
    mapping = {
        "dirichlet": BC_DIRICHLET,
        "neumann": BC_NEUMANN,
        "adiabatic": BC_ADIABATIC,
        "robin": BC_ROBIN,
    }
    return mapping.get(condition_value, BC_ADIABATIC)


def _pack_bc(bc) -> tuple[int, float, float, float]:
    """BoundarySpec を Numba 互換のタプルにパック."""
    return (_bc_to_int(bc.condition.value), bc.value, bc.h_conv, bc.T_inf)


if HAS_NUMBA:

    @njit(cache=True)
    def _bc_coefficients_numba(
        bc_type: int,
        bc_value: float,
        bc_h_conv: float,
        bc_T_inf: float,
        k_c: float,
        d: float,
        d2: float,
    ) -> tuple[float, float]:
        """境界条件の係数を返す（Numba版）."""
        if bc_type == BC_DIRICHLET:
            coeff = 2.0 * k_c / d2
            return coeff, coeff * bc_value
        elif bc_type == BC_NEUMANN:
            return 0.0, bc_value / d
        elif bc_type == BC_ROBIN:
            h = bc_h_conv
            if h <= 0.0:
                return 0.0, 0.0
            u_eff = 2.0 * k_c * h / (2.0 * k_c + h * d)
            coeff = u_eff / d
            return coeff, coeff * bc_T_inf
        else:
            return 0.0, 0.0

    @njit(cache=True)
    def _harmonic_mean_numba(a: float, b: float) -> float:
        """調和平均."""
        s = a + b
        if s == 0.0:
            return 0.0
        return 2.0 * a * b / s

    @njit(cache=True)
    def _gauss_seidel_step_numba(
        T: np.ndarray,
        T_old: np.ndarray,
        k: np.ndarray,
        C: np.ndarray,
        q: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        is_transient: bool,
        bc_xm: tuple[int, float, float, float],
        bc_xp: tuple[int, float, float, float],
        bc_ym: tuple[int, float, float, float],
        bc_yp: tuple[int, float, float, float],
        bc_zm: tuple[int, float, float, float],
        bc_zp: tuple[int, float, float, float],
    ) -> float:
        """ガウスザイデル法の1反復（Numba JIT版）.

        Parameters
        ----------
        T : np.ndarray
            温度場 (nx, ny, nz)。in-place で更新される。
        T_old : np.ndarray
            前タイムステップの温度場。
        k, C, q : np.ndarray
            材料物性・発熱量
        dx, dy, dz : float
            格子幅
        dt : float
            時間刻み
        is_transient : bool
            非定常フラグ
        bc_xm..bc_zp : tuple
            各面の境界条件 (type, value, h_conv, T_inf)

        Returns
        -------
        float
            残差の L2 ノルム
        """
        nx, ny, nz = T.shape
        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz

        residual_sum = 0.0
        n_cells = nx * ny * nz

        for i in range(nx):
            for j in range(ny):
                for ki in range(nz):
                    k_c = k[i, j, ki]
                    a_sum = 0.0
                    flux_sum = 0.0

                    # x- 面
                    if i > 0:
                        k_face = _harmonic_mean_numba(k_c, k[i - 1, j, ki])
                        coeff = k_face / dx2
                        a_sum += coeff
                        flux_sum += coeff * T[i - 1, j, ki]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_xm[0], bc_xm[1], bc_xm[2], bc_xm[3], k_c, dx, dx2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # x+ 面
                    if i < nx - 1:
                        k_face = _harmonic_mean_numba(k_c, k[i + 1, j, ki])
                        coeff = k_face / dx2
                        a_sum += coeff
                        flux_sum += coeff * T[i + 1, j, ki]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_xp[0], bc_xp[1], bc_xp[2], bc_xp[3], k_c, dx, dx2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # y- 面
                    if j > 0:
                        k_face = _harmonic_mean_numba(k_c, k[i, j - 1, ki])
                        coeff = k_face / dy2
                        a_sum += coeff
                        flux_sum += coeff * T[i, j - 1, ki]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_ym[0], bc_ym[1], bc_ym[2], bc_ym[3], k_c, dy, dy2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # y+ 面
                    if j < ny - 1:
                        k_face = _harmonic_mean_numba(k_c, k[i, j + 1, ki])
                        coeff = k_face / dy2
                        a_sum += coeff
                        flux_sum += coeff * T[i, j + 1, ki]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_yp[0], bc_yp[1], bc_yp[2], bc_yp[3], k_c, dy, dy2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # z- 面
                    if ki > 0:
                        k_face = _harmonic_mean_numba(k_c, k[i, j, ki - 1])
                        coeff = k_face / dz2
                        a_sum += coeff
                        flux_sum += coeff * T[i, j, ki - 1]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_zm[0], bc_zm[1], bc_zm[2], bc_zm[3], k_c, dz, dz2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # z+ 面
                    if ki < nz - 1:
                        k_face = _harmonic_mean_numba(k_c, k[i, j, ki + 1])
                        coeff = k_face / dz2
                        a_sum += coeff
                        flux_sum += coeff * T[i, j, ki + 1]
                    else:
                        bc_a, bc_f = _bc_coefficients_numba(
                            bc_zp[0], bc_zp[1], bc_zp[2], bc_zp[3], k_c, dz, dz2
                        )
                        a_sum += bc_a
                        flux_sum += bc_f

                    # 時間項
                    time_coeff = 0.0
                    time_source = 0.0
                    if is_transient:
                        time_coeff = C[i, j, ki] / dt
                        time_source = time_coeff * T_old[i, j, ki]

                    source = q[i, j, ki]
                    a_p = a_sum + time_coeff

                    if a_p == 0.0:
                        continue

                    T_new = (flux_sum + time_source + source) / a_p
                    residual_sum += (T_new - T[i, j, ki]) ** 2
                    T[i, j, ki] = T_new

        if n_cells > 0:
            return (residual_sum / n_cells) ** 0.5
        return 0.0


def solve_gauss_seidel_step_numba(
    T: np.ndarray,
    T_old: np.ndarray,
    inp,
    is_transient: bool,
) -> float:
    """Numba JIT 版ガウスザイデル反復のラッパー.

    HeatTransferInput からパラメータを展開し、Numba カーネルを呼び出す。

    Raises
    ------
    ImportError
        numba がインストールされていない場合
    """
    if not HAS_NUMBA:
        msg = "Numba が必要です。pip install 'xkep-cae-fluid[fast]' でインストールしてください。"
        raise ImportError(msg)

    bc_xm = _pack_bc(inp.bc_xm)
    bc_xp = _pack_bc(inp.bc_xp)
    bc_ym = _pack_bc(inp.bc_ym)
    bc_yp = _pack_bc(inp.bc_yp)
    bc_zm = _pack_bc(inp.bc_zm)
    bc_zp = _pack_bc(inp.bc_zp)

    return _gauss_seidel_step_numba(
        T,
        T_old,
        inp.k.astype(np.float64),
        inp.C.astype(np.float64),
        inp.q.astype(np.float64),
        inp.dx,
        inp.dy,
        inp.dz,
        inp.dt if inp.dt > 0 else 1.0,
        is_transient,
        bc_xm,
        bc_xp,
        bc_ym,
        bc_yp,
        bc_zm,
        bc_zp,
    )
