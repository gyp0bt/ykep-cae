"""3次元自然対流ソルバー (FDM + SIMPLE法).

等間隔直交格子上で SIMPLE 法による圧力-速度連成と
Boussinesq 近似による浮力項を使い、自然対流問題を解く。
固体-流体練成伝熱（Conjugate Heat Transfer）に対応。
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.natural_convection.assembly import (
    build_energy_system,
    build_momentum_system,
    build_pressure_correction_system_rc,
    compute_face_mass_residual,
    compute_rhie_chow_face_velocity,
)
from xkep_cae_fluid.natural_convection.data import (
    NaturalConvectionInput,
    NaturalConvectionResult,
)
from xkep_cae_fluid.scalar_transport.assembly import build_scalar_system
from xkep_cae_fluid.scalar_transport.data import (
    ExtraScalarSpec,
    ScalarFieldSpec,
    ScalarTransportInput,
)

logger = logging.getLogger(__name__)

# --- AMG キャッシュ（圧力方程式用） ---


class _PressureAMGCache:
    """圧力補正方程式用 AMG 階層キャッシュ.

    スパースパターンが不変の間は AMG セットアップを再利用する。
    """

    def __init__(self) -> None:
        self._ml = None
        self._indptr_hash: int | None = None
        self._indices_hash: int | None = None

    def get_solver(self, A_csr):
        """キャッシュ済み AMG ソルバーを返す."""
        import pyamg

        indptr_h = hash(A_csr.indptr.data.tobytes())
        indices_h = hash(A_csr.indices.data.tobytes())

        if self._ml is None or self._indptr_hash != indptr_h or self._indices_hash != indices_h:
            self._ml = pyamg.ruge_stuben_solver(A_csr)
            self._indptr_hash = indptr_h
            self._indices_hash = indices_h

        return self._ml

    def clear(self) -> None:
        self._ml = None
        self._indptr_hash = None
        self._indices_hash = None


_pressure_amg_cache = _PressureAMGCache()


def _solve_linear(A, b, x0=None, tol=1e-6, maxiter=50):
    """BiCGSTAB + ILU前処理で線形方程式を解く."""
    try:
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-4)
        M = spla.LinearOperator(A.shape, matvec=ilu.solve)
    except Exception:
        M = None

    if x0 is None:
        x0 = np.zeros(b.shape[0])

    x, info = spla.bicgstab(A, b, x0=x0, M=M, rtol=tol, maxiter=maxiter)
    return x


def _solve_pressure_amg(A, b, x0=None, tol=1e-6, maxiter=100):
    """AMG前処理 + CG で圧力補正方程式を解く.

    圧力補正方程式はラプラシアン型の対称正定値行列なので
    CG（共役勾配法）が最適。AMG を前処理に使う。
    """
    A_csr = A.tocsr()
    ml = _pressure_amg_cache.get_solver(A_csr)
    M = ml.aspreconditioner()

    if x0 is None:
        x0 = np.zeros(b.shape[0])

    x, info = spla.cg(A_csr, b, x0=x0, M=M, rtol=tol, maxiter=maxiter)
    return x


def _compute_residual_norm(A, x, b):
    """残差ノルムを計算."""
    r = b - A @ x
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-30:
        return np.linalg.norm(r)
    return np.linalg.norm(r) / b_norm


def _simple_convergence_residual(residuals: dict[str, float]) -> float:
    """SIMPLE 収束判定用の残差を返す.

    温度 (T) はエネルギー方程式の RHS が ρCp/dt × T_old で支配されるため
    相対残差が不当に大きくなる。SIMPLE の収束判定には速度・圧力・質量のみを使う。
    """
    keys = ["u", "v", "w", "p", "mass"]
    return max(residuals.get(k, 0.0) for k in keys)


def _correct_velocity(
    inp: NaturalConvectionInput,
    u_star: np.ndarray,
    v_star: np.ndarray,
    w_star: np.ndarray,
    p_prime: np.ndarray,
    a_P_u: np.ndarray,
    a_P_v: np.ndarray,
    a_P_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """速度場を圧力補正で更新.

    u = u* - (V/a_P) * ∂p'/∂x
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    rho = inp.rho

    p_prime_3d = p_prime.reshape(nx, ny, nz)
    a_P_u_3d = a_P_u.reshape(nx, ny, nz)
    a_P_v_3d = a_P_v.reshape(nx, ny, nz)
    a_P_w_3d = a_P_w.reshape(nx, ny, nz)

    u_corr = u_star.copy()
    v_corr = v_star.copy()
    w_corr = w_star.copy()

    # ∂p'/∂x — 中心差分
    dp_dx = np.zeros((nx, ny, nz))
    if nx > 2:
        dp_dx[1:-1, :, :] = (p_prime_3d[2:, :, :] - p_prime_3d[:-2, :, :]) / (2.0 * dx)
    if nx > 1:
        dp_dx[0, :, :] = (p_prime_3d[1, :, :] - p_prime_3d[0, :, :]) / dx
        dp_dx[-1, :, :] = (p_prime_3d[-1, :, :] - p_prime_3d[-2, :, :]) / dx

    dp_dy = np.zeros((nx, ny, nz))
    if ny > 2:
        dp_dy[:, 1:-1, :] = (p_prime_3d[:, 2:, :] - p_prime_3d[:, :-2, :]) / (2.0 * dy)
    if ny > 1:
        dp_dy[:, 0, :] = (p_prime_3d[:, 1, :] - p_prime_3d[:, 0, :]) / dy
        dp_dy[:, -1, :] = (p_prime_3d[:, -1, :] - p_prime_3d[:, -2, :]) / dy

    dp_dz = np.zeros((nx, ny, nz))
    if nz > 2:
        dp_dz[:, :, 1:-1] = (p_prime_3d[:, :, 2:] - p_prime_3d[:, :, :-2]) / (2.0 * dz)
    if nz > 1:
        dp_dz[:, :, 0] = (p_prime_3d[:, :, 1] - p_prime_3d[:, :, 0]) / dz
        dp_dz[:, :, -1] = (p_prime_3d[:, :, -1] - p_prime_3d[:, :, -2]) / dz

    # d = rho / a_P
    safe_aP_u = np.where(a_P_u_3d > 0, a_P_u_3d, 1.0)
    safe_aP_v = np.where(a_P_v_3d > 0, a_P_v_3d, 1.0)
    safe_aP_w = np.where(a_P_w_3d > 0, a_P_w_3d, 1.0)

    u_corr -= (rho / safe_aP_u) * dp_dx
    v_corr -= (rho / safe_aP_v) * dp_dy
    w_corr -= (rho / safe_aP_w) * dp_dz

    # 固体セル: 速度=0
    if inp.solid_mask is not None:
        u_corr[inp.solid_mask] = 0.0
        v_corr[inp.solid_mask] = 0.0
        w_corr[inp.solid_mask] = 0.0

    return u_corr, v_corr, w_corr


def _compute_mass_residual(
    inp: NaturalConvectionInput,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> float:
    """質量残差（連続の式の不整合）を計算."""
    nx, ny, nz = inp.nx, inp.ny, inp.nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz

    # ∂u/∂x + ∂v/∂y + ∂w/∂z の各セルの値
    div = np.zeros((nx, ny, nz))

    # 中心差分
    if nx > 2:
        div[1:-1, :, :] += (u[2:, :, :] - u[:-2, :, :]) / (2.0 * dx)
    if nx > 1:
        div[0, :, :] += (u[1, :, :] - u[0, :, :]) / dx
        div[-1, :, :] += (u[-1, :, :] - u[-2, :, :]) / dx

    if ny > 2:
        div[:, 1:-1, :] += (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)
    if ny > 1:
        div[:, 0, :] += (v[:, 1, :] - v[:, 0, :]) / dy
        div[:, -1, :] += (v[:, -1, :] - v[:, -2, :]) / dy

    if nz > 2:
        div[:, :, 1:-1] += (w[:, :, 2:] - w[:, :, :-2]) / (2.0 * dz)
    if nz > 1:
        div[:, :, 0] += (w[:, :, 1] - w[:, :, 0]) / dz
        div[:, :, -1] += (w[:, :, -1] - w[:, :, -2]) / dz

    if inp.solid_mask is not None:
        div[inp.solid_mask] = 0.0

    return float(np.linalg.norm(div.ravel()))


def _make_scalar_transport_input(
    inp: NaturalConvectionInput,
    spec: ExtraScalarSpec,
    phi0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> ScalarTransportInput:
    """ExtraScalarSpec から ScalarTransportInput を組み立てる.

    natural convection ソルバーが各 SIMPLE 反復で scalar_transport の
    アセンブリを再利用するためのアダプタ。境界条件・拡散係数・ソースは
    spec から、格子・密度・速度場は natural convection 側から供給する。
    dt/t_end は外部ループで制御するため 0 を設定し、RC 面速度を介して
    対流フラックスを渡すので u, v, w の線形補間は使われない。
    """
    return ScalarTransportInput(
        Lx=inp.Lx,
        Ly=inp.Ly,
        Lz=inp.Lz,
        nx=inp.nx,
        ny=inp.ny,
        nz=inp.nz,
        rho=inp.rho,
        u=u,
        v=v,
        w=w,
        field=ScalarFieldSpec(
            name=spec.field.name,
            diffusivity=spec.field.diffusivity,
            phi0=phi0,
            source=spec.field.source,
        ),
        solid_mask=inp.solid_mask,
        bc_xm=spec.bc_xm,
        bc_xp=spec.bc_xp,
        bc_ym=spec.bc_ym,
        bc_yp=spec.bc_yp,
        bc_zm=spec.bc_zm,
        bc_zp=spec.bc_zp,
        dt=inp.dt,
        t_end=inp.t_end,
        max_iter=inp.max_inner_iter,
        tol=inp.tol_inner,
    )


def _solve_extra_scalars(
    inp: NaturalConvectionInput,
    phi_state: dict[str, np.ndarray],
    phi_old_time: dict[str, np.ndarray] | None,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    rc_faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    residuals: dict[str, float],
) -> dict[str, np.ndarray]:
    """各 ExtraScalarSpec を RC 面速度で輸送し、緩和と固体ゼロ化を適用.

    Parameters
    ----------
    phi_state : dict
        現在のスカラー状態 {name: (nx, ny, nz)}。ソルバーの初期推定に使う。
    phi_old_time : dict | None
        前タイムステップのスカラー状態（非定常時のみ）。
    rc_faces : tuple
        Rhie-Chow 面速度 (u_face_xp, v_face_yp, w_face_zp)。
    residuals : dict
        残差辞書（`phi_<name>` キーで線形求解残差を格納）。

    Returns
    -------
    dict[str, np.ndarray]
        緩和・固体ゼロ化後のスカラー状態。
    """
    new_state: dict[str, np.ndarray] = {}
    for spec in inp.extra_scalars:
        name = spec.field.name
        phi_curr = phi_state[name]
        phi_old = phi_old_time[name] if phi_old_time is not None else None
        st_inp = _make_scalar_transport_input(inp, spec, phi_curr, u, v, w)
        A_phi, b_phi = build_scalar_system(
            st_inp, phi_old_time=phi_old, rc_face_velocities=rc_faces
        )
        residuals[f"phi_{name}"] = _compute_residual_norm(A_phi, phi_curr.ravel(), b_phi)
        x = _solve_linear(
            A_phi,
            b_phi,
            phi_curr.ravel(),
            inp.tol_inner,
            inp.max_inner_iter,
        )
        phi_new = x.reshape(inp.nx, inp.ny, inp.nz)
        # 緩和
        alpha = spec.alpha
        phi_new = alpha * phi_new + (1.0 - alpha) * phi_curr
        # 固体セルはスカラーを据え置き（対流・拡散はアセンブリでゼロ化済み）
        new_state[name] = phi_new
    return new_state


def _simple_iteration(
    inp: NaturalConvectionInput,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    u_old_time: np.ndarray | None = None,
    v_old_time: np.ndarray | None = None,
    w_old_time: np.ndarray | None = None,
    T_old_time: np.ndarray | None = None,
    u_old_old_time: np.ndarray | None = None,
    v_old_old_time: np.ndarray | None = None,
    w_old_old_time: np.ndarray | None = None,
    T_old_old_time: np.ndarray | None = None,
    phi_state: dict[str, np.ndarray] | None = None,
    phi_old_time: dict[str, np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
]:
    """SIMPLE法の1反復を実行.

    Returns
    -------
    tuple
        (u_new, v_new, w_new, p_new, T_new, phi_state_new, residuals)
    """
    nx, ny, nz = inp.nx, inp.ny, inp.nz

    residuals: dict[str, float] = {}

    # 1. 運動量方程式を解く → u*, v*, w*
    #    残差は「初期残差」(解く前の残差) を使用 — OpenFOAM 方式。
    #    ||b - A*x_old|| / ||b|| が各反復の収束を正しく評価する。
    A_u, b_u, a_P_u = build_momentum_system(
        inp,
        u,
        v,
        w,
        p,
        T,
        "u",
        u_old_time,
        v_old_time,
        w_old_time,
        u_old_old_time,
        v_old_old_time,
        w_old_old_time,
    )
    residuals["u"] = _compute_residual_norm(A_u, u.ravel(), b_u)
    u_star_flat = _solve_linear(A_u, b_u, u.ravel(), inp.tol_inner, inp.max_inner_iter)

    A_v, b_v, a_P_v = build_momentum_system(
        inp,
        u,
        v,
        w,
        p,
        T,
        "v",
        u_old_time,
        v_old_time,
        w_old_time,
        u_old_old_time,
        v_old_old_time,
        w_old_old_time,
    )
    residuals["v"] = _compute_residual_norm(A_v, v.ravel(), b_v)
    v_star_flat = _solve_linear(A_v, b_v, v.ravel(), inp.tol_inner, inp.max_inner_iter)

    A_w, b_w, a_P_w = build_momentum_system(
        inp,
        u,
        v,
        w,
        p,
        T,
        "w",
        u_old_time,
        v_old_time,
        w_old_time,
        u_old_old_time,
        v_old_old_time,
        w_old_old_time,
    )
    residuals["w"] = _compute_residual_norm(A_w, w.ravel(), b_w)
    w_star_flat = _solve_linear(A_w, b_w, w.ravel(), inp.tol_inner, inp.max_inner_iter)

    u_star = u_star_flat.reshape(nx, ny, nz)
    v_star = v_star_flat.reshape(nx, ny, nz)
    w_star = w_star_flat.reshape(nx, ny, nz)

    # 固体セル: 速度=0 強制
    if inp.solid_mask is not None:
        u_star[inp.solid_mask] = 0.0
        v_star[inp.solid_mask] = 0.0
        w_star[inp.solid_mask] = 0.0

    # d係数の計算方法を連成手法に応じて選択
    # SIMPLE:  d = rho / a_P（対角のみ）
    # SIMPLEC: d = rho / row_sum(A)（Van Doormaal-Raithby, 1984）
    # PISO:    d = rho / a_P + 複数回圧力補正（Issa, 1986）
    if inp.coupling_method == "simplec":
        a_P_u_eff = np.maximum(np.array(A_u.sum(axis=1)).ravel(), 1.0)
        a_P_v_eff = np.maximum(np.array(A_v.sum(axis=1)).ravel(), 1.0)
        a_P_w_eff = np.maximum(np.array(A_w.sum(axis=1)).ravel(), 1.0)
        alpha_p_eff = 1.0
    else:
        a_P_u_eff = a_P_u
        a_P_v_eff = a_P_v
        a_P_w_eff = a_P_w
        alpha_p_eff = 1.0 if inp.coupling_method == "piso" else inp.alpha_p

    # 2. 圧力補正方程式を解く → p'（Rhie-Chow 補間付き）
    A_pp, b_pp = build_pressure_correction_system_rc(
        inp, u_star, v_star, w_star, p, a_P_u_eff, a_P_v_eff, a_P_w_eff
    )
    p_maxiter = inp.max_pressure_iter if inp.max_pressure_iter > 0 else inp.max_inner_iter
    # 圧力補正の初期残差 = ||b_pp|| / ||b_pp|| = RHS norm が指標
    # （p'の初期推定はゼロなので、初期残差 = ||b_pp - A_pp*0|| / ||b_pp|| = 1.0）
    # 代わりに RHS norm を使う（質量不整合の絶対値）
    residuals["p"] = float(np.linalg.norm(b_pp))
    if inp.pressure_solver == "amg":
        p_prime_flat = _solve_pressure_amg(A_pp, b_pp, tol=inp.tol_inner, maxiter=p_maxiter)
    else:
        p_prime_flat = _solve_linear(A_pp, b_pp, tol=inp.tol_inner, maxiter=p_maxiter)

    # 3. 速度を補正
    u_new, v_new, w_new = _correct_velocity(
        inp,
        u_star,
        v_star,
        w_star,
        p_prime_flat,
        a_P_u_eff,
        a_P_v_eff,
        a_P_w_eff,
    )

    # 4. 圧力を更新
    p_new = p + alpha_p_eff * p_prime_flat.reshape(nx, ny, nz)

    # PISO: 追加の圧力補正ステップ（Issa, 1986）
    # 補正済み速度の質量残差を再計算し、圧力を再補正する
    if inp.coupling_method == "piso":
        for _corr in range(inp.n_piso_correctors - 1):
            A_pp2, b_pp2 = build_pressure_correction_system_rc(
                inp, u_new, v_new, w_new, p_new, a_P_u_eff, a_P_v_eff, a_P_w_eff
            )
            if inp.pressure_solver == "amg":
                p_prime2_flat = _solve_pressure_amg(
                    A_pp2, b_pp2, tol=inp.tol_inner, maxiter=p_maxiter
                )
            else:
                p_prime2_flat = _solve_linear(A_pp2, b_pp2, tol=inp.tol_inner, maxiter=p_maxiter)
            u_new, v_new, w_new = _correct_velocity(
                inp,
                u_new,
                v_new,
                w_new,
                p_prime2_flat,
                a_P_u_eff,
                a_P_v_eff,
                a_P_w_eff,
            )
            p_new = p_new + p_prime2_flat.reshape(nx, ny, nz)

    # 速度に緩和を適用（PISO含む全手法で適用）
    # PISO の outer=1 回なら alpha_u=1.0 で実質無効化できる
    u_new = inp.alpha_u * u_new + (1.0 - inp.alpha_u) * u
    v_new = inp.alpha_u * v_new + (1.0 - inp.alpha_u) * v
    w_new = inp.alpha_u * w_new + (1.0 - inp.alpha_u) * w

    # 5. エネルギー方程式を解く → T
    #    Rhie-Chow 面速度を使って対流フラックスを計算（チェッカーボード抑制）
    rc_faces = compute_rhie_chow_face_velocity(
        inp, u_new, v_new, w_new, p_new, a_P_u_eff, a_P_v_eff, a_P_w_eff
    )
    A_T, b_T = build_energy_system(
        inp,
        u_new,
        v_new,
        w_new,
        T_old_time,
        rc_face_velocities=rc_faces,
        T_old_old_time=T_old_old_time,
    )
    residuals["T"] = _compute_residual_norm(A_T, T.ravel(), b_T)
    T_new_flat = _solve_linear(A_T, b_T, T.ravel(), inp.tol_inner, inp.max_inner_iter)
    T_new = T_new_flat.reshape(nx, ny, nz)

    # 温度に緩和を適用
    T_new = inp.alpha_T * T_new + (1.0 - inp.alpha_T) * T

    # 6. 追加スカラーを RC 面速度で同時輸送（Phase 6.1b）
    if inp.extra_scalars and phi_state is not None:
        phi_state_new = _solve_extra_scalars(
            inp, phi_state, phi_old_time, u_new, v_new, w_new, rc_faces, residuals
        )
    else:
        phi_state_new = phi_state if phi_state is not None else {}

    # 7. 質量残差（Rhie-Chow 面速度ベース — 圧力補正と整合的）
    residuals["mass"] = compute_face_mass_residual(
        inp, u_new, v_new, w_new, p_new, a_P_u_eff, a_P_v_eff, a_P_w_eff
    )

    return u_new, v_new, w_new, p_new, T_new, phi_state_new, residuals


class NaturalConvectionFDMProcess(SolverProcess[NaturalConvectionInput, NaturalConvectionResult]):
    """3次元自然対流ソルバー (FDM + SIMPLE法).

    Boussinesq近似による非圧縮性自然対流を、等間隔直交格子上の
    FDM + SIMPLE法で解く。固体-流体練成伝熱に対応。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="NaturalConvectionFDMProcess",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/natural-convection-fdm.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: NaturalConvectionInput) -> NaturalConvectionResult:
        """SIMPLE法で自然対流問題を解く."""
        t_start = time.perf_counter()

        inp = input_data
        nx, ny, nz = inp.nx, inp.ny, inp.nz

        # 初期化
        u = np.zeros((nx, ny, nz))
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        p = np.zeros((nx, ny, nz))
        T = inp.T0.copy() if inp.T0 is not None else np.full((nx, ny, nz), inp.T_ref)

        residual_history: dict[str, list[float]] = {
            "u": [],
            "v": [],
            "w": [],
            "p": [],
            "T": [],
            "mass": [],
        }
        for spec in inp.extra_scalars:
            residual_history[f"phi_{spec.field.name}"] = []

        # 追加スカラーの初期化（extra_scalars が空なら空 dict）
        phi_state: dict[str, np.ndarray] = {
            spec.field.name: spec.field.phi0.astype(np.float64).copy() for spec in inp.extra_scalars
        }

        if inp.is_transient:
            return self._solve_transient(inp, u, v, w, p, T, phi_state, residual_history, t_start)
        else:
            return self._solve_steady(inp, u, v, w, p, T, phi_state, residual_history, t_start)

    @staticmethod
    def _adapt_relaxation(
        inp: NaturalConvectionInput,
        residuals: dict[str, float],
        prev_max_res: float,
    ) -> NaturalConvectionInput:
        """残差に応じて緩和係数を適応的に調整.

        残差が減少 → 緩和を積極化（alpha_u↑, alpha_p↑）
        残差が増大 → 緩和を保守化（alpha_u↓, alpha_p↓）
        """
        max_res = _simple_convergence_residual(residuals)

        if prev_max_res > 0 and max_res > 0:
            ratio = max_res / prev_max_res  # < 1 なら改善
            if ratio < 0.8:
                # 順調に収束 → 緩和を積極化
                new_alpha_u = min(inp.alpha_u * 1.1, 0.9)
                new_alpha_p = min(inp.alpha_p * 1.1, 0.5)
            elif ratio > 1.2:
                # 残差増大 → 緩和を保守化
                new_alpha_u = max(inp.alpha_u * 0.8, 0.1)
                new_alpha_p = max(inp.alpha_p * 0.8, 0.05)
            else:
                return inp
        else:
            return inp

        if new_alpha_u == inp.alpha_u and new_alpha_p == inp.alpha_p:
            return inp

        return NaturalConvectionInput(
            Lx=inp.Lx,
            Ly=inp.Ly,
            Lz=inp.Lz,
            nx=inp.nx,
            ny=inp.ny,
            nz=inp.nz,
            rho=inp.rho,
            mu=inp.mu,
            Cp=inp.Cp,
            k_fluid=inp.k_fluid,
            beta=inp.beta,
            T_ref=inp.T_ref,
            gravity=inp.gravity,
            solid_mask=inp.solid_mask,
            k_solid=inp.k_solid,
            q_vol=inp.q_vol,
            T0=inp.T0,
            bc_xm=inp.bc_xm,
            bc_xp=inp.bc_xp,
            bc_ym=inp.bc_ym,
            bc_yp=inp.bc_yp,
            bc_zm=inp.bc_zm,
            bc_zp=inp.bc_zp,
            dt=inp.dt,
            t_end=inp.t_end,
            max_simple_iter=inp.max_simple_iter,
            max_inner_iter=inp.max_inner_iter,
            tol_simple=inp.tol_simple,
            tol_inner=inp.tol_inner,
            alpha_u=new_alpha_u,
            alpha_p=new_alpha_p,
            alpha_T=inp.alpha_T,
            output_interval=inp.output_interval,
            coupling_method=inp.coupling_method,
            n_piso_correctors=inp.n_piso_correctors,
            convection_scheme=inp.convection_scheme,
            time_scheme=inp.time_scheme,
            pressure_solver=inp.pressure_solver,
            adaptive_relaxation=inp.adaptive_relaxation,
            max_pressure_iter=inp.max_pressure_iter,
            extra_scalars=inp.extra_scalars,
        )

    def _solve_steady(
        self,
        inp: NaturalConvectionInput,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        T: np.ndarray,
        phi_state: dict[str, np.ndarray],
        residual_history: dict[str, list[float]],
        t_start: float,
    ) -> NaturalConvectionResult:
        """定常解析."""
        converged = False
        n_iter = 0
        prev_max_res = 0.0

        for outer in range(inp.max_simple_iter):
            n_iter = outer + 1
            u, v, w, p, T, phi_state, residuals = _simple_iteration(
                inp, u, v, w, p, T, phi_state=phi_state
            )

            for key in residuals:
                if key in residual_history:
                    residual_history[key].append(residuals[key])

            # 収束判定（温度は除外 — RHSが時間項で支配されるため）
            max_res = _simple_convergence_residual(residuals)
            if n_iter % 10 == 0 or n_iter <= 5:
                logger.info(
                    "SIMPLE iter %d: mass=%.2e, u=%.2e, v=%.2e, w=%.2e, p=%.2e, T=%.2e",
                    n_iter,
                    residuals["mass"],
                    residuals["u"],
                    residuals["v"],
                    residuals["w"],
                    residuals["p"],
                    residuals["T"],
                )

            # 適応的緩和
            if inp.adaptive_relaxation:
                inp = self._adapt_relaxation(inp, residuals, prev_max_res)
            prev_max_res = max_res

            # 発散検出
            if np.isnan(max_res) or max_res > 1e20:
                logger.warning("SIMPLE 発散: iter %d, max_residual=%.2e", n_iter, max_res)
                break

            # 最低3反復は実行（初期の偽収束を防ぐ）
            if n_iter >= 3 and max_res < inp.tol_simple:
                converged = True
                logger.info("SIMPLE 収束: %d 反復, max_residual=%.2e", n_iter, max_res)
                break

        elapsed = time.perf_counter() - t_start

        return NaturalConvectionResult(
            u=u,
            v=v,
            w=w,
            p=p,
            T=T,
            converged=converged,
            n_outer_iterations=n_iter,
            residual_history=residual_history,
            elapsed_seconds=elapsed,
            extra_scalars=phi_state,
        )

    @staticmethod
    def _adaptive_dt(
        inp: NaturalConvectionInput,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        cfl_max: float = 0.5,
    ) -> float:
        """CFL 制約に基づく適応的時間刻みを計算.

        dt = min(dt_user, cfl_max * dx / v_max) を各方向で評価し、
        最も厳しい制約を採用する。速度がゼロの場合は dt_user を返す。
        """
        dt = inp.dt
        dx, dy, dz = inp.dx, inp.dy, inp.dz

        u_max = float(np.abs(u).max())
        v_max = float(np.abs(v).max())
        w_max = float(np.abs(w).max())

        if u_max > 1e-30:
            dt = min(dt, cfl_max * dx / u_max)
        if v_max > 1e-30:
            dt = min(dt, cfl_max * dy / v_max)
        if w_max > 1e-30:
            dt = min(dt, cfl_max * dz / w_max)

        # 最小 dt 下限（ユーザ指定の 1/100）
        dt = max(dt, inp.dt * 0.01)
        return dt

    def _solve_transient(
        self,
        inp: NaturalConvectionInput,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        T: np.ndarray,
        phi_state: dict[str, np.ndarray],
        residual_history: dict[str, list[float]],
        t_start: float,
    ) -> NaturalConvectionResult:
        """非定常解析."""
        t_current = 0.0
        n_timesteps = 0
        total_outer = 0
        converged = True

        # BDF2 用: 前々タイムステップの値
        u_old_old: np.ndarray | None = None
        v_old_old: np.ndarray | None = None
        w_old_old: np.ndarray | None = None
        T_old_old: np.ndarray | None = None

        while t_current < inp.t_end - 1e-12 * inp.dt:
            n_timesteps += 1

            # CFL 制約に基づく適応的時間刻み
            dt_actual = self._adaptive_dt(inp, u, v, w)
            # t_end を超えないように調整
            dt_actual = min(dt_actual, inp.t_end - t_current)
            t_current += dt_actual

            # dt_actual を使う一時的な入力を生成
            if dt_actual != inp.dt:
                # frozen dataclass なので object.__setattr__ で一時変更
                inp_step = inp
                # NaturalConvectionInput は frozen なので直接変更できない
                # _simple_iteration 内で inp.dt を参照するため、
                # 一時的に dt を差し替えた入力を使う
                inp_step = NaturalConvectionInput(
                    Lx=inp.Lx,
                    Ly=inp.Ly,
                    Lz=inp.Lz,
                    nx=inp.nx,
                    ny=inp.ny,
                    nz=inp.nz,
                    rho=inp.rho,
                    mu=inp.mu,
                    Cp=inp.Cp,
                    k_fluid=inp.k_fluid,
                    beta=inp.beta,
                    T_ref=inp.T_ref,
                    gravity=inp.gravity,
                    solid_mask=inp.solid_mask,
                    k_solid=inp.k_solid,
                    q_vol=inp.q_vol,
                    T0=inp.T0,
                    bc_xm=inp.bc_xm,
                    bc_xp=inp.bc_xp,
                    bc_ym=inp.bc_ym,
                    bc_yp=inp.bc_yp,
                    bc_zm=inp.bc_zm,
                    bc_zp=inp.bc_zp,
                    dt=dt_actual,
                    t_end=inp.t_end,
                    max_simple_iter=inp.max_simple_iter,
                    max_inner_iter=inp.max_inner_iter,
                    tol_simple=inp.tol_simple,
                    tol_inner=inp.tol_inner,
                    alpha_u=inp.alpha_u,
                    alpha_p=inp.alpha_p,
                    alpha_T=inp.alpha_T,
                    output_interval=inp.output_interval,
                    coupling_method=inp.coupling_method,
                    n_piso_correctors=inp.n_piso_correctors,
                    convection_scheme=inp.convection_scheme,
                    time_scheme=inp.time_scheme,
                    pressure_solver=inp.pressure_solver,
                    adaptive_relaxation=inp.adaptive_relaxation,
                    max_pressure_iter=inp.max_pressure_iter,
                    extra_scalars=inp.extra_scalars,
                )
            else:
                inp_step = inp

            # 前々ステップ → 前ステップ → 現在の順に保存
            u_old_old_step = u_old_old
            v_old_old_step = v_old_old
            w_old_old_step = w_old_old
            T_old_old_step = T_old_old

            u_old = u.copy()
            v_old = v.copy()
            w_old = w.copy()
            T_old = T.copy()
            phi_old_time = {name: arr.copy() for name, arr in phi_state.items()}

            step_converged = False
            n_inner = 0
            prev_max_res_step = 0.0
            for _outer in range(inp_step.max_simple_iter):
                total_outer += 1
                n_inner += 1
                u, v, w, p, T, phi_state, residuals = _simple_iteration(
                    inp_step,
                    u,
                    v,
                    w,
                    p,
                    T,
                    u_old,
                    v_old,
                    w_old,
                    T_old,
                    u_old_old_step,
                    v_old_old_step,
                    w_old_old_step,
                    T_old_old_step,
                    phi_state=phi_state,
                    phi_old_time=phi_old_time,
                )

                for key in residuals:
                    if key in residual_history:
                        residual_history[key].append(residuals[key])

                max_res = _simple_convergence_residual(residuals)

                # 適応的緩和
                if inp_step.adaptive_relaxation:
                    inp_step = self._adapt_relaxation(inp_step, residuals, prev_max_res_step)
                prev_max_res_step = max_res

                if max_res < inp_step.tol_simple:
                    step_converged = True
                    break

            # BDF2 用: old_old を更新
            u_old_old = u_old
            v_old_old = v_old
            w_old_old = w_old
            T_old_old = T_old

            if not step_converged:
                converged = False
                logger.warning(
                    "タイムステップ %d (t=%.4f, dt=%.4e): SIMPLE未収束 (max_res=%.2e)",
                    n_timesteps,
                    t_current,
                    dt_actual,
                    max_res,
                )

            if n_timesteps % max(inp.output_interval, 1) == 0:
                logger.info(
                    "t=%.4f (dt=%.3e): SIMPLE %d iter, mass=%.2e",
                    t_current,
                    dt_actual,
                    n_inner,
                    residuals["mass"],
                )

        elapsed = time.perf_counter() - t_start

        return NaturalConvectionResult(
            u=u,
            v=v,
            w=w,
            p=p,
            T=T,
            converged=converged,
            n_outer_iterations=total_outer,
            residual_history=residual_history,
            elapsed_seconds=elapsed,
            n_timesteps=n_timesteps,
            extra_scalars=phi_state,
        )
