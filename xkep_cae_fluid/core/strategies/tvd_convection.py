"""TVD 対流スキーム（MeshData ベース FVM 離散化）.

Total Variation Diminishing スキームによる対流項離散化。
勾配比 r に基づくフラックスリミッタで風上差分と高次スキームをブレンドし、
数値振動を抑制しつつ2次精度を達成する。

面フラックス:
  φ_f = φ_P + ψ(r) * (φ_N - φ_P) / 2   (ṁ_f > 0 の場合)

リミッタ関数:
  van Leer:  ψ(r) = (r + |r|) / (1 + |r|)
  Superbee:  ψ(r) = max(0, min(2r, 1), min(r, 2))
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import scipy.sparse as sp

from xkep_cae_fluid.core.data import MeshData


class TVDLimiter(Enum):
    """TVD リミッタの種類."""

    VAN_LEER = "van_leer"
    SUPERBEE = "superbee"


def _limiter_van_leer(r: np.ndarray) -> np.ndarray:
    """van Leer リミッタ: ψ(r) = (r + |r|) / (1 + |r|)."""
    abs_r = np.abs(r)
    return (r + abs_r) / (1.0 + abs_r)


def _limiter_superbee(r: np.ndarray) -> np.ndarray:
    """Superbee リミッタ: ψ(r) = max(0, min(2r, 1), min(r, 2))."""
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


_LIMITER_FUNCS = {
    TVDLimiter.VAN_LEER: _limiter_van_leer,
    TVDLimiter.SUPERBEE: _limiter_superbee,
}


class TVDConvectionScheme:
    """TVD 対流スキーム.

    フラックスリミッタにより風上差分と高次スキームをブレンドする。
    Sweby の TVD 領域内で単調性を保持しつつ高精度化を実現。
    """

    def __init__(self, limiter: TVDLimiter = TVDLimiter.VAN_LEER) -> None:
        self._limiter = limiter
        self._limiter_func = _LIMITER_FUNCS[limiter]

    @property
    def limiter(self) -> TVDLimiter:
        """使用中のリミッタ種別."""
        return self._limiter

    def flux(
        self,
        phi: np.ndarray,
        velocity: np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """TVD 対流フラックスを計算.

        Parameters
        ----------
        phi : np.ndarray
            スカラー場 (n_cells,)
        velocity : np.ndarray
            速度場 (n_cells, ndim)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        np.ndarray
            対流フラックス (n_cells,)。正値は流入を意味する。
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)

        mass_flux = self._face_mass_flux(velocity, mesh, n_internal)
        own = owner[:n_internal]

        # 風上値 + TVD 補正
        phi_face = self._tvd_face_value(phi, mass_flux, own, neighbour, mesh)
        face_flux = mass_flux * phi_face

        result = np.zeros(n_cells)
        np.add.at(result, own, -face_flux)
        np.add.at(result, neighbour, face_flux)
        return result

    def matrix_coefficients(
        self,
        velocity: np.ndarray,
        mesh: MeshData,
    ) -> sp.csr_matrix:
        """TVD 対流項の係数行列を構築.

        陰的部分（風上差分）を行列に、TVD 補正は遅延修正として
        右辺ベクトル（deferred_correction）に分離する。

        本メソッドは陰的風上部分のみを返す。遅延修正項は
        deferred_correction() で別途取得する。

        Parameters
        ----------
        velocity : np.ndarray
            速度場 (n_cells, ndim)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        sp.csr_matrix
            対流係数行列（風上部分） (n_cells, n_cells)
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)

        mass_flux = self._face_mass_flux(velocity, mesh, n_internal)
        mf_pos = np.maximum(mass_flux, 0.0)
        mf_neg = np.minimum(mass_flux, 0.0)

        own = owner[:n_internal]

        row_list = []
        col_list = []
        data_list = []

        diag_owner = np.zeros(n_cells)
        np.add.at(diag_owner, own, mf_pos)
        diag_neighbour = np.zeros(n_cells)
        np.add.at(diag_neighbour, neighbour, -mf_neg)
        diag = diag_owner + diag_neighbour

        row_list.append(np.arange(n_cells))
        col_list.append(np.arange(n_cells))
        data_list.append(diag)

        row_list.append(own)
        col_list.append(neighbour)
        data_list.append(mf_neg)

        row_list.append(neighbour)
        col_list.append(own)
        data_list.append(-mf_pos)

        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        data = np.concatenate(data_list)

        return sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))

    def deferred_correction(
        self,
        phi: np.ndarray,
        velocity: np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """TVD 遅延修正項を計算.

        TVD 面値と風上面値の差分を右辺ベクトルとして返す。
        外部反復ループ内で A_upwind * phi = b + correction の形で使用する。

        Parameters
        ----------
        phi : np.ndarray
            現在のスカラー場 (n_cells,)
        velocity : np.ndarray
            速度場 (n_cells, ndim)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        np.ndarray
            遅延修正ベクトル (n_cells,)
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)
        own = owner[:n_internal]

        mass_flux = self._face_mass_flux(velocity, mesh, n_internal)

        # TVD 面値
        phi_tvd = self._tvd_face_value(phi, mass_flux, own, neighbour, mesh)

        # 風上面値
        phi_upwind = np.where(mass_flux >= 0, phi[own], phi[neighbour])

        # 修正フラックス
        correction_flux = mass_flux * (phi_tvd - phi_upwind)

        result = np.zeros(n_cells)
        np.add.at(result, own, correction_flux)
        np.add.at(result, neighbour, -correction_flux)
        return result

    def _tvd_face_value(
        self,
        phi: np.ndarray,
        mass_flux: np.ndarray,
        own: np.ndarray,
        neighbour: np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """TVD リミッタ付き面補間値を計算.

        φ_f = φ_U + 0.5 * ψ(r) * (φ_D - φ_U)

        U: 上流セル, D: 下流セル
        r = (φ_U - φ_UU) / (φ_D - φ_U)  (勾配比)

        ここでは MeshData の勾配情報を使い、遠方上流値 φ_UU を
        2φ_U - φ_D の外挿で近似する（Darwish-Moukalled 手法）。
        """
        phi_own = phi[own]
        phi_neigh = phi[neighbour]

        # 正方向流れ: U=own, D=neighbour
        # 負方向流れ: U=neighbour, D=own
        is_positive = mass_flux >= 0

        phi_u = np.where(is_positive, phi_own, phi_neigh)
        phi_d = np.where(is_positive, phi_neigh, phi_own)

        # 勾配比 r の計算
        # セル中心間ベクトルを利用した勾配近似
        delta_phi = phi_d - phi_u
        eps = 1e-30

        # 遠方上流値の近似: セル中心の勾配から外挿
        # grad(φ)_U · 2d_UD ≈ φ_UU → φ_UU = 2φ_U - φ_D (線形外挿)
        # r = (φ_U - φ_UU) / (φ_D - φ_U) = (2(φ_D - φ_U) - (φ_D - φ_U)) / (φ_D - φ_U)
        # 上記は常に1なので、セル勾配ベースの手法を使う
        #
        # Darwish-Moukalled: r = 2 * grad(φ)_U · d_UD / (φ_D - φ_U) - 1
        grad_phi = self._cell_gradient(phi, own, neighbour, mesh)

        cc_own = mesh.cell_centers[own]
        cc_neigh = mesh.cell_centers[neighbour]
        d_vec = cc_neigh - cc_own  # owner → neighbour

        # 上流セルの勾配を d_UD に射影
        grad_u = np.where(
            is_positive[:, np.newaxis],
            grad_phi[own],
            grad_phi[neighbour],
        )
        grad_dot_d = np.sum(
            grad_u * np.where(is_positive[:, np.newaxis], d_vec, -d_vec),
            axis=1,
        )

        # r = 2 * grad(φ)_U · d_UD / (φ_D - φ_U) - 1
        r = np.where(
            np.abs(delta_phi) > eps,
            2.0 * grad_dot_d / delta_phi - 1.0,
            0.0,
        )

        # リミッタ適用
        psi = self._limiter_func(r)

        # TVD 面値
        return phi_u + 0.5 * psi * delta_phi

    @staticmethod
    def _cell_gradient(
        phi: np.ndarray,
        own: np.ndarray,
        neighbour: np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """Green-Gauss 法によるセル勾配を計算."""
        n_cells = mesh.n_cells
        ndim = mesh.ndim
        n_internal = len(neighbour)

        grad = np.zeros((n_cells, ndim))

        # 面値（線形補間）
        phi_f = 0.5 * (phi[own] + phi[neighbour])

        # 面法線 × 面積
        sf = mesh.face_normals[:n_internal] * mesh.face_areas[:n_internal, np.newaxis]

        # owner: +S_f, neighbour: -S_f
        contrib = phi_f[:, np.newaxis] * sf
        for d in range(ndim):
            np.add.at(grad[:, d], own, contrib[:, d])
            np.add.at(grad[:, d], neighbour, -contrib[:, d])

        # 体積で割る
        vol = mesh.cell_volumes
        safe = vol > 0
        grad[safe] /= vol[safe, np.newaxis]

        return grad

    @staticmethod
    def _face_mass_flux(
        velocity: np.ndarray,
        mesh: MeshData,
        n_internal: int,
    ) -> np.ndarray:
        """内部面の質量流束 ṁ_f = (u_f · n_f) * A_f を計算."""
        owner = mesh.face_owner[:n_internal]
        neighbour = mesh.face_neighbour

        u_face = 0.5 * (velocity[owner] + velocity[neighbour])
        normals = mesh.face_normals[:n_internal]
        areas = mesh.face_areas[:n_internal]

        u_dot_n = np.sum(u_face * normals, axis=1)
        return u_dot_n * areas
