"""非直交補正付き拡散スキーム（MeshData ベース FVM 離散化）.

非直交メッシュにおける拡散項の離散化を提供する。
直交成分（陰的）と非直交補正成分（陽的遅延修正）に分解して精度を確保する。

面フラックス:
  F_diff = Γ_f * |S_f| * (φ_N - φ_P) / |d_PN|   (直交成分)
         + Γ_f * k_f · grad(φ)_f                  (非直交補正)

ここで:
  S_f: 面法線ベクトル × 面積
  d_PN: セル中心間ベクトル
  k_f = S_f - |S_f|^2 / (S_f · e_PN) * e_PN  (非直交補正ベクトル)
  e_PN = d_PN / |d_PN|
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae_fluid.core.data import MeshData


class CorrectedDiffusionScheme:
    """非直交補正付き拡散スキーム.

    直交メッシュでは CentralDiffusionScheme と同等の結果を返す。
    非直交メッシュではセル勾配を用いた補正項で精度を改善する。

    Parameters
    ----------
    max_non_ortho_corrections : int
        非直交補正の最大反復数（デフォルト: 2）。0 で補正なし。
    """

    def __init__(self, max_non_ortho_corrections: int = 2) -> None:
        self._max_corrections = max_non_ortho_corrections

    @property
    def max_non_ortho_corrections(self) -> int:
        """非直交補正の最大反復数."""
        return self._max_corrections

    def flux(
        self,
        phi: np.ndarray,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """拡散フラックスを計算（非直交補正付き）.

        Parameters
        ----------
        phi : np.ndarray
            スカラー場 (n_cells,)
        diffusivity : float | np.ndarray
            拡散係数。スカラーまたはセルごとの配列 (n_cells,)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        np.ndarray
            拡散フラックス (n_cells,)。正値は流入を意味する。
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)
        own = owner[:n_internal]

        gamma_f = self._face_diffusivity(diffusivity, owner, neighbour, n_internal)
        orthog, non_orthog = self._decompose_face_vectors(mesh, n_internal)

        # 直交成分フラックス
        dphi = phi[neighbour] - phi[own]
        d_pn = self._face_distance(mesh, n_internal)
        face_flux = gamma_f * orthog * dphi / d_pn

        # 非直交補正
        if self._max_corrections > 0:
            grad_phi = self._cell_gradient_gauss(phi, own, neighbour, mesh)
            grad_f = 0.5 * (grad_phi[own] + grad_phi[neighbour])
            correction = gamma_f * np.sum(non_orthog * grad_f, axis=1)
            face_flux += correction

        # セルごとの合計
        result = np.zeros(n_cells)
        np.add.at(result, own, face_flux)
        np.add.at(result, neighbour, -face_flux)

        return result

    def matrix_coefficients(
        self,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> sp.csr_matrix:
        """拡散項の係数行列を構築（直交成分のみ、陰的部分）.

        非直交補正は遅延修正として deferred_correction() で取得する。

        Parameters
        ----------
        diffusivity : float | np.ndarray
            拡散係数
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        sp.csr_matrix
            拡散係数行列（直交成分） (n_cells, n_cells)
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)
        own = owner[:n_internal]

        gamma_f = self._face_diffusivity(diffusivity, owner, neighbour, n_internal)
        orthog, _ = self._decompose_face_vectors(mesh, n_internal)
        d_pn = self._face_distance(mesh, n_internal)

        # 面係数: a_f = Γ_f * |orthog| / d_PN
        a_f = gamma_f * orthog / d_pn

        # COO 形式で組立
        row = np.concatenate([own, neighbour])
        col = np.concatenate([neighbour, own])
        data = np.concatenate([-a_f, -a_f])

        diag = np.zeros(n_cells)
        np.add.at(diag, own, a_f)
        np.add.at(diag, neighbour, a_f)

        row = np.concatenate([row, np.arange(n_cells)])
        col = np.concatenate([col, np.arange(n_cells)])
        data = np.concatenate([data, diag])

        return sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))

    def deferred_correction(
        self,
        phi: np.ndarray,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """非直交補正の遅延修正項を計算.

        Parameters
        ----------
        phi : np.ndarray
            現在のスカラー場 (n_cells,)
        diffusivity : float | np.ndarray
            拡散係数
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        np.ndarray
            遅延修正ベクトル (n_cells,)
        """
        if self._max_corrections == 0:
            return np.zeros(mesh.n_cells)

        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)
        own = owner[:n_internal]

        gamma_f = self._face_diffusivity(diffusivity, owner, neighbour, n_internal)
        _, non_orthog = self._decompose_face_vectors(mesh, n_internal)

        grad_phi = self._cell_gradient_gauss(phi, own, neighbour, mesh)
        grad_f = 0.5 * (grad_phi[own] + grad_phi[neighbour])
        correction_flux = gamma_f * np.sum(non_orthog * grad_f, axis=1)

        result = np.zeros(n_cells)
        np.add.at(result, own, correction_flux)
        np.add.at(result, neighbour, -correction_flux)
        return result

    @staticmethod
    def _decompose_face_vectors(
        mesh: MeshData,
        n_internal: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """面ベクトルを直交成分と非直交補正成分に分解.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (orthog_mag, non_orthog_vec)
            orthog_mag: 直交成分の大きさ (n_internal,)
            non_orthog_vec: 非直交補正ベクトル (n_internal, ndim)
        """
        owner = mesh.face_owner[:n_internal]
        neighbour = mesh.face_neighbour

        # 面法線 × 面積 = S_f
        sf = mesh.face_normals[:n_internal] * mesh.face_areas[:n_internal, np.newaxis]

        # セル中心間ベクトル d_PN と単位ベクトル e_PN
        d_vec = mesh.cell_centers[neighbour] - mesh.cell_centers[owner]
        d_mag = np.linalg.norm(d_vec, axis=1)
        d_mag_safe = np.maximum(d_mag, 1e-30)
        e_pn = d_vec / d_mag_safe[:, np.newaxis]

        # 直交成分: |S_f · e_PN|
        sf_dot_e = np.sum(sf * e_pn, axis=1)
        orthog_mag = np.abs(sf_dot_e)

        # 非直交補正: k_f = S_f - (S_f · e_PN) * e_PN
        non_orthog = sf - sf_dot_e[:, np.newaxis] * e_pn

        return orthog_mag, non_orthog

    @staticmethod
    def _face_diffusivity(
        diffusivity: float | np.ndarray,
        owner: np.ndarray,
        neighbour: np.ndarray,
        n_internal: int,
    ) -> np.ndarray:
        """面における拡散係数（調和平均）を計算."""
        if np.isscalar(diffusivity):
            return np.full(n_internal, float(diffusivity))

        gamma_p = diffusivity[owner[:n_internal]]
        gamma_n = diffusivity[neighbour]
        denom = gamma_p + gamma_n
        safe = denom > 0
        result = np.zeros(n_internal)
        result[safe] = 2.0 * gamma_p[safe] * gamma_n[safe] / denom[safe]
        return result

    @staticmethod
    def _face_distance(mesh: MeshData, n_internal: int) -> np.ndarray:
        """内部面に対するセル中心間距離を計算."""
        owner = mesh.face_owner[:n_internal]
        neighbour = mesh.face_neighbour
        delta = mesh.cell_centers[neighbour] - mesh.cell_centers[owner]
        return np.linalg.norm(delta, axis=1)

    @staticmethod
    def _cell_gradient_gauss(
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
        phi_f = 0.5 * (phi[own] + phi[neighbour])
        sf = mesh.face_normals[:n_internal] * mesh.face_areas[:n_internal, np.newaxis]
        contrib = phi_f[:, np.newaxis] * sf

        for d in range(ndim):
            np.add.at(grad[:, d], own, contrib[:, d])
            np.add.at(grad[:, d], neighbour, -contrib[:, d])

        vol = mesh.cell_volumes
        safe = vol > 0
        grad[safe] /= vol[safe, np.newaxis]
        return grad
