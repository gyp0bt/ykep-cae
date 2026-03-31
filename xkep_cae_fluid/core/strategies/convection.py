"""1次風上対流スキーム（MeshData ベース FVM 離散化）.

面フラックス方向に応じて上流側のセル値を採用する風上差分法。
ConvectionSchemeStrategy Protocol の具象実装。

面フラックス:
  F_conv = ṁ_f * φ_upwind
  ṁ_f = ρ * (u · n) * A_f  （面質量流束）

上流側の選択:
  ṁ_f > 0 → φ_upwind = φ_P (owner)
  ṁ_f < 0 → φ_upwind = φ_N (neighbour)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae_fluid.core.data import MeshData


class UpwindConvectionScheme:
    """1次風上対流スキーム.

    面の質量流束方向に基づき上流側セル値を採用する。
    安定性は高いが1次精度（数値拡散あり）。
    """

    def flux(
        self,
        phi: np.ndarray,
        velocity: np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """対流フラックスを計算.

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

        # 面質量流束: ṁ_f = (u_f · n_f) * A_f
        # u_f は owner セルの速度で近似（風上的）
        mass_flux = self._face_mass_flux(velocity, mesh, n_internal)

        # 風上値の選択
        phi_upwind = np.where(
            mass_flux >= 0,
            phi[owner[:n_internal]],
            phi[neighbour],
        )

        # 面フラックス
        face_flux = mass_flux * phi_upwind

        # セルごとの合計（owner に出、neighbour に入）
        result = np.zeros(n_cells)
        np.add.at(result, owner[:n_internal], -face_flux)
        np.add.at(result, neighbour, face_flux)

        return result

    def matrix_coefficients(
        self,
        velocity: np.ndarray,
        mesh: MeshData,
    ) -> sp.csr_matrix:
        """対流項の係数行列を構築.

        A_conv * φ の形で対流項を表現する。
        風上差分による M-matrix（対角優位で非対角が非正）。

        Parameters
        ----------
        velocity : np.ndarray
            速度場 (n_cells, ndim)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        sp.csr_matrix
            対流係数行列 (n_cells, n_cells)
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)

        # 面質量流束
        mass_flux = self._face_mass_flux(velocity, mesh, n_internal)

        # 風上分解: ṁ_f = ṁ_f⁺ + ṁ_f⁻
        # ṁ_f⁺ = max(ṁ_f, 0), ṁ_f⁻ = min(ṁ_f, 0)
        mf_pos = np.maximum(mass_flux, 0.0)
        mf_neg = np.minimum(mass_flux, 0.0)

        # COO 形式で組立
        # owner → owner 対角: +ṁ_f⁺ (流出)
        # neighbour → neighbour 対角: -ṁ_f⁻ (流出)
        # owner → neighbour 非対角: +ṁ_f⁻ (流入)
        # neighbour → owner 非対角: -ṁ_f⁺ (流入)

        own = owner[:n_internal]

        row_list = []
        col_list = []
        data_list = []

        # owner の対角: sum of ṁ_f⁺
        diag_owner = np.zeros(n_cells)
        np.add.at(diag_owner, own, mf_pos)

        # neighbour の対角: sum of -ṁ_f⁻
        diag_neighbour = np.zeros(n_cells)
        np.add.at(diag_neighbour, neighbour, -mf_neg)

        diag = diag_owner + diag_neighbour
        row_list.append(np.arange(n_cells))
        col_list.append(np.arange(n_cells))
        data_list.append(diag)

        # owner → neighbour (非対角): ṁ_f⁻
        row_list.append(own)
        col_list.append(neighbour)
        data_list.append(mf_neg)

        # neighbour → owner (非対角): -ṁ_f⁺
        row_list.append(neighbour)
        col_list.append(own)
        data_list.append(-mf_pos)

        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        data = np.concatenate(data_list)

        return sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))

    @staticmethod
    def _face_mass_flux(
        velocity: np.ndarray,
        mesh: MeshData,
        n_internal: int,
    ) -> np.ndarray:
        """内部面の質量流束 ṁ_f = (u_f · n_f) * A_f を計算.

        面速度は owner/neighbour の平均で補間する。
        """
        owner = mesh.face_owner[:n_internal]
        neighbour = mesh.face_neighbour

        # 面速度: owner と neighbour の平均
        u_face = 0.5 * (velocity[owner] + velocity[neighbour])

        # 面法線方向の速度成分 × 面面積
        normals = mesh.face_normals[:n_internal]
        areas = mesh.face_areas[:n_internal]

        # u · n * A
        u_dot_n = np.sum(u_face * normals, axis=1)
        return u_dot_n * areas
