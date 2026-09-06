"""非直交補正付き拡散スキーム（DiffusionSchemeStrategy の実装、fvm 層の薄い包み）.

直交成分（陰的）と非直交補正成分（陽的遅延修正）に分解して拡散項を離散化する。
実体は面ベース FVM 低レイヤー :mod:`xkep_cae_fluid.fvm` の

- :func:`~xkep_cae_fluid.fvm.assembly.assemble_diffusion`（over-relaxed 分解の陰的成分 Γ_f |E_f| / d_PN）
- :func:`~xkep_cae_fluid.fvm.assembly.nonorthogonal_correction`（遅延補正 Γ_f (∇φ)_f·T_f）
- :func:`~xkep_cae_fluid.fvm.assembly.diffusive_face_flux`

で、本クラスは Strategy Protocol（``flux`` / ``matrix_coefficients`` / ``deferred_correction``）の
形に合わせるだけ。境界面は全てゼロ勾配として扱う（境界条件は呼び出し側が別途課す。
境界条件込みで組むなら fvm 層を直接使う）。

面フラックス（内部面）:
  J_f = −Γ_f [ |E_f| (φ_N − φ_P) / d_PN + (∇φ)_f·T_f ]、S_f = E_f + T_f、|E_f| = A_f / (n_f·e_PN)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae_fluid.core.data import MeshData
from xkep_cae_fluid.fvm.assembly import (
    assemble_diffusion,
    diffusive_face_flux,
    nonorthogonal_correction,
)
from xkep_cae_fluid.fvm.boundary import BoundaryFaces, resolve_boundary


def _zero_gradient_boundary(mesh: MeshData) -> BoundaryFaces:
    """全境界面をゼロ勾配にした境界配列（境界条件は呼び出し側が課す前提）."""
    return resolve_boundary(mesh, {})


class CorrectedDiffusionScheme:
    """非直交補正付き拡散スキーム.

    直交メッシュでは CentralDiffusionScheme と同等の結果を返す。
    非直交メッシュではセル勾配を用いた補正項で精度を改善する。

    Parameters
    ----------
    max_non_ortho_corrections : int
        非直交補正の最大反復数（デフォルト: 2）。0 で補正なし（``deferred_correction`` はゼロ、
        ``flux`` は直交成分のみ）。反復自体は呼び出し側（外部反復）が行う
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
        """拡散フラックスの各セルへの正味流入 (n_cells,)（正値は流入）.

        Parameters
        ----------
        phi : np.ndarray
            スカラー場 (n_cells,)
        diffusivity : float | np.ndarray
            拡散係数。スカラーまたはセルごとの配列 (n_cells,)
        mesh : MeshData
            メッシュデータ（面情報付き）
        """
        bfaces = _zero_gradient_boundary(mesh)
        j = diffusive_face_flux(mesh, phi, diffusivity, bfaces, corrected=self._max_corrections > 0)
        n_int = mesh.n_internal_faces
        result = np.zeros(mesh.n_cells)
        np.add.at(result, mesh.face_owner[:n_int], -j[:n_int])
        np.add.at(result, mesh.face_neighbour, j[:n_int])
        return result

    def matrix_coefficients(
        self,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> sp.csr_matrix:
        """拡散項の係数行列（直交成分のみ、陰的部分）(n_cells, n_cells).

        非直交補正は遅延修正として ``deferred_correction()`` で取得する。
        """
        A, _b = assemble_diffusion(mesh, diffusivity, _zero_gradient_boundary(mesh))
        return sp.csr_matrix(A)

    def deferred_correction(
        self,
        phi: np.ndarray,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """非直交補正の遅延修正項 (n_cells,)（``A φ = b + correction`` の右辺に足す）."""
        if self._max_corrections == 0:
            return np.zeros(mesh.n_cells)
        return nonorthogonal_correction(mesh, phi, diffusivity, _zero_gradient_boundary(mesh))
