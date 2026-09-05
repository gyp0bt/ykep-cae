"""境界パッチ条件（面ベース FVM 共通）.

境界条件は「パッチ名 → :class:`PatchBC`」の写像で与える。パッチ名は
``MeshData.boundary_patches`` のキー（構造格子なら ``XM/XP/YM/YP/ZM/ZP``、
.inp なら ``*SURFACE`` 名、polyMesh なら boundary ファイルのパッチ名）。
:func:`resolve_boundary` がこれを境界面ごとの配列 :class:`BoundaryFaces` に展開し、
組み立て側はパッチ名を意識せずに面ループで係数を足す。

離散化（owner セル P、境界面 b、面積 A_b、セル中心から面中心までの法線距離 d_b）:

- DIRICHLET ``value`` = φ_b: 拡散係数 a_b = Γ_P A_b / d_b を対角に、a_b φ_b を右辺に
- NEUMANN ``flux`` = Γ ∂φ/∂n_in（正 = 流入）: 右辺に flux·A_b
- ROBIN ``J = h (φ_inf − φ_b)``: 合成コンダクタンス U = Γ_P h / (Γ_P + h d_b) を使い
  U A_b を対角、U A_b φ_inf を右辺に
- ZERO_GRADIENT: 何もしない（断熱）

構造格子の既存 FDM（``2Γ/d²``、``U_eff = 2Γh/(2Γ+hd)``）とは d_b = d/2 で一致する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from xkep_cae_fluid.core.data import MeshData


class BCKind(Enum):
    """スカラー場の境界条件種別."""

    ZERO_GRADIENT = "zero_gradient"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"


_KIND_CODE: dict[BCKind, int] = {
    BCKind.ZERO_GRADIENT: 0,
    BCKind.DIRICHLET: 1,
    BCKind.NEUMANN: 2,
    BCKind.ROBIN: 3,
}


@dataclass(frozen=True)
class PatchBC:
    """1 パッチの境界条件.

    Parameters
    ----------
    kind : BCKind
    value : float | np.ndarray
        DIRICHLET の φ_b。パッチの面数と同じ長さの配列でも可
    flux : float | np.ndarray
        NEUMANN の Γ ∂φ/∂n_in（正 = 流入）。配列可
    h : float
        ROBIN の伝達係数
    phi_inf : float
        ROBIN の外部値
    """

    kind: BCKind = BCKind.ZERO_GRADIENT
    value: float | np.ndarray = 0.0
    flux: float | np.ndarray = 0.0
    h: float = 0.0
    phi_inf: float = 0.0

    @staticmethod
    def dirichlet(value: float | np.ndarray) -> PatchBC:
        return PatchBC(BCKind.DIRICHLET, value=value)

    @staticmethod
    def neumann(flux: float | np.ndarray) -> PatchBC:
        return PatchBC(BCKind.NEUMANN, flux=flux)

    @staticmethod
    def robin(h: float, phi_inf: float) -> PatchBC:
        return PatchBC(BCKind.ROBIN, h=h, phi_inf=phi_inf)

    @staticmethod
    def zero_gradient() -> PatchBC:
        return PatchBC(BCKind.ZERO_GRADIENT)


@dataclass(frozen=True)
class BoundaryFaces:
    """境界面ごとに展開した境界条件（長さは全て n_boundary_faces）.

    Parameters
    ----------
    faces : np.ndarray
        面のグローバルインデックス
    owner : np.ndarray
        owner セル
    kind : np.ndarray
        種別コード（0: zero-gradient, 1: Dirichlet, 2: Neumann, 3: Robin）
    value, flux, h, phi_inf : np.ndarray
        各種別の値
    area : np.ndarray
        面積
    distance : np.ndarray
        owner セル中心から面中心までの法線方向距離 d_b
    """

    faces: np.ndarray
    owner: np.ndarray
    kind: np.ndarray
    value: np.ndarray
    flux: np.ndarray
    h: np.ndarray
    phi_inf: np.ndarray
    area: np.ndarray
    distance: np.ndarray

    @property
    def n(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_dirichlet(self) -> np.ndarray:
        return self.kind == _KIND_CODE[BCKind.DIRICHLET]

    @property
    def is_neumann(self) -> np.ndarray:
        return self.kind == _KIND_CODE[BCKind.NEUMANN]

    @property
    def is_robin(self) -> np.ndarray:
        return self.kind == _KIND_CODE[BCKind.ROBIN]


def _broadcast(v: float | np.ndarray, n: int, patch: str, what: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(
            f"パッチ {patch!r} の {what} は長さ {n} の配列かスカラーが必要: {arr.shape}"
        )
    return arr


def resolve_boundary(
    mesh: MeshData,
    bcs: Mapping[str, PatchBC],
    *,
    default: PatchBC | None = None,
) -> BoundaryFaces:
    """パッチ別の境界条件を境界面ごとの配列に展開する.

    Parameters
    ----------
    mesh : MeshData
        ``boundary_patches`` を持つメッシュ
    bcs : Mapping[str, PatchBC]
        パッチ名 → 境界条件。未指定のパッチは ``default``（既定はゼロ勾配）
    default : PatchBC | None
        未指定パッチの条件

    Raises
    ------
    KeyError
        ``bcs`` にメッシュに無いパッチ名が含まれる
    ValueError
        メッシュが面情報／パッチを持たない
    """
    if mesh.face_owner is None or mesh.face_areas is None or mesh.face_centers is None:
        raise ValueError("MeshData に面情報（face_owner / face_areas / face_centers）がありません")
    if mesh.cell_centers is None or mesh.face_normals is None:
        raise ValueError("MeshData に cell_centers / face_normals がありません")
    patches = dict(mesh.boundary_patches or {})
    unknown = sorted(set(bcs) - set(patches))
    if unknown:
        raise KeyError(f"メッシュに無いパッチ名: {unknown}（定義済み: {sorted(patches)}）")

    n_b = mesh.n_boundary_faces
    n_int = mesh.n_internal_faces
    faces = mesh.boundary_faces
    kind = np.zeros(n_b, dtype=np.int64)
    value = np.zeros(n_b)
    flux = np.zeros(n_b)
    h = np.zeros(n_b)
    phi_inf = np.zeros(n_b)

    if default is None:
        default = PatchBC()
    covered = np.zeros(n_b, dtype=bool)
    for name, idx in patches.items():
        bc = bcs.get(name, default)
        local = np.asarray(idx, dtype=np.int64) - n_int
        if local.size and (local.min() < 0 or local.max() >= n_b):
            raise ValueError(f"パッチ {name!r} の面インデックスが境界面の範囲外です")
        m = local.size
        kind[local] = _KIND_CODE[bc.kind]
        value[local] = _broadcast(bc.value, m, name, "value")
        flux[local] = _broadcast(bc.flux, m, name, "flux")
        h[local] = float(bc.h)
        phi_inf[local] = float(bc.phi_inf)
        covered[local] = True
    # パッチに属さない境界面は default
    if not np.all(covered):
        rest = ~covered
        kind[rest] = _KIND_CODE[default.kind]
        value[rest] = _broadcast(default.value, int(rest.sum()), "<default>", "value")
        flux[rest] = _broadcast(default.flux, int(rest.sum()), "<default>", "flux")
        h[rest] = float(default.h)
        phi_inf[rest] = float(default.phi_inf)

    owner = mesh.face_owner[faces]
    normals = mesh.face_normals[faces]
    d_vec = mesh.face_centers[faces] - mesh.cell_centers[owner]
    distance = np.abs(np.sum(d_vec * normals, axis=1))
    return BoundaryFaces(
        faces=faces,
        owner=owner,
        kind=kind,
        value=value,
        flux=flux,
        h=h,
        phi_inf=phi_inf,
        area=mesh.face_areas[faces],
        distance=distance,
    )
