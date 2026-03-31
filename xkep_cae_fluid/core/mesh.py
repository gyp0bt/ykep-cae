"""構造化メッシュ生成 Process.

不等間隔直交格子（ストレッチング対応）を生成し、
MeshData として返す StructuredMeshProcess を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import MeshData

# ---------------------------------------------------------------------------
# 入出力データ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredMeshInput:
    """構造化メッシュ生成の入力.

    Parameters
    ----------
    Lx, Ly, Lz : float
        各方向の領域サイズ [m]。Ly=0 で2D、Lz=0 で1Dとなる。
    nx, ny, nz : int
        各方向のセル数
    stretch_x, stretch_y, stretch_z : tuple[float, ...]
        各方向のストレッチング比率。
        長さが1の場合は等間隔。
        長さがn_cellsの場合は各セルの幅比率を指定。
        長さが2の場合は (ratio, grading) として幾何級数ストレッチング。
          ratio: 最大幅/最小幅の比率
          grading: 1.0=一方向、-1.0=逆方向、0.0=両端集中
    origin : tuple[float, float, float]
        原点座標 (x0, y0, z0)
    """

    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int = 1
    nz: int = 1
    stretch_x: tuple[float, ...] = (1.0,)
    stretch_y: tuple[float, ...] = (1.0,)
    stretch_z: tuple[float, ...] = (1.0,)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StructuredMeshResult:
    """構造化メッシュ生成の出力.

    Parameters
    ----------
    mesh : MeshData
        生成されたメッシュデータ
    dx : np.ndarray
        x方向の各セル幅 (nx,)
    dy : np.ndarray
        y方向の各セル幅 (ny,)
    dz : np.ndarray
        z方向の各セル幅 (nz,)
    """

    mesh: MeshData
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray


# ---------------------------------------------------------------------------
# ストレッチング関数
# ---------------------------------------------------------------------------


def _compute_cell_widths(L: float, n: int, stretch: tuple[float, ...]) -> np.ndarray:
    """ストレッチング指定からセル幅配列を計算する.

    Parameters
    ----------
    L : float
        方向の全長
    n : int
        セル数
    stretch : tuple[float, ...]
        ストレッチング指定

    Returns
    -------
    np.ndarray
        セル幅配列 (n,)、合計が L になるよう正規化
    """
    if len(stretch) == 1:
        # 等間隔
        return np.full(n, L / n)

    if len(stretch) == n:
        # 直接比率指定
        ratios = np.array(stretch, dtype=np.float64)
        return ratios / ratios.sum() * L

    if len(stretch) == 2:
        # 幾何級数ストレッチング
        ratio, grading = stretch
        if ratio <= 0:
            msg = f"ストレッチング比率は正の値が必要: {ratio}"
            raise ValueError(msg)
        if n == 1:
            return np.array([L])

        if abs(grading) < 1e-12:
            # 両端集中: 前半を正方向、後半を逆方向
            n_half = n // 2
            n_rest = n - n_half
            w1 = _compute_cell_widths(L / 2, n_half, (ratio, 1.0))
            w2 = _compute_cell_widths(L / 2, n_rest, (ratio, -1.0))
            return np.concatenate([w1, w2])

        if abs(ratio - 1.0) < 1e-12:
            return np.full(n, L / n)

        # 等比数列: r^0, r^1, ..., r^(n-1) where r = ratio^(1/(n-1))
        if grading > 0:
            r = ratio ** (1.0 / (n - 1))
            widths = r ** np.arange(n, dtype=np.float64)
        else:
            r = ratio ** (1.0 / (n - 1))
            widths = r ** np.arange(n - 1, -1, -1, dtype=np.float64)

        return widths / widths.sum() * L

    msg = f"stretch の長さが不正: {len(stretch)}。1, 2, または {n} が必要。"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# StructuredMeshProcess
# ---------------------------------------------------------------------------


class StructuredMeshProcess(PreProcess["StructuredMeshInput", "StructuredMeshResult"]):
    """不等間隔直交格子（構造化メッシュ）生成.

    ストレッチング対応の直交格子を生成し、MeshData を返す。
    セル中心座標、セル体積、面面積・法線・中心、隣接関係を全て計算する。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="StructuredMesh",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/structured-mesh.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: StructuredMeshInput) -> StructuredMeshResult:
        """メッシュを生成する."""
        inp = input_data
        nx, ny, nz = inp.nx, inp.ny, inp.nz
        x0, y0, z0 = inp.origin

        # セル幅を計算
        dx = _compute_cell_widths(inp.Lx, nx, inp.stretch_x)
        dy = _compute_cell_widths(inp.Ly, ny, inp.stretch_y)
        dz = _compute_cell_widths(inp.Lz, nz, inp.stretch_z)

        # ノード座標（各方向の累積和）
        x_nodes = np.concatenate([[x0], x0 + np.cumsum(dx)])
        y_nodes = np.concatenate([[y0], y0 + np.cumsum(dy)])
        z_nodes = np.concatenate([[z0], z0 + np.cumsum(dz)])

        # セル中心座標
        x_centers = x0 + np.cumsum(dx) - dx / 2
        y_centers = y0 + np.cumsum(dy) - dy / 2
        z_centers = z0 + np.cumsum(dz) - dz / 2

        # 3D meshgrid でセル中心
        xc, yc, zc = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        cell_centers = np.column_stack([xc.ravel(), yc.ravel(), zc.ravel()])

        # セル体積
        dxv, dyv, dzv = np.meshgrid(dx, dy, dz, indexing="ij")
        cell_volumes = (dxv * dyv * dzv).ravel()

        # ノード座標（3D格子の全ノード）
        xn, yn, zn = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
        node_coords = np.column_stack([xn.ravel(), yn.ravel(), zn.ravel()])

        # connectivity: 各セルの8頂点ノードインデックス
        # ノードは (nx+1)*(ny+1)*(nz+1) 個
        def node_idx(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
            return i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k

        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ii_f, jj_f, kk_f = ii.ravel(), jj.ravel(), kk.ravel()

        connectivity = np.column_stack(
            [
                node_idx(ii_f, jj_f, kk_f),
                node_idx(ii_f + 1, jj_f, kk_f),
                node_idx(ii_f + 1, jj_f + 1, kk_f),
                node_idx(ii_f, jj_f + 1, kk_f),
                node_idx(ii_f, jj_f, kk_f + 1),
                node_idx(ii_f + 1, jj_f, kk_f + 1),
                node_idx(ii_f + 1, jj_f + 1, kk_f + 1),
                node_idx(ii_f, jj_f + 1, kk_f + 1),
            ]
        )

        # 面情報の構築
        faces = self._build_faces(nx, ny, nz, dx, dy, dz, x_centers, y_centers, z_centers)

        mesh = MeshData(
            node_coords=node_coords,
            connectivity=connectivity,
            cell_volumes=cell_volumes,
            face_areas=faces["areas"],
            face_normals=faces["normals"],
            face_centers=faces["centers"],
            cell_centers=cell_centers,
            dimensions=(nx, ny, nz),
            face_owner=faces["owner"],
            face_neighbour=faces["neighbour"],
        )

        return StructuredMeshResult(mesh=mesh, dx=dx, dy=dy, dz=dz)

    @staticmethod
    def _build_faces(
        nx: int,
        ny: int,
        nz: int,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        z_centers: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """内部面+境界面の情報を構築する.

        Returns
        -------
        dict
            areas, normals, centers, owner, neighbour
        """

        # セルインデックス関数
        def cell_idx(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
            return i * (ny * nz) + j * nz + k

        all_areas: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        all_centers: list[np.ndarray] = []
        all_owner: list[np.ndarray] = []
        all_neighbour: list[np.ndarray] = []

        # --- 内部面 ---
        # x方向内部面: (nx-1) * ny * nz 面
        if nx > 1:
            ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny), np.arange(nz), indexing="ij")
            ii_f, jj_f, kk_f = ii.ravel(), jj.ravel(), kk.ravel()
            n_faces_x = len(ii_f)

            # 面面積 = dy[j] * dz[k]
            dyv = dy[jj_f]
            dzv = dz[kk_f]
            areas = dyv * dzv
            all_areas.append(areas)

            # 法線 = (1, 0, 0)
            normals = np.zeros((n_faces_x, 3))
            normals[:, 0] = 1.0
            all_normals.append(normals)

            # x方向の面位置 = セル i の右端 = Σdx[0:i+1]
            x_face = np.cumsum(dx)[ii_f]
            centers = np.column_stack([x_face, y_centers[jj_f], z_centers[kk_f]])
            all_centers.append(centers)

            all_owner.append(cell_idx(ii_f, jj_f, kk_f))
            all_neighbour.append(cell_idx(ii_f + 1, jj_f, kk_f))

        # y方向内部面: nx * (ny-1) * nz 面
        if ny > 1:
            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny - 1), np.arange(nz), indexing="ij")
            ii_f, jj_f, kk_f = ii.ravel(), jj.ravel(), kk.ravel()
            n_faces_y = len(ii_f)

            dxv = dx[ii_f]
            dzv = dz[kk_f]
            areas = dxv * dzv
            all_areas.append(areas)

            normals = np.zeros((n_faces_y, 3))
            normals[:, 1] = 1.0
            all_normals.append(normals)

            y_face = np.cumsum(dy)[jj_f]
            centers = np.column_stack([x_centers[ii_f], y_face, z_centers[kk_f]])
            all_centers.append(centers)

            all_owner.append(cell_idx(ii_f, jj_f, kk_f))
            all_neighbour.append(cell_idx(ii_f, jj_f + 1, kk_f))

        # z方向内部面: nx * ny * (nz-1) 面
        if nz > 1:
            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz - 1), indexing="ij")
            ii_f, jj_f, kk_f = ii.ravel(), jj.ravel(), kk.ravel()
            n_faces_z = len(ii_f)

            dxv = dx[ii_f]
            dyv = dy[jj_f]
            areas = dxv * dyv
            all_areas.append(areas)

            normals = np.zeros((n_faces_z, 3))
            normals[:, 2] = 1.0
            all_normals.append(normals)

            z_face = np.cumsum(dz)[kk_f]
            centers = np.column_stack([x_centers[ii_f], y_centers[jj_f], z_face])
            all_centers.append(centers)

            all_owner.append(cell_idx(ii_f, jj_f, kk_f))
            all_neighbour.append(cell_idx(ii_f, jj_f, kk_f + 1))

        # 内部面の結合
        if all_areas:
            areas_arr = np.concatenate(all_areas)
            normals_arr = np.concatenate(all_normals)
            centers_arr = np.concatenate(all_centers)
            owner_arr = np.concatenate(all_owner)
            neighbour_arr = np.concatenate(all_neighbour)
        else:
            areas_arr = np.array([], dtype=np.float64)
            normals_arr = np.zeros((0, 3), dtype=np.float64)
            centers_arr = np.zeros((0, 3), dtype=np.float64)
            owner_arr = np.array([], dtype=np.int64)
            neighbour_arr = np.array([], dtype=np.int64)

        return {
            "areas": areas_arr,
            "normals": normals_arr,
            "centers": centers_arr,
            "owner": owner_arr,
            "neighbour": neighbour_arr,
        }
