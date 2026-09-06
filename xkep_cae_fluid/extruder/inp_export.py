"""展開チャネル 2.5D の諸元 → **汎用記法の .inp**（非構造 NS FVM 経路）を書き出す PreProcess.

専用ソルバー（:class:`~xkep_cae_fluid.extruder.solver.ExtruderFlowProcess`）は
「w の Poisson + 断面 Stokes」を直接組む 2.5D 専用実装で、構造格子と展開チャネルの
境界条件を前提にしている。同じ問題を **汎用記法**（``*NODE`` / ``*ELEMENT`` +
``*NAVIER STOKES``）で書けば、溝形状・フライト断面・不等ピッチなどを .inp の編集だけで
変えられる。この Process はその橋渡しで、``ScrewSpec`` から

- 断面 (x, y) を z 方向 1 セルに押し出した ``C3D8`` メッシュ（フライトは要素ごと抜く）
- ``*SURFACE`` 4 枚（x 両端・z 両端）と ``BARREL``（y = H）
- ``*BOUNDARY, TYPE=PERIODIC`` 2 組（x は 1 ピッチ W_t、z は厚さ dz でオフセット 0 相当）
- 圧力跳びを分解した一様体積力 ``*DLOAD, BF``（f = (−G·cotφ, 0, −G)）
- バレルの移動壁 ``*BOUNDARY, TYPE=VELOCITY``（u = −V sinφ, w = +V cosφ）
- 粘度（``*VISCOSITY`` / ``TYPE=POWER LAW`` / ``TYPE=CARREAU``）

を持つ .inp テキストを組み立てる。設計は ``docs/design/inp-generic-extrusion.md``。

**z 方向を対称面にしてはいけない**（w が 0 に固定される）。1 セル厚の両端を周期にすると
∂/∂z = 0 が厳密に成り立ち、w が自由になる。これが「2.5D を汎用記法で書く」の要点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.extruder.data import ChannelGrid, ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.fvm.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
    ViscosityModelStrategy,
)


@dataclass(frozen=True)
class ExtruderInpInput:
    """:class:`ExtruderChannelInpProcess` の入力.

    Parameters
    ----------
    spec : ScrewSpec
        スクリュー諸元と断面格子の解像度
    G : float
        下流方向圧力勾配 dp/dz [Pa/m]（背圧のある押出は正）
    viscosity : ViscosityModelStrategy
        粘度モデル（:class:`~xkep_cae_fluid.fvm.viscosity.NewtonianViscosity` /
        ``PowerLawViscosity`` / ``CarreauViscosity`` のみ .inp に書ける）
    density : float
        密度 [kg/m³]。クリープ流れ（``INERTIA`` 無し）では結果に影響しないが必須項目
    depth_z : float
        押し出す z 方向の厚さ [m]。0 なら断面の代表寸法から自動（H/10）
    heading : str
        ``*HEADING`` の 1 行
    coupling : str
        ``PRESSURE_VELOCITY=``（既定 ``COUPLED``: 速度と圧力を 1 つの線形系で解く）
    tol, max_outer : float, int
        ``*CONTROLS, PARAMETERS=SOLVER`` の収束閾値と外部反復上限
    alpha_mu : float
        粘度 Picard の緩和係数（``RELAXATION`` の ``VISCOSITY=``）
    output_format : str
        ``*OUTPUT, FIELD, FORMAT=``
    """

    spec: ScrewSpec
    G: float
    viscosity: ViscosityModelStrategy = field(default_factory=lambda: NewtonianViscosity(1000.0))
    density: float = 1000.0
    depth_z: float = 0.0
    heading: str = ""
    coupling: str = "COUPLED"
    tol: float = 1.0e-9
    max_outer: int = 200
    alpha_mu: float = 0.5
    output_format: str = "NPZ"


@dataclass(frozen=True)
class ExtruderInpResult:
    """汎用記法の .inp テキストと、照合に使う諸量.

    Parameters
    ----------
    text : str
        .inp の全文
    grid : ChannelGrid
        元の断面格子（専用ソルバーと同じもの。照合に使う）
    n_cells, n_nodes : int
        書き出した流体セル数・節点数
    depth_z : float
        押し出した z 方向の厚さ [m]
    body_force : tuple[float, float, float]
        ``*DLOAD, BF`` に書いた一様体積力 [N/m³]
    barrel_velocity : tuple[float, float, float]
        バレル（y = H）の移動壁速度 [m/s]
    """

    text: str
    grid: ChannelGrid
    n_cells: int
    n_nodes: int
    depth_z: float
    body_force: tuple[float, float, float]
    barrel_velocity: tuple[float, float, float]


def _num(value: float) -> str:
    """.inp に書ける数値表記（numpy スカラーの repr を避けて往復精度を保つ）."""
    return repr(float(value))


def _viscosity_keyword(model: ViscosityModelStrategy) -> list[str]:
    """粘度モデル Strategy → ``*VISCOSITY`` のキーワード行."""
    if isinstance(model, NewtonianViscosity):
        return ["*VISCOSITY", f" {_num(model.mu)}"]
    if isinstance(model, PowerLawViscosity):
        return [
            "*VISCOSITY, TYPE=POWER LAW",
            f" {_num(model.K)}, {_num(model.n)}, {_num(model.gamma_min)}, {_num(model.mu_max)}",
        ]
    if isinstance(model, CarreauViscosity):
        return [
            "*VISCOSITY, TYPE=CARREAU",
            f" {_num(model.mu_0)}, {_num(model.mu_inf)}, {_num(model.lam)}, {_num(model.n)}",
        ]
    raise ValueError(
        f"粘度モデル {type(model).__name__} は .inp に書けません"
        "（NewtonianViscosity / PowerLawViscosity / CarreauViscosity のみ）"
    )


def build_extruder_inp(input_data: ExtruderInpInput, grid: ChannelGrid) -> ExtruderInpResult:
    """断面格子から汎用記法の .inp テキストを組み立てる."""
    spec = input_data.spec
    nx, ny = grid.nx, grid.ny
    dz = float(input_data.depth_z) if input_data.depth_z > 0.0 else spec.H / 10.0
    xs = np.concatenate([[0.0], np.cumsum(grid.dx)])
    ys = np.concatenate([[0.0], np.cumsum(grid.dy)])
    fluid = ~grid.solid

    # 使う節点だけを書き出す（節点 ID = 1 + i + (nx+1) (j + (ny+1) k)）
    def nid(i: int, j: int, k: int) -> int:
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    used: set[int] = set()
    cells: list[tuple[int, int, int]] = []  # (elem_id, i, j)
    eid = 0
    for j in range(ny):
        for i in range(nx):
            if not fluid[i, j]:
                continue
            eid += 1
            cells.append((eid, i, j))
            for k in (0, 1):
                for jj in (j, j + 1):
                    for ii in (i, i + 1):
                        used.add(nid(ii, jj, k))
    if not cells:
        raise ValueError("流体セルが 1 つもありません（固体マスクが領域全体を覆っています）")

    lines: list[str] = ["*HEADING"]
    lines.append(
        " " + (input_data.heading or f"unrolled channel D={spec.D} lead={spec.lead} N={spec.N}")
    )
    lines.append("*NODE")
    for k in (0, 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = nid(i, j, k)
                if n in used:
                    lines.append(f" {n}, {_num(xs[i])}, {_num(ys[j])}, {_num(k * dz)}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=CHANNEL")
    for e, i, j in cells:
        conn = [
            nid(i, j, 0),
            nid(i + 1, j, 0),
            nid(i + 1, j + 1, 0),
            nid(i, j + 1, 0),
            nid(i, j, 1),
            nid(i + 1, j, 1),
            nid(i + 1, j + 1, 1),
            nid(i, j + 1, 1),
        ]
        lines.append(f" {e}, " + ", ".join(str(c) for c in conn))

    # 面ラベル（C3D8、底面 4 節点が z=0）: S1 = z−, S2 = z+, S3 = y−, S4 = x+, S5 = y+, S6 = x−
    def _elset(name: str, ids: list[int]) -> None:
        lines.append(f"*ELSET, ELSET={name}")
        for a in range(0, len(ids), 8):
            lines.append(" " + ", ".join(str(v) for v in ids[a : a + 8]))

    _elset("EXMIN", [e for e, i, _ in cells if i == 0])
    _elset("EXMAX", [e for e, i, _ in cells if i == nx - 1])
    _elset("EBARREL", [e for e, _, j in cells if j == ny - 1])
    for name, elset, face in (
        ("XPER0", "EXMIN", "S6"),
        ("XPER1", "EXMAX", "S4"),
        ("ZPER0", "CHANNEL", "S1"),
        ("ZPER1", "CHANNEL", "S2"),
        ("BARREL", "EBARREL", "S5"),
    ):
        lines.append(f"*SURFACE, NAME={name}")
        lines.append(f" {elset}, {face}")

    # 周期境界: x は 1 ピッチ W_t、z は 1 セル厚 dz（∂/∂z = 0 を厳密にし w を自由にする）
    lines.append("*BOUNDARY, TYPE=PERIODIC")
    lines.append(f" XPER0, XPER1, {_num(spec.W_t)}, 0.0, 0.0")
    lines.append("*BOUNDARY, TYPE=PERIODIC")
    lines.append(f" ZPER0, ZPER1, 0.0, 0.0, {_num(dz)}")

    lines.append("*MATERIAL, NAME=MELT")
    lines.append("*DENSITY")
    lines.append(f" {_num(input_data.density)}")
    lines.extend(_viscosity_keyword(input_data.viscosity))
    lines.append("*FLUID SECTION, ELSET=CHANNEL, MATERIAL=MELT")

    # 圧力跳び Δp = G·L_turn を P = βx + p̃ に分解した一様体積力（設計文書 §2.1）
    body = (-spec.beta(input_data.G), 0.0, -input_data.G)
    barrel = (spec.u_barrel, 0.0, spec.w_barrel)
    lines.append("*STEP, NAME=channel")
    lines.append("*NAVIER STOKES, STEADY STATE, HEAT TRANSFER=NONE")
    lines.append("*CONTROLS, PARAMETERS=DISCRETIZATION")
    lines.append(f" CONVECTION=NONE, PRESSURE_VELOCITY={input_data.coupling}")
    lines.append("*CONTROLS, PARAMETERS=RELAXATION")
    lines.append(f" VISCOSITY={_num(input_data.alpha_mu)}")
    lines.append("*CONTROLS, PARAMETERS=SOLVER")
    lines.append(f" TOL={_num(input_data.tol)}, MAX_OUTER={input_data.max_outer}")
    lines.append("*BOUNDARY, TYPE=VELOCITY")
    lines.append(f" BARREL, {_num(barrel[0])}, {_num(barrel[1])}, {_num(barrel[2])}")
    lines.append("*DLOAD")
    lines.append(f" CHANNEL, BF, {_num(body[0])}, {_num(body[1])}, {_num(body[2])}")
    lines.append(f"*OUTPUT, FIELD, FORMAT={input_data.output_format}")
    lines.append("*END STEP")
    return ExtruderInpResult(
        text="\n".join(lines) + "\n",
        grid=grid,
        n_cells=len(cells),
        n_nodes=len(used),
        depth_z=dz,
        body_force=body,
        barrel_velocity=barrel,
    )


def axial_throughput(
    velocity: np.ndarray, cell_volumes: np.ndarray, depth_z: float, spec: ScrewSpec
) -> tuple[float, float, float]:
    """汎用経路のセル場から (Q, Q_leak, Q_axial) [m³/s] を求める.

    - ``Q = ∫∫ w dA``（下流方向の押出量）
    - ``Q_leak = (1/W_t) ∫∫ u dA``（断面内は 2D 非圧縮なのでどの x 面でも同じ横断流束）
    - ``Q_axial = Q + L_turn·Q_leak``（設計文書 §2.1.2 の恒等式）
    """
    area = np.asarray(cell_volumes, dtype=np.float64) / depth_z
    u = np.asarray(velocity, dtype=np.float64)
    q = float(np.sum(u[:, 2] * area))
    q_leak = float(np.sum(u[:, 0] * area)) / spec.W_t
    return q, q_leak, q + spec.L_turn * q_leak


class ExtruderChannelInpProcess(PreProcess["ExtruderInpInput", "ExtruderInpResult"]):
    """スクリュー諸元 → 汎用記法（非構造 NS FVM）の .inp テキストを書き出す PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ExtruderChannelInp",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-generic-extrusion.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [ScrewGeometryProcess]

    def process(self, input_data: ExtruderInpInput) -> ExtruderInpResult:
        if input_data.depth_z < 0.0:
            raise ValueError(f"depth_z は 0 以上（0 なら自動）: {input_data.depth_z}")
        if not math.isfinite(input_data.G):
            raise ValueError("G が有限値ではありません")
        grid = ScrewGeometryProcess().execute(input_data.spec)
        return build_extruder_inp(input_data, grid)


__all__ = [
    "ExtruderInpInput",
    "ExtruderInpResult",
    "ExtruderChannelInpProcess",
    "build_extruder_inp",
    "axial_throughput",
]
