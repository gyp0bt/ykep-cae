"""ソルバー非依存の中立ケース定義（``CaseDefinition``）.

.inp のキーワードを意味付けした結果。ykep の各ソルバー Input への変換
（:mod:`xkep_cae_fluid.inp.mapping`）と、将来の OpenFOAM / Fluent 書き出しは
すべてこのデータ構造を起点にする。座標は常に 3 成分（2D は z=0 を補う）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# 幾何・集合
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeTable:
    """節点表."""

    ids: np.ndarray  # (n,) int
    coords: np.ndarray  # (n, 3) float

    @property
    def n_nodes(self) -> int:
        return int(self.ids.shape[0])


@dataclass(frozen=True)
class ElementBlock:
    """``*ELEMENT`` 1 ブロック（同一要素タイプ）."""

    element_type: str  # 例: "C3D8", "CPS4", "FC3D8"
    ids: np.ndarray  # (n,) int
    connectivity: np.ndarray  # (n, n_nodes_per_element) int（節点 ID）
    elset: str | None = None

    @property
    def n_elements(self) -> int:
        return int(self.ids.shape[0])

    @property
    def nodes_per_element(self) -> int:
        return int(self.connectivity.shape[1])

    @property
    def is_3d(self) -> bool:
        return self.nodes_per_element == 8


class SetKind(Enum):
    NODE = "node"
    ELEMENT = "element"


@dataclass(frozen=True)
class SetDefinition:
    """``*NSET`` / ``*ELSET``（GENERATE・他集合参照は解決済み）."""

    name: str
    kind: SetKind
    ids: np.ndarray  # (n,) int（昇順・重複なし）


@dataclass(frozen=True)
class SurfaceEntry:
    """``*SURFACE, TYPE=ELEMENT`` の 1 行: 要素集合（または要素 ID）と面ラベル."""

    target: str  # elset 名 または 要素 ID の文字列
    face: str  # "S1".."S6"（2D は "S1".."S4"）


@dataclass(frozen=True)
class SurfaceDefinition:
    """``*SURFACE``."""

    name: str
    entries: tuple[SurfaceEntry, ...]


# ---------------------------------------------------------------------------
# 物性・セクション
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialDefinition:
    """``*MATERIAL`` とそのサブキーワード（SI 単位固定）.

    Parameters
    ----------
    density : float | None
        ``*DENSITY`` [kg/m³]
    viscosity : float | None
        ``*VISCOSITY`` 粘度 [Pa·s]
    conductivity : float | None
        ``*CONDUCTIVITY`` [W/(m·K)]
    specific_heat : float | None
        ``*SPECIFIC HEAT`` [J/(kg·K)]
    expansion : float | None
        ``*EXPANSION`` 体膨張係数 [1/K]（Boussinesq）
    reference_temperature : float | None
        ``*EXPANSION, ZERO=`` 基準温度 [K]
    permeability : float | None
        ``*PERMEABILITY`` [m²]（Darcy 用、現状は保持のみ）
    """

    name: str
    density: float | None = None
    viscosity: float | None = None
    conductivity: float | None = None
    specific_heat: float | None = None
    expansion: float | None = None
    reference_temperature: float | None = None
    permeability: float | None = None

    def require(self, attr: str) -> float:
        value = getattr(self, attr)
        if value is None:
            raise ValueError(f"材料 {self.name!r} に {attr} が定義されていません")
        return float(value)


class SectionKind(Enum):
    FLUID = "fluid"
    SOLID = "solid"


@dataclass(frozen=True)
class SectionDefinition:
    """``*FLUID SECTION`` / ``*SOLID SECTION``."""

    kind: SectionKind
    elset: str
    material: str


# ---------------------------------------------------------------------------
# 初期条件・境界条件・荷重
# ---------------------------------------------------------------------------


class InitialConditionKind(Enum):
    TEMPERATURE = "temperature"
    VELOCITY = "velocity"
    PRESSURE = "pressure"


@dataclass(frozen=True)
class InitialCondition:
    """``*INITIAL CONDITIONS``. target は nset / elset 名、要素・節点 ID、または ``ALL``."""

    kind: InitialConditionKind
    target: str
    values: tuple[float, ...]


class BoundaryKind(Enum):
    """``*BOUNDARY, TYPE=`` の語彙（自由度番号形式からも変換される）."""

    WALL = "wall"  # 速度 0（no-slip）
    SLIP = "slip"  # すべり壁
    VELOCITY = "velocity"  # 速度指定（流入）
    PRESSURE = "pressure"  # 圧力指定（流出）
    OUTLET = "outlet"  # 対流流出（非反射）
    SYMMETRY = "symmetry"
    TEMPERATURE = "temperature"  # 温度固定


@dataclass(frozen=True)
class BoundaryCondition:
    """``*BOUNDARY`` の 1 行."""

    target: str  # surface 名（予約名 XM/XP/YM/YP/ZM/ZP を含む）
    kind: BoundaryKind
    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FilmCondition:
    """``*SFILM``: 対流熱伝達 q = h (T_inf - T)."""

    target: str
    h: float
    t_inf: float


class FluxLabel(Enum):
    SURFACE = "S"  # 面熱流束 [W/m²]（正=流入）
    BODY = "BF"  # 体積発熱 [W/m³]


@dataclass(frozen=True)
class DistributedFlux:
    """``*DFLUX`` の 1 行."""

    target: str  # S: surface 名 / BF: elset 名
    label: FluxLabel
    magnitude: float


@dataclass(frozen=True)
class DistributedLoad:
    """``*DLOAD`` の 1 行（現状 GRAV のみ）."""

    target: str
    label: str  # "GRAV"
    magnitude: float
    direction: tuple[float, float, float]

    @property
    def vector(self) -> tuple[float, float, float]:
        d = np.asarray(self.direction, dtype=float)
        norm = float(np.linalg.norm(d))
        if norm == 0.0:
            raise ValueError("*DLOAD, GRAV の方向ベクトルがゼロです")
        g = self.magnitude * d / norm
        return (float(g[0]), float(g[1]), float(g[2]))


# ---------------------------------------------------------------------------
# ステップ
# ---------------------------------------------------------------------------


class EquationFamily(Enum):
    """方程式ファミリー（手続きキーワード）."""

    NAVIER_STOKES = "NAVIER STOKES"
    HEAT_TRANSFER = "HEAT TRANSFER"
    DARCY = "DARCY"


@dataclass(frozen=True)
class Procedure:
    """``*NAVIER STOKES`` / ``*HEAT TRANSFER`` / ``*DARCY``.

    Parameters
    ----------
    family : EquationFamily
    steady : bool
        ``STEADY STATE`` フラグ
    turbulence : str
        ``TURBULENCE=``（現状 ``LAMINAR`` のみ）
    heat_transfer : str
        ``HEAT TRANSFER=NONE|COUPLED``（NAVIER STOKES のみ）
    dt : float
        非定常の時間刻み [s]（データ行 1 列目）
    time_period : float
        非定常の解析時間 [s]（データ行 2 列目）
    """

    family: EquationFamily
    steady: bool = True
    turbulence: str = "LAMINAR"
    heat_transfer: str = "NONE"
    dt: float = 0.0
    time_period: float = 0.0


class ControlCategory(Enum):
    DISCRETIZATION = "DISCRETIZATION"
    RELAXATION = "RELAXATION"
    SOLVER = "SOLVER"
    TIME_INCREMENTATION = "TIME INCREMENTATION"


@dataclass(frozen=True)
class ControlSet:
    """``*CONTROLS, PARAMETERS=...``。データ行の ``KEY=VALUE`` を大文字キーで保持."""

    category: ControlCategory
    values: Mapping[str, str] = field(default_factory=dict)


class OutputFormat(Enum):
    NPZ = "NPZ"
    VTK = "VTK"
    HTML = "HTML"  # messi mirador（three.js）3D ビューア


@dataclass(frozen=True)
class OutputRequest:
    """``*OUTPUT, FIELD``（+ ``*ELEMENT OUTPUT`` / ``*NODE OUTPUT`` の変数リスト）."""

    variables: tuple[str, ...] = ()  # 空 = 全変数
    formats: tuple[OutputFormat, ...] = (OutputFormat.NPZ,)
    frequency: int = 1
    formats_explicit: bool = False  # FORMAT= を書いた（書かなければ messi があれば HTML も自動）


@dataclass(frozen=True)
class StepDefinition:
    """``*STEP`` … ``*END STEP``."""

    name: str
    procedure: Procedure
    controls: tuple[ControlSet, ...] = ()
    boundaries: tuple[BoundaryCondition, ...] = ()
    films: tuple[FilmCondition, ...] = ()
    fluxes: tuple[DistributedFlux, ...] = ()
    loads: tuple[DistributedLoad, ...] = ()
    outputs: tuple[OutputRequest, ...] = ()
    max_increments: int = 0  # *STEP, INC=

    def control_values(self, category: ControlCategory) -> dict[str, str]:
        merged: dict[str, str] = {}
        for c in self.controls:
            if c.category == category:
                merged.update(c.values)
        return merged


# ---------------------------------------------------------------------------
# ケース全体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseDefinition:
    """.inp 全体の中立表現.

    Parameters
    ----------
    heading : str
        ``*HEADING`` の自由文
    nodes : NodeTable
    elements : tuple[ElementBlock, ...]
    nsets, elsets : Mapping[str, SetDefinition]
        名前は大文字に正規化
    surfaces : Mapping[str, SurfaceDefinition]
    materials : Mapping[str, MaterialDefinition]
    sections : tuple[SectionDefinition, ...]
    initial_conditions : tuple[InitialCondition, ...]
    boundaries : tuple[BoundaryCondition, ...]
        モデルレベル（``*STEP`` の外）の境界条件。全ステップに適用
    films, fluxes, loads : モデルレベルの ``*SFILM`` / ``*DFLUX`` / ``*DLOAD``
    steps : tuple[StepDefinition, ...]
    parameters : Mapping[str, object]
        ``*PARAMETER`` の最終値（記録用）
    source : str
        由来ファイル
    """

    heading: str
    nodes: NodeTable
    elements: tuple[ElementBlock, ...]
    nsets: Mapping[str, SetDefinition] = field(default_factory=dict)
    elsets: Mapping[str, SetDefinition] = field(default_factory=dict)
    surfaces: Mapping[str, SurfaceDefinition] = field(default_factory=dict)
    materials: Mapping[str, MaterialDefinition] = field(default_factory=dict)
    sections: tuple[SectionDefinition, ...] = ()
    initial_conditions: tuple[InitialCondition, ...] = ()
    boundaries: tuple[BoundaryCondition, ...] = ()
    films: tuple[FilmCondition, ...] = ()
    fluxes: tuple[DistributedFlux, ...] = ()
    loads: tuple[DistributedLoad, ...] = ()
    steps: tuple[StepDefinition, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)
    source: str = ""

    @property
    def n_elements(self) -> int:
        return sum(b.n_elements for b in self.elements)

    @property
    def is_3d(self) -> bool:
        return all(b.is_3d for b in self.elements)

    def element_ids_of(self, target: str) -> np.ndarray:
        """elset 名・要素 ID・``ALL`` を要素 ID 配列に解決する."""
        key = target.strip().upper()
        if key == "ALL":
            return (
                np.concatenate([b.ids for b in self.elements])
                if self.elements
                else np.zeros(0, int)
            )
        if key in self.elsets:
            return self.elsets[key].ids
        if key.isdigit():
            return np.array([int(key)], dtype=int)
        raise KeyError(f"要素集合 {target!r} が定義されていません")

    def node_ids_of(self, target: str) -> np.ndarray:
        """nset 名・節点 ID・``ALL`` を節点 ID 配列に解決する."""
        key = target.strip().upper()
        if key == "ALL":
            return self.nodes.ids
        if key in self.nsets:
            return self.nsets[key].ids
        if key.isdigit():
            return np.array([int(key)], dtype=int)
        raise KeyError(f"節点集合 {target!r} が定義されていません")

    def material_of_section(self, section: SectionDefinition) -> MaterialDefinition:
        key = section.material.upper()
        if key not in self.materials:
            raise KeyError(f"材料 {section.material!r} が定義されていません")
        return self.materials[key]
