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


class OrientationSystem(Enum):
    """``*ORIENTATION, SYSTEM=`` の語彙."""

    RECTANGULAR = "RECTANGULAR"
    CYLINDRICAL = "CYLINDRICAL"


@dataclass(frozen=True)
class OrientationDefinition:
    """``*ORIENTATION``: 速度・角速度の成分を解釈する局所座標系.

    データ行は Abaqus と同じ ``ax, ay, az, bx, by, bz``。

    - ``SYSTEM=RECTANGULAR``（既定）: a は局所 1 軸上の点、b は局所 1–2 平面上の点
    - ``SYSTEM=CYLINDRICAL``: a と b は**軸上の 2 点**。局所 3 軸が軸方向（b − a）になる
      （回転体の軸。局所 1・2 軸は軸に直交する任意の正規直交基底）

    Parameters
    ----------
    name : str
    system : OrientationSystem
    point_a, point_b : tuple[float, float, float]
    """

    name: str
    system: OrientationSystem
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]

    def basis(self) -> np.ndarray:
        """局所基底 (3, 3)（行が局所 1/2/3 軸の単位ベクトル、右手系）を返す."""
        a = np.asarray(self.point_a, dtype=float)
        b = np.asarray(self.point_b, dtype=float)
        if self.system == OrientationSystem.CYLINDRICAL:
            axis = b - a
            norm = float(np.linalg.norm(axis))
            if norm == 0.0:
                raise ValueError(f"*ORIENTATION {self.name}: 軸上の 2 点が同一です")
            e3 = axis / norm
            # 軸に直交する任意の単位ベクトル（最も軸成分の小さい全体軸から作る）
            seed = np.zeros(3)
            seed[int(np.argmin(np.abs(e3)))] = 1.0
            e1 = seed - float(np.dot(seed, e3)) * e3
            e1 /= float(np.linalg.norm(e1))
            return np.stack([e1, np.cross(e3, e1), e3])
        n_a = float(np.linalg.norm(a))
        if n_a == 0.0:
            raise ValueError(f"*ORIENTATION {self.name}: 局所 1 軸の点が原点です")
        e1 = a / n_a
        v = b - float(np.dot(b, e1)) * e1
        n_v = float(np.linalg.norm(v))
        if n_v == 0.0:
            raise ValueError(f"*ORIENTATION {self.name}: 局所 1–2 平面の点が 1 軸上にあります")
        e2 = v / n_v
        return np.stack([e1, e2, np.cross(e1, e2)])

    def to_global(self, components: tuple[float, ...]) -> tuple[float, float, float]:
        """局所成分 (v1, v2, v3) を全体座標系のベクトルにする."""
        v = np.zeros(3)
        v[: len(components)] = np.asarray(components, dtype=float)[:3]
        g = v @ self.basis()
        return (float(g[0]), float(g[1]), float(g[2]))


class MPCKind(Enum):
    """``*MPC`` の拘束種別（面を参照節点の剛体運動に従わせる）."""

    BEAM = "BEAM"  # 並進 + 回転を伝える剛体リンク（Abaqus 互換）
    RIGID = "RIGID"  # BEAM の別名
    TIE = "TIE"  # BEAM の別名


@dataclass(frozen=True)
class MPCDefinition:
    """``*MPC``: 面（従属）を参照節点（独立）の剛体運動に拘束する.

    データ行は ``kind, slave, master``。``slave`` は ``*SURFACE`` 名か予約面名、
    ``master`` は参照節点の ``*NSET`` 名か節点 ID。面上の速度は

        u(x) = v_ref + ω_ref × (x − x_ref)

    になる（``v_ref`` / ``ω_ref`` は参照節点への ``*BOUNDARY`` の自由度 1-3 / 4-6）。
    """

    kind: MPCKind
    slave: str
    master: str


@dataclass(frozen=True)
class PeriodicDefinition:
    """``*BOUNDARY, TYPE=PERIODIC`` の 1 行: 2 つの面を並進で対応付ける周期境界.

    Parameters
    ----------
    master, slave : str
        ``*SURFACE`` 名（または予約面名 XM..ZP）。slave の面は master の面を ``translation`` だけ
        並進した位置にある（x_slave = x_master + t）
    translation : tuple[float, float, float] | None
        並進ベクトル [m]。None なら両面の面中心の平均の差から自動決定
    """

    master: str
    slave: str
    translation: tuple[float, float, float] | None = None


# ---------------------------------------------------------------------------
# 物性・セクション
# ---------------------------------------------------------------------------


class ViscosityModel(Enum):
    """``*VISCOSITY, TYPE=`` の語彙."""

    NEWTONIAN = "NEWTONIAN"
    POWER_LAW = "POWER LAW"
    CARREAU = "CARREAU"


@dataclass(frozen=True)
class ViscosityLaw:
    """``*VISCOSITY, TYPE=POWER LAW | CARREAU`` の粘度モデルとパラメータ.

    Parameters
    ----------
    model : ViscosityModel
    parameters : tuple[float, ...]
        POWER LAW: ``K, n[, gamma_min, mu_max]``（μ = K γ̇^(n−1)）、
        CARREAU: ``mu_0, mu_inf, lambda, n``
    """

    model: ViscosityModel
    parameters: tuple[float, ...]

    @property
    def nominal_viscosity(self) -> float:
        """参照粘度 [Pa·s]（POWER LAW は K = μ(γ̇=1)、CARREAU は μ_0）。ログと初期化に使う."""
        return float(self.parameters[0])


@dataclass(frozen=True)
class MaterialDefinition:
    """``*MATERIAL`` とそのサブキーワード（SI 単位固定）.

    Parameters
    ----------
    density : float | None
        ``*DENSITY`` [kg/m³]
    viscosity : float | None
        ``*VISCOSITY`` 粘度 [Pa·s]（非ニュートンでは参照粘度 ``ViscosityLaw.nominal_viscosity``）
    viscosity_law : ViscosityLaw | None
        ``*VISCOSITY, TYPE=POWER LAW | CARREAU``（None ならニュートン）
    conductivity : float | None
        ``*CONDUCTIVITY`` [W/(m·K)]
    specific_heat : float | None
        ``*SPECIFIC HEAT`` [J/(kg·K)]
    expansion : float | None
        ``*EXPANSION`` 体膨張係数 [1/K]（Boussinesq）
    reference_temperature : float | None
        ``*EXPANSION, ZERO=`` 基準温度 [K]
    permeability : float | None
        ``*PERMEABILITY`` [m²]（Darcy 用）
    forchheimer, specific_storage : float | None
        ``*FORCHHEIMER`` β [1/m] / ``*SPECIFIC STORAGE`` S_s [1/Pa]（Darcy 用）
    """

    name: str
    density: float | None = None
    viscosity: float | None = None
    viscosity_law: ViscosityLaw | None = None
    conductivity: float | None = None
    specific_heat: float | None = None
    expansion: float | None = None
    reference_temperature: float | None = None
    permeability: float | None = None
    forchheimer: float | None = None  # ``*FORCHHEIMER`` β [1/m]（Darcy の慣性補正）
    specific_storage: float | None = None  # ``*SPECIFIC STORAGE`` S_s [1/Pa]（非定常 Darcy）

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
    ROTATION = "rotation"  # 角速度 [rad/s]（自由度 4-6。参照節点に与えて *MPC で面に伝える）


@dataclass(frozen=True)
class BoundaryCondition:
    """``*BOUNDARY`` の 1 行.

    Parameters
    ----------
    target : str
        ``*SURFACE`` 名（予約名 XM/XP/YM/YP/ZM/ZP を含む）、``*ELSET`` 名（内部の吐出・吸入）、
        ``*NSET`` 名 / 節点 ID（参照節点。``*MPC`` で面に伝える）
    kind : BoundaryKind
    values : tuple[float, ...]
        速度なら (ux, uy, uz)、角速度なら (ωx, ωy, ωz) [rad/s]、圧力・温度なら 1 値
    orientation : str
        ``ORIENTATION=`` で指定した ``*ORIENTATION`` 名（空なら全体座標系）。
        速度・角速度の成分をその局所座標系で解釈する
    """

    target: str
    kind: BoundaryKind
    values: tuple[float, ...] = ()
    orientation: str = ""


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


BODY_FORCE_LABELS: tuple[str, ...] = ("BX", "BY", "BZ", "BF")
"""``*DLOAD`` の体積力ラベル（Abaqus の BX/BY/BZ = 成分ごと、BF = ベクトル 3 成分）."""


@dataclass(frozen=True)
class DistributedLoad:
    """``*DLOAD`` の 1 行.

    - ``GRAV``: 重力（``magnitude`` × 方向余弦 ``direction``）
    - ``BX`` / ``BY`` / ``BZ``: 体積力の 1 成分 [N/m³]（``direction`` は単位軸、``magnitude`` が値）
    - ``BF``: 体積力ベクトル [N/m³]（``direction`` にそのまま入り、``magnitude`` = 1）
    """

    target: str
    label: str  # "GRAV" / "BX" / "BY" / "BZ" / "BF"
    magnitude: float
    direction: tuple[float, float, float]

    @property
    def is_body_force(self) -> bool:
        return self.label in BODY_FORCE_LABELS

    @property
    def vector(self) -> tuple[float, float, float]:
        """重力なら大きさ × 単位方向、体積力ならベクトルそのもの [N/m³]."""
        d = np.asarray(self.direction, dtype=float)
        if self.is_body_force:
            g = self.magnitude * d
            return (float(g[0]), float(g[1]), float(g[2]))
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
    periodic : tuple[PeriodicDefinition, ...]
        ``*BOUNDARY, TYPE=PERIODIC``（モデルレベルのみ。メッシュの位相なので全ステップ共通）
    orientations : Mapping[str, OrientationDefinition]
        ``*ORIENTATION``（名前は大文字に正規化）
    mpcs : tuple[MPCDefinition, ...]
        ``*MPC``（面 → 参照節点の剛体拘束）
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
    periodic: tuple[PeriodicDefinition, ...] = ()
    orientations: Mapping[str, OrientationDefinition] = field(default_factory=dict)
    mpcs: tuple[MPCDefinition, ...] = ()
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
