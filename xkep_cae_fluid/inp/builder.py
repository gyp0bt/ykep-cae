"""``KeywordBlock`` 列から :class:`CaseDefinition` を組み立てる.

キーワードごとの意味付け（``*NODE`` → 節点表、``*STEP`` → ステップ等）を担当する。
未知のキーワードは警告して無視し、ykep が解釈できない値（乱流モデル等）は
ここではなく mapping 段で検証する（フォーマットとしては保持できるようにするため）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.inp.case import (
    BoundaryCondition,
    BoundaryKind,
    CaseDefinition,
    ControlCategory,
    ControlSet,
    DistributedFlux,
    DistributedLoad,
    ElementBlock,
    EquationFamily,
    FilmCondition,
    FluxLabel,
    InitialCondition,
    InitialConditionKind,
    MaterialDefinition,
    MPCDefinition,
    MPCKind,
    NodeTable,
    OrientationDefinition,
    OrientationSystem,
    OutputFormat,
    OutputRequest,
    PeriodicDefinition,
    Procedure,
    SectionDefinition,
    SectionKind,
    SetDefinition,
    SetKind,
    StepDefinition,
    SurfaceDefinition,
    SurfaceEntry,
    ViscosityLaw,
    ViscosityModel,
)
from xkep_cae_fluid.inp.parser import InpParseResult, InpSyntaxError, KeywordBlock

logger = logging.getLogger(__name__)

# 予約面名（構造格子の 6 面）。*GRID 使用時は自動定義され、*BOUNDARY 等の target に直接書ける
RESERVED_FACE_NAMES: tuple[str, ...] = ("XM", "XP", "YM", "YP", "ZM", "ZP")
FACE_ALIASES: dict[str, str] = {
    "WEST": "XM",
    "EAST": "XP",
    "SOUTH": "YM",
    "NORTH": "YP",
    "BOTTOM": "ZM",
    "TOP": "ZP",
    "X-": "XM",
    "X+": "XP",
    "Y-": "YM",
    "Y+": "YP",
    "Z-": "ZM",
    "Z+": "ZP",
}

# 自由度番号 → 境界条件種別（Abaqus 互換: 1-3 速度, 8 圧力, 11 温度）
_DOF_KINDS: dict[int, BoundaryKind] = {
    1: BoundaryKind.VELOCITY,
    2: BoundaryKind.VELOCITY,
    3: BoundaryKind.VELOCITY,
    8: BoundaryKind.PRESSURE,
    11: BoundaryKind.TEMPERATURE,
}

_PROCEDURE_KEYWORDS: dict[str, EquationFamily] = {
    "NAVIER STOKES": EquationFamily.NAVIER_STOKES,
    "HEAT TRANSFER": EquationFamily.HEAT_TRANSFER,
    "DARCY": EquationFamily.DARCY,
}

_MATERIAL_SUBKEYWORDS: dict[str, str] = {
    "DENSITY": "density",
    "VISCOSITY": "viscosity",
    "CONDUCTIVITY": "conductivity",
    "SPECIFIC HEAT": "specific_heat",
    "EXPANSION": "expansion",
    "PERMEABILITY": "permeability",
    "FORCHHEIMER": "forchheimer",
    "SPECIFIC STORAGE": "specific_storage",
}

_IGNORED_KEYWORDS: frozenset[str] = frozenset(
    {"PREPRINT", "END ASSEMBLY", "ASSEMBLY", "PART", "END PART", "INSTANCE", "END INSTANCE"}
)


def _norm_name(name: str) -> str:
    return name.strip().upper()


def _float(token: str, block: KeywordBlock, what: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise InpSyntaxError(
            f"{what} の数値が不正: {token!r}", block.source, block.line_no
        ) from exc


def _int(token: str, block: KeywordBlock, what: str) -> int:
    try:
        return int(float(token))
    except ValueError as exc:
        raise InpSyntaxError(
            f"{what} の整数が不正: {token!r}", block.source, block.line_no
        ) from exc


@dataclass
class _Builder:
    """組み立て途中の可変状態（プロセス内部専用）."""

    source: str
    heading: str = ""
    node_ids: list[int] | None = None
    node_coords: list[list[float]] | None = None
    elements: list[ElementBlock] | None = None
    nsets: dict[str, list[int]] | None = None
    elsets: dict[str, list[int]] | None = None
    surfaces: dict[str, SurfaceDefinition] | None = None
    materials: dict[str, MaterialDefinition] | None = None
    sections: list[SectionDefinition] | None = None
    initial_conditions: list[InitialCondition] | None = None
    boundaries: list[BoundaryCondition] | None = None
    films: list[FilmCondition] | None = None
    fluxes: list[DistributedFlux] | None = None
    loads: list[DistributedLoad] | None = None
    periodic: list[PeriodicDefinition] | None = None
    orientations: dict[str, OrientationDefinition] | None = None
    mpcs: list[MPCDefinition] | None = None
    steps: list[StepDefinition] | None = None
    grid_defined: bool = False

    def __post_init__(self) -> None:
        self.node_ids = []
        self.node_coords = []
        self.elements = []
        self.nsets = {}
        self.elsets = {}
        self.surfaces = {}
        self.materials = {}
        self.sections = []
        self.initial_conditions = []
        self.boundaries = []
        self.films = []
        self.fluxes = []
        self.loads = []
        self.periodic = []
        self.orientations = {}
        self.mpcs = []
        self.steps = []


class _StepBuilder:
    def __init__(self, block: KeywordBlock) -> None:
        self.block = block
        self.name = block.get("NAME", "") or f"Step-{block.line_no}"
        self.max_increments = _int(block.get("INC", "0") or "0", block, "INC")
        self.procedure: Procedure | None = None
        self.controls: list[ControlSet] = []
        self.boundaries: list[BoundaryCondition] = []
        self.films: list[FilmCondition] = []
        self.fluxes: list[DistributedFlux] = []
        self.loads: list[DistributedLoad] = []
        self.outputs: list[OutputRequest] = []
        self._pending_output: OutputRequest | None = None

    def finish(self) -> StepDefinition:
        if self.procedure is None:
            raise InpSyntaxError(
                f"*STEP {self.name!r} に手続きキーワード（*NAVIER STOKES 等）がありません",
                self.block.source,
                self.block.line_no,
            )
        self._flush_output()
        return StepDefinition(
            name=self.name,
            procedure=self.procedure,
            controls=tuple(self.controls),
            boundaries=tuple(self.boundaries),
            films=tuple(self.films),
            fluxes=tuple(self.fluxes),
            loads=tuple(self.loads),
            outputs=tuple(self.outputs),
            max_increments=self.max_increments,
        )

    def _flush_output(self) -> None:
        if self._pending_output is not None:
            self.outputs.append(self._pending_output)
            self._pending_output = None

    def add_output(self, req: OutputRequest) -> None:
        self._flush_output()
        self._pending_output = req

    def add_output_variables(self, variables: tuple[str, ...], block: KeywordBlock) -> None:
        if self._pending_output is None:
            raise InpSyntaxError(
                "*ELEMENT OUTPUT / *NODE OUTPUT は *OUTPUT, FIELD の後に置いてください",
                block.source,
                block.line_no,
            )
        merged = tuple(dict.fromkeys(self._pending_output.variables + variables))
        self._pending_output = OutputRequest(
            variables=merged,
            formats=self._pending_output.formats,
            frequency=self._pending_output.frequency,
        )


# ---------------------------------------------------------------------------
# キーワード別ハンドラ
# ---------------------------------------------------------------------------


def _reject_if_grid(b: _Builder, block: KeywordBlock) -> None:
    if b.grid_defined:
        raise InpSyntaxError(
            f"*{block.keyword} は *GRID と併用できません", block.source, block.line_no
        )


def _handle_node(b: _Builder, block: KeywordBlock) -> None:
    _reject_if_grid(b, block)
    nset = block.get("NSET")
    ids: list[int] = []
    for row in block.data:
        if len(row) < 2:
            raise InpSyntaxError(
                "*NODE のデータ行は 'id, x[, y[, z]]'", block.source, block.line_no
            )
        nid = _int(row[0], block, "節点 ID")
        xyz = [_float(t, block, "節点座標") for t in row[1:4]]
        while len(xyz) < 3:
            xyz.append(0.0)
        b.node_ids.append(nid)
        b.node_coords.append(xyz)
        ids.append(nid)
    if nset:
        b.nsets.setdefault(_norm_name(nset), []).extend(ids)


def _handle_element(b: _Builder, block: KeywordBlock) -> None:
    _reject_if_grid(b, block)
    etype = _norm_name(block.require("TYPE"))
    elset = block.get("ELSET")
    ids: list[int] = []
    conn: list[list[int]] = []
    width: int | None = None
    for row in block.data:
        if len(row) < 3:
            raise InpSyntaxError(
                "*ELEMENT のデータ行は 'id, n1, n2, ...'", block.source, block.line_no
            )
        ids.append(_int(row[0], block, "要素 ID"))
        nodes = [_int(t, block, "要素節点") for t in row[1:]]
        if width is None:
            width = len(nodes)
        elif len(nodes) != width:
            raise InpSyntaxError(
                "*ELEMENT の節点数が行ごとに異なります", block.source, block.line_no
            )
        conn.append(nodes)
    if not ids:
        return
    allowed = (4, 5, 6, 8, 10, 15, 20) if "3D" in etype else (3, 4, 6, 8)
    if width not in allowed:
        raise InpSyntaxError(
            f"要素タイプ {etype}（節点数 {width}）は未対応です。"
            "3D は四面体 C3D4/C3D10 / 角錐 C3D5 / 楔 C3D6/C3D15 / 六面体 C3D8/C3D20 系"
            "（4 / 10 / 5 / 6 / 15 / 8 / 20 節点）、"
            "2D は三角形 CPS3/CPS6 / 四辺形 CPS4/CPS8 系（3 / 6 / 4 / 8 節点）のみ"
            "（2 次要素は頂点だけを使う。非構造メッシュ経路のみ）",
            block.source,
            block.line_no,
        )
    b.elements.append(
        ElementBlock(
            element_type=etype,
            ids=np.asarray(ids, dtype=int),
            connectivity=np.asarray(conn, dtype=int),
            elset=_norm_name(elset) if elset else None,
        )
    )
    if elset:
        b.elsets.setdefault(_norm_name(elset), []).extend(ids)


def _resolve_set_rows(block: KeywordBlock, existing: dict[str, list[int]], what: str) -> list[int]:
    ids: list[int] = []
    if block.has("GENERATE"):
        for row in block.data:
            if len(row) < 2:
                raise InpSyntaxError(
                    f"{what}, GENERATE のデータ行は 'first, last[, inc]'",
                    block.source,
                    block.line_no,
                )
            first = _int(row[0], block, "first")
            last = _int(row[1], block, "last")
            inc = _int(row[2], block, "inc") if len(row) > 2 and row[2] else 1
            if inc <= 0:
                raise InpSyntaxError("GENERATE の増分は正", block.source, block.line_no)
            ids.extend(range(first, last + 1, inc))
        return ids
    for row in block.data:
        for token in row:
            if not token:
                continue
            key = _norm_name(token)
            if key in existing:
                ids.extend(existing[key])
            elif key.lstrip("-").isdigit():
                ids.append(int(key))
            else:
                raise InpSyntaxError(
                    f"{what} の項目 {token!r} は ID でも既存集合でもありません",
                    block.source,
                    block.line_no,
                )
    return ids


def _handle_nset(b: _Builder, block: KeywordBlock) -> None:
    name = _norm_name(block.require("NSET"))
    b.nsets.setdefault(name, []).extend(_resolve_set_rows(block, b.nsets, "*NSET"))


def _handle_elset(b: _Builder, block: KeywordBlock) -> None:
    name = _norm_name(block.require("ELSET"))
    b.elsets.setdefault(name, []).extend(_resolve_set_rows(block, b.elsets, "*ELSET"))


def _handle_surface(b: _Builder, block: KeywordBlock) -> None:
    name = _norm_name(block.require("NAME"))
    stype = _norm_name(block.get("TYPE", "ELEMENT") or "ELEMENT")
    if stype != "ELEMENT":
        raise InpSyntaxError(
            f"*SURFACE, TYPE={stype} は未対応（ELEMENT のみ）", block.source, block.line_no
        )
    entries: list[SurfaceEntry] = []
    for row in block.data:
        if len(row) < 2:
            raise InpSyntaxError(
                "*SURFACE のデータ行は 'elset_or_id, S#'", block.source, block.line_no
            )
        face = _norm_name(row[1])
        if not (face.startswith("S") and face[1:].isdigit()):
            raise InpSyntaxError(f"面ラベルが不正: {row[1]!r}", block.source, block.line_no)
        entries.append(SurfaceEntry(target=_norm_name(row[0]), face=face))
    if name in b.surfaces:
        entries = list(b.surfaces[name].entries) + entries
    b.surfaces[name] = SurfaceDefinition(name=name, entries=tuple(entries))


def _handle_material(b: _Builder, block: KeywordBlock) -> str:
    name = _norm_name(block.require("NAME"))
    if name not in b.materials:
        b.materials[name] = MaterialDefinition(name=name)
    return name


def _handle_material_property(
    b: _Builder, block: KeywordBlock, current_material: str | None
) -> None:
    if current_material is None:
        raise InpSyntaxError(
            f"*{block.keyword} は *MATERIAL の後に置いてください", block.source, block.line_no
        )
    if not block.data or not block.data[0]:
        raise InpSyntaxError(f"*{block.keyword} に値がありません", block.source, block.line_no)
    attr = _MATERIAL_SUBKEYWORDS[block.keyword]
    mat = b.materials[current_material]
    if block.keyword == "VISCOSITY":
        law = _parse_viscosity_law(block)
        if law is not None:
            b.materials[current_material] = replace(
                mat, viscosity=law.nominal_viscosity, viscosity_law=law
            )
            return
    value = _float(block.data[0][0], block, block.keyword)
    updates = {attr: value}
    if block.keyword == "EXPANSION":
        zero = block.get("ZERO")
        if zero:
            updates["reference_temperature"] = _float(zero, block, "ZERO")
    b.materials[current_material] = replace(mat, **updates)


_VISCOSITY_PARAMS: dict[ViscosityModel, tuple[int, int, str]] = {
    ViscosityModel.POWER_LAW: (2, 4, "K, n[, gamma_min, mu_max]"),
    ViscosityModel.CARREAU: (4, 4, "mu_0, mu_inf, lambda, n"),
}


def _parse_viscosity_law(block: KeywordBlock) -> ViscosityLaw | None:
    """``*VISCOSITY, TYPE=POWER LAW | CARREAU`` のデータ行。TYPE 無し / NEWTONIAN なら None."""
    type_text = _norm_name(block.get("TYPE", "NEWTONIAN") or "NEWTONIAN")
    try:
        model = ViscosityModel(type_text)
    except ValueError as exc:
        raise InpSyntaxError(
            f"*VISCOSITY, TYPE={type_text} は未対応（NEWTONIAN / POWER LAW / CARREAU）",
            block.source,
            block.line_no,
        ) from exc
    if model == ViscosityModel.NEWTONIAN:
        return None
    n_min, n_max, fmt = _VISCOSITY_PARAMS[model]
    tokens = [t for t in block.data[0] if t != ""]
    if not n_min <= len(tokens) <= n_max:
        raise InpSyntaxError(
            f"*VISCOSITY, TYPE={model.value} のデータ行は '{fmt}'", block.source, block.line_no
        )
    params = tuple(_float(t, block, "粘度パラメータ") for t in tokens)
    if any(p <= 0.0 for p in params):
        raise InpSyntaxError(
            f"*VISCOSITY, TYPE={model.value} のパラメータは正の値", block.source, block.line_no
        )
    return ViscosityLaw(model=model, parameters=params)


def _handle_section(b: _Builder, block: KeywordBlock, kind: SectionKind) -> None:
    b.sections.append(
        SectionDefinition(
            kind=kind,
            elset=_norm_name(block.require("ELSET")),
            material=_norm_name(block.require("MATERIAL")),
        )
    )


def _handle_initial_conditions(b: _Builder, block: KeywordBlock) -> None:
    kind_name = _norm_name(block.require("TYPE"))
    try:
        kind = InitialConditionKind[kind_name]
    except KeyError as exc:
        raise InpSyntaxError(
            f"*INITIAL CONDITIONS, TYPE={kind_name} は未対応", block.source, block.line_no
        ) from exc
    for row in block.data:
        if len(row) < 2:
            raise InpSyntaxError(
                "*INITIAL CONDITIONS のデータ行は 'target, value...'", block.source, block.line_no
            )
        values = tuple(_float(t, block, "初期値") for t in row[1:] if t)
        b.initial_conditions.append(
            InitialCondition(kind=kind, target=_norm_name(row[0]), values=values)
        )


def _handle_orientation(b: _Builder, block: KeywordBlock) -> None:
    name = _norm_name(block.require("NAME"))
    system_text = _norm_name(block.get("SYSTEM", "RECTANGULAR") or "RECTANGULAR")
    try:
        system = OrientationSystem(system_text)
    except ValueError as exc:
        raise InpSyntaxError(
            f"*ORIENTATION, SYSTEM={system_text} は未対応（RECTANGULAR / CYLINDRICAL）",
            block.source,
            block.line_no,
        ) from exc
    tokens = [t for row in block.data for t in row if t != ""]
    if len(tokens) < 6:
        raise InpSyntaxError(
            "*ORIENTATION のデータ行は 'ax, ay, az, bx, by, bz'"
            "（CYLINDRICAL は軸上の 2 点、RECTANGULAR は局所 1 軸上の点と 1–2 平面上の点）",
            block.source,
            block.line_no,
        )
    vals = [_float(t, block, "座標") for t in tokens[:6]]
    b.orientations[name] = OrientationDefinition(
        name=name,
        system=system,
        point_a=(vals[0], vals[1], vals[2]),
        point_b=(vals[3], vals[4], vals[5]),
    )


def _handle_mpc(b: _Builder, block: KeywordBlock) -> None:
    for row in block.data:
        rest = [t for t in row if t != ""]
        if len(rest) < 3:
            raise InpSyntaxError(
                "*MPC のデータ行は 'BEAM, slave_surface, master_node'",
                block.source,
                block.line_no,
            )
        kind_text = _norm_name(rest[0])
        try:
            kind = MPCKind(kind_text)
        except ValueError as exc:
            raise InpSyntaxError(
                f"*MPC の種別 {rest[0]!r} は未対応（{', '.join(k.value for k in MPCKind)}）",
                block.source,
                block.line_no,
            ) from exc
        slave = FACE_ALIASES.get(_norm_name(rest[1]), _norm_name(rest[1]))
        b.mpcs.append(MPCDefinition(kind=kind, slave=slave, master=_norm_name(rest[2])))


def _is_periodic_block(block: KeywordBlock) -> bool:
    return _norm_name(block.get("TYPE", "") or "") == "PERIODIC"


def _parse_periodic_rows(block: KeywordBlock) -> list[PeriodicDefinition]:
    """``*BOUNDARY, TYPE=PERIODIC`` のデータ行 ``master, slave[, tx, ty, tz]``."""
    out: list[PeriodicDefinition] = []
    for row in block.data:
        rest = [t for t in row if t != ""]
        if len(rest) < 2:
            raise InpSyntaxError(
                "*BOUNDARY, TYPE=PERIODIC のデータ行は 'master_surface, slave_surface[, tx, ty, tz]'",
                block.source,
                block.line_no,
            )
        master = FACE_ALIASES.get(_norm_name(rest[0]), _norm_name(rest[0]))
        slave = FACE_ALIASES.get(_norm_name(rest[1]), _norm_name(rest[1]))
        if master == slave:
            raise InpSyntaxError(
                f"周期境界の 2 面が同じです: {master}", block.source, block.line_no
            )
        translation: tuple[float, float, float] | None = None
        if len(rest) > 2:
            vec = [_float(t, block, "並進ベクトル") for t in rest[2:5]]
            while len(vec) < 3:
                vec.append(0.0)
            translation = (vec[0], vec[1], vec[2])
        out.append(PeriodicDefinition(master=master, slave=slave, translation=translation))
    return out


def _parse_boundary_rows(block: KeywordBlock) -> list[BoundaryCondition]:
    type_param = block.get("TYPE")
    orientation = _norm_name(block.get("ORIENTATION", "") or "")
    out: list[BoundaryCondition] = []
    for row in block.data:
        if not row:
            continue
        target = _norm_name(row[0])
        target = FACE_ALIASES.get(target, target)
        rest = [t for t in row[1:] if t != ""]
        if type_param:
            kind_name = _norm_name(type_param)
            try:
                kind = BoundaryKind[kind_name]
            except KeyError as exc:
                raise InpSyntaxError(
                    f"*BOUNDARY, TYPE={kind_name} は未対応", block.source, block.line_no
                ) from exc
            if rest and _norm_name(rest[0]) == "SLIP" and kind == BoundaryKind.WALL:
                out.append(BoundaryCondition(target=target, kind=BoundaryKind.SLIP))
                continue
            values = tuple(_float(t, block, "境界値") for t in rest)
            out.append(
                BoundaryCondition(target=target, kind=kind, values=values, orientation=orientation)
            )
            continue
        # 自由度番号形式: target, first_dof[, last_dof[, magnitude]]
        if not rest:
            raise InpSyntaxError(
                "*BOUNDARY: TYPE= を指定するか自由度番号を書いてください",
                block.source,
                block.line_no,
            )
        first = _int(rest[0], block, "自由度")
        last = _int(rest[1], block, "自由度") if len(rest) > 1 else first
        magnitude = _float(rest[2], block, "境界値") if len(rest) > 2 else 0.0
        dofs = list(range(first, last + 1))
        if all(d in (1, 2, 3) for d in dofs):
            if magnitude == 0.0 and dofs == [1, 2, 3] and not orientation:
                out.append(BoundaryCondition(target=target, kind=BoundaryKind.WALL))
            else:
                vel = [0.0, 0.0, 0.0]
                for d in dofs:
                    vel[d - 1] = magnitude
                out.append(
                    BoundaryCondition(
                        target=target,
                        kind=BoundaryKind.VELOCITY,
                        values=tuple(vel),
                        orientation=orientation,
                    )
                )
        elif all(d in (4, 5, 6) for d in dofs):
            # 自由度 4-6 = 回転（角速度 [rad/s]）。参照節点に与えて *MPC で面に伝える
            omega = [0.0, 0.0, 0.0]
            for d in dofs:
                omega[d - 4] = magnitude
            out.append(
                BoundaryCondition(
                    target=target,
                    kind=BoundaryKind.ROTATION,
                    values=tuple(omega),
                    orientation=orientation,
                )
            )
        elif dofs == [8]:
            out.append(
                BoundaryCondition(target=target, kind=BoundaryKind.PRESSURE, values=(magnitude,))
            )
        elif dofs == [11]:
            out.append(
                BoundaryCondition(target=target, kind=BoundaryKind.TEMPERATURE, values=(magnitude,))
            )
        else:
            raise InpSyntaxError(
                f"自由度 {first}-{last} は未対応（1-3: 速度, 4-6: 角速度, 8: 圧力, 11: 温度）",
                block.source,
                block.line_no,
            )
    return out


def _parse_film_rows(block: KeywordBlock) -> list[FilmCondition]:
    out: list[FilmCondition] = []
    for row in block.data:
        if len(row) < 4:
            raise InpSyntaxError(
                "*SFILM のデータ行は 'surface, F, T_inf, h'", block.source, block.line_no
            )
        target = FACE_ALIASES.get(_norm_name(row[0]), _norm_name(row[0]))
        if _norm_name(row[1]) != "F":
            raise InpSyntaxError(
                f"*SFILM のラベル {row[1]!r} は未対応（F のみ）", block.source, block.line_no
            )
        out.append(
            FilmCondition(
                target=target,
                t_inf=_float(row[2], block, "T_inf"),
                h=_float(row[3], block, "h"),
            )
        )
    return out


def _parse_dflux_rows(block: KeywordBlock) -> list[DistributedFlux]:
    out: list[DistributedFlux] = []
    for row in block.data:
        if len(row) < 3:
            raise InpSyntaxError(
                "*DFLUX のデータ行は 'target, S|BF, magnitude'", block.source, block.line_no
            )
        label_name = _norm_name(row[1])
        try:
            label = FluxLabel(label_name)
        except ValueError as exc:
            raise InpSyntaxError(
                f"*DFLUX のラベル {row[1]!r} は未対応（S / BF）", block.source, block.line_no
            ) from exc
        target = FACE_ALIASES.get(_norm_name(row[0]), _norm_name(row[0]))
        out.append(
            DistributedFlux(target=target, label=label, magnitude=_float(row[2], block, "熱流束"))
        )
    return out


def _parse_dload_rows(block: KeywordBlock) -> list[DistributedLoad]:
    out: list[DistributedLoad] = []
    for row in block.data:
        if len(row) < 3:
            raise InpSyntaxError(
                "*DLOAD のデータ行は 'elset, GRAV, g, nx, ny, nz' / 'elset, BX|BY|BZ, f' / "
                "'elset, BF, fx, fy, fz'",
                block.source,
                block.line_no,
            )
        label = _norm_name(row[1])
        target = _norm_name(row[0])
        if label == "GRAV":
            direction = [_float(t, block, "方向余弦") for t in row[3:6]]
            while len(direction) < 3:
                direction.append(0.0)
            out.append(
                DistributedLoad(
                    target=target,
                    label=label,
                    magnitude=_float(row[2], block, "重力加速度"),
                    direction=(direction[0], direction[1], direction[2]),
                )
            )
        elif label in ("BX", "BY", "BZ"):
            axis = [0.0, 0.0, 0.0]
            axis["XYZ".index(label[1])] = 1.0
            out.append(
                DistributedLoad(
                    target=target,
                    label=label,
                    magnitude=_float(row[2], block, "体積力"),
                    direction=(axis[0], axis[1], axis[2]),
                )
            )
        elif label == "BF":
            vec = [_float(t, block, "体積力") for t in row[2:5] if t != ""]
            while len(vec) < 3:
                vec.append(0.0)
            out.append(
                DistributedLoad(
                    target=target, label=label, magnitude=1.0, direction=(vec[0], vec[1], vec[2])
                )
            )
        else:
            raise InpSyntaxError(
                f"*DLOAD のラベル {row[1]!r} は未対応（GRAV / BX / BY / BZ / BF）",
                block.source,
                block.line_no,
            )
    return out


def _parse_procedure(block: KeywordBlock) -> Procedure:
    family = _PROCEDURE_KEYWORDS[block.keyword]
    steady = block.has("STEADY STATE")
    turbulence = _norm_name(block.get("TURBULENCE", "LAMINAR") or "LAMINAR")
    heat_transfer = _norm_name(block.get("HEAT TRANSFER", "NONE") or "NONE")
    dt = 0.0
    period = 0.0
    if block.data and block.data[0]:
        row = block.data[0]
        dt = _float(row[0], block, "時間刻み") if row[0] else 0.0
        period = _float(row[1], block, "解析時間") if len(row) > 1 and row[1] else 0.0
    if not steady and (dt <= 0.0 or period <= 0.0):
        raise InpSyntaxError(
            f"*{block.keyword}: 非定常には データ行 'dt, time_period' が必要です"
            "（定常なら STEADY STATE を付けてください）",
            block.source,
            block.line_no,
        )
    if steady:
        dt = 0.0
        period = 0.0
    return Procedure(
        family=family,
        steady=steady,
        turbulence=turbulence,
        heat_transfer=heat_transfer,
        dt=dt,
        time_period=period,
    )


def _parse_controls(block: KeywordBlock) -> ControlSet:
    cat_name = _norm_name(block.require("PARAMETERS"))
    try:
        category = ControlCategory(cat_name)
    except ValueError as exc:
        raise InpSyntaxError(
            f"*CONTROLS, PARAMETERS={cat_name} は未対応"
            "（DISCRETIZATION / RELAXATION / SOLVER / TIME INCREMENTATION）",
            block.source,
            block.line_no,
        ) from exc
    values: dict[str, str] = {}
    for row in block.data:
        for token in row:
            if not token:
                continue
            if "=" not in token:
                raise InpSyntaxError(
                    f"*CONTROLS のデータ行は 'KEY=VALUE, ...': {token!r}",
                    block.source,
                    block.line_no,
                )
            key, value = token.split("=", 1)
            values[_norm_name(key)] = value.strip()
    return ControlSet(category=category, values=values)


def _parse_output(block: KeywordBlock) -> OutputRequest:
    if not block.has("FIELD"):
        raise InpSyntaxError("*OUTPUT は FIELD のみ対応", block.source, block.line_no)
    fmt_text = block.get("FORMAT", "NPZ") or "NPZ"
    explicit = block.get("FORMAT") is not None
    formats: list[OutputFormat] = []
    for token in fmt_text.replace("+", " ").replace("/", " ").split():
        try:
            formats.append(OutputFormat(_norm_name(token)))
        except ValueError as exc:
            raise InpSyntaxError(
                f"*OUTPUT, FORMAT={token} は未対応（NPZ / VTK / HTML）", block.source, block.line_no
            ) from exc
    if OutputFormat.NPZ not in formats:
        formats.insert(0, OutputFormat.NPZ)
    variables_text = block.get("VARIABLE", "") or ""
    variables = tuple(_norm_name(v) for v in variables_text.replace(" ", ",").split(",") if v)
    frequency = int(block.get("FREQUENCY", "1") or 1)
    return OutputRequest(
        variables=variables,
        formats=tuple(formats),
        frequency=max(frequency, 1),
        formats_explicit=explicit,
    )


# ---------------------------------------------------------------------------
# *GRID（ykep 拡張）: 構造格子の節点・要素を生成
# ---------------------------------------------------------------------------


def generate_grid(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    elset: str = "ALL",
) -> tuple[NodeTable, ElementBlock]:
    """等間隔直交格子の節点表と C3D8 要素ブロックを生成する（ID は 1 始まり）."""
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("*GRID: NX, NY, NZ は 1 以上")
    xs = origin[0] + np.linspace(0.0, lx, nx + 1)
    ys = origin[1] + np.linspace(0.0, ly, ny + 1)
    zs = origin[2] + np.linspace(0.0, lz, nz + 1)
    # 節点 ID: i + (nx+1)*(j + (ny+1)*k) + 1
    gi, gj, gk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    node_ids = (gi + (nx + 1) * (gj + (ny + 1) * gk) + 1).ravel(order="F")
    coords = np.stack(
        [xs[gi.ravel(order="F")], ys[gj.ravel(order="F")], zs[gk.ravel(order="F")]], axis=1
    )

    def nid(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return i + (nx + 1) * (j + (ny + 1) * k) + 1

    ci, cj, ck = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ci = ci.ravel(order="F")
    cj = cj.ravel(order="F")
    ck = ck.ravel(order="F")
    conn = np.stack(
        [
            nid(ci, cj, ck),
            nid(ci + 1, cj, ck),
            nid(ci + 1, cj + 1, ck),
            nid(ci, cj + 1, ck),
            nid(ci, cj, ck + 1),
            nid(ci + 1, cj, ck + 1),
            nid(ci + 1, cj + 1, ck + 1),
            nid(ci, cj + 1, ck + 1),
        ],
        axis=1,
    )
    elem_ids = np.arange(1, nx * ny * nz + 1)
    return (
        NodeTable(ids=node_ids.astype(int), coords=coords.astype(float)),
        ElementBlock(element_type="C3D8", ids=elem_ids, connectivity=conn.astype(int), elset=elset),
    )


def _handle_grid(b: _Builder, block: KeywordBlock) -> None:
    if b.grid_defined or b.node_ids:
        raise InpSyntaxError(
            "*GRID は *NODE/*ELEMENT と併用できません（1 ケース 1 回）", block.source, block.line_no
        )
    nx = _int(block.require("NX"), block, "NX")
    ny = _int(block.get("NY", "1") or "1", block, "NY")
    nz = _int(block.get("NZ", "1") or "1", block, "NZ")
    lx = _float(block.require("LX"), block, "LX")
    ly = _float(block.get("LY", "1.0") or "1.0", block, "LY")
    lz = _float(block.get("LZ", "1.0") or "1.0", block, "LZ")
    origin_text = block.get("ORIGIN", "") or ""
    origin = (0.0, 0.0, 0.0)
    if origin_text:
        parts = [p for p in origin_text.replace(";", " ").split() if p]
        if len(parts) != 3:
            raise InpSyntaxError(
                "*GRID, ORIGIN= は 'x y z'（空白区切り）", block.source, block.line_no
            )
        origin = tuple(_float(p, block, "ORIGIN") for p in parts)  # type: ignore[assignment]
    elset = _norm_name(block.get("ELSET", "ALL") or "ALL")
    nodes, elements = generate_grid(nx, ny, nz, lx, ly, lz, origin, elset)
    b.node_ids.extend(nodes.ids.tolist())
    b.node_coords.extend(nodes.coords.tolist())
    b.elements.append(elements)
    b.elsets.setdefault(elset, []).extend(elements.ids.tolist())
    b.nsets.setdefault("ALL", []).extend(nodes.ids.tolist())
    b.grid_defined = True


# ---------------------------------------------------------------------------
# 組み立て本体
# ---------------------------------------------------------------------------


def build_case(parsed: InpParseResult) -> CaseDefinition:
    """トークナイズ結果から :class:`CaseDefinition` を組み立てる."""
    b = _Builder(source=parsed.source)
    current_material: str | None = None
    step: _StepBuilder | None = None

    for block in parsed.blocks:
        kw = block.keyword

        # --- ステップ内 ---
        if step is not None:
            if kw == "END STEP":
                b.steps.append(step.finish())
                step = None
            elif kw in _PROCEDURE_KEYWORDS:
                if step.procedure is not None:
                    raise InpSyntaxError(
                        "1 ステップに手続きキーワードは 1 つ", block.source, block.line_no
                    )
                step.procedure = _parse_procedure(block)
            elif kw == "CONTROLS":
                step.controls.append(_parse_controls(block))
            elif kw == "BOUNDARY":
                if _is_periodic_block(block):
                    raise InpSyntaxError(
                        "*BOUNDARY, TYPE=PERIODIC はメッシュの位相なので *STEP の外に置いてください",
                        block.source,
                        block.line_no,
                    )
                step.boundaries.extend(_parse_boundary_rows(block))
            elif kw == "SFILM":
                step.films.extend(_parse_film_rows(block))
            elif kw == "DFLUX":
                step.fluxes.extend(_parse_dflux_rows(block))
            elif kw == "DLOAD":
                step.loads.extend(_parse_dload_rows(block))
            elif kw == "OUTPUT":
                step.add_output(_parse_output(block))
            elif kw in ("ELEMENT OUTPUT", "NODE OUTPUT"):
                variables = tuple(_norm_name(t) for row in block.data for t in row if t)
                step.add_output_variables(variables, block)
            elif kw == "STEP":
                raise InpSyntaxError(
                    "*STEP が入れ子です（*END STEP 忘れ）", block.source, block.line_no
                )
            else:
                logger.warning(
                    "%s: ステップ内の未対応キーワード *%s を無視します", block.location(), kw
                )
            continue

        # --- モデルレベル ---
        if kw == "HEADING":
            b.heading = "\n".join(block.raw_lines)
        elif kw == "NODE":
            _handle_node(b, block)
        elif kw == "ELEMENT":
            _handle_element(b, block)
        elif kw == "GRID":
            _handle_grid(b, block)
        elif kw == "NSET":
            _handle_nset(b, block)
        elif kw == "ELSET":
            _handle_elset(b, block)
        elif kw == "SURFACE":
            _handle_surface(b, block)
        elif kw == "ORIENTATION":
            _handle_orientation(b, block)
        elif kw == "MPC":
            _handle_mpc(b, block)
        elif kw == "MATERIAL":
            current_material = _handle_material(b, block)
        elif kw in _MATERIAL_SUBKEYWORDS:
            _handle_material_property(b, block, current_material)
        elif kw == "FLUID SECTION":
            _handle_section(b, block, SectionKind.FLUID)
        elif kw == "SOLID SECTION":
            _handle_section(b, block, SectionKind.SOLID)
        elif kw == "INITIAL CONDITIONS":
            _handle_initial_conditions(b, block)
        elif kw == "BOUNDARY":
            if _is_periodic_block(block):
                b.periodic.extend(_parse_periodic_rows(block))
            else:
                b.boundaries.extend(_parse_boundary_rows(block))
        elif kw == "SFILM":
            b.films.extend(_parse_film_rows(block))
        elif kw == "DFLUX":
            b.fluxes.extend(_parse_dflux_rows(block))
        elif kw == "DLOAD":
            b.loads.extend(_parse_dload_rows(block))
        elif kw == "STEP":
            step = _StepBuilder(block)
        elif kw == "END STEP":
            raise InpSyntaxError(
                "*END STEP に対応する *STEP がありません", block.source, block.line_no
            )
        elif kw in _PROCEDURE_KEYWORDS or kw in ("CONTROLS", "OUTPUT"):
            raise InpSyntaxError(
                f"*{kw} は *STEP の中に置いてください", block.source, block.line_no
            )
        elif kw in _IGNORED_KEYWORDS:
            continue
        else:
            logger.warning("%s: 未対応キーワード *%s を無視します", block.location(), kw)

    if step is not None:
        raise InpSyntaxError(
            f"*STEP {step.name!r} が *END STEP で閉じられていません",
            step.block.source,
            step.block.line_no,
        )

    if not b.node_ids:
        raise InpSyntaxError("節点がありません（*NODE または *GRID が必要）", parsed.source)
    if not b.elements:
        raise InpSyntaxError("要素がありません（*ELEMENT または *GRID が必要）", parsed.source)

    node_ids = np.asarray(b.node_ids, dtype=int)
    if np.unique(node_ids).size != node_ids.size:
        raise InpSyntaxError("節点 ID が重複しています", parsed.source)
    all_elem_ids = np.concatenate([e.ids for e in b.elements])
    if np.unique(all_elem_ids).size != all_elem_ids.size:
        raise InpSyntaxError("要素 ID が重複しています", parsed.source)

    known_nodes = set(node_ids.tolist())
    for e in b.elements:
        missing = set(e.connectivity.ravel().tolist()) - known_nodes
        if missing:
            raise InpSyntaxError(
                f"要素が未定義の節点を参照: {sorted(missing)[:5]} ...", parsed.source
            )

    def _finalize(sets: dict[str, list[int]], kind: SetKind) -> dict[str, SetDefinition]:
        return {
            name: SetDefinition(name=name, kind=kind, ids=np.unique(np.asarray(ids, dtype=int)))
            for name, ids in sets.items()
        }

    nsets = _finalize(b.nsets, SetKind.NODE)
    elsets = _finalize(b.elsets, SetKind.ELEMENT)
    if "ALL" not in elsets:
        elsets["ALL"] = SetDefinition(name="ALL", kind=SetKind.ELEMENT, ids=np.unique(all_elem_ids))
    if "ALL" not in nsets:
        nsets["ALL"] = SetDefinition(name="ALL", kind=SetKind.NODE, ids=np.unique(node_ids))

    known_elems = set(all_elem_ids.tolist())
    for name, s in elsets.items():
        unknown = set(s.ids.tolist()) - known_elems
        if unknown:
            raise InpSyntaxError(
                f"*ELSET {name} が未定義要素を参照: {sorted(unknown)[:5]}", parsed.source
            )
    for name, s in nsets.items():
        unknown = set(s.ids.tolist()) - known_nodes
        if unknown:
            raise InpSyntaxError(
                f"*NSET {name} が未定義節点を参照: {sorted(unknown)[:5]}", parsed.source
            )

    for sec in b.sections:
        if sec.elset not in elsets:
            raise InpSyntaxError(f"セクションの ELSET {sec.elset!r} が未定義", parsed.source)
        if sec.material not in b.materials:
            raise InpSyntaxError(f"セクションの MATERIAL {sec.material!r} が未定義", parsed.source)
    for surf in b.surfaces.values():
        for entry in surf.entries:
            if entry.target not in elsets and not entry.target.isdigit():
                raise InpSyntaxError(
                    f"*SURFACE {surf.name} の要素集合 {entry.target!r} が未定義", parsed.source
                )
    for mpc in b.mpcs:
        if mpc.slave not in b.surfaces and mpc.slave not in RESERVED_FACE_NAMES:
            raise InpSyntaxError(
                f"*MPC の従属面 {mpc.slave!r} は *SURFACE でも予約面名でもありません", parsed.source
            )
        if mpc.master not in nsets and not mpc.master.isdigit():
            raise InpSyntaxError(
                f"*MPC の参照節点 {mpc.master!r} は *NSET でも節点 ID でもありません", parsed.source
            )
    for bc in list(b.boundaries) + [bc for st in b.steps for bc in st.boundaries]:
        if bc.orientation and bc.orientation not in b.orientations:
            raise InpSyntaxError(
                f"*BOUNDARY, ORIENTATION={bc.orientation} が *ORIENTATION で定義されていません",
                parsed.source,
            )
    used_periodic: set[str] = set()
    for per in b.periodic:
        for name in (per.master, per.slave):
            if name not in b.surfaces and name not in RESERVED_FACE_NAMES:
                raise InpSyntaxError(
                    f"*BOUNDARY, TYPE=PERIODIC の面 {name!r} は *SURFACE でも予約面名でもありません",
                    parsed.source,
                )
            if name in used_periodic:
                raise InpSyntaxError(f"面 {name!r} が複数の周期境界に使われています", parsed.source)
            used_periodic.add(name)

    return CaseDefinition(
        heading=b.heading,
        nodes=NodeTable(ids=node_ids, coords=np.asarray(b.node_coords, dtype=float)),
        elements=tuple(b.elements),
        nsets=nsets,
        elsets=elsets,
        surfaces=dict(b.surfaces),
        materials=dict(b.materials),
        sections=tuple(b.sections),
        initial_conditions=tuple(b.initial_conditions),
        boundaries=tuple(b.boundaries),
        films=tuple(b.films),
        fluxes=tuple(b.fluxes),
        loads=tuple(b.loads),
        periodic=tuple(b.periodic),
        orientations=dict(b.orientations),
        mpcs=tuple(b.mpcs),
        steps=tuple(b.steps),
        parameters=dict(parsed.parameters),
        source=parsed.source,
    )


class InpCaseBuildProcess(PreProcess["InpParseResult", "CaseDefinition"]):
    """:class:`KeywordBlock` 列を :class:`CaseDefinition` に意味付けする PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpCaseBuild",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpParseResult) -> CaseDefinition:
        return build_case(input_data)
