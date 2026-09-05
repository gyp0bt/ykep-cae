"""``CaseDefinition`` + 構造格子 → ykep 各ソルバー Input への変換.

- :class:`InpToNaturalConvectionProcess`: ``*NAVIER STOKES``（層流、等温/伝熱連成）
  → :class:`~xkep_cae_fluid.natural_convection.data.NaturalConvectionInput`
- :class:`InpToHeatTransferProcess`: ``*HEAT TRANSFER``
  → :class:`~xkep_cae_fluid.heat_transfer.data.HeatTransferInput`

ykep が解釈できない指定（乱流モデル、部分面、未対応スキーム等）は
:class:`UnsupportedFeatureError` で明示的に拒否する（黙って無視しない）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition as HTBoundaryCondition,
)
from xkep_cae_fluid.heat_transfer.data import (
    BoundarySpec as HTBoundarySpec,
)
from xkep_cae_fluid.heat_transfer.data import (
    HeatTransferInput,
)
from xkep_cae_fluid.inp.case import (
    BoundaryCondition,
    BoundaryKind,
    CaseDefinition,
    ControlCategory,
    DistributedLoad,
    EquationFamily,
    FilmCondition,
    FluxLabel,
    InitialConditionKind,
    MaterialDefinition,
    SectionKind,
    StepDefinition,
)
from xkep_cae_fluid.inp.grid import FACE_NAMES, StructuredGridMap
from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)

logger = logging.getLogger(__name__)


class UnsupportedFeatureError(ValueError):
    """フォーマット上は正しいが、ykep のソルバーでは扱えない指定."""


@dataclass(frozen=True)
class InpMappingInput:
    """マッピングプロセス共通入力."""

    case: CaseDefinition
    grid: StructuredGridMap
    step_index: int = 0

    @property
    def step(self) -> StepDefinition:
        if not self.case.steps:
            raise UnsupportedFeatureError("*STEP がありません")
        if not 0 <= self.step_index < len(self.case.steps):
            raise IndexError(f"step_index {self.step_index} が範囲外")
        return self.case.steps[self.step_index]


@dataclass(frozen=True)
class HeatTransferMappingResult:
    """伝熱ソルバー用の Input と、プロセス生成時に必要な線形解法名."""

    input: HeatTransferInput
    method: str = "bicgstab"


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------


def _norm_value(text: str) -> str:
    return text.strip().upper().replace("-", "_").replace(" ", "_")


def _as_bool(text: str, key: str) -> bool:
    v = _norm_value(text)
    if v in ("YES", "TRUE", "ON", "1"):
        return True
    if v in ("NO", "FALSE", "OFF", "0"):
        return False
    raise UnsupportedFeatureError(f"*CONTROLS {key}={text!r} は YES/NO で指定")


def _as_float(text: str, key: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise UnsupportedFeatureError(f"*CONTROLS {key}={text!r} は数値") from exc


def _as_int(text: str, key: str) -> int:
    try:
        return int(float(text))
    except ValueError as exc:
        raise UnsupportedFeatureError(f"*CONTROLS {key}={text!r} は整数") from exc


def _check_keys(values: dict[str, str], allowed: set[str], category: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise UnsupportedFeatureError(
            f"*CONTROLS, PARAMETERS={category} の未知のキー: {sorted(unknown)}"
            f"（許容: {sorted(allowed)}）"
        )


def resolve_target_face(target: str, case: CaseDefinition, grid: StructuredGridMap) -> str:
    """境界条件の target（予約面名 or ``*SURFACE`` 名）を領域面名に解決する."""
    key = target.strip().upper()
    if key in FACE_NAMES:
        return key
    if key in case.surfaces:
        return grid.resolve_surface_face(case.surfaces[key], case)
    raise UnsupportedFeatureError(
        f"境界 target {target!r} は *SURFACE でも予約面名（{', '.join(FACE_NAMES)}）でもありません"
    )


def _gravity_vector(case: CaseDefinition, step: StepDefinition) -> tuple[float, float, float]:
    loads: list[DistributedLoad] = [ld for ld in case.loads + step.loads if ld.label == "GRAV"]
    if not loads:
        return (0.0, 0.0, 0.0)
    if len(loads) > 1:
        raise UnsupportedFeatureError("*DLOAD, GRAV は 1 つだけ指定してください")
    return loads[0].vector


def _fluid_material(case: CaseDefinition) -> MaterialDefinition:
    fluid_sections = [s for s in case.sections if s.kind == SectionKind.FLUID]
    if not fluid_sections:
        raise UnsupportedFeatureError("*FLUID SECTION がありません")
    names = {s.material for s in fluid_sections}
    if len(names) > 1:
        raise UnsupportedFeatureError(f"流体材料は 1 種類のみ対応: {sorted(names)}")
    return case.material_of_section(fluid_sections[0])


def _section_coverage(
    case: CaseDefinition, grid: StructuredGridMap
) -> tuple[np.ndarray, list[tuple[np.ndarray, MaterialDefinition, SectionKind]]]:
    """各セクションのマスクと材料。全セルが重複なく覆われていることを検証する."""
    covered = np.zeros(grid.dimensions, dtype=bool)
    out: list[tuple[np.ndarray, MaterialDefinition, SectionKind]] = []
    for sec in case.sections:
        mask = grid.mask_for_elements(case.element_ids_of(sec.elset))
        if np.any(covered & mask):
            raise UnsupportedFeatureError(f"セクションが重複しています（ELSET={sec.elset}）")
        covered |= mask
        out.append((mask, case.material_of_section(sec), sec.kind))
    if not np.all(covered):
        n_missing = int((~covered).sum())
        raise UnsupportedFeatureError(f"セクション未割当のセルが {n_missing} 個あります")
    return covered, out


def _initial_temperature(
    case: CaseDefinition, grid: StructuredGridMap, default: float
) -> np.ndarray:
    """``*INITIAL CONDITIONS, TYPE=TEMPERATURE`` をセル値に展開する.

    Abaqus と同じく節点ベースを基本とし（nset / 節点 ID / ALL）、セル値は要素の
    節点平均で決める。elset / 要素 ID を target にした行はセル値を直接上書きする
    （ykep 拡張）。後に書いた行が優先。未指定セルは ``default``。
    """
    node_ids = case.nodes.ids
    node_T = np.full(case.nodes.n_nodes, np.nan)
    cell_override = np.full(grid.dimensions, np.nan)
    node_index = {int(n): i for i, n in enumerate(node_ids.tolist())}
    for ic in case.initial_conditions:
        if ic.kind != InitialConditionKind.TEMPERATURE:
            continue
        if not ic.values:
            raise UnsupportedFeatureError("*INITIAL CONDITIONS, TYPE=TEMPERATURE に値がありません")
        value = float(ic.values[0])
        key = ic.target
        if key == "ALL":
            node_T[:] = value
            cell_override[:] = np.nan
        elif key in case.nsets:
            node_T[[node_index[int(n)] for n in case.nsets[key].ids.tolist()]] = value
        elif key in case.elsets:
            cell_override[grid.mask_for_elements(case.elsets[key].ids)] = value
        elif key.isdigit() and int(key) in node_index:
            node_T[node_index[int(key)]] = value
        elif key.isdigit():
            cell_override[grid.mask_for_elements(np.array([int(key)]))] = value
        else:
            raise UnsupportedFeatureError(f"*INITIAL CONDITIONS の target {ic.target!r} が未定義")
    defined = ~np.isnan(node_T)
    cell_T = np.full(grid.dimensions, np.nan)
    if np.any(defined):
        cell_T = grid.node_values_to_cells(node_ids[defined], node_T[defined], case)
    cell_T = np.where(np.isnan(cell_override), cell_T, cell_override)
    return np.where(np.isnan(cell_T), default, cell_T)


def _body_flux(
    case: CaseDefinition, step: StepDefinition, grid: StructuredGridMap
) -> np.ndarray | None:
    q = np.zeros(grid.dimensions)
    found = False
    for fl in case.fluxes + step.fluxes:
        if fl.label != FluxLabel.BODY:
            continue
        q[grid.mask_for_elements(case.element_ids_of(fl.target))] += fl.magnitude
        found = True
    return q if found else None


def _surface_flux(
    case: CaseDefinition, step: StepDefinition, grid: StructuredGridMap
) -> dict[str, float]:
    out: dict[str, float] = {}
    for fl in case.fluxes + step.fluxes:
        if fl.label != FluxLabel.SURFACE:
            continue
        face = resolve_target_face(fl.target, case, grid)
        out[face] = out.get(face, 0.0) + fl.magnitude
    return out


def _face_boundaries(
    case: CaseDefinition, step: StepDefinition, grid: StructuredGridMap
) -> dict[str, list[BoundaryCondition]]:
    out: dict[str, list[BoundaryCondition]] = {f: [] for f in FACE_NAMES}
    for bc in case.boundaries + step.boundaries:
        out[resolve_target_face(bc.target, case, grid)].append(bc)
    return out


def _face_films(
    case: CaseDefinition, step: StepDefinition, grid: StructuredGridMap
) -> dict[str, FilmCondition]:
    out: dict[str, FilmCondition] = {}
    for film in case.films + step.films:
        face = resolve_target_face(film.target, case, grid)
        if face in out:
            raise UnsupportedFeatureError(f"面 {face} に *SFILM が重複しています")
        out[face] = film
    return out


def _check_procedure_common(step: StepDefinition, family: EquationFamily) -> None:
    proc = step.procedure
    if proc.family != family:
        raise UnsupportedFeatureError(
            f"このマッピングは *{family.value} 用です（ステップは *{proc.family.value}）"
        )
    if proc.turbulence != "LAMINAR":
        raise UnsupportedFeatureError(
            f"TURBULENCE={proc.turbulence} は未対応（LAMINAR のみ。乱流モデルは Phase 5 予定）"
        )


# ---------------------------------------------------------------------------
# Navier-Stokes（NaturalConvectionFDMProcess）
# ---------------------------------------------------------------------------

_CONVECTION_NC: dict[str, str] = {
    "UPWIND": "upwind",
    "FIRST_ORDER_UPWIND": "upwind",
    "VAN_LEER": "van_leer",
    "VANLEER": "van_leer",
    "SUPERBEE": "superbee",
}
_TIME_NC: dict[str, str] = {"EULER": "euler", "BACKWARD_EULER": "euler", "BDF2": "bdf2"}
_COUPLING_NC: dict[str, str] = {"SIMPLE": "simple", "SIMPLEC": "simplec", "PISO": "piso"}
_PRESSURE_SOLVER_NC: dict[str, str] = {"BICGSTAB": "bicgstab", "AMG": "amg"}

_NC_DISCRETIZATION_KEYS = {"CONVECTION", "TIME", "PRESSURE_VELOCITY", "PISO_CORRECTORS", "LIMITER"}
_NC_RELAXATION_KEYS = {"VELOCITY", "PRESSURE", "TEMPERATURE", "ADAPTIVE"}
_NC_SOLVER_KEYS = {"PRESSURE", "MAX_OUTER", "MAX_INNER", "MAX_PRESSURE_ITER", "TOL", "TOL_INNER"}
_TIME_INC_KEYS = {"OUTPUT_INTERVAL"}


def _nc_fluid_bc(
    face: str,
    bcs: list[BoundaryCondition],
    heat_flux: float | None,
    coupled: bool,
    default_condition: FluidBoundaryCondition,
    t_default: float,
) -> FluidBoundarySpec:
    condition = default_condition
    velocity = (0.0, 0.0, 0.0)
    pressure = 0.0
    thermal = ThermalBoundaryCondition.ADIABATIC
    temperature = t_default
    flux = 0.0
    for bc in bcs:
        if bc.kind == BoundaryKind.WALL:
            condition = FluidBoundaryCondition.NO_SLIP
        elif bc.kind == BoundaryKind.SLIP:
            condition = FluidBoundaryCondition.SLIP
        elif bc.kind == BoundaryKind.SYMMETRY:
            condition = FluidBoundaryCondition.SYMMETRY
        elif bc.kind == BoundaryKind.VELOCITY:
            vel = list(bc.values) + [0.0] * (3 - len(bc.values))
            velocity = (float(vel[0]), float(vel[1]), float(vel[2]))
            condition = (
                FluidBoundaryCondition.NO_SLIP
                if all(v == 0.0 for v in velocity)
                else FluidBoundaryCondition.INLET_VELOCITY
            )
        elif bc.kind == BoundaryKind.PRESSURE:
            condition = FluidBoundaryCondition.OUTLET_PRESSURE
            pressure = float(bc.values[0]) if bc.values else 0.0
        elif bc.kind == BoundaryKind.OUTLET:
            condition = FluidBoundaryCondition.OUTLET_CONVECTIVE
        elif bc.kind == BoundaryKind.TEMPERATURE:
            if not coupled:
                logger.warning("面 %s の温度境界は HEAT TRANSFER=NONE のため無視します", face)
                continue
            if not bc.values:
                raise UnsupportedFeatureError(f"面 {face} の TYPE=TEMPERATURE に値がありません")
            thermal = ThermalBoundaryCondition.DIRICHLET
            temperature = float(bc.values[0])
    if heat_flux is not None:
        if not coupled:
            logger.warning("面 %s の *DFLUX は HEAT TRANSFER=NONE のため無視します", face)
        elif thermal == ThermalBoundaryCondition.DIRICHLET:
            raise UnsupportedFeatureError(f"面 {face} に温度固定と熱流束が同時に指定されています")
        else:
            thermal = ThermalBoundaryCondition.NEUMANN
            flux = heat_flux
    return FluidBoundarySpec(
        condition=condition,
        velocity=velocity,
        pressure=pressure,
        thermal=thermal,
        temperature=temperature,
        heat_flux=flux,
    )


def map_navier_stokes(
    case: CaseDefinition, grid: StructuredGridMap, step: StepDefinition
) -> NaturalConvectionInput:
    """``*NAVIER STOKES`` ステップを :class:`NaturalConvectionInput` に変換する."""
    _check_procedure_common(step, EquationFamily.NAVIER_STOKES)
    proc = step.procedure
    coupled = proc.heat_transfer == "COUPLED"
    if proc.heat_transfer not in ("NONE", "COUPLED"):
        raise UnsupportedFeatureError(f"HEAT TRANSFER={proc.heat_transfer} は NONE / COUPLED のみ")
    if not grid.is_uniform:
        raise UnsupportedFeatureError("*NAVIER STOKES（NaturalConvectionFDM）は等間隔格子のみ対応")

    fluid = _fluid_material(case)
    rho = fluid.require("density")
    mu = fluid.require("viscosity")
    nx, ny, nz = grid.dimensions
    Lx, Ly, Lz = grid.lengths

    _, sections = _section_coverage(case, grid)
    solid_mask = np.zeros(grid.dimensions, dtype=bool)
    solid_k = np.zeros(grid.dimensions)
    has_solid = False
    for mask, mat, kind in sections:
        if kind == SectionKind.SOLID:
            has_solid = True
            solid_mask |= mask
            solid_k[mask] = mat.require("conductivity")

    if coupled:
        Cp = fluid.require("specific_heat")
        k_fluid = fluid.require("conductivity")
        beta = fluid.expansion if fluid.expansion is not None else 0.0
        gravity = _gravity_vector(case, step)
        if beta == 0.0 and any(g != 0.0 for g in gravity):
            logger.warning("重力があるのに *EXPANSION が 0（浮力なし）です")
    else:
        Cp = fluid.specific_heat if fluid.specific_heat is not None else 1000.0
        k_fluid = fluid.conductivity if fluid.conductivity is not None else 1.0
        beta = 0.0
        gravity = _gravity_vector(case, step)

    t_init_default = (
        fluid.reference_temperature if fluid.reference_temperature is not None else 300.0
    )
    T0 = _initial_temperature(case, grid, t_init_default)
    if not coupled:
        T0 = np.full(grid.dimensions, float(np.mean(T0)))
    T_ref = (
        fluid.reference_temperature
        if fluid.reference_temperature is not None
        else float(np.mean(T0))
    )

    k_solid = None
    if has_solid:
        k_solid = np.where(solid_mask, solid_k, k_fluid)

    q_vol = _body_flux(case, step, grid) if coupled else None
    surface_flux = _surface_flux(case, step, grid)
    if _face_films(case, step, grid):
        raise UnsupportedFeatureError("*SFILM（対流熱伝達）は *NAVIER STOKES では未対応")

    face_bcs = _face_boundaries(case, step, grid)
    default_z = (
        FluidBoundaryCondition.SYMMETRY if grid.ndim == 2 else FluidBoundaryCondition.NO_SLIP
    )
    specs: dict[str, FluidBoundarySpec] = {}
    for face in FACE_NAMES:
        default = default_z if face in ("ZM", "ZP") else FluidBoundaryCondition.NO_SLIP
        specs[face] = _nc_fluid_bc(
            face, face_bcs[face], surface_flux.get(face), coupled, default, T_ref
        )

    # --- *CONTROLS ---
    disc = step.control_values(ControlCategory.DISCRETIZATION)
    _check_keys(disc, _NC_DISCRETIZATION_KEYS, "DISCRETIZATION")
    convection = "upwind"
    if "CONVECTION" in disc:
        key = _norm_value(disc["CONVECTION"])
        if key not in _CONVECTION_NC:
            raise UnsupportedFeatureError(
                f"CONVECTION={disc['CONVECTION']} は NaturalConvectionFDM では未対応"
                f"（{sorted(_CONVECTION_NC)}）"
            )
        convection = _CONVECTION_NC[key]
    if "LIMITER" in disc:
        raise UnsupportedFeatureError(
            "LIMITER= は NaturalConvectionFDM では未対応（TVD は CONVECTION= で選択）"
        )
    time_scheme = "euler"
    if "TIME" in disc:
        key = _norm_value(disc["TIME"])
        if key not in _TIME_NC:
            raise UnsupportedFeatureError(f"TIME={disc['TIME']} は未対応（EULER / BDF2）")
        time_scheme = _TIME_NC[key]
    coupling = "simple"
    if "PRESSURE_VELOCITY" in disc:
        key = _norm_value(disc["PRESSURE_VELOCITY"])
        if key not in _COUPLING_NC:
            raise UnsupportedFeatureError(
                f"PRESSURE_VELOCITY={disc['PRESSURE_VELOCITY']} は未対応（SIMPLE / SIMPLEC / PISO）"
            )
        coupling = _COUPLING_NC[key]
    piso_correctors = _as_int(disc.get("PISO_CORRECTORS", "2"), "PISO_CORRECTORS")

    relax = step.control_values(ControlCategory.RELAXATION)
    _check_keys(relax, _NC_RELAXATION_KEYS, "RELAXATION")
    alpha_u = _as_float(relax.get("VELOCITY", "0.7"), "VELOCITY")
    alpha_p = _as_float(relax.get("PRESSURE", "0.3"), "PRESSURE")
    alpha_T = _as_float(relax.get("TEMPERATURE", "0.9"), "TEMPERATURE")
    adaptive = _as_bool(relax.get("ADAPTIVE", "NO"), "ADAPTIVE")

    solver = step.control_values(ControlCategory.SOLVER)
    _check_keys(solver, _NC_SOLVER_KEYS, "SOLVER")
    pressure_solver = "bicgstab"
    if "PRESSURE" in solver:
        key = _norm_value(solver["PRESSURE"])
        if key not in _PRESSURE_SOLVER_NC:
            raise UnsupportedFeatureError(
                f"SOLVER PRESSURE={solver['PRESSURE']} は未対応（BICGSTAB / AMG）"
            )
        pressure_solver = _PRESSURE_SOLVER_NC[key]
    max_outer = _as_int(solver.get("MAX_OUTER", "500"), "MAX_OUTER")
    if step.max_increments > 0 and "MAX_OUTER" not in solver:
        max_outer = step.max_increments
    max_inner = _as_int(solver.get("MAX_INNER", "50"), "MAX_INNER")
    max_pressure_iter = _as_int(solver.get("MAX_PRESSURE_ITER", "0"), "MAX_PRESSURE_ITER")
    tol = _as_float(solver.get("TOL", "1e-5"), "TOL")
    tol_inner = _as_float(solver.get("TOL_INNER", "1e-6"), "TOL_INNER")

    tinc = step.control_values(ControlCategory.TIME_INCREMENTATION)
    _check_keys(tinc, _TIME_INC_KEYS, "TIME INCREMENTATION")
    output_interval = _as_int(tinc.get("OUTPUT_INTERVAL", "0"), "OUTPUT_INTERVAL")
    if output_interval <= 0:
        output_interval = max((o.frequency for o in step.outputs), default=1)

    return NaturalConvectionInput(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=rho,
        mu=mu,
        Cp=Cp,
        k_fluid=k_fluid,
        beta=beta,
        T_ref=T_ref,
        gravity=gravity,
        solid_mask=solid_mask if has_solid else None,
        k_solid=k_solid,
        q_vol=q_vol,
        T0=T0,
        bc_xm=specs["XM"],
        bc_xp=specs["XP"],
        bc_ym=specs["YM"],
        bc_yp=specs["YP"],
        bc_zm=specs["ZM"],
        bc_zp=specs["ZP"],
        dt=proc.dt,
        t_end=proc.time_period,
        max_simple_iter=max_outer,
        max_inner_iter=max_inner,
        tol_simple=tol,
        tol_inner=tol_inner,
        alpha_u=alpha_u,
        alpha_p=alpha_p,
        alpha_T=alpha_T,
        output_interval=output_interval,
        coupling_method=coupling,
        n_piso_correctors=piso_correctors,
        convection_scheme=convection,
        time_scheme=time_scheme,
        pressure_solver=pressure_solver,
        adaptive_relaxation=adaptive,
        max_pressure_iter=max_pressure_iter,
    )


class InpToNaturalConvectionProcess(PreProcess["InpMappingInput", "NaturalConvectionInput"]):
    """``*NAVIER STOKES`` ステップを :class:`NaturalConvectionInput` に変換する PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpToNaturalConvection",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMappingInput) -> NaturalConvectionInput:
        return map_navier_stokes(input_data.case, input_data.grid, input_data.step)


# ---------------------------------------------------------------------------
# Heat transfer（HeatTransferFDMProcess）
# ---------------------------------------------------------------------------

_HT_METHODS = {
    "JACOBI": "jacobi",
    "DIRECT": "direct",
    "BICGSTAB": "bicgstab",
    "AMG": "amg",
    "NUMBA": "numba",
}
_HT_SOLVER_KEYS = {"METHOD", "MAX_ITER", "TOL"}


def _ht_bc(
    face: str, bcs: list[BoundaryCondition], film: FilmCondition | None, flux: float | None
) -> HTBoundarySpec:
    spec = HTBoundarySpec(HTBoundaryCondition.ADIABATIC)
    for bc in bcs:
        if bc.kind == BoundaryKind.TEMPERATURE:
            if not bc.values:
                raise UnsupportedFeatureError(f"面 {face} の TYPE=TEMPERATURE に値がありません")
            spec = HTBoundarySpec(HTBoundaryCondition.DIRICHLET, value=float(bc.values[0]))
        else:
            raise UnsupportedFeatureError(
                f"面 {face} の {bc.kind.name} 境界は *HEAT TRANSFER では使えません（TEMPERATURE のみ）"
            )
    if flux is not None:
        if spec.condition == HTBoundaryCondition.DIRICHLET:
            raise UnsupportedFeatureError(f"面 {face} に温度固定と熱流束が同時に指定されています")
        spec = HTBoundarySpec(HTBoundaryCondition.NEUMANN, value=flux)
    if film is not None:
        if spec.condition != HTBoundaryCondition.ADIABATIC:
            raise UnsupportedFeatureError(
                f"面 {face} に *SFILM と他の熱境界が同時に指定されています"
            )
        spec = HTBoundarySpec(HTBoundaryCondition.ROBIN, h_conv=film.h, T_inf=film.t_inf)
    return spec


def map_heat_transfer(
    case: CaseDefinition, grid: StructuredGridMap, step: StepDefinition
) -> HeatTransferMappingResult:
    """``*HEAT TRANSFER`` ステップを :class:`HeatTransferInput` に変換する."""
    _check_procedure_common(step, EquationFamily.HEAT_TRANSFER)
    proc = step.procedure
    _, sections = _section_coverage(case, grid)
    k = np.zeros(grid.dimensions)
    C = np.zeros(grid.dimensions)
    for mask, mat, _kind in sections:
        k[mask] = mat.require("conductivity")
        if proc.steady:
            rho = mat.density if mat.density is not None else 1.0
            cp = mat.specific_heat if mat.specific_heat is not None else 1.0
        else:
            rho = mat.require("density")
            cp = mat.require("specific_heat")
        C[mask] = rho * cp

    q = _body_flux(case, step, grid)
    if q is None:
        q = np.zeros(grid.dimensions)
    T0 = _initial_temperature(case, grid, 300.0)

    surface_flux = _surface_flux(case, step, grid)
    films = _face_films(case, step, grid)
    face_bcs = _face_boundaries(case, step, grid)
    specs = {
        face: _ht_bc(face, face_bcs[face], films.get(face), surface_flux.get(face))
        for face in FACE_NAMES
    }

    for cat in (ControlCategory.DISCRETIZATION, ControlCategory.RELAXATION):
        if step.control_values(cat):
            raise UnsupportedFeatureError(
                f"*CONTROLS, PARAMETERS={cat.value} は *HEAT TRANSFER では未対応"
            )
    solver = step.control_values(ControlCategory.SOLVER)
    _check_keys(solver, _HT_SOLVER_KEYS, "SOLVER")
    method = "bicgstab"
    if "METHOD" in solver:
        key = _norm_value(solver["METHOD"])
        if key not in _HT_METHODS:
            raise UnsupportedFeatureError(
                f"SOLVER METHOD={solver['METHOD']} は未対応（{sorted(_HT_METHODS)}）"
            )
        method = _HT_METHODS[key]
    max_iter = _as_int(solver.get("MAX_ITER", "10000"), "MAX_ITER")
    tol = _as_float(solver.get("TOL", "1e-6"), "TOL")
    tinc = step.control_values(ControlCategory.TIME_INCREMENTATION)
    _check_keys(tinc, _TIME_INC_KEYS, "TIME INCREMENTATION")
    output_interval = _as_int(tinc.get("OUTPUT_INTERVAL", "0"), "OUTPUT_INTERVAL")
    if output_interval <= 0:
        output_interval = max((o.frequency for o in step.outputs), default=1)

    dx, dy, dz = grid.spacings
    Lx, Ly, Lz = grid.lengths
    ht_input = HeatTransferInput(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        k=k,
        C=C,
        q=q,
        T0=T0,
        bc_xm=specs["XM"],
        bc_xp=specs["XP"],
        bc_ym=specs["YM"],
        bc_yp=specs["YP"],
        bc_zm=specs["ZM"],
        bc_zp=specs["ZP"],
        dt=proc.dt,
        t_end=proc.time_period,
        max_iter=max_iter,
        tol=tol,
        output_interval=output_interval,
        dx_array=None if grid.is_uniform else dx,
        dy_array=None if grid.is_uniform else dy,
        dz_array=None if grid.is_uniform else dz,
    )
    return HeatTransferMappingResult(input=ht_input, method=method)


class InpToHeatTransferProcess(PreProcess["InpMappingInput", "HeatTransferMappingResult"]):
    """``*HEAT TRANSFER`` ステップを :class:`HeatTransferInput` に変換する PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpToHeatTransfer",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMappingInput) -> HeatTransferMappingResult:
        return map_heat_transfer(input_data.case, input_data.grid, input_data.step)
