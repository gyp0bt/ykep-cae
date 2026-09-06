"""``CaseDefinition`` + 構造格子 → ykep 各ソルバー Input への変換.

- :class:`InpToNaturalConvectionProcess`: ``*NAVIER STOKES``（層流、等温/伝熱連成）
  → :class:`~xkep_cae_fluid.natural_convection.data.NaturalConvectionInput`
- :class:`InpToHeatTransferProcess`: ``*HEAT TRANSFER``（構造格子）
  → :class:`~xkep_cae_fluid.heat_transfer.data.HeatTransferInput`
- :class:`InpToHeatTransferFVMProcess`: ``*HEAT TRANSFER``（非構造 :class:`~xkep_cae_fluid.inp.mesh.InpMeshResult` 経由）
  → :class:`~xkep_cae_fluid.heat_transfer.fvm.HeatTransferFVMInput`
- :class:`InpToDarcyProcess`: ``*DARCY``（非構造 :class:`~xkep_cae_fluid.inp.mesh.InpMeshResult` 経由）
  → :class:`~xkep_cae_fluid.darcy.data.DarcyFlowInput`
- :class:`InpToNavierStokesFVMProcess`: ``*NAVIER STOKES``（非構造メッシュ経由）
  → :class:`~xkep_cae_fluid.incompressible.data.NavierStokesFVMInput`

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
from xkep_cae_fluid.darcy.data import DarcyFlowInput, DarcyPatchBC
from xkep_cae_fluid.fvm import BCKind, PatchBC
from xkep_cae_fluid.fvm.momentum import VelocityPatchBC
from xkep_cae_fluid.fvm.viscosity import (
    CarreauViscosity,
    PowerLawViscosity,
    ViscosityModelStrategy,
)
from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition as HTBoundaryCondition,
)
from xkep_cae_fluid.heat_transfer.data import (
    BoundarySpec as HTBoundarySpec,
)
from xkep_cae_fluid.heat_transfer.data import (
    HeatTransferInput,
)
from xkep_cae_fluid.heat_transfer.fvm import HeatTransferFVMInput
from xkep_cae_fluid.incompressible.data import (
    FlowPatchBC,
    InternalCellBC,
    NavierStokesFVMInput,
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
    ViscosityModel,
)
from xkep_cae_fluid.inp.grid import FACE_NAMES, StructuredGridMap
from xkep_cae_fluid.inp.mesh import InpMeshResult
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
class InpMeshMappingInput:
    """非構造メッシュ（:class:`InpMeshResult`）経由のマッピング入力（``*DARCY`` / ``*HEAT TRANSFER`` / ``*NAVIER STOKES``）."""

    case: CaseDefinition
    mesh: InpMeshResult
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


def _body_force_field(
    case: CaseDefinition, step: StepDefinition, mesh: InpMeshResult
) -> np.ndarray | None:
    """``*DLOAD`` の体積力（BX / BY / BZ / BF）を (n_cells, 3) [N/m³] に展開する（無ければ None）."""
    loads = [ld for ld in case.loads + step.loads if ld.is_body_force]
    if not loads:
        return None
    out = np.zeros((mesh.n_cells, 3))
    for ld in loads:
        mask = mesh.mask_for_elements(case.element_ids_of(ld.target))
        out[mask] += np.asarray(ld.vector, dtype=np.float64)
    return out


def _reject_body_force(case: CaseDefinition, step: StepDefinition, where: str) -> None:
    if any(ld.is_body_force for ld in case.loads + step.loads):
        raise UnsupportedFeatureError(
            f"*DLOAD の体積力（BX / BY / BZ / BF）は{where}では未対応（非構造 *NAVIER STOKES のみ）"
        )


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

_NC_DISCRETIZATION_KEYS = {
    "CONVECTION",
    "TIME",
    "PRESSURE_VELOCITY",
    "PISO_CORRECTORS",
    "LIMITER",
    "NONORTHOGONAL_CORRECTORS",
}
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
    _reject_body_force(case, step, "構造格子経路")

    fluid = _fluid_material(case)
    if fluid.viscosity_law is not None:
        raise UnsupportedFeatureError(
            "*VISCOSITY, TYPE=POWER LAW / CARREAU は構造格子経路では未対応（非構造 *NAVIER STOKES のみ）"
        )
    if case.mpcs:
        raise UnsupportedFeatureError(
            "*MPC（回転壁）は構造格子経路では未対応（非構造 *NAVIER STOKES のみ）"
        )
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
    if "NONORTHOGONAL_CORRECTORS" in disc:
        raise UnsupportedFeatureError(
            "NONORTHOGONAL_CORRECTORS= は非構造 NS（NavierStokesFVM）のみ（箱格子では意味が無い）"
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
        solve_energy=coupled,
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
        elif bc.kind == BoundaryKind.WALL:
            # 伝熱では壁 = 断熱（既定と同じ）。バッフルを *BOUNDARY, TYPE=WALL で置くために受理する
            continue
        else:
            raise UnsupportedFeatureError(
                f"面 {face} の {bc.kind.name} 境界は *HEAT TRANSFER では使えません（TEMPERATURE / WALL のみ）"
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


# ---------------------------------------------------------------------------
# 伝熱（HeatTransferFVMProcess、非構造メッシュ経由）
# ---------------------------------------------------------------------------

_LINEAR_SOLVERS = {"DIRECT": "direct", "BICGSTAB": "bicgstab", "AMG": "amg"}


def _spec_to_patch_bc(spec: HTBoundarySpec) -> PatchBC:
    """構造格子版の境界仕様を面ベース FVM の :class:`PatchBC` に写す."""
    if spec.condition == HTBoundaryCondition.DIRICHLET:
        return PatchBC.dirichlet(spec.value)
    if spec.condition == HTBoundaryCondition.NEUMANN:
        return PatchBC.neumann(spec.value)
    if spec.condition == HTBoundaryCondition.ROBIN:
        return PatchBC.robin(spec.h_conv, spec.T_inf)
    return PatchBC.zero_gradient()


def _resolve_patch_name(target: str, mesh: InpMeshResult) -> str:
    name = target.strip().upper()
    patches = mesh.mesh.boundary_patches or {}
    if name not in patches:
        if name in mesh.periodic_surfaces:
            raise UnsupportedFeatureError(
                f"面 {name} は周期境界（*BOUNDARY, TYPE=PERIODIC）なので境界条件を置けません"
            )
        if name in mesh.surface_faces:
            raise UnsupportedFeatureError(
                f"*SURFACE {name} は内部面を含むのでパッチではありません（バッフルにするには"
                f" InpMeshInput.baffle_surfaces に渡す。ykep ランナーは境界条件の target を自動で渡す）"
            )
        raise UnsupportedFeatureError(
            f"境界 target {target!r} は *SURFACE でも予約面名（{', '.join(FACE_NAMES)}）"
            f"でもありません（定義済み: {sorted(patches)}）"
        )
    return name


_BAFFLE_FLOW_KINDS = (BoundaryKind.VELOCITY, BoundaryKind.PRESSURE, BoundaryKind.OUTLET)


def _reject_flow_bc_on_baffle(name: str, bc: BoundaryCondition, mesh: InpMeshResult) -> None:
    """バッフル（厚さゼロ、両側が同じ条件）に流入・流出条件は置けない."""
    if name in mesh.baffle_surfaces and bc.kind in _BAFFLE_FLOW_KINDS:
        raise UnsupportedFeatureError(
            f"バッフル {name}（内部面の *SURFACE）に {bc.kind.name} は置けません"
            f"（WALL / SLIP / SYMMETRY / TEMPERATURE と *DFLUX / *SFILM のみ）"
        )


def _ht_patch_bcs(
    case: CaseDefinition, step: StepDefinition, mesh: InpMeshResult
) -> dict[str, PatchBC]:
    """``*BOUNDARY`` / ``*DFLUX, S`` / ``*SFILM`` をパッチ名 → :class:`PatchBC` に展開する.

    競合規則は構造格子版 :func:`_ht_bc` と同じ（温度固定と熱流束の同時指定は拒否）。
    """
    bcs: dict[str, list[BoundaryCondition]] = {}
    for bc in case.boundaries + step.boundaries:
        bcs.setdefault(_resolve_patch_name(bc.target, mesh), []).append(bc)
    flux: dict[str, float] = {}
    for fl in case.fluxes + step.fluxes:
        if fl.label != FluxLabel.SURFACE:
            continue
        name = _resolve_patch_name(fl.target, mesh)
        flux[name] = flux.get(name, 0.0) + fl.magnitude
    films: dict[str, FilmCondition] = {}
    for film in case.films + step.films:
        name = _resolve_patch_name(film.target, mesh)
        if name in films:
            raise UnsupportedFeatureError(f"面 {name} に *SFILM が重複しています")
        films[name] = film
    out: dict[str, PatchBC] = {}
    for name in sorted(set(bcs) | set(flux) | set(films)):
        spec = _ht_bc(name, bcs.get(name, []), films.get(name), flux.get(name))
        out[name] = _spec_to_patch_bc(spec)
    return out


def _body_flux_unstructured(
    case: CaseDefinition, step: StepDefinition, mesh: InpMeshResult
) -> np.ndarray | None:
    q = np.zeros(mesh.n_cells)
    found = False
    for fl in case.fluxes + step.fluxes:
        if fl.label != FluxLabel.BODY:
            continue
        q[mesh.mask_for_elements(case.element_ids_of(fl.target))] += fl.magnitude
        found = True
    return q if found else None


def map_heat_transfer_fvm(
    case: CaseDefinition, mesh: InpMeshResult, step: StepDefinition
) -> HeatTransferFVMInput:
    """``*HEAT TRANSFER`` ステップを非構造メッシュの :class:`HeatTransferFVMInput` に変換する."""
    _check_procedure_common(step, EquationFamily.HEAT_TRANSFER)
    proc = step.procedure
    k = np.zeros(mesh.n_cells)
    C = np.zeros(mesh.n_cells)
    for mask, mat, _kind in _mesh_section_coverage(case, mesh):
        k[mask] = mat.require("conductivity")
        if proc.steady:
            rho = mat.density if mat.density is not None else 1.0
            cp = mat.specific_heat if mat.specific_heat is not None else 1.0
        else:
            rho = mat.require("density")
            cp = mat.require("specific_heat")
        C[mask] = rho * cp
    q = _body_flux_unstructured(case, step, mesh)
    T0 = _initial_cell_field_unstructured(case, mesh, InitialConditionKind.TEMPERATURE, 300.0)
    bcs = _ht_patch_bcs(case, step, mesh)

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
        if key not in _LINEAR_SOLVERS:
            raise UnsupportedFeatureError(
                f"SOLVER METHOD={solver['METHOD']} は非構造メッシュの伝熱では未対応"
                f"（{sorted(_LINEAR_SOLVERS)}）"
            )
        method = _LINEAR_SOLVERS[key]
    max_iter = _as_int(solver.get("MAX_ITER", "500"), "MAX_ITER")
    tol = _as_float(solver.get("TOL", "1e-8"), "TOL")
    tinc = step.control_values(ControlCategory.TIME_INCREMENTATION)
    _check_keys(tinc, _TIME_INC_KEYS, "TIME INCREMENTATION")
    output_interval = _as_int(tinc.get("OUTPUT_INTERVAL", "0"), "OUTPUT_INTERVAL")
    if output_interval <= 0:
        output_interval = max((o.frequency for o in step.outputs), default=1)
    return HeatTransferFVMInput(
        mesh=mesh.mesh,
        conductivity=k,
        T0=T0,
        heat_capacity=C,
        heat_source=q,
        bcs=bcs,
        dt=proc.dt,
        t_end=proc.time_period,
        output_interval=output_interval,
        linear_solver=method,
        tol=tol,
        max_iter=max_iter,
    )


class InpToHeatTransferFVMProcess(PreProcess["InpMeshMappingInput", "HeatTransferFVMInput"]):
    """``*HEAT TRANSFER`` を非構造メッシュの :class:`HeatTransferFVMInput` に変換する PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpToHeatTransferFVM",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMeshMappingInput) -> HeatTransferFVMInput:
        return map_heat_transfer_fvm(input_data.case, input_data.mesh, input_data.step)


# ---------------------------------------------------------------------------
# Navier–Stokes（NavierStokesFVMProcess、非構造メッシュ経由）
# ---------------------------------------------------------------------------

_COUPLING_FVM: dict[str, str] = {
    "SIMPLE": "simple",
    "SIMPLEC": "simplec",
    "PISO": "piso",
    "COUPLED": "coupled",
}
_NS_FVM_RELAXATION_KEYS = _NC_RELAXATION_KEYS | {"VISCOSITY"}
_DARCY_SOLVER_KEYS = _HT_SOLVER_KEYS | {"MAX_PICARD", "PICARD_TOL"}
_NS_FVM_SOLVER_KEYS = _NC_SOLVER_KEYS | {"MOMENTUM"}
# CONVECTION= の値 → (対流スキーム, リミッタ)。TVD は LIMITER= で既定 van Leer を上書きできる
_CONVECTION_FVM: dict[str, tuple[str, str | None]] = {
    "UPWIND": ("upwind", None),
    "FIRST_ORDER_UPWIND": ("upwind", None),
    "TVD": ("tvd", None),
    "VAN_LEER": ("tvd", "van_leer"),
    "VANLEER": ("tvd", "van_leer"),
    "SUPERBEE": ("tvd", "superbee"),
    "NONE": ("none", None),  # Stokes（対流項なし）
    "STOKES": ("none", None),
}
_LIMITER_FVM: dict[str, str] = {
    "VAN_LEER": "van_leer",
    "VANLEER": "van_leer",
    "SUPERBEE": "superbee",
}


def _oriented(case: CaseDefinition, bc: BoundaryCondition) -> tuple[float, float, float]:
    """境界条件の値ベクトルを全体座標系にする（``ORIENTATION=`` があれば局所 → 全体）."""
    vals = tuple(bc.values) + (0.0,) * (3 - len(bc.values))
    if not bc.orientation:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    ori = case.orientations.get(bc.orientation)
    if ori is None:
        raise UnsupportedFeatureError(
            f"ORIENTATION={bc.orientation} が *ORIENTATION で定義されていません"
        )
    return ori.to_global(vals[:3])


def _reference_node_targets(case: CaseDefinition) -> set[str]:
    """``*MPC`` の参照節点になっている ``*NSET`` 名 / 節点 ID（通常の境界条件から除く）."""
    return {m.master for m in case.mpcs}


def _mpc_patch_bcs(
    case: CaseDefinition, mesh: InpMeshResult, boundaries: list[BoundaryCondition]
) -> dict[str, VelocityPatchBC]:
    """``*MPC`` の従属面 → 参照節点の剛体運動（並進 + 回転）の速度境界条件.

    参照節点への ``*BOUNDARY`` の自由度 1-3 が並進速度、4-6 が角速度 [rad/s]。
    回転中心は参照節点の座標。
    """
    if not case.mpcs:
        return {}
    by_ref: dict[str, dict[BoundaryKind, BoundaryCondition]] = {}
    for bc in boundaries:
        name = bc.target.strip().upper()
        if name not in _reference_node_targets(case):
            continue
        if bc.kind not in (BoundaryKind.VELOCITY, BoundaryKind.ROTATION, BoundaryKind.WALL):
            raise UnsupportedFeatureError(
                f"参照節点 {name} への *BOUNDARY は自由度 1-3（速度）/ 4-6（角速度）のみ"
                f"（{bc.kind.value}）"
            )
        by_ref.setdefault(name, {})[bc.kind] = bc
    node_index = {int(n): i for i, n in enumerate(case.nodes.ids.tolist())}
    out: dict[str, VelocityPatchBC] = {}
    for mpc in case.mpcs:
        ids = case.node_ids_of(mpc.master)
        if ids.size != 1:
            raise UnsupportedFeatureError(
                f"*MPC の参照節点 {mpc.master} は節点 1 つが必要です（{ids.size} 個）"
            )
        center = tuple(float(v) for v in case.nodes.coords[node_index[int(ids[0])]][:3])
        kinds = by_ref.get(mpc.master, {})
        vel = (0.0, 0.0, 0.0)
        omega = (0.0, 0.0, 0.0)
        if BoundaryKind.VELOCITY in kinds:
            vel = _oriented(case, kinds[BoundaryKind.VELOCITY])
        if BoundaryKind.ROTATION in kinds:
            omega = _oriented(case, kinds[BoundaryKind.ROTATION])
        if not kinds:
            logger.warning(
                "*MPC の参照節点 %s に *BOUNDARY がありません（静止壁として扱います）", mpc.master
            )
        patch = _resolve_patch_name(mpc.slave, mesh)
        if patch in out:
            raise UnsupportedFeatureError(f"面 {patch} に *MPC が重複しています")
        out[patch] = VelocityPatchBC.rotating_wall(omega, center, vel)
        logger.info(
            "*MPC %s: 面 %s ← 参照節点 %s（中心 %s, ω=%s rad/s, v=%s m/s）",
            mpc.kind.value,
            patch,
            mpc.master,
            center,
            omega,
            vel,
        )
    return out


def _viscosity_strategy(material: MaterialDefinition) -> ViscosityModelStrategy | None:
    """``*VISCOSITY, TYPE=`` を fvm 層の粘度モデル Strategy にする（ニュートンなら None）."""
    law = material.viscosity_law
    if law is None:
        return None
    if law.model == ViscosityModel.POWER_LAW:
        K, n_idx = law.parameters[0], law.parameters[1]
        gamma_min = law.parameters[2] if len(law.parameters) > 2 else 1.0e-2
        mu_max = law.parameters[3] if len(law.parameters) > 3 else 1.0e8
        return PowerLawViscosity(K=K, n=n_idx, gamma_min=gamma_min, mu_max=mu_max)
    if law.model == ViscosityModel.CARREAU:
        mu_0, mu_inf, lam, n_idx = law.parameters[:4]
        return CarreauViscosity(mu_0=mu_0, mu_inf=mu_inf, lam=lam, n=n_idx)
    raise UnsupportedFeatureError(f"*VISCOSITY, TYPE={law.model.value} は未対応")


def _ns_fvm_patch_bc(
    name: str,
    bcs: list[BoundaryCondition],
    heat_flux: float | None,
    film: FilmCondition | None,
    coupled: bool,
    default: FlowPatchBC,
) -> FlowPatchBC:
    velocity = default.velocity
    thermal: PatchBC | None = default.thermal
    for bc in bcs:
        if bc.kind == BoundaryKind.WALL:
            velocity = VelocityPatchBC.wall()
        elif bc.kind in (BoundaryKind.SLIP, BoundaryKind.SYMMETRY):
            velocity = VelocityPatchBC.slip()
        elif bc.kind == BoundaryKind.VELOCITY:
            vel = list(bc.values) + [0.0] * (3 - len(bc.values))
            v3 = (float(vel[0]), float(vel[1]), float(vel[2]))
            velocity = (
                VelocityPatchBC.wall() if all(v == 0.0 for v in v3) else VelocityPatchBC.inlet(v3)
            )
        elif bc.kind == BoundaryKind.PRESSURE:
            velocity = VelocityPatchBC.outlet(float(bc.values[0]) if bc.values else 0.0)
        elif bc.kind == BoundaryKind.OUTLET:
            velocity = VelocityPatchBC.outflow()
        elif bc.kind == BoundaryKind.TEMPERATURE:
            if not coupled:
                logger.warning("面 %s の温度境界は HEAT TRANSFER=NONE のため無視します", name)
                continue
            if not bc.values:
                raise UnsupportedFeatureError(f"面 {name} の TYPE=TEMPERATURE に値がありません")
            thermal = PatchBC.dirichlet(float(bc.values[0]))
    if heat_flux is not None:
        if not coupled:
            logger.warning("面 %s の *DFLUX は HEAT TRANSFER=NONE のため無視します", name)
        elif thermal is not None and thermal.kind == BCKind.DIRICHLET:
            raise UnsupportedFeatureError(f"面 {name} に温度固定と熱流束が同時に指定されています")
        else:
            thermal = PatchBC.neumann(heat_flux)
    if film is not None:
        if not coupled:
            logger.warning("面 %s の *SFILM は HEAT TRANSFER=NONE のため無視します", name)
        elif thermal is not None:
            raise UnsupportedFeatureError(
                f"面 {name} に *SFILM と他の熱境界が同時に指定されています"
            )
        else:
            thermal = PatchBC.robin(film.h, film.t_inf)
    return FlowPatchBC(velocity=velocity, thermal=thermal)


def map_navier_stokes_fvm(
    case: CaseDefinition, mesh: InpMeshResult, step: StepDefinition
) -> NavierStokesFVMInput:
    """``*NAVIER STOKES`` ステップを非構造メッシュの :class:`NavierStokesFVMInput` に変換する."""
    _check_procedure_common(step, EquationFamily.NAVIER_STOKES)
    proc = step.procedure
    coupled = proc.heat_transfer == "COUPLED"
    if proc.heat_transfer not in ("NONE", "COUPLED"):
        raise UnsupportedFeatureError(f"HEAT TRANSFER={proc.heat_transfer} は NONE / COUPLED のみ")
    fluid = _fluid_material(case)
    rho = fluid.require("density")
    mu = fluid.require("viscosity")
    md = mesh.mesh
    n = mesh.n_cells

    solid_mask = np.zeros(n, dtype=bool)
    k_solid = np.zeros(n)
    has_solid = False
    for mask, mat, kind in _mesh_section_coverage(case, mesh):
        if kind == SectionKind.SOLID:
            has_solid = True
            solid_mask |= mask
            k_solid[mask] = mat.require("conductivity")
    gravity = _gravity_vector(case, step)
    if coupled:
        Cp = fluid.require("specific_heat")
        k_fluid = fluid.require("conductivity")
        beta = fluid.expansion if fluid.expansion is not None else 0.0
        if beta == 0.0 and any(g != 0.0 for g in gravity):
            logger.warning("重力があるのに *EXPANSION が 0（浮力なし）です")
    else:
        Cp = fluid.specific_heat if fluid.specific_heat is not None else 1000.0
        k_fluid = fluid.conductivity if fluid.conductivity is not None else 1.0
        beta = 0.0
    t_default = fluid.reference_temperature if fluid.reference_temperature is not None else 300.0
    T0 = _initial_cell_field_unstructured(case, mesh, InitialConditionKind.TEMPERATURE, t_default)
    if not coupled:
        T0 = np.full(n, float(np.mean(T0)))
    T_ref = (
        fluid.reference_temperature
        if fluid.reference_temperature is not None
        else float(np.mean(T0))
    )
    heat_source = _body_flux_unstructured(case, step, mesh) if coupled else None
    p0 = _initial_cell_field_unstructured(case, mesh, InitialConditionKind.PRESSURE, 0.0)

    # 境界: パッチごとに集約（未指定は静止壁、2D 要素の ZM/ZP は対称面）。
    # target が要素集合（パッチ名ではない elset）なら領域内部の吐出・吸入セル（InternalCellBC）
    bcs_by: dict[str, list[BoundaryCondition]] = {}
    all_bcs = list(case.boundaries + step.boundaries)
    internal_bcs = _internal_cell_bcs(case, mesh, all_bcs, coupled)
    mpc_bcs = _mpc_patch_bcs(case, mesh, all_bcs)
    ref_targets = _reference_node_targets(case)
    patches = md.boundary_patches or {}
    for bc in case.boundaries + step.boundaries:
        name = bc.target.strip().upper()
        if name in ref_targets:
            continue  # *MPC の参照節点（_mpc_patch_bcs が処理済み）
        if name in case.elsets and name not in patches:
            continue
        if bc.kind == BoundaryKind.ROTATION:
            raise UnsupportedFeatureError(
                f"自由度 4-6（角速度）は *MPC の参照節点にだけ与えられます（target={bc.target}）"
            )
        pname = _resolve_patch_name(bc.target, mesh)
        _reject_flow_bc_on_baffle(pname, bc, mesh)
        bcs_by.setdefault(pname, []).append(bc)
    flux: dict[str, float] = {}
    for fl in case.fluxes + step.fluxes:
        if fl.label != FluxLabel.SURFACE:
            continue
        nm = _resolve_patch_name(fl.target, mesh)
        flux[nm] = flux.get(nm, 0.0) + fl.magnitude
    films: dict[str, FilmCondition] = {}
    for film in case.films + step.films:
        nm = _resolve_patch_name(film.target, mesh)
        if nm in films:
            raise UnsupportedFeatureError(f"面 {nm} に *SFILM が重複しています")
        films[nm] = film
    names = set(bcs_by) | set(flux) | set(films) | set(mpc_bcs)
    if mesh.ndim == 2:
        # 2D 要素の z 2 面は既定で対称面（周期にした場合はパッチが無いので付けない）
        names |= {nm for nm in ("ZM", "ZP") if nm in patches}
    bcs: dict[str, FlowPatchBC] = {}
    for nm in sorted(names):
        if nm in mpc_bcs:
            default = FlowPatchBC(velocity=mpc_bcs[nm])
        elif mesh.ndim == 2 and nm in ("ZM", "ZP"):
            default = FlowPatchBC.symmetry()
        else:
            default = FlowPatchBC.wall()
        bcs[nm] = _ns_fvm_patch_bc(
            nm, bcs_by.get(nm, []), flux.get(nm), films.get(nm), coupled, default
        )

    # --- *CONTROLS ---
    disc = step.control_values(ControlCategory.DISCRETIZATION)
    _check_keys(disc, _NC_DISCRETIZATION_KEYS, "DISCRETIZATION")
    convection, limiter = "upwind", "van_leer"
    if "CONVECTION" in disc:
        key = _norm_value(disc["CONVECTION"])
        if key not in _CONVECTION_FVM:
            raise UnsupportedFeatureError(
                f"CONVECTION={disc['CONVECTION']} は未対応（{sorted(_CONVECTION_FVM)}）"
            )
        convection, lim = _CONVECTION_FVM[key]
        if lim is not None:
            limiter = lim
    if "LIMITER" in disc:
        key = _norm_value(disc["LIMITER"])
        if key not in _LIMITER_FVM:
            raise UnsupportedFeatureError(
                f"LIMITER={disc['LIMITER']} は未対応（{sorted(_LIMITER_FVM)}）"
            )
        if convection != "tvd":
            raise UnsupportedFeatureError(
                "LIMITER= は CONVECTION=TVD / VAN_LEER / SUPERBEE と組み合わせる"
            )
        limiter = _LIMITER_FVM[key]
    body_force = _body_force_field(case, step, mesh)
    time_scheme = "euler"
    if "TIME" in disc:
        key = _norm_value(disc["TIME"])
        if key not in _TIME_NC:
            raise UnsupportedFeatureError(f"TIME={disc['TIME']} は未対応（EULER / BDF2）")
        time_scheme = _TIME_NC[key]
    coupling = "simple"
    if "PRESSURE_VELOCITY" in disc:
        key = _norm_value(disc["PRESSURE_VELOCITY"])
        if key not in _COUPLING_FVM:
            raise UnsupportedFeatureError(
                f"PRESSURE_VELOCITY={disc['PRESSURE_VELOCITY']} は未対応"
                "（SIMPLE / SIMPLEC / PISO / COUPLED）"
            )
        coupling = _COUPLING_FVM[key]
    piso_correctors = _as_int(disc.get("PISO_CORRECTORS", "2"), "PISO_CORRECTORS")
    if piso_correctors < 1:
        raise UnsupportedFeatureError("PISO_CORRECTORS は 1 以上")
    relax = step.control_values(ControlCategory.RELAXATION)
    _check_keys(relax, _NS_FVM_RELAXATION_KEYS, "RELAXATION")
    alpha_u = _as_float(relax.get("VELOCITY", "0.7"), "VELOCITY")
    alpha_p = _as_float(relax.get("PRESSURE", "0.3"), "PRESSURE")
    alpha_T = _as_float(relax.get("TEMPERATURE", "0.9"), "TEMPERATURE")
    alpha_mu = _as_float(relax.get("VISCOSITY", "0.5"), "VISCOSITY")
    adaptive = _as_bool(relax.get("ADAPTIVE", "NO"), "ADAPTIVE")
    if adaptive and coupling == "coupled":
        raise UnsupportedFeatureError(
            "ADAPTIVE は COUPLED では意味を持ちません（緩和係数を使わない）"
        )
    viscosity_model = _viscosity_strategy(fluid)
    nonorth = _as_int(disc.get("NONORTHOGONAL_CORRECTORS", "2"), "NONORTHOGONAL_CORRECTORS")
    if nonorth < 1:
        raise UnsupportedFeatureError("NONORTHOGONAL_CORRECTORS は 1 以上")
    solver = step.control_values(ControlCategory.SOLVER)
    _check_keys(solver, _NS_FVM_SOLVER_KEYS, "SOLVER")
    pressure_solver = "bicgstab"
    if "PRESSURE" in solver:
        key = _norm_value(solver["PRESSURE"])
        if key not in _LINEAR_SOLVERS:
            raise UnsupportedFeatureError(
                f"SOLVER PRESSURE={solver['PRESSURE']} は未対応（{sorted(_LINEAR_SOLVERS)}）"
            )
        pressure_solver = _LINEAR_SOLVERS[key]
    momentum_solver = "bicgstab"
    if "MOMENTUM" in solver:
        key = _norm_value(solver["MOMENTUM"])
        if key not in _LINEAR_SOLVERS:
            raise UnsupportedFeatureError(
                f"SOLVER MOMENTUM={solver['MOMENTUM']} は未対応（{sorted(_LINEAR_SOLVERS)}）"
            )
        momentum_solver = _LINEAR_SOLVERS[key]
    max_outer = _as_int(solver.get("MAX_OUTER", "500"), "MAX_OUTER")
    if step.max_increments > 0 and "MAX_OUTER" not in solver:
        max_outer = step.max_increments
    max_inner = _as_int(solver.get("MAX_INNER", "200"), "MAX_INNER")
    max_p = _as_int(solver.get("MAX_PRESSURE_ITER", "0"), "MAX_PRESSURE_ITER")
    if max_p > 0:
        max_inner = max(max_inner, max_p)
    tol = _as_float(solver.get("TOL", "1e-5"), "TOL")
    tol_inner = _as_float(solver.get("TOL_INNER", "1e-8"), "TOL_INNER")
    tinc = step.control_values(ControlCategory.TIME_INCREMENTATION)
    _check_keys(tinc, _TIME_INC_KEYS, "TIME INCREMENTATION")
    output_interval = _as_int(tinc.get("OUTPUT_INTERVAL", "0"), "OUTPUT_INTERVAL")
    if output_interval <= 0:
        output_interval = max((o.frequency for o in step.outputs), default=1)

    return NavierStokesFVMInput(
        mesh=md,
        rho=rho,
        mu=mu,
        bcs=bcs,
        solve_energy=coupled,
        Cp=Cp,
        k_fluid=k_fluid,
        beta=beta,
        T_ref=T_ref,
        gravity=gravity,
        T0=T0,
        p0=p0,
        solid_mask=solid_mask if has_solid else None,
        k_solid=np.where(solid_mask, k_solid, k_fluid) if has_solid else None,
        heat_source=heat_source,
        dt=proc.dt,
        t_end=proc.time_period,
        max_outer_iter=max_outer,
        tol=tol,
        alpha_u=alpha_u,
        alpha_p=alpha_p,
        alpha_T=alpha_T,
        alpha_mu=alpha_mu,
        viscosity_model=viscosity_model,
        coupling=coupling,
        internal_bcs=internal_bcs,
        n_piso_correctors=piso_correctors,
        n_nonorthogonal_correctors=nonorth,
        adaptive_relaxation=adaptive,
        convection=convection,
        limiter=limiter,
        time_scheme=time_scheme,
        body_force=body_force,
        linear_solver=momentum_solver,
        pressure_solver=pressure_solver,
        tol_inner=tol_inner,
        max_inner_iter=max_inner,
        output_interval=output_interval,
    )


def _internal_cell_bcs(
    case: CaseDefinition,
    mesh: InpMeshResult,
    boundaries: list[BoundaryCondition],
    coupled: bool,
) -> tuple[InternalCellBC, ...]:
    """要素集合を target にした ``*BOUNDARY`` を領域内部の吐出・吸入セルに写す.

    - ``TYPE=VELOCITY`` + elset → :meth:`InternalCellBC.inlet`（速度固定、p' = 0）
    - ``TYPE=PRESSURE`` + elset → :meth:`InternalCellBC.outlet`（p' = 0 の圧力基準。値は 0 のみ）
    - ``TYPE=TEMPERATURE`` + elset → 同じ elset の吐出セルの温度（吐出が無ければエラー）
    """
    patches = mesh.mesh.boundary_patches or {}
    by_set: dict[str, dict[str, BoundaryCondition]] = {}
    for bc in boundaries:
        name = bc.target.strip().upper()
        if name not in case.elsets or name in patches:
            continue
        if bc.kind not in (BoundaryKind.VELOCITY, BoundaryKind.PRESSURE, BoundaryKind.TEMPERATURE):
            raise UnsupportedFeatureError(
                f"要素集合 {name} への *BOUNDARY は TYPE=VELOCITY / PRESSURE / TEMPERATURE のみ"
                f"（{bc.kind.value}）"
            )
        by_set.setdefault(name, {})[bc.kind.value] = bc
    out: list[InternalCellBC] = []
    for name, kinds in by_set.items():
        mask = mesh.mask_for_elements(case.elsets[name].ids)
        vel_bc = kinds.get(BoundaryKind.VELOCITY.value)
        p_bc = kinds.get(BoundaryKind.PRESSURE.value)
        t_bc = kinds.get(BoundaryKind.TEMPERATURE.value)
        if vel_bc is not None and p_bc is not None:
            raise UnsupportedFeatureError(
                f"要素集合 {name} に速度固定と圧力基準が同時に指定されています"
            )
        temperature: float | None = None
        if t_bc is not None:
            if vel_bc is None:
                raise UnsupportedFeatureError(
                    f"要素集合 {name} の温度固定は TYPE=VELOCITY（吐出）と組み合わせる"
                )
            if not t_bc.values:
                raise UnsupportedFeatureError(
                    f"要素集合 {name} の TYPE=TEMPERATURE に値がありません"
                )
            if coupled:
                temperature = float(t_bc.values[0])
            else:
                logger.warning("要素集合 %s の温度固定は HEAT TRANSFER=NONE のため無視します", name)
        if vel_bc is not None:
            vel = list(vel_bc.values) + [0.0] * (3 - len(vel_bc.values))
            out.append(
                InternalCellBC.inlet(
                    mask, (float(vel[0]), float(vel[1]), float(vel[2])), temperature, label=name
                )
            )
        elif p_bc is not None:
            if p_bc.values and p_bc.values[0] != 0.0:
                raise UnsupportedFeatureError(
                    f"要素集合 {name} の圧力基準は 0 のみ（p' = 0 のピン留め。相対圧で解く）"
                )
            out.append(InternalCellBC.outlet(mask, label=name))
    return tuple(out)


class InpToNavierStokesFVMProcess(PreProcess["InpMeshMappingInput", "NavierStokesFVMInput"]):
    """``*NAVIER STOKES`` を非構造メッシュの :class:`NavierStokesFVMInput` に変換する PreProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpToNavierStokesFVM",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMeshMappingInput) -> NavierStokesFVMInput:
        return map_navier_stokes_fvm(input_data.case, input_data.mesh, input_data.step)


# ---------------------------------------------------------------------------
# Darcy（DarcyFlowProcess、非構造メッシュ経由）
# ---------------------------------------------------------------------------


def _mesh_section_coverage(
    case: CaseDefinition, mesh: InpMeshResult
) -> list[tuple[np.ndarray, MaterialDefinition, SectionKind]]:
    covered = np.zeros(mesh.n_cells, dtype=bool)
    out: list[tuple[np.ndarray, MaterialDefinition, SectionKind]] = []
    for sec in case.sections:
        mask = mesh.mask_for_elements(case.element_ids_of(sec.elset))
        if np.any(covered & mask):
            raise UnsupportedFeatureError(f"セクションが重複しています（ELSET={sec.elset}）")
        covered |= mask
        out.append((mask, case.material_of_section(sec), sec.kind))
    if not np.all(covered):
        raise UnsupportedFeatureError(
            f"セクション未割当のセルが {int((~covered).sum())} 個あります"
        )
    return out


def _initial_cell_field_unstructured(
    case: CaseDefinition, mesh: InpMeshResult, kind: InitialConditionKind, default: float
) -> np.ndarray:
    """``*INITIAL CONDITIONS`` を非構造メッシュのセル値に展開する（構造格子版と同じ規則）."""
    node_lookup = {int(n): True for n in case.nodes.ids.tolist()}
    node_val = np.full(case.nodes.n_nodes, np.nan)
    node_index = {int(n): i for i, n in enumerate(case.nodes.ids.tolist())}
    cell_override = np.full(mesh.n_cells, np.nan)
    for ic in case.initial_conditions:
        if ic.kind != kind:
            continue
        if not ic.values:
            raise UnsupportedFeatureError(f"*INITIAL CONDITIONS, TYPE={kind.name} に値がありません")
        value = float(ic.values[0])
        key = ic.target
        if key == "ALL":
            node_val[:] = value
            cell_override[:] = np.nan
        elif key in case.nsets:
            node_val[[node_index[int(n)] for n in case.nsets[key].ids.tolist()]] = value
        elif key in case.elsets:
            cell_override[mesh.mask_for_elements(case.elsets[key].ids)] = value
        elif key.isdigit() and int(key) in node_lookup:
            node_val[node_index[int(key)]] = value
        elif key.isdigit():
            cell_override[mesh.mask_for_elements(np.array([int(key)]))] = value
        else:
            raise UnsupportedFeatureError(f"*INITIAL CONDITIONS の target {ic.target!r} が未定義")
    defined = ~np.isnan(node_val)
    cell = np.full(mesh.n_cells, np.nan)
    if np.any(defined):
        cell = mesh.node_values_to_cells(case.nodes.ids[defined], node_val[defined])
    cell = np.where(np.isnan(cell_override), cell, cell_override)
    return np.where(np.isnan(cell), default, cell)


def _darcy_patch_bcs(
    case: CaseDefinition, step: StepDefinition, mesh: InpMeshResult
) -> dict[str, DarcyPatchBC]:
    md = mesh.mesh
    out: dict[str, DarcyPatchBC] = {}
    for bc in case.boundaries + step.boundaries:
        name = _resolve_patch_name(bc.target, mesh)
        _reject_flow_bc_on_baffle(name, bc, mesh)
        if bc.kind == BoundaryKind.PRESSURE:
            if not bc.values:
                raise UnsupportedFeatureError(f"面 {name} の TYPE=PRESSURE に値がありません")
            out[name] = DarcyPatchBC.pressure_bc(float(bc.values[0]))
        elif bc.kind == BoundaryKind.VELOCITY:
            if not bc.values:
                raise UnsupportedFeatureError(f"面 {name} の TYPE=VELOCITY に値がありません")
            if len(bc.values) == 1:
                u_n = float(bc.values[0])
            else:
                vel = np.array(list(bc.values) + [0.0] * (3 - len(bc.values)), dtype=float)[:3]
                faces = md.patch_faces(name)
                n_out = md.face_normals[faces].mean(axis=0)
                norm = float(np.linalg.norm(n_out))
                if norm == 0.0:
                    raise UnsupportedFeatureError(
                        f"面 {name} の法線が定まらないため TYPE=VELOCITY のベクトル指定を"
                        "法線速度に変換できません（1 成分で流入速度を指定してください）"
                    )
                u_n = float(-np.dot(vel, n_out / norm))  # 内向き法線成分（正 = 流入）
            out[name] = DarcyPatchBC.velocity_bc(u_n)
        elif bc.kind in (BoundaryKind.WALL, BoundaryKind.SLIP, BoundaryKind.SYMMETRY):
            out[name] = DarcyPatchBC.wall()
        else:
            raise UnsupportedFeatureError(
                f"面 {name} の {bc.kind.name} 境界は *DARCY では使えません"
                "（PRESSURE / VELOCITY / WALL / SYMMETRY のみ）"
            )
    return out


def map_darcy(case: CaseDefinition, mesh: InpMeshResult, step: StepDefinition) -> DarcyFlowInput:
    """``*DARCY`` ステップを :class:`DarcyFlowInput` に変換する."""
    _check_procedure_common(step, EquationFamily.DARCY)
    proc = step.procedure
    fluid = _fluid_material(case)
    mu = fluid.require("viscosity")
    rho = fluid.density if fluid.density is not None else 1000.0
    permeability = np.zeros(mesh.n_cells)
    forchheimer = np.zeros(mesh.n_cells)
    storage = np.zeros(mesh.n_cells)
    for mask, mat, _kind in _mesh_section_coverage(case, mesh):
        permeability[mask] = mat.require("permeability")
        forchheimer[mask] = mat.forchheimer if mat.forchheimer is not None else 0.0
        storage[mask] = mat.specific_storage if mat.specific_storage is not None else 0.0
    if not proc.steady and not np.any(storage > 0.0):
        raise UnsupportedFeatureError(
            "非定常の *DARCY には *SPECIFIC STORAGE（比貯留係数 S_s > 0）が必要"
        )
    if case.films + step.films or case.fluxes + step.fluxes or case.loads + step.loads:
        raise UnsupportedFeatureError("*SFILM / *DFLUX / *DLOAD は *DARCY では未対応")
    bcs = _darcy_patch_bcs(case, step, mesh)
    p0 = _initial_cell_field_unstructured(case, mesh, InitialConditionKind.PRESSURE, 0.0)

    for cat in (ControlCategory.DISCRETIZATION, ControlCategory.RELAXATION):
        if step.control_values(cat):
            raise UnsupportedFeatureError(f"*CONTROLS, PARAMETERS={cat.value} は *DARCY では未対応")
    tinc = step.control_values(ControlCategory.TIME_INCREMENTATION)
    _check_keys(tinc, _TIME_INC_KEYS, "TIME INCREMENTATION")
    output_interval = _as_int(tinc.get("OUTPUT_INTERVAL", "0"), "OUTPUT_INTERVAL")
    if output_interval <= 0:
        output_interval = max((o.frequency for o in step.outputs), default=1)
    solver = step.control_values(ControlCategory.SOLVER)
    _check_keys(solver, _DARCY_SOLVER_KEYS, "SOLVER")
    max_picard = _as_int(solver.get("MAX_PICARD", "50"), "MAX_PICARD")
    picard_tol = _as_float(solver.get("PICARD_TOL", "1e-8"), "PICARD_TOL")
    method = "direct"
    if "METHOD" in solver:
        key = _norm_value(solver["METHOD"])
        if key not in _LINEAR_SOLVERS:
            raise UnsupportedFeatureError(
                f"SOLVER METHOD={solver['METHOD']} は未対応（{sorted(_LINEAR_SOLVERS)}）"
            )
        method = _LINEAR_SOLVERS[key]
    max_iter = _as_int(solver.get("MAX_ITER", "1000"), "MAX_ITER")
    tol = _as_float(solver.get("TOL", "1e-10"), "TOL")
    return DarcyFlowInput(
        mesh=mesh.mesh,
        permeability=permeability,
        viscosity=mu,
        density=rho,
        bcs=bcs,
        p0=p0,
        linear_solver=method,
        tol=tol,
        max_iter=max_iter,
        forchheimer=forchheimer,
        max_picard_iter=max_picard,
        picard_tol=picard_tol,
        specific_storage=storage,
        dt=proc.dt,
        t_end=proc.time_period,
        output_interval=output_interval,
    )


class InpToDarcyProcess(PreProcess["InpMeshMappingInput", "DarcyFlowInput"]):
    """``*DARCY`` ステップを :class:`DarcyFlowInput` に変換する PreProcess（非構造メッシュ経由）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpToDarcy",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: InpMeshMappingInput) -> DarcyFlowInput:
        return map_darcy(input_data.case, input_data.mesh, input_data.step)
