"""ジョブ実行: .inp → 解析 → 出力を 1 プロセスで束ねる（``ykep -j=<job>.inp int`` の本体）.

方程式ファミリーで振り分ける:

- ``*NAVIER STOKES`` → :class:`NaturalConvectionFDMProcess`
- ``*HEAT TRANSFER`` → :class:`HeatTransferFDMProcess`
- ``*DARCY`` → :class:`DarcyFlowProcess`（:class:`InpMeshProcess` の非構造メッシュ経由）
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import BatchProcess
from xkep_cae_fluid.darcy.solver import DarcyFlowProcess
from xkep_cae_fluid.heat_transfer.solver import HeatTransferFDMProcess
from xkep_cae_fluid.inp.builder import InpCaseBuildProcess
from xkep_cae_fluid.inp.case import CaseDefinition, EquationFamily, StepDefinition
from xkep_cae_fluid.inp.grid import (
    StructuredGridInput,
    StructuredGridMap,
    StructuredGridRecoveryProcess,
)
from xkep_cae_fluid.inp.mapping import (
    InpMappingInput,
    InpMeshMappingInput,
    InpToDarcyProcess,
    InpToHeatTransferProcess,
    InpToNaturalConvectionProcess,
    UnsupportedFeatureError,
)
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess, InpMeshResult
from xkep_cae_fluid.inp.output import FieldOutputInput, InpOutputWriterProcess, git_commit_hash
from xkep_cae_fluid.inp.parameters import ParameterValue
from xkep_cae_fluid.inp.parser import InpKeywordParseProcess, InpParseInput
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InpJobInput:
    """:class:`InpCaseRunnerProcess` の入力.

    Parameters
    ----------
    path : str
        .inp ファイル
    output_dir : str | None
        出力先（None なら .inp と同じディレクトリ）
    job_name : str | None
        出力ファイルのベース名（None なら .inp の拡張子なし名）
    parameters : Mapping[str, ParameterValue]
        ``*PARAMETER`` の初期値（CLI からの上書き用）
    check_only : bool
        True なら解析を実行せず、読込・格子復元・マッピングまでで終了
    """

    path: str
    output_dir: str | None = None
    job_name: str | None = None
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    check_only: bool = False


@dataclass(frozen=True)
class InpStepResult:
    """1 ステップの実行結果."""

    step_name: str
    family: EquationFamily
    converged: bool
    n_iterations: int
    elapsed_seconds: float
    output_paths: tuple[str, ...]
    summary: Mapping[str, Any]
    result: object | None = None  # ソルバー Result（NaturalConvectionResult / HeatTransferResult）


@dataclass(frozen=True)
class InpJobResult:
    """ジョブ全体の結果."""

    job_name: str
    case: CaseDefinition
    grid: StructuredGridMap | None  # 構造格子（*NAVIER STOKES / *HEAT TRANSFER があるとき）
    steps: tuple[InpStepResult, ...]
    mesh: InpMeshResult | None = None  # 非構造メッシュ（*DARCY があるとき）

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.steps)


def _velocity_field(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.stack([u, v, w], axis=-1)


class InpCaseRunnerProcess(BatchProcess["InpJobInput", "InpJobResult"]):
    """.inp を読み、ステップごとにソルバーを実行して出力する BatchProcess."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="InpCaseRunner",
        module="batch",
        version="0.1.0",
        document_path="../../docs/design/inp-format.md",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [
        InpKeywordParseProcess,
        InpCaseBuildProcess,
        StructuredGridRecoveryProcess,
        InpMeshProcess,
        InpToNaturalConvectionProcess,
        InpToHeatTransferProcess,
        InpToDarcyProcess,
        NaturalConvectionFDMProcess,
        HeatTransferFDMProcess,
        DarcyFlowProcess,
        InpOutputWriterProcess,
    ]

    def process(self, input_data: InpJobInput) -> InpJobResult:
        path = Path(input_data.path)
        job_name = input_data.job_name or path.stem
        out_dir = input_data.output_dir or str(path.parent)

        parsed = InpKeywordParseProcess().execute(
            InpParseInput(path=str(path), parameters=input_data.parameters)
        )
        case = InpCaseBuildProcess().execute(parsed)
        if not case.steps:
            raise UnsupportedFeatureError("*STEP がありません")
        families = {step.procedure.family for step in case.steps}
        heading = case.heading.strip().splitlines()[0] if case.heading.strip() else "(no heading)"

        # 構造格子（NS / 伝熱）と非構造メッシュ（Darcy）は使うステップがあるときだけ組む
        grid: StructuredGridMap | None = None
        if families & {EquationFamily.NAVIER_STOKES, EquationFamily.HEAT_TRANSFER}:
            grid = StructuredGridRecoveryProcess().execute(StructuredGridInput(case=case))
            nx, ny, nz = grid.dimensions
            logger.info(
                "ジョブ %s: %s / 格子 %d×%d×%d (%dD 要素) / ステップ %d",
                job_name,
                heading,
                nx,
                ny,
                nz,
                grid.ndim,
                len(case.steps),
            )
        mesh: InpMeshResult | None = None
        if EquationFamily.DARCY in families:
            mesh = InpMeshProcess().execute(InpMeshInput(case=case))
            logger.info(
                "ジョブ %s: %s / 非構造メッシュ セル %d・面 %d（境界 %d、パッチ %s）/ ステップ %d",
                job_name,
                heading,
                mesh.mesh.n_cells,
                mesh.mesh.n_faces,
                mesh.mesh.n_boundary_faces,
                ", ".join(sorted(mesh.mesh.boundary_patches or ())),
                len(case.steps),
            )

        results: list[InpStepResult] = []
        for idx, step in enumerate(case.steps):
            step_job = job_name if len(case.steps) == 1 else f"{job_name}_{idx + 1}"
            results.append(
                self._run_step(input_data, case, grid, mesh, step, idx, step_job, out_dir)
            )
        return InpJobResult(
            job_name=job_name, case=case, grid=grid, steps=tuple(results), mesh=mesh
        )

    def _run_step(
        self,
        job: InpJobInput,
        case: CaseDefinition,
        grid: StructuredGridMap | None,
        mesh: InpMeshResult | None,
        step: StepDefinition,
        idx: int,
        step_job: str,
        out_dir: str,
    ) -> InpStepResult:
        family = step.procedure.family
        base_summary: dict[str, Any] = {
            "job": step_job,
            "inp": str(Path(job.path)),
            "step": step.name,
            "procedure": family.value,
            "steady": step.procedure.steady,
            "turbulence": step.procedure.turbulence,
            "commit": git_commit_hash(),
            "parameters": {k: v for k, v in case.parameters.items()},
        }
        if family == EquationFamily.DARCY:
            assert mesh is not None
            base_summary["mesh"] = {
                "n_cells": int(mesh.mesh.n_cells),
                "n_faces": int(mesh.mesh.n_faces),
                "n_boundary_faces": int(mesh.mesh.n_boundary_faces),
                "patches": sorted(mesh.mesh.boundary_patches or ()),
            }
        else:
            assert grid is not None
            base_summary["grid"] = {
                "nx": grid.dimensions[0],
                "ny": grid.dimensions[1],
                "nz": grid.dimensions[2],
            }
            base_summary["lengths"] = {
                "lx": grid.lengths[0],
                "ly": grid.lengths[1],
                "lz": grid.lengths[2],
            }
        mapping_input = InpMappingInput(case=case, grid=grid, step_index=idx) if grid else None
        t0 = time.perf_counter()

        if family == EquationFamily.NAVIER_STOKES:
            assert mapping_input is not None
            nc_input = InpToNaturalConvectionProcess().execute(mapping_input)
            base_summary["solver"] = {
                "process": "NaturalConvectionFDMProcess",
                "coupling": nc_input.coupling_method,
                "convection": nc_input.convection_scheme,
                "time_scheme": nc_input.time_scheme,
                "pressure_solver": nc_input.pressure_solver,
                "alpha_u": nc_input.alpha_u,
                "alpha_p": nc_input.alpha_p,
                "alpha_T": nc_input.alpha_T,
                "heat_transfer": step.procedure.heat_transfer,
                "dt": nc_input.dt,
                "t_end": nc_input.t_end,
            }
            if job.check_only:
                return InpStepResult(step.name, family, True, 0, 0.0, (), base_summary, None)
            logger.info("ステップ %s: *NAVIER STOKES → NaturalConvectionFDMProcess", step.name)
            res = NaturalConvectionFDMProcess().execute(nc_input)
            fields = {"U": _velocity_field(res.u, res.v, res.w), "P": res.p, "T": res.T}
            fields.update({str(k): v for k, v in res.extra_scalars.items()})
            fields.update(
                res.residual_fields
            )  # 残差マップ res_u / res_v / res_w / res_T / res_mass
            last = {k: (v[-1] if v else None) for k, v in res.residual_history.items()}
            summary = {
                **base_summary,
                "converged": bool(res.converged),
                "n_outer_iterations": int(res.n_outer_iterations),
                "n_timesteps": int(res.n_timesteps),
                "final_residuals": last,
                "elapsed_seconds": float(res.elapsed_seconds),
                "max_abs_velocity": float(np.max(np.abs(fields["U"]))),
                "temperature_range": [float(res.T.min()), float(res.T.max())],
            }
            n_iter = int(res.n_outer_iterations)
            converged = bool(res.converged)
            result: object = res
        elif family == EquationFamily.HEAT_TRANSFER:
            assert mapping_input is not None
            mapped = InpToHeatTransferProcess().execute(mapping_input)
            base_summary["solver"] = {
                "process": "HeatTransferFDMProcess",
                "method": mapped.method,
                "dt": mapped.input.dt,
                "t_end": mapped.input.t_end,
                "max_iter": mapped.input.max_iter,
                "tol": mapped.input.tol,
            }
            if job.check_only:
                return InpStepResult(step.name, family, True, 0, 0.0, (), base_summary, None)
            logger.info(
                "ステップ %s: *HEAT TRANSFER → HeatTransferFDMProcess(%s)", step.name, mapped.method
            )
            res_ht = HeatTransferFDMProcess(method=mapped.method).execute(mapped.input)
            fields = {"T": res_ht.T, **res_ht.residual_fields}
            summary = {
                **base_summary,
                "converged": bool(res_ht.converged),
                "n_timesteps": int(res_ht.n_timesteps),
                "iteration_counts": [int(c) for c in res_ht.iteration_counts],
                "elapsed_seconds": float(res_ht.elapsed_seconds),
                "temperature_range": [float(res_ht.T.min()), float(res_ht.T.max())],
            }
            n_iter = int(sum(res_ht.iteration_counts))
            converged = bool(res_ht.converged)
            result = res_ht
        elif family == EquationFamily.DARCY:
            assert mesh is not None
            darcy_input = InpToDarcyProcess().execute(
                InpMeshMappingInput(case=case, mesh=mesh, step_index=idx)
            )
            base_summary["solver"] = {
                "process": "DarcyFlowProcess",
                "linear_solver": darcy_input.linear_solver,
                "tol": darcy_input.tol,
                "max_iter": darcy_input.max_iter,
                "viscosity": darcy_input.viscosity,
            }
            if job.check_only:
                return InpStepResult(step.name, family, True, 0, 0.0, (), base_summary, None)
            logger.info(
                "ステップ %s: *DARCY → DarcyFlowProcess(%s)", step.name, darcy_input.linear_solver
            )
            res_d = DarcyFlowProcess().execute(darcy_input)
            fields = {"P": res_d.p, "U": res_d.velocity, "res_mass": res_d.mass_residual}
            summary = {
                **base_summary,
                "converged": bool(res_d.converged),
                "residual": float(res_d.residual),
                "elapsed_seconds": float(res_d.elapsed_seconds),
                "max_abs_velocity": float(np.max(np.abs(res_d.velocity))),
                "pressure_range": [float(res_d.p.min()), float(res_d.p.max())],
                "inflow_m3s": float(res_d.inflow),
                "outflow_m3s": float(res_d.outflow),
                "max_mass_residual": float(np.max(np.abs(res_d.mass_residual))),
            }
            n_iter = 1
            converged = bool(res_d.converged)
            result = res_d
        else:  # pragma: no cover
            raise UnsupportedFeatureError(f"未知の手続き {family}")

        summary["wall_seconds"] = time.perf_counter() - t0
        out = InpOutputWriterProcess().execute(
            FieldOutputInput(
                job_name=step_job,
                output_dir=out_dir,
                grid=grid,
                fields=fields,
                summary=summary,
                requests=step.outputs,
                mesh=None if grid is not None else mesh.mesh if mesh is not None else None,
            )
        )
        logger.info(
            "ステップ %s 完了: converged=%s, 反復 %d, %.2f s → %s",
            step.name,
            converged,
            n_iter,
            summary["elapsed_seconds"],
            ", ".join(Path(p).name for p in out.paths),
        )
        return InpStepResult(
            step_name=step.name,
            family=family,
            converged=converged,
            n_iterations=n_iter,
            elapsed_seconds=float(summary["elapsed_seconds"]),
            output_paths=out.paths,
            summary=summary,
            result=result,
        )
