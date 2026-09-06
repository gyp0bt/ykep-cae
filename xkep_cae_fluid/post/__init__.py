"""後処理（PostProcess）パッケージ.

- :mod:`xkep_cae_fluid.post.mirador` -- 構造格子の解析結果を messi の three.js ビューア
  （``mirador``）で 3D レンダリングする :class:`MiradorExportProcess`
- :mod:`xkep_cae_fluid.post.tracking` -- 面流束から再構成したセル内アフィン場を辿る
  Pollock 型の粒子追跡 :class:`ParticleTrackFVMProcess`（非構造メッシュ）
- :mod:`xkep_cae_fluid.post.rtd` -- 粒子追跡結果から滞留時間分布を作る
  :class:`ResidenceTimeProcess`
- :mod:`xkep_cae_fluid.post.statistics` -- 流束重み付きの分位点・経験分布
"""

from xkep_cae_fluid.post.mirador import (
    MiradorExportInput,
    MiradorExportProcess,
    MiradorExportResult,
    MiradorUnavailableError,
    SlicePlane,
    fields_from_heat_transfer,
    fields_from_natural_convection,
    lines_from_structured_mesh,
    load_npz_fields,
)
from xkep_cae_fluid.post.rtd import (
    ResidenceTimeInput,
    ResidenceTimeProcess,
    ResidenceTimeResult,
)
from xkep_cae_fluid.post.statistics import weighted_ecdf, weighted_quantile
from xkep_cae_fluid.post.tracking import (
    CellFaceTable,
    ParticleTrackFVMInput,
    ParticleTrackFVMProcess,
    ParticleTrackFVMResult,
    cell_face_table,
    reconstruct_cell_velocity,
)

__all__ = [
    "CellFaceTable",
    "MiradorExportInput",
    "MiradorExportProcess",
    "MiradorExportResult",
    "MiradorUnavailableError",
    "ParticleTrackFVMInput",
    "ParticleTrackFVMProcess",
    "ParticleTrackFVMResult",
    "ResidenceTimeInput",
    "ResidenceTimeProcess",
    "ResidenceTimeResult",
    "SlicePlane",
    "cell_face_table",
    "fields_from_heat_transfer",
    "fields_from_natural_convection",
    "lines_from_structured_mesh",
    "load_npz_fields",
    "reconstruct_cell_velocity",
    "weighted_ecdf",
    "weighted_quantile",
]
