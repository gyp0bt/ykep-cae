"""単軸押出（展開チャネル 2.5D）解析パッケージ.

設計文書: docs/design/single-screw-extruder.md
実装計画: docs/plans/2026-09-02-single-screw-extruder-impl.md

使い方::

    proc = ExtruderFlowProcess()
    proc.viscosity = PowerLawViscosity(K=2e4, n=0.4)
    flow = proc.process(ExtruderFlowInput(spec=ScrewSpec(...), G=5e6))
    track = ParticleTrackerProcess().process(
        ParticleTrackInput(flow=flow, z_axial=0.200)
    )
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_axial=0.200))

押出量は `flow.Q_axial`（`flow.Q` ではない。設計文書 §2.1.2）。
"""

from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
from xkep_cae_fluid.extruder.data import (
    ChannelGrid,
    CrossChannelInput,
    CrossChannelResult,
    DownChannelInput,
    DownChannelResult,
    ExtruderFlowInput,
    ExtruderFlowResult,
    ParticleTrackInput,
    ParticleTrackResult,
    RTDInput,
    RTDResult,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.rtd import RTDProcess
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
    ViscosityModelStrategy,
    mixing_index,
    strain_rate,
)

__all__ = [
    "CarreauViscosity",
    "ChannelGrid",
    "CrossChannelInput",
    "CrossChannelResult",
    "CrossChannelStokesProcess",
    "DownChannelFlowProcess",
    "DownChannelInput",
    "DownChannelResult",
    "ExtruderFlowInput",
    "ExtruderFlowProcess",
    "ExtruderFlowResult",
    "NewtonianViscosity",
    "ParticleTrackInput",
    "ParticleTrackResult",
    "ParticleTrackerProcess",
    "PowerLawViscosity",
    "RTDInput",
    "RTDProcess",
    "RTDResult",
    "ScrewGeometryProcess",
    "ScrewSpec",
    "ViscosityModelStrategy",
    "metering_flow_rate",
    "mixing_index",
    "shape_factor_drag",
    "shape_factor_pressure",
    "strain_rate",
]
