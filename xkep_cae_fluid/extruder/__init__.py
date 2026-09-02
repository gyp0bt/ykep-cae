"""単軸押出（展開チャネル 2.5D）解析パッケージ.

設計文書: docs/design/single-screw-extruder.md
実装計画: docs/plans/2026-09-02-single-screw-extruder-impl.md
"""

from xkep_cae_fluid.extruder.data import ChannelGrid, ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

__all__ = ["ChannelGrid", "ScrewGeometryProcess", "ScrewSpec"]
