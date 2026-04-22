"""水槽設計 CAE モジュール (Phase 6).

90×30×45 cm 水草水槽の熱流体・物質輸送・生体反応を統合した
持続的設計 CAE のドメイン固有プロセス群。

- `geometry.py`: AquariumGeometryProcess（Phase 6.2a）
- `heater.py`: HeaterProcess（Phase 6.2b）
- `filter.py`: AquariumFilterProcess（Phase 6.3a）
"""

from xkep_cae_fluid.aquarium.filter import (
    AquariumFilterInput,
    AquariumFilterProcess,
    AquariumFilterResult,
    NozzleGeometry,
)
from xkep_cae_fluid.aquarium.geometry import (
    AquariumGeometryInput,
    AquariumGeometryProcess,
    AquariumGeometryResult,
)
from xkep_cae_fluid.aquarium.heater import (
    HeaterGeometry,
    HeaterInput,
    HeaterMode,
    HeaterProcess,
    HeaterResult,
)

__all__ = [
    "AquariumFilterInput",
    "AquariumFilterProcess",
    "AquariumFilterResult",
    "AquariumGeometryInput",
    "AquariumGeometryProcess",
    "AquariumGeometryResult",
    "HeaterGeometry",
    "HeaterInput",
    "HeaterMode",
    "HeaterProcess",
    "HeaterResult",
    "NozzleGeometry",
]
