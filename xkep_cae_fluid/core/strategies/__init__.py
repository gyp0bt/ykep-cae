"""Strategy Protocol 定義（流体解析向け）.

対流スキーム・乱流モデル・圧力ソルバー等の直交する振る舞い軸を
Protocol で規定する。
"""

from xkep_cae_fluid.core.strategies.convection import UpwindConvectionScheme
from xkep_cae_fluid.core.strategies.corrected_diffusion import CorrectedDiffusionScheme
from xkep_cae_fluid.core.strategies.diffusion import CentralDiffusionScheme
from xkep_cae_fluid.core.strategies.protocols import (
    ConvectionSchemeStrategy,
    DiffusionSchemeStrategy,
    LinearSolverStrategy,
    PressureVelocityCouplingStrategy,
    TimeIntegrationStrategy,
    TurbulenceModelStrategy,
)
from xkep_cae_fluid.core.strategies.tvd_convection import (
    TVDConvectionScheme,
    TVDLimiter,
)

__all__ = [
    "ConvectionSchemeStrategy",
    "DiffusionSchemeStrategy",
    "TimeIntegrationStrategy",
    "TurbulenceModelStrategy",
    "PressureVelocityCouplingStrategy",
    "LinearSolverStrategy",
    "CentralDiffusionScheme",
    "CorrectedDiffusionScheme",
    "TVDConvectionScheme",
    "TVDLimiter",
    "UpwindConvectionScheme",
]
