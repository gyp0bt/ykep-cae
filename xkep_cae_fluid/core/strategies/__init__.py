"""Strategy Protocol 定義（流体解析向け）.

対流スキーム・乱流モデル・圧力ソルバー等の直交する振る舞い軸を
Protocol で規定する。
"""

from xkep_cae_fluid.core.strategies.protocols import (
    ConvectionSchemeStrategy,
    DiffusionSchemeStrategy,
    LinearSolverStrategy,
    PressureVelocityCouplingStrategy,
    TimeIntegrationStrategy,
    TurbulenceModelStrategy,
)

__all__ = [
    "ConvectionSchemeStrategy",
    "DiffusionSchemeStrategy",
    "TimeIntegrationStrategy",
    "TurbulenceModelStrategy",
    "PressureVelocityCouplingStrategy",
    "LinearSolverStrategy",
]
