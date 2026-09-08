"""nsbm: nsb（72×48 固定）の初期場を学習で出す別パッケージ（torch 依存は model/train/evaluate のみ）.

families → features → dataset → model → train → evaluate の直線パイプライン。nsb 本体は変更しない。
"""

from nsbm.families import (
    FAMILIES,
    LX,
    LY,
    NX,
    NY,
    Port,
    Theta,
    build_bc,
    build_h,
    build_input,
    connected,
    port_cells,
    sample_theta,
)

__all__ = [
    "FAMILIES",
    "LX",
    "LY",
    "NX",
    "NY",
    "Port",
    "Theta",
    "build_bc",
    "build_h",
    "build_input",
    "connected",
    "port_cells",
    "sample_theta",
]
