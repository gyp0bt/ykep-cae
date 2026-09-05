"""``python -m xkep_cae_fluid.inp -j=case.inp int``."""

from __future__ import annotations

import sys

from xkep_cae_fluid.inp.cli import main

if __name__ == "__main__":
    sys.exit(main())
