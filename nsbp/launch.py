"""MPI 起動の補助: PETSc 同梱 MPICH の mpiexec を探す（petsc4py は import しない）."""

from __future__ import annotations

import os
import shutil


def petsc_dir() -> str | None:
    """petsc4py が組まれた PETSC_DIR（無ければ環境変数 PETSC_DIR）."""
    try:
        import petsc4py

        cfg = petsc4py.get_config()
        d = cfg.get("PETSC_DIR")
        arch = cfg.get("PETSC_ARCH") or ""
        if d:
            return os.path.join(d, arch) if arch else str(d)
    except ImportError:
        pass
    return os.environ.get("PETSC_DIR")


def mpiexec_path() -> str | None:
    """PETSc が同梱ビルドした mpiexec（`--download-mpich`）を優先し、無ければ PATH の mpiexec."""
    d = petsc_dir()
    if d:
        cand = os.path.join(d, "bin", "mpiexec")
        if os.path.exists(cand):
            return cand
    return shutil.which("mpiexec")
