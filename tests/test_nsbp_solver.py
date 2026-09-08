"""nsbp.solver（PETSc 駆動の Newton + 擬似時間）のテスト。petsc4py が無ければ skip."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from nsb.geo import make_case
from nsbp import HAVE_PETSC
from nsbp.launch import mpiexec_path

pytestmark = pytest.mark.skipif(not HAVE_PETSC, reason="petsc4py が無い")

if HAVE_PETSC:
    from nsbp.solver import NSBPSettings, NSBPSolver, solve_steady


class TestNSBPSolverAPI:
    def test_fd_jacobian_newton_on_stokes_residual(self):
        """対流を落とした残差（cs=0）に FD カラーリングのヤコビアンで Newton すると 3 回で 1e-6 以下."""
        inp = make_case("flat", 1, u_in=0.1)
        solver = NSBPSolver(inp, NSBPSettings())
        try:
            X, F = solver.X, solver.F
            X.set(0.0)
            r0 = solver.residual(X, F, tau=False, cs=0.0)
            b = F.duplicate()
            dX = X.duplicate()
            hist = []
            for _ in range(3):
                solver.jacobian_steady(X, cs=0.0)
                F.copy(b)
                b.scale(-1.0)
                n_it, ok, ratio = solver.linear_solve(solver.J, b, dX, rtol=1e-12)
                assert ok and ratio < 1e-10
                X.axpy(1.0, dX)
                hist.append(solver.residual(X, F, tau=False, cs=0.0) / r0)
            assert hist[0] < 0.5
            assert hist[-1] < 1e-6, hist  # FD ヤコビアンの精度（1e-7 程度）なりの 2 次収束
            assert solver.counts["resid"] < 3 * 90  # 幅 2 box の色数 75 + 数回
        finally:
            solver.destroy()

    def test_pc_kind_validated(self):
        inp = make_case("flat", 1, u_in=0.1)
        with pytest.raises(ValueError):
            NSBPSolver(inp, NSBPSettings(pc="nope"))


class TestNSBPSolverConvergence:
    def test_flat_converges_and_matches_nsb(self):
        from nsb.core import NSBSettings
        from nsb.solver import solve_steady as nsb_solve

        inp = make_case("flat", 1, u_in=0.1)
        res = solve_steady(inp, NSBPSettings(), log=None)
        assert res.converged, res.failure_reason
        assert res.rel_steady_residual < 1e-6
        assert abs(res.mass_in - res.mass_out) < 1e-6 * abs(res.mass_in)
        ref = nsb_solve(make_case("flat", 1, u_in=0.1, settings=NSBSettings()), log=None)
        assert ref.converged
        assert np.abs(res.u - ref.u).max() < 1e-4 * np.abs(ref.u).max()
        assert np.abs(res.p - ref.p).max() < 1e-4 * np.abs(ref.p).max()

    def test_uturn_u1_converges(self):
        inp = make_case("uturn", 1, u_in=1.0)
        res = solve_steady(inp, NSBPSettings(), log=None)
        assert res.converged, res.failure_reason
        assert abs(res.mass_in - res.mass_out) < 1e-6 * abs(res.mass_in)


MPI_SCRIPT = """
import json, sys
import numpy as np
from nsb.geo import make_case
from nsbp.solver import NSBPSettings, solve_steady
from petsc4py import PETSc
res = solve_steady(make_case("uturn", 1, u_in=1.0), NSBPSettings(), log=None)
if PETSc.COMM_WORLD.getRank() == 0:
    np.save(sys.argv[1], np.stack([res.u, res.v, res.p]))
    print(json.dumps({"converged": res.converged, "it": res.n_iter, "ranks": res.n_ranks}))
"""


@pytest.mark.slow
class TestNSBPSolverMPI:
    def test_three_ranks_reproduce_serial_solution(self, tmp_path):
        mpiexec = mpiexec_path()
        if mpiexec is None:
            pytest.skip("mpiexec が無い")
        script = tmp_path / "run.py"
        script.write_text(MPI_SCRIPT)
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        outs = {}
        for n in (1, 3):
            out = tmp_path / f"sol{n}.npy"
            cp = subprocess.run(
                [mpiexec, "-n", str(n), sys.executable, str(script), str(out)],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
                check=False,
            )
            assert cp.returncode == 0, cp.stderr[-2000:]
            info = __import__("json").loads(cp.stdout.strip().splitlines()[-1])
            assert info["converged"] and info["ranks"] == n
            outs[n] = np.load(out)
        d = np.abs(outs[3] - outs[1]).max(axis=(1, 2)) / np.abs(outs[1]).max(axis=(1, 2))
        # ブロック Jacobi は分割で前処理が変わり Newton 経路とリミター凍結時点が僅かに動くので、
        # 相対残差 1e-6 で収束した 2 つの解は 1e-6 程度ずれる（実測 2.5e-6）。桁が合えばよい
        assert d.max() < 1e-4, d
