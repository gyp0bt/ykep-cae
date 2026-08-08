"""darcy.py (擬3D ダルシー則 2 目的最適化) のテスト.

プログラムテスト: TestDarcyAPI (勾配・収束・API)
物理テスト:      TestDarcyPhysics (質量保存・エネルギー収支・単調性)
"""

import darcy as dc
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def small():
    cfg = dc.DarcyConfig(n_cols=1, n_rows=2, margin_left=1, margin_right=1, margin_y=1, refine=2)
    return cfg, dc.make_geo(cfg)


@pytest.fixture(scope="module")
def rand_field(small):
    _, geo = small
    rng = np.random.default_rng(7)
    return torch.tensor(rng.uniform(0.0, 0.9, size=(geo.by, geo.bx)))


class TestDarcyAPI:
    def test_sparse_solve_gradient(self):
        # 小さな対角優位系で adjoint 勾配を FD 照合
        rng = np.random.default_rng(0)
        n = 8
        rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
        cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
        w = torch.tensor(rng.uniform(0.5, 1.5, size=rows.shape[0]), requires_grad=True)
        base = torch.tensor(np.where(rows == cols, 4.0, -1.0))
        b = torch.tensor(rng.standard_normal(n))

        def f(wv):
            return (dc.sparse_solve(base * wv, b, rows, cols, n) ** 2).sum()

        j = f(w)
        (g,) = torch.autograd.grad(j, w)
        eps = 1e-6
        for k in (0, 3, n, 2 * n - 2):
            wp = w.detach().clone()
            wp[k] += eps
            wm = w.detach().clone()
            wm[k] -= eps
            fd = (float(f(wp)) - float(f(wm))) / (2 * eps)
            assert abs(fd - float(g[k])) / max(abs(fd), 1e-12) < 1e-6

    def test_objective_gradient(self, small):
        cfg, geo = small
        dp_ref = dc.dp_reference(cfg, geo)
        rng = np.random.default_rng(1)
        xi = torch.tensor(0.5 * rng.standard_normal((geo.by, geo.bx)), requires_grad=True)
        j, _ = dc.objective(cfg, geo, xi, gamma_p=1.0, dp_ref=dp_ref)
        (g,) = torch.autograd.grad(j, xi)
        eps = 1e-6
        err = 0.0
        for k in range(0, xi.numel(), max(1, xi.numel() // 8)):
            i, jj = np.unravel_index(k, xi.shape)
            xp = xi.detach().clone()
            xp[i, jj] += eps
            xm = xi.detach().clone()
            xm[i, jj] -= eps
            jp, _ = dc.objective(cfg, geo, xp, 1.0, dp_ref)
            jm, _ = dc.objective(cfg, geo, xm, 1.0, dp_ref)
            fd = (float(jp) - float(jm)) / (2 * eps)
            an = float(g[i, jj])
            err = max(err, abs(fd - an) / max(abs(fd), abs(an), 1e-12))
        assert err < 1e-5

    def test_optimize_decreases(self, small):
        cfg, geo = small
        dp_ref = dc.dp_reference(cfg, geo)
        rng = np.random.default_rng(0)
        xi0 = torch.tensor(0.05 * rng.standard_normal((geo.by, geo.bx)))
        j0, _ = dc.objective(cfg, geo, xi0, 0.3, dp_ref)
        xi = dc.optimize(cfg, geo, gamma_p=0.3, iters=40, seed=0)
        j1, _ = dc.objective(cfg, geo, xi, 0.3, dp_ref)
        assert float(j1) < float(j0)


class TestDarcyPhysics:
    def test_mass_conservation(self, small, rand_field):
        cfg, geo = small
        with torch.no_grad():
            s = dc.expand(geo, rand_field)
            flow = dc.solve_flow(cfg, geo, s)
        q_v = cfg.m_dot / cfg.rho_f
        n = geo.n_cells
        div = np.zeros(n)
        qf = flow["q_face"].numpy()
        np.add.at(div, geo.fi.numpy(), qf)
        np.add.at(div, geo.fj.numpy(), -qf)
        div[geo.inlet_cells.numpy()] -= q_v / len(geo.inlet_cells)
        div[geo.outlet_cells.numpy()] += flow["q_out"].numpy()
        assert np.abs(div).max() / q_v < 1e-9
        assert abs(float(flow["q_out"].sum()) / q_v - 1.0) < 1e-9

    def test_energy_balance(self, small, rand_field):
        cfg, geo = small
        r = dc.evaluate(cfg, geo, rand_field)
        assert r["heat_balance_rel"] < 1e-8

    def test_fluid_outlet_equals_tref(self, small, rand_field):
        cfg, geo = small
        r = dc.evaluate(cfg, geo, rand_field)
        t_ref = dc.t_ref_scale(cfg, geo)
        assert abs(r["T_fluid_out"] - t_ref) / t_ref < 1e-6

    def test_temperature_ordering(self, small, rand_field):
        cfg, geo = small
        r = dc.evaluate(cfg, geo, rand_field)
        assert r["T_block_min"] > r["T_fluid_out"] > 0.0

    def test_pressure_monotone_in_solidity(self, small):
        cfg, geo = small
        open_field = torch.zeros(geo.by, geo.bx)
        band = torch.zeros(geo.by, geo.bx)
        band[geo.by // 2, :] = 0.8  # 横一列の多孔質バッフル
        dp0 = dc.evaluate(cfg, geo, open_field)["dp"]
        dp1 = dc.evaluate(cfg, geo, band)["dp"]
        assert dp1 > dp0
