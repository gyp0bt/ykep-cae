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


@pytest.fixture(scope="module")
def small_pin(small):
    cfg, geo = small
    from dataclasses import replace

    return replace(cfg, pin_fin=True), geo


class TestPinFin:
    def test_permeability_limits_and_monotone(self, small_pin):
        cfg, _ = small_pin
        s = torch.linspace(0.0, 1.0, 21)
        k = dc.permeability(cfg, s)
        k_plate = cfg.t_chan**2 / 12.0
        assert abs(float(k[0]) - k_plate) / k_plate < 1e-6  # φ=0 で平板に一致
        assert (k[1:] < k[:-1]).all()  # φ 単調で K 単調減少
        assert float(k[-1]) > 0.0

    def test_interlayer_u_monotone_increase(self, small_pin):
        cfg, _ = small_pin
        s = torch.linspace(0.0, 1.0, 21)
        u = dc.interlayer_u(cfg, s)
        assert (u[1:] > u[:-1]).all()  # ピンが増えるほど z 方向実効 h が上がる
        # φ=0 はプライム面のみ = 旧モデルの s=0 (全水) と同じ床
        u0_ref = 1.0 / (cfg.t_base / (2 * cfg.k_s) + cfg.t_chan / (2 * cfg.k_f))
        assert abs(float(u[0]) - u0_ref) / u0_ref < 1e-6

    def test_legacy_mode_unchanged(self, small):
        """pin_fin=False の数値が従来コードと一致し続けること (回帰)."""
        cfg, _ = small
        s = torch.linspace(0.0, 1.0, 5)
        k_legacy = (cfg.t_chan**2 / 12.0) * torch.exp(torch.log(torch.tensor(cfg.k_ratio)) * s)
        assert torch.allclose(dc.permeability(cfg, s), k_legacy)
        r_half = cfg.t_base / (2 * cfg.k_s) + cfg.t_chan / (2 * dc.conductivity(cfg, s))
        assert torch.allclose(dc.interlayer_u(cfg, s), 1.0 / r_half)

    def test_objective_gradient_pin(self, small_pin):
        cfg, geo = small_pin
        dp_ref = dc.dp_reference(cfg, geo)
        rng = np.random.default_rng(2)
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

    def test_conservation_pin(self, small_pin, rand_field):
        cfg, geo = small_pin
        r = dc.evaluate(cfg, geo, rand_field)
        assert r["heat_balance_rel"] < 1e-8
        assert r["T_block_min"] > r["T_fluid_out"] > 0.0

    def test_u_bounded_by_base_conduction(self, small_pin):
        """U'' はベース板半厚み伝導の直列上限 1/r_base を超えない."""
        cfg, _ = small_pin
        u = dc.interlayer_u(cfg, torch.linspace(0.0, 1.0, 21))
        assert float(u.max()) < 2.0 * cfg.k_s / cfg.t_base

    def test_fin_lever_at_small_phi(self, small_pin):
        """少量のピンで z 方向実効 h が大きく立ち上がる (フィン増倍のレバー)."""
        cfg, _ = small_pin
        u = dc.interlayer_u(cfg, torch.tensor([0.0, 0.1]))
        assert float(u[1] / u[0]) > 2.0

    def test_forchheimer_requires_pin_fin(self, small):
        cfg, geo = small
        from dataclasses import replace

        cfg_f = replace(cfg, forchheimer=True)  # pin_fin=False のまま
        with pytest.raises(ValueError):
            dc.solve_flow(cfg_f, geo, torch.zeros(geo.n_cells))

    def test_forchheimer_increases_dp(self, small_pin, rand_field):
        """慣性損失は同一場で ΔP を増やす (全開では β=0 でほぼ不変)."""
        cfg, geo = small_pin
        from dataclasses import replace

        cfg_f = replace(cfg, forchheimer=True)
        dp_d = dc.evaluate(cfg, geo, rand_field)["dp"]
        dp_f = dc.evaluate(cfg_f, geo, rand_field)["dp"]
        assert dp_f > dp_d
        open_field = torch.zeros(geo.by, geo.bx)
        dp_do = dc.evaluate(cfg, geo, open_field)["dp"]
        dp_fo = dc.evaluate(cfg_f, geo, open_field)["dp"]
        assert abs(dp_fo - dp_do) / dp_do < 1e-6

    def test_forchheimer_mass_conservation(self, small_pin, rand_field):
        """Picard 収束後も面フラックスは厳密に発散ゼロ (最終反復が線形解)."""
        cfg, geo = small_pin
        from dataclasses import replace

        cfg_f = replace(cfg, forchheimer=True)
        with torch.no_grad():
            s = dc.expand(geo, rand_field)
            flow = dc.solve_flow(cfg_f, geo, s)
        assert flow["picard_iters"] < cfg_f.picard_max  # 収束している
        q_v = cfg.m_dot / cfg.rho_f
        div = np.zeros(geo.n_cells)
        qf = flow["q_face"].numpy()
        np.add.at(div, geo.fi.numpy(), qf)
        np.add.at(div, geo.fj.numpy(), -qf)
        div[geo.inlet_cells.numpy()] -= q_v / len(geo.inlet_cells)
        div[geo.outlet_cells.numpy()] += flow["q_out"].numpy()
        assert np.abs(div).max() / q_v < 1e-9

    def test_forchheimer_gradient(self, small_pin):
        """Picard 反復ごしの adjoint 勾配を FD 照合."""
        cfg, geo = small_pin
        from dataclasses import replace

        cfg_f = replace(cfg, forchheimer=True)
        dp_ref = dc.dp_reference(cfg_f, geo)
        rng = np.random.default_rng(3)
        xi = torch.tensor(0.5 * rng.standard_normal((geo.by, geo.bx)), requires_grad=True)
        j, _ = dc.objective(cfg_f, geo, xi, gamma_p=1.0, dp_ref=dp_ref)
        (g,) = torch.autograd.grad(j, xi)
        eps = 1e-6
        err = 0.0
        for k in range(0, xi.numel(), max(1, xi.numel() // 6)):
            i, jj = np.unravel_index(k, xi.shape)
            xp = xi.detach().clone()
            xp[i, jj] += eps
            xm = xi.detach().clone()
            xm[i, jj] -= eps
            jp, _ = dc.objective(cfg_f, geo, xp, 1.0, dp_ref)
            jm, _ = dc.objective(cfg_f, geo, xm, 1.0, dp_ref)
            fd = (float(jp) - float(jm)) / (2 * eps)
            an = float(g[i, jj])
            err = max(err, abs(fd - an) / max(abs(fd), abs(an), 1e-12))
        assert err < 1e-4  # Picard 打ち切り分だけ緩め

    def test_pin_more_conservative_than_legacy(self, small_pin, small, rand_field):
        """同一 s 場でピンフィンは旧モデルより板温度が高い (保守側).

        旧モデルは線形混合 k_c を層厚伝導に使うため、少量の固体でも
        「流体温度へ直結する垂直伝導スラブ」となり非物理に楽観的だった。
        ピン版は側面→流体の対流段 (フィン抵抗) を挟むので U が下がり、
        その分だけ温度評価が正直になる — これが本修正の狙いそのもの。
        """
        cfg_pin, geo = small_pin
        cfg_old, _ = small
        t_pin = dc.evaluate(cfg_pin, geo, rand_field)["T_peak"]
        t_old = dc.evaluate(cfg_old, geo, rand_field)["T_peak"]
        assert t_pin > t_old
