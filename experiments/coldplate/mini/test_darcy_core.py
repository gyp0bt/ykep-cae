"""darcy_core.py の検定器 — 写経再構築版はこの 10 件を通れば正しい.

検定の設計思想: 実装の詳細 (行列の組み方・反復の書き方) には依存せず、
物理と数学が要求する不変量だけを検査する。

  保存則     — 質量 (発散ゼロ)・エネルギー (発熱 = 出口エンタルピー)
  随伴の正しさ — adjoint 勾配が有限差分と一致 (単体・目的関数ごし・Picard ごし)
  閉包の性質  — K 単調減少・U'' 単調増加と物理限界
  最適化     — 目的が実際に下がる

実行: OMP_NUM_THREADS=4 python -m pytest test_darcy_core.py -q  (~10 秒)
"""

from dataclasses import replace

import darcy_core as dc
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def small():
    cfg = dc.Config(n_cols=1, n_rows=2, margin_left=1, margin_right=1, margin_y=1, refine=2)
    return cfg, dc.make_geo(cfg)


@pytest.fixture(scope="module")
def rand_field(small):
    _, geo = small
    rng = np.random.default_rng(7)
    return torch.tensor(rng.uniform(0.0, 0.9, size=(geo.by, geo.bx)))


def fd_check(f, x, ks, eps=1e-6, tol=1e-5):
    """f のスカラー出力の勾配を、指定成分 ks で中心差分と照合する."""
    x = x.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(f(x), x)
    for k in ks:
        i = np.unravel_index(k, x.shape)
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[i] += eps
        xm[i] -= eps
        fd = (float(f(xp)) - float(f(xm))) / (2 * eps)
        an = float(g[i])
        assert abs(fd - an) / max(abs(fd), abs(an), 1e-12) < tol, f"成分 {i}: FD={fd} 解析={an}"


class TestAdjoint:
    def test_sparse_solve_gradient(self):
        """随伴 backward の単体検定: 小さな対角優位系で FD 照合."""
        rng = np.random.default_rng(0)
        n = 8
        rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
        cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
        base = torch.tensor(np.where(rows == cols, 4.0, -1.0))
        b = torch.tensor(rng.standard_normal(n))
        w0 = torch.tensor(rng.uniform(0.5, 1.5, size=rows.shape[0]))
        fd_check(lambda w: (dc.sparse_solve(base * w, b, rows, cols, n) ** 2).sum(), w0, [0, 3, 9])

    def test_objective_gradient_darcy(self, small):
        cfg, geo = small
        dp_ref = dc.dp_reference(cfg, geo)
        rng = np.random.default_rng(1)
        xi = torch.tensor(0.5 * rng.standard_normal((geo.by, geo.bx)))
        ks = list(range(0, xi.numel(), max(1, xi.numel() // 6)))
        fd_check(lambda x: dc.objective(cfg, geo, x, 1.0, dp_ref), xi, ks)

    def test_objective_gradient_forchheimer(self, small):
        """Picard 反復ごしでも勾配が通る (打ち切り分だけ tol 緩め)."""
        cfg, geo = small
        cfg_f = replace(cfg, forchheimer=True)
        dp_ref = dc.dp_reference(cfg_f, geo)
        rng = np.random.default_rng(3)
        xi = torch.tensor(0.5 * rng.standard_normal((geo.by, geo.bx)))
        fd_check(lambda x: dc.objective(cfg_f, geo, x, 1.0, dp_ref), xi, [0, 7], tol=1e-4)


class TestConservation:
    @pytest.mark.parametrize("forch", [False, True])
    def test_mass(self, small, rand_field, forch):
        """発散 = 流入 - 流出が全セルでゼロ (Forchheimer 込みでも厳密)."""
        cfg, geo = small
        cfg = replace(cfg, forchheimer=forch)
        with torch.no_grad():
            flow = dc.solve_flow(cfg, geo, dc.expand(geo, rand_field))
        q_v = cfg.m_dot / cfg.rho_f
        div = np.zeros(geo.n_cells)
        qf = flow["q_face"].numpy()
        np.add.at(div, geo.fi.numpy(), qf)
        np.add.at(div, geo.fj.numpy(), -qf)
        div[geo.inlet_cells.numpy()] -= q_v / len(geo.inlet_cells)
        div[geo.outlet_cells.numpy()] += flow["q_out"].numpy()
        assert np.abs(div).max() / q_v < 1e-9

    def test_energy(self, small, rand_field):
        """定常収支: Σ発熱 = 出口エンタルピー流束、出口温度 = ΣQ/(c_p·ṁ)."""
        cfg, geo = small
        with torch.no_grad():
            s = dc.expand(geo, rand_field)
            ht = dc.solve_heat(cfg, geo, s, dc.solve_flow(cfg, geo, s))
        q_in = cfg.q_block_w * len(geo.heater_cells)
        assert abs(float(ht["heat_out"]) - q_in) / q_in < 1e-8


class TestClosures:
    def test_permeability(self, small):
        cfg, _ = small
        k = dc.permeability(cfg, torch.linspace(0.0, 1.0, 21))
        k_plate = cfg.t_chan**2 / 12.0
        assert abs(float(k[0]) - k_plate) / k_plate < 1e-6  # φ=0 で平板
        assert (k[1:] < k[:-1]).all() and float(k[-1]) > 0.0

    def test_interlayer_u(self, small):
        cfg, _ = small
        u = dc.interlayer_u(cfg, torch.linspace(0.0, 1.0, 21))
        assert (u[1:] > u[:-1]).all()  # ピン↑ = 実効 h↑
        assert float(u.max()) < 2.0 * cfg.k_s / cfg.t_base  # ベース伝導の直列上限

    def test_forchheimer_increases_dp(self, small, rand_field):
        cfg, geo = small
        dp_d = float(dc.solve_flow(cfg, geo, dc.expand(geo, rand_field))["dp"])
        cfg_f = replace(cfg, forchheimer=True)
        dp_f = float(dc.solve_flow(cfg_f, geo, dc.expand(geo, rand_field))["dp"])
        assert dp_f > dp_d


class TestOptimize:
    def test_objective_decreases(self, small):
        cfg, geo = small
        dp_ref = dc.dp_reference(cfg, geo)
        rng = np.random.default_rng(0)
        xi0 = torch.tensor(0.05 * rng.standard_normal((geo.by, geo.bx)))
        j0 = float(dc.objective(cfg, geo, xi0, 0.3, dp_ref))
        xi = dc.optimize(cfg, geo, gamma_p=0.3, iters=40, seed=0)
        assert float(dc.objective(cfg, geo, xi, 0.3, dp_ref)) < j0
