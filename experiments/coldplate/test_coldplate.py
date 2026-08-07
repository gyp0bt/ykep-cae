"""coldplate.py v2 のテスト.

実験コードのため CI (testpaths=["tests"]) には含めない。手動実行:

    cd experiments/coldplate && python -m pytest test_coldplate.py -v

小規模問題 (10x12 格子, 6 ブロック) で v2 の P0 修正を検証する。
"""

from __future__ import annotations

import coldplate as cp
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def small_problem() -> cp.Problem:
    return cp.make_problem(
        n_cols=2, n_rows=3, margin_left=3, margin_right=2, margin_y=2, ntu_coef=1e-2
    )


@pytest.fixture(scope="module")
def pipeline_result(small_problem: cp.Problem):
    cfg = cp.Config(vol_frac=0.25)
    x_raw, x_fin, mask = cp.solve_pipeline(
        small_problem, cfg, log_p_max=5.0, maxiter_stage=200, maxiter_reopt=200, verbose=False
    )
    return cfg, x_raw, x_fin, mask


class TestColdplateAPI:
    """プログラムテスト: API・勾配・収束."""

    def test_problem_shape(self, small_problem: cp.Problem) -> None:
        gr = small_problem.graph
        assert gr.n_nodes == gr.nx * gr.ny
        assert gr.n_edges == int(gr.tail.shape[0])
        assert len(small_problem.block_edges) == 6
        for e in small_problem.block_edges:
            assert len(e) == 4  # 発熱体 1 セル = 4 辺

    def test_gradient_full(self, small_problem: cp.Problem) -> None:
        cfg = cp.Config()
        assert cp.check_gradient(small_problem, cfg) < 1e-4

    def test_gradient_masked(self, small_problem: cp.Problem, pipeline_result) -> None:
        cfg, _, _, mask = pipeline_result
        assert cp.check_gradient(small_problem, cfg, mask=mask) < 1e-4

    def test_pipeline_improves_and_converges(
        self, small_problem: cp.Problem, pipeline_result
    ) -> None:
        cfg, _, x_fin, mask = pipeline_result
        r = cp.report(small_problem, cfg, x_fin, mask)
        assert np.isfinite(r["cooling"])
        assert r["blocks_covered"] == 6  # 全発熱体が流路に到達
        assert r["grey"] == 0.0  # rho 凍結により grey は構造的にゼロ

    def test_decode_mask_exact_binary(self, small_problem: cp.Problem, pipeline_result) -> None:
        cfg, _, x_fin, mask = pipeline_result
        with torch.no_grad():
            st = cp.state(small_problem, cfg, torch.tensor(x_fin), mask)
        rho = st["rho"].numpy()
        assert set(np.unique(rho)).issubset({0.0, 1.0})
        # 凍結ポート (port_*_on=False) の softmax 重みは厳密ゼロ
        yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
        assert np.all(yi[~mask.port_in_on] == 0.0)
        assert np.all(yo[~mask.port_out_on] == 0.0)


@pytest.fixture(scope="module")
def single_port_result(small_problem: cp.Problem):
    cfg = cp.Config(vol_frac=0.25)
    _, x_fin, mask = cp.solve_pipeline(
        small_problem,
        cfg,
        log_p_max=5.0,
        maxiter_stage=200,
        maxiter_reopt=200,
        single_ports=True,
        verbose=False,
    )
    return cfg, x_fin, mask


class TestColdplateSinglePortAPI:
    """single-port モード: 物理ポート in/out 各 1 箇所の強制."""

    def test_exactly_one_port_each(self, small_problem: cp.Problem, single_port_result) -> None:
        cfg, x_fin, mask = single_port_result
        assert int(mask.port_in_on.sum()) == 1
        assert int(mask.port_out_on.sum()) == 1
        with torch.no_grad():
            st = cp.state(small_problem, cfg, torch.tensor(x_fin), mask)
        yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
        # 1 要素 softmax は厳密に 1
        assert yi.max() == 1.0
        assert yo.max() == 1.0
        b = st["b"].numpy()
        assert int((b > 0).sum()) == 1
        assert int((b < 0).sum()) == 1

    def test_single_port_leak_free_connected(
        self, small_problem: cp.Problem, single_port_result
    ) -> None:
        cfg, x_fin, mask = single_port_result
        mc = cp.mass_check(small_problem, cfg, x_fin, mask)
        assert mc["leak_inactive"] == 0.0
        assert mc["resid_max"] < 1e-8
        cc = cp.connectivity_check(small_problem, cfg, x_fin, mask=mask)
        assert cc["connected"]


class TestColdplatePhysics:
    """物理テスト: 質量保存・漏れ流れ・連結性・エネルギー整合."""

    def test_mass_conservation_nodes(self, small_problem: cp.Problem, pipeline_result) -> None:
        """節点質量保存: div(q) = b (アクティブ部分グラフの厳密解)."""
        cfg, _, x_fin, mask = pipeline_result
        mc = cp.mass_check(small_problem, cfg, x_fin, mask)
        assert mc["resid_max"] < 1e-8 * small_problem.q_total

    def test_zero_flow_on_inactive_edges(self, small_problem: cp.Problem, pipeline_result) -> None:
        """P0 の核心: 存在しない (凍結) 辺の流量は厳密にゼロ."""
        cfg, x_raw, x_fin, mask = pipeline_result
        # v1 相当の素の SIMP 解は rho<0.5 の辺に漏れ流れを持つ
        mc_raw = cp.mass_check(small_problem, cfg, x_raw)
        assert mc_raw["leak_inactive"] > 1e-3  # 漏れが実在した (回帰確認)
        # v2 の刈り込み後は厳密ゼロ
        mc = cp.mass_check(small_problem, cfg, x_fin, mask)
        assert mc["leak_inactive"] == 0.0

    def test_connectivity(self, small_problem: cp.Problem, pipeline_result) -> None:
        cfg, _, x_fin, mask = pipeline_result
        cc = cp.connectivity_check(small_problem, cfg, x_fin, mask=mask)
        assert cc["connected"]
        assert cc["n_src_components"] == 1
        assert cc["n_snk_components"] == 1

    def test_port_flow_balance(self, small_problem: cp.Problem, pipeline_result) -> None:
        """Σb = 0 (流入 = 流出) がポート softmax により厳密に成立."""
        cfg, _, x_fin, mask = pipeline_result
        with torch.no_grad():
            st = cp.state(small_problem, cfg, torch.tensor(x_fin), mask)
        assert abs(float(st["b"].sum())) < 1e-12

    def test_energy_consistency(self, small_problem: cp.Problem, pipeline_result) -> None:
        """鮮度収支: 流入鮮度 - 流出鮮度 = 全辺の除熱量 (輸送方程式の整合)."""
        cfg, _, x_fin, mask = pipeline_result
        with torch.no_grad():
            st = cp.state(small_problem, cfg, torch.tensor(x_fin), mask)
        fresh_in = float(torch.relu(st["b"]).sum())  # φ=1 で流入
        fresh_out = float((st["phi"] * torch.relu(-st["b"])).sum())
        heat_total = float(st["q_heat"].sum())
        assert fresh_in - fresh_out == pytest.approx(heat_total, rel=1e-6)

    def test_freshness_bounds(self, small_problem: cp.Problem, pipeline_result) -> None:
        cfg, _, x_fin, mask = pipeline_result
        with torch.no_grad():
            st = cp.state(small_problem, cfg, torch.tensor(x_fin), mask)
        phi = st["phi"].numpy()
        assert phi.min() >= -1e-12
        assert phi.max() <= 1.0 + 1e-9

    def test_uniformity_reported(self, small_problem: cp.Problem, pipeline_result) -> None:
        """均一性指標が報告され、worst ブロックが平均の 1/2 を下回らない."""
        cfg, _, x_fin, mask = pipeline_result
        r = cp.report(small_problem, cfg, x_fin, mask)
        assert 0.0 <= r["block_cv"] < 1.0
        assert r["block_min_over_mean"] > 0.5
