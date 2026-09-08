"""nsbm.residual_loss: 展開ループが nsb の残差履歴を再現するか、随伴勾配が FD と合うか（直接解モード）."""

from __future__ import annotations

import numpy as np
import pytest

from nsb.core import NSBSettings
from nsb.solver import solve_steady
from nsbm.families import build_input, sample_theta
from nsbm.features import p_ref
from nsbm.residual_loss import _w, residual_loss, unroll


@pytest.fixture(scope="module")
def case():
    """uniform ファミリの小さな θ: Stokes 場を少し崩したものを初期場にする."""
    theta = sample_theta(11, ("uniform",))
    st = NSBSettings(newton_max_iter=3)
    res0 = solve_steady(build_input(theta, st), log=None)
    rng = np.random.default_rng(0)
    u = res0.u * (1.0 + 0.3 * rng.standard_normal(res0.u.shape))
    v = res0.v * (1.0 + 0.3 * rng.standard_normal(res0.v.shape))
    p = res0.p * (1.0 + 0.1 * rng.standard_normal(res0.p.shape))
    x0 = np.concatenate([u.ravel(), v.ravel(), p.ravel()])
    return theta, st, (u, v, p), x0, res0.residual_ref


def test_w_transforms():
    assert _w(2.0, "ratio") == (2.0, 1.0)
    val, d = _w(1.0, "log")
    assert val == pytest.approx(0.0, abs=1e-9) and d == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError):
        _w(1.0, "bogus")


@pytest.mark.slow
def test_unroll_matches_solve_steady_history(case):
    theta, st, init, x0, r_ref = case
    inp = build_input(theta, st, init)
    res = solve_steady(inp, log=None)
    ref = np.array(res.steady_residual_history) / res.residual_ref
    rec = unroll(inp, x0, st.cfl_init, res.residual_ref, steps=3, need_jacobians=False)
    got = np.array(rec.rhos)
    assert rec.failure == "" and len(got) == len(ref)
    # 同じ JFNK 経路だが、GMRES 許容 1e-3 を満たす δ は前処理の状態（nsb は Stokes 解で使った前処理を
    # 引き継ぐ）で悪条件方向に差が出、悪い初期場からの 1 歩は非線形残差がその差に敏感（実測 13%）。
    # 実機の損失面が持つ固有のノイズなので、傾向の一致だけを見る
    assert np.allclose(got, ref, rtol=0.25), (got, ref)


@pytest.mark.slow
def test_gradient_k0_matches_fd(case):
    theta, st, init, x0, r_ref = case
    inp = build_input(theta, st, init)
    out = residual_loss(inp, x0, st.cfl_init, r_ref, steps=0)
    g = out["grad_x0"]
    n = x0.size // 3
    sc = np.concatenate([np.full(2 * n, theta.u_in), np.full(n, p_ref(theta))])
    rng = np.random.default_rng(1)
    d = rng.standard_normal(x0.size) * sc
    eps = 1e-6
    lp = residual_loss(inp, x0 + eps * d, st.cfl_init, r_ref, steps=0, with_grad=False)["loss"]
    lm = residual_loss(inp, x0 - eps * d, st.cfl_init, r_ref, steps=0, with_grad=False)["loss"]
    assert (lp - lm) / (2 * eps) == pytest.approx(float(np.dot(g, d)), rel=1e-3)
    assert out["grad_logcfl"] == 0.0  # K=0 では cfl は効かない


@pytest.mark.slow
def test_gradient_k2_direct_solve_descends_and_cfl_matches_fd(case):
    """凍結ヤコビアン随伴（直接解モード）: log cfl の勾配は FD と数 % で一致、x0 の勾配は降下方向."""
    theta, st, init, x0, r_ref = case
    inp = build_input(theta, st, init)
    kw = dict(steps=2, linear="direct")
    out = residual_loss(inp, x0, st.cfl_init, r_ref, **kw)
    h = 1e-3
    lp = residual_loss(inp, x0, st.cfl_init * np.exp(h), r_ref, with_grad=False, **kw)["loss"]
    lm = residual_loss(inp, x0, st.cfl_init * np.exp(-h), r_ref, with_grad=False, **kw)["loss"]
    assert (lp - lm) / (2 * h) == pytest.approx(out["grad_logcfl"], rel=0.05)
    g = out["grad_x0"]
    n = x0.size // 3
    sc = np.concatenate([np.full(2 * n, theta.u_in), np.full(n, p_ref(theta))])
    gd = g * sc**2
    gd /= np.linalg.norm(gd / sc)
    eps = 1e-6
    lp = residual_loss(inp, x0 + eps * gd, st.cfl_init, r_ref, with_grad=False, **kw)["loss"]
    lm = residual_loss(inp, x0 - eps * gd, st.cfl_init, r_ref, with_grad=False, **kw)["loss"]
    fd = (lp - lm) / (2 * eps)
    adj = float(np.dot(g, gd))
    assert fd > 0 and adj > 0 and 0.5 < fd / adj < 2.0, (fd, adj)


def test_straight_through_gradient_equals_worker_grads():
    torch = pytest.importorskip("torch")
    from nsbm.residual_loss import straight_through

    f = torch.zeros(2, 3, 4, 4, requires_grad=True)
    c = torch.zeros(2, requires_grad=True)
    outs = [
        {"loss": 1.0, "grad_x0": np.ones((3, 4, 4), np.float32), "grad_logcfl": 2.0},
        {"loss": float("nan"), "grad_x0": np.zeros((3, 4, 4), np.float32), "grad_logcfl": 0.0},
    ]
    loss, n_ok = straight_through(f, c, outs)
    assert n_ok == 1 and loss.item() == 1.0
    loss.backward()
    assert torch.allclose(f.grad[0], torch.ones(3, 4, 4)) and float(f.grad[1].abs().sum()) == 0.0
    assert float(c.grad[0]) == 2.0 and float(c.grad[1]) == 0.0
