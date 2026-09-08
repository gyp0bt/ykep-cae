"""nsbm.model / nsbm.train: UNet と学習ループのテスト（torch が無ければ skip）."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nsbm.model import UNet  # noqa: E402


def test_unet_shape_and_backward():
    net = UNet()
    x = torch.randn(2, 8, 72, 48)
    y = net(x)
    assert y.shape == (2, 3, 72, 48)
    y.square().mean().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_unet_param_count():
    n = sum(p.numel() for p in UNet().parameters())
    assert 1_000_000 <= n <= 6_000_000, n


def test_unet_deterministic_eval():
    torch.manual_seed(0)
    net = UNet().eval()
    x = torch.randn(1, 8, 72, 48)
    with torch.no_grad():
        a, b = net(x), net(x)
    assert torch.equal(a, b)


def test_unet_accepts_numpy_batch_roundtrip():
    net = UNet().eval()
    xb = np.random.default_rng(0).normal(size=(3, 8, 72, 48)).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(xb)).numpy()
    assert out.shape == (3, 3, 72, 48) and out.dtype == np.float32


def test_train_smoke(tmp_path):
    from nsbm.dataset import Sample
    from nsbm.families import sample_theta
    from nsbm.train import load_model, split_by_family, train

    rng = np.random.default_rng(0)
    samples = [
        Sample(
            theta=sample_theta(s),
            x=rng.normal(size=(8, 72, 48)).astype(np.float32),
            y=rng.normal(size=(3, 72, 48)).astype(np.float32),
            n_iter=5,
            converged=True,
            n_gmres_total=1,
            residual_ref=1.0,
            elapsed=0.0,
        )
        for s in range(16)
    ]
    split = {"train": list(range(10)), "val": list(range(10, 13)), "test": list(range(13, 16))}
    res = train(
        samples,
        tmp_path,
        split=split,
        epochs=2,
        batch=4,
        widths=(8, 16, 32, 64),
        threads=2,
        log=None,
    )
    assert (
        res.best_path.exists()
        and (tmp_path / "history.csv").exists()
        and (tmp_path / "split.json").exists()
    )
    net = load_model(res.best_path)
    with torch.no_grad():
        assert net(torch.from_numpy(samples[0].x[None])).shape == (1, 3, 72, 48)
    sp = split_by_family(samples, 0)
    assert set(sp) == {"train", "val", "test"} and sum(len(v) for v in sp.values()) == 16


def test_knn_predict_recovers_exact_neighbor():
    from nsbm.evaluate import knn_predict

    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 8, 72, 48)).astype(np.float32)
    y = rng.normal(size=(6, 3, 72, 48)).astype(np.float32)
    out = knn_predict(x, y, x[2], k=1)
    assert np.allclose(out, y[2])


def test_summarize_counts():
    from nsbm.evaluate import summarize

    rows = [
        {
            "family": "uniform",
            "stokes_n_iter": 10,
            "stokes_converged": True,
            "stokes_r0_ratio": 1.0,
            "knn_n_iter": 8,
            "knn_converged": True,
            "knn_r0_ratio": 0.2,
            "unet_n_iter": 6,
            "unet_converged": True,
            "unet_r0_ratio": 0.1,
        },
        {
            "family": "pins",
            "stokes_n_iter": 10,
            "stokes_converged": True,
            "stokes_r0_ratio": 1.0,
            "knn_n_iter": 12,
            "knn_converged": True,
            "knn_r0_ratio": 0.5,
            "unet_n_iter": 12,
            "unet_converged": True,
            "unet_r0_ratio": 0.4,
        },
    ]
    s = summarize(rows)
    assert s["all"]["n"] == 2 and s["all"]["unet"]["wins"] == 1 and s["all"]["unet"]["losses"] == 1
    assert s["all"]["unet_vs_knn"] == {"wins": 1, "ties": 1, "losses": 0}
    assert set(s["by_family"]) == {"uniform", "pins"}


def test_divergence_loss_zero_for_uniform_flow_and_positive_for_source():
    from nsbm.train import divergence_loss

    x = torch.zeros(1, 8, 72, 48)  # h = h0 一様
    y = torch.zeros(1, 3, 72, 48)
    y[:, 0] = 1.0  # 一様流 u = u_in → 発散 0
    assert float(divergence_loss(x, y)) == pytest.approx(0.0, abs=1e-12)
    y2 = torch.zeros(1, 3, 72, 48)
    y2[:, 0, 30:40, :] = 1.0  # 途中で始まる流れ → 発散あり
    assert float(divergence_loss(x, y2)) > 0
