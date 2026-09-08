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
