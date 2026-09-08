"""nsbm.model / nsbm.train: UNet と学習ループのテスト（torch が無ければ skip）."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nsbm.model import UNet  # noqa: E402


def test_unet_shape_and_backward():
    net = UNet()
    x = torch.randn(2, 8, 72, 48)
    y, logcfl = net(x)
    assert y.shape == (2, 3, 72, 48) and logcfl.shape == (2,)
    (y.square().mean() + logcfl.sum()).backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_unet_cfl_head_starts_at_default_and_is_bounded():
    from nsbm.model import CFL_BASE, CFL_LOG_RANGE

    net = UNet().eval()
    with torch.no_grad():
        _, logcfl = net(torch.randn(3, 8, 72, 48))
    assert torch.allclose(torch.exp(logcfl), torch.full((3,), CFL_BASE))
    lo, hi = CFL_BASE * np.exp(-CFL_LOG_RANGE), CFL_BASE * np.exp(CFL_LOG_RANGE)
    assert lo < 0.01 and 10 < hi < 20


def test_unet_param_count():
    n = sum(p.numel() for p in UNet().parameters())
    assert 1_000_000 <= n <= 6_000_000, n


def test_unet_deterministic_eval():
    torch.manual_seed(0)
    net = UNet().eval()
    x = torch.randn(1, 8, 72, 48)
    with torch.no_grad():
        a, b = net(x)[0], net(x)[0]
    assert torch.equal(a, b)


def test_unet_accepts_numpy_batch_roundtrip():
    net = UNet().eval()
    xb = np.random.default_rng(0).normal(size=(3, 8, 72, 48)).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(xb))[0].numpy()
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
        assert net(torch.from_numpy(samples[0].x[None]))[0].shape == (1, 3, 72, 48)
    hist = (tmp_path / "history.csv").read_text().splitlines()
    assert hist[0].startswith("epoch,train_mse,train_res,val_mse,val_res,val_obj")
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


def test_parse_method():
    from nsbm.evaluate import parse_cfl, parse_method

    assert parse_method("stokes") == ("stokes", 0)
    assert parse_method("unet_n2") == ("unet", 2)
    assert parse_method("knn_n1") == ("knn", 1)
    assert parse_method("unet_n2@pred") == ("unet", 2)
    assert parse_cfl("stokes") is None
    assert parse_cfl("stokes@4") == 4.0
    assert parse_cfl("unet@pred") == "pred"


def test_field_metrics_perfect_and_shifted():
    from nsbm.evaluate import field_metrics

    rng = np.random.default_rng(0)
    y = rng.normal(size=(5, 3, 72, 48)).astype(np.float32)
    m = field_metrics(y, y)
    for c in ("u", "v", "p"):
        assert m[c]["r2_pooled"] == pytest.approx(1.0) and m[c]["max_err_abs_median"] == 0.0
    m2 = field_metrics(y + 0.5, y)  # 一様に +0.5: 最大・最小とも +0.5、R² は落ちる
    assert m2["u"]["max_err_abs_median"] == pytest.approx(0.5, abs=1e-6)
    assert m2["u"]["min_err_signed_mean"] == pytest.approx(0.5, abs=1e-6)
    assert m2["u"]["r2_pooled"] < 1.0


def test_load_model_accepts_checkpoint_without_cfl_head(tmp_path):
    from nsbm.train import load_model

    net = UNet(widths=(8, 16, 32, 64))
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith("cfl_head.")}
    torch.save({"state_dict": sd, "widths": (8, 16, 32, 64)}, tmp_path / "old.pt")
    loaded = load_model(tmp_path / "old.pt")
    with torch.no_grad():
        _, logcfl = loaded(torch.randn(1, 8, 72, 48))
    assert float(torch.exp(logcfl)[0]) == pytest.approx(0.25)
    torch.save(
        {"state_dict": {**sd, "bogus": torch.zeros(1)}, "widths": (8, 16, 32, 64)},
        tmp_path / "bad.pt",
    )
    with pytest.raises(KeyError):
        load_model(tmp_path / "bad.pt")


def test_train_floor_smoke(tmp_path):
    """床モード: 学習前の予測は床（Stokes 解）に一致し、学習後も best.pt に in_ch=11 / floor=True が残る."""
    from nsbm.dataset import Sample
    from nsbm.families import sample_theta
    from nsbm.floor import floor_input, floor_predict, instance_scale
    from nsbm.train import load_model, train

    rng = np.random.default_rng(0)
    samples, floor = [], {}
    for s in range(12):
        th = sample_theta(s)
        x = rng.normal(size=(8, 72, 48)).astype(np.float32)
        x[0] = np.where(rng.random((72, 48)) < 0.2, np.log(1.0 / 100.0), 0.0)  # 2 割を閉塞に
        ys = rng.normal(size=(3, 72, 48)).astype(np.float32)
        samples.append(
            Sample(
                theta=th,
                x=x,
                y=(ys + 0.1 * rng.normal(size=ys.shape)).astype(np.float32),
                n_iter=5,
                converged=True,
                n_gmres_total=1,
                residual_ref=1.0,
                elapsed=0.0,
            )
        )
        floor[th.seed] = ys
    split = {"train": list(range(8)), "val": list(range(8, 10)), "test": list(range(10, 12))}
    # 学習前: ゼロ初期化ヘッド → y = 床（閉塞セルの u,v は 0）
    from nsbm.model import UNet

    net = UNet(in_ch=11, widths=(8, 16, 32, 64)).eval()
    torch.nn.init.zeros_(net.head.weight)
    torch.nn.init.zeros_(net.head.bias)
    x_ext, sc, open_mask = floor_input(samples[0].x, floor[samples[0].theta.seed])
    assert x_ext.shape == (11, 72, 48) and sc.shape == (3,)
    assert np.allclose(sc, instance_scale(floor[samples[0].theta.seed], open_mask))
    with torch.no_grad():
        c, _ = net(torch.from_numpy(x_ext[None]))
        y = floor_predict(
            torch.from_numpy(floor[samples[0].theta.seed][None]),
            torch.from_numpy(sc[None]),
            c,
            torch.from_numpy(open_mask[None, None]),
        ).numpy()[0]
    ys0 = floor[samples[0].theta.seed]
    assert np.allclose(y[2], ys0[2]) and np.allclose(y[0][open_mask], ys0[0][open_mask])
    assert np.all(y[0][~open_mask] == 0.0)
    res = train(
        samples,
        tmp_path,
        split=split,
        epochs=2,
        batch=4,
        widths=(8, 16, 32, 64),
        threads=2,
        floor=floor,
        log=None,
    )
    ck = torch.load(res.best_path, map_location="cpu", weights_only=False)
    assert ck["in_ch"] == 11 and ck["floor"] is True
    net2 = load_model(res.best_path)
    assert net2.in_ch == 11
