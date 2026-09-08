"""床 + 補正: Stokes–Brinkman 解 y_S を床にして、UNet は補正量 c だけを出す（y = y_S + s·c）.

[なぜ] 場→場の教師あり UNet（unet-a）は R² 0.9 で飽和し、圧力はサンプル別 R² が崩れた（変動の小さいケースで
  大域参照 p_ref のスケールが合わない）。messi の 1 か月の実測（2026-09-09 の助言）: 安い物理解を床にして差分だけ学ぶ、
  インスタンスごとに rms(床解) でスケールする、壁セルは出力を硬く 0・損失は開きセルだけ、が効いた 3 点。
[スケール] s_u = rms(|u_S|)（開きセル）、s_p = std(p_S)（開きセル、ゲージを除く）。正規化単位（u/u_in, p/p_ref）の中で取る。
[入力] 8 ch + (u_S/s_u, v_S/s_u, p_S/s_p) の 3 ch = 11 ch。
[出力] c (3 ch)。予測 y = y_S + s·c、閉塞セルの u, v は 0。損失 = 開きセルの ((y − y_true)/s)² の平均（= c の MSE）。
[出発点] ヘッドをゼロ初期化するので学習前は y = y_S（床の精度から始まり、悪くなる方向には学びにくい）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from nsbm.features import blocked_mask_from_x

FLOOR_CH = 3


def load_stokes(path: Path) -> dict[int, np.ndarray]:
    z = np.load(path)
    seeds, ys = z["seed"], z["ys"]
    return {int(s): ys[k] for k, s in enumerate(seeds)}


def instance_scale(ys: np.ndarray, open_mask: np.ndarray) -> np.ndarray:
    """(s_u, s_u, s_p): 床解の開きセルでの rms 速さと圧力の標準偏差（下限 1e-6）."""
    if open_mask.sum() == 0:
        open_mask = np.ones_like(open_mask, dtype=bool)
    u, v, p = ys[0][open_mask], ys[1][open_mask], ys[2][open_mask]
    s_u = float(np.sqrt(np.mean(u**2 + v**2)))
    s_p = float(np.std(p))
    s_u, s_p = max(s_u, 1e-6), max(s_p, 1e-6)
    return np.array([s_u, s_u, s_p], dtype=np.float32)


def floor_input(x: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(11ch 入力, スケール (3,), 開きマスク (H,W)) を返す."""
    open_mask = ~blocked_mask_from_x(x)
    s = instance_scale(ys, open_mask)
    x_ext = np.concatenate([x, (ys / s[:, None, None]).astype(np.float32)], axis=0)
    return x_ext, s, open_mask


def floor_predict(ys: Tensor, s: Tensor, c: Tensor, open_mask: Tensor) -> Tensor:
    """y = y_S + s·c、閉塞セルの u, v は 0（torch、バッチ）. ys (B,3,H,W), s (B,3), c (B,3,H,W), open_mask (B,1,H,W)."""
    y = ys + s[:, :, None, None] * c
    keep = torch.cat([open_mask, open_mask, torch.ones_like(open_mask)], dim=1).to(y.dtype)
    return y * keep


def floor_loss(y_pred: Tensor, y_true: Tensor, s: Tensor, open_mask: Tensor) -> Tensor:
    """開きセルだけの ((y − y_true)/s)² の平均（3 チャネル等重み）."""
    d = (y_pred - y_true) / s[:, :, None, None]
    m = open_mask.to(d.dtype).expand_as(d)
    return (d**2 * m).sum() / m.sum().clamp_min(1.0)
