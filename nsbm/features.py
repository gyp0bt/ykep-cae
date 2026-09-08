"""入力画像（8 チャネル）と正解場の正規化 / 逆変換.

[入力] 0: log(h/h0)、1: log(h0/1e-3)、2: log(u_in)（1, 2 は定数面）、3-4: inlet 境界セルの u_in·n（内向き法線）、
  5: outlet 境界セルの 1、6-7: x/LX, y/LY。
[出力] u/u_in, v/u_in, p/p_ref。p_ref = 12 μ u_in LX / h0² + ρ u_in²（Brinkman 支配と慣性支配の圧力スケールの和）。
  スケールは θ だけから決まるので、推論結果を `denormalize_y` で物理量に戻せる。
"""

from __future__ import annotations

import numpy as np

from nsbm.families import LX, LY, NX, NY, Theta, X, Y, port_cells, port_normal

IN_CH = 8
OUT_CH = 3
RHO, MU = 1000.0, 1.0e-3  # nsb 既定物性
H_UNIT = 1.0e-3


def p_ref(theta: Theta) -> float:
    return 12.0 * MU * theta.u_in * LX / theta.h0**2 + RHO * theta.u_in**2


def make_x(theta: Theta, h: np.ndarray) -> np.ndarray:
    x = np.zeros((IN_CH, NX, NY), dtype=np.float32)
    x[0] = np.log(h / theta.h0)
    x[1] = np.log(theta.h0 / H_UNIT)
    x[2] = np.log(theta.u_in)
    i, j = port_cells(theta.inlet)
    nxv, nyv = port_normal(theta.inlet)
    x[3, i, j] = theta.u_in * nxv
    x[4, i, j] = theta.u_in * nyv
    io, jo = port_cells(theta.outlet)
    x[5, io, jo] = 1.0
    x[6] = X / LX
    x[7] = Y / LY
    return x


def normalize_y(theta: Theta, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.stack([u / theta.u_in, v / theta.u_in, p / p_ref(theta)]).astype(np.float32)


def denormalize_y(theta: Theta, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    return y[0] * theta.u_in, y[1] * theta.u_in, y[2] * p_ref(theta)
