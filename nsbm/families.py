"""θ = (h 場ファミリ, u_in, inlet, outlet) のサンプルと、nsb 入力（厚さ場・境界条件）の構築.

[固定] 格子 72×48、領域 0.7×0.4 m、nsb 既定物性。
[θ] `Theta` は JSON 化できる純データ。`build_h(theta)` は決定的（乱数場は `params["noise_seed"]` から再生）。
[ファミリ] 滑らか系（uniform / sin2d / quad / grf）は全域開き、log(h/h0) を ±2 に留める。
  閉塞系（uturn / serpentine / pins / blobs）は h_blocked = h0/100 の閉塞セルを持ち、ポートから内側へ
  最初の開きセルまで廊下を切った上で、inlet と outlet が 4 近傍で連結していることを確認する。
  連結しない θ は棄却して再抽選する（`sample_theta` のループ）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from nsb.core import BC, NSBInput, NSBResult, NSBSettings
from nsb.data import MaskFn, east_span, north_span, south_span, west_span
from nsb.geo import make_uturn_h

NX, NY = 72, 48
LX, LY = 0.7, 0.4
DX, DY = LX / NX, LY / NY
XC = (np.arange(NX) + 0.5) * DX
YC = (np.arange(NY) + 0.5) * DY
X, Y = np.meshgrid(XC, YC, indexing="ij")

FAMILIES = ("uniform", "sin2d", "quad", "grf", "uturn", "serpentine", "pins", "blobs")
BLOCKED_FAMILIES = frozenset({"uturn", "serpentine", "pins", "blobs"})
WALLS = ("west", "east", "north", "south")
WALL_LENGTH = {"west": LY, "east": LY, "north": LX, "south": LX}
H0_RANGE = (0.3e-3, 3.0e-3)
U_IN_RANGE = (0.1, 2.0)
PORT_LENGTH_RANGE = (0.05, 0.15)
BLOCK_RATIO = 100.0  # h_blocked = h0 / BLOCK_RATIO
LOG_CLIP = 2.0  # 滑らか系の log(h/h0) の上下限
MAX_RESAMPLE = 50


@dataclass(frozen=True)
class Port:
    """境界ポート: 壁と、壁に沿った区間 [m]（west/east は y、north/south は x）."""

    wall: str
    s0: float
    s1: float

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass(frozen=True)
class Theta:
    """1 ケースの形状パラメータ（JSON 化可）."""

    seed: int
    family: str
    h0: float
    u_in: float
    inlet: Port
    outlet: Port
    params: dict[str, float] = field(default_factory=dict)

    @property
    def h_blocked(self) -> float:
        return self.h0 / BLOCK_RATIO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Theta:
        d = dict(d)
        d["inlet"] = Port(**d["inlet"])
        d["outlet"] = Port(**d["outlet"])
        d["params"] = {k: float(v) for k, v in d.get("params", {}).items()}
        return Theta(**d)


# ---------------------------------------------------------------- ポート
def port_cells(port: Port) -> tuple[np.ndarray, np.ndarray]:
    """ポートに接する境界セルの (i, j) index 配列（BC マスクと同じ「面中心が区間内」の判定）."""
    if port.wall in ("west", "east"):
        j = np.nonzero((YC > port.s0) & (YC < port.s1))[0]
        i = np.full_like(j, 0 if port.wall == "west" else NX - 1)
    else:
        i = np.nonzero((XC > port.s0) & (XC < port.s1))[0]
        j = np.full_like(i, 0 if port.wall == "south" else NY - 1)
    return i, j


def port_normal(port: Port) -> tuple[float, float]:
    """内向き法線."""
    return {"west": (1.0, 0.0), "east": (-1.0, 0.0), "south": (0.0, 1.0), "north": (0.0, -1.0)}[
        port.wall
    ]


def port_mask(port: Port) -> MaskFn:
    if port.wall == "west":
        return west_span(port.s0, port.s1)
    if port.wall == "east":
        return east_span(port.s0, port.s1, LX)
    if port.wall == "south":
        return south_span(port.s0, port.s1)
    return north_span(port.s0, port.s1, LY)


def _sample_port(rng: np.random.Generator, walls: tuple[str, ...] = WALLS) -> Port:
    wall = str(rng.choice(walls))
    length = float(rng.uniform(*PORT_LENGTH_RANGE))
    s0 = float(rng.uniform(0.0, WALL_LENGTH[wall] - length))
    return Port(wall, s0, s0 + length)


def _ports_overlap(a: Port, b: Port) -> bool:
    return a.wall == b.wall and not (a.s1 <= b.s0 or b.s1 <= a.s0)


def _sample_ports(rng: np.random.Generator, walls: tuple[str, ...]) -> tuple[Port, Port]:
    inlet = _sample_port(rng, walls)
    for _ in range(20):
        outlet = _sample_port(rng, walls)
        if not _ports_overlap(inlet, outlet):
            return inlet, outlet
    # 20 回で引けない（同じ壁に長い区間 2 つ）場合は構成的に置く: inlet の残り側の長い方に outlet を詰める
    wall_len = WALL_LENGTH[inlet.wall]
    length = float(rng.uniform(*PORT_LENGTH_RANGE))
    before, after = inlet.s0, wall_len - inlet.s1
    if max(before, after) < length:
        length = max(before, after)
    if after >= before:
        s0 = float(rng.uniform(inlet.s1, wall_len - length))
    else:
        s0 = float(rng.uniform(0.0, inlet.s0 - length))
    return inlet, Port(inlet.wall, s0, s0 + length)


# ---------------------------------------------------------------- θ のサンプル
def _sample_params(family: str, rng: np.random.Generator) -> dict[str, float]:
    if family == "uniform":
        return {}
    if family == "sin2d":
        return {
            "a": float(rng.uniform(0.2, 1.0)),
            "kx": float(rng.uniform(0.5, 4.0)),
            "ky": float(rng.uniform(0.5, 4.0)),
            "phi": float(rng.uniform(0.0, 2 * np.pi)),
            "psi": float(rng.uniform(0.0, 2 * np.pi)),
        }
    if family == "quad":
        return {
            "a": float(rng.uniform(-1.0, 1.0)),
            "s": float(rng.choice([-1.0, 1.0])),
            "x0": float(rng.uniform(0.0, LX)),
            "y0": float(rng.uniform(0.0, LY)),
        }
    if family == "grf":
        return {
            "corr": float(rng.uniform(0.03, 0.2)),
            "sigma": float(rng.uniform(0.3, 1.0)),
            "noise_seed": float(rng.integers(0, 2**31 - 1)),
        }
    if family == "uturn":
        return {"width": float(rng.uniform(0.05, 0.2))}
    if family == "serpentine":
        n = int(rng.integers(2, 6))
        return {
            "n": float(n),
            "hc": float(rng.uniform(0.03, min(0.1, 0.8 * LY / n))),
            "wc": float(rng.uniform(0.05, 0.1)),
            "margin": float(rng.uniform(0.03, 0.08)),
        }
    if family == "pins":
        return {
            "pitch": float(rng.uniform(0.05, 0.15)),
            "d_ratio": float(rng.uniform(0.3, 0.6)),
            "hex": float(rng.integers(0, 2)),
            "ox": float(rng.uniform(0.0, 1.0)),
            "oy": float(rng.uniform(0.0, 1.0)),
        }
    if family == "blobs":
        return {
            "corr": float(rng.uniform(0.03, 0.15)),
            "open_frac": float(rng.uniform(0.4, 0.8)),
            "noise_seed": float(rng.integers(0, 2**31 - 1)),
        }
    raise ValueError(f"unknown family: {family}")


def sample_theta(seed: int, families: tuple[str, ...] | list[str] = FAMILIES) -> Theta:
    """seed から θ を 1 つ引く（閉塞系は inlet-outlet が連結するまで再抽選、決定的）."""
    for attempt in range(MAX_RESAMPLE):
        rng = np.random.default_rng([int(seed), attempt])
        family = str(rng.choice(list(families)))
        h0 = float(np.exp(rng.uniform(*np.log(H0_RANGE))))
        u_in = float(np.exp(rng.uniform(*np.log(U_IN_RANGE))))
        walls: tuple[str, ...] = ("west",) if family == "uturn" else WALLS
        inlet, outlet = _sample_ports(rng, walls)
        theta = Theta(int(seed), family, h0, u_in, inlet, outlet, _sample_params(family, rng))
        if family not in BLOCKED_FAMILIES:
            return theta
        if connected(build_h(theta), inlet, outlet, theta.h_blocked):
            return theta
    raise RuntimeError(f"seed={seed}: 連結な閉塞形状を {MAX_RESAMPLE} 回で引けませんでした")


# ---------------------------------------------------------------- h 場
def _smooth(theta: Theta, log_ratio: np.ndarray) -> np.ndarray:
    return theta.h0 * np.exp(np.clip(log_ratio, -LOG_CLIP, LOG_CLIP))


def _h_uniform(theta: Theta) -> np.ndarray:
    return np.full((NX, NY), theta.h0)


def _h_sin2d(theta: Theta) -> np.ndarray:
    p = theta.params
    g = np.sin(2 * np.pi * p["kx"] * X / LX + p["phi"]) * np.sin(
        2 * np.pi * p["ky"] * Y / LY + p["psi"]
    )
    return _smooth(theta, p["a"] * g)


def _h_quad(theta: Theta) -> np.ndarray:
    p = theta.params
    g = ((X - p["x0"]) / LX) ** 2 + p["s"] * ((Y - p["y0"]) / LY) ** 2
    return _smooth(theta, p["a"] * g)


def _gaussian_field(corr: float, noise_seed: float) -> np.ndarray:
    """相関長 corr [m] のガウス平滑白色雑音、分散 1."""
    rng = np.random.default_rng(int(noise_seed))
    g = ndimage.gaussian_filter(rng.standard_normal((NX, NY)), sigma=(corr / DX, corr / DY))
    return (g - g.mean()) / max(g.std(), 1e-12)


def _h_grf(theta: Theta) -> np.ndarray:
    p = theta.params
    return _smooth(theta, p["sigma"] * _gaussian_field(p["corr"], p["noise_seed"]))


def _h_uturn(theta: Theta) -> np.ndarray:
    return make_uturn_h(
        NX,
        NY,
        h_channel=theta.h0,
        h_blocked=theta.h_blocked,
        width=theta.params["width"],
        inlet_y=(theta.inlet.s0, theta.inlet.s1),
        outlet_y=(theta.outlet.s0, theta.outlet.s1),
    )


def _h_serpentine(theta: Theta) -> np.ndarray:
    """n 本の水平流路を端で交互に繋いだ蛇行流路."""
    p = theta.params
    n, hc, wc, m = int(p["n"]), p["hc"], p["wc"], p["margin"]
    open_ = np.zeros((NX, NY), dtype=bool)
    yk = [LY * (k + 0.5) / n for k in range(n)]
    for k, y in enumerate(yk):
        open_ |= (X > m) & (X < LX - m) & (np.abs(Y - y) < hc / 2)
        if k + 1 < n:
            x_lo, x_hi = (LX - m - wc, LX - m) if k % 2 == 0 else (m, m + wc)
            open_ |= (X > x_lo) & (X < x_hi) & (Y > y) & (Y < yk[k + 1])
    return np.where(open_, theta.h0, theta.h_blocked)


def _h_pins(theta: Theta) -> np.ndarray:
    """円柱ピンフィン配列（正方 or 六方格子）で閉塞."""
    p = theta.params
    pitch, r = p["pitch"], 0.5 * p["d_ratio"] * p["pitch"]
    hexa = p["hex"] > 0.5
    ox, oy = p["ox"] * pitch, p["oy"] * pitch
    blocked = np.zeros((NX, NY), dtype=bool)
    row_pitch = pitch * (np.sqrt(3) / 2 if hexa else 1.0)
    ny_rows = int(LY / row_pitch) + 2
    nx_cols = int(LX / pitch) + 2
    for jr in range(-1, ny_rows):
        cy = oy + jr * row_pitch
        shift = 0.5 * pitch if (hexa and jr % 2) else 0.0
        for ic in range(-1, nx_cols):
            cx = ox + ic * pitch + shift
            blocked |= (X - cx) ** 2 + (Y - cy) ** 2 < r**2
    return np.where(blocked, theta.h_blocked, theta.h0)


def _h_blobs(theta: Theta) -> np.ndarray:
    """ガウス場を分位点で 2 値化した島状閉塞（開口率 open_frac）."""
    p = theta.params
    g = _gaussian_field(p["corr"], p["noise_seed"])
    open_ = g < np.quantile(g, p["open_frac"])
    return np.where(open_, theta.h0, theta.h_blocked)


_BUILDERS = {
    "uniform": _h_uniform,
    "sin2d": _h_sin2d,
    "quad": _h_quad,
    "grf": _h_grf,
    "uturn": _h_uturn,
    "serpentine": _h_serpentine,
    "pins": _h_pins,
    "blobs": _h_blobs,
}


def _open_ports(h: np.ndarray, theta: Theta) -> np.ndarray:
    """各ポート境界セルから内向きに、最初の開きセル（最低 2 セル）まで廊下を h0 で開ける."""
    h = h.copy()
    thr = theta.h_blocked * 1.5
    for port in (theta.inlet, theta.outlet):
        ii, jj = port_cells(port)
        nxv, nyv = port_normal(port)
        di, dj = int(nxv), int(nyv)
        for i0, j0 in zip(ii, jj, strict=True):
            i, j, step = int(i0), int(j0), 0
            while 0 <= i < NX and 0 <= j < NY:
                if step >= 2 and h[i, j] > thr:
                    break
                h[i, j] = theta.h0
                i, j, step = i + di, j + dj, step + 1
    return h


def build_h(theta: Theta) -> np.ndarray:
    """θ から厚さ場 (72, 48) を組む（決定的）."""
    h = _BUILDERS[theta.family](theta)
    if theta.family in BLOCKED_FAMILIES:
        h = _open_ports(h, theta)
    return h


def connected(h: np.ndarray, inlet: Port, outlet: Port, h_blocked: float) -> bool:
    """inlet セルと outlet セルが開きセル（h > 1.5 h_blocked）の 4 近傍連結成分で繋がっているか."""
    labels, _ = ndimage.label(h > h_blocked * 1.5)
    li = set(labels[port_cells(inlet)].tolist()) - {0}
    lo = set(labels[port_cells(outlet)].tolist()) - {0}
    return bool(li & lo)


# ---------------------------------------------------------------- nsb 入力
def build_bc(theta: Theta) -> BC:
    return BC(
        patches=(
            BC.velocity_inlet(port_mask(theta.inlet), theta.u_in),
            BC.pressure_outlet(port_mask(theta.outlet)),
        )
    )


def build_input(
    theta: Theta,
    settings: NSBSettings | None = None,
    init: NSBResult | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> NSBInput:
    """θ → NSBInput。init に NSBResult か (u0, v0, p0) を渡すと初期場にする."""
    kw: dict[str, np.ndarray] = {}
    if isinstance(init, NSBResult):
        kw = {"u0": init.u, "v0": init.v, "p0": init.p}
    elif init is not None:
        kw = {"u0": init[0], "v0": init[1], "p0": init[2]}
    return NSBInput(
        nx=NX,
        ny=NY,
        lx=LX,
        ly=LY,
        h=build_h(theta),
        bc=build_bc(theta),
        settings=settings or NSBSettings(),
        **kw,
    )
