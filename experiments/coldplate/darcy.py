"""擬3D ダルシー則モデルによる冷却流路の 2 目的最適化 (圧損 × 温度均一化).

ネットワークモデル (coldplate.py) の層流一定 Nu 近似では「濡れ面積が唯一のレバー」
になってしまい流速分布の寄与が見えない。そこで近似式を捨て、同一形状を

  - 流れ: Hele-Shaw 深さ平均のダルシー則 (K_open = t_chan²/12)。
    設計変数はセルごとではなく 5mm 設計ブロックごとのダルシー係数
    (セル単位だと枝状の奇妙な最適解が出るため)。0/1 位相ではなく
    連続な透過率場 K ∈ [K_min, K_open] を対数スケールで直接設計する
    (物理的にはブロックごとのピンフィン密度グレーディングに相当)。
    細分セル (pitch/refine) の FVM で圧力ポアソン ∇·(K t/μ ∇p) = 0 を解く。
  - 熱: z 方向 2 層の擬 3D。ベース板 (伝導のみ, 発熱入力) と流路層
    (ダルシー面流束による風上移流 + k(s) 面内伝導) を、半厚み伝導の
    直列コンダクタンスで層間結合。Nu 相関は使わない。

で解き直す。目的は 2 目的の重み付き和:

  J = J_T + γ·ΔP/ΔP_ref,   J_T = LSE_β(T_blocks)/T_ref + μ_tvar·var(T_blocks)/T_ref²

γ をスイープしてパレートフロントを得る。線形系はスパース LU (splu) を
adjoint 付き autograd.Function でラップし、設計勾配を厳密に流す。

README.md 参照。単発実験 (Process クラス統合なし)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu

torch.set_default_dtype(torch.float64)


# ==========================================================================
# 設定
# ==========================================================================
@dataclass
class DarcyConfig:
    # 幾何 (coldplate.make_problem と同じ設計ブロック配置)
    n_cols: int = 2
    n_rows: int = 4
    block_pitch: int = 3
    margin_left: int = 4
    margin_right: int = 3
    margin_y: int = 3
    refine: int = 4  # 設計ブロック (5mm) あたりの細分セル数
    cell_pitch: float = 5e-3  # 設計ブロックピッチ [m]
    t_base: float = 3e-3  # ベース板厚 [m]
    t_chan: float = 2e-3  # 流路層深さ [m]
    # 物性 (水 / Al 6061)
    mu: float = 1.0e-3  # 粘性係数 [Pa·s]
    rho_f: float = 998.0  # 密度 [kg/m³]
    cp: float = 4180.0  # 比熱 [J/kgK]
    k_f: float = 0.6  # 水の熱伝導率 [W/mK]
    k_s: float = 167.0  # Al の熱伝導率 [W/mK]
    m_dot: float = 5e-3  # 総質量流量 [kg/s]
    q_block_w: float = 10.0  # 発熱量 / ブロック [W]
    # 設計変数: 固体度 s ∈ [0,1] → K(s) = K_open·k_ratio^s (対数補間)
    k_ratio: float = 1e-5  # K_min/K_open (s=1 でほぼ不透過)
    # ピンフィンモデル (pin_fin=True で有効): s をピン充填率 φ = phi_max·s と解釈し
    # K(φ)=平行平板+Gebart 円柱列の直列、層間 U(φ)=プライム面+フィン増倍で閉じる
    pin_fin: bool = False
    d_pin: float = 1.0e-3  # ピン直径 [m]
    phi_max: float = 0.6  # s=1 でのピン充填率 (正方配列の幾何限界 π/4 未満)
    nu_pin: float = 4.0  # ピン周り Nu (速度非依存の伝導床)
    # Forchheimer 慣性補正 (pin_fin 前提): ∇p = -(μ/K)U - ρβ(φ)|U|U
    forchheimer: bool = False
    c_ergun: float = 1.75  # Ergun 型 β = c_E·φ/((1-φ)³·d) の係数
    picard_max: int = 60  # Picard 反復上限
    picard_tol: float = 1e-12  # dp 相対収束判定
    picard_relax: float = 0.5  # 線形化フラックスの緩和 (固定点の減衰振動対策)
    # 目的
    beta_t: float = 2.0  # ピーク温度 smooth-max の鋭さ [1/K]
    mu_tvar: float = 10.0  # ブロック温度分散罰則 (T_ref² 正規化後)


@dataclass(frozen=True)
class Geo:
    """細分格子と設計ブロックのインデックス一式 (全て構築時に確定)."""

    bx: int  # 設計ブロック数 x
    by: int  # 設計ブロック数 y
    ncx: int  # 細分セル数 x
    ncy: int  # 細分セル数 y
    h: float  # 細分セル寸法 [m]
    fi: torch.Tensor  # 面の両側セル (long, F)
    fj: torch.Tensor
    inlet_cells: torch.Tensor  # 左辺中央の流入セル
    outlet_cells: torch.Tensor  # 右辺中央の流出セル
    heater_cells: list[torch.Tensor] = field(default_factory=list)
    heater_boxes: list[tuple[int, int]] = field(default_factory=list)  # 設計ブロック座標

    @property
    def n_cells(self) -> int:
        return self.ncx * self.ncy


def make_geo(cfg: DarcyConfig) -> Geo:
    span_x = (cfg.n_cols - 1) * cfg.block_pitch + 1
    span_y = (cfg.n_rows - 1) * cfg.block_pitch + 1
    bx = cfg.margin_left + span_x + cfg.margin_right
    by = 2 * cfg.margin_y + span_y
    r = cfg.refine
    ncx, ncy = bx * r, by * r

    # 面 (x 方向 / y 方向) の両側セル id
    ix = np.arange(ncx - 1)
    iy = np.arange(ncy)
    fx_i = (iy[:, None] * ncx + ix[None, :]).ravel()
    ix2 = np.arange(ncx)
    iy2 = np.arange(ncy - 1)
    fy_i = (iy2[:, None] * ncx + ix2[None, :]).ravel()
    fi = np.concatenate([fx_i, fy_i])
    fj = np.concatenate([fx_i + 1, fy_i + ncx])

    # ポート: 左辺 / 右辺の中央 5mm (= refine セル)
    j0 = ncy // 2 - r // 2
    port_j = np.arange(j0, j0 + r)
    inlet = port_j * ncx
    outlet = port_j * ncx + (ncx - 1)

    # 発熱体: 設計ブロック (x0, y0) が 1 ブロック占有
    heater_cells: list[torch.Tensor] = []
    boxes: list[tuple[int, int]] = []
    for c in range(cfg.n_cols):
        x0 = cfg.margin_left + c * cfg.block_pitch
        for row in range(cfg.n_rows):
            y0 = cfg.margin_y + row * cfg.block_pitch
            cx = np.arange(x0 * r, (x0 + 1) * r)
            cy = np.arange(y0 * r, (y0 + 1) * r)
            cells = (cy[:, None] * ncx + cx[None, :]).ravel()
            heater_cells.append(torch.tensor(cells, dtype=torch.long))
            boxes.append((x0, y0))

    return Geo(
        bx=bx,
        by=by,
        ncx=ncx,
        ncy=ncy,
        h=cfg.cell_pitch / r,
        fi=torch.tensor(fi, dtype=torch.long),
        fj=torch.tensor(fj, dtype=torch.long),
        inlet_cells=torch.tensor(inlet, dtype=torch.long),
        outlet_cells=torch.tensor(outlet, dtype=torch.long),
        heater_cells=heater_cells,
        heater_boxes=boxes,
    )


# ==========================================================================
# スパース直接法 (adjoint 付き autograd)
# ==========================================================================
class _SparseSolve(torch.autograd.Function):
    """x = A⁻¹b。A は COO (rows, cols, vals) で重複エントリは加算。

    backward: λ = A⁻ᵀ ḡ,  ∂J/∂vals_k = -λ[row_k]·x[col_k],  ∂J/∂b = λ。
    """

    @staticmethod
    def forward(ctx, vals, b, rows, cols, n):  # noqa: ANN001, ANN205
        a = sp.csc_matrix((vals.detach().numpy(), (rows, cols)), shape=(n, n))
        lu = splu(a)
        x = torch.from_numpy(lu.solve(np.ascontiguousarray(b.detach().numpy())))
        ctx.save_for_backward(x)
        ctx.lu, ctx.rows, ctx.cols = lu, rows, cols
        return x

    @staticmethod
    def backward(ctx, gout):  # noqa: ANN001, ANN205
        (x,) = ctx.saved_tensors
        lam = torch.from_numpy(ctx.lu.solve(np.ascontiguousarray(gout.detach().numpy()), trans="T"))
        gvals = -lam[ctx.rows] * x[ctx.cols]
        return gvals, lam, None, None, None


def sparse_solve(
    vals: torch.Tensor, b: torch.Tensor, rows: np.ndarray, cols: np.ndarray, n: int
) -> torch.Tensor:
    return _SparseSolve.apply(vals, b, rows, cols, n)


# ==========================================================================
# 設計場: ブロック固体度 s → セル物性
# ==========================================================================
def expand(geo: Geo, blocks: torch.Tensor) -> torch.Tensor:
    """(by, bx) ブロック値 → (n_cells,) セル値."""
    r_y = blocks.repeat_interleave(geo.ncy // geo.by, dim=0)
    return r_y.repeat_interleave(geo.ncx // geo.bx, dim=1).reshape(-1)


GEBART_C1 = 16.0 / (9.0 * np.pi * np.sqrt(2.0))  # 正方配列・横断流の Gebart 係数
GEBART_PHI_C = np.pi / 4.0  # 正方配列の最大充填率


def pin_phi(cfg: DarcyConfig, s: torch.Tensor) -> torch.Tensor:
    """ピン充填率 φ = phi_max·s (φ=0 で div/NaN が出ないよう下限クランプ)."""
    return (cfg.phi_max * s).clamp_min(1e-12)


def permeability(cfg: DarcyConfig, s: torch.Tensor) -> torch.Tensor:
    """透過率 K(s).

    - 旧モード: K = K_open · k_ratio^s (対数補間、幾何的裏付けなし)
    - pin_fin:  1/K = 1/K_plate + 1/K_pin(φ)。平行平板 Hele-Shaw (t²/12) と
      正方配列円柱の横断流 (Gebart 1992: K = C1·R²·(√(φ_c/φ)-1)^{5/2}) の直列。
      φ→0 で K_pin→∞ となり K→K_plate に漸近 (全開 = ピンなし平板流路)。
    """
    k_plate = cfg.t_chan**2 / 12.0
    if not cfg.pin_fin:
        return k_plate * torch.exp(float(np.log(cfg.k_ratio)) * s)
    phi = pin_phi(cfg, s)
    r_pin = cfg.d_pin / 2.0
    k_pin = GEBART_C1 * r_pin**2 * (torch.sqrt(GEBART_PHI_C / phi) - 1.0).clamp_min(1e-9) ** 2.5
    return 1.0 / (1.0 / k_plate + 1.0 / k_pin)


def conductivity(cfg: DarcyConfig, s: torch.Tensor) -> torch.Tensor:
    """流路層の面内熱伝導率 (線形混合則).

    旧モード: k_c = k_f + (k_s-k_f)·s。pin_fin: 同式を φ = phi_max·s で評価
    (ピンは層を貫通する Al 柱なので面内は並列混合の上限側で近似)。
    """
    x = pin_phi(cfg, s) if cfg.pin_fin else s
    return cfg.k_f + (cfg.k_s - cfg.k_f) * x


def interlayer_u(cfg: DarcyConfig, s: torch.Tensor) -> torch.Tensor:
    """ベース板→流路層の層間コンダクタンス U'' [W/m²K] (単位底面積あたり).

    - 旧モード: 半厚み伝導の直列 1/(t_base/2k_s + t_chan/2k_c(s))
    - pin_fin: ベース半厚み伝導と「プライム面 + ピンフィン」の直列。
        プライム面: (1-φ)·h0, h0 = 2k_f/t_chan (半深さ伝導床)
        ピン: 側面積比 a_fin = 4φ·t_chan/d, フィン効率 η = tanh(mL)/mL,
              m = √(4h_pin/(k_s·d)), h_pin = Nu_pin·k_f/d (速度非依存)
      φ↑ で濡れ面積が増え U'' が上がる — 「流れを絞る」と「z 方向 h を上げる」が
      同じ変数 s の別の顔になり、s の物理的意味が閉じる。
    """
    r_base = cfg.t_base / (2.0 * cfg.k_s)
    if not cfg.pin_fin:
        return 1.0 / (r_base + cfg.t_chan / (2.0 * conductivity(cfg, s)))
    phi = pin_phi(cfg, s)
    h0 = 2.0 * cfg.k_f / cfg.t_chan
    h_pin = cfg.nu_pin * cfg.k_f / cfg.d_pin
    m_fin = float(np.sqrt(4.0 * h_pin / (cfg.k_s * cfg.d_pin)))
    eta = float(np.tanh(m_fin * cfg.t_chan) / (m_fin * cfg.t_chan))
    h_eff = (1.0 - phi) * h0 + eta * h_pin * 4.0 * phi * cfg.t_chan / cfg.d_pin
    return 1.0 / (r_base + 1.0 / h_eff)


# ==========================================================================
# 流れ: ∇·(K t/μ ∇p) = 0, 流入フラックス / 流出 Dirichlet(半セル)
# Forchheimer 有効時は g_eff(|q|) の Picard 反復 (グラフを通して微分可能)
# ==========================================================================
def solve_flow(cfg: DarcyConfig, geo: Geo, s: torch.Tensor) -> dict[str, torch.Tensor]:
    n = geo.n_cells
    m = permeability(cfg, s) * cfg.t_chan / cfg.mu  # セル移動度 [m³/(Pa·s)]
    m_i, m_j = m[geo.fi], m[geo.fj]
    g_f = 2.0 * m_i * m_j / (m_i + m_j)  # 調和平均 (面幅 h と距離 h が相殺)
    g_out = 2.0 * m[geo.outlet_cells]  # 流出境界 p=0 (半セル距離)

    fi_np, fj_np = geo.fi.numpy(), geo.fj.numpy()
    out_np = geo.outlet_cells.numpy()
    rows = np.concatenate([fi_np, fj_np, fi_np, fj_np, out_np])
    cols = np.concatenate([fi_np, fj_np, fj_np, fi_np, out_np])

    q_v = cfg.m_dot / cfg.rho_f  # 体積流量 [m³/s]
    b = torch.zeros(n)
    b[geo.inlet_cells] = q_v / len(geo.inlet_cells)

    if not cfg.forchheimer:
        vals = torch.cat([g_f, g_f, -g_f, -g_f, g_out])
        p = sparse_solve(vals, b, rows, cols, n)
        q_face = g_f * (p[geo.fi] - p[geo.fj])  # >0 で fi→fj
        q_out = g_out * p[geo.outlet_cells]
        dp = p[geo.inlet_cells].mean()  # 流入は一様フラックスなので単純平均
        return {"p": p, "q_face": q_face, "q_out": q_out, "dp": dp, "g_f": g_f, "picard_iters": 0}

    # --- Forchheimer: Δp_F = ρ β |U| U · L,  U = q/(t·h) ---
    if not cfg.pin_fin:
        msg = "forchheimer=True は pin_fin=True が前提 (β(φ) がピン形状に依存)"
        raise ValueError(msg)
    phi = pin_phi(cfg, s)
    beta = cfg.c_ergun * phi / ((1.0 - phi).clamp_min(1e-6) ** 3 * cfg.d_pin)  # [1/m]
    beta_face = 0.5 * (beta[geo.fi] + beta[geo.fj])  # 経路半セルずつの算術平均
    beta_out = 0.5 * beta[geo.outlet_cells]  # 流出は半セル距離
    coef = cfg.rho_f / (cfg.t_chan**2 * geo.h)  # R_F = coef·β_face·|q|

    # 線形化点 q_lin は緩和付きで更新 (裸の固定点は減衰振動して遅い)。
    # 返す q_face/q_out は最終線形解そのものなので発散ゼロは厳密に保たれる。
    q_lin = torch.zeros(geo.fi.shape[0])
    qo_lin = torch.zeros(len(geo.outlet_cells))
    dp_old = float("inf")
    n_it = 0
    w = cfg.picard_relax
    for n_it in range(1, cfg.picard_max + 1):  # noqa: B007 (反復数を picard_iters で返す)
        g_eff = 1.0 / (1.0 / g_f + coef * beta_face * q_lin.abs())
        g_oeff = 1.0 / (1.0 / g_out + coef * beta_out * qo_lin.abs())
        vals = torch.cat([g_eff, g_eff, -g_eff, -g_eff, g_oeff])
        p = sparse_solve(vals, b, rows, cols, n)
        q_face = g_eff * (p[geo.fi] - p[geo.fj])
        q_out = g_oeff * p[geo.outlet_cells]
        dp = p[geo.inlet_cells].mean()
        dp_new = float(dp.detach())
        if abs(dp_new - dp_old) <= cfg.picard_tol * abs(dp_new):
            break
        dp_old = dp_new
        q_lin = w * q_face + (1.0 - w) * q_lin
        qo_lin = w * q_out + (1.0 - w) * qo_lin
    return {
        "p": p,
        "q_face": q_face,
        "q_out": q_out,
        "dp": dp,
        "g_f": g_eff,
        "picard_iters": n_it,
    }


# ==========================================================================
# 熱: 2 層 (ベース板 T_s + 流路層 T_c) 連成、風上移流
# ==========================================================================
def solve_heat(
    cfg: DarcyConfig, geo: Geo, s: torch.Tensor, flow: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    n = geo.n_cells
    fi_np, fj_np = geo.fi.numpy(), geo.fj.numpy()
    out_np = geo.outlet_cells.numpy()
    cells = np.arange(n)

    # 面内伝導: ベース板 (一定) / 流路層 k_c(s)
    g_s = torch.full((geo.fi.shape[0],), cfg.k_s * cfg.t_base)
    k_c = conductivity(cfg, s)
    kc_i, kc_j = k_c[geo.fi], k_c[geo.fj]
    g_c = cfg.t_chan * 2.0 * kc_i * kc_j / (kc_i + kc_j)

    # 移流 (風上): F± = ρ c_p · relu(±q_face)
    rcp = cfg.rho_f * cfg.cp
    f_pos = rcp * torch.relu(flow["q_face"])
    f_neg = rcp * torch.relu(-flow["q_face"])
    f_out = rcp * torch.relu(flow["q_out"])

    # 層間結合 (面積 h²): 旧=半厚み伝導直列 / pin_fin=プライム面+フィン増倍
    u_cpl = geo.h**2 * interlayer_u(cfg, s)

    rows = np.concatenate(
        [
            fi_np,
            fj_np,
            fi_np,
            fj_np,  # ベース伝導
            fi_np + n,
            fj_np + n,
            fi_np + n,
            fj_np + n,  # 流路伝導
            fi_np + n,
            fj_np + n,
            fj_np + n,
            fi_np + n,  # 移流
            out_np + n,  # 流出エンタルピー
            cells,
            cells,
            cells + n,
            cells + n,  # 層間結合
        ]
    )
    cols = np.concatenate(
        [
            fi_np,
            fj_np,
            fj_np,
            fi_np,
            fi_np + n,
            fj_np + n,
            fj_np + n,
            fi_np + n,
            fi_np + n,
            fi_np + n,
            fj_np + n,
            fj_np + n,
            out_np + n,
            cells,
            cells + n,
            cells + n,
            cells,
        ]
    )
    vals = torch.cat(
        [
            g_s,
            g_s,
            -g_s,
            -g_s,
            g_c,
            g_c,
            -g_c,
            -g_c,
            f_pos,
            -f_pos,
            f_neg,
            -f_neg,
            f_out,
            u_cpl,
            -u_cpl,
            u_cpl,
            -u_cpl,
        ]
    )

    b = torch.zeros(2 * n)
    q_cell = cfg.q_block_w / (cfg.refine**2)
    for nd in geo.heater_cells:
        b[nd] += q_cell

    t = sparse_solve(vals, b, rows, cols, 2 * n)
    t_s, t_c = t[:n], t[n:]
    heat_out = (f_out * t_c[geo.outlet_cells]).sum()
    return {"t_s": t_s, "t_c": t_c, "heat_out": heat_out}


def block_temps(geo: Geo, t_s: torch.Tensor) -> torch.Tensor:
    return torch.stack([t_s[nd].mean() for nd in geo.heater_cells])


def t_ref_scale(cfg: DarcyConfig, geo: Geo) -> float:
    return cfg.q_block_w * len(geo.heater_cells) / (cfg.cp * cfg.m_dot)


# ==========================================================================
# 目的関数と最適化
# ==========================================================================
def forward(
    cfg: DarcyConfig, geo: Geo, xi: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    s_b = torch.sigmoid(xi)
    s = expand(geo, s_b)
    flow = solve_flow(cfg, geo, s)
    ht = solve_heat(cfg, geo, s, flow)
    return s_b, {**flow, **ht, "s": s}


def objective(
    cfg: DarcyConfig, geo: Geo, xi: torch.Tensor, gamma_p: float, dp_ref: float
) -> tuple[torch.Tensor, dict[str, float]]:
    _, st = forward(cfg, geo, xi)
    t_ref = t_ref_scale(cfg, geo)
    t_b = block_temps(geo, st["t_s"])
    j_t = torch.logsumexp(cfg.beta_t * t_b, dim=0) / cfg.beta_t / t_ref
    j_t = j_t + cfg.mu_tvar * t_b.var() / t_ref**2
    j_p = st["dp"] / dp_ref
    j = j_t + gamma_p * j_p
    parts = {
        "j": float(j.detach()),
        "j_t": float(j_t.detach()),
        "dp": float(st["dp"].detach()),
        "t_peak": float(t_b.detach().max()),
        "t_std": float(t_b.detach().std()),
    }
    return j, parts


def dp_reference(cfg: DarcyConfig, geo: Geo) -> float:
    """全開 (s=0) の圧損。J_p の正規化基準."""
    with torch.no_grad():
        flow = solve_flow(cfg, geo, torch.zeros(geo.n_cells))
    return float(flow["dp"])


def optimize(
    cfg: DarcyConfig,
    geo: Geo,
    gamma_p: float,
    iters: int = 500,
    lr: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
    log_fn: Callable[[str], None] = print,
) -> torch.Tensor:
    """γ 固定の重み付き和を Adam で最小化。戻り値: ロジット ξ (by, bx).

    設計変数は連続な透過率場なので 0/1 射影・継続法は不要。
    100 反復ごとにロジットを [-6, 6] にクランプ (シグモイド勾配死の予防)。
    """
    dp_ref = dp_reference(cfg, geo)
    rng = np.random.default_rng(seed)
    xi = torch.tensor(0.05 * rng.standard_normal((geo.by, geo.bx)), requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        j, parts = objective(cfg, geo, xi, gamma_p, dp_ref)
        j.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            with torch.no_grad():
                xi.clamp_(-6.0, 6.0)
        if verbose and (it % 100 == 0 or it == iters - 1):
            log_fn(
                f"    it={it:3d} J={parts['j']:.4f} J_t={parts['j_t']:.4f} "
                f"dp={parts['dp']:.2f}Pa T_peak={parts['t_peak']:.2f}K "
                f"T_std={parts['t_std']:.3f}K"
            )
    return xi.detach()


def evaluate(cfg: DarcyConfig, geo: Geo, s_blocks: torch.Tensor) -> dict[str, float]:
    """与えたブロック固体度場 (by, bx) を評価."""
    with torch.no_grad():
        s = expand(geo, s_blocks)
        flow = solve_flow(cfg, geo, s)
        ht = solve_heat(cfg, geo, s, flow)
        t_b = block_temps(geo, ht["t_s"])
        q_in = cfg.q_block_w * len(geo.heater_cells)
        f_out = cfg.rho_f * cfg.cp * torch.relu(flow["q_out"])
        t_out = float((f_out * ht["t_c"][geo.outlet_cells]).sum() / f_out.sum())
        sb = s_blocks.reshape(-1)
    return {
        "dp": float(flow["dp"]),
        "T_peak": float(t_b.max()),
        "T_block_mean": float(t_b.mean()),
        "T_block_std": float(t_b.std()),
        "T_block_min": float(t_b.min()),
        "T_fluid_out": t_out,
        "heat_balance_rel": abs(float(ht["heat_out"]) - q_in) / q_in,
        "solidity_mean": float(sb.mean()),
    }


# ==========================================================================
# 可視化
# ==========================================================================
def cell_speed(cfg: DarcyConfig, geo: Geo, flow: dict[str, torch.Tensor]) -> np.ndarray:
    """面フラックス → セル中心流速 [m/s] (流路断面 t_chan·h で割る)."""
    n = geo.n_cells
    nfx = (geo.ncx - 1) * geo.ncy
    qx = flow["q_face"][:nfx].numpy()
    qy = flow["q_face"][nfx:].numpy()
    ux = np.zeros(n)
    uy = np.zeros(n)
    cnt_x = np.zeros(n)
    cnt_y = np.zeros(n)
    fi_np, fj_np = geo.fi.numpy(), geo.fj.numpy()
    np.add.at(ux, fi_np[:nfx], qx)
    np.add.at(ux, fj_np[:nfx], qx)
    np.add.at(cnt_x, fi_np[:nfx], 1)
    np.add.at(cnt_x, fj_np[:nfx], 1)
    np.add.at(uy, fi_np[nfx:], qy)
    np.add.at(uy, fj_np[nfx:], qy)
    np.add.at(cnt_y, fi_np[nfx:], 1)
    np.add.at(cnt_y, fj_np[nfx:], 1)
    ux /= np.maximum(cnt_x, 1) * cfg.t_chan * geo.h
    uy /= np.maximum(cnt_y, 1) * cfg.t_chan * geo.h
    return np.sqrt(ux**2 + uy**2)


def panel(
    ax,
    geo: Geo,
    field: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fig=None,
    norm=None,
) -> None:
    from matplotlib.patches import Rectangle

    ext = (0.0, geo.ncx * geo.h * 1e3, 0.0, geo.ncy * geo.h * 1e3)
    scale = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(
        field.reshape(geo.ncy, geo.ncx),
        origin="lower",
        extent=ext,
        cmap=cmap,
        interpolation="nearest",
        **scale,
    )
    r = geo.ncx // geo.bx
    pitch = geo.h * r * 1e3
    for x0, y0 in geo.heater_boxes:
        ax.add_patch(
            Rectangle(
                (x0 * pitch, y0 * pitch),
                pitch,
                pitch,
                fill=False,
                edgecolor="w",
                lw=1.0,
                ls="--",
            )
        )
    ymid = geo.ncy * geo.h * 1e3 / 2
    ax.annotate("in", (0, ymid), color="cyan", fontsize=9, ha="right")
    ax.annotate("out", (geo.ncx * geo.h * 1e3, ymid), color="orange", fontsize=9, ha="left")
    ax.set_title(title, color="w", fontsize=9)
    ax.set_facecolor("#0e1015")
    ax.tick_params(colors="w", labelsize=7)
    for sp_ in ax.spines.values():
        sp_.set_color("w")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(colors="w", labelsize=7)
