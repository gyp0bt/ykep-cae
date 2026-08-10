"""coldplate 擬3D ダルシーモデル — 写経用最小核 (~360 行).

darcy.py 本体から可視化・ポート設計・粒度一般化・ウォームスタートを剥がし、
物理と adjoint だけを残した再構築用のリファレンス。写経の推奨順:

  1. test_darcy_core.py     — 何が保証されるべきかを先に読む (検定器)
  2. SparseSolve            — 離散 adjoint の全て (backward の 5 行)
  3. permeability / interlayer_u / ergun_beta — φ の 3 閉包 (物理的主張)
  4. solve_flow / solve_heat — FVM 行列の組み立てと 2 層連成
  5. objective / optimize   — 2 目的の重み付き和と Adam ループ

再構築版がテスト 10 件を通れば、写経の正しさは構成的に保証される。

モデル (README.md 詳細):
  流れ: 深さ平均ダルシー(+Forchheimer)  ∇·(K t/μ ∇p) = 0
  熱:   ベース板 (伝導+発熱) と流路層 (風上移流+面内伝導) を U''(φ) で層間結合
  設計: ブロックごとのピン充填率 φ = φ_max·sigmoid(ξ)
  目的: J = LSE_β(T_blocks)/T_ref + μ·var(T)/T_ref² + γ·ΔP/ΔP_ref
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu

torch.set_default_dtype(torch.float64)


# ==========================================================================
# 設定と幾何
# ==========================================================================
@dataclass
class Config:
    # 幾何: n_cols×n_rows の発熱体アレイ、5mm 設計ブロック、refine 細分
    n_cols: int = 2
    n_rows: int = 4
    block_pitch: int = 3  # 発熱体間隔 [ブロック]
    margin_left: int = 4
    margin_right: int = 3
    margin_y: int = 3
    refine: int = 4
    cell_pitch: float = 5e-3  # 設計ブロックピッチ [m]
    t_base: float = 3e-3  # ベース板厚 [m]
    t_chan: float = 2e-3  # 流路層深さ [m]
    # 物性 (水 / Al)
    mu: float = 1.0e-3
    rho_f: float = 998.0
    cp: float = 4180.0
    k_f: float = 0.6
    k_s: float = 167.0
    m_dot: float = 5e-3  # 総質量流量 [kg/s]
    q_block_w: float = 10.0  # 発熱 [W/block]
    # ピンフィン: 設計変数 s ∈ [0,1] → 充填率 φ = phi_max·s
    d_pin: float = 1.0e-3
    phi_max: float = 0.6
    nu_pin: float = 4.0  # ピン周り Nu (速度非依存の伝導床)
    # Forchheimer 慣性補正
    forchheimer: bool = False
    c_ergun: float = 1.75
    picard_max: int = 60
    picard_tol: float = 1e-12
    picard_relax: float = 0.5  # 裸の固定点は減衰振動する — 緩和必須
    # 目的
    beta_t: float = 2.0  # smooth-max の鋭さ [1/K]
    mu_tvar: float = 10.0  # 温度分散罰則


@dataclass(frozen=True)
class Geo:
    bx: int
    by: int
    ncx: int
    ncy: int
    h: float  # 細分セル寸法 [m]
    fi: torch.Tensor  # 面の両側セル id (x方向面 → y方向面 の順)
    fj: torch.Tensor
    inlet_cells: torch.Tensor  # 左辺中央 5mm
    outlet_cells: torch.Tensor  # 右辺中央 5mm
    heater_cells: list[torch.Tensor] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return self.ncx * self.ncy


def make_geo(cfg: Config) -> Geo:
    span_x = (cfg.n_cols - 1) * cfg.block_pitch + 1
    span_y = (cfg.n_rows - 1) * cfg.block_pitch + 1
    bx = cfg.margin_left + span_x + cfg.margin_right
    by = 2 * cfg.margin_y + span_y
    r = cfg.refine
    ncx, ncy = bx * r, by * r

    ix, iy = np.arange(ncx - 1), np.arange(ncy)
    fx_i = (iy[:, None] * ncx + ix[None, :]).ravel()
    ix2, iy2 = np.arange(ncx), np.arange(ncy - 1)
    fy_i = (iy2[:, None] * ncx + ix2[None, :]).ravel()
    fi = np.concatenate([fx_i, fy_i])
    fj = np.concatenate([fx_i + 1, fy_i + ncx])

    j0 = ncy // 2 - r // 2
    port_j = np.arange(j0, j0 + r)

    heater_cells = []
    for c in range(cfg.n_cols):
        x0 = cfg.margin_left + c * cfg.block_pitch
        for row in range(cfg.n_rows):
            y0 = cfg.margin_y + row * cfg.block_pitch
            cx = np.arange(x0 * r, (x0 + 1) * r)
            cy = np.arange(y0 * r, (y0 + 1) * r)
            heater_cells.append(
                torch.tensor((cy[:, None] * ncx + cx[None, :]).ravel(), dtype=torch.long)
            )

    return Geo(
        bx=bx,
        by=by,
        ncx=ncx,
        ncy=ncy,
        h=cfg.cell_pitch / r,
        fi=torch.tensor(fi, dtype=torch.long),
        fj=torch.tensor(fj, dtype=torch.long),
        inlet_cells=torch.tensor(port_j * ncx, dtype=torch.long),
        outlet_cells=torch.tensor(port_j * ncx + (ncx - 1), dtype=torch.long),
        heater_cells=heater_cells,
    )


# ==========================================================================
# 離散 adjoint: x = A⁻¹b の backward は転置系を 1 回解くだけ
#   J(x(A,b)) に対し  λ = A⁻ᵀ (∂J/∂x),  ∂J/∂A_kl = -λ_k x_l,  ∂J/∂b = λ
# 設計変数が何千個でも勾配コストは前進解 1 回分 — adjoint 法の核心
# ==========================================================================
class SparseSolve(torch.autograd.Function):
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
        return -lam[ctx.rows] * x[ctx.cols], lam, None, None, None


def sparse_solve(vals, b, rows, cols, n):  # noqa: ANN001, ANN201
    return SparseSolve.apply(vals, b, rows, cols, n)


# ==========================================================================
# φ の 3 閉包 — 物理的主張はここに集約されている
# ==========================================================================
GEBART_C1 = 16.0 / (9.0 * np.pi * np.sqrt(2.0))
GEBART_PHI_C = np.pi / 4.0  # 正方配列の幾何限界


def pin_phi(cfg: Config, s: torch.Tensor) -> torch.Tensor:
    return (cfg.phi_max * s).clamp_min(1e-12)


def permeability(cfg: Config, s: torch.Tensor) -> torch.Tensor:
    """1/K = 1/K_plate + 1/K_pin: 平行平板と Gebart 円柱列の抵抗直列.

    φ→0 で K_pin→∞ となり K→t²/12 (ピンなし平板) に連続漸近する。
    """
    k_plate = cfg.t_chan**2 / 12.0
    phi = pin_phi(cfg, s)
    r_pin = cfg.d_pin / 2.0
    k_pin = GEBART_C1 * r_pin**2 * (torch.sqrt(GEBART_PHI_C / phi) - 1.0).clamp_min(1e-9) ** 2.5
    return 1.0 / (1.0 / k_plate + 1.0 / k_pin)


def conductivity(cfg: Config, s: torch.Tensor) -> torch.Tensor:
    """流路層の面内熱伝導率 (ピンは層を貫通する Al 柱 → 並列混合上限)."""
    return cfg.k_f + (cfg.k_s - cfg.k_f) * pin_phi(cfg, s)


def interlayer_u(cfg: Config, s: torch.Tensor) -> torch.Tensor:
    """ベース板→流体の層間コンダクタンス U'' [W/m²K].

    プライム面 (1-φ)·h0 + ピンフィン増倍 (側面積比 4φt/d × フィン効率 η)。
    ピン側面→流体の対流段 (フィン抵抗) を挟むのが物理的な要点 —
    「固体を層厚伝導スラブ扱いする」楽観 (旧モデルの一桁過大) を排除する。
    """
    r_base = cfg.t_base / (2.0 * cfg.k_s)
    phi = pin_phi(cfg, s)
    h0 = 2.0 * cfg.k_f / cfg.t_chan
    h_pin = cfg.nu_pin * cfg.k_f / cfg.d_pin
    m_fin = float(np.sqrt(4.0 * h_pin / (cfg.k_s * cfg.d_pin)))
    eta = float(np.tanh(m_fin * cfg.t_chan) / (m_fin * cfg.t_chan))
    h_eff = (1.0 - phi) * h0 + eta * h_pin * 4.0 * phi * cfg.t_chan / cfg.d_pin
    return 1.0 / (r_base + 1.0 / h_eff)


def ergun_beta(cfg: Config, s: torch.Tensor) -> torch.Tensor:
    """Forchheimer 係数 β(φ) [1/m]。φ=0 で形状抗力ゼロ (平板に帰着)."""
    phi = pin_phi(cfg, s)
    return cfg.c_ergun * phi / ((1.0 - phi).clamp_min(1e-6) ** 3 * cfg.d_pin)


def expand(geo: Geo, blocks: torch.Tensor) -> torch.Tensor:
    """(by, bx) ブロック値 → (n_cells,) セル値."""
    r_y = blocks.repeat_interleave(geo.ncy // geo.by, dim=0)
    return r_y.repeat_interleave(geo.ncx // geo.bx, dim=1).reshape(-1)


# ==========================================================================
# 流れ: ∇·(K t/μ ∇p) = 0。流入フラックス / 流出 Robin (半セル)
# Forchheimer 有効時は面抵抗直列 1/g_eff = 1/g_D + R_F(|q|) を緩和付き
# Picard で反復 (グラフを通すので微分可能。返す流束は最終線形解 = 保存厳密)
# ==========================================================================
def solve_flow(cfg: Config, geo: Geo, s: torch.Tensor) -> dict[str, torch.Tensor]:
    n = geo.n_cells
    m = permeability(cfg, s) * cfg.t_chan / cfg.mu  # セル移動度 [m³/(Pa·s)]
    m_i, m_j = m[geo.fi], m[geo.fj]
    g_f = 2.0 * m_i * m_j / (m_i + m_j)  # 調和平均 (面幅 h と距離 h が相殺)
    g_out = 2.0 * m[geo.outlet_cells]

    fi_np, fj_np = geo.fi.numpy(), geo.fj.numpy()
    out_np = geo.outlet_cells.numpy()
    rows = np.concatenate([fi_np, fj_np, fi_np, fj_np, out_np])
    cols = np.concatenate([fi_np, fj_np, fj_np, fi_np, out_np])

    q_v = cfg.m_dot / cfg.rho_f
    b = torch.zeros(n)
    b[geo.inlet_cells] = q_v / len(geo.inlet_cells)

    if not cfg.forchheimer:
        vals = torch.cat([g_f, g_f, -g_f, -g_f, g_out])
        p = sparse_solve(vals, b, rows, cols, n)
        q_face = g_f * (p[geo.fi] - p[geo.fj])
        return {
            "p": p,
            "q_face": q_face,
            "q_out": g_out * p[geo.outlet_cells],
            "dp": p[geo.inlet_cells].mean(),
        }

    beta = ergun_beta(cfg, s)
    beta_face = 0.5 * (beta[geo.fi] + beta[geo.fj])
    beta_out = 0.5 * beta[geo.outlet_cells]
    coef = cfg.rho_f / (cfg.t_chan**2 * geo.h)  # R_F = coef·β·|q|

    q_lin = torch.zeros(geo.fi.shape[0])
    qo_lin = torch.zeros(len(geo.outlet_cells))
    dp_old = float("inf")
    w = cfg.picard_relax
    for _ in range(cfg.picard_max):
        g_eff = 1.0 / (1.0 / g_f + coef * beta_face * q_lin.abs())
        g_oeff = 1.0 / (1.0 / g_out + coef * beta_out * qo_lin.abs())
        vals = torch.cat([g_eff, g_eff, -g_eff, -g_eff, g_oeff])
        p = sparse_solve(vals, b, rows, cols, n)
        q_face = g_eff * (p[geo.fi] - p[geo.fj])
        q_out = g_oeff * p[geo.outlet_cells]
        dp = p[geo.inlet_cells].mean()
        if abs(float(dp.detach()) - dp_old) <= cfg.picard_tol * abs(float(dp.detach())):
            break
        dp_old = float(dp.detach())
        q_lin = w * q_face + (1.0 - w) * q_lin
        qo_lin = w * q_out + (1.0 - w) * qo_lin
    return {"p": p, "q_face": q_face, "q_out": q_out, "dp": dp}


# ==========================================================================
# 熱: 2 層 (ベース板 T_s + 流路層 T_c) を 1 つの 2N×2N 系に束ねて解く
# ==========================================================================
def solve_heat(cfg: Config, geo: Geo, s: torch.Tensor, flow: dict) -> dict[str, torch.Tensor]:
    n = geo.n_cells
    fi_np, fj_np = geo.fi.numpy(), geo.fj.numpy()
    out_np = geo.outlet_cells.numpy()
    cells = np.arange(n)

    g_s = torch.full((geo.fi.shape[0],), cfg.k_s * cfg.t_base)  # ベース板 面内伝導
    k_c = conductivity(cfg, s)
    kc_i, kc_j = k_c[geo.fi], k_c[geo.fj]
    g_c = cfg.t_chan * 2.0 * kc_i * kc_j / (kc_i + kc_j)  # 流路層 面内伝導

    rcp = cfg.rho_f * cfg.cp  # 風上移流: F± = ρc_p·relu(±q)
    f_pos, f_neg = rcp * torch.relu(flow["q_face"]), rcp * torch.relu(-flow["q_face"])
    f_out = rcp * torch.relu(flow["q_out"])

    u_cpl = geo.h**2 * interlayer_u(cfg, s)  # 層間結合 (面積 h²)

    rows = np.concatenate(
        [fi_np, fj_np, fi_np, fj_np]  # ベース伝導
        + [fi_np + n, fj_np + n, fi_np + n, fj_np + n]  # 流路伝導
        + [fi_np + n, fj_np + n, fj_np + n, fi_np + n]  # 移流
        + [out_np + n, cells, cells, cells + n, cells + n]  # 流出・層間
    )
    cols = np.concatenate(
        [fi_np, fj_np, fj_np, fi_np]
        + [fi_np + n, fj_np + n, fj_np + n, fi_np + n]
        + [fi_np + n, fi_np + n, fj_np + n, fj_np + n]
        + [out_np + n, cells, cells + n, cells + n, cells]
    )
    vals = torch.cat(
        [g_s, g_s, -g_s, -g_s, g_c, g_c, -g_c, -g_c]
        + [f_pos, -f_pos, f_neg, -f_neg]
        + [f_out, u_cpl, -u_cpl, u_cpl, -u_cpl]
    )

    b = torch.zeros(2 * n)
    for nd in geo.heater_cells:
        b[nd] += cfg.q_block_w / (cfg.refine**2)

    t = sparse_solve(vals, b, rows, cols, 2 * n)
    t_s, t_c = t[:n], t[n:]
    return {"t_s": t_s, "t_c": t_c, "heat_out": (f_out * t_c[geo.outlet_cells]).sum()}


# ==========================================================================
# 目的関数と最適化
# ==========================================================================
def t_ref_scale(cfg: Config, geo: Geo) -> float:
    """出口流体温度の理論値 ΣQ/(c_p·ṁ) — エネルギー収支テストの基準."""
    return cfg.q_block_w * len(geo.heater_cells) / (cfg.cp * cfg.m_dot)


def dp_reference(cfg: Config, geo: Geo) -> float:
    with torch.no_grad():
        return float(solve_flow(cfg, geo, torch.zeros(geo.n_cells))["dp"])


def objective(
    cfg: Config, geo: Geo, xi: torch.Tensor, gamma_p: float, dp_ref: float
) -> torch.Tensor:
    s = expand(geo, torch.sigmoid(xi))
    flow = solve_flow(cfg, geo, s)
    ht = solve_heat(cfg, geo, s, flow)
    t_b = torch.stack([ht["t_s"][nd].mean() for nd in geo.heater_cells])
    t_ref = t_ref_scale(cfg, geo)
    j_t = torch.logsumexp(cfg.beta_t * t_b, dim=0) / cfg.beta_t / t_ref
    j_t = j_t + cfg.mu_tvar * t_b.var() / t_ref**2
    return j_t + gamma_p * flow["dp"] / dp_ref


def optimize(cfg: Config, geo: Geo, gamma_p: float, iters: int = 500, seed: int = 0):  # noqa: ANN201
    dp_ref = dp_reference(cfg, geo)
    rng = np.random.default_rng(seed)
    xi = torch.tensor(0.05 * rng.standard_normal((geo.by, geo.bx)), requires_grad=True)
    opt = torch.optim.Adam([xi], lr=0.1)
    for it in range(iters):
        opt.zero_grad()
        objective(cfg, geo, xi, gamma_p, dp_ref).backward()
        opt.step()
        if (it + 1) % 100 == 0:
            with torch.no_grad():
                xi.clamp_(-6.0, 6.0)  # シグモイド飽和による勾配死の予防
    return xi.detach()
