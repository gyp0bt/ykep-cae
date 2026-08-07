"""coldplate.py -- 格子グラフ上の冷却流路 概念設計ソルバ (単一ファイル統合版 v2).

freshnet.py .. freshnet6.py の試行錯誤を 1 ファイルに統合したもの (v1) に、
P0 (連結性の破れ = 孤立領域への漏れ流れ) の修正を入れた版。
外部依存は numpy / scipy / torch / matplotlib のみ。

--------------------------------------------------------------------------
v2 での修正 (v1 からの差分)
--------------------------------------------------------------------------
[P0] 「孤立しているところにも液が流れる」偽解の排除
  v1 は SIMP の rho_floor (1e-4) により、消したはずの辺にも
  g = rho_floor^p w^4/L > 0 の伝導度が残り、Kirchhoff 解が全域に微小流を
  配っていた。節点レベルの質量保存 (div q = b) 自体は満たされているが、
  「存在しない流路」が流量と除熱を運ぶため、目的関数に偽の得点が混入し、
  connectivity_check が False の解が最適に見えていた。

  対策 (逐次刈り込み + アクティブ部分グラフ厳密再解):
    1. prune():  rho > rho_cut の辺で連結成分を取り、inlet/outlet の両方を
       含む主成分だけを残す。それ以外の辺は rho = 0 (厳密ゼロ) で凍結。
    2. physics(edge_on=...):  アクティブ節点のみで Kirchhoff を解く。
       非アクティブ辺は g = 0 なので流量は「厳密に」ゼロ。
    3. optimise(mask=...):  凍結マスク付きで残り変数を再最適化。
       ポート softmax も主成分内のポートに制限 (自動で再正規化)。
    4. mass_check():  節点質量残差と非アクティブ辺の漏れ流量を検証。
  これを prune -> reopt で数ラウンド繰り返す (ground structure 法の定石)。

[P1] grey 削減: penal / beta / mu_bin の continuation (solve_pipeline)

[均一性] ブロック別除熱の min/mean, CV を報告。刈り込み後ラウンドでは
  softmin の beta を強化して worst-block を直接押し上げる。
  必要なら mu_uni (log 除熱の分散罰則) を有効化できる。

--------------------------------------------------------------------------
定式化 (v1 と同一)
--------------------------------------------------------------------------
格子間隔 P = 1 ≡ 流路ピッチ = 流路幅 d + 壁厚 t   (幾何整合が必須)

設計変数 (すべて実数, sigmoid/softmax で箱制約を内点化)
  rho_e ∈ [0,1]          辺の存在度 (SIMP)
  w_e   ∈ [d_min,d_max]  流路幅
  a_in, a_out            左辺ポートへの流入/流出配分ロジット (softmax)

物理 (すべて微分可能, torch.autograd で厳密勾配)
  g_e   = rho_e^p · w_e^4 / L_e                Hagen-Poiseuille (p=3 SIMP)
  b     = Q_tot · (softmax(a_in) - softmax(a_out))   節点湧き出し (Σb=0)
  L(g) p = b                                   Kirchhoff (基準節点固定)
  q     = g ⊙ (A p)
  NTU_e = c · rho_e · L_e · |q_e|^-0.75 · w_e^-0.25
  tau_e = exp(-NTU_e)
  鮮度 φ を上流風で輸送:  M(q,tau) φ = b⁺
  辺の除熱量:  q_heat_e = |q_e| · φ_up,e · (1 - tau_e)

目的
  (1) max  min_b Σ_{e ∈ block b} q_heat_e      softmin で平滑化
  (2) s.t. 散逸 p·b  ≤  exp(log_p_max)         ε制約 (罰則)
  (3) s.t. 体積 Σ rho L w²  ≤  V0              不等式 (両側は死枝が出る)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)

EPS = 1e-12
QEPS = 1e-9
NEG_INF = -1e30


# ==========================================================================
# グラフ
# ==========================================================================
@dataclass(frozen=True)
class GridGraph:
    nx: int
    ny: int
    tail: torch.Tensor
    head: torch.Tensor
    elen: torch.Tensor

    @property
    def n_nodes(self) -> int:
        return self.nx * self.ny

    @property
    def n_edges(self) -> int:
        return int(self.tail.shape[0])

    def nid(self, i: int, j: int) -> int:
        return j * self.nx + i

    def coords(self) -> np.ndarray:
        idx = np.arange(self.n_nodes)
        return np.stack([idx % self.nx, idx // self.nx], axis=1).astype(float)

    def grad(self, p: torch.Tensor) -> torch.Tensor:
        """A @ p = p_tail - p_head."""
        return p[self.tail] - p[self.head]

    def div(self, q: torch.Tensor) -> torch.Tensor:
        """Aᵀ @ q = 節点正味湧き出し."""
        return (
            torch.zeros(self.n_nodes, dtype=q.dtype)
            .index_add_(0, self.tail, q)
            .index_add_(0, self.head, -q)
        )

    def laplacian(self, g: torch.Tensor) -> torch.Tensor:
        """Aᵀ diag(g) A を scatter で構築。密 matmul より 30x 速い。"""
        rows = torch.cat([self.tail, self.head, self.tail, self.head])
        cols = torch.cat([self.tail, self.head, self.head, self.tail])
        vals = torch.cat([g, g, -g, -g])
        return torch.zeros(self.n_nodes, self.n_nodes, dtype=g.dtype).index_put_(
            (rows, cols), vals, accumulate=True
        )

    @staticmethod
    def build(nx: int, ny: int, diagonals: bool = False) -> GridGraph:
        t: list[int] = []
        h: list[int] = []
        ln: list[float] = []

        def add(a: int, b: int, length: float) -> None:
            t.append(a)
            h.append(b)
            ln.append(length)

        r2 = float(np.sqrt(2.0))
        for j in range(ny):
            for i in range(nx):
                a = j * nx + i
                if i + 1 < nx:
                    add(a, a + 1, 1.0)
                if j + 1 < ny:
                    add(a, a + nx, 1.0)
                if diagonals and i + 1 < nx and j + 1 < ny:
                    add(a, a + nx + 1, r2)
                    add(a + 1, a + nx, r2)
        return GridGraph(nx, ny, torch.tensor(t), torch.tensor(h), torch.tensor(ln))


# ==========================================================================
# 設定
# ==========================================================================
@dataclass
class Problem:
    graph: GridGraph
    ports: torch.Tensor  # 候補ポート節点 (左辺)
    block_edges: list[torch.Tensor]  # 各発熱体 footprint 内の辺
    block_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    ntu_coef: float = 1e-2
    q_total: float = 1.0
    beta: float = 8.0  # softmin の鋭さ


@dataclass
class Config:
    # 幾何 (格子ピッチ = 1 に対する比)
    d_min: float = 0.60
    d_max: float = 0.75
    # SIMP
    penal: float = 3.0
    rho_floor: float = 1e-4
    # 制約
    vol_frac: float = 0.25
    mu_vol: float = 200.0
    mu_diss: float = 30.0
    # 正則化
    mu_bin: float = 3.0  # rho(1-rho) 罰則
    mu_conc: float = 0.0  # ポート集中化 (0 なら分散を許す)
    mu_overlap: float = 5.0  # inlet/outlet の同一節点重なり禁止
    mu_uni: float = 0.0  # ブロック別 log 除熱の分散罰則 (均一性)
    logit_scale: float = 1.0


@dataclass
class DesignMask:
    """刈り込みで確定した位相の凍結マスク (v2 新設).

    edge_on が False の辺は rho = 0 (厳密ゼロ)、True の辺は rho = 1 に
    凍結され (grey の構造的排除)、
    port_on が False のポートは softmax から除外 (ロジット -inf)、
    block_on が False のブロックは目的関数の softmin から除外される
    (凍結位相では到達不能なため。放置すると softmin が飽和して
    勾配が死ぬ。除外したブロックは orphan として報告する)。
    """

    edge_on: np.ndarray  # bool (E,)
    port_on: np.ndarray  # bool (P,)
    block_on: np.ndarray  # bool (B,)

    def same_as(self, other: DesignMask | None) -> bool:
        if other is None:
            return False
        return (
            bool(np.array_equal(self.edge_on, other.edge_on))
            and bool(np.array_equal(self.port_on, other.port_on))
            and bool(np.array_equal(self.block_on, other.block_on))
        )


# ==========================================================================
# 順方向
# ==========================================================================
def decode(
    x: torch.Tensor, cfg: Config, n_e: int, n_p: int, mask: DesignMask | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = cfg.rho_floor + (1.0 - cfg.rho_floor) * torch.sigmoid(x[:n_e])
    w = cfg.d_min + (cfg.d_max - cfg.d_min) * torch.sigmoid(x[n_e : 2 * n_e])
    a_in = x[2 * n_e : 2 * n_e + n_p] * cfg.logit_scale
    a_out = x[2 * n_e + n_p :] * cfg.logit_scale
    if mask is not None:
        # 位相確定後は rho を厳密 0/1 に固定する (grey の構造的排除)。
        # 凍結辺は rho_floor すら残さない。ここが漏れ流れ排除の本体。
        rho = torch.from_numpy(mask.edge_on.astype(np.float64))
        off = torch.from_numpy(~mask.port_on)
        a_in = a_in.masked_fill(off, NEG_INF)
        a_out = a_out.masked_fill(off, NEG_INF)
    return rho, w, torch.softmax(a_in, 0), torch.softmax(a_out, 0)


def _active_nodes(gr: GridGraph, edge_on: np.ndarray) -> np.ndarray:
    act = np.zeros(gr.n_nodes, dtype=bool)
    act[gr.tail.numpy()[edge_on]] = True
    act[gr.head.numpy()[edge_on]] = True
    return np.nonzero(act)[0]


def physics(
    prob: Problem,
    cfg: Config,
    rho: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    edge_on: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    gr = prob.graph
    n = gr.n_nodes
    g = rho**cfg.penal * w**4 / gr.elen

    # --- Kirchhoff ---
    # edge_on 指定時はアクティブ節点だけで解く。非アクティブ辺は g = 0 なので
    # q = g·(Ap) が厳密にゼロになり、孤立領域への漏れ流れは存在しない。
    lap = gr.laplacian(g)
    if edge_on is None:
        idx = torch.arange(1, n)  # 節点 0 を基準に固定
    else:
        act = _active_nodes(gr, edge_on)
        if len(act) < 2:
            raise ValueError("active subgraph is empty; prune failed")
        idx = torch.from_numpy(act[1:])  # act[0] を基準に固定
    p_r = torch.linalg.solve(lap[idx][:, idx] + EPS * torch.eye(len(idx)), b[idx])
    p = torch.zeros(n).index_copy(0, idx, p_r)
    q = g * gr.grad(p)
    qa = q.abs() + QEPS

    # --- NTU / 減衰 ---
    ntu = prob.ntu_coef * rho * gr.elen * qa ** (-0.75) * w ** (-0.25)
    tau = torch.exp(-ntu)

    # --- 鮮度の上流風輸送 ---
    qp, qm = torch.relu(q), torch.relu(-q)
    b_in = torch.relu(b)
    inflow = torch.zeros(n).index_add_(0, gr.head, qp).index_add_(0, gr.tail, qm) + b_in
    rows = torch.cat([gr.head, gr.tail])
    cols = torch.cat([gr.tail, gr.head])
    vals = torch.cat([qp * tau, qm * tau])
    tmat = torch.zeros(n, n).index_put_((rows, cols), vals, accumulate=True)
    phi = torch.linalg.solve(torch.diag(inflow + 1e-9) - tmat, b_in.clone())

    phi_up = (qp * phi[gr.tail] + qm * phi[gr.head]) / qa
    return {
        "q": q,
        "p": p,
        "phi": phi,
        "tau": tau,
        "q_heat": qa * phi_up * (1.0 - tau),
        "volume": (rho * gr.elen * w**2).sum(),
        "diss": (p * b).sum(),  # = ΣQ²/g だが SIMP の数値発散を避けられる
    }


def state(
    prob: Problem, cfg: Config, x: torch.Tensor, mask: DesignMask | None = None
) -> dict[str, torch.Tensor]:
    gr = prob.graph
    rho, w, win, wout = decode(x, cfg, gr.n_edges, len(prob.ports), mask)
    b = torch.zeros(gr.n_nodes).index_add_(0, prob.ports, prob.q_total * (win - wout))
    st = physics(prob, cfg, rho, w, b, None if mask is None else mask.edge_on)
    st.update({"rho": rho, "w": w, "w_in": win, "w_out": wout, "b": b})
    return st


def _per_block(prob: Problem, q_heat: torch.Tensor) -> torch.Tensor:
    return torch.stack([q_heat[e].sum() for e in prob.block_edges])


def objective(
    prob: Problem,
    cfg: Config,
    x: torch.Tensor,
    log_p_max: float | None = None,
    mask: DesignMask | None = None,
) -> torch.Tensor:
    gr = prob.graph
    st = state(prob, cfg, x, mask)

    per_block = _per_block(prob, st["q_heat"])
    if mask is not None:
        # 凍結位相で到達不能なブロックは softmin から外す (勾配死の防止)。
        per_block = per_block[torch.from_numpy(mask.block_on)]
    logs = torch.log(per_block + 1e-14)
    worst = -torch.logsumexp(-prob.beta * logs, dim=0) / prob.beta

    d_nom = 0.5 * (cfg.d_min + cfg.d_max)
    v_target = cfg.vol_frac * float((gr.elen * d_nom**2).sum())

    j = -worst
    j = j + cfg.mu_vol * torch.relu(st["volume"] / v_target - 1.0) ** 2
    if cfg.mu_bin > 0:
        j = j + cfg.mu_bin * (st["rho"] * (1.0 - st["rho"])).mean()
    if cfg.mu_conc > 0:
        j = j + cfg.mu_conc * (2.0 - (st["w_in"] ** 2).sum() - (st["w_out"] ** 2).sum())
    if cfg.mu_overlap > 0:
        j = j + cfg.mu_overlap * (st["w_in"] * st["w_out"]).sum()
    if cfg.mu_uni > 0:
        j = j + cfg.mu_uni * logs.var()
    if log_p_max is not None:
        j = j + cfg.mu_diss * torch.relu(torch.log(st["diss"] + EPS) - log_p_max) ** 2
    return j


# ==========================================================================
# 最適化
# ==========================================================================
def optimise(
    prob: Problem,
    cfg: Config,
    x0: np.ndarray | None = None,
    maxiter: int = 1200,
    seed: int = 0,
    log_p_max: float | None = None,
    mask: DesignMask | None = None,
) -> np.ndarray:
    n_e, n_p = prob.graph.n_edges, len(prob.ports)
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = np.concatenate(
            [
                -1.5 + 0.3 * rng.standard_normal(n_e),  # rho: 疎側から始める
                0.3 * rng.standard_normal(n_e),  # w
                0.3 * rng.standard_normal(n_p),  # a_in
                0.3 * rng.standard_normal(n_p),  # a_out
            ]
        )

    def fg(xv: np.ndarray) -> tuple[float, np.ndarray]:
        xt = torch.tensor(xv, requires_grad=True)
        j = objective(prob, cfg, xt, log_p_max, mask)
        (grad,) = torch.autograd.grad(j, xt)
        return j.item(), grad.numpy()

    res = minimize(
        fg,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0)] * len(x0),
        options={"maxiter": maxiter, "maxcor": 30},
    )
    return res.x


def pareto_sweep(
    prob: Problem, cfg: Config, caps: Sequence[float], maxiter: int = 1200
) -> list[dict[str, object]]:
    """散逸上限を掃引。前解を初期値に使う (continuation)。

    刻みを粗くすると自明解に飛ぶので細かく取ること。
    """
    out: list[dict[str, object]] = []
    x0: np.ndarray | None = None
    for cap in caps:
        x = optimise(prob, cfg, x0=x0, maxiter=maxiter, log_p_max=cap)
        x0 = x.copy()
        out.append({"cap": cap, "x": x, **report(prob, cfg, x)})
    return out


# ==========================================================================
# 刈り込み (v2: P0 対策の本体)
# ==========================================================================
def _union_find_labels(
    n_nodes: int, tail: np.ndarray, head: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    parent = np.arange(n_nodes)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in np.nonzero(keep)[0]:
        ra, rb = find(int(tail[e])), find(int(head[e]))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def _main_component(
    prob: Problem,
    keep: np.ndarray,
    yi: np.ndarray,
    yo: np.ndarray,
) -> tuple[int, float] | None:
    """inlet/outlet 双方の重みを最も多く含む連結成分を返す (label, min_share)."""
    gr = prob.graph
    tail, head = gr.tail.numpy(), gr.head.numpy()
    comp = _union_find_labels(gr.n_nodes, tail, head, keep)
    has_edge = np.zeros(gr.n_nodes, dtype=bool)
    has_edge[comp[tail[keep]]] = True  # 辺を 1 本以上持つ成分ラベル

    pn = prob.ports.numpy()
    share_in: dict[int, float] = {}
    share_out: dict[int, float] = {}
    for k, nd in enumerate(pn):
        c = int(comp[nd])
        if not has_edge[c]:
            continue
        share_in[c] = share_in.get(c, 0.0) + float(yi[k])
        share_out[c] = share_out.get(c, 0.0) + float(yo[k])

    best, best_score = None, 0.0
    for c in share_in:
        score = min(share_in[c], share_out.get(c, 0.0))
        if score > best_score:
            best, best_score = c, score
    if best is None or best_score <= 0.02:
        return None
    return best, best_score


def _bridge_edges(gr: GridGraph, edge_on: np.ndarray, port_on_nodes: np.ndarray) -> np.ndarray:
    """仮想ノード経由で全ポートを結んだグラフ G' の橋 (実辺のみ) を返す.

    G' = アクティブ辺 + {各ポート節点 - 仮想ノード} の補助辺。
    G' で橋である実辺は、どのポート配分でも定常流を運べない
    (片側にポートが無い袋小路)。逆に橋でない辺は in-out 経路または
    ループ上にあり、ポート配分次第で流れ得る。
    """
    n_e = gr.n_edges
    tail, head = gr.tail.numpy(), gr.head.numpy()
    virt = gr.n_nodes
    adj: list[list[tuple[int, int]]] = [[] for _ in range(gr.n_nodes + 1)]
    for e in np.nonzero(edge_on)[0]:
        u, v = int(tail[e]), int(head[e])
        adj[u].append((v, int(e)))
        adj[v].append((u, int(e)))
    for k, nd in enumerate(port_on_nodes):
        adj[int(nd)].append((virt, n_e + k))
        adj[virt].append((int(nd), n_e + k))

    n_all = gr.n_nodes + 1
    disc = np.full(n_all, -1, dtype=np.int64)
    low = np.zeros(n_all, dtype=np.int64)
    is_bridge = np.zeros(n_e, dtype=bool)
    timer = 0
    for s in range(n_all):
        if disc[s] != -1 or not adj[s]:
            continue
        disc[s] = low[s] = timer
        timer += 1
        st = [(s, -1, iter(adj[s]))]
        while st:
            u, peid, it = st[-1]
            advanced = False
            for v, eid in it:
                if eid == peid:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    st.append((v, eid, iter(adj[v])))
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if not advanced:
                st.pop()
                if st:
                    pu = st[-1][0]
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu] and 0 <= peid < n_e:
                        is_bridge[peid] = True
    return is_bridge


def prune(
    prob: Problem,
    cfg: Config,
    x: np.ndarray,
    mask: DesignMask | None = None,
    rho_cut: float = 0.5,
    q_cut_rel: float = 1e-6,
    verbose: bool = False,
) -> DesignMask:
    """rho 閾値 + 連結成分抽出 + 死枝除去で凍結マスクを作る.

    round 0 (mask=None) — 位相能力ベース:
      1. rho > cut の辺で連結成分を取り、inlet/outlet の重みを両方含む
         主成分を候補にする。cut は複数試し、「流れ得る」ブロック被覆数が
         最大の cut を採用 (同数なら高い cut)。低い cut は grey の半端な
         実流路を主成分に取り込めるため、被覆と清潔さのトレードオフを
         ここで解く。
      2. G' (全ポートを仮想ノードで結合) の橋 = どのポート配分でも定常流を
         運べない袋小路のみ除去する。現在のポート配分での実流量は使わない
         (まだ重みが乗っていないフィーダを再最適化前に切らないため)。
    round 1+ (mask 有り) — 実流量ベース:
      再最適化済みの解で |q| < q_cut_rel·Q_tot の辺 (使われなかった流路) を
      落とし、主成分を取り直す。
    最後に block_on (除熱が届くブロック) を判定する。
    """
    gr = prob.graph
    tail, head = gr.tail.numpy(), gr.head.numpy()
    with torch.no_grad():
        st = state(prob, cfg, torch.tensor(x), mask)
    rho = st["rho"].numpy()
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    pn = prob.ports.numpy()
    blk_edge_ids = [e.numpy() for e in prob.block_edges]

    def flow_capable(
        keep: np.ndarray, label: int, comp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        on = keep & (comp[tail] == label)
        p_on = comp[pn] == label
        br = _bridge_edges(gr, on, pn[p_on])
        return on & ~br, p_on

    if mask is None:
        # --- round 0: 「流れ得る」被覆が最大になる cut を採用 ---
        chosen = None
        best_cov = -1
        for cut in (rho_cut, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02):
            keep = rho > cut
            found = _main_component(prob, keep, yi, yo)
            if found is None:
                continue
            comp_c = _union_find_labels(gr.n_nodes, tail, head, keep)
            on_c, p_on_c = flow_capable(keep, found[0], comp_c)
            cov = sum(1 for ids in blk_edge_ids if on_c[ids].any())
            if cov > best_cov:
                best_cov = cov
                chosen = (cut, on_c, found[1])
        if chosen is None:
            raise ValueError("no component contains both inlet and outlet flow")
        cut, edge_on, score = chosen
        # 橋除去で主成分が割れることがあるので取り直す
        found = _main_component(prob, edge_on, yi, yo)
        if found is None:
            raise ValueError("bridge removal destroyed the main component")
        comp = _union_find_labels(gr.n_nodes, tail, head, edge_on)
        edge_on = edge_on & (comp[tail] == found[0])
        port_on = comp[pn] == found[0]
        if verbose:
            print(
                f"  prune: rho_cut={cut:.2f} -> flow-capable main comp "
                f"{int(edge_on.sum())} edges  port_share={score:.3f}  "
                f"coverage {best_cov}/{len(blk_edge_ids)}"
            )
    else:
        # --- round 1+: 使われなかった流路を実流量で落とす ---
        q0 = st["q"].numpy()
        edge_on = mask.edge_on & (np.abs(q0) > q_cut_rel * prob.q_total)
        found = _main_component(prob, edge_on, yi, yo)
        if found is not None:
            comp = _union_find_labels(gr.n_nodes, tail, head, edge_on)
            edge_on = edge_on & (comp[tail] == found[0])
            port_on = comp[pn] == found[0]
        else:  # 全滅したら現状維持
            edge_on, port_on = mask.edge_on, mask.port_on
        if verbose:
            print(f"  prune: unused-channel removal -> {int(edge_on.sum())} edges")

    # --- block_on: 除熱が届くブロックの判定 ---
    m2 = DesignMask(edge_on, port_on, np.ones(len(prob.block_edges), dtype=bool))
    with torch.no_grad():
        st2 = state(prob, cfg, torch.tensor(x), m2)
    pb = _per_block(prob, st2["q_heat"]).numpy()
    block_on = pb > 1e-6 * max(float(pb.max()), 1e-30)
    if mask is None:
        # round 0 は位相的に流れ得るブロックも残す (再最適化が流れを配る余地)
        touch = np.array([bool(edge_on[ids].any()) for ids in blk_edge_ids])
        block_on = block_on | touch
    if verbose:
        print(f"  prune: blocks covered {int(block_on.sum())}/{len(block_on)}")
    return DesignMask(edge_on, port_on, block_on)


def solve_pipeline(
    prob: Problem,
    cfg: Config,
    log_p_max: float = 5.0,
    seed: int = 0,
    maxiter_stage: int = 400,
    maxiter_reopt: int = 400,
    n_prune_rounds: int = 3,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, DesignMask]:
    """continuation -> 刈り込み -> 再最適化 の全パイプライン.

    Returns
    -------
    (x_raw, x_final, mask):
        x_raw   continuation 後・刈り込み前の SIMP 解 (漏れ流れを含む)
        x_final 凍結マスク付き再最適化後の解
        mask    最終凍結マスク (edge_on の外の流量は厳密ゼロ)
    """
    # --- stage A: continuation (penal / beta / mu_bin を段階強化) ---
    stages = [(1.0, 4.0, 0.3), (2.0, 6.0, 1.0), (3.0, 8.0, 3.0)]
    x: np.ndarray | None = None
    for penal, beta, mu_bin in stages:
        cfg.penal, prob.beta, cfg.mu_bin = penal, beta, mu_bin
        x = optimise(prob, cfg, x0=x, maxiter=maxiter_stage, seed=seed, log_p_max=log_p_max)
        if verbose:
            r = report(prob, cfg, x)
            print(
                f"  stage penal={penal:.0f} beta={beta:.0f}: "
                f"cooling={r['cooling']:.3f} grey={r['grey']:.3f} "
                f"leak_frac={r['leak_flow_frac']:.3f}"
            )
    assert x is not None
    x_raw = x.copy()

    # --- stage B: prune -> reopt を繰り返す (逐次刈り込み) ---
    prob.beta = 12.0  # 刈り込み後は worst-block を直接押し上げる
    mask: DesignMask | None = None
    for rnd in range(n_prune_rounds):
        new_mask = prune(prob, cfg, x, mask, verbose=verbose)
        if new_mask.same_as(mask):
            break
        mask = new_mask
        x = optimise(prob, cfg, x0=x, maxiter=maxiter_reopt, log_p_max=log_p_max, mask=mask)
        if verbose:
            r = report(prob, cfg, x, mask)
            print(
                f"  round {rnd}: edges={int(mask.edge_on.sum())} "
                f"blocks={int(mask.block_on.sum())}/{len(mask.block_on)} "
                f"cooling={r['cooling']:.3f} block_min/mean={r['block_min_over_mean']:.3f}"
            )
    assert mask is not None
    return x_raw, x, mask


# ==========================================================================
# 評価・検証
# ==========================================================================
def report(
    prob: Problem, cfg: Config, x: np.ndarray, mask: DesignMask | None = None
) -> dict[str, object]:
    gr = prob.graph
    with torch.no_grad():
        st = state(prob, cfg, torch.tensor(x), mask)
        per_block_all = _per_block(prob, st["q_heat"])
        if mask is not None:
            per_block = per_block_all[torch.from_numpy(mask.block_on)]
        else:
            per_block = per_block_all
        logs = torch.log(per_block + 1e-14)
        worst = -torch.logsumexp(-prob.beta * logs, dim=0) / prob.beta

    r = st["rho"].numpy()
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    inblk = np.zeros(gr.n_edges, dtype=bool)
    for e in prob.block_edges:
        inblk[e.numpy()] = True
    qh = st["q_heat"].numpy()
    qn = st["q"].numpy()
    elen = gr.elen.numpy()

    # 漏れ流れ診断: rho < 0.5 (=存在しないはずの辺) が運ぶ流量·距離の割合
    weak = r < 0.5
    denom = float((np.abs(qn) * elen).sum())
    leak_frac = float((np.abs(qn[weak]) * elen[weak]).sum() / max(denom, 1e-30))
    # 凍結マスクの外の漏れ (v2 では厳密ゼロであるべき)
    leak_abs = float(np.abs(qn[~mask.edge_on]).sum()) if mask is not None else float("nan")
    # 節点質量残差
    with torch.no_grad():
        resid = (gr.div(st["q"]) - st["b"]).abs().max()

    pb = per_block.numpy()
    return {
        "cooling": float(worst),
        "log_diss": float(np.log(max(st["diss"].item(), 1e-12))),
        "n_on": int((r > 0.5).sum()),
        "grey": float(((r > 0.05) & (r < 0.95)).mean()),
        "w_mean": float(st["w"].numpy()[r > 0.5].mean()) if (r > 0.5).any() else float("nan"),
        "w_std": float(st["w"].numpy()[r > 0.5].std()) if (r > 0.5).any() else float("nan"),
        "phi_out": float((st["phi"] * torch.relu(-st["b"])).sum() / prob.q_total),
        "heat_in_block": float(qh[inblk].sum() / max(qh.sum(), 1e-30)),
        "eff_ports_in": float(1.0 / (yi**2).sum()),
        "eff_ports_out": float(1.0 / (yo**2).sum()),
        "block_min": float(pb.min()),
        "block_max": float(pb.max()),
        "block_cv": float(pb.std() / max(pb.mean(), 1e-30)),
        "block_min_over_mean": float(pb.min() / max(pb.mean(), 1e-30)),
        "blocks_covered": int(mask.block_on.sum()) if mask is not None else len(pb),
        "mass_resid_max": float(resid),
        "leak_flow_frac": leak_frac,
        "leak_flow_masked": leak_abs,
    }


def mass_check(
    prob: Problem, cfg: Config, x: np.ndarray, mask: DesignMask | None = None
) -> dict[str, float]:
    """質量保存の検証 (v2 新設).

    - resid_max:   max_i |div(q)_i - b_i|  (Kirchhoff 解の質量残差)
    - leak_inactive: 非アクティブ辺 (存在しないはずの流路) の流量合計。
      mask 付きなら厳密に 0 でなければならない。
      mask なし (素の SIMP) では rho<0.5 辺の流量で、v1 の偽解では O(0.1) 出る。
    """
    gr = prob.graph
    with torch.no_grad():
        st = state(prob, cfg, torch.tensor(x), mask)
        resid = float((gr.div(st["q"]) - st["b"]).abs().max())
    qn = st["q"].numpy()
    if mask is not None:
        leak = float(np.abs(qn[~mask.edge_on]).sum())
    else:
        leak = float(np.abs(qn[st["rho"].numpy() < 0.5]).sum())
    return {"resid_max": resid, "leak_inactive": leak}


def check_gradient(
    prob: Problem,
    cfg: Config,
    seed: int = 1,
    n_probe: int = 10,
    log_p_max: float | None = 5.0,
    mask: DesignMask | None = None,
    x0: np.ndarray | None = None,
) -> float:
    """autograd 勾配を中心差分で検証。1e-4 以下なら健全.

    AD と FD の両方がノイズ床 (線形ソルバの丸めに由来、目的値 O(1) と
    h=1e-6 に対して FD 分解能 ~1e-7) を下回る成分は「ゼロ勾配と整合」と
    みなしスキップする。さもないと感度が厳密ゼロに近い成分 (凍結辺の w 等) で
    ノイズ同士の比が O(1) となり偽陽性を出す。
    """
    rng = np.random.default_rng(seed)
    n = 2 * prob.graph.n_edges + 2 * len(prob.ports)
    xv = 0.3 * rng.standard_normal(n) if x0 is None else x0.copy()
    xt = torch.tensor(xv, requires_grad=True)
    (g_ad,) = torch.autograd.grad(objective(prob, cfg, xt, log_p_max, mask), xt)
    noise_floor = 1e-6 * (1.0 + float(g_ad.abs().max()))
    h, errs = 1e-6, []
    for i in rng.choice(n, size=n_probe, replace=False):
        xp, xm = xv.copy(), xv.copy()
        xp[i] += h
        xm[i] -= h
        fd = (
            objective(prob, cfg, torch.tensor(xp), log_p_max, mask).item()
            - objective(prob, cfg, torch.tensor(xm), log_p_max, mask).item()
        ) / (2 * h)
        if abs(fd) < noise_floor and abs(float(g_ad[i])) < noise_floor:
            continue  # ゼロ勾配成分: FD はソルバノイズしか見えない
        errs.append(abs(fd - float(g_ad[i])) / (abs(fd) + 1e-8))
    return float(np.max(errs)) if errs else 0.0


def connectivity_check(
    prob: Problem,
    cfg: Config,
    x: np.ndarray,
    rho_cut: float = 0.5,
    mask: DesignMask | None = None,
) -> dict[str, object]:
    """rho>rho_cut の辺だけで inlet 群と outlet 群が連結か判定する.

    散逸制約が緩いと SIMP は平然と断線し、rho_floor の数値的な辺を
    天文学的な Δp で流れる偽解を出す。必ず検査すること。
    """
    gr = prob.graph
    with torch.no_grad():
        st = state(prob, cfg, torch.tensor(x), mask)
    keep = st["rho"].numpy() > rho_cut
    comp = _union_find_labels(gr.n_nodes, gr.tail.numpy(), gr.head.numpy(), keep)

    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    pn = prob.ports.numpy()
    src = {int(comp[int(pn[k])]) for k in np.nonzero(yi > 0.02)[0]}
    snk = {int(comp[int(pn[k])]) for k in np.nonzero(yo > 0.02)[0]}
    return {
        "connected": bool(src & snk) and len(src | snk) > 0 and len(src) == 1 and len(snk) == 1,
        "n_src_components": len(src),
        "n_snk_components": len(snk),
        "n_kept_edges": int(keep.sum()),
    }


# ==========================================================================
# 問題生成
# ==========================================================================
def make_problem(
    n_cols: int = 4,
    n_rows: int = 8,
    block_pitch: int = 3,
    margin_left: int = 4,
    margin_right: int = 3,
    margin_y: int = 3,
    **kw: float,
) -> Problem:
    """発熱体 n_cols × n_rows を中央に配置。各発熱体は 1 セル (2×2 節点).

    幾何の前提: 格子ピッチ = 流路ピッチ = d + 壁厚。
    発熱体幅 ≈ 2·d_min ≈ 1.2 セル なので 1 セルを占める。
    その 4 辺がその発熱体の下を通り得る流路。
    """
    span_x = (n_cols - 1) * block_pitch + 1
    span_y = (n_rows - 1) * block_pitch + 1
    nx = margin_left + span_x + margin_right + 1
    ny = 2 * margin_y + span_y + 1
    gr = GridGraph.build(nx, ny, diagonals=False)

    tail_np, head_np = gr.tail.numpy(), gr.head.numpy()
    block_edges: list[torch.Tensor] = []
    boxes: list[tuple[int, int, int, int]] = []
    for c in range(n_cols):
        x0 = margin_left + c * block_pitch
        for r in range(n_rows):
            y0 = margin_y + r * block_pitch
            inside = np.zeros(gr.n_nodes, dtype=bool)
            for j in (y0, y0 + 1):
                for i in (x0, x0 + 1):
                    inside[gr.nid(i, j)] = True
            block_edges.append(
                torch.tensor(np.nonzero(inside[tail_np] & inside[head_np])[0], dtype=torch.long)
            )
            boxes.append((x0, y0, 1, 1))

    return Problem(
        graph=gr,
        ports=torch.tensor([gr.nid(0, j) for j in range(ny)], dtype=torch.long),
        block_edges=block_edges,
        block_boxes=boxes,
        **kw,
    )


# ==========================================================================
# 可視化 (実スケール厳守)
# ==========================================================================
def plot(
    prob: Problem,
    cfg: Config,
    x: np.ndarray,
    fig,
    ax,
    title: str = "",
    rho_cut: float = 0.35,
    mask: DesignMask | None = None,
) -> None:
    """流路幅をデータ座標に忠実に描く.

    lw をデータ座標と無関係な係数で決めると幾何の破綻が見えなくなる。
    lw [pt] = w [セル] × (1 セルあたり pt 数) を厳守すること。
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    gr = prob.graph
    with torch.no_grad():
        st = state(prob, cfg, torch.tensor(x), mask)
    xy = gr.coords()
    rn, wn, phin = st["rho"].numpy(), st["w"].numpy(), st["phi"].numpy()
    cmap = plt.get_cmap("turbo")

    ax.set_facecolor("#0e1015")
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, gr.nx)
    ax.set_ylim(-1.2, gr.ny)
    ax.axis("off")
    fig.canvas.draw()
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((1, 0))
    ppc = float(np.hypot(*(p1 - p0))) * 72 / fig.dpi  # points per cell

    for x0, y0, bw, bh in prob.block_boxes:
        ax.add_patch(
            mpatches.Rectangle(
                (x0 - 0.1, y0 - 0.1),
                bw + 0.2,
                bh + 0.2,
                fc="none",
                ec="#ff3b30",
                lw=1.1,
                zorder=5,
            )
        )
    draw = np.nonzero(mask.edge_on if mask is not None else (rn > rho_cut))[0]
    for e in draw:
        a, b = int(gr.tail[e]), int(gr.head[e])
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            lw=wn[e] * ppc,
            alpha=float(min(1.0, rn[e])),
            color=cmap(0.5 * (phin[a] + phin[b])),
            solid_capstyle="round",
            zorder=2,
        )

    pn = prob.ports.numpy()
    yi, yo = st["w_in"].numpy(), st["w_out"].numpy()
    for k, nd in enumerate(pn):
        if yi[k] > 0.01:
            ax.scatter(
                xy[nd, 0] - 1.1,
                xy[nd, 1],
                s=40 + 260 * yi[k],
                c="#00e5ff",
                marker="o",
                zorder=6,
            )
        if yo[k] > 0.01:
            ax.scatter(
                xy[nd, 0] - 1.1,
                xy[nd, 1],
                s=40 + 260 * yo[k],
                c="#ffffff",
                marker="X",
                zorder=6,
            )
    ax.set_title(title, color="w", fontsize=10)


# ==========================================================================
if __name__ == "__main__":
    import time
    from pathlib import Path

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    prob = make_problem(ntu_coef=1e-2)
    cfg = Config(vol_frac=0.25, mu_conc=0.0)
    gr = prob.graph
    print(
        f"grid {gr.nx}x{gr.ny}  nodes={gr.n_nodes}  edges={gr.n_edges}  "
        f"blocks={len(prob.block_edges)}  ports={len(prob.ports)}"
    )
    print(f"gradient check (max rel err) = {check_gradient(prob, cfg):.2e}")

    t0 = time.perf_counter()
    x_raw, x_fin, mask = solve_pipeline(prob, cfg, log_p_max=3.5, verbose=True)
    print(f"pipeline done in {time.perf_counter() - t0:.0f}s")

    print(f"gradient check (masked)      = {check_gradient(prob, cfg, mask=mask, n_probe=40):.2e}")

    print("\n--- raw SIMP solution (before prune; v1 相当) ---")
    cfg_raw = Config(vol_frac=cfg.vol_frac, penal=3.0, mu_bin=3.0)  # stage A 最終段相当
    for k, v in report(prob, cfg_raw, x_raw).items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    print(f"  mass_check           {mass_check(prob, cfg_raw, x_raw)}")
    print(f"  connectivity         {connectivity_check(prob, cfg_raw, x_raw)}")

    print("\n--- pruned + re-optimised solution (v2) ---")
    for k, v in report(prob, cfg, x_fin, mask).items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    mc = mass_check(prob, cfg, x_fin, mask)
    print(f"  mass_check           {mc}")
    cc = connectivity_check(prob, cfg, x_fin, mask=mask)
    print(f"  connectivity         {cc}")
    assert cc["connected"], "final design must be connected"
    assert mc["leak_inactive"] == 0.0, "inactive edges must carry exactly zero flow"
    assert mc["resid_max"] < 1e-8 * prob.q_total, "nodal mass residual too large"
    print("\nP0 checks passed: connected, zero leak on inactive edges, mass conserved.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 9), facecolor="#0e1015")
    plot(prob, cfg_raw, x_raw, fig, axes[0], title="(a) raw SIMP (leaky, v1)")
    plot(prob, cfg, x_fin, fig, axes[1], title="(b) pruned + re-optimised (v2)", mask=mask)
    fig.suptitle(
        "marker size = port flow share | color = freshness phi | TRUE SCALE",
        color="w",
        fontsize=11,
    )
    fig.savefig(out_dir / "coldplate_v2.png", dpi=110, bbox_inches="tight")
    print(f"saved: {out_dir / 'coldplate_v2.png'}")

    np.savez(
        out_dir / "coldplate_v2_result.npz",
        x_raw=x_raw,
        x_final=x_fin,
        edge_on=mask.edge_on,
        port_on=mask.port_on,
        block_on=mask.block_on,
    )
    print(f"saved: {out_dir / 'coldplate_v2_result.npz'}")
