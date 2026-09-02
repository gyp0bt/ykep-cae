"""単軸押出 展開チャネル 2.5D のデータ契約.

座標系:
  x  横断方向（フライトに直交）。x=0 と x=W_t は周期。+x が下流側の隣チャネル
  y  深さ。y=0 スクリュー根元、y=H バレル
  z  下流方向（フライトに沿う）。完全発達を仮定し ∂/∂z = 0

幾何恒等式（docs/design/single-screw-extruder.md §2.1.1）:
  W_t    = πD·sinφ      チャネル 1 ピッチのフライト直交幅
  L_turn = πD·cosφ      隣チャネルまでの下流距離
  W_t / L_turn = tanφ           （D にもリードにも依らない）
  β = G·L_turn/W_t = G·cotφ     横断方向の一様圧力勾配（同上）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae_fluid.core.data import MeshData


@dataclass(frozen=True)
class ScrewSpec:
    """スクリュー諸元と格子解像度.

    Parameters
    ----------
    D : float
        バレル内径 [m]
    lead : float
        リード（1 回転あたりの軸方向前進量）[m]
    H : float
        計量部チャネル深さ [m]
    e : float
        フライト幅（フライト直交方向）[m]
    delta : float
        フライト隙間 [m]。0.0 で閉チャネル（G1/G2 用）
    N : float
        回転数 [1/s]。rpm ではないことに注意
    nx_channel : int
        チャネル部（フライト以外）の x 方向セル数
    nx_land : int
        フライト頂部（ランド）の x 方向セル数
    ny_bulk : int
        隙間より下のバルク部の y 方向セル数
    n_gap : int
        隙間 delta の中に入れる y 方向セル数。delta=0 なら無視される
    """

    D: float
    lead: float
    H: float
    e: float
    delta: float
    N: float
    nx_channel: int = 200
    nx_land: int = 48
    ny_bulk: int = 60
    n_gap: int = 20

    @property
    def phi(self) -> float:
        """リード角 [rad]. tanφ = lead / (πD)."""
        return math.atan(self.lead / (math.pi * self.D))

    @property
    def W_t(self) -> float:
        """チャネル 1 ピッチのフライト直交幅 W_t = πD·sinφ [m]."""
        return math.pi * self.D * math.sin(self.phi)

    @property
    def W(self) -> float:
        """チャネル幅（フライトを除く）[m]."""
        return self.W_t - self.e

    @property
    def L_turn(self) -> float:
        """隣チャネルまでの下流距離 L_turn = πD·cosφ [m]."""
        return math.pi * self.D * math.cos(self.phi)

    @property
    def V(self) -> float:
        """バレルの相対周速 V = πDN [m/s]."""
        return math.pi * self.D * self.N

    @property
    def u_barrel(self) -> float:
        """バレルの横断方向速度 [m/s]. 負（-x = 上流側）."""
        return -self.V * math.sin(self.phi)

    @property
    def w_barrel(self) -> float:
        """バレルの下流方向速度 [m/s]. 正（下流向き）."""
        return self.V * math.cos(self.phi)

    def beta(self, G: float) -> float:
        """横断方向の一様圧力勾配 β = dP/dx = G·cotφ [Pa/m].

        断面内運動量には体積力 f_x = -β として入る。
        """
        return G / math.tan(self.phi)


@dataclass(frozen=True)
class ChannelGrid:
    """展開チャネル断面の不等間隔格子.

    Parameters
    ----------
    dx, dy : np.ndarray
        セル幅 (nx,), (ny,) [m]
    xc, yc : np.ndarray
        セル中心座標 (nx,), (ny,) [m]
    solid : np.ndarray
        (nx, ny) bool。True = フライト（固体）
    spec : ScrewSpec
        元の諸元
    mesh : MeshData
        StructuredMeshProcess が生成した MeshData（来歴保持用）
    """

    dx: np.ndarray
    dy: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    solid: np.ndarray
    spec: ScrewSpec
    mesh: MeshData

    @property
    def nx(self) -> int:
        return int(self.dx.shape[0])

    @property
    def ny(self) -> int:
        return int(self.dy.shape[0])

    @property
    def area_free(self) -> float:
        """流体セルの断面積和 [m²]."""
        cell = self.dx[:, None] * self.dy[None, :]
        return float(cell[~self.solid].sum())


@dataclass(frozen=True)
class DownChannelInput:
    """下流方向流れ w の入力.

    Parameters
    ----------
    grid : ChannelGrid
        断面格子
    mu : np.ndarray
        (nx, ny) 粘度場 [Pa·s]。ニュートンなら定数配列
    G : float
        下流方向圧力勾配 dp/dz [Pa/m]。押出（背圧あり）は正
    """

    grid: ChannelGrid
    mu: np.ndarray
    G: float


@dataclass(frozen=True)
class DownChannelResult:
    """下流方向流れ w の結果.

    Parameters
    ----------
    w : np.ndarray
        (nx, ny) 下流方向速度 [m/s]。固体セルは 0
    Q : float
        体積流量 [m³/s]（断面積分 ∫∫ w dx dy）
    """

    w: np.ndarray
    Q: float


@dataclass(frozen=True)
class CrossChannelInput:
    """断面内 Stokes の入力.

    Parameters
    ----------
    grid : ChannelGrid
        断面格子
    mu : np.ndarray
        (nx, ny) 粘度場 [Pa·s]
    G : float
        下流方向圧力勾配 [Pa/m]。横断方向体積力は f_x = −spec.beta(G) = −G·cotφ
    p_pin_value : float
        圧力の定数自由度を消すためのピン留め値 [Pa]
    """

    grid: ChannelGrid
    mu: np.ndarray
    G: float
    p_pin_value: float = 0.0


@dataclass(frozen=True)
class CrossChannelResult:
    """断面内 Stokes の結果.

    Parameters
    ----------
    u, v : np.ndarray
        (nx, ny) セル中心速度 [m/s]。固体セルは 0
    u_face : np.ndarray
        (nx, ny) x 面の u [m/s]。面 i はセル i の西面（周期なので nx 枚）
    v_face : np.ndarray
        (nx, ny+1) y 面の v [m/s]。面 j はセル j の南面
    p : np.ndarray
        (nx, ny) 圧力の周期部分 p̃ [Pa]
    psi : np.ndarray
        (nx+1, ny+1) 節点上の流れ関数 [m²/s]。面流束を積分して作るので
        離散的に厳密に発散ゼロ。粒子追跡はこれを使う
    div_max : float
        セル発散の最大値を代表せん断速度 |u_barrel|/H で規格化した無次元量
    psi_periodicity : float
        psi[nx,:] と psi[0,:] のずれを |u_barrel|·H で規格化した無次元量。
        0 でなければ質量保存が破れている
    """

    u: np.ndarray
    v: np.ndarray
    u_face: np.ndarray
    v_face: np.ndarray
    p: np.ndarray
    psi: np.ndarray
    div_max: float
    psi_periodicity: float


@dataclass(frozen=True)
class ExtruderFlowInput:
    """押出流れ解析の入力.

    Parameters
    ----------
    spec : ScrewSpec
        スクリュー諸元と格子解像度
    G : float
        下流方向圧力勾配 dp/dz [Pa/m]。押出（背圧あり）は正
    max_iter : int
        Picard 反復の上限
    tol : float
        粘度場の相対変化に対する収束閾値
    relax_mu : float
        粘度の緩和係数 ω。μ^{k+1} = (1−ω)μ^k + ω·μ(γ̇^k)
    """

    spec: ScrewSpec
    G: float
    max_iter: int = 100
    tol: float = 1.0e-6
    relax_mu: float = 0.5


@dataclass(frozen=True)
class ExtruderFlowResult:
    """押出流れ解析の結果.

    Parameters
    ----------
    grid : ChannelGrid
        使用した断面格子
    u, v, w : np.ndarray
        (nx, ny) セル中心の速度 3 成分 [m/s]
    u_face : np.ndarray
        (nx, ny) x 面の u [m/s]
    v_face : np.ndarray
        (nx, ny+1) y 面の v [m/s]
    psi : np.ndarray
        (nx+1, ny+1) 節点流れ関数 [m²/s]
    p : np.ndarray
        (nx, ny) 圧力の周期部分 [Pa]
    mu : np.ndarray
        (nx, ny) 収束した粘度場 [Pa·s]
    gamma_dot : np.ndarray
        (nx, ny) せん断速度 [1/s]
    Q : float
        下流方向の体積流量 ∫∫w dA [m³/s]。**押出量ではない**（Q_axial を見ること）
    Q_leak : float
        フライトランド中央を通る正味横断流束 [m²/s]。負が上流への漏れ。
        断面内は 2D 非圧縮なのでどの x 面を取っても同じ値になる
    Q_axial : float
        押出量（軸方向の正味体積流量）[m³/s]。Q_axial = Q + L_turn·Q_leak。
        隙間がなければ Q_leak=0 で Q と一致する
    converged : bool
        粘度場の Picard 反復が収束したか
    n_iter : int
        Picard 反復回数
    mu_history : tuple[float, ...]
        各反復における粘度場の相対変化
    div_max : float
        断面内の最大セル発散（|u_barrel|/H で規格化）
    elapsed_seconds : float
        計算時間 [s]
    """

    grid: ChannelGrid
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    u_face: np.ndarray
    v_face: np.ndarray
    psi: np.ndarray
    p: np.ndarray
    mu: np.ndarray
    gamma_dot: np.ndarray
    Q: float
    Q_leak: float
    Q_axial: float
    converged: bool
    n_iter: int
    mu_history: tuple[float, ...] = ()
    div_max: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ParticleTrackInput:
    """粒子追跡の入力.

    Parameters
    ----------
    flow : ExtruderFlowResult
        収束済みの流れ場
    z_axial : float
        計量部の軸方向長さ [m]。粒子は ζ >= z_axial で脱出したとみなす
    stride : int
        種まきセルの間引き。1 で全流体セル、2 で 1 つおき
    cfl : float
        時間刻みの安全率。dt = cfl·min(dx/|u|, dy/|v|)。
        実測では 1.0 が最良（0.5 だと外挿に回る粒子が増えて逆に精度が落ち、
        2.0 だと時間積分誤差が 0.8% 出る）
    max_steps : int
        1 粒子あたりの最大ステップ数。超えた粒子は軸方向の進行率から外挿して
        閉じる（extrapolated=True）
    """

    flow: ExtruderFlowResult
    z_axial: float
    stride: int = 1
    cfl: float = 1.0
    max_steps: int = 50_000


@dataclass(frozen=True)
class ParticleTrackResult:
    """粒子追跡の結果（全て (n_particles,) の配列）.

    Parameters
    ----------
    weight : np.ndarray
        流束重み [m³/s]。ζ=0 面を通る体積流束。総和が Q_axial に一致する
    t_res : np.ndarray
        滞留時間 [s]
    gamma_total : np.ndarray
        累積せん断ひずみ ∫γ̇ dt [-]
    lambda_mean : np.ndarray
        経路に沿った混合指数の時間平均 [-]
    n_wraps : np.ndarray
        x 周期を跨いだ回数（正 = 下流側へ、負 = 上流側へ）
    x0, y0 : np.ndarray
        初期位置 [m]
    x, y : np.ndarray
        最終位置 [m]
    escaped : np.ndarray
        ζ >= z_axial に到達したか（bool）。外挿で閉じたものも True
    extrapolated : np.ndarray
        ステップ上限に達したので軸方向進行率から外挿したか（bool）。
        バレル直下・根元直上の境界層は軸方向速度が 0 に漸近するため
        滞留時間が発散し、有限ステップでは閉じない
    n_steps : np.ndarray
        使用したステップ数
    """

    weight: np.ndarray
    t_res: np.ndarray
    gamma_total: np.ndarray
    lambda_mean: np.ndarray
    n_wraps: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x: np.ndarray
    y: np.ndarray
    escaped: np.ndarray
    extrapolated: np.ndarray
    n_steps: np.ndarray


@dataclass(frozen=True)
class RTDInput:
    """滞留時間分布の集計入力.

    Parameters
    ----------
    track : ParticleTrackResult
        粒子追跡の結果
    flow : ExtruderFlowResult
        流れ場（理論値 ⟨t⟩ = z_axial·A_free/(sinφ·Q) の照合に使う）
    z_axial : float
        追跡に使った軸方向長さ [m]
    n_bins : int
        E(t) のヒストグラム区間数
    """

    track: ParticleTrackResult
    flow: ExtruderFlowResult
    z_axial: float
    n_bins: int = 200


@dataclass(frozen=True)
class RTDResult:
    """滞留時間分布と混練性指標.

    Parameters
    ----------
    t_edges : np.ndarray
        (n_bins+1,) ヒストグラムの区間端 [s]
    E : np.ndarray
        (n_bins,) 滞留時間の確率密度 [1/s]。∫E dt = 1
    F : np.ndarray
        (n_bins+1,) 累積分布 [-]。F(t_edges[0])=0, F(t_edges[-1])=1
    t_mean : float
        平均滞留時間 [s]（流束重み付き）
    t_mean_theory : float
        理論値 z_axial·A_free/(sinφ·Σweight) [s]。厳密関係
    t_min, t_p10, t_p50, t_p90 : float
        最短・10/50/90 パーセンタイル滞留時間 [s]
    spread : float
        t_p90 / t_p10。分布の広がり（1 に近いほど揃った履歴）
    gamma_mean, gamma_p10, gamma_p50, gamma_p90 : float
        累積せん断ひずみの平均とパーセンタイル [-]
    lambda_mean : float
        混合指数の流束重み付き平均 [-]
    extrapolated_weight_fraction : float
        外挿で閉じた粒子の重み割合 [-]（境界層の長い裾）
    unresolved_weight_fraction : float
        脱出も外挿もできなかった粒子の重み割合 [-]
    """

    t_edges: np.ndarray
    E: np.ndarray
    F: np.ndarray
    t_mean: float
    t_mean_theory: float
    t_min: float
    t_p10: float
    t_p50: float
    t_p90: float
    spread: float
    gamma_mean: float
    gamma_p10: float
    gamma_p50: float
    gamma_p90: float
    lambda_mean: float
    extrapolated_weight_fraction: float
    unresolved_weight_fraction: float
