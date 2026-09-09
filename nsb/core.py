"""型宣言: 境界条件・入力・設定・結果."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nsb.data import (
    BoundaryKind,
    BoundaryPatch,
    BrinkmanFlowInput,
    BrinkmanGeometry,
    ConvectionSchemeType,
    MaskFn,
    WeightFn,
)

FaceType = (
    BoundaryKind  # 互換エイリアス（WALL / VELOCITY_INLET / MASS_FLOW_INLET / PRESSURE_OUTLET）
)


@dataclass(frozen=True)
class BC:
    """境界条件: 座標マスク関数で指定する境界パッチの列.

    各パッチは mask(x, y) -> bool を領域 4 辺の境界面中心で評価して面を選ぶ。
    どのパッチにも属さない面は WALL。後のパッチが優先。

    Parameters
    ----------
    patches : tuple[BoundaryPatch, ...]
        境界パッチ
    """

    patches: tuple[BoundaryPatch, ...]

    @staticmethod
    def velocity_inlet(mask: MaskFn, u_in: float, name: str = "inlet") -> BoundaryPatch:
        return BoundaryPatch(BoundaryKind.VELOCITY_INLET, mask, velocity=u_in, name=name)

    @staticmethod
    def mass_flow_inlet(mask: MaskFn, mass_flow: float, name: str = "inlet") -> BoundaryPatch:
        """質量流量 [kg/s]（厚さ込み 3 次元値）指定の inlet."""
        return BoundaryPatch(BoundaryKind.MASS_FLOW_INLET, mask, mass_flow=mass_flow, name=name)

    @staticmethod
    def pressure_outlet(mask: MaskFn, p: float = 0.0, name: str = "outlet") -> BoundaryPatch:
        return BoundaryPatch(BoundaryKind.PRESSURE_OUTLET, mask, pressure=p, name=name)

    # --- 領域内マニホールド（紙面垂直方向）: マスク/重みはセル中心で評価 ---
    @staticmethod
    def interior_source(
        mask: MaskFn | None,
        mass_flow: float,
        name: str = "manifold_in",
        weight: WeightFn | None = None,
    ) -> BoundaryPatch:
        """流量指定の注入マニホールド [kg/s]（面内運動量ゼロで注入）。weight で滑らかな窓も可."""
        return BoundaryPatch(
            BoundaryKind.INTERIOR_MASS_SOURCE, mask, mass_flow=mass_flow, weight=weight, name=name
        )

    @staticmethod
    def interior_sink(
        mask: MaskFn | None,
        mass_flow: float,
        name: str = "manifold_out",
        weight: WeightFn | None = None,
    ) -> BoundaryPatch:
        """流量指定の吸出マニホールド [kg/s]（局所運動量を持ち出す）。圧力基準が別に必要."""
        return BoundaryPatch(
            BoundaryKind.INTERIOR_MASS_SINK, mask, mass_flow=mass_flow, weight=weight, name=name
        )

    @staticmethod
    def interior_pressure_sink(
        mask: MaskFn | None,
        conductance: float,
        p: float = 0.0,
        name: str = "manifold_p",
        weight: WeightFn | None = None,
    ) -> BoundaryPatch:
        """圧力指定マニホールド: q = conductance (p - p_manifold) [kg/s]。圧力基準を与える."""
        return BoundaryPatch(
            BoundaryKind.INTERIOR_PRESSURE_SINK,
            mask,
            conductance=conductance,
            pressure=p,
            weight=weight,
            name=name,
        )

    @property
    def u_inlet(self) -> float:
        """VELOCITY_INLET の最大流速（MASS_FLOW_INLET のみの場合は 0。速度スケールは離散化側で決まる）."""
        return max(
            (p.velocity for p in self.patches if p.kind is BoundaryKind.VELOCITY_INLET),
            default=0.0,
        )


@dataclass(frozen=True)
class NSBSettings:
    """Newton + 擬似時間の制御則.

    status-40 で「実験で一方が常に劣ると分かった切替」（静止場発進、LU 直接・defect correction、
    運動量 Jacobi、CFL backtracking、速度下限なし、numpy 残差、SA 階層の毎回構築）を落とし、
    数値パラメータと本当に一長一短の切替だけにした。参照場（収束判定の基準 r0 と初期 CFL）は
    常に Stokes–Brinkman 解で、初期場は `NSBInput.u0/v0/p0` があればそれ、無ければ Stokes 解。

    Parameters
    ----------
    convection : str
        残差の対流スキーム "sou"（2 次風上 + Venkatakrishnan）/ "fou"（1 次風上）。
        ヤコビアンは常に 1 次風上
    venkat_k : float
        Venkatakrishnan 定数 K
    linear_solver : str
        "jfnk_simple"（既定。有限差分 J v を FGMRES、SIMPLE 型ブロック前処理 `nsb.precond`: 運動量 ILU +
        Schur 補元 SA-AMG。大格子で速く pyamg が要る）/ "jfnk"（同 FGMRES、PARDISO 疎 LU(J1) 前処理。
        GMRES 反復は少なく頑健だが三角解が 1 スレッドで大格子では遅い）
    jacobian : str
        前処理に組む行列。"fou"（既定。1 次風上・RC 係数凍結の J1）/ "fd"（残差関数を色分け有限差分した
        厳密ヤコビアン `nsb.fdjac.colored_fd_jacobian`。半径 fd_jacobian_radius のボックスで 3(2r+1)² 回の
        残差評価。高 Re で J1 の速度ブロックが真の作用素と O(1) ずれて GMRES が壊れる件の対策、status-45）。
        **"fd" は "jfnk"（PARDISO LU）と組むこと**: SIMPLE 型前処理（"jfnk_simple"）は J1 の構造（M 行列の運動量
        ブロック + 凍結 RC の Schur 補元）を前提にしており、厳密ヤコビアンから組むと蛇行流路 0.0015 kg/s の初手で
        GMRES 200 反復・真の残差比 5〜170 で棄却され、J1 + SIMPLE の 22 反復が 58 反復以上に悪化した（status-46）
    pseudo_compressibility : float
        圧力の擬似時間項（人工圧縮性）β。0 で無効（既定）。連続式に c_p (p − p_prev)、
        c_p = τ_cell / (ρ (β u_scale)²) を加える（`pseudo_time_in_residual` に従い残差にも入れる）。
        τ は速度しか凍らせないので、Stokes 出発点が NS 解から遠い高 Re の蛇行流路では Newton ステップが
        「速度を凍らせたまま圧力だけで対流の不釣り合いを釣り合わせる」静水圧的な解（レベルシフト 1e4〜1e5 Pa）
        になり、CFL を下げるほど悪化する。β を入れると圧力レベルも減衰し、閉塞セルの圧力の谷も埋まる（status-45）
    line_search_halvings : int
        定常残差のラインサーチ: 修正量 δ を α = 1, 1/2, …, 1/2^k で試し、|R_steady| が減る最初の α を採る。
        全て失敗なら修正量を捨てて CFL を ser_shrink 倍にする。0 で無効（既定）。蛇行流路では残差が折れ点
        （風上切替・sink の clip）で Newton 方向に沿って単調でなく、棄却が連鎖して CFL が 1e-30 まで潰れる
        （status-46: 内部ポート継続法 0.005 段、ラインサーチ有 120 反復未達 / 無 69 反復収束）ので参照ケース以外では非推奨。
        Stokes 出発点が遠い高 Re 蛇行流路では cfl 0.25 のステップで定常残差が 56 倍になり、擬似時間残差だけを
        見る SER がそれを受理して偽収束（|R_τ| → 0、|R| は 2000 倍）に入る対策（status-45）
    limiter_freeze_rel : float
        [リミター凍結] 定常残差比 |R|/|R_ref| がこれを下回り、かつ Venkatakrishnan ψ が動いていない
        （|Δψ| > limiter_freeze_delta のセルが limiter_freeze_stable_frac 未満）ときに ψ を凍結する。
        凍結後は対流項が φ に線形になり Newton が折れ点（面ごとの min・クリップ・d_max/d_min の分岐）に
        引っかからなくなる。0 で無効（既定）。序盤（ψ ≈ 1 の Stokes 場）で凍結すると柵の無い外挿になって
        発散するので残差条件と ψ 安定条件の両方を要求する（nsbp status-42、gyp さんの手元の知見）。
        収束後は凍結を解いた真の残差も報告する（`NSBResult.residual_unfrozen`）
    limiter_freeze_delta, limiter_freeze_stable_frac : float
        上記の「動いていない」判定（既定 0.1 / 0.01）
    limiter_unfreeze_ratio, limiter_unfreeze_patience : float, int
        凍結後に定常残差が凍結以降の最小値の ratio 倍を超える状態が patience 反復続いたら解凍する
        （既定 3 / 3。1 反復で解凍すると跳ねのたびに前処理と CFL がリセットされ、蛇行流路の内部ポートで
        凍結↔解凍を 5 往復して 0.005 kg/s が 120 反復でも未収束だった）
    limiter_refreeze_max : int
        凍結問題が収束したあと解凍した真の残差が判定を満たさないとき、現在の ψ で凍結し直す回数の上限
        （ψ の Picard 反復）。0（既定）なら凍結問題の収束をそのまま収束とし、解凍した真の残差は
        `NSBResult.residual_unfrozen` に報告するだけ。凍結時点の ψ は解の ψ と違うので凍結問題の解の
        真の残差は 1e-4〜1e-3 に留まり、Picard 反復は 1 回あたり 0.78 倍（uturn r1 U=1）しか縮まない
        （ψ の場依存が O(1) で、それを Newton に入れるかどうかの差がそのまま出る）ので既定では回さない
    fd_jacobian_radius : int
        "fd" のステンシル半径（2 次風上 + リミター + RC で 2。3 と 4 で非零パターン・値とも同一を確認、status-45）
    precond_lag : int
        前処理（LU(J1) または SIMPLE 型）の遅延更新: 1 回の組立を最大この回数の Newton 反復で
        使い回す。1 で毎反復組立。GMRES が収束しなかったら即組み直して解き直す
    precond_refresh_gmres : int
        直前の GMRES 反復数がこれを超えたら次の Newton 反復で前処理を組み直す（古くなった兆候）
    precond_cfl_ratio : float
        組立時の CFL から現在の CFL がこの倍率以上変わったら組み直す。擬似時間対角 ρV/Δτ が
        CFL に反比例するので、SER で CFL が伸びる局面では前処理の対角が過大になり GMRES が
        遅くなる（status-32: 倍率無制限だと GMRES 反復 +70%）。0 以下で無効
    simple_schur_cycles : int
        SIMPLE 型前処理の Schur 補元に当てる AMG V サイクル数（既定 1。2 で反復数 −2〜20%、適用 +40%）
    simple_ilu_drop_tol, simple_ilu_fill_factor : float
        運動量 ILU（scipy `spilu`）の drop_tol / fill_factor。零ピボットなら drop_tol 1/10・fill 2 倍で
        最大 3 回組み直す（1e-2 / 1.5 は 288×192 で零ピボット・発散した実績があり、1e-3 / 3.0 を既定にする）
    cfl_init, cfl_max, ser_growth, ser_shrink : float
        擬似時間 CFL の初期値（Stokes 参照場に対する値。初期場が参照場より良ければ
        cfl_init·|R_ref|/|R_init| から出発する）・上限・SER の 1 反復あたり成長率上限・
        残差が増えたときの 1 反復あたり減少率下限（cfl *= clip(|R_prev|/|R_new|, ser_shrink, ser_growth)）。
        cfl_init は 0.5 → 0.25 に下げた（status-41: 良い初期場から出発すると CFL 6 以上で始まり、
        流れ場が形成途中の CFL 10〜40 で線形解が崩れて発散する走行があった）
    reject_lin_ratio : float
        線形解（再試行後）の真の残差比 |b − A δ|/|b| がこれを超えたら、その修正量を捨てて CFL を
        ser_shrink 倍にし同じ場からやり直す（0 で無効）。JFNK の差分ノイズで「未収束」判定が常に立つ
        （残差比 1e-3 程度）のとは別に、GMRES が実質失敗した（残差比 0.3 超）ごみステップだけを弾く。
        これがないと uturn 144×96 U=1/2 で残差が 1 ステップで 1e9 倍に跳ねて回復しない（status-41）
    local_dtau : bool
        True: セル局所 Δτ、False: 局所 Δτ の全セル最小値を一律に使う（大域 Δτ は同じ CFL で
        減衰が約 10 倍強く高 CFL に寛容だが収束は遅い）
    velocity_floor_ratio : float
        Δτ の速度スケール下限 = velocity_floor_ratio × 最大流入速度（`BrinkmanDiscretization.u_scale`）。
        下限なし（0）だと静止・低速セルで Δτ→∞ となり Newton が素になって停滞する（status-30）
    pseudo_time_in_residual : bool
        True: 残差にも ρV(u - u_prev)/Δτ を加える（dual-time 型）。収束判定・SER も
        その残差で行う。False（既定、status-46 で切替）: 対角補強のみ（残差は Δτ 非依存）で、
        収束判定・SER は定常残差で行う。τ の 2 重カウントを直して作用素 J+τ と前処理 J1+τ を揃えた（status-45）
        結果、擬似時間残差は各 Newton ステップでほぼゼロになり、それを見る SER は「進んだ」と誤認して CFL を
        育てず定常残差が這う（flat r1 U=1: 80 反復で定常 rel 9e-3 のまま |R_τ| 3e-6。定常残差 SER なら 15 反復）。
        True は非定常（`nsb.unsteady`）向けの残差形で、定常解法では偽収束の元
    sub_iters : int
        1 擬似時間ステップあたりの Newton 反復数（u_prev を凍結）。1 で通常の擬似時間 Newton
    rc_with_pseudo_time : bool
        Rhie–Chow 係数を d_f = V/(a_P + ρV/Δτ) にする
    alpha_u : float
        陰的緩和（運動量対角を a_P/α_u）。1.0（既定）で無し。速度下限ありなら緩和なしが最速
    newton_tol, newton_max_iter : float, int
        相対残差 |R|/|R(Stokes 場)| の収束判定と反復上限（擬似時間ステップ数 × sub_iters が上限）
    gmres_tol, gmres_restart, gmres_maxiter : float, int, int
        右前処理 FGMRES（`nsb.krylov.fgmres`）の相対許容・再出発次元・再出発回数。指定した rtol に届いた
        ところで止まる（scipy `gmres` は内部で許容を締めて 1e-2 指定でも 2e-3 まで解いていた）
    divergence_ratio : float
        ||R||/||R_ref|| がこれを超えたら発散停止
    """

    convection: str = "sou"
    venkat_k: float = 5.0
    linear_solver: str = "jfnk_simple"
    jacobian: str = "fou"
    fd_jacobian_radius: int = 2
    pseudo_compressibility: float = 0.0
    line_search_halvings: int = 0
    limiter_freeze_rel: float = 0.0
    limiter_freeze_delta: float = 0.1
    limiter_freeze_stable_frac: float = 0.01
    limiter_unfreeze_ratio: float = 3.0
    limiter_unfreeze_patience: int = 3
    limiter_refreeze_max: int = 0
    cfl_init: float = 0.25
    cfl_max: float = 1.0e6
    ser_growth: float = 2.0
    ser_shrink: float = 0.1
    reject_lin_ratio: float = 0.3
    local_dtau: bool = True
    velocity_floor_ratio: float = 0.1
    pseudo_time_in_residual: bool = False
    sub_iters: int = 1
    rc_with_pseudo_time: bool = False
    alpha_u: float = 1.0
    newton_tol: float = 1.0e-6
    newton_max_iter: int = 80
    gmres_tol: float = 1.0e-3
    gmres_restart: int = 40
    gmres_maxiter: int = 5
    precond_lag: int = 4
    precond_refresh_gmres: int = 30
    precond_cfl_ratio: float = 2.0
    simple_schur_cycles: int = 1
    simple_ilu_drop_tol: float = 1.0e-3
    simple_ilu_fill_factor: float = 3.0
    divergence_ratio: float = 1.0e6

    @property
    def scheme(self) -> ConvectionSchemeType:
        return {
            "sou": ConvectionSchemeType.SECOND_ORDER_UPWIND,
            "fou": ConvectionSchemeType.FIRST_ORDER_UPWIND,
        }[self.convection]


@dataclass(frozen=True)
class NSBInput:
    """ソルバー入力.

    Parameters
    ----------
    nx, ny : int
        分割数
    lx, ly : float
        領域サイズ [m]
    h : np.ndarray
        厚さ場 (nx, ny) [m]
    bc : BC
        境界条件
    rho, mu, mu_b : float
        密度・粘度・Brinkman 粘度
    settings : NSBSettings
        ソルバー設定
    u0, v0, p0 : np.ndarray | None
        初期場（None ならゼロ）
    friction_re_crit, friction_exponent, friction_blend : float
        [摩擦則] 隙間流れの Re 依存抗力（`BrinkmanFlowInput` と同じ。re_crit <= 0 で層流のみ）
    """

    nx: int
    ny: int
    lx: float
    ly: float
    h: np.ndarray
    bc: BC
    rho: float = 1000.0
    mu: float = 1.0e-3
    mu_b: float = 1.0e-3
    settings: NSBSettings = field(default_factory=NSBSettings)
    u0: np.ndarray | None = None
    v0: np.ndarray | None = None
    p0: np.ndarray | None = None
    friction_re_crit: float = 0.0
    friction_exponent: float = 0.75
    friction_blend: float = 4.0

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    def to_flow_input(self) -> BrinkmanFlowInput:
        """共有離散化（BrinkmanDiscretization）用の入力へ変換."""
        return BrinkmanFlowInput(
            nx=self.nx,
            ny=self.ny,
            thickness=self.h,
            geometry=BrinkmanGeometry(lx=self.lx, ly=self.ly),
            rho=self.rho,
            mu=self.mu,
            mu_brinkman=self.mu_b,
            brinkman_factor=12.0,
            friction_re_crit=self.friction_re_crit,
            friction_exponent=self.friction_exponent,
            friction_blend=self.friction_blend,
            u_inlet=self.bc.u_inlet,
            boundaries=self.bc.patches,
        )


@dataclass(frozen=True)
class NSBResult:
    """ソルバー結果.

    Parameters
    ----------
    u, v, p : np.ndarray
        セル中心の速度・圧力 (nx, ny)
    converged : bool
        収束フラグ
    failure_reason : str
        未収束理由（"" なら収束）
    n_iter : int
        実行した Newton 反復数（sub_iters 込み）
    residual_ref : float
        収束判定の基準 |R(Stokes 場)|（対流項込みの定常残差。Re→0 で 0 になる場合は静止場の残差）
    residual_history : tuple[float, ...]
        反復ごとの残差ノルム（pseudo_time_in_residual=True なら擬似時間項込み）。先頭は初期場の定常残差
    steady_residual_history : tuple[float, ...]
        反復ごとの Δτ 非依存の定常残差ノルム
    cfl_history : tuple[float, ...]
        反復ごとの CFL
    mass_in, mass_out : float
        inlet / outlet 質量流量 [kg/s]
    elapsed : float
        計算時間 [s]
    n_factorizations : int
        前処理 LU（PARDISO）の分解回数（Stokes 初期場の 1 回を含む）
    n_gmres_total : int
        GMRES 内部反復の総数（JFNK の残差評価回数の目安）
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    converged: bool
    failure_reason: str
    n_iter: int
    residual_history: tuple[float, ...]
    steady_residual_history: tuple[float, ...]
    cfl_history: tuple[float, ...]
    mass_in: float
    mass_out: float
    elapsed: float
    residual_ref: float = 1.0
    n_factorizations: int = 0
    n_gmres_total: int = 0
    limiter_frozen_at: int = -1  # [リミター凍結] 凍結した Newton 反復（-1 なら凍結せず）
    residual_unfrozen: float = -1.0  # 凍結を解いた真の定常残差 |R|（凍結していなければ -1）

    @property
    def rel_residual(self) -> float:
        return self.residual_history[-1] / self.residual_ref

    @property
    def rel_steady_residual(self) -> float:
        return self.steady_residual_history[-1] / self.residual_ref
