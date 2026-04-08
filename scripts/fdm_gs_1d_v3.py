"""1D過渡固体電熱解析スクリプト (Gauss-Seidel)

区間ごとに物性値と断面積が異なる1次元ロッドのジュール加熱を
制御体積法 + 陰的Euler + Gauss-Seidel反復で解く。

物理:
    rho * c * A * dT/dt = d/dx [k * A * dT/dx] + I^2 * rho_e(T) / A

    rho_e(T) = rho_e0 * (1 + alpha * (T - T_ref))
    q_vol = I^2 * rho_e(T) / A^2  [W/m^3]

離散化:
    セル中心有限体積法、陰的Euler時間積分
    面コンダクタンスは直列抵抗モデル (熱抵抗の直列接続):
        1/C_{i+1/2} = dx_i/(2*k_i*A_i) + dx_{i+1}/(2*k_{i+1}*A_{i+1})

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def solve_1d_joule_gs(
    # ---- メッシュ ----
    n_cells_per_seg: list[int],  # 各区間のセル数
    seg_lengths: list[float],  # 各区間の長さ [m]
    seg_areas: list[float],  # 各区間の断面積 [m^2]
    seg_areas_conv: list[float],  # 各区間の放熱面積 [m^2]
    seg_h: list[float],  # 各区間の熱伝達係数 [W/m^2K]
    # ---- 物性 (区間ごと) ----
    seg_rho: list[float],  # 密度 [kg/m^3]
    seg_c: list[float],  # 比熱 [J/(kg*K)]
    seg_k: list[float],  # 熱伝導率 [W/(m*K)]
    seg_rho_e0: list[float],  # 基準電気抵抗率 [Ohm*m]
    seg_alpha: list[float],  # 抵抗温度係数 [1/K]
    # ---- 条件 ----
    current_I: float,  # 電流 [A]
    T_init: float,  # 初期温度 [K]
    T_env: float,  # 環境温度 [K]
    T_ref: float,  # 抵抗の基準温度 [K]
    dt: float,  # 時間刻み [s]
    t_end: float,  # 終了時刻 [s]
    T_left: Optional[float] = None,  # 左境界温度 [K]
    T_right: Optional[float] = None,  # 右境界温度 [K]
    T_melt: Optional[float] = None,  # 溶断温度 [K] (超えたら停止)
    # ---- ソルバー ----
    gs_tol: float = 1e-10,
    gs_max_iter: int = 500,
    outer_max_iter: int = 10,
    print_every: int = 0,
):
    """1D過渡ジュール電熱をGauss-Seidelで解く。

    Returns: (T, x_center, t_final, info_dict)
    """
    n_seg = len(seg_lengths)

    # ---- セル配列の構築 ----
    dx_list, rho_list, c_list, k_list, A_list, A_conv_list, seg_h_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rho_e0_list, alpha_list = [], []
    for s in range(n_seg):
        cell_dx = seg_lengths[s] / n_cells_per_seg[s]
        for _ in range(n_cells_per_seg[s]):
            dx_list.append(cell_dx)
            rho_list.append(seg_rho[s])
            c_list.append(seg_c[s])
            k_list.append(seg_k[s])
            A_list.append(seg_areas[s])
            A_conv_list.append(seg_areas_conv[s] / n_cells_per_seg[s])
            seg_h_list.append(seg_h[s])
            rho_e0_list.append(seg_rho_e0[s])
            alpha_list.append(seg_alpha[s])

    N = len(dx_list)
    dx = np.array(dx_list)
    rho = np.array(rho_list)
    c = np.array(c_list)
    k = np.array(k_list)
    A = np.array(A_list)
    A_conv = np.array(A_conv_list)
    h = np.array(seg_h_list)
    rho_e0 = np.array(rho_e0_list)
    alpha_e = np.array(alpha_list)

    # セル中心座標
    x_center = np.zeros(N)
    x_center[0] = dx[0] / 2.0
    for i in range(1, N):
        x_center[i] = x_center[i - 1] + dx[i - 1] / 2.0 + dx[i] / 2.0

    # ---- 面コンダクタンス (直列抵抗モデル) ----
    # 内部面 i+1/2 (i=0..N-2):
    #   1/C = dx_i/(2*k_i*A_i) + dx_{i+1}/(2*k_{i+1}*A_{i+1})
    C_face = np.zeros(N - 1)
    for i in range(N - 1):
        R = dx[i] / (2.0 * k[i] * A[i]) + dx[i + 1] / (2.0 * k[i + 1] * A[i + 1])
        C_face[i] = 1.0 / R

    # 境界面コンダクタンス (半セル距離)
    C_left_bnd = k[0] * A[0] / (dx[0] / 2.0)
    C_right_bnd = k[-1] * A[-1] / (dx[-1] / 2.0)

    # ---- 係数 (時間不変部分) ----
    # 過渡項: a_P0 = rho * c * A * dx / dt
    V = A * dx  # セル体積
    # a_P0 = rho * c * V / dt はループ内で this_dt に応じて再計算

    # 西側・東側係数
    a_W = np.zeros(N)
    a_E = np.zeros(N)
    for i in range(N):
        if i > 0:
            a_W[i] = C_face[i - 1]
        if i < N - 1:
            a_E[i] = C_face[i]

    # 境界寄与 (a_Pへの加算分と、bへの定数項)
    a_P_bnd = np.zeros(N)
    b_bnd = np.zeros(N)
    a_P_bnd[0] = C_left_bnd
    if T_left is not None:
        b_bnd[0] = C_left_bnd * T_left
    a_P_bnd[-1] = C_right_bnd
    if T_right is not None:
        b_bnd[-1] = C_right_bnd * T_right

    # ---- 時間進行 ----
    T = np.full(N, T_init)
    t = 0.0
    step = 0
    melted = False

    while t < t_end - 1e-15:
        this_dt = min(dt, t_end - t)
        # a_P0を現在のdtで再計算 (最終ステップでdtが変わる場合)
        # a_P0_now = rho * c * V / this_dt
        a_P0_now = rho * c * V / this_dt
        a_P1_now = h * A_conv
        T_old = T.copy()

        # 外側反復 (非線形ソース項の再評価)
        for _outer in range(outer_max_iter):
            T_star = T.copy()

            # ジュール発熱: q_vol = I^2 * rho_e(T*) / A^2
            rho_e = rho_e0 * (1.0 + alpha_e * (T_star - T_ref))
            source = current_I**2 * rho_e / A**2 * V  # [W] per cell

            # Gauss-Seidel 内側反復
            converged = False
            for _gs in range(gs_max_iter):
                T_prev = T.copy()
                for i in range(N):
                    a_P = a_P0_now[i] + a_P1_now[i] + a_W[i] + a_E[i] + a_P_bnd[i]
                    b_i = (
                        a_P0_now[i] * T_old[i]
                        + a_P1_now[i] * T_env
                        + source[i]
                        + b_bnd[i]
                    )
                    if i > 0:
                        b_i += a_W[i] * T[i - 1]
                    if i < N - 1:
                        b_i += a_E[i] * T[i + 1]
                    T[i] = b_i / a_P

                if np.max(np.abs(T - T_prev)) < gs_tol:
                    converged = True
                    break

            # 外側収束判定
            if np.max(np.abs(T - T_star)) < gs_tol:
                break

        t += this_dt
        step += 1

        if print_every > 0 and step % print_every == 0:
            print(
                f"  t={t:.6e}s  step={step}  T_max={np.max(T):.2f}K  conv={converged}"
            )

        if T_melt is not None and np.max(T) >= T_melt:
            melted = True
            if print_every > 0:
                print(f"  ** 溶断温度 {T_melt}K 到達 at t={t:.6e}s **")
            break

    # ---- エネルギー収支 ----
    rho_e_final = rho_e0 * (1.0 + alpha_e * (T - T_ref))
    q_vol_final = current_I**2 * rho_e_final / A**2
    Q_joule = np.sum(q_vol_final * V)
    Q_left_out = C_left_bnd * (T[0] - T_left) if T_left is not None else 0.0
    Q_right_out = C_right_bnd * (T[-1] - T_right) if T_right is not None else 0.0
    balance_err = Q_joule - Q_left_out - Q_right_out

    info = {
        "t_final": t,
        "steps": step,
        "melted": melted,
        "T_max": np.max(T),
        "T_max_pos": x_center[np.argmax(T)],
        "Q_joule": Q_joule,
        "Q_left": Q_left_out,
        "Q_right": Q_right_out,
        "balance_err": balance_err,
        "dx": dx,
        "k": k,
        "A": A,
        "V": V,
        "N": N,
    }
    return T, x_center, info


def analytical_steady_uniform(x, L, T_bc, current, rho_e0, k, A):
    """均一ロッド定常解 (alpha=0): T(x) = T_bc + I²ρ_e0/(2kA²) * x(L-x)"""
    return T_bc + (current**2 * rho_e0) / (2.0 * k * A**2) * x * (L - x)


@dataclass
class LineArea:
    # 電流 [A]
    current: float = 5.0

    # 銅物性
    k: float = 386.0
    density: float = 8950
    c: float = 380.0
    rho: float = 1.68e-8
    alpha: float = 3.93e-3
    T_m: float = 1085.0
    # 長さ当たり熱容量 [J/K/m]
    C_l: Optional[float] = None

    # パターンサイズ
    # パターン長 [m]
    length: float = 7.5 * 1e-3
    # 発熱断面積 [m^2]
    A0: float = 0.0035e-6
    # 放熱面積 [m^2]
    A1: float = 1.36e-4 * length * 2.0
    Rth_l: Optional[float] = None

    # 輻射率
    epsilon: float = 0.1
    sigma: float = 5.670374419e-8

    # 環境温度 [℃]
    T_env: float = 25.0
    h_env: float = 5.0


if __name__ == "__main__":
    current = 5.0
    mat_data = dict(
        k=67.0, density=8900.0, c=377.0, rho=1.92e-7, alpha=8.0e-8, epsilon=0.1
    )

    """溶断温度まで加熱する過渡解析 — 10分割 vs 40分割"""
    print("=" * 70)
    print("Test 4: 過渡解析 (溶断温度 1000K まで) — メッシュ依存性チェック")
    print("=" * 70)

    # ニクロム線: 断面積を小さくして溶断に至る条件
    # A=1e-8 m² (0.1mm×0.1mm相当), I=5A → j=5e8 A/m²
    # q_vol = I²*rho_e0/A² = 25*1.1e-6/1e-16 = 2.75e11 W/m³
    # 断熱昇温率: q_vol/(rho*c) ≈ 2.75e11/(8400*450) ≈ 7.3e4 K/s
    A_val = 1e-8
    L = 0.01
    rho_val = 8400.0
    c_val = 450.0
    k_val = 11.3
    tau = rho_val * c_val * L**2 / (np.pi**2 * k_val)
    print(f"  熱時定数 tau = {tau:.4f} s")
    print(f"  断熱昇温率 ≈ {25 * 1.1e-6 / A_val**2 / (rho_val * c_val):.1f} K/s")
    print()
    n_cells = 10

    print(f"  --- N={n_cells} 分割 ---")
    T, xc, info = solve_1d_joule_gs(
        n_cells_per_seg=[n_cells],
        seg_lengths=[L],
        seg_areas=[A_val],
        seg_rho=[rho_val],
        seg_c=[c_val],
        seg_k=[k_val],
        seg_h=[5.0],
        seg_areas_conv=[1.0e-6],
        seg_rho_e0=[1.1e-6],
        seg_alpha=[4e-4],
        current_I=5.0,
        # T_left=300.0,
        T_left=None,
        # T_right=300.0,
        T_right=None,
        T_init=300.0,
        T_env=300.0,
        T_ref=300.0,
        dt=1e-5,
        t_end=0.05,
        T_melt=1000.0,
        print_every=500,
    )

    print()
    print(f"  最終時刻   : {info['t_final']:.6e} s")
    print(f"  溶断到達   : {info['melted']}")
    print(
        f"  T_max      : {info['T_max']:.2f} K  at x = {info['T_max_pos'] * 1000:.2f} mm"
    )
    print(f"  ステップ数 : {info['steps']}")
    print()
    print("  温度プロファイル:")
    # 15点に間引いて表示
    n_show = min(len(T), 15)
    indices = np.linspace(0, len(T) - 1, n_show, dtype=int)
    for idx in indices:
        bar = "#" * max(0, int((T[idx] - 300) / 10))
        print(f"    x={xc[idx] * 1000:6.3f}mm  T={T[idx]:8.2f}K  {bar}")
    print()
