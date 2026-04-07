#!/usr/bin/env python3
"""1D過渡固体電熱解析 検算スクリプト (Gauss-Seidel)

区間ごとに物性値と断面積が異なる1次元ロッドのジュール加熱を
制御体積法 + 陰的Euler + Gauss-Seidel反復で解く。

検算目的のローテクスクリプト。メッシュ依存性バグの検証用。

物理:
    rho * c * A * dT/dt = d/dx [k * A * dT/dx] + I^2 * rho_e(T) / A

    rho_e(T) = rho_e0 * (1 + alpha * (T - T_ref))
    q_vol = I^2 * rho_e(T) / A^2  [W/m^3]

離散化:
    セル中心有限体積法、陰的Euler時間積分
    面コンダクタンスは直列抵抗モデル (熱抵抗の直列接続):
        1/C_{i+1/2} = dx_i/(2*k_i*A_i) + dx_{i+1}/(2*k_{i+1}*A_{i+1})

    これが正しく実装されていれば、均一物性でメッシュ依存性は出ない。

メッシュ依存性バグの典型原因:
    1. 面コンダクタンスで kA の直列抵抗でなく k の調和平均を使っている
    2. ジュール発熱の体積積分で A が抜けている (q_vol*dx vs q_vol*A*dx)
    3. 境界面で dx/2 でなく dx を使っている
    4. 過渡項で rho*c*A*dx でなく rho*c*dx を使っている
"""

import numpy as np


def solve_1d_joule_gs(
    # ---- メッシュ ----
    n_cells_per_seg,  # list[int]: 各区間のセル数
    seg_lengths,  # list[float]: 各区間の長さ [m]
    seg_areas,  # list[float]: 各区間の断面積 [m^2]
    # ---- 物性 (区間ごと) ----
    seg_rho,  # list[float]: 密度 [kg/m^3]
    seg_c,  # list[float]: 比熱 [J/(kg*K)]
    seg_k,  # list[float]: 熱伝導率 [W/(m*K)]
    seg_rho_e0,  # list[float]: 基準電気抵抗率 [Ohm*m]
    seg_alpha,  # list[float]: 抵抗温度係数 [1/K]
    # ---- 条件 ----
    current_I,  # float: 電流 [A]
    T_left,  # float: 左境界温度 [K]
    T_right,  # float: 右境界温度 [K]
    T_init,  # float: 初期温度 [K]
    T_ref,  # float: 抵抗の基準温度 [K]
    dt,  # float: 時間刻み [s]
    t_end,  # float: 終了時刻 [s]
    T_melt=None,  # float or None: 溶断温度 [K] (超えたら停止)
    # ---- ソルバー ----
    gs_tol=1e-10,
    gs_max_iter=500,
    outer_max_iter=10,
    print_every=0,
):
    """1D過渡ジュール電熱をGauss-Seidelで解く。

    Returns: (T, x_center, t_final, info_dict)
    """
    n_seg = len(seg_lengths)

    # ---- セル配列の構築 ----
    dx_list, rho_list, c_list, k_list, A_list = [], [], [], [], []
    rho_e0_list, alpha_list = [], []
    for s in range(n_seg):
        cell_dx = seg_lengths[s] / n_cells_per_seg[s]
        for _ in range(n_cells_per_seg[s]):
            dx_list.append(cell_dx)
            rho_list.append(seg_rho[s])
            c_list.append(seg_c[s])
            k_list.append(seg_k[s])
            A_list.append(seg_areas[s])
            rho_e0_list.append(seg_rho_e0[s])
            alpha_list.append(seg_alpha[s])

    N = len(dx_list)
    dx = np.array(dx_list)
    rho = np.array(rho_list)
    c = np.array(c_list)
    k = np.array(k_list)
    A = np.array(A_list)
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
    b_bnd[0] = C_left_bnd * T_left
    a_P_bnd[-1] = C_right_bnd
    b_bnd[-1] = C_right_bnd * T_right

    # ---- 時間進行 ----
    T = np.full(N, T_init)
    t = 0.0
    step = 0
    melted = False

    while t < t_end - 1e-15:
        this_dt = min(dt, t_end - t)
        # a_P0を現在のdtで再計算 (最終ステップでdtが変わる場合)
        a_P0_now = rho * c * V / this_dt
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
                    a_P = a_P0_now[i] + a_W[i] + a_E[i] + a_P_bnd[i]
                    b_i = a_P0_now[i] * T_old[i] + source[i] + b_bnd[i]
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
            print(f"  t={t:.6e}s  step={step}  T_max={np.max(T):.2f}K  conv={converged}")

        if T_melt is not None and np.max(T) >= T_melt:
            melted = True
            if print_every > 0:
                print(f"  ** 溶断温度 {T_melt}K 到達 at t={t:.6e}s **")
            break

    # ---- エネルギー収支 ----
    rho_e_final = rho_e0 * (1.0 + alpha_e * (T - T_ref))
    q_vol_final = current_I**2 * rho_e_final / A**2
    Q_joule = np.sum(q_vol_final * V)
    Q_left_out = C_left_bnd * (T[0] - T_left)
    Q_right_out = C_right_bnd * (T[-1] - T_right)
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


# ======================================================================
# テストケース
# ======================================================================


def test_uniform_steady():
    """均一ロッドの定常解析解との比較 + メッシュ収束性テスト"""
    print("=" * 70)
    print("Test 1: 均一ロッド定常解 (alpha=0) + メッシュ収束性")
    print("=" * 70)

    # 物性 (ニクロム線的)
    rho_val = 8400.0
    c_val = 450.0
    k_val = 11.3
    rho_e0_val = 1.1e-6
    L = 0.01  # 1cm
    A_val = 1e-6  # 1mm^2
    cur = 5.0  # 電流 [A]
    T_bc = 300.0

    # 解析解の最大温度
    T_max_exact = T_bc + (cur**2 * rho_e0_val) / (2 * k_val * A_val**2) * (L / 2) ** 2
    print(f"  解析解 T_max = {T_max_exact:.6f} K")
    print(f"  ΔT_max = {T_max_exact - T_bc:.6f} K")
    print()

    # 時定数: tau = rho*c*L^2 / (pi^2*k)
    tau = rho_val * c_val * L**2 / (np.pi**2 * k_val)
    print(f"  熱時定数 tau = {tau:.4f} s")
    t_end = 10 * tau  # 十分定常に到達
    print(f"  t_end = {t_end:.4f} s (10*tau)")
    print()

    N_list = [5, 10, 20, 40, 80, 160]
    errors = []

    print(
        f"  {'N':>5s}  {'dx[mm]':>8s}  {'T_max数値':>14s}  {'T_max解析':>14s}"
        f"  {'誤差[K]':>12s}  {'次数':>6s}"
    )
    print(f"  {'-' * 5}  {'-' * 8}  {'-' * 14}  {'-' * 14}  {'-' * 12}  {'-' * 6}")

    for i_n, Nc in enumerate(N_list):
        T, xc, info = solve_1d_joule_gs(
            n_cells_per_seg=[Nc],
            seg_lengths=[L],
            seg_areas=[A_val],
            seg_rho=[rho_val],
            seg_c=[c_val],
            seg_k=[k_val],
            seg_rho_e0=[rho_e0_val],
            seg_alpha=[0.0],
            current_I=cur,
            T_left=T_bc,
            T_right=T_bc,
            T_init=T_bc,
            T_ref=T_bc,
            dt=tau / 10,
            t_end=t_end,
        )
        T_exact = analytical_steady_uniform(xc, L, T_bc, cur, rho_e0_val, k_val, A_val)
        err = np.max(np.abs(T - T_exact))
        errors.append(err)

        if i_n == 0:
            order_str = "---"
        else:
            order = np.log(errors[i_n - 1] / errors[i_n]) / np.log(N_list[i_n] / N_list[i_n - 1])
            order_str = f"{order:.2f}"

        dx_mm = L / Nc * 1000
        print(
            f"  {Nc:5d}  {dx_mm:8.4f}  {np.max(T):14.8f}  {T_max_exact:14.8f}"
            f"  {err:12.4e}  {order_str:>6s}"
        )

    print()
    # 判定
    final_order = np.log(errors[-2] / errors[-1]) / np.log(N_list[-1] / N_list[-2])
    if abs(final_order - 2.0) < 0.3:
        print(f"  PASS: 2次収束確認 (次数={final_order:.2f})")
    else:
        print(f"  FAIL: 2次収束でない (次数={final_order:.2f})")

    # エネルギー収支 (最密メッシュ)
    print(f"\n  エネルギー収支 (N={N_list[-1]}):")
    print(f"    Q_joule  = {info['Q_joule']:.6e} W")
    print(f"    Q_left   = {info['Q_left']:.6e} W")
    print(f"    Q_right  = {info['Q_right']:.6e} W")
    print(f"    収支誤差 = {info['balance_err']:.6e} W")
    print()


def test_multi_segment():
    """銅-ニクロム-銅 の3区間ロッド"""
    print("=" * 70)
    print("Test 2: 多区間ロッド (銅-ニクロム-銅) + エネルギー収支")
    print("=" * 70)

    # 銅: rho=8960, c=385, k=401, rho_e0=1.68e-8
    # ニクロム: rho=8400, c=450, k=11.3, rho_e0=1.1e-6
    cur = 5.0  # 電流 [A]
    T_bc = 300.0

    # ニクロム部の時定数: tau = 8400*450*0.01^2/(pi^2*11.3) = 3.4s
    # 定常到達に 30*tau = 100s 以上必要
    T, xc, info = solve_1d_joule_gs(
        n_cells_per_seg=[10, 20, 10],
        seg_lengths=[0.005, 0.01, 0.005],  # 5mm-10mm-5mm
        seg_areas=[1e-6, 0.5e-6, 1e-6],  # 1mm^2-0.5mm^2-1mm^2
        seg_rho=[8960.0, 8400.0, 8960.0],
        seg_c=[385.0, 450.0, 385.0],
        seg_k=[401.0, 11.3, 401.0],
        seg_rho_e0=[1.68e-8, 1.1e-6, 1.68e-8],
        seg_alpha=[0.0, 0.0, 0.0],
        current_I=cur,
        T_left=T_bc,
        T_right=T_bc,
        T_init=T_bc,
        T_ref=T_bc,
        dt=0.5,
        t_end=200.0,
        print_every=40,
    )

    print()
    print(f"  T_max = {info['T_max']:.4f} K  at x = {info['T_max_pos'] * 1000:.2f} mm")
    print(f"  Q_joule  = {info['Q_joule']:.6e} W")
    print(f"  Q_left   = {info['Q_left']:.6e} W")
    print(f"  Q_right  = {info['Q_right']:.6e} W")
    print(f"  収支誤差 = {info['balance_err']:.6e} W")
    print(f"  相対誤差 = {abs(info['balance_err']) / info['Q_joule']:.6e}")
    print()

    # 温度プロファイル
    print("  温度プロファイル:")
    print(f"    {'x[mm]':>8s}  {'T[K]':>10s}")
    for idx in np.linspace(0, info["N"] - 1, 12, dtype=int):
        print(f"    {xc[idx] * 1000:8.3f}  {T[idx]:10.4f}")
    print()


def test_temp_dependent():
    """温度依存抵抗率 (alpha!=0) + メッシュ収束性"""
    print("=" * 70)
    print("Test 3: 温度依存抵抗率 (alpha=4e-4/K) + メッシュ収束性")
    print("=" * 70)

    rho_val = 8400.0
    c_val = 450.0
    k_val = 11.3
    rho_e0_val = 1.1e-6
    alpha_val = 4e-4  # 穏やかな温度係数
    L = 0.01
    A_val = 1e-6
    cur = 5.0  # 電流 [A]
    T_bc = 300.0

    tau = rho_val * c_val * L**2 / (np.pi**2 * k_val)
    t_end = 10 * tau

    N_list = [10, 20, 40, 80, 160]
    T_max_list = []

    for Nc in N_list:
        T, xc, info = solve_1d_joule_gs(
            n_cells_per_seg=[Nc],
            seg_lengths=[L],
            seg_areas=[A_val],
            seg_rho=[rho_val],
            seg_c=[c_val],
            seg_k=[k_val],
            seg_rho_e0=[rho_e0_val],
            seg_alpha=[alpha_val],
            current_I=cur,
            T_left=T_bc,
            T_right=T_bc,
            T_init=T_bc,
            T_ref=T_bc,
            dt=tau / 10,
            t_end=t_end,
            outer_max_iter=20,
        )
        T_max_list.append(np.max(T))

    T_ref_val = T_max_list[-1]
    print(f"\n  {'N':>5s}  {'T_max[K]':>14s}  {'|ΔT_max|':>12s}  {'次数':>6s}")
    print(f"  {'-' * 5}  {'-' * 14}  {'-' * 12}  {'-' * 6}")

    for i, Nc in enumerate(N_list):
        err = abs(T_max_list[i] - T_ref_val)
        if i == 0 or err == 0 or abs(T_max_list[i - 1] - T_ref_val) == 0:
            order_str = "---"
        else:
            order = np.log(abs(T_max_list[i - 1] - T_ref_val) / err) / np.log(
                N_list[i] / N_list[i - 1]
            )
            order_str = f"{order:.2f}"
        print(f"  {Nc:5d}  {T_max_list[i]:14.8f}  {err:12.4e}  {order_str:>6s}")

    # エネルギー収支
    print(f"\n  エネルギー収支 (N={N_list[-1]}):")
    print(f"    Q_joule  = {info['Q_joule']:.6e} W")
    print(f"    Q_left   = {info['Q_left']:.6e} W")
    print(f"    Q_right  = {info['Q_right']:.6e} W")
    print(f"    収支誤差 = {info['balance_err']:.6e} W")
    print()


def test_transient_to_melt():
    """溶断温度まで加熱する過渡解析"""
    print("=" * 70)
    print("Test 4: 過渡解析 (溶断温度 1000K まで)")
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
    T, xc, info = solve_1d_joule_gs(
        n_cells_per_seg=[10],
        seg_lengths=[L],
        seg_areas=[A_val],
        seg_rho=[rho_val],
        seg_c=[c_val],
        seg_k=[k_val],
        seg_rho_e0=[1.1e-6],
        seg_alpha=[4e-4],
        current_I=5.0,
        T_left=300.0,
        T_right=300.0,
        T_init=300.0,
        T_ref=300.0,
        dt=1e-5,
        t_end=0.05,
        T_melt=1000.0,
        print_every=500,
    )

    print()
    print(f"  最終時刻   : {info['t_final']:.6e} s")
    print(f"  溶断到達   : {info['melted']}")
    print(f"  T_max      : {info['T_max']:.2f} K  at x = {info['T_max_pos'] * 1000:.2f} mm")
    print(f"  ステップ数 : {info['steps']}")
    print()
    print("  温度プロファイル:")
    for i in range(len(T)):
        bar = "#" * int((T[i] - 300) / 10)
        print(f"    x={xc[i] * 1000:6.3f}mm  T={T[i]:8.2f}K  {bar}")
    print()


# ======================================================================
# メイン
# ======================================================================


def main():
    print()
    print("################################################################")
    print("#  1D過渡固体電熱解析 検算スクリプト (Gauss-Seidel)             #")
    print("#  制御体積法 + 陰的Euler + 直列抵抗モデル面コンダクタンス      #")
    print("################################################################")
    print()

    test_uniform_steady()
    test_multi_segment()
    test_temp_dependent()
    test_transient_to_melt()

    print("=" * 70)
    print("全テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
