"""残差評価の numba 版（JFNK の matvec 用。`BrinkmanDiscretization.residual_fast` から呼ぶ）.

`compute_state` + `residual_from_state`（numpy 配列演算、288×192 で 5.8 ms、うち 1 スレッドの中間配列生成が大半）と
同じ値を、面ごと・セルごとの `prange` ループ 7 パスで計算する（同格子で 0.3〜0.5 ms、status-39）。
状態配列（面流束・a_P 等）は作らない。ヤコビアン組立には従来の `compute_state` を使う。

numba が無ければ `HAVE_NUMBA=False` で、呼び出し側は numpy 経路に落ちる。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - 環境依存
    HAVE_NUMBA = False

    def njit(*_a, **_k):  # type: ignore[misc]
        def deco(f):
            return f

        return deco

    prange = range  # type: ignore[assignment]


@njit(cache=True, parallel=True, fastmath=False)
def residual_kernel(  # noqa: PLR0913
    u,
    v,
    p,
    rho,
    mu,
    dx,
    dy,
    w_outlet,
    e_outlet,
    s_outlet,
    n_outlet,
    w_p,
    e_p,
    s_p,
    n_p,
    u_w,
    v_w,
    u_e,
    v_e,
    u_s,
    v_s,
    u_n,
    v_n,
    diff_diag,
    drag_vol,
    pseudo_diag,
    use_pseudo,
    q_src,
    q_sink,
    c_sink,
    cp_sink,
    first_order,
    venkat_k,
    cs,
    psi_u_f,
    psi_v_f,
    use_psi,
    thickness,
    fr_re_crit,
    fr_exp,
    fr_blend,
):
    nx, ny = u.shape
    vol = dx * dy
    # ---- [摩擦則] 抗力倍率（BrinkmanDiscretization.drag_factor と同じ式）----
    drag_eff = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            if fr_re_crit > 0.0:
                sp = np.sqrt(u[i, j] * u[i, j] + v[i, j] * v[i, j])
                re_h = rho * sp * 2.0 * thickness[i, j] / mu
                fac = (1.0 + (re_h / fr_re_crit) ** (fr_exp * fr_blend)) ** (1.0 / fr_blend)
            else:
                fac = 1.0
            drag_eff[i, j] = drag_vol[i, j] * fac
    # ---- 線形補間の面値 ----
    ufx = np.empty((nx + 1, ny))
    vfx = np.empty((nx + 1, ny))
    pfx = np.empty((nx + 1, ny))
    for i in prange(nx + 1):
        for j in range(ny):
            if i == 0:
                if w_outlet[j]:
                    ufx[0, j] = u[0, j]
                    vfx[0, j] = v[0, j]
                    pfx[0, j] = w_p[j]
                else:
                    ufx[0, j] = u_w[j]
                    vfx[0, j] = v_w[j]
                    pfx[0, j] = p[0, j]
            elif i == nx:
                if e_outlet[j]:
                    ufx[nx, j] = u[nx - 1, j]
                    vfx[nx, j] = v[nx - 1, j]
                    pfx[nx, j] = e_p[j]
                else:
                    ufx[nx, j] = u_e[j]
                    vfx[nx, j] = v_e[j]
                    pfx[nx, j] = p[nx - 1, j]
            else:
                ufx[i, j] = 0.5 * (u[i - 1, j] + u[i, j])
                vfx[i, j] = 0.5 * (v[i - 1, j] + v[i, j])
                pfx[i, j] = 0.5 * (p[i - 1, j] + p[i, j])
    ufy = np.empty((nx, ny + 1))
    vfy = np.empty((nx, ny + 1))
    pfy = np.empty((nx, ny + 1))
    for i in prange(nx):
        for j in range(ny + 1):
            if j == 0:
                if s_outlet[i]:
                    ufy[i, 0] = u[i, 0]
                    vfy[i, 0] = v[i, 0]
                    pfy[i, 0] = s_p[i]
                else:
                    ufy[i, 0] = u_s[i]
                    vfy[i, 0] = v_s[i]
                    pfy[i, 0] = p[i, 0]
            elif j == ny:
                if n_outlet[i]:
                    ufy[i, ny] = u[i, ny - 1]
                    vfy[i, ny] = v[i, ny - 1]
                    pfy[i, ny] = n_p[i]
                else:
                    ufy[i, ny] = u_n[i]
                    vfy[i, ny] = v_n[i]
                    pfy[i, ny] = p[i, ny - 1]
            else:
                ufy[i, j] = 0.5 * (u[i, j - 1] + u[i, j])
                vfy[i, j] = 0.5 * (v[i, j - 1] + v[i, j])
                pfy[i, j] = 0.5 * (p[i, j - 1] + p[i, j])
    # ---- RC 用 a_P、d_cell、圧力セル勾配 ----
    d_cell = np.empty((nx, ny))
    gpx = np.empty((nx, ny))
    gpy = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            fe = rho * dy * ufx[i + 1, j]
            fw = rho * dy * ufx[i, j]
            fn = rho * dx * vfy[i, j + 1]
            fs = rho * dx * vfy[i, j]
            a_p = (
                max(fe, 0.0)
                + max(-fw, 0.0)
                + max(fn, 0.0)
                + max(-fs, 0.0)
                + diff_diag[i, j]
                + drag_eff[i, j]
            )
            if use_pseudo:
                d_cell[i, j] = vol / (a_p + pseudo_diag[i, j])
            else:
                d_cell[i, j] = vol / a_p
            gpx[i, j] = (pfx[i + 1, j] - pfx[i, j]) / dx
            gpy[i, j] = (pfy[i, j + 1] - pfy[i, j]) / dy
    # ---- Rhie–Chow 質量流束 ----
    fx = np.empty((nx + 1, ny))
    fy = np.empty((nx, ny + 1))
    for i in prange(nx + 1):
        for j in range(ny):
            if i == 0 or i == nx:
                fx[i, j] = rho * dy * ufx[i, j]
            else:
                dfx = 0.5 * (d_cell[i - 1, j] + d_cell[i, j])
                corr = dfx * ((p[i, j] - p[i - 1, j]) / dx - 0.5 * (gpx[i - 1, j] + gpx[i, j]))
                fx[i, j] = rho * dy * (ufx[i, j] - corr)
    for i in prange(nx):
        for j in range(ny + 1):
            if j == 0 or j == ny:
                fy[i, j] = rho * dx * vfy[i, j]
            else:
                dfy = 0.5 * (d_cell[i, j - 1] + d_cell[i, j])
                corr = dfy * ((p[i, j] - p[i, j - 1]) / dy - 0.5 * (gpy[i, j - 1] + gpy[i, j]))
                fy[i, j] = rho * dx * (vfy[i, j] - corr)
    # ---- 2 次風上の外挿量 ex, ey（Venkatakrishnan） ----
    ex_u = np.zeros((nx, ny))
    ey_u = np.zeros((nx, ny))
    ex_v = np.zeros((nx, ny))
    ey_v = np.zeros((nx, ny))
    if not first_order:
        eps2 = (venkat_k * min(dx, dy)) ** 3
        for i in prange(nx):
            for j in range(ny):
                for comp in range(2):
                    if comp == 0:
                        phi = u
                        phifx = ufx
                        phify = ufy
                    else:
                        phi = v
                        phifx = vfx
                        phify = vfy
                    phi_p = phi[i, j]
                    gx = (phifx[i + 1, j] - phifx[i, j]) / dx
                    gy = (phify[i, j + 1] - phify[i, j]) / dy
                    nb_e = phi[i + 1, j] if i < nx - 1 else phifx[nx, j]
                    nb_w = phi[i - 1, j] if i > 0 else phifx[0, j]
                    nb_n = phi[i, j + 1] if j < ny - 1 else phify[i, ny]
                    nb_s = phi[i, j - 1] if j > 0 else phify[i, 0]
                    nb_max = max(max(nb_e, nb_w), max(nb_n, nb_s))
                    nb_min = min(min(nb_e, nb_w), min(nb_n, nb_s))
                    d_max = max(nb_max - phi_p, 0.0)
                    d_min = min(nb_min - phi_p, 0.0)
                    psi = 1.0
                    nq = 0 if use_psi else 4  # [リミター凍結] 凍結時は ψ を再計算しない
                    if use_psi:
                        psi = psi_u_f[i, j] if comp == 0 else psi_v_f[i, j]
                    for q in range(nq):
                        if q == 0:
                            d_f = 0.5 * dx * gx
                        elif q == 1:
                            d_f = -0.5 * dx * gx
                        elif q == 2:
                            d_f = 0.5 * dy * gy
                        else:
                            d_f = -0.5 * dy * gy
                        d_p = d_max if d_f > 0.0 else d_min
                        if abs(d_f) > 1e-300:
                            num = (d_p * d_p + eps2) + 2.0 * d_f * d_p
                            den = d_p * d_p + 2.0 * d_f * d_f + d_f * d_p + eps2
                            psi_f = num / den
                        else:
                            psi_f = 1.0
                        psi_f = min(max(psi_f, 0.0), 1.0)
                        psi = min(psi, psi_f)
                    if comp == 0:
                        ex_u[i, j] = 0.5 * dx * psi * gx
                        ey_u[i, j] = 0.5 * dy * psi * gy
                    else:
                        ex_v[i, j] = 0.5 * dx * psi * gx
                        ey_v[i, j] = 0.5 * dy * psi * gy
    # ---- 対流面値 × 流束（境界面は線形面値 = 境界値、内部面は風上 + 外挿） ----
    cfx_u = np.empty((nx + 1, ny))
    cfx_v = np.empty((nx + 1, ny))
    for i in prange(nx + 1):
        for j in range(ny):
            if i == 0 or i == nx:
                cfx_u[i, j] = fx[i, j] * ufx[i, j]
                cfx_v[i, j] = fx[i, j] * vfx[i, j]
            elif fx[i, j] >= 0.0:
                cfx_u[i, j] = fx[i, j] * (u[i - 1, j] + ex_u[i - 1, j])
                cfx_v[i, j] = fx[i, j] * (v[i - 1, j] + ex_v[i - 1, j])
            else:
                cfx_u[i, j] = fx[i, j] * (u[i, j] - ex_u[i, j])
                cfx_v[i, j] = fx[i, j] * (v[i, j] - ex_v[i, j])
    cfy_u = np.empty((nx, ny + 1))
    cfy_v = np.empty((nx, ny + 1))
    for i in prange(nx):
        for j in range(ny + 1):
            if j == 0 or j == ny:
                cfy_u[i, j] = fy[i, j] * ufy[i, j]
                cfy_v[i, j] = fy[i, j] * vfy[i, j]
            elif fy[i, j] >= 0.0:
                cfy_u[i, j] = fy[i, j] * (u[i, j - 1] + ey_u[i, j - 1])
                cfy_v[i, j] = fy[i, j] * (v[i, j - 1] + ey_v[i, j - 1])
            else:
                cfy_u[i, j] = fy[i, j] * (u[i, j] - ey_u[i, j])
                cfy_v[i, j] = fy[i, j] * (v[i, j] - ey_v[i, j])
    # ---- 残差 ----
    r_u = np.empty((nx, ny))
    r_v = np.empty((nx, ny))
    r_p = np.empty((nx, ny))
    hx = 0.5 * dx
    hy = 0.5 * dy
    for i in prange(nx):
        for j in range(ny):
            # 拡散: 面勾配（Dirichlet 面は (境界値 − セル値)/(d/2)、outlet 面は 0）
            if i == 0:
                gw_u = 0.0 if w_outlet[j] else (u[0, j] - u_w[j]) / hx
                gw_v = 0.0 if w_outlet[j] else (v[0, j] - v_w[j]) / hx
            else:
                gw_u = (u[i, j] - u[i - 1, j]) / dx
                gw_v = (v[i, j] - v[i - 1, j]) / dx
            if i == nx - 1:
                ge_u = 0.0 if e_outlet[j] else (u_e[j] - u[nx - 1, j]) / hx
                ge_v = 0.0 if e_outlet[j] else (v_e[j] - v[nx - 1, j]) / hx
            else:
                ge_u = (u[i + 1, j] - u[i, j]) / dx
                ge_v = (v[i + 1, j] - v[i, j]) / dx
            if j == 0:
                gs_u = 0.0 if s_outlet[i] else (u[i, 0] - u_s[i]) / hy
                gs_v = 0.0 if s_outlet[i] else (v[i, 0] - v_s[i]) / hy
            else:
                gs_u = (u[i, j] - u[i, j - 1]) / dy
                gs_v = (v[i, j] - v[i, j - 1]) / dy
            if j == ny - 1:
                gn_u = 0.0 if n_outlet[i] else (u_n[i] - u[i, ny - 1]) / hy
                gn_v = 0.0 if n_outlet[i] else (v_n[i] - v[i, ny - 1]) / hy
            else:
                gn_u = (u[i, j + 1] - u[i, j]) / dy
                gn_v = (v[i, j + 1] - v[i, j]) / dy
            diff_u = mu * (dy * ge_u - dy * gw_u + dx * gn_u - dx * gs_u)
            diff_v = mu * (dy * ge_v - dy * gw_v + dx * gn_v - dx * gs_v)
            q_c = c_sink[i, j] * p[i, j] - cp_sink[i, j]
            q_in = q_src[i, j] + max(-q_c, 0.0)
            q_out = q_sink[i, j] + max(q_c, 0.0)
            conv_u = cfx_u[i + 1, j] - cfx_u[i, j] + cfy_u[i, j + 1] - cfy_u[i, j]
            conv_v = cfx_v[i + 1, j] - cfx_v[i, j] + cfy_v[i, j + 1] - cfy_v[i, j]
            r_u[i, j] = (
                cs * conv_u
                - diff_u
                + (pfx[i + 1, j] - pfx[i, j]) * dy
                + drag_eff[i, j] * u[i, j]
                + q_out * u[i, j]
            )
            r_v[i, j] = (
                cs * conv_v
                - diff_v
                + (pfy[i, j + 1] - pfy[i, j]) * dx
                + drag_eff[i, j] * v[i, j]
                + q_out * v[i, j]
            )
            r_p[i, j] = fx[i + 1, j] - fx[i, j] + fy[i, j + 1] - fy[i, j] - q_in + q_out
    return r_u, r_v, r_p
