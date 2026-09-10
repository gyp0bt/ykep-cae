"""ゴースト付き局所パッチ上の残差カーネル（numba）.

`nsb.fastres.residual_kernel` と同じ離散化（線形面補間、Rhie–Chow、2 次風上 + Venkatakrishnan、
Dirichlet 面の拡散 2μA/d、outlet 面の零勾配）を、DMDA が配る **ゴースト幅 2 の局所パッチ**上で評価する。

パッチの 4 辺は「物理境界」か「分割の切れ目」のどちらか。切れ目側の面は outlet と同じ零勾配コピーで
埋めるが、その影響が届くのはパッチ端 2 セル（ゴースト）だけで、所有セルの残差には入らない
（残差 (i) の依存範囲は u(i−2 … i+2)。理由は nsbp/README.md「ステンシル幅」参照）。

[壁セル] `wall_x` / `wall_y`（0: 内部面 / 1: 右・上セルが流体 / 2: 左・下セルが流体 / 3: 両側固体）で
領域内部の no-slip 壁面を表す（`nsb.assembly` と同じ）。パッチの 4 辺の面は従来どおり
物理境界／切れ目として扱い、壁マスクは内部面にだけ効かせる。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def venkat_psi(u, v, ufx, vfx, ufy, vfy, dx, dy, venkat_k, wall_x, wall_y):
    """セルごとの Venkatakrishnan リミター ψ（u, v 成分）。nsb.fastres と同じ式."""
    nx, ny = u.shape
    eps2 = (venkat_k * min(dx, dy)) ** 3
    psi_u = np.ones((nx, ny))
    psi_v = np.ones((nx, ny))
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
                # [壁セル] 固体側の隣は壁面値 0 を見る
                if i == nx - 1:
                    nb_e = phifx[nx, j]
                else:
                    nb_e = phi[i + 1, j] if wall_x[i + 1, j] == 0 else 0.0
                if i == 0:
                    nb_w = phifx[0, j]
                else:
                    nb_w = phi[i - 1, j] if wall_x[i, j] == 0 else 0.0
                if j == ny - 1:
                    nb_n = phify[i, ny]
                else:
                    nb_n = phi[i, j + 1] if wall_y[i, j + 1] == 0 else 0.0
                if j == 0:
                    nb_s = phify[i, 0]
                else:
                    nb_s = phi[i, j - 1] if wall_y[i, j] == 0 else 0.0
                nb_max = max(max(nb_e, nb_w), max(nb_n, nb_s))
                nb_min = min(min(nb_e, nb_w), min(nb_n, nb_s))
                d_max = max(nb_max - phi_p, 0.0)
                d_min = min(nb_min - phi_p, 0.0)
                psi = 1.0
                for q in range(4):
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
                    psi_u[i, j] = psi
                else:
                    psi_v[i, j] = psi
    return psi_u, psi_v


@njit(cache=True, parallel=True, fastmath=False)
def face_values(
    u,
    v,
    p,
    w_phys,
    e_phys,
    s_phys,
    n_phys,
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
    wall_x,
    wall_y,
):
    """線形補間の面値（x 面: (nx+1, ny)、y 面: (nx, ny+1)）。境界面は境界値、切れ目は零勾配コピー."""
    nx, ny = u.shape
    ufx = np.empty((nx + 1, ny))
    vfx = np.empty((nx + 1, ny))
    pfx = np.empty((nx + 1, ny))
    for i in prange(nx + 1):
        for j in range(ny):
            if i == 0:
                if (not w_phys) or w_outlet[j]:
                    ufx[0, j] = u[0, j]
                    vfx[0, j] = v[0, j]
                    pfx[0, j] = w_p[j] if w_phys else p[0, j]
                else:
                    ufx[0, j] = u_w[j]
                    vfx[0, j] = v_w[j]
                    pfx[0, j] = p[0, j]
            elif i == nx:
                if (not e_phys) or e_outlet[j]:
                    ufx[nx, j] = u[nx - 1, j]
                    vfx[nx, j] = v[nx - 1, j]
                    pfx[nx, j] = e_p[j] if e_phys else p[nx - 1, j]
                else:
                    ufx[nx, j] = u_e[j]
                    vfx[nx, j] = v_e[j]
                    pfx[nx, j] = p[nx - 1, j]
            elif wall_x[i, j] == 0:
                ufx[i, j] = 0.5 * (u[i - 1, j] + u[i, j])
                vfx[i, j] = 0.5 * (v[i - 1, j] + v[i, j])
                pfx[i, j] = 0.5 * (p[i - 1, j] + p[i, j])
            else:  # [壁セル] 速度 0、圧力は流体側セル値
                ufx[i, j] = 0.0
                vfx[i, j] = 0.0
                if wall_x[i, j] == 1:
                    pfx[i, j] = p[i, j]
                elif wall_x[i, j] == 2:
                    pfx[i, j] = p[i - 1, j]
                else:
                    pfx[i, j] = 0.0
    ufy = np.empty((nx, ny + 1))
    vfy = np.empty((nx, ny + 1))
    pfy = np.empty((nx, ny + 1))
    for i in prange(nx):
        for j in range(ny + 1):
            if j == 0:
                if (not s_phys) or s_outlet[i]:
                    ufy[i, 0] = u[i, 0]
                    vfy[i, 0] = v[i, 0]
                    pfy[i, 0] = s_p[i] if s_phys else p[i, 0]
                else:
                    ufy[i, 0] = u_s[i]
                    vfy[i, 0] = v_s[i]
                    pfy[i, 0] = p[i, 0]
            elif j == ny:
                if (not n_phys) or n_outlet[i]:
                    ufy[i, ny] = u[i, ny - 1]
                    vfy[i, ny] = v[i, ny - 1]
                    pfy[i, ny] = n_p[i] if n_phys else p[i, ny - 1]
                else:
                    ufy[i, ny] = u_n[i]
                    vfy[i, ny] = v_n[i]
                    pfy[i, ny] = p[i, ny - 1]
            elif wall_y[i, j] == 0:
                ufy[i, j] = 0.5 * (u[i, j - 1] + u[i, j])
                vfy[i, j] = 0.5 * (v[i, j - 1] + v[i, j])
                pfy[i, j] = 0.5 * (p[i, j - 1] + p[i, j])
            else:
                ufy[i, j] = 0.0
                vfy[i, j] = 0.0
                if wall_y[i, j] == 1:
                    pfy[i, j] = p[i, j]
                elif wall_y[i, j] == 2:
                    pfy[i, j] = p[i, j - 1]
                else:
                    pfy[i, j] = 0.0
    return ufx, vfx, pfx, ufy, vfy, pfy


@njit(cache=True, parallel=True, fastmath=False)
def residual_patch(  # noqa: PLR0913
    u,
    v,
    p,
    rho,
    mu,
    dx,
    dy,
    w_phys,
    e_phys,
    s_phys,
    n_phys,
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
    q_src,
    q_sink,
    c_sink,
    cp_sink,
    first_order,
    venkat_k,
    cs,
    psi_u,
    psi_v,
    use_frozen,
    wall_x,
    wall_y,
):
    nx, ny = u.shape
    vol = dx * dy
    ufx, vfx, pfx, ufy, vfy, pfy = face_values(
        u,
        v,
        p,
        w_phys,
        e_phys,
        s_phys,
        n_phys,
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
        wall_x,
        wall_y,
    )
    # ---- RC 用 a_P → d_cell、圧力セル勾配 ----
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
                + drag_vol[i, j]
            )
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
            elif wall_x[i, j] != 0:
                fx[i, j] = 0.0  # [壁セル] 壁面は質量を通さない
            else:
                dfx = 0.5 * (d_cell[i - 1, j] + d_cell[i, j])
                corr = dfx * ((p[i, j] - p[i - 1, j]) / dx - 0.5 * (gpx[i - 1, j] + gpx[i, j]))
                fx[i, j] = rho * dy * (ufx[i, j] - corr)
    for i in prange(nx):
        for j in range(ny + 1):
            if j == 0 or j == ny:
                fy[i, j] = rho * dx * vfy[i, j]
            elif wall_y[i, j] != 0:
                fy[i, j] = 0.0
            else:
                dfy = 0.5 * (d_cell[i, j - 1] + d_cell[i, j])
                corr = dfy * ((p[i, j] - p[i, j - 1]) / dy - 0.5 * (gpy[i, j - 1] + gpy[i, j]))
                fy[i, j] = rho * dx * (vfy[i, j] - corr)
    # ---- 2 次風上の外挿量（Venkatakrishnan、または凍結した ψ） ----
    ex_u = np.zeros((nx, ny))
    ey_u = np.zeros((nx, ny))
    ex_v = np.zeros((nx, ny))
    ey_v = np.zeros((nx, ny))
    if not first_order:
        if use_frozen:
            pu = psi_u
            pv = psi_v
        else:
            pu, pv = venkat_psi(u, v, ufx, vfx, ufy, vfy, dx, dy, venkat_k, wall_x, wall_y)
        for i in prange(nx):
            for j in range(ny):
                gx = (ufx[i + 1, j] - ufx[i, j]) / dx
                gy = (ufy[i, j + 1] - ufy[i, j]) / dy
                ex_u[i, j] = 0.5 * dx * pu[i, j] * gx
                ey_u[i, j] = 0.5 * dy * pu[i, j] * gy
                gx = (vfx[i + 1, j] - vfx[i, j]) / dx
                gy = (vfy[i, j + 1] - vfy[i, j]) / dy
                ex_v[i, j] = 0.5 * dx * pv[i, j] * gx
                ey_v[i, j] = 0.5 * dy * pv[i, j] * gy
    # ---- 対流面値 × 流束 ----
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
            if i == 0:
                if (not w_phys) or w_outlet[j]:
                    gw_u = 0.0
                    gw_v = 0.0
                else:
                    gw_u = (u[0, j] - u_w[j]) / hx
                    gw_v = (v[0, j] - v_w[j]) / hx
            elif wall_x[i, j] == 0:
                gw_u = (u[i, j] - u[i - 1, j]) / dx
                gw_v = (v[i, j] - v[i - 1, j]) / dx
            elif wall_x[i, j] == 1:  # [壁セル] このセルが流体側
                gw_u = u[i, j] / hx
                gw_v = v[i, j] / hx
            elif wall_x[i, j] == 2:
                gw_u = -u[i - 1, j] / hx
                gw_v = -v[i - 1, j] / hx
            else:
                gw_u = 0.0
                gw_v = 0.0
            if i == nx - 1:
                if (not e_phys) or e_outlet[j]:
                    ge_u = 0.0
                    ge_v = 0.0
                else:
                    ge_u = (u_e[j] - u[nx - 1, j]) / hx
                    ge_v = (v_e[j] - v[nx - 1, j]) / hx
            elif wall_x[i + 1, j] == 0:
                ge_u = (u[i + 1, j] - u[i, j]) / dx
                ge_v = (v[i + 1, j] - v[i, j]) / dx
            elif wall_x[i + 1, j] == 1:
                ge_u = u[i + 1, j] / hx
                ge_v = v[i + 1, j] / hx
            elif wall_x[i + 1, j] == 2:  # このセルが流体側
                ge_u = -u[i, j] / hx
                ge_v = -v[i, j] / hx
            else:
                ge_u = 0.0
                ge_v = 0.0
            if j == 0:
                if (not s_phys) or s_outlet[i]:
                    gs_u = 0.0
                    gs_v = 0.0
                else:
                    gs_u = (u[i, 0] - u_s[i]) / hy
                    gs_v = (v[i, 0] - v_s[i]) / hy
            elif wall_y[i, j] == 0:
                gs_u = (u[i, j] - u[i, j - 1]) / dy
                gs_v = (v[i, j] - v[i, j - 1]) / dy
            elif wall_y[i, j] == 1:
                gs_u = u[i, j] / hy
                gs_v = v[i, j] / hy
            elif wall_y[i, j] == 2:
                gs_u = -u[i, j - 1] / hy
                gs_v = -v[i, j - 1] / hy
            else:
                gs_u = 0.0
                gs_v = 0.0
            if j == ny - 1:
                if (not n_phys) or n_outlet[i]:
                    gn_u = 0.0
                    gn_v = 0.0
                else:
                    gn_u = (u_n[i] - u[i, ny - 1]) / hy
                    gn_v = (v_n[i] - v[i, ny - 1]) / hy
            elif wall_y[i, j + 1] == 0:
                gn_u = (u[i, j + 1] - u[i, j]) / dy
                gn_v = (v[i, j + 1] - v[i, j]) / dy
            elif wall_y[i, j + 1] == 1:
                gn_u = u[i, j + 1] / hy
                gn_v = v[i, j + 1] / hy
            elif wall_y[i, j + 1] == 2:
                gn_u = -u[i, j] / hy
                gn_v = -v[i, j] / hy
            else:
                gn_u = 0.0
                gn_v = 0.0
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
                + drag_vol[i, j] * u[i, j]
                + q_out * u[i, j]
            )
            r_v[i, j] = (
                cs * conv_v
                - diff_v
                + (pfy[i, j + 1] - pfy[i, j]) * dx
                + drag_vol[i, j] * v[i, j]
                + q_out * v[i, j]
            )
            r_p[i, j] = fx[i + 1, j] - fx[i, j] + fy[i, j + 1] - fy[i, j] - q_in + q_out
    return r_u, r_v, r_p


@njit(cache=True, parallel=True)
def dtau_patch(u, v, dx, dy, cfl, u_floor):
    """セル局所の擬似時間増分 Δτ = CFL·min(Δx,Δy)/max(|u|+|v|, u_floor)（nsb.solver.compute_dtau と同じ）."""
    nx, ny = u.shape
    out = np.empty((nx, ny))
    hmin = min(dx, dy)
    for i in prange(nx):
        for j in range(ny):
            speed = abs(u[i, j]) + abs(v[i, j])
            if speed < u_floor:
                speed = u_floor
            if speed > 0.0:
                out[i, j] = min(cfl * hmin / max(speed, 1e-300), 1e30)
            else:
                out[i, j] = 1e30
    return out


def limiter_psi(u, v, p, args_bc, dx, dy, venkat_k, wall_x, wall_y):
    """凍結用: パッチ上の Venkatakrishnan ψ（u, v 成分）を現在の場から計算する."""
    ufx, vfx, _pfx, ufy, vfy, _pfy = face_values(u, v, p, *args_bc[:20], wall_x, wall_y)
    return venkat_psi(u, v, ufx, vfx, ufy, vfy, dx, dy, venkat_k, wall_x, wall_y)
