"""2D Brinkman 補正 Navier-Stokes の同位置格子 FVM 離散化.

残差（1 次風上 / 2 次風上 + Venkatakrishnan リミター）を配列演算で評価し、
1 次風上ヤコビアンを疎行列オペレータの合成として解析的に組み立てる。

未知数ベクトルは x = [u.ravel(), v.ravel(), p.ravel()]（各 nx*ny、C 順）。
セル id は k = i*ny + j、x 面 id は i*ny + j (i=0..nx)、y 面 id は i*(ny+1) + j (j=0..ny)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from xkep_cae_fluid.brinkman_flow.data import BrinkmanFlowInput, ConvectionSchemeType


@dataclass
class StateArrays:
    """1 回の残差評価で共有する中間量（面速度・質量流束・a_P など）."""

    fx: np.ndarray  # (nx+1, ny) Rhie-Chow 面質量流束 [kg/s]（+x 方向正）
    fy: np.ndarray  # (nx, ny+1)
    ufx: np.ndarray  # (nx+1, ny) 線形補間の u 面値（RC 補正前）
    vfx: np.ndarray
    ufy: np.ndarray  # (nx, ny+1)
    vfy: np.ndarray
    pfx: np.ndarray  # (nx+1, ny) p 面値
    pfy: np.ndarray
    a_p: np.ndarray  # (nx, ny) 運動量対角係数 [kg/s]（RC 用、緩和なし）
    dfx: np.ndarray  # (nx+1, ny) RC 係数 d_f（境界面は 0）
    dfy: np.ndarray  # (nx, ny+1)
    conv_ufx: np.ndarray  # 対流面値（選択スキーム）
    conv_vfx: np.ndarray
    conv_ufy: np.ndarray
    conv_vfy: np.ndarray


class BrinkmanDiscretization:
    """格子・境界条件に依存する離散化オペレータを保持し、残差とヤコビアンを提供する."""

    def __init__(self, inp: BrinkmanFlowInput) -> None:
        self.inp = inp
        self.nx, self.ny = inp.nx, inp.ny
        self.n = inp.nx * inp.ny
        self.dx, self.dy = inp.dx, inp.dy
        self.vol = self.dx * self.dy
        self.rho, self.mu = inp.rho, inp.mu
        self.drag = inp.brinkman_factor * inp.mu_brinkman / inp.thickness**2  # (nx, ny) [kg/(m³s)]

        yc = (np.arange(self.ny) + 0.5) * self.dy
        g = inp.geometry
        self.inlet_mask = (yc > g.inlet_y0) & (yc < g.inlet_y1)  # (ny,)
        self.outlet_mask = (yc > g.outlet_y0) & (yc < g.outlet_y1)
        if not self.inlet_mask.any() or not self.outlet_mask.any():
            raise ValueError("inlet / outlet に対応するセルが存在しません（分割数不足）")

        # 左壁の境界値（u, v）と、拡散のディリクレ係数
        self.u_left = np.where(self.inlet_mask, inp.u_inlet, 0.0)
        self.v_left = np.zeros(self.ny)
        self.left_dirichlet = ~self.outlet_mask  # outlet はゼロ勾配

        dxx = self.mu * self.dy / self.dx
        dyy = self.mu * self.dx / self.dy
        diff_diag = np.full((self.nx, self.ny), 2.0 * dxx + 2.0 * dyy)
        diff_diag[0, :] += np.where(self.left_dirichlet, dxx, -dxx)  # Dirichlet: 2D, outlet: 0
        diff_diag[-1, :] += dxx
        diff_diag[:, 0] += dyy
        diff_diag[:, -1] += dyy
        self.diff_diag = diff_diag
        self._dxx, self._dyy = dxx, dyy

        self._build_operators()

    # ------------------------------------------------------------------
    # 定数オペレータ
    # ------------------------------------------------------------------
    def _cell(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return i * self.ny + j

    def _build_operators(self) -> None:
        nx, ny, n = self.nx, self.ny, self.n
        nfx, nfy = (nx + 1) * ny, nx * (ny + 1)
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cells = self._cell(ii, jj).ravel()

        # 発散: 東 - 西 / 北 - 南
        fe = (ii + 1) * ny + jj
        fw = ii * ny + jj
        self.Dx = sparse.csr_matrix(
            (
                np.r_[np.ones(n), -np.ones(n)],
                (np.r_[cells, cells], np.r_[fe.ravel(), fw.ravel()]),
            ),
            shape=(n, nfx),
        )
        fn = ii * (ny + 1) + jj + 1
        fs = ii * (ny + 1) + jj
        self.Dy = sparse.csr_matrix(
            (
                np.r_[np.ones(n), -np.ones(n)],
                (np.r_[cells, cells], np.r_[fn.ravel(), fs.ravel()]),
            ),
            shape=(n, nfy),
        )

        # 内部 x 面 (i=1..nx-1): 左セル (i-1), 右セル (i)
        fi, fj = np.meshgrid(np.arange(1, nx), np.arange(ny), indexing="ij")
        fx_int = (fi * ny + fj).ravel()
        cl = self._cell(fi - 1, fj).ravel()
        cr = self._cell(fi, fj).ravel()
        self.fx_int = fx_int
        self.fx_cl, self.fx_cr = cl, cr
        # 内部 y 面 (j=1..ny-1)
        gi, gj = np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij")
        fy_int = (gi * (ny + 1) + gj).ravel()
        cs = self._cell(gi, gj - 1).ravel()
        cn = self._cell(gi, gj).ravel()
        self.fy_int = fy_int
        self.fy_cs, self.fy_cn = cs, cn

        # 左境界面 (i=0), 右境界面 (i=nx)
        j_all = np.arange(ny)
        fx_left = j_all
        fx_right = nx * ny + j_all
        c_left = self._cell(np.zeros(ny, dtype=int), j_all)
        c_right = self._cell(np.full(ny, nx - 1), j_all)
        i_all = np.arange(nx)
        fy_bot = i_all * (ny + 1)
        fy_top = i_all * (ny + 1) + ny
        c_bot = self._cell(i_all, np.zeros(nx, dtype=int))
        c_top = self._cell(i_all, np.full(nx, ny - 1))

        def mat(rows, cols, vals, shape):
            return sparse.csr_matrix((vals, (rows, cols)), shape=shape)

        # 面平均（内部）
        self.Ax = mat(
            np.r_[fx_int, fx_int],
            np.r_[cl, cr],
            np.r_[np.full(n - ny, 0.5), np.full(n - ny, 0.5)],
            (nfx, n),
        )
        self.Ay = mat(
            np.r_[fy_int, fy_int],
            np.r_[cs, cn],
            np.r_[np.full(n - nx, 0.5), np.full(n - nx, 0.5)],
            (nfy, n),
        )

        # 速度の面補間（線形部）: 内部 0.5/0.5、outlet 面はセル値（ゼロ勾配）、他境界 0
        out_j = j_all[self.outlet_mask]
        self.Ux = self.Ax + mat(
            fx_left[self.outlet_mask], c_left[self.outlet_mask], np.ones(len(out_j)), (nfx, n)
        )
        self.Uy = self.Ay.copy()

        # 圧力の面補間: 内部 0.5/0.5、壁/inlet はセル値、outlet は 0
        nout_j = ~self.outlet_mask
        self.Px = (
            self.Ax
            + mat(fx_left[nout_j], c_left[nout_j], np.ones(nout_j.sum()), (nfx, n))
            + mat(fx_right, c_right, np.ones(ny), (nfx, n))
        )
        self.Py = (
            self.Ay
            + mat(fy_bot, c_bot, np.ones(nx), (nfy, n))
            + mat(fy_top, c_top, np.ones(nx), (nfy, n))
        )

        # 面勾配（内部のみ、RC 用）
        self.Fgx_int = mat(
            np.r_[fx_int, fx_int],
            np.r_[cr, cl],
            np.r_[np.full(n - ny, 1.0 / self.dx), np.full(n - ny, -1.0 / self.dx)],
            (nfx, n),
        )
        self.Fgy_int = mat(
            np.r_[fy_int, fy_int],
            np.r_[cn, cs],
            np.r_[np.full(n - nx, 1.0 / self.dy), np.full(n - nx, -1.0 / self.dy)],
            (nfy, n),
        )

        # 速度の面勾配（拡散用、Dirichlet 境界は 2/d、outlet は 0）
        dl = self.left_dirichlet
        self.Fgx_vel = (
            self.Fgx_int
            + mat(fx_left[dl], c_left[dl], np.full(dl.sum(), 2.0 / self.dx), (nfx, n))
            + mat(fx_right, c_right, np.full(ny, -2.0 / self.dx), (nfx, n))
        )
        self.Fgy_vel = (
            self.Fgy_int
            + mat(fy_bot, c_bot, np.full(nx, 2.0 / self.dy), (nfy, n))
            + mat(fy_top, c_top, np.full(nx, -2.0 / self.dy), (nfy, n))
        )

        # セル勾配（圧力）
        self.Gx = (self.Dx @ self.Px) / self.dx
        self.Gy = (self.Dy @ self.Py) / self.dy

        # 拡散オペレータ（R_diff = -Ldiff φ + 定数）
        self.Ldiff = self.mu * (
            self.dy * (self.Dx @ self.Fgx_vel) + self.dx * (self.Dy @ self.Fgy_vel)
        )
        self.drag_v = sparse.diags((self.drag * self.vol).ravel())

    # ------------------------------------------------------------------
    # 状態依存量
    # ------------------------------------------------------------------
    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        shape = (self.nx, self.ny)
        return x[:n].reshape(shape), x[n : 2 * n].reshape(shape), x[2 * n :].reshape(shape)

    def _linear_face_values(self, u: np.ndarray, v: np.ndarray, p: np.ndarray):
        nx, ny = self.nx, self.ny
        ufx = np.empty((nx + 1, ny))
        vfx = np.empty((nx + 1, ny))
        ufx[1:-1] = 0.5 * (u[:-1] + u[1:])
        vfx[1:-1] = 0.5 * (v[:-1] + v[1:])
        ufx[0] = np.where(self.outlet_mask, u[0], self.u_left)
        vfx[0] = np.where(self.outlet_mask, v[0], self.v_left)
        ufx[-1] = 0.0
        vfx[-1] = 0.0
        ufy = np.zeros((nx, ny + 1))
        vfy = np.zeros((nx, ny + 1))
        ufy[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
        vfy[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
        pfx = np.empty((nx + 1, ny))
        pfx[1:-1] = 0.5 * (p[:-1] + p[1:])
        pfx[0] = np.where(self.outlet_mask, 0.0, p[0])
        pfx[-1] = p[-1]
        pfy = np.empty((nx, ny + 1))
        pfy[:, 1:-1] = 0.5 * (p[:, :-1] + p[:, 1:])
        pfy[:, 0] = p[:, 0]
        pfy[:, -1] = p[:, -1]
        return ufx, vfx, ufy, vfy, pfx, pfy

    def compute_state(
        self, x: np.ndarray, scheme: ConvectionSchemeType, venkat_k: float
    ) -> StateArrays:
        """面速度・RC 質量流束・対流面値を計算する."""
        u, v, p = self.split(x)
        ufx, vfx, ufy, vfy, pfx, pfy = self._linear_face_values(u, v, p)
        rho, dx, dy, vol = self.rho, self.dx, self.dy, self.vol

        # RC 用 a_P（線形補間流束に基づく、緩和なし）
        fx_lin = rho * dy * ufx
        fy_lin = rho * dx * vfy
        a_p = (
            np.maximum(fx_lin[1:], 0.0)
            + np.maximum(-fx_lin[:-1], 0.0)
            + np.maximum(fy_lin[:, 1:], 0.0)
            + np.maximum(-fy_lin[:, :-1], 0.0)
            + self.diff_diag
            + self.drag * vol
        )
        d_cell = vol / a_p
        dfx = np.zeros((self.nx + 1, self.ny))
        dfy = np.zeros((self.nx, self.ny + 1))
        dfx[1:-1] = 0.5 * (d_cell[:-1] + d_cell[1:])
        dfy[:, 1:-1] = 0.5 * (d_cell[:, :-1] + d_cell[:, 1:])

        gpx = (pfx[1:] - pfx[:-1]) / dx
        gpy = (pfy[:, 1:] - pfy[:, :-1]) / dy
        ufx_rc = ufx.copy()
        vfy_rc = vfy.copy()
        ufx_rc[1:-1] -= dfx[1:-1] * ((p[1:] - p[:-1]) / dx - 0.5 * (gpx[:-1] + gpx[1:]))
        vfy_rc[:, 1:-1] -= dfy[:, 1:-1] * (
            (p[:, 1:] - p[:, :-1]) / dy - 0.5 * (gpy[:, :-1] + gpy[:, 1:])
        )
        fx = rho * dy * ufx_rc
        fy = rho * dx * vfy_rc

        conv_ufx, conv_ufy = self._convected_values(u, ufx, ufy, fx, fy, scheme, venkat_k)
        conv_vfx, conv_vfy = self._convected_values(v, vfx, vfy, fx, fy, scheme, venkat_k)
        return StateArrays(
            fx=fx,
            fy=fy,
            ufx=ufx,
            vfx=vfx,
            ufy=ufy,
            vfy=vfy,
            pfx=pfx,
            pfy=pfy,
            a_p=a_p,
            dfx=dfx,
            dfy=dfy,
            conv_ufx=conv_ufx,
            conv_vfx=conv_vfx,
            conv_ufy=conv_ufy,
            conv_vfy=conv_vfy,
        )

    def _convected_values(
        self,
        phi: np.ndarray,
        phifx: np.ndarray,
        phify: np.ndarray,
        fx: np.ndarray,
        fy: np.ndarray,
        scheme: ConvectionSchemeType,
        venkat_k: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """対流面値（境界面は線形面値 = 境界値、内部面は風上）."""
        cfx = phifx.copy()
        cfy = phify.copy()
        up_x = fx[1:-1] >= 0.0  # True: 風上 = 左セル
        up_y = fy[:, 1:-1] >= 0.0  # True: 風上 = 下セル
        if scheme is ConvectionSchemeType.FIRST_ORDER_UPWIND:
            cfx[1:-1] = np.where(up_x, phi[:-1], phi[1:])
            cfy[:, 1:-1] = np.where(up_y, phi[:, :-1], phi[:, 1:])
            return cfx, cfy

        gx = (phifx[1:] - phifx[:-1]) / self.dx
        gy = (phify[:, 1:] - phify[:, :-1]) / self.dy
        psi = self._venkatakrishnan(phi, phifx, phify, gx, gy, venkat_k)
        ex = 0.5 * self.dx * psi * gx  # セル中心→東西面への外挿量
        ey = 0.5 * self.dy * psi * gy
        cfx[1:-1] = np.where(up_x, phi[:-1] + ex[:-1], phi[1:] - ex[1:])
        cfy[:, 1:-1] = np.where(up_y, phi[:, :-1] + ey[:, :-1], phi[:, 1:] - ey[:, 1:])
        return cfx, cfy

    def _venkatakrishnan(
        self,
        phi: np.ndarray,
        phifx: np.ndarray,
        phify: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        k: float,
    ) -> np.ndarray:
        """Venkatakrishnan リミター ψ (nx, ny)。境界では境界面値を隣接値として扱う."""
        nx, ny = self.nx, self.ny
        nb = np.empty((4, nx, ny))
        nb[0, :-1] = phi[1:]
        nb[0, -1] = phifx[-1]
        nb[1, 1:] = phi[:-1]
        nb[1, 0] = phifx[0]
        nb[2, :, :-1] = phi[:, 1:]
        nb[2, :, -1] = phify[:, -1]
        nb[3, :, 1:] = phi[:, :-1]
        nb[3, :, 0] = phify[:, 0]
        d_max = np.maximum(nb.max(axis=0) - phi, 0.0)
        d_min = np.minimum(nb.min(axis=0) - phi, 0.0)
        eps2 = (k * min(self.dx, self.dy)) ** 3

        psi = np.ones_like(phi)
        for d_f in (
            0.5 * self.dx * gx,
            -0.5 * self.dx * gx,
            0.5 * self.dy * gy,
            -0.5 * self.dy * gy,
        ):
            d_p = np.where(d_f > 0.0, d_max, d_min)
            num = (d_p**2 + eps2) + 2.0 * d_f * d_p
            den = d_p**2 + 2.0 * d_f**2 + d_f * d_p + eps2
            psi_f = np.where(np.abs(d_f) > 1e-300, num / den, 1.0)
            psi = np.minimum(psi, np.clip(psi_f, 0.0, 1.0))
        return psi

    # ------------------------------------------------------------------
    # 残差
    # ------------------------------------------------------------------
    def residual(self, x: np.ndarray, scheme: ConvectionSchemeType, venkat_k: float) -> np.ndarray:
        st = self.compute_state(x, scheme, venkat_k)
        return self.residual_from_state(x, st)

    def residual_from_state(self, x: np.ndarray, st: StateArrays) -> np.ndarray:
        u, v, p = self.split(x)
        dx, dy, mu, vol = self.dx, self.dy, self.mu, self.vol

        def div(fxv: np.ndarray, fyv: np.ndarray) -> np.ndarray:
            return fxv[1:] - fxv[:-1] + fyv[:, 1:] - fyv[:, :-1]

        def diffusion(phi: np.ndarray, phi_left: np.ndarray) -> np.ndarray:
            # 面勾配（+x, +y 方向）
            gxf = np.empty((self.nx + 1, self.ny))
            gxf[1:-1] = (phi[1:] - phi[:-1]) / dx
            gxf[0] = np.where(self.left_dirichlet, (phi[0] - phi_left) / (0.5 * dx), 0.0)
            gxf[-1] = (0.0 - phi[-1]) / (0.5 * dx)
            gyf = np.empty((self.nx, self.ny + 1))
            gyf[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / dy
            gyf[:, 0] = (phi[:, 0] - 0.0) / (0.5 * dy)
            gyf[:, -1] = (0.0 - phi[:, -1]) / (0.5 * dy)
            return mu * div(dy * gxf, dx * gyf)  # 流入側が正

        r_u = (
            div(st.fx * st.conv_ufx, st.fy * st.conv_ufy)
            - diffusion(u, self.u_left)
            + (st.pfx[1:] - st.pfx[:-1]) * dy
            + self.drag * vol * u
        )
        r_v = (
            div(st.fx * st.conv_vfx, st.fy * st.conv_vfy)
            - diffusion(v, self.v_left)
            + (st.pfy[:, 1:] - st.pfy[:, :-1]) * dx
            + self.drag * vol * v
        )
        r_p = div(st.fx, st.fy)
        return np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])

    # ------------------------------------------------------------------
    # 1 次風上ヤコビアン
    # ------------------------------------------------------------------
    def jacobian_first_order(
        self, st: StateArrays, newton_convection: bool = True
    ) -> sparse.csr_matrix:
        """1 次風上・RC 係数凍結のヤコビアン（3N×3N、ブロック順 [u, v, p]）."""
        n = self.n
        rho, dx, dy = self.rho, self.dx, self.dy
        nfx, nfy = (self.nx + 1) * self.ny, self.nx * (self.ny + 1)

        fx = st.fx.ravel()
        fy = st.fy.ravel()
        # 風上セレクタ
        up_x = fx[self.fx_int] >= 0.0
        Wx = sparse.csr_matrix(
            (np.ones(len(self.fx_int)), (self.fx_int, np.where(up_x, self.fx_cl, self.fx_cr))),
            shape=(nfx, n),
        )
        out_faces = np.arange(self.ny)[self.outlet_mask]
        Wx = Wx + sparse.csr_matrix(
            (np.ones(len(out_faces)), (out_faces, out_faces)), shape=(nfx, n)
        )  # outlet 面: φ_f = φ_P（セル id = j）
        up_y = fy[self.fy_int] >= 0.0
        Wy = sparse.csr_matrix(
            (np.ones(len(self.fy_int)), (self.fy_int, np.where(up_y, self.fy_cs, self.fy_cn))),
            shape=(nfy, n),
        )
        conv = self.Dx @ sparse.diags(fx) @ Wx + self.Dy @ sparse.diags(fy) @ Wy

        # RC 質量流束の p 依存: ∂Fx/∂p = -ρ dy diag(dfx) (Fgx_int - Ax Gx)
        Mx_p = -rho * dy * sparse.diags(st.dfx.ravel()) @ (self.Fgx_int - self.Ax @ self.Gx)
        My_p = -rho * dx * sparse.diags(st.dfy.ravel()) @ (self.Fgy_int - self.Ay @ self.Gy)
        dFx_du = rho * dy * self.Ux
        dFy_dv = rho * dx * self.Uy

        base = conv - self.Ldiff + self.drag_v
        J_up = dy * (self.Dx @ self.Px)
        J_vp = dx * (self.Dy @ self.Py)
        J_uu = base
        J_vv = base
        J_uv = sparse.csr_matrix((n, n))
        J_vu = sparse.csr_matrix((n, n))
        if newton_convection:
            Du_x = self.Dx @ sparse.diags(st.conv_ufx.ravel())
            Du_y = self.Dy @ sparse.diags(st.conv_ufy.ravel())
            Dv_x = self.Dx @ sparse.diags(st.conv_vfx.ravel())
            Dv_y = self.Dy @ sparse.diags(st.conv_vfy.ravel())
            J_uu = J_uu + Du_x @ dFx_du
            J_uv = Du_y @ dFy_dv
            J_up = J_up + Du_x @ Mx_p + Du_y @ My_p
            J_vu = Dv_x @ dFx_du
            J_vv = J_vv + Dv_y @ dFy_dv
            J_vp = J_vp + Dv_x @ Mx_p + Dv_y @ My_p

        J_pu = self.Dx @ dFx_du
        J_pv = self.Dy @ dFy_dv
        J_pp = self.Dx @ Mx_p + self.Dy @ My_p

        J = sparse.bmat([[J_uu, J_uv, J_up], [J_vu, J_vv, J_vp], [J_pu, J_pv, J_pp]], format="csr")
        return J

    def mass_flow(self, st: StateArrays) -> tuple[float, float]:
        """inlet / outlet 質量流量 [kg/s]（正 = 流入）."""
        m_in = float(st.fx[0][self.inlet_mask].sum())
        m_out = float(-st.fx[0][self.outlet_mask].sum())
        return m_in, m_out
