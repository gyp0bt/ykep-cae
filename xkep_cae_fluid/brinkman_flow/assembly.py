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

from xkep_cae_fluid.brinkman_flow.data import (
    BoundaryKind,
    BoundaryPatch,
    BrinkmanFlowInput,
    ConvectionSchemeType,
)


@dataclass
class BoundarySide:
    """領域 1 辺の境界面配列（W/E: 長さ ny、S/N: 長さ nx）.

    un: 内向き法線方向の流入速度（inlet 以外は 0）、p: outlet 圧力（他は未使用）。
    is_outlet の面は速度ゼロ勾配・圧力 Dirichlet、それ以外は速度 Dirichlet・圧力ゼロ勾配。
    """

    kind: np.ndarray  # object 配列 (BoundaryKind)
    is_outlet: np.ndarray
    un: np.ndarray
    p: np.ndarray
    x: np.ndarray  # 面中心座標
    y: np.ndarray

    @property
    def is_dirichlet(self) -> np.ndarray:
        return ~self.is_outlet

    @property
    def is_inlet(self) -> np.ndarray:
        return np.array(
            [k in (BoundaryKind.VELOCITY_INLET, BoundaryKind.MASS_FLOW_INLET) for k in self.kind]
        )


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

        patches = inp.effective_boundaries()
        self.sides = self._resolve_boundaries(tuple(b for b in patches if not b.is_interior))
        self._resolve_interior(tuple(b for b in patches if b.is_interior))
        W, E, S, N = self.sides["W"], self.sides["E"], self.sides["S"], self.sides["N"]
        n_in = sum(int(sd.is_inlet.sum()) for sd in self.sides.values()) + int(
            (self.q_src > 0.0).sum()
        )
        n_out = sum(int(sd.is_outlet.sum()) for sd in self.sides.values()) + int(
            ((self.q_sink > 0.0) | (self.c_sink > 0.0)).sum()
        )
        if n_in == 0 or n_out == 0:
            raise ValueError(
                "inlet / outlet に対応する境界面・セルが存在しません（マスクか分割数を確認）"
            )
        if (
            not any(sd.is_outlet.any() for sd in self.sides.values())
            and not (self.c_sink > 0).any()
        ):
            raise ValueError(
                "圧力の基準がありません: PRESSURE_OUTLET か INTERIOR_PRESSURE_SINK が必要です"
            )
        # 擬似時間の速度スケール（最大流入速度。領域内ソースは周長 4√A から見積もる）
        u_b = max(float(np.abs(sd.un).max()) for sd in self.sides.values())
        self.u_scale = max(u_b, self._interior_velocity_scale())

        # 境界面の速度成分（W: u=+un, E: u=-un, S: v=+un, N: v=-un）
        self.u_w, self.v_w = W.un, np.zeros(self.ny)
        self.u_e, self.v_e = -E.un, np.zeros(self.ny)
        self.u_s, self.v_s = np.zeros(self.nx), S.un
        self.u_n, self.v_n = np.zeros(self.nx), -N.un

        dxx = self.mu * self.dy / self.dx
        dyy = self.mu * self.dx / self.dy
        diff_diag = np.full((self.nx, self.ny), 2.0 * dxx + 2.0 * dyy)
        # 境界面: Dirichlet は 2μA/d（内部の 2 倍）、outlet（ゼロ勾配）は 0
        diff_diag[0, :] += np.where(W.is_dirichlet, dxx, -dxx)
        diff_diag[-1, :] += np.where(E.is_dirichlet, dxx, -dxx)
        diff_diag[:, 0] += np.where(S.is_dirichlet, dyy, -dyy)
        diff_diag[:, -1] += np.where(N.is_dirichlet, dyy, -dyy)
        self.diff_diag = diff_diag
        self._dxx, self._dyy = dxx, dyy

        self._build_operators()

    # ------------------------------------------------------------------
    # 境界パッチの解決
    # ------------------------------------------------------------------
    def _resolve_boundaries(self, patches: tuple[BoundaryPatch, ...]) -> dict[str, BoundarySide]:
        """座標マスクを 4 辺の境界面中心で評価し、面ごとの種別・流入速度・圧力に展開する.

        MASS_FLOW_INLET は u_n = mass_flow / (ρ Σ_f h_f A_f)（h_f は隣接セルの厚さ）で
        パッチ内一様の流入速度に換算する。
        """
        nx, ny = self.nx, self.ny
        lx, ly = self.inp.geometry.lx, self.inp.geometry.ly
        xc = (np.arange(nx) + 0.5) * self.dx
        yc = (np.arange(ny) + 0.5) * self.dy
        h = self.inp.thickness
        # (x, y, 隣接セル厚さ, 面積)
        geom = {
            "W": (np.zeros(ny), yc, h[0, :], self.dy),
            "E": (np.full(ny, lx), yc, h[-1, :], self.dy),
            "S": (xc, np.zeros(nx), h[:, 0], self.dx),
            "N": (xc, np.full(nx, ly), h[:, -1], self.dx),
        }
        sides: dict[str, BoundarySide] = {}
        for key, (x, y, _hf, _a) in geom.items():
            m = x.size
            sides[key] = BoundarySide(
                kind=np.array([BoundaryKind.WALL] * m, dtype=object),
                is_outlet=np.zeros(m, dtype=bool),
                un=np.zeros(m),
                p=np.zeros(m),
                x=x,
                y=y,
            )
        for patch in patches:
            hits = {
                key: np.asarray(patch.mask(x, y), dtype=bool) for key, (x, y, _, _) in geom.items()
            }
            if patch.kind is BoundaryKind.MASS_FLOW_INLET:
                area_h = sum(float((geom[k][2][hit] * geom[k][3]).sum()) for k, hit in hits.items())
                if area_h <= 0.0:
                    raise ValueError(
                        f"MASS_FLOW_INLET '{patch.name}' のマスクに一致する境界面がありません"
                    )
                un = patch.mass_flow / (self.rho * area_h)
            elif patch.kind is BoundaryKind.VELOCITY_INLET:
                un = patch.velocity
            else:
                un = 0.0
            for key, hit in hits.items():
                sd = sides[key]
                sd.kind[hit] = patch.kind
                sd.is_outlet[hit] = patch.kind is BoundaryKind.PRESSURE_OUTLET
                sd.un[hit] = un
                sd.p[hit] = patch.pressure if patch.kind is BoundaryKind.PRESSURE_OUTLET else 0.0
        return sides

    def _resolve_interior(self, patches: tuple[BoundaryPatch, ...]) -> None:
        """領域内マニホールド（セル中心でマスク評価）を単位深さのセルソースに展開する.

        q_src  : 注入 [kg/s]（面内運動量ゼロ）
        q_sink : 質量流量指定の吸出 [kg/s]（局所運動量を持ち出す）
        c_sink : 圧力指定マニホールドの単位深さコンダクタンス [kg/(s·Pa)]、p_sink: その基準圧力
        いずれも 3 次元値 X を X · V_c / Σ_c h_c V_c で按分（Σ は同一パッチ内）。
        """
        nx, ny = self.nx, self.ny
        xc = (np.arange(nx) + 0.5) * self.dx
        yc = (np.arange(ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        h = self.inp.thickness
        self.q_src = np.zeros((nx, ny))
        self.q_sink = np.zeros((nx, ny))
        self.c_sink = np.zeros((nx, ny))
        self.cp_sink = np.zeros((nx, ny))  # Σ_k c_k p_k（複数の圧力指定パッチが重なっても可）
        self.interior_mask = np.zeros((nx, ny), dtype=bool)
        for patch in patches:
            w = patch.weights(X, Y)
            hv = float((w * h * self.vol).sum())
            if hv <= 0.0:
                raise ValueError(
                    f"領域内パッチ '{patch.name}' のマスク/重みに一致するセルがありません"
                )
            share = w * self.vol / hv
            # 領域内パッチは加算（滑らかな窓は裾が重なるため、上書きではなく重ね合わせ）
            if patch.kind is BoundaryKind.INTERIOR_MASS_SOURCE:
                self.q_src += patch.mass_flow * share
            elif patch.kind is BoundaryKind.INTERIOR_MASS_SINK:
                self.q_sink += patch.mass_flow * share
            else:
                self.c_sink += patch.conductance * share
                self.cp_sink += patch.conductance * share * patch.pressure
            self.interior_mask |= w > 1e-3
        with np.errstate(invalid="ignore", divide="ignore"):
            self.p_sink = np.where(self.c_sink > 0.0, self.cp_sink / self.c_sink, 0.0)

    def _interior_velocity_scale(self) -> float:
        """領域内ソースの速度スケール: 総流量 / (ρ · 周長 4√A)。ソースが無ければ 0."""
        core = (
            self.q_src > 1e-3 * self.q_src.max()
            if self.q_src.max() > 0
            else np.zeros_like(self.q_src, bool)
        )
        if not core.any():
            return 0.0
        area = float(core.sum()) * self.vol
        return float(self.q_src.sum()) / (self.rho * 4.0 * np.sqrt(area))

    def interior_fluxes(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(注入 q_in ≥ 0, 吸出 q_out ≥ 0) [kg/s]（単位深さ、セルごと）.

        圧力指定マニホールドは q = c (p - p_sink) で、正なら吸出、負なら注入（運動量ゼロ）。
        """
        q_c = self.c_sink * p - self.cp_sink
        q_in = self.q_src + np.maximum(-q_c, 0.0)
        q_out = self.q_sink + np.maximum(q_c, 0.0)
        return q_in, q_out

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

        W, E, S, N = self.sides["W"], self.sides["E"], self.sides["S"], self.sides["N"]

        # 速度の面補間（線形部）: 内部 0.5/0.5、outlet 面はセル値（ゼロ勾配）、Dirichlet 面は定数（行列は 0）
        self.Ux = self.Ax + sum(
            mat(f[m], c[m], np.ones(int(m.sum())), (nfx, n))
            for f, c, m in ((fx_left, c_left, W.is_outlet), (fx_right, c_right, E.is_outlet))
        )
        self.Uy = self.Ay + sum(
            mat(f[m], c[m], np.ones(int(m.sum())), (nfy, n))
            for f, c, m in ((fy_bot, c_bot, S.is_outlet), (fy_top, c_top, N.is_outlet))
        )

        # 圧力の面補間: 内部 0.5/0.5、壁/inlet はセル値（ゼロ勾配）、outlet は定数（行列は 0）
        self.Px = self.Ax + sum(
            mat(f[m], c[m], np.ones(int(m.sum())), (nfx, n))
            for f, c, m in ((fx_left, c_left, W.is_dirichlet), (fx_right, c_right, E.is_dirichlet))
        )
        self.Py = self.Ay + sum(
            mat(f[m], c[m], np.ones(int(m.sum())), (nfy, n))
            for f, c, m in ((fy_bot, c_bot, S.is_dirichlet), (fy_top, c_top, N.is_dirichlet))
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

        # 速度の面勾配（拡散用、Dirichlet 境界は ±2/d、outlet は 0）
        self.Fgx_vel = (
            self.Fgx_int
            + mat(
                fx_left[W.is_dirichlet],
                c_left[W.is_dirichlet],
                np.full(int(W.is_dirichlet.sum()), 2.0 / self.dx),
                (nfx, n),
            )
            + mat(
                fx_right[E.is_dirichlet],
                c_right[E.is_dirichlet],
                np.full(int(E.is_dirichlet.sum()), -2.0 / self.dx),
                (nfx, n),
            )
        )
        self.Fgy_vel = (
            self.Fgy_int
            + mat(
                fy_bot[S.is_dirichlet],
                c_bot[S.is_dirichlet],
                np.full(int(S.is_dirichlet.sum()), 2.0 / self.dy),
                (nfy, n),
            )
            + mat(
                fy_top[N.is_dirichlet],
                c_top[N.is_dirichlet],
                np.full(int(N.is_dirichlet.sum()), -2.0 / self.dy),
                (nfy, n),
            )
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
        """線形補間の面値。境界面: outlet は速度セル値・圧力指定値、他は速度指定値・圧力セル値."""
        nx, ny = self.nx, self.ny
        W, E, S, N = self.sides["W"], self.sides["E"], self.sides["S"], self.sides["N"]
        ufx = np.empty((nx + 1, ny))
        vfx = np.empty((nx + 1, ny))
        ufx[1:-1] = 0.5 * (u[:-1] + u[1:])
        vfx[1:-1] = 0.5 * (v[:-1] + v[1:])
        ufx[0] = np.where(W.is_outlet, u[0], self.u_w)
        vfx[0] = np.where(W.is_outlet, v[0], self.v_w)
        ufx[-1] = np.where(E.is_outlet, u[-1], self.u_e)
        vfx[-1] = np.where(E.is_outlet, v[-1], self.v_e)
        ufy = np.empty((nx, ny + 1))
        vfy = np.empty((nx, ny + 1))
        ufy[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
        vfy[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
        ufy[:, 0] = np.where(S.is_outlet, u[:, 0], self.u_s)
        vfy[:, 0] = np.where(S.is_outlet, v[:, 0], self.v_s)
        ufy[:, -1] = np.where(N.is_outlet, u[:, -1], self.u_n)
        vfy[:, -1] = np.where(N.is_outlet, v[:, -1], self.v_n)
        pfx = np.empty((nx + 1, ny))
        pfx[1:-1] = 0.5 * (p[:-1] + p[1:])
        pfx[0] = np.where(W.is_outlet, W.p, p[0])
        pfx[-1] = np.where(E.is_outlet, E.p, p[-1])
        pfy = np.empty((nx, ny + 1))
        pfy[:, 1:-1] = 0.5 * (p[:, :-1] + p[:, 1:])
        pfy[:, 0] = np.where(S.is_outlet, S.p, p[:, 0])
        pfy[:, -1] = np.where(N.is_outlet, N.p, p[:, -1])
        return ufx, vfx, ufy, vfy, pfx, pfy

    def compute_state(
        self,
        x: np.ndarray,
        scheme: ConvectionSchemeType,
        venkat_k: float,
        pseudo_diag: np.ndarray | None = None,
    ) -> StateArrays:
        """面速度・RC 質量流束・対流面値を計算する.

        pseudo_diag に (nx, ny) の擬似時間対角 ρV/Δτ を渡すと、RC 係数を
        d_f = V/(a_P + ρV/Δτ) として組む（既定 None: d_f = V/a_P）。
        """
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
        d_cell = vol / a_p if pseudo_diag is None else vol / (a_p + pseudo_diag)
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

    def residual_from_state(
        self, x: np.ndarray, st: StateArrays, convection: bool = True
    ) -> np.ndarray:
        """残差 [R_u, R_v, R_p]。convection=False で運動量の対流項を落とす（Stokes–Brinkman）."""
        u, v, p = self.split(x)
        dx, dy, mu, vol = self.dx, self.dy, self.mu, self.vol
        cs = 1.0 if convection else 0.0

        def div(fxv: np.ndarray, fyv: np.ndarray) -> np.ndarray:
            return fxv[1:] - fxv[:-1] + fyv[:, 1:] - fyv[:, :-1]

        W, E, S, N = self.sides["W"], self.sides["E"], self.sides["S"], self.sides["N"]

        def diffusion(
            phi: np.ndarray, bw: np.ndarray, be: np.ndarray, bs: np.ndarray, bn: np.ndarray
        ) -> np.ndarray:
            # 面勾配（+x, +y 方向）。Dirichlet 面は (境界値 - セル値)/(d/2)、outlet 面は 0
            gxf = np.empty((self.nx + 1, self.ny))
            gxf[1:-1] = (phi[1:] - phi[:-1]) / dx
            gxf[0] = np.where(W.is_dirichlet, (phi[0] - bw) / (0.5 * dx), 0.0)
            gxf[-1] = np.where(E.is_dirichlet, (be - phi[-1]) / (0.5 * dx), 0.0)
            gyf = np.empty((self.nx, self.ny + 1))
            gyf[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / dy
            gyf[:, 0] = np.where(S.is_dirichlet, (phi[:, 0] - bs) / (0.5 * dy), 0.0)
            gyf[:, -1] = np.where(N.is_dirichlet, (bn - phi[:, -1]) / (0.5 * dy), 0.0)
            return mu * div(dy * gxf, dx * gyf)  # 流入側が正

        # 領域内マニホールド: 連続式に -q_in + q_out、吸出は局所運動量 q_out u_i を持ち出す
        q_in, q_out = self.interior_fluxes(p)
        r_u = (
            cs * div(st.fx * st.conv_ufx, st.fy * st.conv_ufy)
            - diffusion(u, self.u_w, self.u_e, self.u_s, self.u_n)
            + (st.pfx[1:] - st.pfx[:-1]) * dy
            + self.drag * vol * u
            + q_out * u
        )
        r_v = (
            cs * div(st.fx * st.conv_vfx, st.fy * st.conv_vfy)
            - diffusion(v, self.v_w, self.v_e, self.v_s, self.v_n)
            + (st.pfy[:, 1:] - st.pfy[:, :-1]) * dx
            + self.drag * vol * v
            + q_out * v
        )
        r_p = div(st.fx, st.fy) - q_in + q_out
        return np.concatenate([r_u.ravel(), r_v.ravel(), r_p.ravel()])

    # ------------------------------------------------------------------
    # 1 次風上ヤコビアン
    # ------------------------------------------------------------------
    def jacobian_first_order(
        self,
        st: StateArrays,
        newton_convection: bool = True,
        convection: bool = True,
        x: np.ndarray | None = None,
    ) -> sparse.csr_matrix:
        """1 次風上・RC 係数凍結のヤコビアン（3N×3N、ブロック順 [u, v, p]）.

        convection=False で運動量の対流項（Newton 項含む）を落とす（Stokes–Brinkman 用）。
        """
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
        # outlet 面: φ_f = φ_P（ゼロ勾配）
        W, E, S, N = self.sides["W"], self.sides["E"], self.sides["S"], self.sides["N"]
        j_all, i_all = np.arange(self.ny), np.arange(self.nx)
        for faces, cellids in (
            (
                j_all[W.is_outlet],
                self._cell(np.zeros(int(W.is_outlet.sum()), dtype=int), j_all[W.is_outlet]),
            ),
            (
                self.nx * self.ny + j_all[E.is_outlet],
                self._cell(np.full(int(E.is_outlet.sum()), self.nx - 1), j_all[E.is_outlet]),
            ),
        ):
            Wx = Wx + sparse.csr_matrix((np.ones(len(faces)), (faces, cellids)), shape=(nfx, n))
        up_y = fy[self.fy_int] >= 0.0
        Wy = sparse.csr_matrix(
            (np.ones(len(self.fy_int)), (self.fy_int, np.where(up_y, self.fy_cs, self.fy_cn))),
            shape=(nfy, n),
        )
        for faces, cellids in (
            (
                i_all[S.is_outlet] * (self.ny + 1),
                self._cell(i_all[S.is_outlet], np.zeros(int(S.is_outlet.sum()), dtype=int)),
            ),
            (
                i_all[N.is_outlet] * (self.ny + 1) + self.ny,
                self._cell(i_all[N.is_outlet], np.full(int(N.is_outlet.sum()), self.ny - 1)),
            ),
        ):
            Wy = Wy + sparse.csr_matrix((np.ones(len(faces)), (faces, cellids)), shape=(nfy, n))
        conv = self.Dx @ sparse.diags(fx) @ Wx + self.Dy @ sparse.diags(fy) @ Wy

        # RC 質量流束の p 依存: ∂Fx/∂p = -ρ dy diag(dfx) (Fgx_int - Ax Gx)
        Mx_p = -rho * dy * sparse.diags(st.dfx.ravel()) @ (self.Fgx_int - self.Ax @ self.Gx)
        My_p = -rho * dx * sparse.diags(st.dfy.ravel()) @ (self.Fgy_int - self.Ay @ self.Gy)
        dFx_du = rho * dy * self.Ux
        dFy_dv = rho * dx * self.Uy

        base = (conv if convection else sparse.csr_matrix((n, n))) - self.Ldiff + self.drag_v
        J_up = dy * (self.Dx @ self.Px)
        J_vp = dx * (self.Dy @ self.Py)
        J_uu = base
        J_vv = base
        J_uv = sparse.csr_matrix((n, n))
        J_vu = sparse.csr_matrix((n, n))
        if newton_convection and convection:
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

        # 領域内マニホールド（x が無ければ p=0 として評価）
        if self.interior_mask.any():
            if x is None:
                u_, v_, p_ = np.zeros((3, n))
            else:
                u_, v_, p_ = (a.ravel() for a in self.split(x))
            q_in, q_out = self.interior_fluxes(p_.reshape(self.nx, self.ny))
            q_out = q_out.ravel()
            c = self.c_sink.ravel()
            q_c = c * p_ - self.cp_sink.ravel()
            out_active = (q_c > 0.0).astype(float)  # 吸出側のみ運動量項
            J_uu = J_uu + sparse.diags(q_out)
            J_vv = J_vv + sparse.diags(q_out)
            J_up = J_up + sparse.diags(c * out_active * u_)
            J_vp = J_vp + sparse.diags(c * out_active * v_)
            J_pp = J_pp + sparse.diags(c)

        J = sparse.bmat([[J_uu, J_uv, J_up], [J_vu, J_vv, J_vp], [J_pu, J_pv, J_pp]], format="csr")
        return J

    def mass_flow(self, st: StateArrays, x: np.ndarray | None = None) -> tuple[float, float]:
        """inlet / outlet 質量流量 [kg/s]（単位深さ、正 = 流入 / 流出）。領域内マニホールド分を含む.

        圧力指定マニホールドの流量には p が要るので x を渡す（無ければ p=0 で評価）。
        """
        # 各辺の内向き流束
        inward = {
            "W": st.fx[0],
            "E": -st.fx[-1],
            "S": st.fy[:, 0],
            "N": -st.fy[:, -1],
        }
        m_in = sum(float(inward[k][sd.is_inlet].sum()) for k, sd in self.sides.items())
        m_out = sum(float(-inward[k][sd.is_outlet].sum()) for k, sd in self.sides.items())
        if self.interior_mask.any():
            p = np.zeros((self.nx, self.ny)) if x is None else self.split(x)[2]
            q_in, q_out = self.interior_fluxes(p)
            m_in += float(q_in.sum())
            m_out += float(q_out.sum())
        return m_in, m_out

    def boundary_report(self) -> list[dict[str, object]]:
        """辺ごとの境界面種別と流入速度の要約（デバッグ用）."""
        out = []
        for key, sd in self.sides.items():
            for kind in BoundaryKind:
                m = np.array([k is kind for k in sd.kind])
                if kind is BoundaryKind.WALL or not m.any():
                    continue
                out.append(
                    {
                        "side": key,
                        "kind": kind.value,
                        "n_faces": int(m.sum()),
                        "u_n": float(np.abs(sd.un[m]).max()),
                        "span": (
                            float(sd.x[m].min()),
                            float(sd.x[m].max()),
                            float(sd.y[m].min()),
                            float(sd.y[m].max()),
                        ),
                    }
                )
        return out
