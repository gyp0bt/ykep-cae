"""PETSc 駆動の Newton + 擬似時間（SER）ソルバー.

nsb.solver.solve_steady と同じ制御則（Stokes 参照場、古典形出発 cfl_init·|R_ref|/|R_init|、
乗法形 SER、線形解が実質失敗したステップの棄却）を、部品だけ PETSc に置き換えたもの:

  [分割]     DMDA(dof=3, box ステンシル幅 2) が格子を MPI ランクに配り、ゴースト交換をする
  [残差]     各ランクが自分のゴースト付きパッチで numba カーネル（nsbp.kernels）を回す
  [ヤコビアン] SNES の FD カラーリング（DMDA の色分け、幅 2 box で 75 色）で **リミター込みの厳密 J** を
             残差 75 回で組む。nsb の JFNK（差分 matvec の非線形雑音で GMRES が空回り）を根から消す
  [線形]     FGMRES（右前処理）+ PCFIELDSPLIT Schur（運動量 bjacobi/ILU、Schur は selfp + hypre BoomerAMG）。
             nsb の SIMPLE 型前処理（ILU + SA-AMG）と同じ構造を C / MPI で
  [擬似時間] 対角 ρV/Δτ は J_steady に足すだけ（残差にも入れる dual-time 型、nsb と同じ）

ランク 0 だけがログを出す。結果の場は全ランクに集約して返す。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from petsc4py import PETSc

from nsb.core import NSBInput
from nsb.data import ConvectionSchemeType
from nsbp.kernels import dtau_patch, limiter_psi, residual_patch
from nsbp.problem import Patch, PatchCoefficients, make_discretization

LogFn = Callable[[str], None]
PC_KINDS = ("schur", "bjacobi", "asm", "lu")


@dataclass(frozen=True)
class NSBPSettings:
    """制御則と PETSc 部品の設定.

    Parameters
    ----------
    convection, venkat_k
        残差の対流スキーム "sou"/"fou" と Venkatakrishnan 定数（nsb と同じ）
    cfl_init, cfl_max, ser_growth, ser_shrink, reject_lin_ratio, local_dtau, velocity_floor_ratio
        擬似時間 SER の制御則（nsb.core.NSBSettings と同じ意味・同じ既定値）
    newton_tol, newton_max_iter, divergence_ratio
        収束判定 |R|/|R_ref| と反復上限、発散判定
    ksp_rtol, ksp_restart, ksp_max_it
        FGMRES の相対許容・再出発次元・総反復上限
    pc
        "asm"（既定。重なり 2 の加法 Schwarz + ILU(2)。逐次では ILU(2) そのもの。8 ランクで KSP 反復を 1.5 倍に抑える）/
        "bjacobi"（重なり無し。分割で圧力の大域結合が切れ 4 ランクで KSP 反復 6 倍、並列で速くならない）/
        "schur"（PCFIELDSPLIT Schur: 運動量 bjacobi/ILU、Schur 補元 selfp + hypre。flat では動くが uturn の
        Brinkman 抗力の 1e4 倍のコントラストで hypre の V サイクルが不安定になり KSP が壊れる）/ "lu"（直接法、逐次のみ）
    schur_fact
        Schur 分解の形 "full"/"upper"/"lower"/"diag"（PETSc の pc_fieldsplit_schur_fact_type）
    schur_amg
        Schur 補元近似 Ŝ = D − C diag(A)⁻¹ B に当てる AMG "hypre"（BoomerAMG）/ "gamg"
    jacobian_lag
        J_steady を組み直す間隔（Newton 反復数）。1 で毎反復（厳密 Newton）。2 以上は修正 Newton
    pc_lag
        前処理の setup（ILU 分解 + AMG 階層）を使い回す反復数。1 で毎反復組み直し
    limiter_freeze_tol
        定常残差 |R|/|R_ref| がこれを割ったら Venkatakrishnan の ψ をその時点の場で凍結する（0 で無効）。
        リミターの分岐切替に厳密ヤコビアンの Newton が追従して 1e-4〜1e-5 で振動・停滞するのを止める
        （1 次風上やリミター無し 2 次風上なら 1e-6 まで 13〜15 反復で落ちる）。凍結後は滑らかな 2 次風上なので
        2 次収束する。凍結時点以降の ψ の変化ぶんだけ厳密なリミター解とはずれる（差は status-42 の表）
    petsc_options
        追加の PETSc オプション文字列（prefix "nsbp_" 付きで解釈。既定を上書きできる）
    petsc_options_global
        prefix なしでそのまま入れるオプション（SNES 内部の MatFDColoring は prefix を継がないので
        `-mat_fd_coloring_err 1e-5` などはこちら）
    stokes_tol, stokes_rtol
        Stokes 参照場の Newton 相対許容（|R_stokes|/|R_stokes(0)|）と KSP 相対許容。参照場は収束判定の基準
        |R_ref| = |R_NS(x_stokes)| を決めるだけなので 1e-2（|R_ref| が 1% 程度動く）で足りる。擬似時間対角のない
        鞍点系は ILU に重く、288×192 で 1e-8 まで解くと 16 s、1e-4 で 8 s、1e-2 で数 s
    """

    convection: str = "sou"
    venkat_k: float = 5.0
    cfl_init: float = 0.25
    cfl_max: float = 1.0e6
    ser_growth: float = 2.0
    ser_shrink: float = 0.1
    reject_lin_ratio: float = 0.3
    local_dtau: bool = True
    velocity_floor_ratio: float = 0.1
    newton_tol: float = 1.0e-6
    newton_max_iter: int = 80
    divergence_ratio: float = 1.0e6
    ksp_rtol: float = 1.0e-3
    ksp_restart: int = 40
    ksp_max_it: int = 200
    pc: str = "asm"
    schur_fact: str = "full"
    schur_amg: str = "hypre"
    jacobian_lag: int = 1
    pc_lag: int = 1
    limiter_freeze_tol: float = 1.0e-3
    petsc_options: str = ""
    petsc_options_global: str = ""
    stokes_tol: float = 1.0e-2
    stokes_rtol: float = 1.0e-4

    @property
    def scheme(self) -> ConvectionSchemeType:
        return {
            "sou": ConvectionSchemeType.SECOND_ORDER_UPWIND,
            "fou": ConvectionSchemeType.FIRST_ORDER_UPWIND,
        }[self.convection]


@dataclass(frozen=True)
class NSBPResult:
    """結果（全ランクに同じ場を持つ）. 意味は nsb.core.NSBResult と同じ。timings は段階別の累積秒.

    収束判定と SER は定常残差（steady_residual_history）で行う。residual_history は擬似時間項込み |R_τ| の記録
    （先頭は初期場の定常残差）で、厳密ヤコビアンでは毎ステップ小さく出るので目安にしかならない。
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
    n_jacobians: int = 0
    n_pc_setups: int = 0
    n_ksp_total: int = 0
    n_residuals: int = 0
    n_ranks: int = 1
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def rel_residual(self) -> float:
        return self.steady_residual_history[-1] / self.residual_ref

    @property
    def rel_steady_residual(self) -> float:
        return self.steady_residual_history[-1] / self.residual_ref


def _numba_threads_for(size: int) -> int:
    import numba

    ncpu = os.cpu_count() or 1
    n = max(1, min(numba.config.NUMBA_NUM_THREADS, ncpu // max(size, 1)))
    numba.set_num_threads(n)
    return n


class NSBPSolver:
    """1 問題ぶんの PETSc オブジェクト（DMDA / SNES / KSP / ベクトル）を保持して solve() する."""

    def __init__(
        self,
        inp: NSBInput,
        settings: NSBPSettings | None = None,
        comm: PETSc.Comm | None = None,
        prefix: str = "nsbp_",
    ) -> None:
        self.inp = inp
        self.s = settings or NSBPSettings()
        if self.s.pc not in PC_KINDS:
            raise ValueError(f"pc は {PC_KINDS} のいずれか: {self.s.pc!r}")
        self.comm = comm or PETSc.COMM_WORLD
        self.rank = self.comm.getRank()
        self.size = self.comm.getSize()
        self.prefix = prefix
        self.numba_threads = _numba_threads_for(self.size)
        self.disc = make_discretization(inp)
        nx, ny = inp.nx, inp.ny

        self.da = PETSc.DMDA().create(
            dim=2,
            dof=3,
            sizes=(nx, ny),
            boundary_type=(PETSc.DM.BoundaryType.NONE, PETSc.DM.BoundaryType.NONE),
            stencil_type=PETSc.DMDA.StencilType.BOX,
            stencil_width=2,
            comm=self.comm,
            setup=True,
        )
        for k, name in enumerate(("u", "v", "p")):
            self.da.setFieldName(k, name)
        (xs, xe), (ys, ye) = self.da.getRanges()
        (gxs, gxe), (gys, gye) = self.da.getGhostRanges()
        self.patch = Patch(nx, ny, xs, xe, ys, ye, gxs, gxe, gys, gye)
        self.coef = PatchCoefficients(self.disc, self.patch)
        self.nxl, self.nyl = xe - xs, ye - ys

        self.X = self.da.createGlobalVec()
        self.F = self.X.duplicate()
        self.xl = self.da.createLocalVec()
        self.J = self._create_jacobian_13pt()
        self.Jtau = self.J.duplicate()

        # 擬似時間項（所有セル、(nyl, nxl) の PETSc 配列順）
        self._tau_on = False
        self._cs = 1.0
        self._frozen = False
        self._psi_u = np.ones((self.patch.gnx, self.patch.gny))
        self._psi_v = np.ones((self.patch.gnx, self.patch.gny))
        self.tau_T = np.zeros((self.nyl, self.nxl))
        self.xprev = self.X.duplicate()

        self.snes = PETSc.SNES().create(comm=self.comm)
        self.snes.setOptionsPrefix(prefix)
        self.snes.setDM(self.da)
        self.snes.setFunction(self._form_function, self.F)
        self.snes.setJacobian(None, self.J, self.J)
        # FD カラーリングは -snes_fd_color（SNESSetFromOptions が DMDA の色分けで
        # SNESComputeJacobianDefaultColor を据える標準経路）で入れる
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOptionsPrefix(prefix)
        # KSP に DM は渡さない: 渡すと PCASM が DM の領域分割を使って KSP 反復が倍（851 → 1755）になった。
        # DMDA の幾何 MG（-pc_type mg -pc_mg_galerkin、Q0 補間、ILU 平滑化）も試したが真の残差比 24 で発散（status-42）
        self._set_default_options()
        self.snes.setFromOptions()
        self.ksp.setFromOptions()
        self.snes.setUp()

        self.counts = {"resid": 0, "jac": 0, "pc_setup": 0, "ksp": 0}
        self.last_ksp_reason = 0
        self.timings = {"resid": 0.0, "jac": 0.0, "ksp": 0.0, "stokes": 0.0}

    # ------------------------------------------------------------------
    # ヤコビアンの疎パターン
    # ------------------------------------------------------------------
    STENCIL_13 = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    def _create_jacobian_13pt(self) -> PETSc.Mat:
        """残差の真の依存（13 セル × 3 dof = 1 行 39 非零）だけを持つ AIJ を DMDA の番号付けで確保する.

        `da.createMatrix()` は box 幅 2 の 25 セル（75 非零/行）を零で埋めた構造を返し、matvec と ILU が
        倍のコストになる。残差 R(i,j) が依存するのは (0,0)、(±1,0)、(0,±1)、(±2,0)、(0,±2)、(±1,±1) の 13 セル
        （面流束の上流側外挿 + Venkatakrishnan の隣接極値 + Rhie–Chow 係数 a_P の面速度）。
        DMDA の box 幅 2 の色分けはこの部分パターンでもそのまま有効（同色の列は行を共有しない）。
        """
        nx, ny = self.inp.nx, self.inp.ny
        pt = self.patch
        dof = 3
        n_loc = self.nxl * self.nyl * dof
        # 行ごとの対角ブロック/非対角ブロックの非零数（自ランク所有セルの列は d_nnz）
        d_nnz = np.zeros(n_loc, dtype=np.int32)
        o_nnz = np.zeros(n_loc, dtype=np.int32)
        cols_per_cell: list[list[int]] = []
        lgmap = self.da.getLGMap()

        # ゴースト付きローカル番号 (i, j) → グローバル行番号
        def gidx(i: int, j: int) -> int:
            return lgmap.apply([((j - pt.gys) * pt.gnx + (i - pt.gxs)) * dof])[0]

        k = 0
        for j in range(pt.ys, pt.ye):
            for i in range(pt.xs, pt.xe):
                cols = []
                for di, dj in self.STENCIL_13:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        owned = pt.xs <= ii < pt.xe and pt.ys <= jj < pt.ye
                        g = gidx(ii, jj)
                        cols.append(g)
                        for c in range(dof):
                            if owned:
                                d_nnz[k + c] += dof
                            else:
                                o_nnz[k + c] += dof
                cols_per_cell.append(cols)
                k += dof
        A = PETSc.Mat().create(comm=self.comm)
        A.setSizes(((n_loc, None), (n_loc, None)))
        A.setBlockSize(dof)
        A.setType(PETSc.Mat.Type.AIJ)
        A.setPreallocationNNZ((d_nnz, o_nnz))
        A.setLGMap(lgmap, lgmap)
        A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # 構造を固定するため零を入れておく（FD カラーリングは既存構造の位置にだけ値を入れる）
        rs, _re = A.getOwnershipRange()
        k = 0
        for cols in cols_per_cell:
            rows = np.array([rs + k + c for c in range(dof)], dtype=np.int32)
            cidx = np.array([g + c for g in cols for c in range(dof)], dtype=np.int32)
            A.setValues(rows, cidx, np.zeros((rows.size, cidx.size)))
            k += dof
        A.assemble()
        A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        return A

    # ------------------------------------------------------------------
    # PETSc オプション
    # ------------------------------------------------------------------
    def _set_default_options(self) -> None:
        s = self.s
        o = PETSc.Options()
        pre = self.prefix
        base: dict[str, object] = {
            "ksp_type": "fgmres",
            "ksp_gmres_restart": s.ksp_restart,
            "ksp_max_it": s.ksp_max_it,
            "ksp_rtol": s.ksp_rtol,
            "ksp_atol": 0.0,
            "ksp_pc_side": "right",
            "ksp_norm_type": "unpreconditioned",
            "snes_fd_color": True,
        }
        if s.pc == "schur":
            base.update(
                {
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": s.schur_fact,
                    "pc_fieldsplit_schur_precondition": "selfp",
                    "pc_fieldsplit_0_fields": "0,1",
                    "pc_fieldsplit_1_fields": "2",
                    "fieldsplit_0_ksp_type": "preonly",
                    "fieldsplit_0_pc_type": "bjacobi",
                    "fieldsplit_0_sub_pc_type": "ilu",
                    "fieldsplit_0_sub_pc_factor_levels": 2,
                    "fieldsplit_1_ksp_type": "preonly",
                }
            )
            if s.schur_amg == "hypre":
                base.update(
                    {
                        "fieldsplit_1_pc_type": "hypre",
                        "fieldsplit_1_pc_hypre_type": "boomeramg",
                        "fieldsplit_1_pc_hypre_boomeramg_strong_threshold": 0.25,
                        "fieldsplit_1_pc_hypre_boomeramg_max_iter": 1,
                    }
                )
            elif s.schur_amg == "gamg":
                base.update({"fieldsplit_1_pc_type": "gamg", "fieldsplit_1_pc_gamg_type": "agg"})
            else:
                raise ValueError(f"schur_amg は 'hypre' か 'gamg': {s.schur_amg!r}")
        elif s.pc == "bjacobi":
            base.update({"pc_type": "bjacobi", "sub_pc_type": "ilu", "sub_pc_factor_levels": 2})
        elif s.pc == "asm":
            base.update(
                {
                    "pc_type": "asm",
                    "pc_asm_overlap": 2,
                    "sub_pc_type": "ilu",
                    "sub_pc_factor_levels": 2,
                }
            )
        elif s.pc == "lu":
            base.update({"pc_type": "lu"})
        for k, val in base.items():
            o[pre + k] = val
        if s.petsc_options:
            o.insertString(
                " ".join(
                    f"-{pre}{t.lstrip('-')}" if t.startswith("-") else t
                    for t in s.petsc_options.split()
                )
            )
        if s.petsc_options_global:
            o.insertString(s.petsc_options_global)

    # ------------------------------------------------------------------
    # 残差
    # ------------------------------------------------------------------
    def _patch_fields(self, X: PETSc.Vec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.da.globalToLocal(X, self.xl)
        arr = self.xl.getArray(readonly=True).reshape(self.patch.gny, self.patch.gnx, 3)
        u = np.ascontiguousarray(arr[:, :, 0].T)
        v = np.ascontiguousarray(arr[:, :, 1].T)
        p = np.ascontiguousarray(arr[:, :, 2].T)
        return u, v, p

    def _steady_residual_arrays(
        self, X: PETSc.Vec, cs: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """所有セルの定常残差 (r_u, r_v, r_p) を (nxl, nyl) で返す."""
        c = self.coef
        u, v, p = self._patch_fields(X)
        r_u, r_v, r_p = residual_patch(
            u,
            v,
            p,
            c.rho,
            c.mu,
            c.dx,
            c.dy,
            *c.args_bc,
            self.s.scheme is ConvectionSchemeType.FIRST_ORDER_UPWIND,
            float(self.s.venkat_k),
            cs,
            self._psi_u,
            self._psi_v,
            self._frozen,
        )
        ox, oy = self.patch.own
        return r_u[ox, oy], r_v[ox, oy], r_p[ox, oy]

    def _form_function(self, snes: PETSc.SNES, X: PETSc.Vec, F: PETSc.Vec) -> None:
        t0 = time.perf_counter()
        r_u, r_v, r_p = self._steady_residual_arrays(X, self._cs)
        f = F.getArray().reshape(self.nyl, self.nxl, 3)
        f[:, :, 0] = r_u.T
        f[:, :, 1] = r_v.T
        f[:, :, 2] = r_p.T
        if self._tau_on:
            xo = X.getArray(readonly=True).reshape(self.nyl, self.nxl, 3)
            xp = self.xprev.getArray(readonly=True).reshape(self.nyl, self.nxl, 3)
            f[:, :, 0] += self.tau_T * (xo[:, :, 0] - xp[:, :, 0])
            f[:, :, 1] += self.tau_T * (xo[:, :, 1] - xp[:, :, 1])
        self.counts["resid"] += 1
        self.timings["resid"] += time.perf_counter() - t0

    def freeze_limiter(self, X: PETSc.Vec) -> None:
        """現在の場で Venkatakrishnan ψ を計算してパッチに凍結する（以後の残差・ヤコビアンは固定 ψ）."""
        c = self.coef
        u, v, p = self._patch_fields(X)
        pu, pv = limiter_psi(u, v, p, c.args_bc, c.dx, c.dy, float(self.s.venkat_k))
        self._psi_u, self._psi_v = pu, pv
        self._frozen = True

    def residual(self, X: PETSc.Vec, F: PETSc.Vec, tau: bool, cs: float = 1.0) -> float:
        """F ← 残差（tau=True で擬似時間項込み）、戻り値は 2 ノルム."""
        self._tau_on, self._cs = tau, cs
        self.snes.computeFunction(X, F)
        self._tau_on, self._cs = False, 1.0
        return float(F.norm())

    def jacobian_steady(self, X: PETSc.Vec, cs: float = 1.0) -> None:
        """self.J ← FD カラーリングの定常ヤコビアン（擬似時間項なし）."""
        t0 = time.perf_counter()
        self._tau_on, self._cs = False, cs
        # SNESComputeJacobianDefaultColor は「X が snes の解ベクトルなら snes の F を基準値に使う」ので、
        # 解ベクトルを登録し、同じフラグで F を評価しておく（未登録だと NULL 参照で落ちる）
        self.snes.setSolution(X)
        self.snes.computeFunction(X, self.F)
        self.snes.computeJacobian(X, self.J)
        self._cs = 1.0
        self.counts["jac"] += 1
        self.timings["jac"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 場の出し入れ
    # ------------------------------------------------------------------
    def set_fields(self, X: PETSc.Vec, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
        pt = self.patch
        a = X.getArray().reshape(self.nyl, self.nxl, 3)
        a[:, :, 0] = u[pt.xs : pt.xe, pt.ys : pt.ye].T
        a[:, :, 1] = v[pt.xs : pt.xe, pt.ys : pt.ye].T
        a[:, :, 2] = p[pt.xs : pt.xe, pt.ys : pt.ye].T

    def gather_fields(self, X: PETSc.Vec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """全ランクに (nx, ny) の u, v, p を集める."""
        xn = self.da.createNaturalVec()
        self.da.globalToNatural(X, xn)
        sc, xall = PETSc.Scatter.toAll(xn)
        sc.scatter(xn, xall, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        arr = xall.getArray(readonly=True).reshape(self.inp.ny, self.inp.nx, 3)
        out = (arr[:, :, 0].T.copy(), arr[:, :, 1].T.copy(), arr[:, :, 2].T.copy())
        sc.destroy()
        xall.destroy()
        xn.destroy()
        return out

    def owned_uv(self, X: PETSc.Vec) -> tuple[np.ndarray, np.ndarray]:
        a = X.getArray(readonly=True).reshape(self.nyl, self.nxl, 3)
        return np.ascontiguousarray(a[:, :, 0].T), np.ascontiguousarray(a[:, :, 1].T)

    def allreduce_min(self, val: float) -> float:
        """全ランクの min（mpi4py に依存せず、要素 1/ランクの Vec で取る）."""
        if self.size == 1:
            return float(val)
        v = PETSc.Vec().createMPI((1, None), comm=self.comm)
        v.setValue(self.rank, float(val))
        v.assemble()
        out = float(v.min()[1])
        v.destroy()
        return out

    # ------------------------------------------------------------------
    # 線形解
    # ------------------------------------------------------------------
    def linear_solve(
        self,
        A: PETSc.Mat,
        b: PETSc.Vec,
        x: PETSc.Vec,
        rtol: float | None = None,
        reuse_pc: bool = False,
    ) -> tuple[int, bool, float]:
        """A x = b を FGMRES で解く。戻り値 (反復数, 収束フラグ, 真の残差比 |b − A x|/|b|)."""
        t0 = time.perf_counter()
        self.ksp.setOperators(A, A)
        self.ksp.getPC().setReusePreconditioner(reuse_pc)
        if not reuse_pc:
            self.counts["pc_setup"] += 1
        if rtol is not None:
            self.ksp.setTolerances(rtol=rtol)
        else:
            self.ksp.setTolerances(rtol=self.s.ksp_rtol)
        x.set(0.0)
        self.ksp.solve(b, x)
        n_it = int(self.ksp.getIterationNumber())
        reason = int(self.ksp.getConvergedReason())
        ok = reason > 0
        self.last_ksp_reason = reason
        r = b.duplicate()
        A.mult(x, r)
        r.aypx(-1.0, b)
        bn = float(b.norm())
        ratio = float(r.norm()) / bn if bn > 0.0 else 0.0
        r.destroy()
        self.counts["ksp"] += n_it
        self.timings["ksp"] += time.perf_counter() - t0
        return n_it, ok, ratio

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    def solve(self, log: LogFn | None = print) -> NSBPResult:
        t_start = time.perf_counter()
        s = self.s
        inp = self.inp
        disc = self.disc
        emit = (
            (log if log is not None else (lambda _m: None)) if self.rank == 0 else (lambda _m: None)
        )
        u_floor = s.velocity_floor_ratio * disc.u_scale
        X, F = self.X, self.F
        dX = X.duplicate()
        b = X.duplicate()

        # [参照場] Stokes–Brinkman（対流を落とした残差 cs=0）を静止場から Newton で解く。
        # nsb は 1 次風上ヤコビアン（a_P 凍結）で 1 ステップだけ解くが、FD ヤコビアンは Rhie–Chow 係数
        # d_f = V/a_P の速度依存まで含むので 1 ステップでは残差が 2 割ほど残る。数回回して収束させる
        t0 = time.perf_counter()
        X.set(0.0)
        r_init_norm = self.residual(X, F, tau=False, cs=0.0)
        r_st = r_init_norm
        n_g0 = 0
        ok0 = True
        ratio0 = 0.0
        stokes_its = 0
        while stokes_its < 6 and r_st > s.stokes_tol * r_init_norm:
            self.jacobian_steady(X, cs=0.0)
            F.copy(b)
            b.scale(-1.0)
            n_g, ok, ratio = self.linear_solve(self.J, b, dX, rtol=s.stokes_rtol)
            n_g0 += n_g
            ok0 = ok0 and ok
            ratio0 = max(ratio0, ratio)
            X.axpy(1.0, dX)
            r_st = self.residual(X, F, tau=False, cs=0.0)
            stokes_its += 1
        x_stokes = X.copy()
        r_ref = self.residual(X, F, tau=False)
        if r_ref == 0.0:
            X.set(0.0)
            r_ref = self.residual(X, F, tau=False)
        r_ref = max(r_ref, 1e-300)
        self.timings["stokes"] = time.perf_counter() - t0
        us, vs, _ = self.gather_fields(x_stokes)
        emit(
            f"[nsbp] ranks={self.size} numba_threads={self.numba_threads} grid={inp.nx}x{inp.ny} "
            f"pc={s.pc}/{s.schur_fact}/{s.schur_amg}"
        )
        emit(
            f"[nsbp] stokes ref (newton={stokes_its} ksp={n_g0}{'' if ok0 else ' not converged'} "
            f"ratio={ratio0:.1e} |R_stokes|/|R_stokes(0)|={r_st / r_init_norm:.1e}): "
            f"|R_stokes(0)|={r_init_norm:.4e} |R_ref|={r_ref:.4e} speed_max={np.hypot(us, vs).max():.3g} m/s "
            f"t={self.timings['stokes']:.2f}s"
        )

        # [初期場]
        if inp.u0 is not None or inp.v0 is not None or inp.p0 is not None:
            shape = (inp.nx, inp.ny)
            u0 = np.zeros(shape) if inp.u0 is None else np.asarray(inp.u0, dtype=float)
            v0 = np.zeros(shape) if inp.v0 is None else np.asarray(inp.v0, dtype=float)
            p0 = np.zeros(shape) if inp.p0 is None else np.asarray(inp.p0, dtype=float)
            self.set_fields(X, u0, v0, p0)
            init_how = "u0/v0/p0"
        else:
            x_stokes.copy(X)
            init_how = "stokes"

        r_norm = self.residual(X, F, tau=False)
        r0 = r_ref
        if init_how != "stokes" and r_norm > r_ref:
            # 与えられた初期場が参照場より悪い（uturn の入れ子で双一次補間が閉塞セルへ流速を持ち込み
            # 抗力残差が参照の 100 倍超になる等）なら Stokes 発進に戻す。古典形出発 CFL が 1e-3 まで
            # 落ちた状態で ILU が壊れ、棄却が連鎖して回復しない（status-42）
            emit(
                f"[nsbp] init={init_how} is worse than the Stokes field (rel={r_norm / r0:.3e}), fall back to stokes"
            )
            x_stokes.copy(X)
            r_norm = self.residual(X, F, tau=False)
            init_how = "stokes(fallback)"
        cfl = float(min(s.cfl_max, s.cfl_init * r0 / max(r_norm, 1e-300)))
        hist = [r_norm]
        hist_steady = [r_norm]
        cfl_hist: list[float] = []
        converged = False
        failure = ""
        n_iter = 0
        jac_age = 10**9
        pc_age = 10**9
        emit(f"[nsbp] it=0 init={init_how} |R|={r_norm:.4e} rel={r_norm / r0:.3e} cfl={cfl:.3g}")

        while n_iter < s.newton_max_iter:
            if not np.isfinite(r_norm):
                failure = "nan"
                break
            if r_norm / r0 < s.newton_tol:
                converged = True
                break
            if r_norm / r0 > s.divergence_ratio:
                failure = "diverged"
                break

            # ---- [リミター凍結] 定常残差が閾値を割ったら ψ を固定して滑らかな問題にする ----
            if (
                s.limiter_freeze_tol > 0.0
                and not self._frozen
                and s.convection == "sou"
                and r_norm / r0 < s.limiter_freeze_tol
            ):
                self.freeze_limiter(X)
                r_norm = self.residual(X, F, tau=False)
                jac_age = 10**9
                emit(f"[nsbp] it={n_iter} limiter frozen (rel={r_norm / r0:.3e} after freeze)")

            # ---- 擬似時間ステップ: Δτ を決めて凍結 ----
            uu, vv = self.owned_uv(X)
            dtau = dtau_patch(uu, vv, disc.dx, disc.dy, cfl, u_floor)
            if not s.local_dtau:
                dtau = np.full_like(dtau, self.allreduce_min(float(dtau.min())))
            dt_min = self.allreduce_min(float(dtau.min())) if self.size > 1 else float(dtau.min())
            dt_max = -self.allreduce_min(-float(dtau.max())) if self.size > 1 else float(dtau.max())
            self.tau_T = (inp.rho * disc.vol / dtau).T.copy()
            X.copy(self.xprev)

            # ---- 残差 R_τ と (J_steady + diag) ----
            self.residual(X, F, tau=True)
            if jac_age >= max(1, s.jacobian_lag):
                self.jacobian_steady(X)
                jac_age = 0
            self.J.copy(self.Jtau, PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            dvec = X.duplicate()
            da_ = dvec.getArray().reshape(self.nyl, self.nxl, 3)
            da_[:, :, 0] = self.tau_T
            da_[:, :, 1] = self.tau_T
            da_[:, :, 2] = 0.0
            self.Jtau.setDiagonal(dvec, PETSc.InsertMode.ADD_VALUES)
            dvec.destroy()
            F.copy(b)
            b.scale(-1.0)
            reuse = pc_age < max(1, s.pc_lag)
            n_ksp, lin_ok, lin_ratio = self.linear_solve(self.Jtau, b, dX, reuse_pc=reuse)
            pc_age = pc_age + 1 if reuse else 1
            jac_age += 1
            if not np.isfinite(lin_ratio) or not np.isfinite(float(dX.norm())):
                failure = "ksp_breakdown"
                break
            if s.reject_lin_ratio > 0.0 and not (lin_ratio <= s.reject_lin_ratio):
                n_iter += 1
                cfl = float(cfl * s.ser_shrink)
                cfl_hist.append(cfl)
                pc_age = 10**9  # 次は組み直す
                emit(
                    f"[nsbp] it={n_iter} linear solve failed (ksp={n_ksp} reason={self.last_ksp_reason}, "
                    f"|b-Ax|/|b|={lin_ratio:.2e}), rejected, cfl -> {cfl:.3g}"
                )
                continue
            X.axpy(1.0, dX)
            n_iter += 1
            r_new = self.residual(X, F, tau=True)
            r_steady_new = self.residual(X, F, tau=False)
            hist.append(r_new)
            hist_steady.append(r_steady_new)
            emit(
                f"[nsbp] it={n_iter} |R_tau|={r_new:.4e} rel={r_new / r0:.3e} "
                f"|R_steady|/|R_ref|={r_steady_new / r0:.3e} cfl={cfl:.3g} "
                f"dtau=[{dt_min:.2e},{dt_max:.2e}] ksp={n_ksp} jac_age={jac_age} pc_age={pc_age}"
                f"{'' if lin_ok else f' (ksp not converged: {lin_ratio:.1e} reason={self.last_ksp_reason})'}"
            )
            if not np.isfinite(r_new) or not np.isfinite(r_steady_new):
                r_norm = float("nan")
                continue
            # ---- [SER] 定常残差の比で CFL を更新 ----
            # 厳密ヤコビアンでは擬似時間込み残差 |R_τ| は毎ステップ 1e-2〜1e-3 倍に落ちる（擬似ステップ内の
            # Newton がほぼ厳密に解けるため）ので、制御・収束判定には Δτ 非依存の定常残差を使う。
            # nsb（JFNK、gmres_tol 1e-3）では |R_τ| と定常残差が CFL 成長とともに一致していくので |R_τ| で足りていた
            ratio = r_norm / r_steady_new if r_steady_new > 0.0 else s.ser_shrink
            cfl = float(min(s.cfl_max, cfl * float(np.clip(ratio, s.ser_shrink, s.ser_growth))))
            cfl_hist.append(cfl)
            r_norm = r_steady_new
        else:
            if np.isfinite(r_norm) and r_norm / r0 < s.newton_tol:
                converged = True
            elif not np.isfinite(r_norm):
                failure = "nan"
            else:
                failure = "max_iter"

        u, v, p = self.gather_fields(X)
        x_full = np.concatenate([u.ravel(), v.ravel(), p.ravel()])
        m_in, m_out = disc.mass_flow(disc.compute_state(x_full, s.scheme, s.venkat_k), x_full)
        elapsed = time.perf_counter() - t_start
        tm = dict(self.timings)
        tm["total"] = elapsed
        emit(
            f"[nsbp] done converged={converged} reason='{failure}' it={n_iter} "
            f"m_in={m_in:.4e} m_out={m_out:.4e} jacobians={self.counts['jac']} pc_setups={self.counts['pc_setup']} "
            f"ksp_total={self.counts['ksp']} residuals={self.counts['resid']} "
            f"t[resid={tm['resid']:.1f} jac={tm['jac']:.1f} ksp={tm['ksp']:.1f} stokes={tm['stokes']:.1f}] "
            f"elapsed={elapsed:.1f}s"
        )
        dX.destroy()
        b.destroy()
        x_stokes.destroy()
        return NSBPResult(
            u=u,
            v=v,
            p=p,
            converged=converged,
            failure_reason=failure,
            n_iter=n_iter,
            residual_history=tuple(hist),
            steady_residual_history=tuple(hist_steady),
            cfl_history=tuple(cfl_hist),
            mass_in=float(m_in),
            mass_out=float(m_out),
            elapsed=elapsed,
            residual_ref=r_ref,
            n_jacobians=self.counts["jac"],
            n_pc_setups=self.counts["pc_setup"],
            n_ksp_total=self.counts["ksp"],
            n_residuals=self.counts["resid"],
            n_ranks=self.size,
            timings=tm,
        )

    def destroy(self) -> None:
        for obj in (
            self.ksp,
            self.snes,
            self.Jtau,
            self.J,
            self.xl,
            self.F,
            self.X,
            self.xprev,
            self.da,
        ):
            obj.destroy()


def solve_steady(
    inp: NSBInput, settings: NSBPSettings | None = None, log: LogFn | None = print
) -> NSBPResult:
    """NSBInput（settings は無視し、nsbp 側の settings を使う）を PETSc で解く."""
    solver = NSBPSolver(inp, settings)
    try:
        return solver.solve(log)
    finally:
        solver.destroy()
