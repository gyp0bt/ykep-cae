"""面流束ベースの粒子追跡（Pollock 型）を非構造メッシュ上で行う PostProcess.

構造格子の :class:`~xkep_cae_fluid.extruder.tracker.ParticleTrackerProcess` は
節点流れ関数 ψ の双一次補間で発散ゼロを担保していたが、ψ は 2 次元の構造格子でしか
作れない。汎用記法（``*NODE`` / ``*ELEMENT``）で書いた .inp の非構造メッシュでは、
**セル中心速度を補間するのではなく面流束から速度場を再構成する**。

セル内の再構成
--------------
セル ``c`` の内部で速度をアフィン場

    u(x) = a_c + B_c (x − x_c)

とし、**そのセルの全ての面について流束を厳密に再現する**ように係数を決める:

    (a_c + B_c d_f)·S_f = q_f     （d_f = x_f − x_c、S_f = 外向き面ベクトル、q_f = 外向き流束）

未知数は nd + nd² 個（3 次元で 12）、拘束は面数（六面体で 6）なので不足決定になる。
残る自由度は**最小ノルム解**で閉じる（``np.linalg.pinv``）。この閉じ方には
2 つの良い性質がある。

- 直交六面体では、対向する 2 面の拘束が軸ごとに分離して
  ``u_x`` が x だけの 1 次関数……という **Pollock（1988）の再構成そのもの**になる。
  非対角成分は拘束に現れないので最小ノルムがゼロにする
- 四面体では拘束 4 本が「定数ベクトル + 等方な B」を決め、非圧縮（Σq_f = 0）なら
  B = 0 の**一定速度**になる。これは最低次 Raviart–Thomas 要素（RT0）と一致する

発散は ∇·u = tr(B_c) = Σ_f q_f / V_c なので、**離散連続式を満たす面流束を渡す限り
セル内で恒等的にゼロ**になる。セル中心速度を直接補間する追跡と違って、粒子が
渦心に落ち込んだり壁に貼り付いたりしない。

セルからセルへの受け渡し
------------------------
1 ステップは「面平面までの残り時間」で刻む。アフィン場なので直線近似の到達時刻
τ_f = −s_f / (u·n̂_f)（s_f は面平面までの符号付き距離）が良い予測になり、
RK4 で進めた後に面を跨いだら false position で面上に落として隣接セルへ渡す。
セル内で場が厳密にアフィンなので、刻み幅は精度ではなく**面の検出**だけで決まる。

周期面（``MeshData.face_offset``）は内部面として跨げる。跨ぐときに位置へ並進を
掛け、その総和 ``shift_total`` を持ち回るので ``x + shift_total`` が連続な
「巻き戻さない座標」になる。押出の ζ（軸方向座標）はこの座標の軸方向成分そのもの。

種まきと終了条件
----------------
- ``seed="patch"``: 流入する境界面 1 枚につき 1 粒子、重み = 流入流束 [m³/s]
- ``seed="axial"``: 流体セル 1 個につき 1 粒子、重み = max(u_c·â, 0)·V_c。
  軸 â 方向に周期的な領域では、進行度 ζ = â·(x + shift_total) が ``length`` に
  達したら脱出とする

``seed="axial"`` の理論平均滞留時間は

    ⟨t⟩ = length · V_total / Σ_c (u_c·â) V_c

（ζ=const 面の断面積を A_ζ、周期の ζ 長さを Δζ とすると V_total = A_ζ Δζ、
面流束 Q = Σ_c (u_c·â) V_c / Δζ、⟨t⟩ = length·A_ζ/Q）。
Δζ を知らなくても書けるのが要点で、構造格子版の
⟨t⟩ = z_axial·A_free/(sinφ·Q_axial) はこの式の 2.5D 特殊形になっている。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PostProcess
from xkep_cae_fluid.core.data import MeshData

__all__ = [
    "CellFaceTable",
    "ParticleTrackFVMInput",
    "ParticleTrackFVMProcess",
    "ParticleTrackFVMResult",
    "SEED_MODES",
    "cell_face_table",
    "reconstruct_cell_velocity",
]

SEED_MODES = ("axial", "patch", "explicit")

_TINY = 1e-300
_DT_MAX_FRACTION = 0.02
"""1 ステップの時間刻みの上限（理論平均滞留時間に対する比）.

淀み点（|u| → 0）では面までの到達時刻が発散し、1 ステップで理論滞留時間の
10 桁上を踏んでしまう。位置は動かないので害は無いように見えるが、経過時間が
壊れて ⟨t⟩ が飛ぶ（構造格子版で実測）。理論値は種まき時点で分かるので上限を置く。
"""


# ---------------------------------------------------------------------------
# セル → 面のテーブル
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellFaceTable:
    """セルを囲む面の CSR 表.

    Parameters
    ----------
    start : np.ndarray
        (n_cells+1,) CSR の区切り。セル c の項目は ``[start[c]:start[c+1]]``
    face : np.ndarray
        (n_entries,) 面インデックス
    sign : np.ndarray
        (n_entries,) セルが owner なら +1、neighbour なら −1。
        面法線・面流束にこれを掛けると**そのセルから見た外向き**になる
    shift : np.ndarray
        (n_entries, nd) 面中心をそのセルの座標系に置くための並進。
        通常の面はゼロ、周期面の neighbour 側は ``−face_offset``

    周期方向のセルが 1 層しか無いとき（押出の 2.5D 例題の z 方向）、周期対は
    owner == neighbour の内部面 1 本に併合される。そのセルは同じ面を符号 +1 と −1 で
    2 項目持ち、``shift`` が違うので**別の平面**として正しく扱える。
    """

    start: np.ndarray
    face: np.ndarray
    sign: np.ndarray
    shift: np.ndarray

    @property
    def count(self) -> np.ndarray:
        """(n_cells,) セルごとの面数."""
        return np.diff(self.start)


def _cell_length(volumes: np.ndarray, nd: int) -> np.ndarray:
    """セルの代表長さ（3 次元は体積の 1/3 乗、2 次元は面積の 1/2 乗）."""
    v = np.maximum(volumes, _TINY)
    return np.cbrt(v) if nd == 3 else np.sqrt(v)


def cell_face_table(mesh: MeshData) -> CellFaceTable:
    """``MeshData`` の owner / neighbour 配列からセル → 面の CSR 表を作る."""
    if mesh.face_owner is None or mesh.face_neighbour is None:
        raise ValueError("MeshData に face_owner / face_neighbour がありません")
    if mesh.face_normals is None or mesh.face_centers is None or mesh.cell_centers is None:
        raise ValueError("MeshData に面幾何（normals/centers）がありません")
    n_int = mesh.n_internal_faces
    n_faces = mesh.n_faces
    nd = mesh.face_normals.shape[1]

    cells = np.concatenate([mesh.face_owner, mesh.face_neighbour])
    faces = np.concatenate([np.arange(n_faces, dtype=np.int64), np.arange(n_int, dtype=np.int64)])
    signs = np.concatenate([np.ones(n_faces), -np.ones(n_int)])

    order = np.argsort(cells, kind="stable")
    cells = cells[order]
    faces = faces[order]
    signs = signs[order]

    shift = np.zeros((faces.shape[0], nd))
    if mesh.face_offset is not None:
        off = np.asarray(mesh.face_offset, dtype=np.float64)[:, :nd]
        nb_entry = signs < 0.0
        shift[nb_entry] = -off[faces[nb_entry]]

    counts = np.bincount(cells, minlength=mesh.n_cells)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if int(counts.min()) < nd + 1:
        bad = int(np.argmin(counts))
        raise ValueError(
            f"セル {bad} の面が {int(counts.min())} 枚しかありません（{nd + 1} 枚以上必要）"
        )
    return CellFaceTable(start=start, face=faces, sign=signs, shift=shift)


# ---------------------------------------------------------------------------
# セル内速度場の再構成
# ---------------------------------------------------------------------------


def reconstruct_cell_velocity(
    mesh: MeshData, face_flux: np.ndarray, table: CellFaceTable | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """面流束を厳密に再現するセル内アフィン速度場 u(x) = a_c + B_c (x − x_c).

    Parameters
    ----------
    mesh : MeshData
    face_flux : np.ndarray
        (n_faces,) **体積**流束 [m³/s]。owner → neighbour 向きが正
    table : CellFaceTable | None
        省略すると :func:`cell_face_table` で作る

    Returns
    -------
    (a, B) : (n_cells, nd) と (n_cells, nd, nd)

    最小ノルム解は無次元化した ``B̃ = B·L_c``（L_c はセルの代表長さ）に対して取る。
    a [m/s] と B [1/s] を素のまま並べると次元が揃わず、セルの大きさで結果が変わる。
    """
    tab = table if table is not None else cell_face_table(mesh)
    nd = mesh.face_normals.shape[1]
    n_cells = mesh.n_cells
    q_all = np.asarray(face_flux, dtype=np.float64)
    if q_all.shape != (mesh.n_faces,):
        raise ValueError(f"face_flux は (n_faces={mesh.n_faces},) が必要: {q_all.shape}")

    xc = mesh.cell_centers[:, :nd]
    length = _cell_length(mesh.cell_volumes, nd)

    counts = tab.count
    a = np.zeros((n_cells, nd))
    B = np.zeros((n_cells, nd, nd))
    cell_of_entry = np.repeat(np.arange(n_cells, dtype=np.int64), counts)

    s_vec = tab.sign[:, None] * mesh.face_normals[tab.face, :nd] * mesh.face_areas[tab.face, None]
    d_vec = mesh.face_centers[tab.face, :nd] + tab.shift - xc[cell_of_entry]
    q_out = tab.sign * q_all[tab.face]

    for m in np.unique(counts):
        sel = np.nonzero(counts == m)[0]
        idx = (tab.start[sel][:, None] + np.arange(m)[None, :]).ravel()
        s_m = s_vec[idx].reshape(sel.size, m, nd)
        d_m = d_vec[idx].reshape(sel.size, m, nd)
        q_m = q_out[idx].reshape(sel.size, m)
        # 行: [S_i ,  d_j S_i / L]  （未知数 [a_i, B̃_ij]）
        rows = np.empty((sel.size, m, nd + nd * nd))
        rows[:, :, :nd] = s_m
        rows[:, :, nd:] = (
            s_m[:, :, :, None] * d_m[:, :, None, :] / length[sel][:, None, None, None]
        ).reshape(sel.size, m, nd * nd)
        z = np.einsum("kij,kj->ki", np.linalg.pinv(rows, rcond=1e-12), q_m)
        a[sel] = z[:, :nd]
        B[sel] = z[:, nd:].reshape(sel.size, nd, nd) / length[sel][:, None, None]
    return a, B


# ---------------------------------------------------------------------------
# 入出力
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleTrackFVMInput:
    """非構造メッシュ粒子追跡の入力.

    Parameters
    ----------
    mesh : MeshData
        面情報（owner / neighbour / areas / normals / centers）を持つメッシュ
    face_flux : np.ndarray
        (n_faces,) 面流束。``density`` で割って体積流束にする。
        :attr:`~xkep_cae_fluid.incompressible.data.NavierStokesFVMResult.mass_flux` を
        そのまま渡し、``density`` に ρ を入れるのが標準的な使い方
    density : float
        面流束を体積流束にするための除数 [kg/m³]。既定 1.0（既に体積流束のとき）
    seed : str
        ``"axial"`` / ``"patch"`` / ``"explicit"``
    axis : tuple[float, ...] | None
        進行方向の単位ベクトル â。``seed="axial"`` では必須
    length : float
        脱出とみなす進行度 ζ = â·(x + shift_total) [m]。``axis`` を与えたときに使う
    inlet_patch : str | None
        ``seed="patch"`` の流入パッチ名
    positions, weights, cells : np.ndarray | None
        ``seed="explicit"`` の初期位置 (n, nd)・重み (n,)・初期セル (n,)。
        点からセルを探す機能は持たないので、初期セルは呼び出し側が与える
    scalars : Mapping[str, np.ndarray]
        経路に沿って ∫s dt を取るセル中心場（γ̇ や混合指数など）
    stride : int
        ``seed="axial"`` のセル間引き。重みは stride 倍して総流束を保つ
    cfl : float
        1 ステップで面までの予測到達時刻の何倍進むか。既定 1.0（面を跨いだら戻す）
    max_steps : int
        1 粒子あたりの最大ステップ数
    t_ref : float | None
        時間刻み上限の基準時間 [s]。省略すると理論平均滞留時間を使う
    dt_max_fraction : float
        時間刻み上限 / ``t_ref``
    extrapolation_min_progress : float
        ステップ上限に達した粒子を外挿してよい最小進行率（``length`` に対する比）。
        これ未満は未解決として報告する（淀みに捕まった粒子の外挿は ⟨t⟩ を壊す）
    wall_flux_tol : float
        境界面の流束がこの値（総流束に対する比）以下なら壁とみなし、
        跨いだ粒子を領域内へ押し戻す
    """

    mesh: MeshData
    face_flux: np.ndarray
    density: float = 1.0
    seed: str = "axial"
    axis: tuple[float, ...] | None = None
    length: float = 0.0
    inlet_patch: str | None = None
    positions: np.ndarray | None = None
    weights: np.ndarray | None = None
    cells: np.ndarray | None = None
    scalars: Mapping[str, np.ndarray] = field(default_factory=dict)
    stride: int = 1
    cfl: float = 1.0
    max_steps: int = 20_000
    t_ref: float | None = None
    dt_max_fraction: float = _DT_MAX_FRACTION
    extrapolation_min_progress: float = 0.1
    wall_flux_tol: float = 1e-10


@dataclass(frozen=True)
class ParticleTrackFVMResult:
    """粒子追跡の結果（配列は全て (n_particles,) か (n_particles, nd)）.

    Parameters
    ----------
    weight : np.ndarray
        流束重み。``seed="patch"`` は [m³/s]、``seed="axial"`` は max(u·â,0)·V_c [m⁴/s]
    t_res : np.ndarray
        滞留時間 [s]
    progress : np.ndarray
        到達した進行度 ζ [m]（``axis`` を与えたときのみ意味を持つ）
    x0, x : np.ndarray
        初期位置と最終位置 [m]
    cell0, cell : np.ndarray
        初期セルと最終セル
    shift_total : np.ndarray
        周期面を跨いで積み上げた並進の総和 [m]。``x + shift_total`` が巻き戻さない座標
    escaped : np.ndarray
        脱出したか（進行度到達・流出境界通過・外挿のいずれか）
    extrapolated : np.ndarray
        ステップ上限に達したので進行率から外挿して閉じたか
    exit_patch : np.ndarray
        流出した境界パッチの番号（``patch_names`` の添字）。未流出は −1
    n_steps : np.ndarray
        踏んだステップ数
    integrals : dict[str, np.ndarray]
        経路に沿った ∫s dt
    patch_names : tuple[str, ...]
        ``exit_patch`` が指すパッチ名
    volume : float
        メッシュの総体積 [m³]
    axial_flux : float
        重みの総和
    t_mean_theory : float
        理論平均滞留時間 [s]。``seed="axial"`` は length·V/Σweight、
        ``seed="patch"`` は V/Q_in
    """

    weight: np.ndarray
    t_res: np.ndarray
    progress: np.ndarray
    x0: np.ndarray
    x: np.ndarray
    cell0: np.ndarray
    cell: np.ndarray
    shift_total: np.ndarray
    escaped: np.ndarray
    extrapolated: np.ndarray
    exit_patch: np.ndarray
    n_steps: np.ndarray
    integrals: dict[str, np.ndarray] = field(default_factory=dict)
    patch_names: tuple[str, ...] = ()
    volume: float = 0.0
    axial_flux: float = 0.0
    t_mean_theory: float = 0.0

    @property
    def n_particles(self) -> int:
        return int(self.t_res.shape[0])


# ---------------------------------------------------------------------------
# 追跡
# ---------------------------------------------------------------------------


def _rk4(
    a: np.ndarray, b: np.ndarray, xc: np.ndarray, cells: np.ndarray, x: np.ndarray, dt: np.ndarray
) -> np.ndarray:
    """セル内アフィン場での RK4 を 1 ステップ.

    セル内で場が厳密にアフィンなので、RK4 は解 ``x(t) = e^{Bt}·…`` の 4 次 Taylor
    打ち切りそのものになる（刻みは面の検出のためだけに要る）。
    """
    a_c = a[cells]
    b_c = b[cells]
    x_c = xc[cells]
    h = dt[:, None]

    def vel(y: np.ndarray) -> np.ndarray:
        return a_c + np.einsum("kij,kj->ki", b_c, y - x_c)

    k1 = vel(x)
    k2 = vel(x + 0.5 * h * k1)
    k3 = vel(x + 0.5 * h * k2)
    k4 = vel(x + h * k3)
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _face_patch_map(mesh: MeshData) -> tuple[np.ndarray, tuple[str, ...]]:
    """面 → 境界パッチ番号（内部面は −1）と、パッチ名のタプル."""
    ids = np.full(mesh.n_faces, -1, dtype=np.int64)
    names = tuple(sorted(mesh.boundary_patches or ()))
    for k, name in enumerate(names):
        ids[np.asarray(mesh.boundary_patches[name], dtype=np.int64)] = k
    return ids, names


class ParticleTrackFVMProcess(PostProcess["ParticleTrackFVMInput", "ParticleTrackFVMResult"]):
    """面流束から再構成したセル内アフィン場を辿る Pollock 型の粒子追跡.

    全粒子を同時に（それぞれ固有の刻みで）進める。粒子ごとに時刻がずれるが、
    定常流なので各粒子の経過時間だけが意味を持ち、同期は要らない。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ParticleTrackFVM",
        module="post",
        version="0.1.0",
        document_path="../../docs/design/particle-tracking-fvm.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: ParticleTrackFVMInput) -> ParticleTrackFVMResult:
        """種をまき、面を跨ぎながら脱出条件まで追跡する."""
        inp = input_data
        mesh = inp.mesh
        if inp.seed not in SEED_MODES:
            raise ValueError(f"seed は {SEED_MODES} のいずれか: {inp.seed!r}")
        if inp.stride < 1:
            raise ValueError(f"stride は 1 以上が必要: {inp.stride}")
        if inp.density == 0.0:
            raise ValueError("density が 0 です")
        if inp.max_steps < 1:
            raise ValueError(f"max_steps は 1 以上が必要: {inp.max_steps}")

        table = cell_face_table(mesh)
        nd = mesh.face_normals.shape[1]
        n_int = mesh.n_internal_faces
        q = np.asarray(inp.face_flux, dtype=np.float64) / inp.density
        a, b_mat = reconstruct_cell_velocity(mesh, q, table)

        xc = mesh.cell_centers[:, :nd]
        normals = mesh.face_normals[:, :nd]
        fcent = mesh.face_centers[:, :nd]
        # 全面長に伸ばした neighbour / 周期並進（境界面は使わないのでゼロ詰め）
        nb_pad = np.zeros(mesh.n_faces, dtype=np.int64)
        nb_pad[:n_int] = mesh.face_neighbour
        off_pad = np.zeros((mesh.n_faces, nd))
        if mesh.face_offset is not None:
            off_pad[:n_int] = np.asarray(mesh.face_offset, dtype=np.float64)[:, :nd]
        counts = table.count
        cell_len = _cell_length(mesh.cell_volumes, nd)
        span = float(np.linalg.norm(mesh.node_coords.max(axis=0) - mesh.node_coords.min(axis=0)))
        tol = 1e-12 * span

        axis = None
        if inp.axis is not None:
            axis = np.asarray(inp.axis, dtype=np.float64)[:nd]
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError("axis がゼロベクトルです")
            axis = axis / norm

        x, cell, weight = self._seed(inp, mesh, a, xc, normals, fcent, cell_len, axis)
        n = x.shape[0]
        if n == 0:
            raise ValueError("種をまける位置が 1 つもありません（流入面 / 軸方向流束を確認）")

        volume = float(np.sum(mesh.cell_volumes))
        total_w = float(weight.sum())
        if total_w <= 0.0:
            raise ValueError("重みの総和が 0 以下です")
        if inp.seed == "axial":
            if axis is None:
                raise ValueError('seed="axial" には axis が必要です')
            if inp.length <= 0.0:
                raise ValueError(f"length は正が必要: {inp.length}")
            t_theory = inp.length * volume / total_w
        else:
            t_theory = volume / total_w
        dt_max = inp.dt_max_fraction * (inp.t_ref if inp.t_ref is not None else t_theory)
        if dt_max <= 0.0:
            raise ValueError(f"時間刻み上限が正になりません（t_ref={inp.t_ref}）")

        scal = {k: np.asarray(v, dtype=np.float64) for k, v in inp.scalars.items()}
        for k, v in scal.items():
            if v.shape != (mesh.n_cells,):
                raise ValueError(f"scalars[{k!r}] は (n_cells,) が必要: {v.shape}")

        patch_of_face, patch_names = _face_patch_map(mesh)
        wall = np.abs(q) <= inp.wall_flux_tol * max(float(np.max(np.abs(q))), _TINY)

        x0 = x.copy()
        cell0 = cell.copy()
        shift_total = np.zeros((n, nd))
        t = np.zeros(n)
        progress = np.zeros(n)
        integrals = {k: np.zeros(n) for k in scal}
        steps = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        escaped = np.zeros(n, dtype=bool)
        exit_patch = np.full(n, -1, dtype=np.int64)
        track_progress = axis is not None and inp.length > 0.0

        for _ in range(inp.max_steps):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            c = cell[idx]
            xa = x[idx]
            cnt = counts[c]
            seg = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            pp = np.repeat(np.arange(idx.size), cnt)
            flat = np.repeat(table.start[c], cnt) + (
                np.arange(int(cnt.sum())) - np.repeat(seg, cnt)
            )
            f = table.face[flat]
            sg = table.sign[flat]
            nout = normals[f] * sg[:, None]
            pc = fcent[f] + table.shift[flat]

            s0 = np.sum((xa[pp] - pc) * nout, axis=1)
            u = a[c] + np.einsum("kij,kj->ki", b_mat[c], xa - xc[c])
            un = np.sum(u[pp] * nout, axis=1)
            tau = np.where(un > _TINY, -s0 / np.where(un > _TINY, un, 1.0), np.inf)
            tau = np.maximum(tau, 0.0)
            dt_face = inp.cfl * np.minimum.reduceat(tau, seg)
            dt = np.where(np.isfinite(dt_face), np.minimum(dt_face, dt_max), dt_max)

            if track_progress:
                # ζ の行き過ぎを抑える（脱出時刻の線形内挿を効かせるため）
                rate = u @ axis
                remain = inp.length - progress[idx]
                dt = np.where(
                    rate > 0.0, np.minimum(dt, 2.0 * remain / np.where(rate > 0.0, rate, 1.0)), dt
                )
            dt = np.maximum(dt, 0.0)

            x1 = _rk4(a, b_mat, xc, c, xa, dt)
            s1 = np.sum((x1[pp] - pc) * nout, axis=1)
            smax = np.maximum.reduceat(s1, seg)
            # 面拘束で刻んだステップは面に**ちょうど**乗る（s1 = 0）。不等号だけで
            # 判定すると受け渡しが起きず、以降 dt = 0 のまま動かなくなる。
            touch = np.isfinite(dt_face) & (dt >= dt_face) & (smax > -1e-9 * cell_len[c])
            crossed = (smax > tol) | touch

            frac = np.ones(idx.size)
            trans = np.zeros((idx.size, nd))
            new_cell = c.copy()
            leaves = np.zeros(idx.size, dtype=bool)
            patch_hit = np.full(idx.size, -1, dtype=np.int64)

            if crossed.any():
                hit = np.nonzero(s1 >= smax[pp])[0]
                ent = np.zeros(idx.size, dtype=np.int64)
                ent[pp[hit]] = hit
                cid = np.nonzero(crossed)[0]
                e = ent[cid]
                x1[cid], frac[cid] = self._land_on_face(
                    a, b_mat, xc, c[cid], xa[cid], dt[cid], s0[e], s1[e], pc[e], nout[e]
                )
                f_e = f[e]
                sg_e = sg[e]
                internal = f_e < n_int
                # 内部面: owner ⇄ neighbour。周期面は位置に並進を掛ける
                nxt = np.where(sg_e > 0.0, nb_pad[f_e], mesh.face_owner[f_e])
                new_cell[cid] = np.where(internal, nxt, c[cid])
                off_e = np.where(internal[:, None], -sg_e[:, None] * off_pad[f_e], 0.0)
                x1[cid] += off_e
                trans[cid] = off_e
                # 境界面: 流束が有るなら流出、無いなら壁なので領域内へ戻す
                out = (~internal) & (~wall[f_e])
                leaves[cid] = out
                patch_hit[cid] = np.where(out, patch_of_face[f_e], -1)
                back = (~internal) & wall[f_e]
                if back.any():
                    x1[cid[back]] -= (1e-9 * cell_len[c[cid[back]]])[:, None] * nout[e][back]

            dt_eff = dt * frac
            prog_new = progress[idx].copy()
            if track_progress:
                # 巻き戻さない座標 x + shift_total（並進 T を掛けたら shift_total は −T）
                prog_new = np.sum((x1 + shift_total[idx] - trans) * axis, axis=1) - np.sum(
                    x0[idx] * axis, axis=1
                )
                reached = prog_new >= inp.length
                if reached.any():
                    d_prog = prog_new - progress[idx]
                    safe = np.where(d_prog > 0.0, d_prog, 1.0)
                    fz = np.where(d_prog > 0.0, (inp.length - progress[idx]) / safe, 1.0)
                    fz = np.clip(fz, 0.0, 1.0)
                    dt_eff = np.where(reached, dt_eff * fz, dt_eff)
                    prog_new = np.where(reached, inp.length, prog_new)

            t[idx] += dt_eff
            for k, v in scal.items():
                integrals[k][idx] += v[c] * dt_eff
            x[idx] = x1
            shift_total[idx] -= trans
            cell[idx] = new_cell
            progress[idx] = prog_new
            steps[idx] += 1

            done = leaves.copy()
            if track_progress:
                done |= prog_new >= inp.length
            if done.any():
                gone = idx[done]
                escaped[gone] = True
                alive[gone] = False
                exit_patch[gone] = patch_hit[done]

        extrapolated = np.zeros(n, dtype=bool)
        if track_progress:
            # ステップ上限に達した粒子を進行率から外挿して閉じる。
            # 定常な周回軌道では ζ が t に比例するので t·length/ζ が良い近似になるが、
            # ζ ≈ 0 の粒子（淀み・二次渦に捕まったもの）は前提を満たしていない。
            # 係数が発散して ⟨t⟩ を壊すので、進行率が信用できる粒子だけ外挿し、
            # 残りは未解決として正直に報告する。
            stuck = alive & (progress >= inp.extrapolation_min_progress * inp.length)
            if stuck.any():
                factor = inp.length / progress[stuck]
                t[stuck] *= factor
                for k in integrals:
                    integrals[k][stuck] *= factor
                extrapolated[stuck] = True
                escaped[stuck] = True

        return ParticleTrackFVMResult(
            weight=weight,
            t_res=t,
            progress=progress,
            x0=x0,
            x=x,
            cell0=cell0,
            cell=cell,
            shift_total=shift_total,
            escaped=escaped,
            extrapolated=extrapolated,
            exit_patch=exit_patch,
            n_steps=steps,
            integrals=integrals,
            patch_names=patch_names,
            volume=volume,
            axial_flux=total_w,
            t_mean_theory=float(t_theory),
        )

    @staticmethod
    def _land_on_face(
        a: np.ndarray,
        b_mat: np.ndarray,
        xc: np.ndarray,
        cells: np.ndarray,
        x: np.ndarray,
        dt: np.ndarray,
        s_lo: np.ndarray,
        s_hi: np.ndarray,
        pc: np.ndarray,
        nout: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """面を跨いだステップを false position で面上まで戻す（位置と刻みの割合）."""
        lo = np.zeros(cells.shape[0])
        hi = np.ones(cells.shape[0])
        s_lo = np.minimum(s_lo, 0.0)
        frac = np.clip(-s_lo / np.maximum(s_hi - s_lo, _TINY), 0.0, 1.0)
        xt = x
        for _ in range(4):
            xt = _rk4(a, b_mat, xc, cells, x, dt * frac)
            st = np.sum((xt - pc) * nout, axis=1)
            inside = st <= 0.0
            lo = np.where(inside, frac, lo)
            s_lo = np.where(inside, st, s_lo)
            hi = np.where(inside, hi, frac)
            s_hi = np.where(inside, s_hi, st)
            width = np.maximum(s_hi - s_lo, _TINY)
            frac = np.clip(lo + (hi - lo) * (-s_lo) / width, 0.0, 1.0)
        xt = _rk4(a, b_mat, xc, cells, x, dt * frac)
        # 面平面へ厳密に落とす（法線は単位ベクトル）
        xt = xt - np.sum((xt - pc) * nout, axis=1)[:, None] * nout
        return xt, frac

    @staticmethod
    def _seed(
        inp: ParticleTrackFVMInput,
        mesh: MeshData,
        a: np.ndarray,
        xc: np.ndarray,
        normals: np.ndarray,
        fcent: np.ndarray,
        cell_len: np.ndarray,
        axis: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(初期位置, 初期セル, 重み) を返す."""
        if inp.seed == "explicit":
            if inp.positions is None or inp.weights is None or inp.cells is None:
                raise ValueError('seed="explicit" には positions / weights / cells が必要です')
            pos = np.asarray(inp.positions, dtype=np.float64)
            w = np.asarray(inp.weights, dtype=np.float64)
            cells = np.asarray(inp.cells, dtype=np.int64)
            if pos.shape[0] != w.shape[0] or pos.shape[0] != cells.shape[0]:
                raise ValueError("positions / weights / cells の長さが揃っていません")
            return pos.copy(), cells.copy(), w.copy()

        if inp.seed == "patch":
            if not inp.inlet_patch:
                raise ValueError('seed="patch" には inlet_patch が必要です')
            faces = mesh.patch_faces(inp.inlet_patch)
            q = np.asarray(inp.face_flux, dtype=np.float64)[faces] / inp.density
            take = q < 0.0  # 外向き流束が負 = 流入
            if not take.any():
                raise ValueError(f"パッチ {inp.inlet_patch!r} に流入面がありません")
            fs = faces[take]
            own = mesh.face_owner[fs]
            # 面中心から owner 側へわずかに押し込む（法線は領域外向き）
            pos = fcent[fs] - (1e-6 * cell_len[own])[:, None] * normals[fs]
            return pos, own.astype(np.int64), -q[take]

        if axis is None:
            raise ValueError('seed="axial" には axis が必要です')
        # セル中心の速度は再構成場では a そのもの（追跡に使う場と食い違わない）
        ua = a @ axis
        keep = np.zeros(mesh.n_cells, dtype=bool)
        keep[:: inp.stride] = True
        mask = keep & (ua > 0.0)
        cells = np.nonzero(mask)[0]
        w = ua[cells] * mesh.cell_volumes[cells] * float(inp.stride)
        return xc[cells].copy(), cells.astype(np.int64), w
