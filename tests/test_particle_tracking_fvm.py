"""非構造メッシュの粒子追跡（Pollock 型）と滞留時間分布のテスト.

再構成が面流束を厳密に再現すること、既知の流れで軌跡と滞留時間が解析解に一致すること、
そして押出の展開チャネルで**構造格子専用のトラッカー**（流れ関数の双一次補間 + RK4、
ゲート G4a/G4b/G5 を通過済み）と同じ RTD が出ることを確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.incompressible import (
    FlowPatchBC,
    NavierStokesFVMInput,
    NavierStokesFVMProcess,
)
from xkep_cae_fluid.inp.builder import build_case
from xkep_cae_fluid.inp.mesh import InpMeshInput, InpMeshProcess
from xkep_cae_fluid.inp.parser import parse_inp_text
from xkep_cae_fluid.post.rtd import ResidenceTimeInput, ResidenceTimeProcess
from xkep_cae_fluid.post.tracking import (
    ParticleTrackFVMInput,
    ParticleTrackFVMProcess,
    cell_face_table,
    reconstruct_cell_velocity,
)


def _box(nx: int, ny: int, nz: int = 1, lx: float = 1.0, ly: float = 1.0, lz: float = 0.1):
    return (
        StructuredMeshProcess()
        .execute(StructuredMeshInput(Lx=lx, Ly=ly, Lz=lz, nx=nx, ny=ny, nz=nz))
        .mesh
    )


def _periodic_box(nx: int, ny: int, lx: float, ly: float, lz: float = 0.05):
    """x と z を周期にした箱格子（.inp 経由）."""
    text = (
        f"*GRID, NX={nx}, NY={ny}, NZ=1, LX={lx}, LY={ly}, LZ={lz}\n"
        "*BOUNDARY, TYPE=PERIODIC\n XM, XP\n"
        "*BOUNDARY, TYPE=PERIODIC\n ZM, ZP\n"
    )
    return InpMeshProcess().execute(InpMeshInput(case=build_case(parse_inp_text(text)))).mesh


def _exact_flux(mesh, field) -> np.ndarray:
    """面中心の u·n·A（面が平面で場が 1 次なら面積分と厳密に一致する）."""
    return np.sum(field(mesh.face_centers) * mesh.face_normals, axis=1) * mesh.face_areas


def _uniform(vec):
    return lambda x: np.tile(np.asarray(vec, dtype=np.float64), (x.shape[0], 1))


def _shear(s: float):
    return lambda x: np.stack([s * x[:, 1], np.zeros(len(x)), np.zeros(len(x))], axis=1)


def _flux_residual(mesh, a: np.ndarray, b: np.ndarray, q: np.ndarray) -> float:
    """再構成場が各面の流束をどれだけ再現しているか（相対）."""
    tab = cell_face_table(mesh)
    cell = np.repeat(np.arange(mesh.n_cells), tab.count)
    d = mesh.face_centers[tab.face] + tab.shift - mesh.cell_centers[cell]
    s_vec = tab.sign[:, None] * mesh.face_normals[tab.face] * mesh.face_areas[tab.face, None]
    u = a[cell] + np.einsum("kij,kj->ki", b[cell], d)
    res = np.sum(u * s_vec, axis=1) - tab.sign * q[tab.face]
    return float(np.max(np.abs(res)) / max(float(np.max(np.abs(q))), 1e-300))


class TestCellVelocityReconstruction:
    """面流束からのセル内アフィン場の再構成（最小ノルム拘束最小二乗）."""

    def test_reproduces_every_face_flux(self):
        mesh = _box(4, 3, 2)
        q = _exact_flux(mesh, _shear(3.0))
        a, b = reconstruct_cell_velocity(mesh, q)
        assert _flux_residual(mesh, a, b, q) < 1e-12

    def test_uniform_flow_gives_constant_field(self):
        mesh = _box(4, 3, 2)
        a, b = reconstruct_cell_velocity(mesh, _exact_flux(mesh, _uniform((2.0, 0.0, 0.0))))
        assert np.max(np.abs(a - np.array([2.0, 0.0, 0.0]))) < 1e-12
        assert np.max(np.abs(b)) < 1e-12

    def test_shear_is_piecewise_constant_across_the_shear_direction(self):
        """Pollock 型は「その方向の面流束が無い変化」を落とす（せん断は y に階段状）.

        u = S y の x 方向流束は ±x 面にしか現れず、y 依存はセル間の差でしか
        表現されない。セル内では y に一定（B = 0）になるのが正しい振る舞い。
        """
        mesh = _box(4, 8, 1)
        s = 3.0
        a, b = reconstruct_cell_velocity(mesh, _exact_flux(mesh, _shear(s)))
        assert np.max(np.abs(a[:, 0] - s * mesh.cell_centers[:, 1])) < 1e-12
        assert np.max(np.abs(b)) < 1e-12

    def test_linear_divergence_free_field_is_recovered_inside_cells(self):
        """u = (S x, −S y, 0) はセル内でも 1 次なので B まで厳密に復元できる."""
        mesh = _box(4, 4, 2)
        s = 2.0
        field = lambda x: np.stack(  # noqa: E731
            [s * x[:, 0], -s * x[:, 1], np.zeros(len(x))], axis=1
        )
        a, b = reconstruct_cell_velocity(mesh, _exact_flux(mesh, field))
        assert np.max(np.abs(a - field(mesh.cell_centers))) < 1e-12
        exact = np.zeros((mesh.n_cells, 3, 3))
        exact[:, 0, 0] = s
        exact[:, 1, 1] = -s
        assert np.max(np.abs(b - exact)) < 1e-10

    def test_divergence_equals_flux_imbalance(self):
        """∇·u = tr(B) = Σq_f / V。非圧縮な面流束ならセル内で恒等的にゼロ."""
        mesh = _box(3, 3, 3)
        q = _exact_flux(mesh, _shear(1.5))
        _, b = reconstruct_cell_velocity(mesh, q)
        assert np.max(np.abs(np.trace(b, axis1=1, axis2=2))) < 1e-10

    def test_rejects_wrong_flux_shape(self):
        mesh = _box(2, 2, 1)
        with pytest.raises(ValueError, match="face_flux"):
            reconstruct_cell_velocity(mesh, np.zeros(3))

    def test_periodic_faces_appear_twice_with_opposite_signs(self):
        """1 層しかない周期方向は owner == neighbour の自己面になる（別平面として持つ）."""
        mesh = _periodic_box(4, 4, 1.0, 1.0)
        tab = cell_face_table(mesh)
        assert np.all(tab.count == 6)
        c0 = slice(tab.start[0], tab.start[1])
        faces = tab.face[c0]
        # z 方向の周期対は同じ面インデックスを符号 ±1 で 2 回持つ
        dup = [f for f in set(faces.tolist()) if (faces == f).sum() == 2]
        assert dup, "自己面（owner == neighbour）が見つからない"
        for f in dup:
            sel = faces == f
            assert set(tab.sign[c0][sel]) == {1.0, -1.0}
            assert np.max(np.abs(np.diff(tab.shift[c0][sel], axis=0))) > 0.0


@binds_to(ParticleTrackFVMProcess)
class TestParticleTrackFVMAPI:
    def test_meta(self):
        meta = ParticleTrackFVMProcess.meta
        assert meta.name == "ParticleTrackFVM" and meta.module == "post"

    @pytest.mark.parametrize(
        "kw,msg",
        [
            (dict(seed="bogus"), "seed は"),
            (dict(seed="patch", inlet_patch=None), "inlet_patch"),
            (dict(seed="patch", inlet_patch="XM", stride=0), "stride"),
            (dict(seed="axial", axis=None), "axis"),
            (dict(seed="axial", axis=(1.0, 0.0, 0.0), length=0.0), "length"),
            (dict(seed="axial", axis=(0.0, 0.0, 0.0), length=1.0), "ゼロベクトル"),
            (dict(seed="explicit"), "positions"),
            (dict(seed="patch", inlet_patch="XM", max_steps=0), "max_steps"),
            (dict(seed="patch", inlet_patch="XM", density=0.0), "density"),
        ],
    )
    def test_validation(self, kw: dict, msg: str):
        mesh = _box(3, 3, 1)
        q = _exact_flux(mesh, _uniform((1.0, 0.0, 0.0)))
        with pytest.raises(ValueError, match=msg):
            ParticleTrackFVMProcess().execute(ParticleTrackFVMInput(mesh=mesh, face_flux=q, **kw))

    def test_unknown_patch_raises(self):
        mesh = _box(3, 3, 1)
        q = _exact_flux(mesh, _uniform((1.0, 0.0, 0.0)))
        with pytest.raises(KeyError):
            ParticleTrackFVMProcess().execute(
                ParticleTrackFVMInput(mesh=mesh, face_flux=q, seed="patch", inlet_patch="NOPE")
            )

    def test_outflow_patch_has_no_inflow(self):
        mesh = _box(3, 3, 1)
        q = _exact_flux(mesh, _uniform((1.0, 0.0, 0.0)))
        with pytest.raises(ValueError, match="流入面がありません"):
            ParticleTrackFVMProcess().execute(
                ParticleTrackFVMInput(mesh=mesh, face_flux=q, seed="patch", inlet_patch="XP")
            )

    def test_unit_scalar_integrates_to_the_residence_time(self):
        """∫1 dt = t なので、経路積分の配線が正しければ滞留時間そのものになる."""
        mesh = _box(5, 3, 1)
        q = _exact_flux(mesh, _shear(2.0))
        out = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=q,
                seed="patch",
                inlet_patch="XM",
                scalars={"one": np.ones(mesh.n_cells)},
            )
        )
        assert out.escaped.all()
        assert np.max(np.abs(out.integrals["one"] - out.t_res)) < 1e-15 * out.t_res.max()

    def test_explicit_seed_requires_matching_lengths(self):
        mesh = _box(3, 3, 1)
        q = _exact_flux(mesh, _uniform((1.0, 0.0, 0.0)))
        with pytest.raises(ValueError, match="長さが揃って"):
            ParticleTrackFVMProcess().execute(
                ParticleTrackFVMInput(
                    mesh=mesh,
                    face_flux=q,
                    seed="explicit",
                    positions=mesh.cell_centers[:2],
                    weights=np.ones(2),
                    cells=np.array([0]),
                )
            )

    def test_stride_keeps_the_total_weight(self):
        mesh = _periodic_box(6, 6, 1.0, 1.0)
        q = _exact_flux(mesh, _uniform((1.0, 0.0, 0.0)))
        kw = dict(mesh=mesh, face_flux=q, seed="axial", axis=(1.0, 0.0, 0.0), length=1.0)
        full = ParticleTrackFVMProcess().execute(ParticleTrackFVMInput(**kw))
        thin = ParticleTrackFVMProcess().execute(ParticleTrackFVMInput(stride=2, **kw))
        assert thin.n_particles < full.n_particles
        assert thin.axial_flux == pytest.approx(full.axial_flux, rel=1e-12)


class TestParticleTrackFVMPhysics:
    """既知の流れで軌跡・滞留時間・質量保存が解析解に一致する."""

    def test_uniform_flow_is_plug_flow(self):
        """一様流では全粒子が同じ滞留時間 L/U で出口パッチから出る."""
        u, lx = 2.0, 1.0
        mesh = _box(4, 3, 2, lx=lx)
        out = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=_exact_flux(mesh, _uniform((u, 0.0, 0.0))),
                seed="patch",
                inlet_patch="XM",
            )
        )
        assert out.escaped.all()
        assert np.ptp(out.t_res) < 1e-12
        assert out.t_res.mean() == pytest.approx(lx / u, rel=1e-5)
        assert set(out.exit_patch.tolist()) == {out.patch_names.index("XP")}
        assert out.t_mean_theory == pytest.approx(lx / u, rel=1e-12)

    def test_simple_shear_trajectories_stay_on_their_streamline(self):
        """単純せん断 u = (S y, 0, 0): 軌跡は直線、滞留時間は L/(S y)（ゲート G4a 相当）."""
        s, lx = 3.0, 1.0
        mesh = _box(4, 6, 1, lx=lx)
        out = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh, face_flux=_exact_flux(mesh, _shear(s)), seed="patch", inlet_patch="XM"
            )
        )
        assert out.escaped.all()
        assert np.max(np.abs(out.x[:, 1] - out.x0[:, 1])) < 1e-12
        assert np.max(np.abs(out.x[:, 2] - out.x0[:, 2])) < 1e-12
        exact = lx / (s * out.x0[:, 1])
        assert np.max(np.abs(out.t_res / exact - 1.0)) < 1e-5

    def test_walls_are_not_crossed(self):
        """流束ゼロの境界面（壁）は跨げない。キャビティの循環流で全粒子が中に残る."""
        mesh = _box(12, 12, 1)
        bcs = {
            "XM": FlowPatchBC.wall(),
            "XP": FlowPatchBC.wall(),
            "YM": FlowPatchBC.wall(),
            "YP": FlowPatchBC.wall(velocity=(1.0, 0.0, 0.0)),
        }
        res = NavierStokesFVMProcess().execute(
            NavierStokesFVMInput(
                mesh=mesh,
                bcs=bcs,
                rho=1.0,
                mu=0.05,
                coupling="coupled",
                convection="none",
                tol=1e-11,
                max_outer_iter=20,
            )
        )
        assert res.converged
        out = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=res.mass_flux,
                seed="explicit",
                positions=mesh.cell_centers.copy(),
                weights=np.ones(mesh.n_cells),
                cells=np.arange(mesh.n_cells),
                t_ref=10.0,
                max_steps=500,
            )
        )
        assert not out.escaped.any()  # 出口が無いので誰も出られない
        lo = mesh.node_coords.min(axis=0)
        hi = mesh.node_coords.max(axis=0)
        assert np.all(out.x >= lo - 1e-9) and np.all(out.x <= hi + 1e-9)

    @pytest.mark.parametrize("ny,tol,tol_mean", [(12, 8e-2, 1.5e-2), (24, 2.2e-2, 4e-3)])
    def test_periodic_poiseuille_residence_time(self, ny: int, tol: float, tol_mean: float):
        """周期流路（体積力駆動）の滞留時間 t(y) = L/u(y) と ⟨t⟩ = V/Q.

        ⟨t⟩ が理論値 length·V/Σw に**厳密に**一致するのは、種まき重み・面の受け渡し・
        周期の巻き戻し・脱出時刻の内挿が全て整合しているときだけ（ゲート G4b 相当）。
        """
        h, lx, lz, f, mu, rho = 0.1, 0.4, 0.05, 2.0, 0.01, 1.0
        mesh = _periodic_box(4, ny, lx, h, lz)
        res = NavierStokesFVMProcess().execute(
            NavierStokesFVMInput(
                mesh=mesh,
                bcs={"YM": FlowPatchBC.wall(), "YP": FlowPatchBC.wall()},
                rho=rho,
                mu=mu,
                body_force=(f, 0.0, 0.0),
                coupling="coupled",
                convection="none",
                tol=1e-12,
                max_outer_iter=20,
            )
        )
        assert res.converged
        length = 1.0
        out = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=res.mass_flux,
                density=rho,
                seed="axial",
                axis=(1.0, 0.0, 0.0),
                length=length,
                max_steps=20000,
            )
        )
        assert out.escaped.all() and not out.extrapolated.any()
        w, t = out.weight, out.t_res
        t_mean = float((w * t).sum() / w.sum())
        # 厳密関係（離散化に依らず成り立つ）
        assert t_mean == pytest.approx(out.t_mean_theory, rel=1e-12)
        # 解析的な平均滞留時間 V/Q（こちらは離散化誤差が乗る。ny 12 → 24 で 1.4% → 0.35%）
        flow_rate = f * h**3 / (12.0 * mu) * lz
        assert t_mean == pytest.approx(length * h * lz / flow_rate, rel=tol_mean)
        # 粒子ごとの厳密解 t = length/u(y)
        y = mesh.cell_centers[out.cell0, 1]
        assert np.max(np.abs(t / (length / (f / (2 * mu) * y * (h - y))) - 1.0)) < tol
        # 周期方向に何度も巻き戻している
        assert np.abs(out.shift_total[:, 0]).max() > lx


@binds_to(ResidenceTimeProcess)
class TestResidenceTimeAPI:
    def test_meta(self):
        meta = ResidenceTimeProcess.meta
        assert meta.name == "ResidenceTime" and meta.module == "post"

    def _track(self, **kw):
        mesh = _box(5, 4, 1)
        return ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=_exact_flux(mesh, _shear(2.0)),
                seed="patch",
                inlet_patch="XM",
                scalars={"one": np.ones(mesh.n_cells)},
                **kw,
            )
        )

    def test_validation(self):
        tr = self._track()
        with pytest.raises(ValueError, match="n_bins"):
            ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr, n_bins=0))
        with pytest.raises(ValueError, match="未知の名前"):
            ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr, rate_scalars=("nope",)))

    def test_no_escaped_particle_raises(self):
        tr = self._track(max_steps=1)
        with pytest.raises(ValueError, match="脱出した粒子が 1 つも無い"):
            ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr))

    def test_distribution_is_normalised(self):
        rtd = ResidenceTimeProcess().execute(ResidenceTimeInput(track=self._track(), n_bins=40))
        assert float(np.sum(rtd.E * np.diff(rtd.t_edges))) == pytest.approx(1.0, rel=1e-12)
        assert rtd.F[0] == 0.0 and rtd.F[-1] == pytest.approx(1.0, rel=1e-12)
        assert np.all(np.diff(rtd.F) >= -1e-15)
        assert np.all(np.diff(rtd.F_ecdf) >= -1e-15)
        assert rtd.t_p10 <= rtd.t_p50 <= rtd.t_p90
        assert rtd.escaped_fraction == pytest.approx(1.0)
        assert rtd.unresolved_weight_fraction == 0.0
        assert rtd.exit_weight_fraction == {"XP": pytest.approx(1.0)}

    def test_rate_scalar_is_a_time_average(self):
        """``rate_scalars`` に入れた量は ∫s dt / t になる（∫1 dt / t = 1）."""
        tr = self._track()
        plain = ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr))
        rate = ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr, rate_scalars=("one",)))
        assert rate.scalar_mean["one"] == pytest.approx(1.0, rel=1e-12)
        assert plain.scalar_mean["one"] == pytest.approx(plain.t_mean, rel=1e-12)


class TestResidenceTimePhysics:
    def test_plug_flow_has_no_spread(self):
        mesh = _box(4, 3, 1)
        tr = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh,
                face_flux=_exact_flux(mesh, _uniform((2.0, 0.0, 0.0))),
                seed="patch",
                inlet_patch="XM",
            )
        )
        rtd = ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr, n_bins=20))
        assert rtd.spread == pytest.approx(1.0, rel=1e-9)
        assert rtd.t_p50 == pytest.approx(0.5, rel=1e-5)

    def test_shear_flow_spread_matches_the_analytic_distribution(self):
        """単純せん断の RTD は F(t) = 1 − t_min/t（重み u ∝ y、t ∝ 1/y の直接計算）."""
        s, lx = 3.0, 1.0
        mesh = _box(4, 40, 1, lx=lx)
        tr = ParticleTrackFVMProcess().execute(
            ParticleTrackFVMInput(
                mesh=mesh, face_flux=_exact_flux(mesh, _shear(s)), seed="patch", inlet_patch="XM"
            )
        )
        rtd = ResidenceTimeProcess().execute(ResidenceTimeInput(track=tr))
        # t = lx/(s y)、重み ∝ y なので t の分布は F(t) = 1 − (t_min/t)²
        f_exact = 1.0 - (rtd.t_min / rtd.t_ecdf) ** 2
        assert np.max(np.abs(rtd.F_ecdf - f_exact)) < 3e-2
        assert rtd.t_p90 / rtd.t_p10 > 2.0
