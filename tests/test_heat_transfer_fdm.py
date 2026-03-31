"""HeatTransferFDMProcess テスト.

API テスト（契約準拠）と物理テスト（解析解との比較）を含む。
"""

import numpy as np

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.heat_transfer.data import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferInput,
    HeatTransferResult,
)
from xkep_cae_fluid.heat_transfer.solver import HeatTransferFDMProcess

# ---------------------------------------------------------------------------
# API テスト
# ---------------------------------------------------------------------------


@binds_to(HeatTransferFDMProcess)
class TestHeatTransferFDMAPI:
    """HeatTransferFDMProcess の Process 契約準拠テスト."""

    def test_meta_exists(self):
        """ProcessMeta が定義されていること."""
        assert HeatTransferFDMProcess.meta.name == "HeatTransferFDM"
        assert HeatTransferFDMProcess.meta.module == "solve"

    def test_process_returns_result(self):
        """process() が HeatTransferResult を返すこと."""
        nx, ny, nz = 3, 3, 3
        inp = HeatTransferInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            k=np.ones((nx, ny, nz)),
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.ones((nx, ny, nz)) * 300.0,
        )
        solver = HeatTransferFDMProcess()
        result = solver.process(inp)
        assert isinstance(result, HeatTransferResult)
        assert result.T.shape == (nx, ny, nz)

    def test_steady_uniform_no_source(self):
        """発熱なし・全面断熱の定常解析で初期温度が維持されること."""
        nx, ny, nz = 4, 4, 4
        T_init = 350.0
        inp = HeatTransferInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            k=np.ones((nx, ny, nz)),
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.full((nx, ny, nz), T_init),
        )
        solver = HeatTransferFDMProcess()
        result = solver.process(inp)
        assert result.converged
        np.testing.assert_allclose(result.T, T_init, atol=1e-10)

    def test_transient_returns_history(self):
        """非定常解析でタイムステップ履歴が返ること."""
        nx, ny, nz = 3, 3, 3
        inp = HeatTransferInput(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            k=np.ones((nx, ny, nz)),
            C=np.ones((nx, ny, nz)) * 1000.0,
            q=np.zeros((nx, ny, nz)),
            T0=np.ones((nx, ny, nz)) * 300.0,
            dt=0.1,
            t_end=0.5,
            output_interval=1,
        )
        solver = HeatTransferFDMProcess()
        result = solver.process(inp)
        assert result.n_timesteps == 5
        assert len(result.time_history) == 5
        assert len(result.T_history) == 5


# ---------------------------------------------------------------------------
# 物理テスト
# ---------------------------------------------------------------------------


class TestHeatTransferFDMPhysics:
    """伝熱解析の物理的妥当性テスト."""

    def test_1d_steady_dirichlet_both_ends(self):
        """1D定常: 両端Dirichlet境界 → 線形温度分布.

        T(x=0) = T_L, T(x=L) = T_R の場合、
        解析解: T(x) = T_L + (T_R - T_L) * x / L
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        T_L, T_R = 100.0, 500.0

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=0.1,
            Lz=0.1,
            k=np.ones((nx, ny, nz)) * 10.0,
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.ones((nx, ny, nz)) * 300.0,
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_L),
            bc_xp=BoundarySpec(BoundaryCondition.DIRICHLET, T_R),
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # セル中心座標
        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = T_L + (T_R - T_L) * x_cell / Lx

        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.01,
            err_msg="1D定常Dirichlet両端: 線形分布と一致しない",
        )

    def test_1d_steady_dirichlet_neumann(self):
        """1D定常: 一端Dirichlet、他端Neumann → 定常解.

        T(x=0) = T_L, q_flux(x=L) = q_R
        発熱なしの場合、定常解: T(x) = T_L + q_R * x / k
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        T_L = 200.0
        q_R = 500.0  # W/m²
        k_val = 10.0

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=0.1,
            Lz=0.1,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.ones((nx, ny, nz)) * 300.0,
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_L),
            bc_xp=BoundarySpec(BoundaryCondition.NEUMANN, q_R),
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = T_L + q_R * x_cell / k_val

        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.02,
            err_msg="1D定常Dirichlet-Neumann: 解析解と一致しない",
        )

    def test_adiabatic_preserves_energy(self):
        """全面断熱 + 発熱 → エネルギー保存.

        全面断熱の場合、系のエネルギー増加 = 発熱量 * 時間
        """
        nx, ny, nz = 5, 5, 5
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        C_val = 1000.0
        q_val = 100.0
        dt = 0.01
        t_end = 0.1

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            k=np.ones((nx, ny, nz)) * 10.0,
            C=np.ones((nx, ny, nz)) * C_val,
            q=np.ones((nx, ny, nz)) * q_val,
            T0=np.ones((nx, ny, nz)) * 300.0,
            dt=dt,
            t_end=t_end,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # エネルギー増加 = C * ΔT
        dT_expected = q_val * t_end / C_val
        T_expected = 300.0 + dT_expected

        # 全面断熱なので均一に温度上昇するはず
        np.testing.assert_allclose(
            result.T.mean(),
            T_expected,
            rtol=1e-6,
            err_msg="断熱+均一発熱: エネルギー保存していない",
        )

    def test_1d_steady_with_source(self):
        """1D定常: 両端Dirichlet + 均一発熱.

        解析解: T(x) = T_L + (T_R - T_L)*x/L + q/(2k) * x * (L - x)
        """
        nx, ny, nz = 30, 1, 1
        Lx = 1.0
        T_L, T_R = 100.0, 100.0
        q_val = 1000.0
        k_val = 10.0

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=0.1,
            Lz=0.1,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)),
            q=np.ones((nx, ny, nz)) * q_val,
            T0=np.ones((nx, ny, nz)) * 100.0,
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_L),
            bc_xp=BoundarySpec(BoundaryCondition.DIRICHLET, T_R),
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = (
            T_L + (T_R - T_L) * x_cell / Lx + q_val / (2 * k_val) * x_cell * (Lx - x_cell)
        )

        np.testing.assert_allclose(
            result.T[:, 0, 0], T_analytical, rtol=0.02, err_msg="1D定常+発熱: 解析解と一致しない"
        )

    def test_heterogeneous_conductivity(self):
        """不均一熱伝導率: 2層構造の定常伝熱.

        k1 の領域と k2 の領域で温度勾配が異なる。
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        T_L, T_R = 100.0, 500.0
        k1, k2 = 10.0, 50.0

        k_arr = np.ones((nx, ny, nz))
        k_arr[: nx // 2] = k1
        k_arr[nx // 2 :] = k2

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=0.1,
            Lz=0.1,
            k=k_arr,
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.ones((nx, ny, nz)) * 300.0,
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_L),
            bc_xp=BoundarySpec(BoundaryCondition.DIRICHLET, T_R),
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 界面温度の検証: T_interface = T_L + (T_R - T_L) * k2 / (k1 + k2)
        # 定常熱流束 q = (T_R - T_L) / (L1/k1 + L2/k2)
        L1 = Lx / 2
        L2 = Lx / 2
        q_flux = (T_R - T_L) / (L1 / k1 + L2 / k2)
        T_interface = T_L + q_flux * L1 / k1

        # 界面付近のセルの温度をチェック
        i_left = nx // 2 - 1
        i_right = nx // 2
        T_avg_interface = (result.T[i_left, 0, 0] + result.T[i_right, 0, 0]) / 2

        np.testing.assert_allclose(
            T_avg_interface,
            T_interface,
            rtol=0.05,
            err_msg="不均一熱伝導率: 界面温度が解析解と一致しない",
        )
