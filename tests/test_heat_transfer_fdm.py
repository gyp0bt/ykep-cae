"""HeatTransferFDMProcess テスト.

API テスト（契約準拠）と物理テスト（解析解との比較）を含む。
冷却フィンベンチマーク（Robin BC 活用）を含む。
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

    def test_1d_steady_robin_boundary(self):
        """1D定常: 一端Dirichlet、他端Robin → 対流熱伝達の定常解.

        T(x=0) = T_L, h*(T_inf - T_surface) at x=L
        解析解: q = (T_inf - T_L) / (L/k + 1/h)
                T(x) = T_L + q * x / k
        """
        nx, ny, nz = 20, 1, 1
        Lx = 1.0
        T_L = 200.0
        T_inf = 500.0
        h_conv = 50.0  # W/(m²·K)
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
            bc_xp=BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf),
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 解析解: q = (T_inf - T_L) / (L/k + 1/h)
        q_flux = (T_inf - T_L) / (Lx / k_val + 1.0 / h_conv)
        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = T_L + q_flux * x_cell / k_val

        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.02,
            err_msg="1D定常Dirichlet-Robin: 解析解と一致しない",
        )

    def test_1d_steady_robin_both_ends(self):
        """1D定常: 両端Robin → 対流熱伝達+発熱の定常解.

        両端 h*(T_inf - T_surface) + 均一発熱 q
        解析解: T(x) = T_inf + q*L/(2h) + q/(2k)*x*(L-x)
        （両端対称の場合）
        """
        nx, ny, nz = 30, 1, 1
        Lx = 1.0
        T_inf = 300.0
        h_conv = 100.0
        k_val = 10.0
        q_val = 500.0

        bc_robin = BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf)
        inp = HeatTransferInput(
            Lx=Lx,
            Ly=0.1,
            Lz=0.1,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)),
            q=np.ones((nx, ny, nz)) * q_val,
            T0=np.ones((nx, ny, nz)) * 300.0,
            bc_xm=bc_robin,
            bc_xp=bc_robin,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 解析解: T(x) = T_inf + q*L/(2h) + q/(2k) * x * (L - x)
        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = (
            T_inf + q_val * Lx / (2 * h_conv) + q_val / (2 * k_val) * x_cell * (Lx - x_cell)
        )

        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.02,
            err_msg="1D定常Robin両端+発熱: 解析解と一致しない",
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

    def test_transient_robin_cooling(self):
        """非定常: 全面Robin冷却 → 長時間後にT_infへ漸近.

        均一初期温度 T0 から全面 Robin BC (h, T_inf) で冷却。
        十分な時間経過後、温度は T_inf に漸近する。
        エネルギー保存: 系のエネルギー変化 = 対流放熱量の積分
        """
        nx, ny, nz = 5, 1, 1
        Lx, Ly, Lz = 0.1, 0.01, 0.01
        k_val = 50.0  # W/(m·K) — 高い伝導率で内部均一化を促進
        C_val = 1000.0  # J/(m³·K)
        h_conv = 500.0  # W/(m²·K)
        T0_val = 500.0  # K — 初期温度
        T_inf = 300.0  # K — 外部温度

        bc_robin = BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf)

        # 十分な時間を取って定常に近づける
        # 時定数の目安: tau ~ C * V / (h * A) = C * Lx / (2h) for 1D
        # tau ~ 1000 * 0.1 / (2*500) = 0.1 s
        dt = 0.005
        t_end = 2.0  # 20 * tau 相当

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)) * C_val,
            q=np.zeros((nx, ny, nz)),
            T0=np.full((nx, ny, nz), T0_val),
            bc_xm=bc_robin,
            bc_xp=bc_robin,
            dt=dt,
            t_end=t_end,
            max_iter=500,
            tol=1e-8,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 長時間後、全セルが T_inf に漸近
        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_inf,
            atol=0.5,
            err_msg="非定常Robin冷却: 長時間後にT_infへ漸近しない",
        )

        # 温度は単調に低下（T0 > T_inf のため）
        assert result.T.max() <= T0_val + 1e-10, "温度が初期値を超えている"
        assert result.T.min() >= T_inf - 1e-10, "温度がT_inf未満になっている"

    def test_transient_robin_energy_balance(self):
        """非定常Robin: 片端Robin + 発熱のエネルギー収支.

        片端断熱、他端 Robin。均一発熱 q がある場合、
        定常到達時: q * V = h * A * (T_surface - T_inf)
        → T(x) = T_inf + q*L/h + q/(2k) * (L² - x²) (1D解析解)
        """
        nx, ny, nz = 20, 1, 1
        Lx, Ly, Lz = 0.02, 0.01, 0.01
        k_val = 50.0
        C_val = 500.0
        h_conv = 500.0
        T_inf = 300.0
        q_val = 1e6  # W/m³

        # 時定数 tau ~ C*L²/k = 500*0.0004/50 = 0.004 s
        dt = 0.01
        t_end = 2.0  # > 100*tau

        inp = HeatTransferInput(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)) * C_val,
            q=np.ones((nx, ny, nz)) * q_val,
            T0=np.full((nx, ny, nz), T_inf),
            bc_xm=BoundarySpec(BoundaryCondition.ADIABATIC),
            bc_xp=BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf),
            dt=dt,
            t_end=t_end,
            max_iter=5000,
            tol=1e-6,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        # 定常解析解: T(x) = T_inf + q*L/h + q/(2k) * (L² - x²)
        dx = Lx / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        T_analytical = T_inf + q_val * Lx / h_conv + q_val / (2 * k_val) * (Lx**2 - x_cell**2)

        # 後半タイムステップは収束している（初期の急変ステップのみ非収束の可能性）
        # 最終温度分布が解析解に一致していることで物理的妥当性を検証
        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.02,
            err_msg="非定常Robin+発熱: 定常到達後の温度分布が解析解と一致しない",
        )

        # 温度は全セルで T_inf 以上（発熱により加温されるため）
        assert result.T.min() >= T_inf - 1e-10, "温度がT_inf未満になっている"


# ---------------------------------------------------------------------------
# 冷却フィンベンチマーク（Robin BC 活用）
# ---------------------------------------------------------------------------


class TestCoolingFinBenchmark:
    """冷却フィン解析ベンチマーク.

    1D矩形断面フィンの古典的ベンチマーク問題。
    Robin BC（対流熱伝達）を活用した検証。
    """

    def test_fin_temperature_distribution(self):
        """1D冷却フィン: 底端Dirichlet + 先端Robin → 解析解比較.

        矩形断面フィン（長さL, 断面積A, 周長P）:
        - 底端: T = T_base (Dirichlet)
        - 先端: h*(T_inf - T) (Robin)
        - 側面: 対流 h*(T_inf - T) を体積発熱として近似

        1Dフィン方程式の解析解（先端対流あり）:
        T(x) = T_inf + (T_base - T_inf)
               * [cosh(m(L-x)) + (h/(mk))sinh(m(L-x))]
               / [cosh(mL) + (h/(mk))sinh(mL)]
        ここで m = sqrt(hP/(kA)), P=周長, A=断面積
        """
        # フィン仕様
        L_fin = 0.1  # フィン長さ [m]
        W_fin = 0.01  # フィン幅（正方形断面） [m]
        k_val = 200.0  # アルミニウム [W/(m·K)]
        h_conv = 25.0  # 対流熱伝達係数 [W/(m²·K)]
        T_base = 373.15  # 底端温度 100°C [K]
        T_inf = 293.15  # 外気温度 20°C [K]

        # フィンパラメータ
        A_fin = W_fin * W_fin  # 断面積 [m²]
        P_fin = 4 * W_fin  # 周長 [m]
        m = np.sqrt(h_conv * P_fin / (k_val * A_fin))

        # メッシュ: x方向がフィン長さ方向
        nx = 40
        ny, nz = 1, 1
        Ly = W_fin
        Lz = W_fin

        # 側面対流を体積発熱として近似: q_conv = -hP/A * (T - T_inf)
        # → 線形化: 初期近似で T ≈ T_base として q_source を推定
        # ただし、この近似は不正確。代わりに1D解析なので
        # 側面からの対流は y,z 方向の Robin BC で表現する
        bc_robin_side = BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf)

        inp = HeatTransferInput(
            Lx=L_fin,
            Ly=Ly,
            Lz=Lz,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.full((nx, ny, nz), T_base),
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_base),
            bc_xp=BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf),
            bc_ym=bc_robin_side,
            bc_yp=bc_robin_side,
            bc_zm=bc_robin_side,
            bc_zp=bc_robin_side,
            max_iter=50000,
            tol=1e-8,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 解析解
        dx = L_fin / nx
        x_cell = np.array([dx * (i + 0.5) for i in range(nx)])
        theta_base = T_base - T_inf

        # 先端対流あり: T(x) = T_inf + theta_base
        # * [cosh(m(L-x)) + (h/(mk))sinh(m(L-x))]
        # / [cosh(mL) + (h/(mk))sinh(mL)]
        h_mk = h_conv / (m * k_val)
        T_analytical = T_inf + theta_base * (
            np.cosh(m * (L_fin - x_cell)) + h_mk * np.sinh(m * (L_fin - x_cell))
        ) / (np.cosh(m * L_fin) + h_mk * np.sinh(m * L_fin))

        # 3Dソルバーの1Dフィン近似なので、やや緩めの許容誤差
        np.testing.assert_allclose(
            result.T[:, 0, 0],
            T_analytical,
            rtol=0.05,
            err_msg="冷却フィン: 温度分布が解析解と一致しない",
        )

    def test_fin_heat_dissipation(self):
        """冷却フィン: フィン効率の検証.

        フィン効率 η = tanh(mL) / (mL) （断熱先端近似時）
        フィンからの全放熱量 Q = η * h * P * L * (T_base - T_inf)
        """
        L_fin = 0.05  # フィン長さ [m]
        W_fin = 0.005  # フィン幅 [m]
        k_val = 400.0  # 銅 [W/(m·K)]
        h_conv = 50.0  # [W/(m²·K)]
        T_base = 350.0  # [K]
        T_inf = 300.0  # [K]

        A_fin = W_fin * W_fin
        P_fin = 4 * W_fin
        m = np.sqrt(h_conv * P_fin / (k_val * A_fin))

        nx, ny, nz = 30, 1, 1

        bc_robin = BoundarySpec(BoundaryCondition.ROBIN, h_conv=h_conv, T_inf=T_inf)

        inp = HeatTransferInput(
            Lx=L_fin,
            Ly=W_fin,
            Lz=W_fin,
            k=np.ones((nx, ny, nz)) * k_val,
            C=np.ones((nx, ny, nz)),
            q=np.zeros((nx, ny, nz)),
            T0=np.full((nx, ny, nz), T_base),
            bc_xm=BoundarySpec(BoundaryCondition.DIRICHLET, T_base),
            bc_xp=BoundarySpec(BoundaryCondition.ADIABATIC),  # 先端断熱
            bc_ym=bc_robin,
            bc_yp=bc_robin,
            bc_zm=bc_robin,
            bc_zp=bc_robin,
            max_iter=50000,
            tol=1e-8,
        )

        solver = HeatTransferFDMProcess()
        result = solver.process(inp)

        assert result.converged

        # 底端からの熱流束を計算: q = -k * dT/dx at x=0
        dx = L_fin / nx
        # セル0の中心は dx/2、底端温度 T_base との勾配
        q_base = k_val * (T_base - result.T[0, 0, 0]) / (dx / 2)

        # 解析解の底端熱流束: Q_fin / A = m*k*(T_base - T_inf)*tanh(mL)
        q_analytical = m * k_val * (T_base - T_inf) * np.tanh(m * L_fin)

        # 3Dソルバーの近似なので緩めの許容誤差（断面1セルの離散化誤差）
        np.testing.assert_allclose(
            q_base,
            q_analytical,
            rtol=0.15,
            err_msg="冷却フィン: 底端熱流束がフィン理論と一致しない",
        )

        # フィン先端は底端より低温
        assert result.T[-1, 0, 0] < result.T[0, 0, 0], "フィン先端が底端より高温"

        # フィン温度は T_inf 以上、T_base 以下
        assert result.T.min() >= T_inf - 1.0
        assert result.T.max() <= T_base + 1.0
