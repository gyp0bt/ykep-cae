"""4層多層シート中心発熱の温度マップ描画.

4層構造の平たいシート（厚み1mm, 幅7mm）を1/8対称条件でモデル化し、
中心付近の発熱による定常温度分布を可視化する。

モデル概要:
    - 全幅: 7mm, 全長: 7mm, 全厚: 1mm
    - 1/8対称: x=3.5mm, y=3.5mm, z=0.5mm
    - 4層構造（z方向に積層、対称配置）:
        Layer 1 (z=0.00-0.25mm): セラミック(alumina) k=25 W/(m·K)
        Layer 2 (z=0.25-0.50mm): 鋼(steel)           k=50 W/(m·K)
        Layer 3 (z=0.50-0.75mm): 鋼(steel)           k=50 W/(m·K) [対称]
        Layer 4 (z=0.75-1.00mm): セラミック(alumina)  k=25 W/(m·K) [対称]
      ※ 対称配置のため半厚モデル(z=0-0.5mm)で Layer1 + Layer2 を解析
    - 中心付近（対称面近傍、x<1mm, y<1mm 領域）で Layer2(鋼) に発熱
    - 対称面（x=0, y=0, z=0）は断熱
    - 外面（x+, y+）は温度固定 25°C、z+（シート表面）は対流相当の温度固定
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xkep_cae_fluid.heat_transfer import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferFDMProcess,
    HeatTransferInput,
    TemperatureMapInput,
    TemperatureMapProcess,
)


def build_multilayer_sheet_input() -> HeatTransferInput:
    """4層多層シートの1/8対称モデル入力を構築する."""

    # --- ジオメトリ（1/8対称） ---
    Lx = 3.5e-3  # 半幅 [m]
    Ly = 3.5e-3  # 半長 [m]
    Lz = 0.5e-3  # 半厚 [m]

    # --- メッシュ ---
    nx = 70  # x方向セル数
    ny = 70  # y方向セル数
    nz = 20  # z方向セル数（各層10セル）

    # --- 材料物性（対称配置の半厚分） ---
    # Layer 1: z = 0 〜 0.25mm (セル 0-9)  : セラミック (alumina)
    # Layer 2: z = 0.25 〜 0.50mm (セル 10-19): 鋼 (steel)
    k = np.zeros((nx, ny, nz))
    C = np.zeros((nx, ny, nz))

    n_half = nz // 2  # 各層10セル

    # Layer 1: セラミック (alumina)
    k[:, :, :n_half] = 25.0  # W/(m·K)
    C[:, :, :n_half] = 3900.0 * 880.0  # rho=3900 kg/m3, Cp=880 J/(kg·K)

    # Layer 2: 鋼 (steel)
    k[:, :, n_half:] = 50.0  # W/(m·K)
    C[:, :, n_half:] = 7800.0 * 500.0  # rho=7800 kg/m3, Cp=500 J/(kg·K)

    # --- 発熱源（中心付近 Layer2 鋼層のみ） ---
    dx = Lx / nx
    dy = Ly / ny
    q = np.zeros((nx, ny, nz))
    heat_nx = int(1.0e-3 / dx)  # 1mm 範囲
    heat_ny = int(1.0e-3 / dy)  # 1mm 範囲
    # Layer2 (鋼) の対称面側に発熱（シート中央面付近が発熱源）
    q[:heat_nx, :heat_ny, n_half:] = 5e9  # W/m3

    # --- 初期温度・境界条件 ---
    T0 = np.full((nx, ny, nz), 298.15)

    bc_adiabatic = BoundarySpec(BoundaryCondition.ADIABATIC)
    bc_fixed = BoundarySpec(BoundaryCondition.DIRICHLET, value=298.15)

    return HeatTransferInput(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        k=k,
        C=C,
        q=q,
        T0=T0,
        bc_xm=bc_adiabatic,  # symmetry
        bc_ym=bc_adiabatic,  # symmetry
        bc_zm=bc_adiabatic,  # symmetry (midplane)
        bc_xp=bc_fixed,  # outer edge
        bc_yp=bc_fixed,  # outer edge
        bc_zp=bc_fixed,  # sheet surface (cooling)
        dt=0.0,  # steady state
        max_iter=80000,
        tol=1e-8,
    )


def main() -> None:
    """メイン実行."""
    print("=" * 60)
    print("4-layer sheet thermal analysis (1/8 symmetry)")
    print("=" * 60)

    inp = build_multilayer_sheet_input()
    print(f"Mesh: {inp.nx} x {inp.ny} x {inp.nz} = {inp.nx * inp.ny * inp.nz} cells")
    print(f"Domain: {inp.Lx * 1e3:.1f} x {inp.Ly * 1e3:.1f} x {inp.Lz * 1e3:.1f} mm")
    print(f"Cell size: {inp.dx * 1e6:.0f} x {inp.dy * 1e6:.0f} x {inp.dz * 1e6:.0f} um")
    print()

    print("Running steady-state analysis...")
    solver = HeatTransferFDMProcess(vectorized=True)
    result = solver.process(inp)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iteration_counts[0]}")
    print(f"Elapsed: {result.elapsed_seconds:.2f} s")
    print(f"Final residual: {result.residual_history[0][-1]:.2e}")
    print(f"T range: {result.T.min():.2f} ~ {result.T.max():.2f} K")
    print(f"         ({result.T.min() - 273.15:.2f} ~ {result.T.max() - 273.15:.2f} deg C)")
    print()

    if not result.converged:
        print("WARNING: Not converged. Results are approximate.")
        print()

    # --- Visualization ---
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    Lx, Ly, Lz = inp.Lx, inp.Ly, inp.Lz
    layer_boundary_z = 0.25e-3  # Layer1/Layer2 boundary

    viz = TemperatureMapProcess()

    # (1) x-z cross-section (y=0 symmetry plane)
    viz_inp_xz = TemperatureMapInput(
        result=result,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        slice_axis="y",
        slice_index=0,
        title="x-z cross-section (y=0 symmetry plane)",
        output_path=output_dir / "multilayer_sheet_xz.png",
        cmap="hot",
        layer_boundaries=(layer_boundary_z,),
        layer_labels=("L1: Ceramic", "L2: Steel"),
        figsize=(10, 4),
    )
    out_xz = viz.process(viz_inp_xz)
    print(f"x-z section: T_min={out_xz.T_min:.2f} K, T_max={out_xz.T_max:.2f} K")
    print(f"  -> {out_xz.saved_path}")

    # (2) x-y cross-section (z=0 symmetry plane, ceramic layer center)
    viz_inp_xy = TemperatureMapInput(
        result=result,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        slice_axis="z",
        slice_index=0,
        title="x-y cross-section (z=0 symmetry, L1 Ceramic center)",
        output_path=output_dir / "multilayer_sheet_xy.png",
        cmap="hot",
        figsize=(8, 8),
    )
    out_xy = viz.process(viz_inp_xy)
    print(f"x-y section: T_min={out_xy.T_min:.2f} K, T_max={out_xy.T_max:.2f} K")
    print(f"  -> {out_xy.saved_path}")

    # (3) y-z cross-section (x=0 symmetry plane)
    viz_inp_yz = TemperatureMapInput(
        result=result,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        slice_axis="x",
        slice_index=0,
        title="y-z cross-section (x=0 symmetry plane)",
        output_path=output_dir / "multilayer_sheet_yz.png",
        cmap="hot",
        layer_boundaries=(layer_boundary_z,),
        layer_labels=("L1: Ceramic", "L2: Steel"),
        figsize=(10, 4),
    )
    out_yz = viz.process(viz_inp_yz)
    print(f"y-z section: T_min={out_yz.T_min:.2f} K, T_max={out_yz.T_max:.2f} K")
    print(f"  -> {out_yz.saved_path}")

    import matplotlib.pyplot as plt

    plt.close("all")

    print()
    print("Done. Temperature maps saved to output/ directory.")


if __name__ == "__main__":
    main()
