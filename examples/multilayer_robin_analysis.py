"""MultilayerBuilderProcess + HeatTransferFDMProcess 連携例.

3層基板（銅-FR4-銅）に Robin BC（対流冷却）を適用した定常伝熱解析。
MultilayerBuilderProcess で物性値配列を自動構築し、
HeatTransferFDMProcess で温度分布を計算する。

モデル概要:
    - 3層基板: Cu(35μm) / FR4(1.0mm) / Cu(35μm) = 1.07mm 厚
    - 面内サイズ: 10mm x 10mm
    - 上面Cu層の中心 2mm x 2mm 領域に発熱（IC チップ想定）
    - 下面: Robin BC（自然対流 h=10 W/(m²·K), T_inf=25°C）
    - 上面: Robin BC（強制対流 h=50 W/(m²·K), T_inf=25°C）
    - 側面: 断熱
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xkep_cae_fluid.heat_transfer import (
    BoundaryCondition,
    BoundarySpec,
    HeatTransferFDMProcess,
    HeatTransferInput,
    LayerSpec,
    MultilayerBuilderProcess,
    MultilayerInput,
    TemperatureMapInput,
    TemperatureMapProcess,
)


def main() -> None:
    """メイン実行."""
    print("=" * 60)
    print("MultilayerBuilder + HeatTransferFDM 連携例")
    print("3層基板 (Cu/FR4/Cu) + Robin BC 対流冷却")
    print("=" * 60)

    # --- 1. MultilayerBuilderProcess で物性値配列を構築 ---
    layers = (
        LayerSpec(thickness=35e-6, k=398.0, C=8960.0 * 385.0, name="Cu-bottom"),
        LayerSpec(thickness=1.0e-3, k=0.3, C=1850.0 * 1100.0, name="FR4"),
        LayerSpec(thickness=35e-6, k=398.0, C=8960.0 * 385.0, name="Cu-top"),
    )

    Lx, Ly = 10e-3, 10e-3  # 10mm x 10mm
    nx, ny = 40, 40

    ml_input = MultilayerInput(
        layers=layers,
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        T0_default=298.15,
        nz_per_meter=10000.0,  # 0.1mm あたり 1セル
    )

    builder = MultilayerBuilderProcess()
    ml_output = builder.process(ml_input)

    print("\n[MultilayerBuilder 出力]")
    print(f"  層数: {len(layers)}")
    for i, name in enumerate(ml_output.layer_names):
        print(f"    {i + 1}. {name} (k={layers[i].k} W/(m·K))")
    print(f"  z方向セル数: {ml_output.nz}")
    print(f"  全厚: {ml_output.Lz * 1e3:.3f} mm")
    print(f"  配列形状: ({nx}, {ny}, {ml_output.nz})")

    # --- 2. 発熱源の設定（上面Cu層の中心 2mm x 2mm） ---
    q = ml_output.q.copy()  # MultilayerBuilder の出力をベースに
    dx, dy = Lx / nx, Ly / ny
    heat_x = int(2e-3 / dx)  # 2mm 分のセル数
    heat_y = int(2e-3 / dy)
    x_start = (nx - heat_x) // 2
    y_start = (ny - heat_y) // 2

    # 上面Cu層（最後の層）のみに発熱
    # 層境界から z方向のオフセットを計算
    nz_cu_bottom = max(1, round(35e-6 * ml_input.nz_per_meter))
    nz_fr4 = max(1, round(1.0e-3 * ml_input.nz_per_meter))
    z_top_start = nz_cu_bottom + nz_fr4

    q[x_start : x_start + heat_x, y_start : y_start + heat_y, z_top_start:] = 5e9  # W/m³

    print("\n[発熱源]")
    print(f"  位置: 上面Cu層 中心 {heat_x * dx * 1e3:.1f}mm x {heat_y * dy * 1e3:.1f}mm")
    print("  発熱密度: 5e9 W/m³")

    # --- 3. HeatTransferFDMProcess で解析 ---
    T_inf = 298.15  # 25°C

    inp = HeatTransferInput(
        Lx=Lx,
        Ly=Ly,
        Lz=ml_output.Lz,
        k=ml_output.k,
        C=ml_output.C,
        q=q,
        T0=ml_output.T0,
        bc_xm=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_xp=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_ym=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_yp=BoundarySpec(BoundaryCondition.ADIABATIC),
        bc_zm=BoundarySpec(BoundaryCondition.ROBIN, h_conv=10.0, T_inf=T_inf),
        bc_zp=BoundarySpec(BoundaryCondition.ROBIN, h_conv=50.0, T_inf=T_inf),
        max_iter=200000,
        tol=1e-6,
    )

    print("\n[解析条件]")
    print(f"  メッシュ: {inp.nx} x {inp.ny} x {inp.nz} = {inp.nx * inp.ny * inp.nz} cells")
    print(f"  下面: Robin (h=10 W/(m²·K), T_inf={T_inf - 273.15:.0f}°C)")
    print(f"  上面: Robin (h=50 W/(m²·K), T_inf={T_inf - 273.15:.0f}°C)")
    print("  側面: 断熱")

    print("\n解析実行中...")
    solver = HeatTransferFDMProcess(vectorized=True)
    result = solver.process(inp)

    print("\n[解析結果]")
    print(f"  収束: {result.converged}")
    print(f"  反復数: {result.iteration_counts[0]}")
    print(f"  計算時間: {result.elapsed_seconds:.2f} s")
    print(f"  最終残差: {result.residual_history[0][-1]:.2e}")
    print(f"  温度範囲: {result.T.min() - 273.15:.2f} ~ {result.T.max() - 273.15:.2f} °C")

    # --- 4. TemperatureMapProcess で可視化 ---
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    viz = TemperatureMapProcess()

    # (1) x-z 断面（y=中央）
    viz_xz = TemperatureMapInput(
        result=result,
        Lx=Lx,
        Ly=Ly,
        Lz=ml_output.Lz,
        slice_axis="y",
        slice_index=ny // 2,
        title="x-z cross-section (Cu/FR4/Cu, center y)",
        output_path=output_dir / "multilayer_robin_xz.png",
        cmap="hot",
        layer_boundaries=ml_output.layer_boundaries,
        layer_labels=list(ml_output.layer_names),
        figsize=(10, 4),
    )
    out_xz = viz.process(viz_xz)
    print(f"\n  x-z断面: {out_xz.saved_path}")

    # (2) x-y 断面（上面Cu層）
    viz_xy = TemperatureMapInput(
        result=result,
        Lx=Lx,
        Ly=Ly,
        Lz=ml_output.Lz,
        slice_axis="z",
        slice_index=ml_output.nz - 1,
        title="x-y cross-section (top Cu layer)",
        output_path=output_dir / "multilayer_robin_xy.png",
        cmap="hot",
        figsize=(8, 8),
    )
    out_xy = viz.process(viz_xy)
    print(f"  x-y断面: {out_xy.saved_path}")

    import matplotlib.pyplot as plt

    plt.close("all")

    print("\nDone.")


if __name__ == "__main__":
    main()
