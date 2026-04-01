"""お線香の煙の参考流速パターン図 + 現ソルバー結果との比較.

お線香の煙に見られる自然対流パターン（層流プルーム → 遷移 → 乱流）を
模式的に描画し、現ソルバーの結果との差異を明確にする。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_incense_reference(ax):
    """お線香の煙の理想的な自然対流パターンを模式的に描画."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 5.0)
    ax.set_aspect("equal")

    # 発熱源（お線香の先端）
    ax.plot(0, 0, "ro", markersize=10, zorder=5)
    ax.annotate(
        "Heat source\n(発熱源)",
        (0, 0),
        (-1.0, -0.3),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )

    # 層流領域のプルーム（下部）- 滑らかな上昇流
    y_lam = np.linspace(0.1, 2.0, 50)
    # プルーム幅が高さとともに広がる
    width = 0.15 + 0.08 * y_lam
    ax.fill_betweenx(y_lam, -width, width, alpha=0.15, color="gray")

    # 層流の速度ベクトル（まっすぐ上向き）
    for y in np.arange(0.3, 2.0, 0.3):
        w = 0.15 + 0.08 * y
        for x in np.linspace(-w * 0.6, w * 0.6, 3):
            speed = 0.4 * (1.0 - abs(x) / w)  # 中心が速い
            ax.annotate(
                "",
                (x, y + speed * 0.5),
                (x, y),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            )

    # エントレインメント（周囲空気の巻き込み）
    for y in np.arange(0.5, 1.8, 0.4):
        w = 0.15 + 0.08 * y
        # 左から
        ax.annotate(
            "",
            (-w * 0.9, y),
            (-w * 0.9 - 0.4, y),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.0, alpha=0.6),
        )
        # 右から
        ax.annotate(
            "",
            (w * 0.9, y),
            (w * 0.9 + 0.4, y),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.0, alpha=0.6),
        )

    # 遷移領域（波打ち始め）
    y_trans = np.linspace(2.0, 3.0, 30)
    np.random.seed(42)
    for i, y in enumerate(y_trans):
        w = 0.2 + 0.1 * y
        wobble = 0.05 * np.sin(y * 8) * (y - 2.0)
        ax.plot(
            [wobble - w, wobble + w],
            [y, y],
            color="gray",
            alpha=0.1 + 0.02 * i,
            lw=2,
        )

    # 乱流領域（ランダムな渦）
    y_turb = np.linspace(3.0, 4.5, 40)
    for y in y_turb:
        w = 0.3 + 0.15 * (y - 3.0)
        n_pts = 8
        angles = np.random.uniform(0, 2 * np.pi, n_pts)
        rs = np.random.uniform(0, w, n_pts)
        xs = rs * np.cos(angles)
        ys_scatter = y + rs * np.sin(angles) * 0.3
        ax.scatter(xs, ys_scatter, s=2, color="gray", alpha=0.3)

    # ラベル
    ax.text(1.2, 0.8, "Laminar\n(層流)", fontsize=9, ha="center", color="blue")
    ax.text(1.2, 2.5, "Transition\n(遷移)", fontsize=9, ha="center", color="orange")
    ax.text(1.2, 3.8, "Turbulent\n(乱流)", fontsize=9, ha="center", color="red")

    # 境界線
    ax.axhline(2.0, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(3.0, color="red", linestyle="--", alpha=0.5)

    # 巻き込み注釈
    ax.text(
        -1.3,
        1.2,
        "Entrainment\n(巻き込み)",
        fontsize=7,
        ha="center",
        color="green",
    )

    ax.set_title("理想的なお線香の煙\n(自然対流プルーム)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y (上向き)")


def draw_expected_solver(ax):
    """q_vol中心発熱源で期待されるソルバー結果を模式的に描画."""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")

    # 発熱体（中心）
    rect = plt.Rectangle((2, 2), 1, 1, facecolor="red", alpha=0.3, edgecolor="red", lw=2)
    ax.add_patch(rect)
    ax.text(2.5, 2.5, "q_vol\nheat", ha="center", va="center", fontsize=7, color="red")

    # 期待される流れ: 中心から上向きプルーム
    # 上昇流（発熱体直上）
    for x in [2.2, 2.5, 2.8]:
        for y_base in np.arange(3.1, 4.5, 0.4):
            speed = 0.3 * (1.0 - abs(x - 2.5) / 0.5)
            ax.annotate(
                "",
                (x, y_base + speed),
                (x, y_base),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            )

    # 発熱体下方: 巻き込み（下から上へ弱い流れ）
    for x in [2.2, 2.5, 2.8]:
        for y_base in np.arange(0.5, 1.8, 0.5):
            ax.annotate(
                "",
                (x, y_base + 0.15),
                (x, y_base),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.0, alpha=0.5),
            )

    # 側方: エントレインメント
    for y in np.arange(2.5, 4.0, 0.5):
        ax.annotate(
            "",
            (1.9, y),
            (1.2, y),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.0, alpha=0.5),
        )
        ax.annotate(
            "",
            (3.1, y),
            (3.8, y),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.0, alpha=0.5),
        )

    # 上部: 流出（広がり）
    for x, dx in [(2.0, -0.15), (2.5, 0.0), (3.0, 0.15)]:
        ax.annotate(
            "",
            (x + dx, 4.8),
            (x, 4.5),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.0, alpha=0.7),
        )

    # 温度等高線（模式）
    for r, alpha in [(0.4, 0.4), (0.7, 0.25), (1.0, 0.15), (1.5, 0.08)]:
        circle = plt.Circle((2.5, 3.0), r, fill=False, edgecolor="orange", alpha=alpha, lw=1)
        ax.add_patch(circle)

    # ラベル
    ax.text(2.5, 4.9, "outflow (流出)", fontsize=7, ha="center", color="blue")
    ax.text(0.8, 3.0, "entrainment", fontsize=7, ha="center", color="green", rotation=90)

    ax.set_title("期待されるソルバー結果\n(q_vol中心発熱, 密閉BC)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def draw_actual_problems(ax):
    """現ソルバー結果で確認された問題を模式的に描画."""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")

    # 発熱体（中心）
    rect = plt.Rectangle((2, 2), 1, 1, facecolor="red", alpha=0.3, edgecolor="red", lw=2)
    ax.add_patch(rect)
    ax.text(2.5, 2.5, "q_vol\nheat", ha="center", va="center", fontsize=7, color="red")

    # 問題1: チェッカーボード温度
    for i in range(10):
        for j in range(10):
            x = 0.25 + i * 0.5
            y = 0.25 + j * 0.5
            if (i + j) % 2 == 0:
                color = "yellow"
            else:
                color = "black"
            rect_cb = plt.Rectangle(
                (x - 0.2, y - 0.2),
                0.4,
                0.4,
                facecolor=color,
                alpha=0.08,
                edgecolor="none",
            )
            ax.add_patch(rect_cb)

    # 問題2: 流速方向逆転（下向き矢印が発熱体上に）
    for x in [2.2, 2.5, 2.8]:
        for y_base in np.arange(3.2, 4.5, 0.4):
            ax.annotate(
                "",
                (x, y_base - 0.25),
                (x, y_base),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

    # 問題3: 高温部が発熱体外
    hot_spots = [(0.8, 1.2), (4.0, 3.5), (1.5, 4.2)]
    for hx, hy in hot_spots:
        ax.plot(hx, hy, "o", color="yellow", markersize=15, alpha=0.5)
        ax.plot(hx, hy, "x", color="red", markersize=10)

    # 注釈
    ax.annotate(
        "Flow reversed!\n(流速逆転 -y)",
        (2.5, 3.8),
        (3.8, 4.5),
        fontsize=8,
        ha="center",
        color="red",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax.annotate(
        "Hot spot at\nwrong location",
        (0.8, 1.2),
        (0.3, 0.3),
        fontsize=7,
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    ax.text(
        2.5,
        0.2,
        "Checkerboard T pattern",
        fontsize=8,
        ha="center",
        color="purple",
        fontweight="bold",
    )

    ax.set_title(
        "現ソルバー結果の問題\n(フォーカスガード登録済み)",
        fontsize=10,
        fontweight="bold",
        color="red",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    draw_incense_reference(axes[0])
    draw_expected_solver(axes[1])
    draw_actual_problems(axes[2])

    fig.suptitle(
        "自然対流プルーム: 期待パターン vs 現ソルバー結果の問題",
        fontsize=13,
        fontweight="bold",
    )

    out_path = output_dir / "incense_smoke_reference.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
