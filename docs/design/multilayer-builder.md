# 多層シート物性値ビルダー 設計仕様

[← README](../../README.md) | [← 設計文書索引](README.md)

## 概要

多層構造体（積層シート等）の物性値配列（k, C, q）を
層定義リストから自動構築する PreProcess。

## 入力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `layers` | `tuple[LayerSpec, ...]` | 層定義（下から上へ） |
| `nx, ny` | `int` | x, y 方向のセル数 |
| `Lx, Ly` | `float` | x, y 方向の領域サイズ [m] |
| `T0_default` | `float` | デフォルト初期温度 [K] |

### LayerSpec

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `thickness` | `float` | 層厚 [m] |
| `k` | `float` | 熱伝導率 [W/(m·K)] |
| `C` | `float` | 体積熱容量 [J/(m³·K)] |
| `q` | `float` | 体積発熱量 [W/m³] |
| `name` | `str` | 層名（可視化用） |

## 出力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `k` | `np.ndarray` | 熱伝導率配列 (nx, ny, nz) |
| `C` | `np.ndarray` | 熱容量配列 (nx, ny, nz) |
| `q` | `np.ndarray` | 発熱量配列 (nx, ny, nz) |
| `T0` | `np.ndarray` | 初期温度配列 (nx, ny, nz) |
| `Lz` | `float` | z方向の合計厚み [m] |
| `nz` | `int` | z方向の合計セル数 |
| `layer_boundaries` | `tuple[float, ...]` | 層境界z座標 [m] |
| `layer_names` | `tuple[str, ...]` | 層名一覧 |

## 離散化方針

- z方向は層厚に基づいて各層のセル数を決定
- 最小1セル/層を保証
- z方向の格子幅は全体で均一（dz = Lz / nz_total）
- 各層の厚みに最も近いセル数を割り当て

## プロセス設計

### MultilayerBuilderProcess

- **カテゴリ**: `PreProcess[MultilayerInput, MultilayerOutput]`
- **依存**: なし
