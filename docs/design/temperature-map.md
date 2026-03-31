# 温度マップ可視化 PostProcess 設計仕様

[← README](../../README.md) | [← 設計文書一覧](README.md)

## 概要

伝熱解析結果の温度場を2Dスライスとして可視化する PostProcess。

## プロセス情報

| 項目 | 値 |
|------|-----|
| クラス名 | `TemperatureMapProcess` |
| カテゴリ | PostProcess |
| 入力 | `TemperatureMapInput` |
| 出力 | `TemperatureMapOutput` |
| 安定性 | experimental |

## 機能

- 任意軸（x, y, z）でのスライス断面表示
- カラーマップによる温度分布の可視化
- 層境界線・層ラベルの描画（多層構造対応）
- PNG ファイルへの保存

## 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| result | HeatTransferResult | 伝熱解析結果 |
| Lx, Ly, Lz | float | 領域サイズ [m] |
| slice_axis | str | スライス軸 ("x", "y", "z") |
| slice_index | int/None | スライス位置（None=中央） |
| layer_boundaries | tuple[float] | 層境界位置 [m] |
| layer_labels | tuple[str] | 各層のラベル |
| output_path | Path/None | 保存先パス |

## 出力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| fig | Figure | matplotlib Figure オブジェクト |
| saved_path | Path/None | 保存先パス |
| T_min | float | スライス内最小温度 |
| T_max | float | スライス内最大温度 |
