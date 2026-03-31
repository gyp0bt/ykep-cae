# status-3: ソルバー高速化 + 可視化 PostProcess + 多層シート温度マップ

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-03-31

## 概要

status-2 の TODO を消化しつつ、4層多層シート中心発熱の温度マップ描画を実現。

## 実装内容

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae_fluid/heat_transfer/solver_vectorized.py` | NumPy ベクトル化ヤコビ法ソルバー |
| `xkep_cae_fluid/heat_transfer/visualize.py` | TemperatureMapProcess（可視化 PostProcess） |
| `tests/test_temperature_map.py` | TemperatureMapProcess テスト6件 |
| `examples/multilayer_sheet_temperature.py` | 4層多層シート温度マップ描画スクリプト |
| `docs/design/temperature-map.md` | 温度マップ可視化設計仕様書 |

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae_fluid/heat_transfer/solver.py` | ベクトル化版統合、`vectorized` パラメータ追加 |
| `xkep_cae_fluid/heat_transfer/__init__.py` | 可視化クラスの公開API追加 |

### 機能詳細

#### 1. ソルバー NumPy ベクトル化（status-2 TODO消化）

- Python 3重ループをNumPy配列演算に置換（ヤコビ法）
- `HeatTransferFDMProcess(vectorized=True)` でベクトル化版を使用（デフォルト）
- `vectorized=False` で従来のガウスザイデル法（参照実装）を選択可能
- 既存テスト9件全てパス

#### 2. 可視化 PostProcess（status-2 TODO消化）

- `TemperatureMapProcess`: 温度場の2Dスライス可視化
- 任意軸（x, y, z）でのスライス断面表示
- 層境界線・層ラベルの描画（多層構造対応）
- カラーマップ、vmin/vmax、figsize、dpi 等のカスタマイズ
- PNG ファイル保存機能

#### 3. 4層多層シート温度マップ

- 厚み1mm, 幅7mmの4層シートを1/8対称でモデル化
- 4層構造（対称配置）: セラミック(k=25) + 鋼(k=50) | 鋼(k=50) + セラミック(k=25)
- 中心付近（鋼層）に発熱源 5×10⁹ W/m³
- メッシュ: 70×70×20 = 98,000セル
- 定常解析: 3,636反復で収束、計算時間約28秒
- 温度範囲: 25.00〜28.03°C
- 3断面の温度マップ（x-z, x-y, y-z）を output/ に保存

## テスト結果

- テスト数: 31（既存25 + 新規6）
- 契約違反: 0件
- 全テスト PASSED

## status-2 TODO 消化状況

- [x] ソルバーの高速化（NumPy ベクトル化）
- [x] 可視化 PostProcess の実装
- [ ] Robin境界条件（対流熱伝達）の追加
- [ ] 2D/3D のより本格的なベンチマーク
- [ ] Phase 2（メッシュ生成・離散化スキーム）への接続

## TODO

- [ ] Robin境界条件（対流熱伝達 h(T-T∞)）の追加
- [ ] 多層シートで層ごとの物性値を簡便に指定するユーティリティ
- [ ] CJK日本語フォント対応（現環境は英語ラベルのみ対応）
- [ ] 解の対称性を活用した全体ドメインミラーリング表示
- [ ] Phase 2（メッシュ生成・離散化スキーム）への接続

## 設計上の懸念

- ヤコビ法はガウスザイデル法より収束が遅い（約2倍の反復数）が、ベクトル化による高速化の恩恵の方が大きい
- 現在の可視化は matplotlib の Agg バックエンドを固定使用。インタラクティブ表示が必要な場合はバックエンド切替が必要
