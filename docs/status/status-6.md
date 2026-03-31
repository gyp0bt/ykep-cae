# status-6: status-5 TODO消化 — 疎行列ソルバー・フィンアレイ2D/3D・CI整備・Phase 2設計

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-03-31

## 概要

status-5 の TODO 4件を消化。SciPy 疎行列ソルバー（直接解法/ILU+BiCGSTAB）導入、冷却フィンアレイ2D/3D拡張テスト、GitHub Actions CI ワークフロー、Phase 2 設計方針をroadmapに追記。

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae_fluid/heat_transfer/solver_sparse.py` | SciPy 疎行列ソルバー新規作成 |
| `xkep_cae_fluid/heat_transfer/solver.py` | `method` パラメータ追加（jacobi/direct/bicgstab） |
| `tests/test_heat_transfer_fdm.py` | 疎行列テスト7件 + フィンアレイテスト3件追加 |
| `.github/workflows/ci.yml` | GitHub Actions CI ワークフロー新規作成 |
| `docs/roadmap.md` | Phase 2 設計方針追記 |
| `README.md` | テスト数・パッケージ構成更新 |

### 機能詳細

#### 1. SciPy 疎行列ソルバー（solver_sparse.py）

- `build_sparse_system()`: FDM離散化の係数行列 A と右辺 b を CSC 疎行列で組み立て
  - 全方向（x±, y±, z±）の内部面・境界面を NumPy ベクトル化で処理
  - Dirichlet / Neumann / Robin / Adiabatic 全境界条件対応
  - 非定常項（時間項）対応
- `solve_sparse_direct()`: SuperLU 直接解法（1ステップで厳密解）
- `solve_sparse_iterative()`: ILU(drop_tol=1e-4) 前処理付き BiCGSTAB 反復法

#### 2. HeatTransferFDMProcess の method パラメータ

- `method="jacobi"`（デフォルト）: 従来のヤコビ/ガウスザイデル反復
- `method="direct"`: SciPy 疎行列直接解法
- `method="bicgstab"`: ILU 前処理付き BiCGSTAB
- 定常・非定常の両方に対応

#### 3. 冷却フィンアレイ 2D/3D 拡張テスト（3件）

- `test_2d_fin_cross_section`: 断面 ny=5 での温度分布（1D解析解との比較、Bi数小条件）
- `test_3d_fin_array_base_heat`: 2本フィン配列の底端熱流束・温度勾配検証
- `test_fin_efficiency_with_mesh_refinement`: メッシュ細分化（nx=10→30）で解析解に収束

#### 4. GitHub Actions CI

- Python 3.10 / 3.11 / 3.12 マトリクスビルド
- ruff check + format check + 契約検証 + テスト実行
- slow/external マーカー付きテストは除外

#### 5. Phase 2 設計方針（roadmap 追記）

- MeshData スキーマ設計（セル中心、面面積、体積、隣接行列）
- StructuredMeshProcess（不等間隔直交格子）
- 非構造化メッシュ読み込み（OpenFOAM互換）
- 離散化スキームは Strategy Pattern で実装

## テスト結果

- テスト数: **59**（既存49 + 疎行列7 + フィンアレイ3）
- 契約違反: **0件**（4プロセス登録）
- 全テスト PASSED

## status-5 TODO 消化状況

- [x] MultilayerBuilder 連携例の収束高速化（前処理付き反復法の検討） → 疎行列ソルバー導入
- [x] 冷却フィンの2D/3D拡張（放熱面積が大きいフィンアレイ） → 断面メッシュ・アレイテスト追加
- [x] CI環境での scipy/numpy/matplotlib 依存管理 → GitHub Actions CI ワークフロー追加
- [x] Phase 2（メッシュ生成・離散化スキーム）への接続 → roadmap に設計方針追記

## TODO

- [ ] Phase 2 MeshData スキーマ設計・実装
- [ ] StructuredMeshProcess 実装（不等間隔直交格子）
- [ ] 既存伝熱ソルバーの MeshData 対応リファクタリング
- [ ] PyAMG マルチグリッド前処理の検討（疎行列ソルバーの更なる高速化）
- [ ] Numba JIT によるガウスザイデル法の高速化

## 設計上の懸念

- 疎行列直接解法は中規模（～10万セル）まで実用的だが、大規模問題には反復法 + マルチグリッドが必要
- BiCGSTAB の ILU 前処理で `drop_tol=1e-4` を使用しているが、問題によっては調整が必要な場合がある
- Phase 2 で MeshData を導入する際、既存の `dx/dy/dz` ベースの離散化との互換性維持が課題

## 開発運用メモ

- 効果的: 疎行列ソルバーにより、異種材料問題でヤコビ法の数千反復が1ステップに短縮
- 効果的: `method` パラメータによる切り替えで、既存テストに影響なく新ソルバーを追加できた
- 効果的: 直接解法をテストに使用することで、テスト実行時間が大幅に短縮（フィンアレイ3件で0.8秒）
