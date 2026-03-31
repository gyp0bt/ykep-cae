# status-7: status-6 TODO消化 — StructuredMeshProcess + PyAMG + Numba JIT

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-03-31

## 概要

status-6 の TODO 5件のうち3件を完全消化、2件を次フェーズへ繰越。
StructuredMeshProcess（不等間隔直交格子生成）、PyAMG マルチグリッド前処理、
Numba JIT ガウスザイデル法を実装。

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae_fluid/core/mesh.py` | StructuredMeshProcess + StructuredMeshInput/Result 新規作成 |
| `xkep_cae_fluid/core/__init__.py` | StructuredMesh エクスポート追加 |
| `xkep_cae_fluid/heat_transfer/solver_sparse.py` | `solve_sparse_amg()` 追加（PyAMG前処理付きCG） |
| `xkep_cae_fluid/heat_transfer/solver_numba.py` | Numba JIT版ガウスザイデル法 新規作成 |
| `xkep_cae_fluid/heat_transfer/solver.py` | `method="amg"/"numba"` 追加、`_iterate_step()` リファクタ |
| `tests/test_structured_mesh.py` | メッシュテスト21件 新規作成 |
| `tests/test_heat_transfer_fdm.py` | AMGテスト4件 + Numbaテスト4件 追加 |
| `docs/design/structured-mesh.md` | StructuredMeshProcess 設計文書 新規作成 |
| `docs/roadmap.md` | Phase 2 タスク更新 |
| `README.md` | テスト数・パッケージ構成更新 |

### 機能詳細

#### 1. StructuredMeshProcess（core/mesh.py）

- **StructuredMeshInput**: Lx/Ly/Lz + nx/ny/nz + ストレッチング指定
- **ストレッチング方式**:
  - 等間隔: `(1.0,)`
  - 幾何級数: `(ratio, grading)` — 一方向/逆方向/両端集中
  - 直接比率: `(r1, r2, ..., rn)` — 各セルの幅比率
- **StructuredMeshResult**: MeshData + dx/dy/dz 配列
- **MeshData生成内容**:
  - ノード座標 (n_nodes, 3)
  - セル接続 (n_cells, 8) — 六面体
  - セル体積、セル中心
  - 内部面: 面積、法線、中心、owner/neighbour

#### 2. PyAMG マルチグリッド前処理（solver_sparse.py）

- `solve_sparse_amg()`: Ruge-Stüben AMG + CG
- 伝熱方程式の係数行列は SPD なので CG が使用可能
- `method="amg"` で定常・非定常に対応
- 大規模問題で ILU+BiCGSTAB より高速な収束が期待できる

#### 3. Numba JIT ガウスザイデル法（solver_numba.py）

- `_gauss_seidel_step_numba()`: @njit(cache=True) で3重ループをコンパイル
- BoundaryCondition enum を整数値に変換して Numba 互換にパック
- `method="numba"` で既存の Python GS と切り替え可能
- 初回コンパイル後はキャッシュにより2回目以降は即座に実行

#### 4. HeatTransferFDMProcess のリファクタリング

- `_iterate_step()` ヘルパーで反復ディスパッチを統一
- method選択肢: `"jacobi"`, `"direct"`, `"bicgstab"`, `"amg"`, `"numba"`

## テスト結果

- テスト数: **88**（既存59 + メッシュ21 + AMG 4 + Numba 4）
- 契約違反: **0件**（5プロセス登録）
- 全テスト PASSED

## status-6 TODO 消化状況

- [x] Phase 2 MeshData スキーマ設計・実装 → 既存 core/data.py の MeshData を活用、StructuredMeshProcess で面情報を完全生成
- [x] StructuredMeshProcess 実装（不等間隔直交格子） → core/mesh.py
- [ ] 既存伝熱ソルバーの MeshData 対応リファクタリング → 次フェーズ（Phase 2 完了時にメッシュ依存部分をリファクタ）
- [x] PyAMG マルチグリッド前処理の検討 → solve_sparse_amg() として実装
- [x] Numba JIT によるガウスザイデル法の高速化 → solver_numba.py として実装

## TODO

- [ ] 既存伝熱ソルバーの MeshData 対応リファクタリング（HeatTransferInput.dx/dy/dz → MeshData 経由）
- [ ] 非構造化メッシュ読み込み Process（OpenFOAM polyMesh 互換）
- [ ] 中心差分拡散スキーム実装（MeshData ベース）
- [ ] 1次風上対流スキーム実装
- [ ] Numba JIT の性能ベンチマーク（Python GS / Vectorized Jacobi / Numba GS 比較）
- [ ] CI に pyamg/numba のオプション依存テストを追加

## 設計上の懸念

- StructuredMeshProcess は現在内部面のみ生成。境界面情報は伝熱ソルバーの MeshData 対応時に追加が必要
- PyAMG の AMG構築コストは1回限りだが、非定常解析で毎ステップ構築している。時間項のみ変化する場合は再利用可能
- Numba の初回コンパイルに 2-3 秒かかる。テスト実行時間への影響を監視

## 開発運用メモ

- 効果的: pyproject.toml に `amg`/`fast` オプション依存が事前定義済みだったため、即座に実装に着手できた
- 効果的: `_iterate_step()` による反復ディスパッチの統一で、新しい method 追加が容易になった
- 注意: Numba は BoundaryCondition の Enum を直接扱えないため、整数値へのパック変換が必要
