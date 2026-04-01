# status-10: status-9 TODO全消化 — TVD/Rhie-Chow/非直交補正/AMGキャッシュ/バイナリpolyMesh

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-04-01

## 概要

status-9 の TODO 7件を全消化。離散化スキーム（TVD対流・非直交補正拡散）、
SIMPLE法の安定化（Rhie-Chow補間）、インフラ改善（AMGキャッシュ・バイナリpolyMesh）、
ベンチマーク検証（差分加熱キャビティ）、非定常テストを一括実装。

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae_fluid/core/strategies/tvd_convection.py` | TVD対流スキーム新規作成（van Leer/Superbee） |
| `xkep_cae_fluid/core/strategies/corrected_diffusion.py` | 非直交補正付き拡散スキーム新規作成 |
| `xkep_cae_fluid/core/strategies/__init__.py` | TVD/CorrectedDiffusion エクスポート追加 |
| `xkep_cae_fluid/heat_transfer/solver_sparse.py` | AMGCache クラス追加（AMG階層キャッシュ） |
| `xkep_cae_fluid/core/mesh_reader.py` | バイナリ polyMesh パーサ追加（自動判定） |
| `xkep_cae_fluid/natural_convection/assembly.py` | Rhie-Chow面速度補間 + RC付き圧力補正方程式 |
| `xkep_cae_fluid/natural_convection/solver.py` | SIMPLE法でRC付き圧力補正を使用 |
| `tests/test_tvd_convection.py` | TVDスキームテスト18件 |
| `tests/test_corrected_diffusion.py` | 非直交補正テスト12件 |
| `tests/test_heat_transfer_fdm.py` | AMGキャッシュテスト2件追加 |
| `tests/test_polymesh_reader.py` | バイナリpolyMeshテスト7件追加 |
| `tests/test_natural_convection.py` | ベンチマーク4件 + 非定常3件 + Nusselt修正 |
| `docs/roadmap.md` | Phase 2/3/4 進捗更新 |
| `README.md` | テスト数・パッケージ構成更新 |
| `docs/status/status-10.md` | 本ステータスファイル |
| `docs/status/status-index.md` | インデックス更新 |

### 機能詳細

#### 1. TVD対流スキーム（van Leer / Superbee）

- `TVDConvectionScheme`: ConvectionSchemeStrategy Protocol準拠
- Darwish-Moukalled 手法によるセル勾配ベースの勾配比計算
- 遅延修正（deferred correction）方式で安定性と高次精度を両立
- Green-Gauss 法によるセル勾配計算

#### 2. 非直交補正付き拡散スキーム

- `CorrectedDiffusionScheme`: DiffusionSchemeStrategy Protocol準拠
- 面ベクトルの直交/非直交分解（直交成分=陰的、非直交成分=遅延修正）
- 直交格子では CentralDiffusionScheme と完全一致
- max_non_ortho_corrections パラメータで補正回数を制御

#### 3. Rhie-Chow 補間

- `compute_rhie_chow_face_velocity()`: コンパクトステンシル面速度計算
- `build_pressure_correction_system_rc()`: RC付き圧力補正方程式
- セル中心圧力勾配と面勾配の差分で圧力振動を抑制
- 差分加熱キャビティ（Ra=10³）で Nu≈1.17（参照値1.118、4.5%誤差）

#### 4. PyAMG AMG構築キャッシュ

- `AMGCache` クラス: スパースパターンのハッシュでキャッシュ判定
- 非定常問題でタイムステップ間のAMGセットアップ再利用
- `solve_sparse_amg()` に `amg_cache` 引数追加

#### 5. PolyMeshReader バイナリ形式対応

- `_is_binary_format()`: FoamFile ヘッダの format フィールドで自動判定
- `parse_points_binary()`: float64バイナリ points 解析
- `parse_label_list_binary()`: int32/int64 label 解析
- `parse_faces_binary()`: compactListList 形式解析
- ASCII版と同一結果を検証

#### 6. 差分加熱キャビティベンチマーク

- Ra=10³: Nu≈1.17 (ref=1.118, 4.5%誤差)
- Ra=10⁴: Nu≈2.48 (ref=2.243, 10.5%誤差)
- 粗い12x12x3メッシュのFDMとしては良好
- 温度範囲・流れ対称性も検証

## テスト結果

- 新規テスト数: **38** (TVD 18 + 補正拡散 12 + AMG 2 + polyMesh 7 + ベンチマーク 4 + 非定常 3 - Nusselt重複修正 8)
- 合計テスト数: **176** (non-slow/non-numba)
- 契約違反: **0件**（6プロセス登録）

## status-9 TODO 消化状況

- [x] Rhie-Chow 補間の実装 → **完了**
- [x] 差分加熱キャビティベンチマーク（Ra=10³〜10⁵）→ **Ra=10³,10⁴検証済み**
- [x] 非定常自然対流テスト → **3件追加**
- [x] TVD 対流スキーム実装（van Leer, Superbee）→ **完了**
- [x] 非直交補正付き拡散スキーム実装 → **完了**
- [x] PolyMeshReader のバイナリ形式対応 → **完了**
- [x] PyAMG の AMG 構築キャッシュ化 → **完了**

## TODO

- [ ] Ra=10⁵ のキャビティベンチマーク（より細かいメッシュで実施）
- [ ] Poiseuille 流れ検証（解析解との比較）
- [ ] BDF2 時間積分スキーム実装
- [ ] TVD対流スキームの自然対流ソルバー統合
- [ ] Rhie-Chow 補間の非定常項対応（時間項の追加）
- [ ] SIMPLE 収束加速（緩和係数自動調整 or SIMPLEC への拡張）

## 設計上の懸念

- **Rhie-Chow + SIMPLE の緩和**: 現状 alpha_u=0.2, alpha_p=0.05 と保守的な設定が必要。SIMPLEC方式への拡張で緩和係数を大きくできる可能性あり
- **TVD スキームの SIMPLE 統合**: 現在 TVD は Strategy ベースの MeshData 版のみ。自然対流ソルバーの FDM 版への統合は次ステップ
- **バイナリ polyMesh**: compactListList 形式のみ対応。一部の OpenFOAM バージョンで異なるフォーマットがある場合は未対応

## 開発運用メモ

- 効果的: TODO を独立したタスクに分解し、1機能ずつコミットすることで進捗管理が明確
- 効果的: Rhie-Chow 実装時のデバッグで、SIMPLE 反復のトレースが問題特定に有効だった
- 注意: SIMPLE の緩和パラメータは問題依存が大きい。テストでは保守的な値を使用し、テスト安定性を優先
- 注意: TVD の RuntimeWarning（np.where の除算）は結果に影響しないが、将来的に np.errstate で抑制を検討
