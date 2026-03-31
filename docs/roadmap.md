# xkep-cae-fluid ロードマップ

[<- README](../README.md)

## 目標

FDM/FVM による非圧縮性 Navier-Stokes ソルバーを Process Architecture 上に構築する。

## Phase 1: 基盤移植（現在）

- [x] Process Architecture 移植（AbstractProcess, Registry, StrategySlot, Diagnostics）
- [x] 流体向け Strategy Protocol 定義（Convection, Diffusion, Turbulence, PV-Coupling）
- [x] 流体向けデータスキーマ（MeshData, FlowFieldData, SolverInputData）
- [x] 契約検証スクリプト（validate_process_contracts.py）
- [x] CLAUDE.md / README.md / pyproject.toml
- [ ] 初期テストスイート

## Phase 2: メッシュ・離散化（予定）

- [ ] 構造化直交メッシュ生成 Process
- [ ] 非構造化メッシュ読み込み Process（OpenFOAM互換）
- [ ] 中心差分拡散スキーム実装
- [ ] 1次風上対流スキーム実装
- [ ] TVD対流スキーム実装（van Leer, Superbee）

## Phase 3: SIMPLE ソルバー（予定）

- [ ] 運動量方程式アセンブリ Process
- [ ] SIMPLE 圧力-速度連成 Process
- [ ] 線形ソルバー Strategy（直接法 + 反復法）
- [ ] Lid-driven cavity ベンチマーク
- [ ] Poiseuille 流れ検証

## Phase 4: 時間進行（予定）

- [ ] 陰的オイラー時間積分
- [ ] BDF2 時間積分
- [ ] 非定常キャビティ流れ検証
- [ ] カルマン渦列（円柱まわり流れ）

## Phase 5: 乱流モデル（予定）

- [ ] 標準 k-epsilon モデル
- [ ] k-omega SST モデル
- [ ] 壁関数
- [ ] 乱流チャネル流れ検証

## 将来構想

- LES / DES
- 多相流（VOF）
- 熱連成
- 適応格子細分化（AMR）

---

## 推奨ソルバー構成（初期）

- **対流スキーム**: TVD (van Leer) — 安定性と精度のバランス
- **拡散スキーム**: 中心差分 + 非直交補正
- **圧力-速度連成**: SIMPLE
- **線形ソルバー**: 圧力=AMG, 速度=BiCGSTAB+ILU
- **時間積分**: 1次陰的オイラー（初期）→ BDF2
