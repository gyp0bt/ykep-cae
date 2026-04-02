# ステータスインデックス

[<- README](../../README.md)

| # | 日付 | テスト数 | 概要 |
|---|------|---------|------|
| 1 | 2026-03-31 | 16 | 初期移植: xkep-cae から Process Architecture 移植 |
| 2 | 2026-03-31 | 25 | 3次元非定常伝熱解析 (FDM) HeatTransferFDMProcess 実装 |
| 3 | 2026-03-31 | 31 | ソルバー高速化 + 可視化PostProcess + 多層シート温度マップ |
| 4 | 2026-03-31 | 39 | Robin BC + 多層ビルダー + CJK対応 + ミラーリング表示 |
| 5 | 2026-03-31 | 49 | status-4 TODO消化: 非定常Robin BC・冷却フィン・連携例・scipy復旧 |
| 6 | 2026-03-31 | 59 | status-5 TODO消化: 疎行列ソルバー・フィンアレイ2D/3D・CI整備・Phase 2設計 |
| 7 | 2026-03-31 | 88 | status-6 TODO消化: StructuredMeshProcess + PyAMG + Numba JIT |
| 8 | 2026-03-31 | 124 | status-7 TODO全消化: 離散化スキーム + MeshData対応 + polyMesh読込 |
| 9 | 2026-04-01 | 138 | 3D自然対流ソルバー (SIMPLE法 + Boussinesq) + 固体-流体練成 |
| 10 | 2026-04-01 | 176 | status-9 TODO全消化: TVD/Rhie-Chow/非直交補正/AMGキャッシュ/バイナリpolyMesh |
| 11 | 2026-04-01 | 180 | 自然対流調査: q_vol追加 + パラメトリックスタディ(24ケース) + 設計指針 |
| 12 | 2026-04-01 | 180 | ソ��バー安定性改善: adaptive dt + RC energy + 収束判定修正 |
| 13 | 2026-04-02 | 197 | SIMPLEC連成 + BDF2���間積分 + Poiseuille検証 + 保守的緩和テスト |
| 14 | 2026-04-02 | 214 | PISO連成 + TVD対流スキーム統合 + 対流流出BC |
| 15 | 2026-04-02 | 214 | 空気実物性 収束評価 + PISO速度緩和修正 |
| 16 | 2026-04-02 | 221 | AMG圧力ソルバー + 面ベース質量残差修正 + 適応的緩和 |
