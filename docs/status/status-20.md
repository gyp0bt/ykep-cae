# status-20: 持続的水槽設計 CAE システム Phase 6 ロードマップ策定

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-20
**ブランチ**: `claude/aquarium-cae-roadmap-5yi47`
**テスト数**: 224（ドキュメントのみ、変更なし）
**契約違反**: 0 件（変更なし）

## 概要

90×30×45 cm 水草水槽を題材とする持続的設計 CAE システム（Phase 6）のロードマップを策定。
ユーザー決裁により 4 つの基本設計方針を確定し、Phase 6.0〜6.9 を計 14 PR（+ 並行 0.5〜1 PR）
に分割した詳細計画を `docs/roadmap-aquarium.md` に記述。

## 既存 TODO の凍結

ユーザー指示により、これまでのロードマップ未消化 TODO（Phase 5 乱流、非定常キャビティ、
カルマン渦列、偽時間進行内蔵化、陰的対流スキーム等）は**凍結**し、Phase 6 水槽設計に
注力する。ただし Phase 6.0 として CLAUDE.md 記載の mu=1.85e-5 mass 残差課題は並行別 PR で
対応する。

## ユーザー決裁（2026-04-20）

| 観点 | 決定 |
|------|------|
| 時間スケール | まず定常解、後半で 24h 日周期へ段階拡張 |
| CO2 添加モデル | 気泡プルーム簡易（上昇速度＋界面積→溶存ソース項） |
| mu=1.85e-5 mass 残差課題 | 並行別 PR で対応（水槽は水のため非ブロッカー） |
| 自由表面 | Rigid-lid + Robin BC（ヘンリー則） |

## 対象機材

- 水槽: 90×30×45 cm（W×D×H、Lx=0.9, Ly=0.3, Lz=0.45 m）
- 外部フィルター: エーハイム相当（インレット/アウトレット、吐出量 Q[L/h]）
- CO2 添加: ADA CO2 キット（拡散器配置・添加速度）
- 植物ライト: 上部吊下式 ×2（PAR 出力、配光角）
- ヒーター: 200 W 級、定温制御 or 定熱流束
- 底床: ADA コロラドサンド、深さ 3〜8 cm、K≈1e-9 m², 空隙率 0.35

## Phase 6 全体構成

| Phase | 内容 | PR 数 | 予定 status |
|-------|------|------:|-------------|
| 6.0 | SIMPLEC/PISO mass 残差改善（並行別 PR） | 0.5〜1 | 並行採番 |
| 6.1 | 汎用スカラー輸送 `ScalarTransportProcess` | 2 | 21, 22 |
| **6.2** | **水槽ジオメトリ + ヒーター最小デモ** | **2** | **23, 24** |
| 6.3 | 外部フィルター `InternalFaceBC` | 2 | 25, 26 |
| 6.4 | 多孔質媒体 Darcy-Forchheimer | 2 | 27, 28 |
| 6.5 | 植物ライト Beer-Lambert | 1 | 29 |
| 6.6 | 生体反応（光合成/呼吸） | 2 | 30, 31 |
| 6.7 | ガス交換 + CO2 気泡プルーム | 2 | 32, 33 |
| **6.8** | **水槽システム統合デモ** | **1〜2** | **34** |
| 6.9 | 日周期 + 設計最適化の足場 | 将来 | — |

**最短価値到達**: Phase 6.2b（PR 4 本目、status-24）でヒーター自然対流の可視化デモ。
**統合デモ完成**: Phase 6.8（PR 12 本目、status-34）。

## 再利用する既存資産

- `natural_convection/assembly.py` のエネルギー方程式構造 → スカラー輸送へ汎用化
- `core/strategies/{convection,diffusion,tvd_convection,corrected_diffusion}.py`
- `heat_transfer/` の Robin BC 機構 → scalar_transport に移植
- `NaturalConvectionInput.q_vol` 経路（ヒーター・光合成・CO2 添加で再利用）
- `NaturalConvectionInput.solid_mask` → 砂層・ガラス・ヒーター形状
- `core/mesh.py::StructuredMeshProcess`（不等間隔格子対応済み）
- AMG+CG / BiCGSTAB+ILU 圧力ソルバー、BDF2 時間積分

## 次 PR（Phase 6.1a）の入口条件

1. 本 PR（ロードマップ策定）をメインブランチにマージ
2. `xkep_cae_fluid/scalar_transport/` ディレクトリ作成
3. `ScalarFieldSpec`, `ScalarBoundarySpec`, `ScalarTransportInput`, `ScalarTransportResult` 設計
4. 既存 `natural_convection/assembly.py` からエネルギー方程式アセンブリロジックを共通ヘルパーへ切り出す設計
5. テスト設計: 純拡散解析解（1D 誤差関数解）、1D Taylor-Green 対流保存則

## 変更ファイル

- **新規**: `docs/roadmap-aquarium.md`
- **新規**: `docs/status/status-20.md`（本ファイル）
- **編集**: `docs/roadmap.md`（Phase 6 概要追記）
- **編集**: `docs/README.md`（水槽ロードマップへのリンク追加）
- **編集**: `README.md`（水槽ロードマップへのリンク追加）
- **編集**: `docs/status/status-index.md`（本 status 行追加）

## TODO（次 PR 以降で消化）

### Phase 6.0（並行 PR）
- [ ] `natural_convection/assembly.py` d 係数計算の見直し
- [ ] PISO 第 2 圧力補正の再チェック
- [ ] 空気実物性 mu=1.85e-5、t=10s、mass 残差 < 0.1 を達成

### 物性値・パラメータ調査（Phase 6.6, 6.7 着手前）
- [ ] 気泡プルーム: 気泡径・上昇速度のデフォルト値（文献 Clift-Grace-Weber 相関）
- [ ] 光合成 Michaelis-Menten 定数（代表的水草種プリセット化）
- [ ] ADA コロラドサンド物性（K, 空隙率）の文献値 or メーカー値
- [ ] ヘンリー則定数（CO2, O2、水温依存）

### 将来構想
- [ ] 底床生物膜（硝化菌）モデル — Phase 6.6 追加 or Phase 7
- [ ] 硝酸塩/アンモニア循環（窒素サイクル）— Phase 7
- [ ] 設計最適化目的関数の定式化 — Phase 6.9c
- [ ] 乱流モデル適用検討（多くのケースで層流仮定で足りる見込み）

## 設計上の懸念・運用メモ

- **ドメイン拡張の位置づけ**: `xkep_cae_fluid/aquarium/` は FDM/FVM 基盤に対する
  ドメインアダプタ層として位置付ける。汎用化可能なもの（スカラー輸送、多孔質媒体）は
  `scalar_transport/`, `porous/` など基盤側に置き、水槽固有（ヒーター・ライト・生体反応）
  は `aquarium/` に置く設計。
- **STA2 防止の徹底**: 水槽統合デモは複雑で未収束リスクがあるため、各フェーズで
  単体テスト + 保存則テストを必須とし、統合デモは収束した結果のみ報告する。
- **コンテキスト肥大化対策**: CLAUDE.md 指示のとおり 1 PR = 1 status の粒度を厳守。
  特に Phase 6.1, 6.4, 6.6, 6.7 は単体実装と統合を別 PR に分割済。
- **テスト数推移**: Phase 6 完了時に + 50〜80 件程度を想定（各 Process に API/収束/物理テスト）。

## 運用上の気づき

- ユーザー判断事項を AskUserQuestion で 4 問まとめて確認できたため、プラン精度が向上。
  今後もドメイン拡張時は「時間スケール」「モデル複雑度」「既知課題の処理」「境界条件近似」の
  4 軸を早期確定する運用を推奨。
- 既存資産の再利用マッピング（scalar_transport ← natural_convection のエネルギー方程式等）を
  ロードマップに明記したことで、実装時の迷いが減る見込み。
