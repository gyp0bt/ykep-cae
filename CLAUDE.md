# xkep-cae-fluid コーディング規約

## 基本

- 全ての回答・設計仕様は**日本語**で記述
- markdown 文書には `README.md` へのバックリンクを貼る
- lint/format: `ruff check xkep_cae_fluid/ tests/` && `ruff format xkep_cae_fluid/ tests/`
- 機能は可能な限りprocessクラスとして実装すること。

## xkep-cae との関係

- **共通基盤**: Process Architecture（AbstractProcess, ProcessMeta, ProcessMetaclass, Registry, StrategySlot 等）は xkep-cae と同一設計
- **ドメイン差異**: xkep-cae は FEM（有限要素法）、本リポジトリは FDM（差分法）/ FVM（有限体積法）
- **Strategy Protocol**: 流体固有のProtocol（ConvectionScheme, TurbulenceModel, PressureVelocityCoupling 等）を定義

## 2交代制運用（Codex / Claude Code）

常に互いへの引き継ぎを想定。statusファイルに状況を詳細記録。

### ステータス管理

- `docs/status/status-{index}.md` に記録（index最大が現在の状況）
- `docs/status/status-index.md` に一覧管理
- status に書いた内容は **commit メッセージと整合**を取る

### 作業完了時の必須手順

1. README.md 更新 → 2. status 新規作成/更新 → 3. status-index.md 更新 → 4. roadmap.md 更新
5. 不整合はその場で修正 or TODO追加 → 6. feature ごとにコミット → push

### ログ出力ルール

- 計算実行は**必ず tee でファイル出力**: `python script.py 2>&1 | tee /tmp/log-$(date +%s).log`
- `| tail -N` のみは禁止（途中経過が残らない）

## テストの分類

### プログラムテスト（API・収束）
- ソルバー収束、例外発生、API仕様準拠
- クラス名: `Test〇〇API`, `Test〇〇Convergence`

### 物理テスト（物理的妥当性）
- 速度場、圧力場、質量保存、運動量保存
- クラス名: `Test〇〇Physics`

## STA2 防止ルール（STAP細胞の二の舞防止）

- **結果の再現性**: 全ての収束結果は tee でログ保存し、YAML 出力と照合可能にすること
- **数値の捏造禁止**: 収束しない場合は「収束しなかった」と報告する
- **ベンチマーク条件の記録**: テスト名、ブランチ名、コミットハッシュ、実行コマンドを記録

### セッション開始時の確認手順
1. `docs/status/status-index.md` → 最新 status 番号を確認
2. 最新 `docs/status/status-{N}.md` を読む
3. `python contracts/validate_process_contracts.py` を実行し、エラー一覧を確認

## フォーカスガード（AI セッション向け）

### やってはいけないこと
- 管理上processクラスとすべきロジックをあえてプライベート関数や迂回ロジックに替えること
- 収束トライ時に目標を緩和して本質的対策を先送りにすること

### 未解決の物理的不整合（status-11 発覚、status-12 で部分修正）

NaturalConvectionFDMProcess の q_vol 付き・空気実物性解析で以下が残存。
**新機能追加よりもこれらの修正を優先すること。**

**status-12 で修正済み:**
- CFL制約付き適応的時間刻み（CFL>>1 による発散を防止）
- エネルギー方程式に Rhie-Chow 面速度を適用（チェッカーボード抑制）
- SIMPLE 収束判定から温度残差を除外（偽の「未収束」判定を修正）

**残存問題（空気実物性 mu=1.85e-5 のみ）:**
1. **mass残差 O(1-100)**: SIMPLE 圧力-速度連成が各ステップ内で収束不十分。
   SIMPLEC or PISO への拡張が必要。
2. **長時間の流速方向不安定化**: 累積誤差により v < 0（下向き）が発生。
   上記 mass 残差の改善で解決する見込み。
3. **温度チェッカーボードの残留**: RC面速度適用で改善したが完全解消はSIMPLE改良後。

**注意**: 人工物性（mu=0.01, k=1.0）では全て正常動作。
差分加熱キャビティ（既存ベンチマーク）も正常。問題は低粘性の実物性限定。
