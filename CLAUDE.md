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

### 未解決の物理的不整合（status-11 調査で発覚）

以下の問題が NaturalConvectionFDMProcess の q_vol 付き解析で確認されている。
**新機能追加よりもこれらの修正を優先すること。**

1. **チェッカーボード温度パターン**: 全条件で温度場にチェッカーボード状の振動が発生。
   Rhie-Chow 補間がエネルギー方程式に適用されていない（運動量方程式のみ）可能性。
   または OUTLET_PRESSURE 境界での温度処理が不適切。
2. **高温部が発熱体位置と不一致**: q_vol を中心に設定しているにもかかわらず、
   高温域が発熱体外の非物理的な位置に出現する。エネルギー方程式の対流項の
   風上差分方向、または境界条件の温度固定が正しく機能していない疑い。
3. **流速方向の逆転**: 発熱体付近で浮力により +y（上向き）の流速が期待されるが、
   -y（下向き）の流速が観測される。浮力項 `-ρβ(T-T_ref)g` の符号、
   圧力勾配の離散化、または SIMPLE 速度補正のいずれかにバグの可能性。

これらは互いに関連している可能性が高い（例: 温度場の誤りが浮力項を通じて
速度方向を逆転させる）。修正時は以下の手順を推奨:
- まず差分加熱キャビティ（既存ベンチマーク、q_vol不使用）が正常に動くことを再確認
- q_vol のみの純粋伝導問題（beta=0, 重力なし）で温度場が正しいことを確認
- 1方向のみの浮力問題（1D的）で流速方向が正しいことを確認
- 段階的に複雑さを増やして原因を特定
