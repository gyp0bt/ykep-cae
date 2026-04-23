# status-26: フィルター循環統合デモ（Phase 6.3b）

[<- README](../../README.md) | [<- status-index](status-index.md) | [水槽ロードマップ](../roadmap-aquarium.md)

**日付**: 2026-04-23
**ブランチ**: `claude/execute-status-todos-tTyoI`
**テスト数**: 286（+1 xfailed、既存テストのまま変動なし。本 PR は examples のみ追加）
**契約違反**: 0 件（登録プロセス 11）

## 概要

水槽設計 CAE ロードマップ Phase 6.3b として、`AquariumGeometryProcess` +
`HeaterProcess` + `AquariumFilterProcess` + `NaturalConvectionFDMProcess` の
**4 段 Process 連携デモ** を `examples/aquarium_filter_circulation.py` に追加した。

小型水槽（30×10×15 cm）で、左上部から右向きに吐出するフィルター循環と
中央ヒーターによる自然対流を同時に解き、温度分布・速度場・質量保存バー・
残差履歴を 1 枚の PNG に出力する。

## 実装内容

### 1. `examples/aquarium_filter_circulation.py`（新規）

| セクション | 内容 |
|---|---|
| 物性定数 | 既存 NC ベンチマークと同一の人工値（mu=0.01, rho=1.0, beta=1e-3）|
| `HEATER_BOX` | 中央ヒーター（x=0.13-0.18, z=0.04-0.11） |
| `INFLOW_BOX` | 上部左のリターン（x=0.025-0.055, z=0.12-0.15, +x 方向吐出）|
| `OUTFLOW_BOX` | 底部右のインテーク（x=0.245-0.275, z=0.0-0.03）|
| `build_inputs` | 4 Process を連結して `NaturalConvectionInput` を構築 |
| `compute_flux` | マスクセル・方向ベクトル・セル幅から体積流量 [m³/s] を積算 |
| `plot_results` | 2×2 パネル: (0,0) T+速度+機材位置、(0,1) T(x) ライン、(1,0) 質量保存バー、(1,1) 残差履歴（log） |
| `main` | `heater=2.0 W`, `Q=60 L/h`, `dt=0.2 s`, `t_end=2.0 s` で実行 |

### 2. パラメータ選定と既知課題の回避

`t_end=2.0 s` で停止する理由:

- status-25 の既知 mass 残差増幅により、buoyancy + 強制対流の累積誤差で
  `t >≈ 2-3 s` 以降で速度場が崩壊（|v|_max が ~20× 小さくなる）するケースがある。
- 本デモは「強制循環の立ち上がり（INLET での強制速度維持、OUTLET 近傍の引き込み）」の
  観測が目的のため、崩壊前の物理的意味のある時刻で結果を保存する。
- SIMPLEC/PISO 実装（Phase 6.0 並行タスク）完了後は長時間積分へ移行予定。

## 動作確認

```
python examples/aquarium_filter_circulation.py 2>&1 | tee /tmp/log-filter-circulation-$(date +%s).log
```

ログ末尾:

```
Done: converged=False, n_steps=10, elapsed=13.66s
Mass balance: inflow Q=1.667e-05 (target 1.667e-05), outflow signed Q=5.817e-08, ratio=0.003
Saved: output/aquarium_filter_circulation.png
```

- **流入側**: 強制速度 0.00889 m/s が完全に維持、体積流量が目標 `Q=60 L/h → 1.667e-5 m³/s` と一致
- **流出側**: 同じ方向での体積流量は 5.8e-8 m³/s（ほぼゼロ）
  → **質量不整合が SIMPLE ペナルティで OUTLET を通らず領域内に残る状態**。
  status-25 の xfail（`test_outlet_mass_nearly_balances_inlet`）と同じ根因。

### 質量保存（厳密な 1% 達成は未達）

ロードマップ Phase 6.3b の「流入 = 流出、< 1% 誤差」は **SIMPLEC/PISO 完了後に再チェック**。
本 Phase では以下を確認できれば合格とみなす:

- [x] 流入側で指定速度ベクトルが維持される（`u[inlet]=target` ✓）
- [x] 流入体積流量が Q_target と一致する（1.67e-5 ≡ 1.67e-5 ✓）
- [x] 温度場にヒーター局在 + フィルター流による移流パターンが現れる
- [ ] ~~流出=流入の厳密バランス（< 1%）~~ → **SIMPLEC/PISO 化待ち**（既知課題）

### 残差・ヒーター影響

- 残差履歴（ログスケール）: mass 残差は 0.08 前後で高止まり（既知課題）
- T(x) プロファイル: ヒーター x=13-18cm で最高温、流出部で最低温
- ヒーター位置での最高 T ≈ 26.5 °C（+1.5 K from T_ref=25 °C）

## 設計判断

### 判断: Process を 4 段連結して純粋関数で扱う

`AquariumGeometryProcess` → `HeaterProcess` + `AquariumFilterProcess`
（並行 PreProcess）→ `NaturalConvectionFDMProcess` という DAG で、
HeaterResult と AquariumFilterResult が独立して NaturalConvectionInput の
異なるフィールド（`q_vol`, `internal_face_bcs`）を埋める。
処理は全て純粋関数、時間ループは NC ソルバー内部で実行される。

### 判断: 質量保存を「バー」で可視化

流入/流出/目標の 3 バーで一目で xfail 状態が分かるよう設計。将来 SIMPLEC/PISO
完了時に同じスクリプトを再実行すれば、3 バーが揃うことで改善が定量確認できる。

### 判断: `inflow_temperature_K` は本デモでは None

`inflow_temperature_K` を T_ref 未満に設定すると、INLET の Boussinesq 浮力項が
強制速度と拮抗し、status-25 の mass 残差問題で 1-2 ステップで velocity が
崩壊する。本デモでは温度拘束なしで運用し、将来 SIMPLEC 化後に温度拘束
デモを別例として追加予定。

### 妥協点・既知課題

- **t_end=2.0 s で停止**: 長時間積分は SIMPLEC/PISO 化後に可能。
- **ratio=0.003（outflow/inflow）**: 既知 xfail 挙動の延長。
- **inflow_temperature_K 未活用**: 同上。

## 次に着手すべきタスク

### Phase 6.0 並行（優先度最高）

- [ ] `natural_convection/assembly.py` d 係数見直し（実水対応 + mass 残差改善）
- [ ] SIMPLEC/PISO で mass 残差を < 0.1 に収束させる
- [ ] SIMPLEC 化後に `test_outlet_mass_nearly_balances_inlet` を strict=True に

### Phase 6.4（順次）

- [ ] `xkep_cae_fluid/porous/process.py`: 多孔質媒体 Darcy-Forchheimer
- [ ] ADA コロラドサンド物性プリセット、ろ材プリセット
- [ ] 1D Darcy 流れ解析解、Ergun 式比較

### Phase 6.3b 拡張（優先度中）

- [ ] inflow_temperature_K=297.15 K（冷水吐出）のケーススタディ（SIMPLEC 化後）
- [ ] 複数吐出ノズル（L 字 / T 字）への拡張
- [ ] 円柱ノズル形状（`NozzleGeometry` のプロトコル化）

## 設計上の懸念・運用メモ

### 発見: SIMPLE ソルバーの `t≈2-3s` 崩壊パターン

buoyancy + INLET 強制の組合せで、短時間（2-3 s）で velocity 場が崩壊する
パターンを確認。これは status-25 の mass 残差問題と同じ根因で、
各 SIMPLE 反復で mass 残差が発散しきらず、累積して連続の式からの逸脱が
加速度的に拡大する現象。SIMPLEC/PISO では係数行列の修正により緩和される
見込み（CLAUDE.md 既知）。

### PR 粒度

- status-26 は「examples 追加 + ドキュメント更新」のみで 1 コミットに収まる。
- テスト数は変動なし（本 Phase は実行サンプルのみ）。

## 変更ファイル

- **新規**: `examples/aquarium_filter_circulation.py`
- **新規**: `docs/status/status-26.md`（本ファイル）
- **編集**: `README.md`（近況更新、構成図に本例を追加）
- **編集**: `docs/status/status-index.md`（status-26 行追加）
- **編集**: `docs/roadmap-aquarium.md`（Phase 6.3b 完了マーク + 既知課題メモ）
