# status-13: SIMPLEC連成 + BDF2時間積分 + Poiseuille検証 + 保守的緩和テスト

[<- status-index](status-index.md) | [<- README](../../README.md)

## 日付

2026-04-02

## 概要

status-12 の TODO を全消化。SIMPLEC 圧力-速度連成手法、BDF2 時間積分、
Poiseuille 流れ検証テスト、保守的緩和係数の安定性テストを実装。

## 実装内容

### 1. SIMPLEC 圧力-速度連成 (Van Doormaal-Raithby, 1984)

| ファイル | 変更内容 |
|---------|---------|
| `data.py` | `coupling_method` フィールド追加 ("simple" / "simplec") |
| `solver.py` | SIMPLEC 分岐: 行列行和からd係数計算、alpha_p=1.0 自動設定 |

**技術詳細:**
- SIMPLE: `d = rho / a_P`（対角係数のみ）
- SIMPLEC: `d = rho / row_sum(A)`（行列行和 = a_P - Σ|a_nb|）
- 行列行和は `A.sum(axis=1)` で取得、assembly 変更不要
- alpha_p=1.0 を自動適用（SIMPLEC は圧力補正が正確）
- 速度緩和 alpha_u はユーザ指定のまま

### 2. BDF2 時間積分

| ファイル | 変更内容 |
|---------|---------|
| `data.py` | `time_scheme` フィールド追加 ("euler" / "bdf2") |
| `assembly.py` | 運動量・エネルギー方程式のBDF2時間項離散化 |
| `solver.py` | old_old 値の管理とBDF2への伝搬 |

**離散化:**
- Euler後退: `(φ^{n+1} - φ^n) / dt` → diag += ρ/dt, rhs += ρ/dt * φ^n
- BDF2: `(3φ^{n+1} - 4φ^n + φ^{n-1}) / (2dt)` → diag += 3ρ/(2dt), rhs += 2ρ/dt * φ^n - ρ/(2dt) * φ^{n-1}
- 最初のステップは φ^{n-1} がないため自動的に Euler にフォールバック

### 3. Poiseuille 流れ検証テスト

INLET_VELOCITY / OUTLET_PRESSURE BC を使ったチャネル流の物理テスト:
- 出口付近の放物線速度プロファイル（壁面<中心、最大値が中心付近）
- 横方向速度がほぼゼロ
- 温度場が一様（断熱壁、浮力なし）

**注意:** 高粘性(mu=0.1)・低Re条件でのみ安定。mu=0.01 では発散が発生。
入出口 BC での SIMPLE 収束は壁面 BC のみの場合より困難。

### 4. 保守的緩和係数テスト

- SIMPLE + 保守的緩和 (alpha_u=0.1, alpha_p=0.03) で低粘性(mu=1e-3)の安定性確認
- SIMPLEC + 保守的緩和 (alpha_u=0.3, alpha_p=0.1) で非定常解析の安定性確認

## テスト結果

| テストクラス | 件数 | 結果 |
|------------|------|------|
| TestSIMPLECAPI | 3 | PASS |
| TestSIMPLECPhysics | 2 | PASS |
| TestPoiseuillePhysics | 3 | PASS |
| TestBDF2API | 2 | PASS |
| TestBDF2Physics | 2 | PASS |
| TestConservativeRelaxation | 2 | PASS |
| 既存テスト (Ra=1e4除く) | 21 | PASS |
| **合計** | **35** | **PASS** |

※ Ra=1e4 の Nusselt 数テストは既存の問題（status-12 以前から）で FAIL。今回の変更と無関係。

## 残存問題

### 空気実物性 (mu=1.85e-5) での mass 残差

status-12 の残存問題は引き続き存在:
- SIMPLEC は d係数を増大させるが、空気の極端な低粘性レジームでは
  行列行和が非常に小さく（対角優位が弱い）、効果が限定的
- PISO（2段圧力補正）への拡張が次の改善候補

### Poiseuille 流（入出口BC）の制限

- mu=0.01 (Re_cell ≈ 1) で SIMPLE が発散
- 現行の出口BC（ゼロ勾配）は完全に非反射ではなく、圧力波が反射する
- 特性ベース非反射BC の実装で改善する見込み

### Ra=1e4 ベンチマークの不一致

- Nu=5.198 (ref=2.243, err=132%)
- 粗いメッシュ + 1次風上差分による数値拡散の可能性
- TVD スキームの自然対流ソルバーへの統合で改善する可能性

## TODO

- [ ] 特性ベース非反射境界条件（OUTLET_PRESSURE 改良）
- [ ] 非定常キャビティ流れ検証
- [ ] PISO（2段圧力補正）の検討
- [ ] TVD 対流スキームの自然対流ソルバーへの統合
- [ ] Ra=1e4 ベンチマーク修正（メッシュ細分化 or TVD スキーム適用）

## ユーザーからの助言

- Ansys Fluent では温度の二次微分を無視するオプションで収束を改善する手法がある
  → エネルギー方程式の対流項の高次差分に関連する可能性。要調査。

## 開発運用メモ

- SIMPLEC の実装は「行列行和をd係数として使う」という手法で assembly 変更不要だった
  → API 変更を最小化する設計の好例
- BDF2 の初回 Euler フォールバックは vel_old_old=None チェックで自然に実現
- Poiseuille 流テストでは mu=0.01 → mu=0.1 に変更して安定化した。入出口 BC が
  SIMPLE の安定性に大きく影響することを確認
