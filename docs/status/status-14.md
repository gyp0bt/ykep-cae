# status-14: PISO連成 + TVD対流スキーム統合 + 対流流出BC

[<- status-index](status-index.md) | [<- README](../../README.md)

## 日付

2026-04-02

## 概要

status-13 TODO のうち最優先項目（PISO、TVD統合、非反射BC）を実装。
自然対流ソルバーの圧力-速度連成・対流離散化・出口境界を大幅に強化。

## 実装内容

### 1. PISO 圧力-速度連成 (Issa, 1986)

| ファイル | 変更内容 |
|---------|---------|
| `data.py` | `coupling_method="piso"`, `n_piso_correctors` フィールド追加 |
| `solver.py` | PISO分岐（追加圧力補正ループ、速度緩和不要） |

**アルゴリズム:**
1. 運動量予測子 → u* （SIMPLE と同じ）
2. 第1圧力補正 → p', u** の補正
3. 第2圧力補正 → p'', u*** の補正（補正済み速度の質量残差を再計算）
4. (オプション) 第3以降の補正

**SIMPLE との違い:**
- 速度の under-relaxation が不要（alpha_u=1.0 相当）
- 複数回の圧力補正で1反復あたりの質量保存を大幅改善
- 非定常解析で特に有効

### 2. TVD 対流スキーム統合

| ファイル | 変更内容 |
|---------|---------|
| `data.py` | `convection_scheme` フィールド追加 ("upwind"/"van_leer"/"superbee") |
| `assembly.py` | `_tvd_deferred_correction()` + リミッター関数追加 |

**遅延補正法 (Deferred Correction):**
- 行列: 1次風上差分を保持（対角優位性・安定性を確保）
- RHS: TVD補正をソース項として追加
- `Δ = F × 0.5 × ψ(r) × (φ_D - φ_U)`
- 勾配比 `r` は構造格子上のUU（upstream-upstream）セルから計算
- 境界面（UUセルなし）では ψ=0（1次風上にフォールバック）

**リミッター:**
- van Leer: `ψ(r) = (r + |r|) / (1 + |r|)` — 滑らかで安定
- Superbee: `ψ(r) = max(0, min(2r,1), min(r,2))` — 高精度だが振動リスク

### 3. 対流流出境界条件 (OUTLET_CONVECTIVE)

| ファイル | 変更内容 |
|---------|---------|
| `data.py` | `FluidBoundaryCondition.OUTLET_CONVECTIVE` 追加 |
| `assembly.py` | 運動量方程式の対流流出処理 |

**従来の OUTLET_PRESSURE との違い:**
- OUTLET_PRESSURE: ゼロ勾配（何もしない）→ 流出フラックスが未処理
- OUTLET_CONVECTIVE: 境界面の流出対流フラックスを陽的に処理
  `F_out = ρ × max(v_n, 0) / d` → 対角に追加

チャネル流等の入出口問題での安定性が向上。

## テスト結果

| テストクラス | 件数 | 結果 |
|------------|------|------|
| TestPISOAPI | 3 | PASS |
| TestPISOPhysics | 3 | PASS |
| TestTVDConvectionAPI | 3 | PASS |
| TestTVDConvectionPhysics | 2 | PASS |
| TestOutletConvectiveAPI | 2 | PASS |
| TestOutletConvectivePhysics | 2 | PASS |
| 既存テスト（Ra=1e4除く） | 37 | PASS |
| **合計** | **52** | **PASS** |

## 残存問題

### 空気実物性 (mu=1.85e-5)

PISO + TVD + 対流流出BC の組み合わせにより、以下の改善が期待される：
- PISO: 各ステップ内の mass 残差を大幅低減
- TVD: 数値拡散を低減しつつ安定性を維持
- 対流流出BC: 出口での流出フラックスの適切な処理

ただし空気実物性のような極端な低粘性レジームでの検証は未実施。
次のステップとして空気実物性パラメトリックスタディが必要。

### Ra=1e4 ベンチマーク

Nu=5.198（ref=2.243, 132%誤差）は未修正。
TVD スキーム適用でも改善されない場合、メッシュ細分化が必要。

## TODO

- [ ] 空気実物性(mu=1.85e-5) + PISO + TVD での検証テスト
- [ ] Ra=1e4 ベンチマーク修正（TVD + メッシュ細分化）
- [ ] 非定常キャビティ流れ検証
- [ ] カルマン渦列（円柱まわり流れ）
- [ ] k-epsilon 乱流モデル

## 開発運用メモ

- PISO は SIMPLE の `_simple_iteration` 内に追加圧力補正ループとして実装。
  SIMPLE/SIMPLEC/PISO が同一関数内で切り替え可能な設計。
- TVD 遅延補正は assembly 内の独立関数として実装。運動量・エネルギーで共用。
  行列変更なし（1次風上を保持）なので既存の収束性に影響を与えない。
- OUTLET_CONVECTIVE は assembly の境界ループ内に自然に追加。
  OUTLET_PRESSURE と同じ場所に条件分岐で追加。
- 3機能を独立にコミットし、各機能ごとにテストを通過させた。
