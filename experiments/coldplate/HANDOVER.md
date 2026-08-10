# 引き継ぎ: 冷却流路 概念設計ソルバ実験 (coldplate)

[README.md](README.md) に戻る

**作成日**: 2026-08-09 / **ブランチ**: `claude/coldplate-handover-1to5` /
**前セッションの HANDOVER 候補 1-5 を全て消化済み** (PR #25 は master へマージ済み)

リポジトリ本体 (Process Architecture) とは独立した実験コード。CI 対象外、
status-index には登録しない方針 (本ファイルが引き継ぎの正)。追加依存: torch。

## 1. 現在地 — モデルの系譜

| 世代 | ファイル/フラグ | 物理 | 状態 |
|------|---------------|------|------|
| v2 ネットワーク | `coldplate.py` | Hagen-Poiseuille 格子回路 + 鮮度/熱 | 完成・凍結 |
| 擬3D ダルシー (logK) | `darcy.py` (既定) | Hele-Shaw + 対数補間 K(s) | 完成・凍結 (回帰テストで保護) |
| **ピンフィン** | `darcy.py` `pin_fin=True` | K(φ)=平板+Gebart 直列, U''(φ)=プライム+フィン増倍 | **本線** (Task 1) |
| **+Forchheimer** | `forchheimer=True` | + Ergun 慣性項, 緩和付き Picard | **本線** (Task 2) |

追加機能: ポート位置設計 `optimize_ports` (Task 3)、任意設計粒度
`optimize(design_shape=)` (Task 4)、ウォームスタート `optimize(xi0=)` (Task 5)。
テスト 48 件 (darcy 28 + coldplate 20) 全合格。詳細な結果表は README 参照。

## 2. 今セッションで確定した知見 (機構ベース)

1. **旧 logK モデルの層間結合は一桁楽観だった**: 線形混合 k_c(s) を層厚伝導に
   使うと少量の固体でも「流体へ直結する垂直伝導スラブ」になる (s=0.17 で
   U''≈22,700 W/m²K)。ピンフィン版はピン側面→流体の対流段 (フィン抵抗) を
   挟んで U''≈2,300 と正直になり、同一場の温度は保守側へ動く。
2. **設計の機構が変わった**: 旧=固体は流れを導く壁 (solidity≤0.25) →
   ピン版=発熱体直上に伝熱面として敷く (γ=0.01 で 0.66)。
3. **慣性損失は無視できない** (Re_gap≈780): ΔP はダルシー比 2.0-2.6 倍。
   D-F 物理で直接最適化すると同等 T_peak で ΔP 半減 (601→312 Pa)。
   モチーフは局所高速を避けるサーペンタイン状へ。
4. **対角カウンターフロー選好は連続場では再現しない** (Task 3): 対角初期値
   からでも中央へ自力回帰。ネットワーク版の対角選好は境界一周の経路長合わせ
   だったが、連続場は設計場自身が分配を担うため中央給水が勝つ。ポート
   自由化は利得ゼロ、共進化でむしろ 3-8% 劣化。
5. **粒度仮説「セル単位で枝的」は形態的に正しいが病的ではない** (Task 4):
   2.5mm 以下で開水路×ピン壁ラメラが出現、J は単調改善 (11%)。ただし縞幅が
   t_chan=2mm を下回ると深さ平均近似の妥当性外 → 概念設計は 5mm 採用。
6. **パレート前線は滑らか** (Task 5, 15 点): T_peak 23.9→9.0 K / ΔP 17→222 Pa
   (ダルシー)。warm start はコールドと J±1% 一致。膝は γ≈1-3。

## 3. ファイル構成 (今セッション追加分)

```
darcy.py            pin_fin / forchheimer / ports / design_shape / xi0 を追加
run_pinfin.py       ピンフィン γ スイープ → coldplate_pinfin.{png,npz}
run_forchheimer.py  Part A: 同一設計 D vs D-F / Part B: D-F 再最適化 → coldplate_forchheimer.*
run_ports.py        fixed-center / free-center / free-diag × γ{1,0.1} → coldplate_ports.*
run_granularity.py  10/5/2.5/1.25mm 粒度比較 (margin_right=4 形状) → coldplate_granularity.*
run_sweep.py        γ 15 点継続法 + D-F 再評価 → coldplate_sweep.*
test_darcy.py       28 件 (旧 8 + ピン/Forchheimer 12 + 粒度 4 + ポート 4)
logs/log-{pinfin,forchheimer,granularity,ports,sweep}-*.log  (全実行 tee 済み)
```

## 4. 検証手順 (環境構築後に必ず)

```bash
cd experiments/coldplate
# venv: リポジトリ直下 .venv (uv venv + torch CPU + numpy scipy matplotlib pytest ruff)
OMP_NUM_THREADS=4 ../../.venv/bin/python -m pytest test_darcy.py test_coldplate.py -q  # 48 passed (~1分)
../../.venv/bin/ruff check *.py && ../../.venv/bin/ruff format --check *.py
```

**罠**: OMP_NUM_THREADS を絞らないと 20 コア機で torch のスレッドスピンにより
小規模 LU が 20-30 倍遅くなる (テスト 127s vs 4s を実測)。

## 5. 失敗記録 (再発防止, 今セッション分)

- **Forchheimer の裸の固定点反復は減衰振動して収束しない** (40 反復で未達):
  線形化フラックスに緩和 ω=0.5 を入れて解消。返す流束は最終線形解なので
  質量保存は厳密のまま
- **「同一場でピン版が旧モデルより冷える」という当初のテスト仮定は逆**:
  ピン版は非物理な熱ショートカットを除去する分、温度は上がる (保守側)。
  test_pin_more_conservative_than_legacy に狙いを文書化済み
- 旧セッション分 (0/1 位相化の 2 度の失敗、シグモイド飽和) は README 参照

## 6. 次の候補 (未着手、優先順は次セッションの裁量)

1. **h の流速依存性**: 現状 Nu_pin=4 固定 (伝導床)。ピン周り対流の
   Re 依存 (Zukauskas 等) を入れると「速い流れ×高密度ピン」の相乗が
   モデルに入り、ラメラ構造の評価がさらに正直になる
2. **3D 検証**: 5mm ブロック採用の根拠 (深さ平均の妥当性限界) を、代表 2-3
   設計の 3D 共役伝熱 (OpenFOAM) で確認。特にラメラ解の実力チェック
3. **製造制約**: ピン径 d と φ の離散カタログ化 (実在フィンピッチへの丸め)、
   丸め後の性能劣化評価
4. **ε 制約法**: 重み付き和は前線の非凸部を拾えない。ΔP 上限制約での
   直接解と比較して前線の凸性を確認
5. ネットワーク版の残課題 (P3 校正、P5 ヘッダ) は据え置き

## 7. 運用ルール (このレポの掟)

- 回答・文書は日本語。計算実行は必ず tee でログ保存 (`| tail` のみ禁止)
- 収束しない/失敗した実験もログを残して正直に報告 (STA2 防止ルール)
- 再現条件 (ブランチ・コミット・コマンド・seed) を結果と一緒に記録
- feature ごとにコミットし、PR の本文を更新
- 本実験は Process クラス統合・status-index 登録の対象外 (独立実験)
