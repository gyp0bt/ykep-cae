# 引き継ぎ: 冷却流路 概念設計ソルバ実験 (coldplate)

[README.md](README.md) に戻る

**作成日**: 2026-08-08 / **ブランチ**: `claude/cooling-channel-topology-solver-1qchon` /
**PR**: [#25](https://github.com/gyp0bt/ykep-cae/pull/25) (open) /
**最新コミット**: 5d66f00 (差分可視化) ← 640e986 (darcy.py) ← 64189bb (thermal 実行結果)

リポジトリ本体 (Process Architecture) とは独立した実験コード。CI 対象外、
status-index には登録しない方針 (本ファイルが引き継ぎの正)。追加依存: torch。

## 1. 現在地 — 3 世代のモデルがある

| 世代 | ファイル | 物理 | 設計変数 | 状態 |
|------|---------|------|---------|------|
| v2 ネットワーク (φ) | `coldplate.py` (mode="fresh") | Hagen-Poiseuille 格子回路 + 鮮度スカラー | 辺 rho/幅 + ポート | 完成・知見出し済み |
| v2 ネットワーク (熱) | `coldplate.py` (mode="thermal") | + 固体 2 層熱回路、h=層流一定 Nu | 同上 | 完成。ただし h が流速非依存 → 「濡れ面積が唯一のレバー」になる限界を確認 |
| 擬3D ダルシー | `darcy.py` | Hele-Shaw ダルシー流 + z 方向 2 層熱 (Nu 相関なし) | **5mm ブロックごとの連続透過率** | 完成・最新。ここを拡張するのが本線 |

直近の流れ: ネットワーク+一定 Nu では ΔT 削減の機構が「総流路長 (濡れ面積)」
だけに縮退することが判明 → ユーザー指示で近似を捨て、ダルシー係数場の
2 目的最適化 (圧損 × 温度均一) に移行した。

## 2. darcy.py の要点 (最新モデル)

- 幾何: 2×4 ヒーターアレイ (coldplate と同一配置)、設計ブロック 11×16 個
  (5mm 角)、細分セル 44×64 (1.25mm)。ポートは左辺中央 in / 右辺中央 out の
  各 5mm **固定** (最適化対象外)
- 流れ: 深さ平均ダルシー ∇·(K t/μ ∇p)=0。K(s)=K_open·(1e-5)^s の対数補間、
  s∈[0,1] がブロックごとの設計変数 (シグモイドロジット)。K_open=t²/12
- 熱: ベース板 (Al 3mm、伝導+発熱 10W/block) と流路層 (水 2mm、風上移流 +
  k(s) 面内伝導) を半厚み伝導直列で層間結合した 2N×2N 連成系
- 線形系: scipy splu を adjoint 付き `_SparseSolve` (autograd.Function) で
  ラップ。**勾配 FD 照合 2e-7 / 質量保存 1e-15 / エネルギー収支 1e-13**
- 目的: J = J_T + γ·ΔP/ΔP_ref、J_T = LSE_β(T_blocks)/T_ref + 10·var/T_ref²。
  Adam 600 反復 (~40 秒/γ、seed=0 で決定的に再現)

### 実測パレート (log: logs/log-darcy-1786196138.log)

| ケース | ΔP [Pa] | T_peak [K] | T_std [K] | 平均固体度 |
|--------|---------|-----------|-----------|-----------|
| 全開一様 | 16.2 | 41.4 | 1.42 | 0 |
| γ=10 | 26.4 | 8.9 | 0.21 | 0.043 |
| γ=1 (膝) | 29.1 | 7.5 | 0.10 | 0.060 |
| γ=0.1 | 44.5 | 6.3 | 0.11 | 0.128 |
| γ=0.01 | 117.3 | 5.9 | 0.11 | 0.250 |

### 知見 (差分図 coldplate_darcy_diff.png 参照)

1. 全開一様は in→out 中央直線にショートサーキットして最悪 (41.4 K)。
   **+10 Pa の抵抗勾配づけだけで 8.9 K に激減** — 流速分布そのものが支配レバー
2. 全 γ が**同一モチーフの濃さ違い**: 「発熱体右列〜出口側に抵抗、外周と左を
   開ける」パターンを γ↓ で深くするだけ (γ=10: K 最小 10^-0.57 → γ=0.01:
   10^-4.9 のほぼ壁)。パレート移動 = 同一設計のコントラストノブ
3. γ<1 の追加利得は上下端の発熱体列への流量再配分に集中 (ΔT マップで確認)

## 3. ファイル構成

```
coldplate.py        ネットワークモデル本体 (v2 + mode="thermal")
darcy.py            擬3D ダルシーモデル本体 (Geo/solve_flow/solve_heat/optimize/evaluate/panel)
run_darcy.py        γ スイープ → coldplate_darcy.png + coldplate_darcy_result.npz
run_darcy_diff.py   npz から差分可視化 (再最適化なし) → coldplate_darcy_diff.png
run_thermal.py      ネットワーク thermal 3 ケース比較
run_rect24.py / run_uniform.py / run_single_port.py / run_compare.py  (旧世代の実行系)
test_coldplate.py   20 件 / test_darcy.py 8 件 — 全合格 (計 28)
output/*.npz        再現用設計場。coldplate_darcy_result.npz: キー s_all-open,
                    s_g=0.01, ..., gammas, metrics, metric_keys
logs/               tee ログ 18 本 (STA2 エビデンス、失敗実験ログ含む)
```

## 4. 検証手順 (環境構築後に必ず)

```bash
cd experiments/coldplate
python -m pytest test_darcy.py test_coldplate.py -q   # 28 passed (~50 秒)
python run_darcy_diff.py 2>&1 | tee logs/log-darcy-diff-$(date +%s).log  # npz から数秒で再現
# フル再現 (~3 分、seed=0 で表の数値がビット単位一致するはず):
python run_darcy.py 2>&1 | tee logs/log-darcy-$(date +%s).log
```

lint: `ruff check darcy.py run_darcy*.py test_darcy.py && ruff format --check 同`

## 5. 失敗記録 (再発防止)

- **0/1 位相化は 2 回失敗して不採用**: (a) RAMP α_max=1e5 固定 → 中間 θ の
  圧損爆発で全開の自明解に潰れる。(b) α_max 継続 (50→1e5) + β 射影 +
  グレー罰則でも「θ̄≈0.5 の漏れ壁」(2値化で消える壁) を悪用した見かけの
  最適解が残り、2値化後 T_peak 15→39 K に劣化。ユーザー指定が「ブロックごとの
  ダルシー係数を変数に」だったため連続場設計に切替えて解消
- **シグモイド飽和による勾配死** (2 回目): ネットワーク版 w ロジットと同症状。
  darcy.optimize は 100 反復ごとにロジットを ±6 にクランプして予防
- ネットワーク thermal の mu_tvar が無効になるバグの経緯は README と
  logs/log-thermal-1786146683.log (失敗ログ) 参照

## 6. 次の候補 (未着手、優先順は次セッションの裁量)

1. **ピンフィン物理の整合**: 現状 s=1 は「流れを止める固体 = 良伝導体」。
   実機のピンフィン密度グレーディングなら「流れをある程度通し、かつ z 方向
   実効 h を上げる」中間状態のはず。K(s) と層間結合 U(s) の同時モデル化
   (多孔質フィン相関) にすると s の物理的意味が閉じる
2. **Forchheimer / Brinkman 補正**: ダルシー則は慣性項なし。全開域の流速
   (~0.4 m/s, Re_gap~800) では慣性損失が無視できない可能性。ΔP の絶対値の
   信頼性を上げるなら必要
3. **ポート位置の設計変数化** (現状は左右中央固定)。ネットワーク版の知見では
   対角カウンターフローが選好された — ダルシーでも同じかは未検証
4. **設計ブロック粒度の感度**: ユーザー仮説「セル単位だと枝的最適解」は
   未検証のまま採用した。refine/ブロックサイズを振って確認する価値あり
5. **γ 連続スイープ or ε 制約法**でパレートを密に引く (現在 4 点)
6. ネットワーク版の残課題 (P3 校正、P5 ヘッダ、多スタート) は README 末尾

## 7. 運用ルール (このレポの掟)

- 回答・文書は日本語。計算実行は必ず tee でログ保存 (`| tail` のみ禁止)
- 収束しない/失敗した実験もログを残して正直に報告 (STA2 防止ルール)
- 再現条件 (ブランチ・コミット・コマンド・seed) を結果と一緒に記録
- feature ごとにコミットし、PR #25 の本文を更新
- 本実験は Process クラス統合・status-index 登録の対象外 (独立実験)
