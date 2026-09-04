# status-28: 単軸押出解析 Phase 1 / 1.5 実装 — ゲート G1〜G4 全通過

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/single-screw-extruder.md) | [実装計画](../plans/2026-09-02-single-screw-extruder-impl.md)

**日付**: 2026-09-03
**ブランチ**: `claude/single-screw-extruder-impl`（master `9de63fc` から分岐、status-27 の設計を実装）
**テスト数**: 286 → **438 passed**（10 failed は pyamg/numba 未導入の既存分で不変、8 skipped, 1 xfailed）
**契約違反**: 0 件（登録プロセス 11 → **17**）

## 概要

設計文書（status-27）の 2.5D 定式化を `xkep_cae_fluid/extruder/` に実装し、
検証の階段 G1 → G2 → G2b → G3 → G4a → G4b を**全て通過**した。
成果物は 3 つの層に分かれる。

```
  解析解の層   shape_factors.py   Fd/Fp 級数（閉形式 + 指数減衰の残余）
               ↓ 真値
  流れの層     geometry.py        展開チャネルの不等間隔格子（隙間を等比で刻む）
               down_channel.py    w: 可変係数 Poisson（splu）
               cross_channel.py   u,v,p: MAC Stokes 鞍点系（splu、圧力スケーリング）
               viscosity.py       Newtonian / PowerLaw / Carreau + Green-Gauss γ̇
               solver.py          ExtruderFlowProcess — Picard 結合、Q_axial
               ↓ 速度場
  混練の層     tracker.py         ψ 双一次補間の発散ゼロ場で RK4 追跡（ζ 座標）
               rtd.py             流束重み付き RTD、パーセンタイル、累積せん断
```

図解は [押出機の中の8秒 / 押出断面のフィールド](../reports/extruder/README.md)（公開 Artifact）。
OpenFOAM 検算は [G3 レポート](../reports/extruder/g3-openfoam.md)（公開 Artifact: https://claude.ai/code/artifact/c5512b21-f3ba-42fe-9441-cec76ec4e9bb ）。

## 検証ゲート（閾値で規格化した比、合格は比 < 1.00）

| ゲート | 内容 | 閾値 | 比 | 備考 |
|---|---|---|---|---|
| G1 | 純引きずり `Q = VHW·Fd/2` | 0.1% | **0.07** | 格子 429×96 |
| G2 | 引きずり + 圧力の直線（G = −2e6…+2e6 の 5 点） | 0.1% | **0.07〜0.08** | Fp の式の誤りを捕まえた |
| G2b | 断面内 1D 解 `u = U(3η²−2η)` | 観測次数 2 | **2.00** | ny=160 で相対 2.9e-5 |
| G3a | OpenFOAM ニュートン（Q / Q_leak / w(y) / u(y)） | 1% | **5e-10 / 0.40 / 3e-10 / 0.006** | 同一格子 16960 セル |
| G3b | OpenFOAM べき乗則 K=2e4, n=0.4 | 1% | **0.15 / 0.52 / 0.046 / 0.088** | G3a 収束解から起動 |
| G3 較正 | 1D powerLaw ↔ 厳密解 | 0.5% | **0.025 / 0.040 / 0.048** | k = K, n = n を確定 |
| G4a | 1D 厳密軌跡（y 保存・t_res） | 1e-8 | **機械精度** | |
| G4b | `⟨t⟩ = z_axial·A_free/(sinφ·Q)` | 1.5% | **0.35〜0.67** | 隙間ありで判定 |
| 隙間収束 | Q_leak の n_gap 次数 | 2 | **1.8〜2.1** | a02 基準より 50 倍良い |

## 決まったこと（実装で確定した設計上の論点）

| 論点 | 結論 | メカニズム |
|---|---|---|
| 非ニュートンの反復 | **Picard（ω=0.5）** | 線形解が毎回厳密なので擬似時間の制約が無く、10 回程度で不動点に着く。Newton は見かけ粘度のヤコビアンが密で組み立てが線形解 1 回分を超える |
| 押出量 | **`Q_axial = Q + L_turn·Q_leak`** | `∫∫w dA` は隙間で**増える**（バレル直下に流路が足される）。軸方向面の流束を取ると漏れ 1 単位が材料を L_turn 戻す。古典的「隙間は押出量を減らす」は Q_axial の話 |
| 追跡座標 | **ζ（軸方向）** | x 周期の同一視で z は跳ぶが ζ = x cosφ + z sinφ は不変。計量部長も実機は軸方向で測る |
| 種まき | **決定論的・流束重み** | 双一次場はセル中心値がセル平均に厳密一致するので 1 点求積が厳密、モンテカルロ誤差ゼロ |
| RTD の判定 | **隙間ありで行う** | 閉チャネルは隅の閉じた流線がデッドゾーンになり ⟨t⟩ が格子で収束しない。隙間はそのトラップを壊す（実機の隙間は材料が淀まない条件でもある） |
| G3 の格子 | **同一格子で比較** | blockMesh の多区間 grading で ykep の等比区間を厳密再現。格子差を切り分けから消す |
| simpleFoam の緩和 | **U 0.999 / p 1.0（SIMPLEC）** | 緩和係数 α は擬似時間刻み α/(1−α)·V/a_P。隙間セルで a_P ≈ 2ν/Δy² が巨大になり、α=0.9 では減衰率 1−7e-5/反復で実質収束しない。α=1 は SIMPLEC が 0 割で落ちる |
| 隙間の格子基準 | **n_gap=20 で 0.01%** | a02 基準（1% に 20 セル）は等間隔ボクセルの話。境界適合の等比格子なら 50 倍良い |

## 実測リソース（NN 学習との共存の根拠）

| 処理 | 格子 | 時間 | 最大 RSS |
|---|---|---|---|
| ExtruderFlowProcess ニュートン | 248×80 | 0.4 s | 199 MB |
| ExtruderFlowProcess べき乗則（Picard 43 反復） | 248×80 | 25 s | 〜200 MB |
| 追跡 + RTD（16960 粒子） | 248×80 | 〜1 min | < 500 MB |
| OpenFOAM G3a（Docker, 1 CPU, 1200 MB 上限） | 16960 セル | 15519 反復 / 597 s | < 1200 MB |
| OpenFOAM G3b | 同 | 15469 反復 / 1062 s | < 1200 MB |
| pytest 全体（OMP 2 スレッド、457 件） | — | 6 min 35 s | 424 MB |

4 GB / 2 スレッドの予算内。OpenFOAM は別コンテナで 1 CPU。

## 設計文書 §0 の訂正記録

`L_turn = πD cosφ`（旧版は `πD/sinφ` = 414 mm と誤り）。`W_t cosφ = L_turn sinφ` の恒等式で
ζ 座標が x 周期に対して不変になる。`docs/` を grep して旧記法の残存は無し。

## 未決事項 / 次にやること

- **実機データとの突き合わせ**（Phase 2 に入る前に必須）。諸元は仮の 40 mm 機
- Phase 2: 粘性発熱 `Φ = μγ̇²` と温度依存粘度（Picard に温度を 1 本足す）
- Phase 3: 混練エレメント（螺旋対称性が壊れるので 3D、messi + OpenFOAM）
- G3 で最も差が大きいのは Q_leak（比 0.40 / 0.52）と断面内速度 v の全域 L2（1.1% / 1.8%、ゲート外）。MAC と Rhie–Chow の圧力結合の差で、
  格子細分で両者とも同じ極限に向かうが、断面内速度を 0.1% で使う用途が出たら格子を増やす
