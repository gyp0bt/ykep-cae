# status-27: 単軸押出解析（展開チャネル 2.5D）設計

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/single-screw-extruder.md)

**日付**: 2026-09-02
**ブランチ**: `claude/single-screw-extruder-design`
**テスト数**: 286（変動なし。本 PR は設計文書のみ、実装なし）
**契約違反**: 0 件（登録プロセス 11）

## 概要

単軸押出機の混練性・滞留時間分布（RTD）を求めるための設計を策定した。
**実装は未着手**。次のセッションが設計文書から実装計画に入れる状態にしてある。

設計の中核は「螺旋対称性を使って 2.5D に落とす」こと。詳細は
[docs/design/single-screw-extruder.md](../design/single-screw-extruder.md)。

## 決まったこと

| 論点 | 結論 | 理由 |
|---|---|---|
| 主目的 | **混練性・滞留時間分布（RTD）** | 圧力特性や設計最適化ではなくこちら |
| スライディングメッシュ | **不要** | スクリュー座標系＋展開でバレルが接線方向に滑るだけになる。壁の接線移動は BC であってメッシュ運動ではない |
| 自由表面 | **不要** | RTD は充満した定常部で定義される量。起動時・飢餓供給・ダイスウェルは射程外 |
| 次元 | **2D 断面で 3 成分** | 一定ピッチは下流方向に断面不変。`∂/∂z = 0` で 3 成分が全部出る |
| フライト隙間 | **不等間隔格子で解像する** | 混練への寄与が最大の部位。`StructuredMeshProcess` が既に不等間隔を持つ |
| 対象部位 | **まず計量部、後で混練エレメント** | 計量部は厳密解があるので検証が固まる |
| 実装方式 | **ykep-cae 主体 + OpenFOAM で検算** | 解析解を真値、OpenFOAM を独立な第三の目に置く |

## 調査結果（ykep-cae 側の棚卸し）

```
  ✓ ある    不等間隔構造格子 (StructuredMeshProcess)
            SIMPLE / SIMPLEC / PISO、TVD、非直交補正
            スカラー輸送（対流-拡散-ソース、Dirichlet/Neumann/Robin）
            伝熱（Robin 対応）、Process Architecture、InternalFaceBC

  ✗ 無い    移動壁 BC        FluidBoundaryCondition = NO_SLIP / SLIP /
                             INLET_VELOCITY / OUTLET_PRESSURE /
                             OUTLET_CONVECTIVE / SYMMETRY のみ
            場としての粘度   natural_convection の mu はスカラー定数
            粘性発熱         エネルギー式に散逸項なし（Phase 2 で必要）
```

**`FluidProperties.power_law_n` / `power_law_k` は宣言済みだが未使用**（参照 1 箇所＝宣言のみ）。
非ニュートンの席は既に用意されていて、座る人がいない状態。

## 次にやること

1. 実装計画の作成（superpowers:writing-plans）
2. Phase 1: 等温・ニュートン → **ゲート G1/G2**（形状係数の厳密解と一致）
3. 非ニュートン → 隙間あり → **G3**（OpenFOAM と 1% 以内）
4. Phase 1.5: 粒子追跡と RTD → **G4**

**G1/G2 を通らなければ先へ進まないこと。**

## 外部依存（別プロジェクトの資産）

OpenFOAM 検算には `~/work/1a/a02` の資産を使う。

- `~/work/1a/a02/tools/of` — Docker ラッパ（資源上限つき、`OF_IMG` でイメージ切替）
- `opencfd/openfoam-run:2312` は**コンパイラを持たない**。`codedSource` を使うなら
  `OF_IMG=opencfd/openfoam-dev:2312`
- `1a/a02` のボクセルメッシュ品質ベンチマークの結論
  （**誤差は最狭方向のセル数だけで決まる。1% なら 20 セル、0.1% なら 63 セル**）を
  隙間の格子設計にそのまま適用する

## 未決事項

- `V sinφ` の向きと `x` 周期の圧力跳び `Δp` の符号（幾何から一意だが図で確認すること）
- 形状係数 `Fd` / `Fp` の級数の具体形と桁落ち対策
- 非ニュートンの反復スキーム（Picard か Newton か、粘度の緩和が要るか）
- 粒子追跡の時間刻みと隙間内での適応制御
- RTD の統計を取る粒子数と初期配置（流量重み付きが正しい）
- 諸元は仮（40mm 押出機）。実機に合わせる場合は差し替える
