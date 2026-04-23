# 持続的水槽設計 CAE システム ロードマップ（Phase 6）

[<- README](../README.md) | [<- ロードマップ(本体)](roadmap.md) | [<- ステータス一覧](status/status-index.md)

## 目標

xkep-cae-fluid を基盤として、**90×30×45 cm（W×D×H）水草水槽**の熱流体・物質輸送・生体反応・照明を統合した持続的設計 CAE を構築する。最終的には以下の定量設計を可能にする:

- ヒーター配置と水温均一性
- 外部フィルターのインレット/アウトレット位置と吐出方向の最適化
- CO2 添加量・拡散器配置と溶存 CO2 分布
- 植物ライト配光と光合成ポテンシャル
- 底面（多孔質）フィルターを活用した循環設計
- 昼夜サイクルでの CO2/O2 動態

## 対象機材（初期ターゲット）

| 機材 | 仕様・モデル化 |
|------|---------------|
| 水槽 | 90×30×45 cm（Lx=0.9, Ly=0.3, Lz=0.45 m） |
| 外部フィルター | エーハイム相当（インレット/アウトレット、吐出量 Q[L/h] をパラメータ化） |
| CO2 添加キット | ADA 相当、拡散器の位置・添加速度 [bubbles/sec] |
| 植物ライト | 上部吊下式 ×2、PAR 出力と配光角を指定 |
| 水槽用ヒーター | 棒状、200 W 級、定温 or 定熱流束モード |
| 底床 | ADA コロラドサンド、深さ 3〜8 cm、K≈1e-9 m²、空隙率 0.35 |

## 確定設計方針（2026-04-20 ユーザー決裁）

| 観点 | 決定 |
|------|------|
| 時間スケール | まず **定常解**、後半で **24h 日周期** に段階拡張 |
| CO2 添加モデル | **気泡プルーム簡易**（上昇速度＋界面積から溶存ソース導出） |
| mu=1.85e-5 mass 残差課題（CLAUDE.md 既知） | **並行別 PR** で対応（本系列は水が主対象のためブロッカーではない） |
| 自由表面 | **Rigid-lid + Robin BC**（ヘンリー則でガス交換） |

## 既存資産の再利用

| 用途 | 既存ファイル/クラス |
|------|---------------------|
| スカラー輸送の土台 | `natural_convection/assembly.py` のエネルギー方程式アセンブリ |
| 対流/拡散スキーム | `core/strategies/{convection,diffusion,tvd_convection,corrected_diffusion}.py` |
| Robin BC 機構 | `heat_transfer/solver*.py` の Robin 実装 → scalar_transport に移植 |
| 体積源の流し込み | `NaturalConvectionInput.q_vol` 経路（ヒーター・光合成・CO2 添加で再利用） |
| 固体マスク | `NaturalConvectionInput.solid_mask` → 砂層・ガラス・ヒーター形状に流用 |
| メッシュ生成 | `core/mesh.py::StructuredMeshProcess`（不等間隔対応済み） |
| 圧力ソルバー | AMG+CG / BiCGSTAB+ILU（既存のまま活用） |
| 時間積分 | BDF2 / 陰的 Euler（既存のまま活用） |

---

## フェーズ計画

### Phase 6.0: 前提整備（並行別 PR、0.5〜1 PR）

**背景**: CLAUDE.md 記載の空気実物性 mu=1.85e-5 mass 残差 O(1-100) 問題。水槽は水(mu≈1e-3)のため実害は限定的だが、SIMPLEC/PISO 連成の健全性確保のため並行で改善する。

- [ ] `natural_convection/assembly.py` の d 係数（運動量-圧力連成）計算見直し
- [ ] PISO 第 2 圧力補正の再チェック、速度緩和の再評価
- [ ] 残差正規化方式の再検証（面ベース質量残差）
- [ ] 検証: 既存 Ra=10⁴ ベンチマーク維持、空気物性 t=10s で mass 残差 < 0.1

### Phase 6.1: 汎用スカラー輸送基盤（2 PR）

**目的**: CO2 / O2 / 任意スカラーを既存流体場に乗せる共通フレーム。

#### 6.1a [status-21 完了 ✅]: ScalarTransportProcess 新設

- 新規: `xkep_cae_fluid/scalar_transport/`
  - `data.py`: `ScalarFieldSpec`, `ScalarBoundarySpec`（Dirichlet/Neumann/Adiabatic/Robin）, `ScalarTransportInput`, `ScalarTransportResult`
  - `assembly.py`: 対流-拡散-ソース疎行列アセンブリ（1 次風上 + 中心差分）
  - `solver.py`: `ScalarTransportProcess`（陰的 Euler + BiCGSTAB+ILU）
- 既存 `core/strategies/` の再利用は Phase 6.1b で判断（natural_convection の共通化含む）
- テスト: 1D 純拡散線形プロファイル、Neumann 勾配、Robin 平衡、1D 対流の質量保存

#### 6.1b [status-22 完了 ✅]: NaturalConvection 統合

- `NaturalConvectionInput.extra_scalars: tuple[ExtraScalarSpec, ...]` 追加
- `NaturalConvectionResult.extra_scalars: dict[str, np.ndarray]` で最終場を返却
- SIMPLE 外部反復内で Rhie-Chow 面速度を共有し、温度と他スカラーを同時輸送
- 物理テスト: 温度と共にトレーサー濃度が運搬される、閉じた系での質量保存

### Phase 6.2: 水槽ジオメトリ + ヒーター最小デモ（2 PR）← **最初の価値到達点**

#### 6.2a [status-23 完了 ✅]: AquariumGeometryProcess

- 新規: `xkep_cae_fluid/aquarium/geometry.py`
  - 90×30×45 cm 領域、底床マスク（砂層 3〜8 cm、不均一厚対応）、ガラス面マスク
  - 既存 `StructuredMeshProcess` を薄くラップ（底床と水面近傍を細かく不等間隔化）
  - 推奨重力ベクトル `(0, 0, -9.81)` を返却（z 軸鉛直）

#### 6.2b [status-24 完了 ✅]: HeaterProcess + 最小デモ

- 新規: `xkep_cae_fluid/aquarium/heater.py`
  - 定温制御（ヒステリシス ON/OFF、`prev_on` 呼出し側管理）と定熱流束の 2 モード
  - 既存 `q_vol` 経路に流し込む
- 新規: `examples/aquarium_heater_natural_convection.py`
  - Geometry + Heater + NC 3 段連携 → 温度分布と自然対流循環の PNG 出力
  - 現状は人工物性（実物性対応は Phase 6.0 完了後）

### Phase 6.3: 外部フィルター（強制対流統合、2 PR）

#### 6.3a [status-25 完了 ✅]: InternalFaceBC + AquariumFilterProcess

- 実装: `InternalFaceBC` / `InternalFaceBCKind`（`natural_convection/data.py`）と
  対応するアセンブリ拡張（`natural_convection/assembly.py`：ペナルティ法で INLET 速度/温度/
  p'=0、OUTLET で p'=0 ピン留め）、ソルバー post-correction 再強制（`solver.py`）
- 実装: `AquariumFilterProcess`（`aquarium/filter.py`）で Eheim 相当のバウンディングボックス
  ノズル + 流量 Q [L/h] + 吐出方向ベクトルから `InternalFaceBC(INLET/OUTLET)` ペアを自動構築
- 既知課題: 強制入口による mass 残差増幅（SIMPLEC/PISO 化まで厳密質量保存は未達、
  当該テストを `@pytest.mark.xfail(strict=False)` でトラッキング）

#### 6.3b [status-26 完了 ✅]: 強制＋自然対流統合例

- 実装: `examples/aquarium_filter_circulation.py`
  - Geometry + Heater + Filter + NC の 4 段連携、2×2 パネル PNG を出力
  - 流入（上部左、+x 吐出）＋ 流出（底部右） + ヒーター（中央、定熱流束 2 W）
- 検証:
  - [x] 流入側で強制速度が維持（`u[inlet]=target` ✓）
  - [x] 流入体積流量が `Q=60 L/h → 1.667e-5 m³/s` と一致
  - [ ] 流出=流入の厳密バランス（< 1% 誤差）→ **SIMPLEC/PISO 化まで未達**（status-25 xfail と同根因）
- 既知課題: buoyancy + INLET 強制で `t≈2-3s` 以降に velocity 崩壊パターン。
  `t_end=2.0s` で停止させて立ち上がり段階を可視化（SIMPLEC/PISO 化後に長時間積分）。

### Phase 6.4: 多孔質媒体 Darcy-Forchheimer（2 PR）

#### 6.4a [status-27 予定]: PorousMediumProcess

- 新規: `xkep_cae_fluid/porous/process.py`
  - 運動量方程式に `-μ/K·u - C_F·ρ|u|·u` を体積ソース項として追加（`assembly.py` の Source 配線拡張）
  - ADA コロラドサンド物性プリセット、ろ材プリセット

#### 6.4b [status-28 予定]: 物理検証

- 1D Darcy 流れ解析解、Ergun 式比較、底床内部流速の妥当性

### Phase 6.5: 植物ライト Beer-Lambert（1 PR）

#### 6.5 [status-29 予定]: LightFieldProcess

- 新規: `xkep_cae_fluid/aquarium/lighting.py`
  - `I(x,y,z) = I0(x,y) · exp(-∫κ_water dz - Σ κ_i·c_i·path)` 上面下向き
  - 2 灯の配置（x-y 位置、配光角、PAR 出力）を `LightSourceSpec` で指定
  - 出力: 光量場 [μmol/(m²·s)] を scalar として返却（光合成で参照）

### Phase 6.6: 生体反応ソース項（2 PR）

#### 6.6a [status-30 予定]: BiologicalReactionProcess

- 新規: `xkep_cae_fluid/aquarium/biology.py`
  - 光合成: `R_p = V_max · I/(I+K_I) · CO2/(CO2+K_c) · Q10^((T-T0)/10)`
  - 呼吸: `R_r = R_0 · Q10^((T-T0)/10)`（常時）
  - 植物マスク領域のみ活性、非負制約、O2 生成と CO2 消費の化学量論一致

#### 6.6b [status-31 予定]: 保存則テスト

- 夜間（I=0）呼吸のみ → CO2 単調増加・O2 単調減少
- 光 ON での CO2 消費量と光量・植物量の比例関係

### Phase 6.7: ガス交換界面 + CO2 気泡プルーム（2 PR）

#### 6.7a [status-32 予定]: AirWaterInterfaceBC（Rigid-lid）

- 上面に Robin BC: `J = k_L·(c_eq - c)`（c_eq はヘンリー則）
- 既存 `heat_transfer` の Robin 機構を `scalar_transport` に移植

#### 6.7b [status-33 予定]: CO2BubblePlumeProcess

- 新規: `xkep_cae_fluid/aquarium/co2_injection.py`
- 簡易プルーム: 拡散器セル上の気泡上昇軌跡に沿って溶解ソース項を分配
- 入力: 添加量 [bubbles/sec], 気泡径 d_b, 拡散器位置
- 気泡上昇速度 `u_b = f(d_b)`（Clift-Grace-Weber 相関）→ 滞留時間 `τ(z) = (H-z)/u_b`
- 物質伝達 `J_b = k_L · a_b · (c_eq - c)`（a_b: 気泡比界面積）を体積セルへ分配
- 出力: 溶存 CO2 体積ソース場 [mol/m³/s]

### Phase 6.8: 水槽システム統合デモ（1〜2 PR）

#### 6.8 [status-34 予定]: 定常フル統合

- 新規: `examples/aquarium_full_steady.py`
- 定常解: ヒーター + フィルター循環 + 砂層多孔質 + 光 + 光合成 + CO2 添加 + ガス交換
- 出力: 温度分布、CO2/O2 濃度分布、流速分布、光量分布の 4 連画像
- STA2 防止: 収束判定を通過した結果のみ報告する。未収束の場合は緩和せず「未収束」と明記

### Phase 6.9: 日周期拡張 + 設計最適化の足場（将来）

- [ ] **6.9a**: 24h 日周期シミュレーション（PISO + BDF2、t=86400 s、`output_interval` 調整）
- [ ] **6.9b**: 設計パラメトリックスタディ（吐出方向、ヒーター位置、CO2 添加量、光量）
- [ ] **6.9c**: 目的関数（温度均一性、CO2 分布、酸素最小値）を定義して設計評価

---

## 最短価値到達パス

| マイルストーン | 完了 PR 数 | 得られるもの |
|---|---:|---|
| **初デモ**（ヒーター自然対流） | **PR 4**（Phase 6.2b） | 水槽内温度+流れ可視化 |
| 循環デモ（フィルター統合） | PR 6（Phase 6.3b） | 強制+自然対流の流速場 |
| **統合デモ**（全機能定常） | **PR 12**（Phase 6.8） | 設計検討可能な全物理 |
| 日周期 + 最適化 | PR 14+（Phase 6.9） | 定量的設計最適化 |

## 新規作成ファイル一覧（計画）

```
xkep_cae_fluid/
├── scalar_transport/         # Phase 6.1
│   ├── data.py
│   └── solver.py
├── porous/                   # Phase 6.4
│   └── process.py
└── aquarium/                 # Phase 6.2, 6.5, 6.6, 6.7
    ├── geometry.py           # 6.2a
    ├── heater.py             # 6.2b
    ├── lighting.py           # 6.5
    ├── biology.py            # 6.6
    ├── co2_injection.py      # 6.7b
    └── interface.py          # 6.7a (AirWaterInterfaceBC)

examples/
├── aquarium_heater_natural_convection.py   # 6.2b
├── aquarium_filter_circulation.py          # 6.3b
└── aquarium_full_steady.py                 # 6.8
```

## 未確定・将来検討事項

- 気泡プルームの気泡径・上昇速度のデフォルト値（実測 or 文献調査必要）
- 光合成 Michaelis-Menten 定数（水草種ごとに異なる → プリセット化）
- 底床生物膜（硝化菌）の扱い — Phase 6.6 追加 or Phase 7 構想
- 硝酸塩/アンモニア循環（窒素サイクル）— Phase 7 構想
- 設計最適化の目的関数設計（Phase 6.9c で詳細化）
- 乱流モデル（Phase 5 完了後に適用検討、水槽は低 Re のため層流仮定で多くのケースは足りる見込み）

## 品質ゲート

- すべての PR で `ruff check xkep_cae_fluid/ tests/` と `ruff format --check` を通過
- `python contracts/validate_process_contracts.py` で契約違反 0 件維持
- 各物理テストは解析解 or 保存則を基準として検証、**STA2 防止**として未収束は未収束と明記
- 計算実行ログは `tee` でファイル出力し、YAML 出力と照合可能にする（CLAUDE.md 準拠）
