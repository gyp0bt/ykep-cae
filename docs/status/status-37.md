# status-37: 非構造メッシュの粒子追跡と滞留時間分布（Pollock 型、Phase 12 完了）

[<- README](../../README.md) | [<- docs](../README.md) | [<- ステータス一覧](status-index.md) | [<- roadmap](../roadmap.md)

- 日付: 2026-09-06
- ブランチ: `claude/phase11-roadmap-todo-n3s0lj`
- 前: [status-36](status-36.md)（汎用記法で押出級の流れを書く / Phase 12）

---

## 1. 何を求められたか

status-36 の残件の筆頭:

> 非構造メッシュの**粒子追跡 / RTD**（構造格子の ψ 双一次補間のみ）。
> 汎用経路で RTD を出すには面流束ベースの追跡（Pollock 型）が要る

押出の第一目的は流速場ではなく**混練性と滞留時間分布（RTD）**なので、汎用記法で書いた
.inp から E(t)・累積せん断ひずみ γ・混合指数 λ が出せないと Phase 12 は片肺のままだった。

## 2. なぜ既存のトラッカーが使えないか

構造格子版（[`extruder/tracker.py`](../../xkep_cae_fluid/extruder/tracker.py)）は**節点流れ関数 ψ**を
双一次補間して `u = ∂ψ/∂y, v = −∂ψ/∂x` とすることでセル内の発散ゼロを恒等的に担保している。
ψ は 2 次元の構造格子でしか作れない。

かといってセル中心速度を線形補間すると離散的な発散ゼロが壊れ、粒子が渦心に落ち込んだり
壁に貼り付いたりして RTD の裾が偽物になる。**FVM が実際に満たしているのは面流束の総和
Σ_f q_f = 0** なので、そこから出発するのが筋。

## 3. やったこと

設計文書: [particle-tracking-fvm.md](../design/particle-tracking-fvm.md)（新規）

### 3.1 セル内の再構成（`post/tracking.py`）

セル内の速度を `u(x) = a_c + B_c (x − x_c)` とし、**そのセルの全ての面について流束を厳密に
再現する**拘束

```
(a_c + B_c d_f)·S_f = q_f
```

を課す。未知数 12（3 次元）に対し拘束は面数（六面体で 6）なので不足決定で、残りは
**最小ノルム解**（`np.linalg.pinv`）で閉じる。ノルムは無次元化した `B̃ = B·L_c` に対して取る。

- 直交六面体では**Pollock（1988）の再構成そのもの**になる（軸ごとに分離、非対角は最小ノルムが 0）
- 四面体では**最低次 Raviart–Thomas（RT0）**と一致する（非圧縮なら一定速度）
- `∇·u = tr(B_c) = Σ_f q_f/V_c` なので、離散連続式を満たす面流束を渡せばセル内で恒等的に 0
- セル形状は問わない（六面体・楔・四面体・角錐の混在可）

### 3.2 セルからセルへの受け渡し

刻みは面平面までの直線近似到達時刻 `τ_f = −s_f/(u·n̂_f)` で決め、RK4 で 1 歩進めて
false position（4 回）で面上に落とし、法線方向に厳密に射影して隣接セルへ渡す。
セル内で場が厳密にアフィンなので **RK4 は解の 4 次 Taylor 打ち切りそのもの**になり、
刻み幅は精度ではなく面の検出のためだけに要る。

- **周期面**は内部面としてそのまま跨げる。並進 T を位置に掛け `shift_total ← shift_total − T`
  を持ち回るので `x + shift_total` が連続な「巻き戻さない座標」になる。押出の ζ（軸方向座標）は
  この座標の軸方向成分そのもの
- 周期方向が 1 層のとき（押出 2.5D の z）、周期対は **owner == neighbour の自己面**に併合される。
  セル → 面テーブルが同じ面を符号 ±1 で 2 項目持ち、面平面の位置が並進分ずれるので別平面として通る
- 流束が（総流束比で）`wall_flux_tol` 以下の境界面は**壁**として押し戻し、流束のある境界面は
  **流出**として脱出（どのパッチから出たかを記録）

### 3.3 種まきと脱出（`seed=`）

| `seed` | 種まき | 重み | 脱出 |
|---|---|---|---|
| `"patch"` | 流入する境界面 1 枚につき 1 粒子 | 流入流束 [m³/s] | 流束のある境界面を跨ぐ |
| `"axial"` | 流体セル 1 個につき 1 粒子 | `max(u_c·â, 0)·V_c` | ζ = â·(x + shift_total) が `length` に達する |
| `"explicit"` | 呼び出し側が位置・重み・セルを与える | 任意 | 同上 |

種まき速度は**追跡に使うのと同じ再構成場**から取る（セル中心では `u = a_c`）。
ステップ上限に達した粒子は進行率から外挿するが、**ζ ≈ 0 の粒子は外挿しない**
（淀みに捕まった粒子の外挿は ⟨t⟩ を桁で壊す。構造格子版と同じ歯止め）。

### 3.4 厳密関係 ⟨t⟩ = length·V/Σw

`seed="axial"` の理論平均滞留時間は

```
⟨t⟩ = length · V_total / Σ_c (u_c·â) V_c
```

（ζ 面の断面積を A_ζ、周期の ζ 長さを Δζ とすると V = A_ζΔζ、Q = Σ(u·â)V_c/Δζ、
⟨t⟩ = length·A_ζ/Q）。**Δζ を知らなくても書ける**のが要点で、構造格子版の
`⟨t⟩ = z_axial·A_free/(sinφ·Q_axial)` はこの 2.5D 特殊形になる。

再構成の誤り・種まき重みの誤り・面の取りこぼし・周期の記帳ミス・脱出時刻の内挿ミスを
**同時に**捕まえる、RTD の最も鋭い検査。

### 3.5 RTD の集計（`post/rtd.py`）

`ResidenceTimeProcess` が流束重み付きで E(t) / F(t) / 分位点 / 経路積分スカラーの統計を作る。
ビン幅に依存しない重み付き経験分布 `t_ecdf` / `F_ecdf` も返す。
`rate_scalars` に入れた名前は `∫s dt / t`（時間平均）として扱う（混合指数 λ 用）。
重み付き分位点・経験分布は `extruder/rtd.py` から [`post/statistics.py`](../../xkep_cae_fluid/post/statistics.py) へ移して共有した。

### 3.6 NS 結果に γ̇ と λ を常時出す

`NavierStokesFVMResult.strain_rate`（γ̇ = sqrt(2 D:D)）を粘度モデルの有無に依らず出し、
`mixing_index`（λ = |D|/(|D|+|Ω|)）を追加した。収束後の速度勾配から 1 回だけ作る。
`.inp` の `*OUTPUT` では `GAMMA` / `LAMBDA`。混練性の評価にはニュートン流体でも要る。

## 4. 検証（数値は実測）

### 4.1 再構成と軌跡

| 検査 | 結果 |
|---|---|
| 全ての面の流束を再現 | 相対残差 < 1e-12 |
| 一様流 / せん断 / 線形発散ゼロ場 | `a`・`B` とも厳密（< 1e-10） |
| 発散 `tr(B) = Σq/V` | < 1e-10 |
| 一様流（プラグフロー） | 全粒子が同じ滞留時間（ptp < 1e-12）、`t = L/U` と 1e-5、広がり 1.0 |
| 単純せん断（G4a 相当） | 軌跡が直線（Δy < 1e-12）、`t = L/(S y)` と 1e-5、F(t) = 1 − (t_min/t)² と 3e-2 |
| 蓋駆動キャビティ（Stokes） | 全粒子が領域内に残る（壁を跨がない） |
| 周期 Poiseuille（G4b 相当） | **⟨t⟩ = length·V/Σw と 1e-12**。解析解 V/Q とは ny=12 で 1.4%、ny=24 で 0.35%（2 次収束） |

### 4.2 押出の展開チャネル（構造格子トラッカーとの照合）

40 mm 機の計量部（μ=1000 Pa·s, G=1e5 Pa/m, z_axial=0.05 m）。参照は専用 2.5D ソルバー +
ψ 双一次補間トラッカー（ゲート G4a/G4b/G5 通過済み）。**流れ場の作り方も追跡の原理も別物**。

例題 [extruder-channel-1](../../examples/inp/extruder-channel-1.inp)（2016 セル、両者同解像度。
ログ [extruder-channel-1-rtd.log](../../examples/inp/results/extruder-channel-1-rtd.log)、
再現は `python examples/extruder_generic_rtd.py`）:

| 量 | 汎用 | 参照 | 相対差 |
|---|---|---|---|
| ⟨t⟩ [s] | 3.11257 | 3.13296 | 6.5e-3 |
| ⟨t⟩ 理論 [s] | 3.13651 | 3.14891 | 3.9e-3 |
| t_p10 [s] | 1.99184 | 1.99525 | 1.7e-3 |
| t_p50 [s] | 2.49600 | 2.49350 | 1.0e-3 |
| t_p90 [s] | 4.33312 | 4.32463 | 2.0e-3 |
| 広がり t_p90/t_p10 | 2.17544 | 2.16746 | 3.7e-3 |
| γ = ∫γ̇dt | 121.813 | 122.771 | 7.8e-3 |
| 混合指数 λ | 0.499580 | 0.499568 | 2.4e-5 |

解像度を上げると縮む（テスト用の別解像度での実測、z_axial=0.05）:

| 量 | 1184 セル | 4736 セル |
|---|---|---|
| ⟨t⟩ | 5.1e-3 | 3.9e-3 |
| t_p10 / t_p50 / t_p90 | 8.4e-3 / 6.5e-3 / 1.1e-2 | 2.2e-4 / 3.8e-4 / 9.6e-4 |
| γ | 8.3e-2 | 1.3e-2 |
| λ | 5.3e-5 | 4.7e-6 |

## 5. 途中で分かったこと

- **面拘束で刻んだステップは面に「ちょうど」乗る**（丸めで `s = 0`）。跨いだ判定を
  `s > 0` の不等号だけにすると受け渡しが起きず、以降 `dt = 0` のまま止まる。
  刻みが面拘束で決まったときは「到達」も跨いだ扱いにする必要がある
- **Pollock 型の精度の性格**: せん断流 `u = S y` を直交格子で再構成すると `u_x` は y に
  階段状（セル内で一定）になる。y 方向は 1 次で ψ 双一次補間より粗い。代わりに
  「どんなセル形状でも局所質量保存が厳密」が手に入る。精度は細分化で回復する
- **計算量が周期方向のセル数に効く**。押出 2.5D は z が 1 セルなので下流へ進むたびに
  自己面を跨ぎ、例題で 1 粒子あたり 250〜60000 ステップ・1920 粒子で 70 s 掛かる
- 押出の x 周期は「(x, z) ~ (x − W_t, z + L_turn)」だが、汎用メッシュでは純粋な x 並進で
  書いてある。それでも ζ = x_u cosφ + z_u sinφ は正しい — 並進 W_t が ζ を W_t cosφ =
  L_turn sinφ だけ進めるので、物理的な同一視と一致する

## 5.5 後方互換の除去とテスト実行時間

### 後方互換をすべて切った

| 消したもの | 置き換え |
|---|---|
| `RegistryProxy` と `AbstractProcess._registry` | `ProcessRegistry.default()` を直接使う |
| `ProcessMeta.deprecated` / `deprecated_by`、`DeprecatedProcessError`、実行時の deprecated 判定、`uses` の deprecated 警告、`ProcessRegistry.non_deprecated`、doc 生成の DEPRECATED 行 | deprecated なプロセスは 1 つも無いので機構ごと削除 |
| メタクラスの `warnings.catch_warnings` 再送出と `ProcessExecutionEntry.warning_type` | deprecated 検出のためだけの仕掛けだった。警告は素通しでよい |
| `extruder/viscosity.py` の粘度モデル再輸出 | `xkep_cae_fluid.fvm.viscosity` から直接 import |
| `extruder/rtd.py` の `weighted_quantile` / `weighted_ecdf` 再輸出 | `xkep_cae_fluid.post.statistics` から直接 import |

### 全件テストが 14 分 26 秒 → 2 分 28 秒

`--durations` を取ると、時間の半分以上が押出の構造格子 RTD/ソルバーに集中していた。
**同じ諸元の流れ場を各テストが独立に解き直していた**のが原因で、テストの数ではなかった。

- `flow_of` / `pipeline` / `run` / `track` を `functools.cache` で共有（返り値は読み取り専用で扱う）。
  `pytest.ini` に `--dist loadfile` を入れて、`-n` 並列でもファイル単位で 1 ワーカーに固める
- `test_back_pressure_lengthens_residence` は理論平均滞留時間だけを見ていたので、
  粒子追跡をやめて流れ場（`Q_axial`）から計算するようにした（88 s → 2 s）
- 格子収束・パラメータ非感受性・前処理遅延の 6 件を `slow` に移した
  （`TestDeadZone`、`TestNonNewtonian` の γ̇ クランプと緩和、`TestStreamlineDrift`、
  `TestLaggedPreconditionerConvergence`、`test_fixed_config_matches_process_solver`）
- 開発依存に `pytest-xdist` を追加。`python -m pytest tests/ -q -m "not slow" -n 4` で 2 分 28 秒

**実測（本環境、4 並列）**: slow 除外 **838 passed / 15 skipped / 1 xfailed、2 分 28 秒**。
`slow` のみ **29 passed / 1 failed、4 分 25 秒**。

失敗した 1 件は `test_natural_convection.py::TestAMGPressureSolver::test_adaptive_relaxation`
で、**本ブランチの変更前（コミット 17506e8）でも同じく失敗する既存の未解決問題**
（CLAUDE.md「未解決の物理的不整合」の空気実物性 μ=1.85e-5 での SIMPLE 連成不足）。
今回の変更が原因ではない。`slow` を含む全件はこれまで CI でも回っていなかったので、
ここで初めて表に出た。

## 6. 残件

- `.inp` から RTD を出す**キーワード（`*RTD` 相当）は未実装**。いまは Python API と
  [`examples/extruder_generic_rtd.py`](../../examples/extruder_generic_rtd.py) のみ
- **自己面（1 層周期）の複数回横断をまとめる最適化**（追跡が 1 粒子数万ステップになる）
- **非定常流には未対応**（定常場のみ）。点からセルを探す機能も無い（`seed="explicit"` は
  初期セルも与える）
- セルが**凸**であることを仮定している
- status-36 からの持ち越し: 回転周期・螺旋周期、COUPLED の Krylov 化と `OUTFLOW`、
  粘性発熱 `Φ = μγ̇²` と温度依存粘度
- `slow` の `TestAMGPressureSolver::test_adaptive_relaxation` が落ちたまま（既存問題。
  空気実物性の SIMPLE 連成。CLAUDE.md の最優先事項）
