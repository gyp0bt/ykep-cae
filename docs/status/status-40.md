# status-40: nsb の制御則を一長一短の切替だけに絞る — Stokes 参照場・粗格子解からの発進

[<- README](../../README.md) | [<- docs](../README.md) | [<- ステータス一覧](status-index.md) | [<- roadmap](../roadmap.md)

- 日付: 2026-09-08
- ブランチ: `claude/nsb-fgmres-hierarchy-numba`（status-39 と同じ PR #35）
- 前: [status-39](status-39.md)（FGMRES・SA 階層再利用・numba 残差）
- 環境: 手元機（20 コア）。依存は status-39 と同じ

---

## 1. 何を求められたか

status-39 の議論で、Newton 反復数（288×192 で 22〜36 回）が総時間の分散を支配していると分かった。
gyp さんの方針: **静止場発進は不要なので切る。収束判定の基準 R0 は何があっても Stokes 解で計算する。
比較実験用に立てていた非効率オプションも切って、一長一短のオプション群だけの構成にする。**
以前、粗格子解を初期場にする試みは「擬似時間増分の戦略がいけてなくて効果を正しく見れなかった」ので、
その原因もここで潰す。

## 2. 全体像: 何を落とし、何を残したか

```
                 status-39 まで                         status-40
  参照場 r0   |R(初期場)|（初期場が良いほど判定が厳しくなる）   |R(Stokes 場)|（常に同じ物差し）
  初期場      "zero" / "stokes"                             Stokes 解、または NSBInput.u0/v0/p0（粗格子解の補間）
  初期 CFL    cfl_init 固定                                  cfl_init·|R_ref|/|R_init|（SER の古典形）
  線形        jfnk / lu / jfnk_simple / dc_simple            jfnk_simple（既定）/ jfnk
  運動量近似  ilu / jacobi                                    ilu
  棄却        reject_growth / max_rejects / cfl_min           なし
  速度下限    velocity_floor [m/s]（既定 0 = 下限なし）        velocity_floor_ratio（既定 0.1、u_scale 比）
  残差        fast_residual True/False                        numba（無ければ numpy）
  SA 階層     reuse_hierarchy True/False                      再利用のみ
  既定        alpha_u 0.7 / precond_cfl_ratio 4               alpha_u 1.0 / precond_cfl_ratio 2
```

落とした側はいずれも実験で常に劣った（静止場: status-30、Jacobi・dc_simple・lu: status-38、
backtracking・下限なし: status-30、numpy 残差・毎回構築: status-39）。残した切替は [nsb README の表](../../nsb/README.md)
に「何と何のトレードオフか」を書いた（`local_dtau`、`pseudo_time_in_residual`、`linear_solver`、
`velocity_floor_ratio`、CFL 則、`alpha_u`、前処理の使い回し、GMRES 許容、対流スキーム）。

## 3. 機構

### 3.1 参照場を Stokes 解に固定する

**現象**: 粗格子解を初期場にすると |R(初期場)| が小さく、相対判定 `|R|/|R(初期場)| < 1e-6` が
その分だけ厳しくなり、同じ解に到達しても「収束していない」扱いになる。

**対策**: 参照 r_ref を「Stokes–Brinkman 解で評価した完全 NS の定常残差」に固定する。Stokes 場は
決定的で格子に整合し、その残差は対流の不釣り合い ρU²Δy のスケールを持つ。初期場が何であっても
同じ物差しで判定できる。Re → 0 で Stokes 場が厳密解になり r_ref = 0 になる場合だけ静止場の残差に落とす。
Stokes 解は常に解く（288×192 で SIMPLE 前処理付き GMRES 1 回、1〜2 s）。`NSBResult.residual_ref` に記録し、
`rel_residual` / `rel_steady_residual` もこれで割る。

### 3.2 SER の出発点

**現象**（以前の粗格子初期場が効かなかった理由）: SER は毎反復の残差比 `cfl *= |R_prev|/|R_new|` で
`cfl_init=0.5` から出発する。良い初期場では残差比が 1 に近く CFL が育たないので、解の近傍で
低 CFL の擬似時間反復を延々やる。初期場の質が CFL に反映されない。

**対策**: SER の古典形 `CFL_k = CFL_0·|R_0|/|R_k|` の |R_0| に r_ref を入れ、出発だけ
`cfl = cfl_init·r_ref/|R(初期場)|` で直接決める（以後は今の比の積のまま）。初期場が参照場より
良ければその分だけ大きい CFL から始まる。

### 3.3 速度下限を比で与える

`velocity_floor` [m/s] の既定 0 は「下限なし」で停滞の主因だった（status-30）。本体側
（`xkep_cae_fluid.brinkman_flow`）と同じく `velocity_floor_ratio × u_scale`（`BrinkmanDiscretization.u_scale`
= 最大流入速度、領域内ソースは周長 4√A から見積もる）にして、ケースの流速に依らず既定 0.1 で動くようにした。

## 4. 実測: 粗格子解からの発進（`scratchpad/nested.py`、flat、U=1、既定構成、20 コア）

144×96 の収束解を 2×2 の区分定数注入で 288×192 に持ち上げ、`NSBInput.u0/v0/p0` に渡した。
参照 r_ref = 39.07 は両者で共通。

| 発進 | 初期 \|R\|/r_ref | 初期 CFL | Newton | GMRES | 組立 | 時間 | 解の差 |
|---|---|---|---|---|---|---|---|
| (a) Stokes 解 | 1.00 | 0.5 | 36 | 1985 | 31 | 42.3 s | — |
| (b) 144×96 解の注入 | 0.50 | 1.01 | **16** | **694** | 14 | **15.1 s**（+ 粗格子 3.7 s） | 4.9e-6 |

- 総時間 42.3 s → 18.8 s（2.2×）。Newton の前半（低 CFL で残差を 1e-2 まで落とす区間）が丸ごと消える
- 注入が区分定数なので初期残差は参照の 0.5 止まり（補間誤差は格子 1 段分の O(h)）。双一次補間なら
  初期残差 0.1〜0.2、初期 CFL 3〜5 から始められる見込み
- (b) でも残差が 5e-3 → 3e-3 → 1e-2 と跳ねて CFL が 11.5 → 3.5 に落ちる区間があり、SER の経路敏感さ
  （status-39 §3.5）は残る。これは次の論点

## 5. 残件

- [ ] 粗格子 → 細格子の補間器（双一次）と 72×48 → 144×96 → 288×192 の入れ子反復ドライバを `nsb` に正式に置く
- [ ] SER の経路敏感さ: 残差が跳ねたときの CFL 減少率 0.1 と成長率 2 の非対称、`precond_cfl_ratio=2` で
      組み直しが CFL 倍化と同期している点。連続的な制御則の検討 — gyp さんと相談
- [ ] 前処理適用の numba 化（status-39 残件）
- master 由来の C3（BenchmarkRunnerProcess）は未修正（status-39 残件）

## テスト実行

```
python -m pytest tests/test_nsb_krylov.py tests/test_nsb_fastres.py tests/test_nsb_precond.py tests/test_nsb_linalg.py tests/test_nsb_standalone.py tests/test_nsb.py tests/test_nsb_adjoint.py -q -n 4
→ 69 passed（落とした切替のテスト 6 件を削除）
ruff check nsb/ tests/ experiments/nsb/ main.py && ruff format --check → All checks passed
```

## ファイル

- 変更: `nsb/{core,solver,precond,utils}.py`（設定の整理、Stokes 参照場、SER 出発則、`NSBResult.residual_ref`）、
  `nsb/README.md`（設定表を「一長一短」に書き換え）、`nsb/theory.md`、`main.py`（比較構成 dual / fixed / global）、
  `experiments/nsb/*.py`（廃止オプションの除去）、`tests/test_nsb_*.py`、`docs/status/status-40.md`、
  `README.md`、`docs/roadmap.md`、`docs/status/status-index.md`
