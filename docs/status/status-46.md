# status-46: nsb 蛇行流路 — リミター凍結・隙間の摩擦則・物理時間の非定常を入れて反復数を測る（0.15 kg/s は未収束）

[<- README](../../README.md) | [status-index](status-index.md) | [status-45](status-45.md) | [図解レポート §10](../reports/nsb-trama-convergence.md#10-リミター凍結隙間の摩擦則物理時間の非定常2026-09-10-朝gyp-さんの追加依頼) | [roadmap](../roadmap.md)

日付: 2026-09-10 / ブランチ: `claude/nsbm-learned-init` / gyp さんの指示「リミター凍結、隙間の摩擦則追加、非定常計算トライをお願いします」
+「今回渡している題材はベストプラクティスで結局収束していますか？」+ 手元の知見「リミター凍結は残差後半に入れるとすっと収束するが序盤に入れたら爆発する」。

本文・図・数字は [報告書 §10](../reports/nsb-trama-convergence.md)。ここには結論・改修・成果物・既知の問題だけ書く。

## 1. 結論（gyp さんの問いへの答え）

**0.15 kg/s（Re_h 1450）は収束していない。** 到達点:

| 到達した流量 | 方法 | 反復数 |
|---|---|---|
| 内部ポート 0.005 kg/s（Re_h 48） | 継続法 + 定常残差 SER + リミター凍結（patience 3） | 46 + 69（0.015 段で停滞 rel 2.7e-3） |
| 壁ポート 0.015 kg/s（Re_h 145） | 同上 | 18 + 43 + 99（0.05 段で停滞 rel 5.8e-2） |
| 0.15 kg/s | 直接解法・継続法・摩擦則: 未収束。物理時間の非定常（Δx 1.5 mm）: 過渡を追跡中（t = 87 ms で定常残差比 1.5e-2、1 ステップ 25 s） | — |

| 対策 | 効き方 |
|---|---|
| リミター凍結 | 継続法を 1 段先まで登らせる（内部 0.005、壁 0.015 が初めて収束）。効くのは「残差比 1e-3 を切った後」で、次の段の停滞は 3e-3〜6e-2 で起きるので届かない。解凍は 1 反復の跳ねで即やると凍結↔解凍を往復して育たない（patience 3 が要）。ラインサーチとの併用は逆効果（棄却の連鎖で CFL 1e-60） |
| 隙間の摩擦則（Blasius 型） | Re_2h 2900 で抗力 1.3 倍。慣性/抗力比 11.6 のままで収束性は悪化（0.0015 段 46 → 116、0.005 段で停滞）。物理補正としては正しいが収束対策ではない |
| 物理時間の非定常 | 細格子は Stokes 場から Δt 7.8e-5 → 5 ms に育てながら過渡を追える（Δt 後退が必須）。粗格子 Δx 3 mm は初手でラインサーチが降下する α を見つけられず打ち切り。定常解の存在は結果待ち（L/U ≈ 1.4 s を 5 ms で刻むと 1 周 2 時間） |

## 2. 改修（nsb）

| 改修 | 場所 |
|---|---|
| リミター凍結 | `BrinkmanDiscretization.limiter()` / `compute_state(psi=)` / `residual_fast(psi=)`、numba 核に凍結 ψ を渡す口。`NSBSettings.limiter_freeze_rel / limiter_freeze_delta / limiter_freeze_stable_frac / limiter_unfreeze_ratio / limiter_unfreeze_patience / limiter_refreeze_max`。`NSBResult.limiter_frozen_at / residual_unfrozen`（凍結問題の解の真の残差を必ず報告） |
| 隙間の摩擦則 | `BrinkmanDiscretization.drag_factor()`（(1 + (Re_Dh/Re_c)^(0.75 m))^(1/m)）、`StateArrays.drag_fac`、`NSBInput.friction_re_crit / friction_exponent / friction_blend`、J1 の抗力対角にも反映 |
| 物理時間の非定常 | `nsb/unsteady.py::solve_unsteady`（後退 Euler、各ステップ Newton、Δt 後退、プローブ・スナップショット）、`stokes_reference()` を `solver.py` から切り出し |
| **既定の切替** | `NSBSettings.pseudo_time_in_residual` True → **False**（定常残差で SER・収束判定）。理由は §4 |
| ケースドライバ | `trama_case.py --freeze --refreeze --friction --unsteady --n-steps --newton-per-step --save-every --step-tol`、継続法の段ごとに残差履歴・凍結反復・解凍残差を yaml に保存 |
| 図 | `trama_plots2.py` → `results/trama_figs/f8_continuation_freeze.png`（継続法の残差履歴 + 凍結/解凍）・`f9_unsteady.png`（非定常の時系列） |
| テスト（nsb 一式 89 件通過。nsbm residual_loss 1 件は上記の既知） | `tests/test_nsb_freeze_friction.py`（凍結 ψ の numba/numpy 一致と連続性、uturn の凍結収束と解凍残差、Picard 再凍結、摩擦則の極限値と J1 整合、uturn の後退 Euler が定常 Newton の解に一致） |

## 3. 落とし穴（今回踏んだもの）

1. **fd ヤコビアンは SIMPLE 前処理と組まない。** 最初の 6 走行を fd + SIMPLE で出して 0.0015 段の初手から線形解が壊れ（真の残差比 5〜170）、
   全部止めて出し直した。SIMPLE の Schur 近似は J1 の構造前提。fd は PARDISO LU（`jfnk`）と組む。docstring に明記。
2. **fd + PARDISO でも GMRES が 200 反復で落ちるステップがある**（uturn r1 でも 11, 208, 21, 215）。FD ヤコビアンの列差分と
   JFNK の方向差分が折れ点の反対側を見ると別の作用素になる。非定常には J1 の LU を使った。
3. **ラインサーチは蛇行流路では逆効果**（定常残差が Newton 方向に単調でない）。参照ケース以外で使わない。

## 4. τ 修正の副作用（status-45 の訂正）

status-45 の τ 2 重カウント修正（作用素 J+2τ → J+τ）で **既存テスト 7 件が壊れていた**（`test_nsb_nested` 2・`test_nsb_precond` 1・
`test_nsb_linalg` 2・`test_nsb_ser` 2。5d8bd16 では通り、4fa445a で落ちる。status-45 の「37 件通過」は部分集合）。

- `test_nsb_ser` 2 件は monkeypatch の `fake()` が新しいキーワード引数（`fd_diag`, `steady_resid_fn`）を受けないだけ → テスト修正。
- 残り 5 件は **flat r1/r2 U=1 が既定設定で収束しなくなっていた**（80 反復で定常 rel 9.1e-3、|R_τ| 3.1e-6、CFL 0.05〜0.9）。
  機構: 作用素と前処理が揃った結果、各 Newton ステップは擬似時間残差 R_τ = R + τδ をほぼゼロにする。R_τ を見る既定の SER は
  「進んだ」と誤認して CFL を育てず、定常残差は前進 Euler 型の時間発展で這う（status-45 の「0.0015 が 22 反復で収束」も
  定常残差は 5.5e-5 の偽収束だった）。τ 修正前は作用素が J+2τ で R_τ が半分残り、たまたま SER が働いていた。
- 未解決 1 件: `test_nsbm_residual_loss::test_unroll_matches_solve_steady_history`（5d8bd16 で通り、τ 修正後に落ちる）。
  nsbm の unroll は (J1+τ) の厳密解で歩き、solve_steady は真の作用素 J+τ（JFNK）で歩くので履歴が 40% ずれる
  （τ 修正前は J+2τ の減衰で偶然 25% 以内だった）。nsbm は否定的結論で閉じた実験なので unroll 側は直していない
  （`residual_loss` は擬似時間残差形 `pseudo_time_in_residual=True` を明示するようにだけ直した）。
- 対処: **既定を定常残差駆動（`pseudo_time_in_residual=False`）に切替。** flat U=1 15 反復、uturn U=1 21（凍結あり 14）で収束。
  True は非定常向けの残差形として残す。

## 5. 成果物

| 種別 | パス |
|---|---|
| 走行ログ（tee） | `experiments/nsb/logs/trama-{A2,A3,A4,A5,B2,B3,C2,E2,F2,F3,F4,G2,G3,G4}-*.log`（gitignore） |
| 結果 yaml / npz | `experiments/nsb/results/trama_{A3-int-cont-sser-frz-pat3,B2-wall-cont-sser-frz,…}.yaml`（npz は未コミット） |
| 図 | `experiments/nsb/results/trama_figs/f8_continuation_freeze.png`, `f9_unsteady.png` |
| 走行コマンド | `scratchpad/launch_batch{2,3,4}.sh` の内容は報告書 §10 付録に転記 |

## 6. 次にやること（roadmap に転記）

1. G4（非定常 Δx 1.5 mm）を最後まで走らせ、定常化するか振動するかを判定。定常化するなら「非定常で吸引域まで運んでから定常 Newton + 凍結」を試す。
2. 停滞段（内部 0.015、壁 0.05）の跳ねの正体（風上切替か sink の clip か）をステップごとに分解して同定。§9.3 の解剖と同じ手順。
3. 粗格子 Δx 3 mm の非定常が初手で降下しない理由（sink 円板が半径 6 セルしかない）。
4. τ 修正後の SER 制御則の再掃引（status-41 の掃引は J+2τ の作用素で行われていた）。
