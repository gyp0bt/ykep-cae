# status-47: OpenFOAM による独立検算 — trama 0.15 kg/s は「解けない」のではなく「定常解が無い」

[<- README](../../README.md) | [status-index](status-index.md) | [status-46](status-46.md) | [図解レポート §11](../reports/nsb-trama-convergence.md) | [roadmap](../roadmap.md)

日付: 2026-09-10 / ブランチ: `claude/nsbm-learned-init` / gyp さんの指示（07:44）
「どうにも上手くいきませんね、openfoam で同じ問題を解いてくれますか？」
+ 追加（07:48）「この環境に nsb と openfoam と比較したオラクルなかったでしたっけ？」（→ ゲート G3 の一式を転用）。

本文・図・数字は [docs/reports/nsb-trama-convergence.md §11](../reports/nsb-trama-convergence.md) にある。
ここには結論と成果物だけ書く。

## 1. 結論

**nsb の解法が悪かったのではない。0.15 kg/s ではこの問題に定常解が存在しない。**

| 問い | 答え |
|---|---|
| OpenFOAM なら定常で解けるか | 解けない。`simpleFoam` も 8000 反復で p 初期残差が 8.4e-2（`walls`）/ 1.1e-1（`porous`）に平坦に張り付く |
| 決めているのは何か | N = L_drag/w = ρuh²/(12μw) = ṁh/(12μw²)。渦が抗力で消えるまでの移動距離を流路幅で測った比。0.15 kg/s で **13.3** |
| 境目はどこか | N 1.33（0.015 kg/s）まで収束、N 2.22（0.025 kg/s）から床。**nsb の継続法の境目と一致** |
| Reynolds 数ではないのか | 違う。流量 0.15 のまま隙間を 12 分の 1 にすると Re_h 1449・Re_w 157895 のまま N が 1.11 に落ち、233 反復で収束する |
| N だけで決まるのか | 決まらない。Re_w を 658 に落とすと N 2.66 でも収束する。**N ≳ 2 かつ Re_w ≳ 2000** の両方が要る |
| 乱流モデルで定常解を作れるか | 作れない。必要な渦粘性 4.2e-4 m²/s は隙間乱流の目安 0.07 u* h = 2.0e-5 の 22 倍 |
| 0.15 kg/s の答えは | 非定常の時間平均。入口の必要圧力ヘッド **25.5 ± 2.3 kPa**（17.6〜29.8 kPa）、速度変動は時間平均流速の 45% |

## 2. nsb と OpenFOAM の一致

壁ポート・リミター凍結・定常残差 SER の継続法（`--continuation 0.0015,0.005,0.015,0.025,0.035,0.05`）:

| 段 [kg/s] | N | nsb | OpenFOAM |
|---|---|---|---|
| 0.0015 | 0.13 | 18 反復で収束 | 65 反復で収束 |
| 0.005 | 0.44 | 43 反復で収束 | 床 3.0e-6（実質収束） |
| 0.015 | 1.33 | 99 反復で収束 | 251 反復で収束 |
| 0.025 | 2.22 | **停滞 1.04e-3** | **床 3.1e-4** |
| 0.05 | 4.43 | 停滞 5.8e-2（status-46） | 床 2.2e-2 |
| 0.15 | 13.3 | 未収束（全手法） | 床 8.4e-2 |

離散化も線形化も前処理もポートの与え方も違う 2 つのコードが同じ流量で壁に当たる。

## 3. 費用の差は物理ではなく実装

| | 1 ステップ | 物理時間 3 秒ぶん |
|---|---|---|
| OpenFOAM `pimpleFoam`（24054 セル、4 コア、Δt 1 ms） | 0.20 s | **615 s** |
| nsb `solve_unsteady`（93200 セル、Δt 2.5 ms） | 25 s | 8.3 時間 |

nsb は (a) 死んだ閉塞セル 68312 個（全体の 73%）も一緒に解き、(b) 毎ステップ直接 LU を組み直す。
どちらも非定常計算では外せる。

## 4. 成果物

| 種別 | パス |
|---|---|
| OpenFOAM ケース生成（blockMesh + topoSet + subsetMesh + fvOptions） | `experiments/nsb/trama_of_case.py` |
| 実行ドライバ（Docker、メッシュ → simpleFoam / pimpleFoam） | `experiments/nsb/run_trama_of.py` |
| 流量掃引のまとめと図 f12 / f13 | `experiments/nsb/trama_of_sweep.py` |
| 場の載せ替えと図 f10 / f11 | `experiments/nsb/trama_of_compare.py` |
| 非定常の時間平均・変動・入口圧の図 f14 | `experiments/nsb/trama_of_transient.py` |
| nsb と OpenFOAM の場の突き合わせと図 f15 | `experiments/nsb/trama_of_verify.py` |
| Foam フィールド読み書き（G3 から再利用、symmTensor 対応を追加） | `experiments/extruder/foam_io.py` |
| 走行結果 | `experiments/nsb/results/trama_of_*.json` |
| ログ | `experiments/nsb/logs/of-trama-*.log`, `trama-N-wall-cont-fine-*.log` |

## 5. 落とし穴（同じ轍を踏まないために）

- `subsetMesh -patch <name>` は露出面を **empty 型**のパッチで作る。直さないと `polyMesh::calcDirections` が
  x, y まで「空」と判定し、`checkMesh` が "0 geometric directions" と言う（2 次元解が立たない）。
- `subsetMesh` は時刻ディレクトリのフィールドも部分集合に写して書き戻す。初期場は `0.orig` に置く。
  古い時刻ディレクトリが残っていると旧格子のサイズで落ちる。
- ポートの刳り抜き半径を流路幅の半分ちょうどにすると、円周が流路と閉塞域の境目に乗る。`porous` 変種では
  入口の外周が栓に接し、そこへ流量を押し込んで入口圧が 16.7 kPa → 1.51 MPa に化ける。2 セル縮める。
- `explicitPorositySource` は `selectionMode` を `explicitPorositySourceCoeffs` の**中**に書く。`all` は使えず
  `cellZone` が要る。名前に反して `eqn -= porosityEqn` と行列ごと引くので抗力は対角に陰的に入る。
- `volFieldValue` に `maxMag` 演算は無い。`mag` 関数オブジェクトで `magU` を作ってから `max` を取る。
- `bash -c "cmd | tee log"` は tee の終了状態を返すので、`set -o pipefail` が無いと失敗が成功に見える。

## 6. 既知の未解決

- `tests/test_nsbm_residual_loss.py::test_unroll_matches_solve_steady_history` は status-46 から続く既知の失敗
  （unroll は J1+τ の厳密解、`solve_steady` は真の J+τ。履歴が 40% ずれる）。nsb 一式は 129 件通過。
- nsb の非定常 G4（0.15 kg/s、Δx 1.5 mm）は t = 0.21 s まで進んで走行中。OpenFOAM の結果からは
  定常化せず振動が続くはずで、時間平均を取るべきもの。
