# nsb: 手元構成ミラーの Brinkman-NS 実験パッケージ

[<- README](../README.md) | [設計文書（共有離散化）](../docs/design/brinkman-flow-fvm.md) | [status-28](../docs/status/status-28.md)

手元の 2D FVM Brinkman 補正 Navier-Stokes コードと**同じファイル構成・同じ制御則**で比較するための薄いレイヤ。
離散化（残差、1 次風上ヤコビアン、Rhie–Chow、境界条件）は
`xkep_cae_fluid.brinkman_flow.assembly.BrinkmanDiscretization` を共有し、
Newton + 擬似時間の制御則だけを `solver.solve_steady` に関数として書き下している。

| ファイル | 役割 |
|---|---|
| `core.py` | 型宣言: `FaceType`, `BC`, `NSBSettings`, `NSBInput`, `NSBResult` |
| `solver.py` | メイン: `solve_steady`, `compute_dtau`, `solve_linear` |
| `utils.py` | ポスト処理、面値⇄セル値変換、要約、npz 保存 |
| `geo.py` | uturn / flat の厚さ場、左壁 inlet/outlet の BC プリセット、`run_uturn`, `run_flat`, `make_case` |
| `../main.py` | パラメータスタディ（構成 × モデル × 細分化 × 流速） |

## `NSBSettings` の「踏んではいけない線」スイッチ

| 設定 | 既定（手元構成の推定） | 修正構成 | 影響（status-28 の実験結果） |
|---|---|---|---|
| `local_dtau` | True（局所 Δτ） | True / False | 大域 Δτ は同じ CFL で減衰が約 10 倍強く、高 CFL に寛容 |
| `velocity_floor` [m/s] | 0（下限なし） | 0.1·U_in | 下限なしだと静止・低速セルで Δτ→∞ となり Newton が素になる。停滞の主因 |
| `pseudo_time_in_residual` | True（残差に ρV(u−u_prev)/Δτ） | False | 収束判定・SER が擬似時間項込みの残差で動く。u_prev の更新が正しければ定常解は同じ |
| `sub_iters` | 1 | 1 | 1 擬似時間ステップあたりの Newton 反復数（u_prev 凍結） |
| `rc_with_pseudo_time` | False | False | RC 係数 d_f に ρV/Δτ を含める。本実装では致命的ではない |
| `cfl_init` | 0.5 | 0.5 | 局所 Δτ では 5 で発散、0.5 なら可 |
| `alpha_u` | 0.7 | 1.0 | 速度下限ありなら緩和なしが最速（uturn 36 反復 vs 102）。下限なしでは効きがモデルごとに逆転 |
| `init_field` | "zero" | "stokes" | Stokes–Brinkman 初期場で反復 25〜35% 減。対流項込み残差で作ると max\|u\| が U_in の 14 倍になるので注意 |
| `reject_growth` / `cfl_min` | 0（無効） | 0 | CFL backtracking は効かず。Δτ→0 で圧力が発散するので cfl_min が要る |

## 使い方

```bash
python main.py --models uturn flat --refine 1 --u 0.1 1 2 --configs mine fixed \
    2>&1 | tee experiments/nsb/logs/main-$(date +%s).log
```

```python
from nsb import NSBSettings, run_uturn

inp, res = run_uturn(refine=1, u_in=2.0, settings=NSBSettings(velocity_floor=0.2))
print(res.converged, res.rel_residual, res.rel_steady_residual)
```

- テスト: `tests/test_nsb.py`
- 結果: `experiments/nsb/results/*.yaml`、ログ: `experiments/nsb/logs/`
