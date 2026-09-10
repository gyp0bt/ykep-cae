# 設計: 刳り抜きポート（nsb の領域内 inlet / outlet を実パッチにする）

[← README](../../../README.md) ｜ [status-49](../../status/status-49.md) ｜ [図解レポート §13](../../reports/nsb-trama-convergence.md)

2026-09-10。gyp さん「nsb の inner cell 境界条件を openfoam と合わせたい。outlet はセル内圧力固定境界を実装したい」。

## 決めたこと

| 論点 | 決定 | 理由 |
|---|---|---|
| ポートの表し方 | 円板セルを**刳り抜く**（未知数から外す）+ リング面に面 BC | OpenFOAM の `cylinderToCell` + `subsetMesh` と 1 対 1。レポート §12.6 が「ポートを実パッチに近づけない限り差は残る」と結論していた |
| 面種別のエンコード | `wall_x`（向き）と `pkind_x`（種別）の **2 配列** | 1 つの int8 に混ぜると「向きだけ見たい箇所」と「種別も見たい箇所」が混ざって読めない |
| outlet | リング面の**圧力 Dirichlet** | 従来 Robin `q = C(p−p_out)` の `C → ∞`。`max(q_c, 0)` の折れ（レポート §6）が消える |
| 逆流時の outlet | 4 辺と揃えて**ゼロ勾配** | OF は `inletOutlet` だが、折れを 1 つ消して別の折れを入れ直すのは筋が悪い |
| 既存 `INTERIOR_*` | **残す** | 紙面垂直方向のマニホールドという別の物理。`smooth_disk` の連続設計変数（status-31）が要る |
| 滑らかな `weight` | ポートでは**拒否** | 刳り抜きは離散的で、設計変数に対して滑らかでない |
| ポート半径 | 既定 `w/2 − 2Δx` | OF の `port_shrink_cells=2` と同じ。円周が閉塞域に接すると OF 側で圧力が跳ねる |
| 実装範囲 | nsb 本体 + 検算まで | nsbp / 随伴は別タスク（明示的に `NotImplementedError`） |

## 面 1 枚の扱い（4 辺の式をそのまま内部面へ）

| | 面速度 | 面圧力 | 速度の面勾配 | RC 係数 d_f | 対流面値 |
|---|---|---|---|---|---|
| 壁（status-48） | 0 | 流体側セル値 | ±2/d | 0 | （流束 0） |
| inlet | 法線に u_n | 流体側セル値 | ±2/d | 0 | 面値 |
| outlet | 流体側セル値 | 指定 p | 0 | 0 | 面値 |

`u_n = ṁ / (ρ Σ_f h_f A_f)`（Σ は実際に生えたリング面、h_f は流体側セルの厚さ）。
4 辺の `MASS_FLOW_INLET` と同一式で、OF の `flowRateInletVelocity`（`Q/(L_perim·tz)`、`tz = h`）と一致。

## `__init__` の順序

```
1. _resolve_boundaries      4 辺
2. _mark_port_cells   (新)  ポートセルを確定し active から落とす
3. _resolve_interior        体積マニホールド（ポートには配らない）
4. _resolve_solid           孤立塊の刈り込み → _build_face_masks
5. _resolve_port_faces (新) リング面の種別・u_n・指定圧力（u_n は実面積から）
```

孤立塊の刈り込みは「圧力基準に繋がらない塊を殺す」判定なので、outlet リング面を圧力基準、
inlet リング面を流入として登録する必要がある。

## 触った場所

`nsb/data.py`（`PORT_*` 種別）、`nsb/core.py`（`BC.port_inlet` / `port_outlet`）、
`nsb/assembly.py`（刳り抜き・面種別・面値・リミター・対流面値・拡散・J1 の 6 箇所・質量集計）、
`nsb/fastres.py`（numba カーネル）、`nsb/utils.py`（`inlet_cells`）、
`nsb/adjoint.py` / `nsbp/problem.py`（ガード）、`experiments/nsb/trama_case.py`（`--port carve`）、
`tests/test_nsb_ports.py`（新規 15 件）。

## 受け入れ試験

**J1 と色分け FD ヤコビアン（`nsb/fdjac.py`）の一致**（貫通項支配で RC 係数を速度非依存にした状態）。
面種別を足すときに触る 6 箇所（`Ux` / `Px` / `Fgx_vel` / `Wx` / `diff_diag` / `dfx`）のどれか 1 つでも
取りこぼすと必ず落ちるので、この作業の受け入れ試験としてほぼ完全。
