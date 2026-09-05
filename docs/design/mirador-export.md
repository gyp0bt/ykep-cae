# 3D レンダリング（messi mirador 連携）PostProcess 設計仕様

[← README](../../README.md) | [← 設計文書一覧](README.md) | [← roadmap](../roadmap.md)

## 概要

ykep の構造格子解析結果（`NaturalConvectionFDMProcess` / `HeatTransferFDMProcess` / `.inp` ジョブ出力）を
[messi](https://github.com/gyp0bt/messi) の three.js ビューア **mirador** で 3D レンダリングする。
セル値をそのまま messi の六面体メッシュ（C3D8）の要素場として載せ、ブラウザで回せる自己完結 HTML
（`<job>.html`）を書き出す。表示にはネット接続が要る（three.js を CDN から読む）。

```
ykep 結果（x/y/z 格子線 + セル場 U/P/T/…）
  └─ MiradorExportProcess ──► messi.Mesher（C3D8 + elset: domain / 断面スラブ）
                                 └─ mesh.export_html(fields, vectors) ──► <job>.html
```

messi 側の対応（同名ブランチ `claude/ykep-3d-rendering-mirador-bhyquf`、messi v0.10.0）:
`export_html` / `render_html` に **要素場のカラーマップ**（登録済み `element_fields` を自動で
モードボタン化、ベクトル場は `|U|` と `U_x/U_y/U_z` に展開）、**セル中心の矢印**（スライダーで長さ調整）、
`init_mode`（最初に表示する場）、`hidden_groups`（開いた時点で隠す elset）を追加。
あわせて VTK legacy（`.vtk` RECTILINEAR_GRID + CELL_DATA）リーダを追加したので、
ykep の `FORMAT=VTK` 出力は `messi mirador -i <job>.vtk` でもそのまま表示できる（断面なし）。

## プロセス情報

| 項目 | 値 |
|------|-----|
| クラス名 | `MiradorExportProcess` |
| カテゴリ | PostProcess |
| 入力 | `MiradorExportInput` |
| 出力 | `MiradorExportResult` |
| 安定性 | experimental |
| 依存 | messi（任意依存。未導入なら `MiradorUnavailableError`） |
| 配置 | `xkep_cae_fluid/post/mirador.py` |

## 断面（スラブ）の考え方

外皮だけを描くと内部の場が見えない。messi の mirador には「elset を隠すと、隣接する別 elset との
共有面が表示側の外表面として立ち上がる」機能（`interface_faces`）があるので、**1 セル厚の断面層を
別 elset** にしておけば、`domain`（外皮）を隠すだけで断面上の場が見える。断面の全内部面を
出力するより遥かに軽く、複数断面（x/y/z 中央面など）も elset の切替で見比べられる。

- `slices` 省略時: セル数 3 以上の各軸の中央層に 1 枚ずつ（`x=0.0458` のような elset 名）
- `SlicePlane(axis, position | index, name)` で任意位置。重なるセルは先に指定した断面に入る
- `hide_domain=True`（既定）: 断面があれば `domain` を開いた時点で非表示（凡例で戻せる）
- `mask`（`(nx,ny,nz)` bool）で固体・ガラスなどのセルを描画から外せる（水槽 CAE 用）

## 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| x_lines, y_lines, z_lines | np.ndarray | 格子線（昇順、長さ n+1）。z が 1 点の 2D 結果は面内セル幅の平均で 1 層押し出し |
| fields | Mapping[str, np.ndarray] | スカラー `(nx,ny,nz)`、ベクトル `(nx,ny,nz,3)`。2D の `(nx,ny)` / `(nx,ny,2)` も可 |
| output_path | str | 出力 HTML |
| title | str | ページタイトル |
| slices | tuple[SlicePlane, ...] | 断面スラブ（空 + `auto_slices` で中央断面） |
| auto_slices | bool | 中央断面の自動挿入（既定 True） |
| mask | np.ndarray \| None | 描画するセル（False は除外） |
| vector_field | str \| None | 矢印にするベクトル場（None: 最初のベクトル場、"" で矢印なし） |
| vector_scale | float \| None | 単位大きさ当たりの矢印長 [m]（None: 最大矢印 = セル代表長×1.5） |
| init_mode | str \| None | 最初に表示する場（None: T → P → 最初のスカラー → \|U\|） |
| hide_domain | bool | 断面があるとき外皮を初期非表示 |
| domain_name | str | 外皮 elset 名（既定 `domain`） |
| panel_collapsed | bool | 操作パネルを畳んだ状態で開く（`ykep view --collapse-panel`） |

## 出力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| path | str | 書き出した HTML |
| n_cells, n_nodes | int | 六面体数・節点数（mask 適用後） |
| n_triangles | int | HTML に埋め込まれた描画三角形数（外皮 + 断面の界面） |
| field_names | tuple[str, ...] | ビューアのモード名（`T`, `P`, `\|U\|`, `U_x`, …） |
| slice_names | tuple[str, ...] | 断面 elset 名 |
| n_vectors | int | 矢印数（= ベクトル場を持つセル数） |
| init_mode | str | 初期表示モード |

出力の統計は書き出した HTML の埋め込み JSON（`const DATA = {...}`）から読み戻しているので、
Python 側の値とビューアの表示が食い違わない（STA2 防止: 照合可能）。

## 使い方

```python
from xkep_cae_fluid.post import (
    MiradorExportInput, MiradorExportProcess, SlicePlane,
    fields_from_natural_convection, lines_from_structured_mesh, load_npz_fields,
)

# (a) ソルバー結果から
x, y, z = lines_from_structured_mesh(mesh_result)          # StructuredMeshResult
fields = fields_from_natural_convection(nc_result)          # {"U", "P", "T", 追加スカラー}
res = MiradorExportProcess().execute(MiradorExportInput(
    x, y, z, fields, "view.html", title="cavity",
    slices=(SlicePlane("z"), SlicePlane("x", position=0.05)),
))

# (b) ykep の NPZ から
x, y, z, fields = load_npz_fields("results/cavity-nc-1.npz")
MiradorExportProcess().execute(MiradorExportInput(x, y, z, fields, "results/cavity-nc-1.html"))
```

```bash
# (c) .inp から: *OUTPUT, FIELD, FORMAT=VTK+HTML で <job>.html を同時出力
#     （FORMAT= を書かなければ messi のある環境では自動で HTML も出る）
ykep -j=examples/inp/cavity-nc-1.inp int -o=examples/inp/results
# (d) 既に解いた NPZ を後から可視化（解析は走らない）
ykep -j=examples/inp/cavity-nc-1.inp view -o=examples/inp/results --slice=x=0.05 --slice=z=0.0125
```

`view` のオプション: `--slice=<axis>=<座標>`（複数可）、`--no-slices`（外皮のみ）、`--no-vectors`（矢印なし）。

## 残差マップとカラーマップ

- ソルバーが最終反復の**セル別残差**を `residual_fields` として返す（`NaturalConvectionResult`:
  `res_u / res_v / res_w / res_T` = |b − A x| / ‖b‖ の分布（L2 ノルムがスカラー残差と一致）、`res_mass` =
  Rhie-Chow 面速度による連続の式の不整合 [kg/m³/s]、追加スカラーは `res_phi_<name>`。`HeatTransferResult`: 定常のみ `res_T`）。
  `fields_from_natural_convection` / `.inp` ランナーはこれを場として含めるので、mirador のモードボタンに `res_*` が並ぶ。
  どこで収束が悪いか（角・境界層・ヒーター周り）を断面で確認できる
- カラーマップは messi 側で **Abaqus 既定の 12 段レインボー**が既定（`colormap="abaqus"`）。ビューアのセレクトで
  `Abaqus（連続）` / `Turbo 風（従来）` に切替。`MiradorExportInput` からは messi の既定に任せる

## ビューア側の操作

- 上段ボタン: `skewness / aspect / detJ`（メッシュ品質）、`|U| U_x U_y U_z P T res_* …`（解析場・残差マップ）、`elset / cluster`
- カラーマップのセレクト（Abaqus 12 段 / 連続 / Turbo 風）
- 凡例の elset 名クリックで表示切替（`domain` を戻すと外皮、断面 elset を隠すとその断面が消える）
- 「矢印 U」チェックとスライダー（×0.1〜×10）で速度矢印
- `probe` で要素をクリックすると品質と全ての場の値を表示
- 操作パネルはタイトル行の「▾/▸」ボタンか `h` キーで畳める（`panel_collapsed=True` で最初から畳む）

## テスト

`tests/test_post_mirador.py`（messi 未導入環境では skip）:

- `TestMiradorExportAPI`: 六面体メッシュ構築（ラベル順・接続・mask・断面 elset）、断面解決、
  HTML 出力と埋め込み JSON の整合（三角形数 = 外皮 + 界面、場の値がセル値と一致、矢印数）、
  2D 入力の押し出し、NPZ 読み戻し、例外（形状不一致・範囲外断面・未知のベクトル場）
- `TestInpOutputWriterAPI`（既存）: `FORMAT=HTML` で `<job>.html` が出ること

## 既知の制約 / TODO

- three.js は CDN（unpkg）から読むためオフラインでは表示できない（messi 側の仕様）
- 断面は軸に垂直な 1 セル厚のみ（任意平面の切断は未対応）
- 非構造格子（polyMesh 読込結果）は未対応。`MeshData` の connectivity から Mesher を組めば同じ経路で載る
- 時系列（`T_history`）はスナップショット 1 枚のみ。フレーム切替は messi 側の拡張が必要
