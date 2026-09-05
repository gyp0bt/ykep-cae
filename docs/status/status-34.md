# status-34: 3D レンダリング — messi mirador で ykep の解析結果を表示（`MiradorExportProcess`）

[<- README](../../README.md) | [<- status-index](status-index.md) | [設計文書](../design/mirador-export.md) | [前: status-33](status-33.md)

**日付**: 2026-09-05
**ブランチ**: `claude/ykep-3d-rendering-mirador-bhyquf`（ykep-cae / messi の両リポジトリで同名）
**テスト数**: 615 = 613 + 2（追記 6。本環境の `pytest --collect-only` は 599 件 + `tests/test_nsb*.py` 3 ファイルが pypardiso 未導入で収集不可。`tests/test_post_mirador.py` +11、`tests/test_inp_runner.py` +3、`tests/test_inp_parser.py` +1。本セッション環境で `pytest tests/ -m "not slow and not external"` の結果は本文末尾）
**契約違反**: 0 件（登録プロセス 27。+1: MiradorExport）

## 目的

ykep の構造格子の計算結果（速度 U・圧力 P・温度 T・追加スカラー）を、
[messi](https://github.com/gyp0bt/messi) の three.js ビューア **mirador** でブラウザ上で回して眺められるようにする。
両リポジトリを跨ぐ変更なので、messi 側も同名ブランチで拡張した（messi v0.10.0）。

## 実装

### ykep-cae 側（`xkep_cae_fluid/post/mirador.py`、Process）

| モジュール | Process / 関数 | 内容 |
|---|---|---|
| `post/mirador.py` | `MiradorExportProcess` | 格子線 + セル場 → messi `Mesher`（C3D8、i 最速のラベル）→ `export_html`。出力統計は HTML の埋め込み JSON から読み戻して照合 |
| 〃 | `build_structured_hex_mesh` / `resolve_slices` | messi 非依存の中間表現 `StructuredHexMesh`。`mask` で固体セルを除外、**断面スラブ（1 セル厚）を別 elset** に |
| 〃 | `fields_from_natural_convection` / `fields_from_heat_transfer` / `lines_from_structured_mesh` / `load_npz_fields` | ソルバー結果・`StructuredMeshResult`・`<job>.npz` からの入力アダプタ |
| `inp/output.py` | `InpOutputWriterProcess`（`uses = [MiradorExportProcess]`） | `*OUTPUT, FIELD, FORMAT=HTML`（`VTK+HTML` 併記可）で `<job>.html`。messi 未導入なら `RuntimeWarning` で他の出力は続行 |
| `inp/cli.py` | `ykep -j=<job> view [-o=dir] [--slice=x=0.05 ...] [--no-slices] [--no-vectors] [--collapse-panel]` | 解析せず既存の NPZ から HTML を生成 |

断面の見せ方: mirador の「elset を隠すと隣接 elset との共有面が外表面として立ち上がる」機能を使う。
既定で各軸中央（セル数 3 以上の軸）に 1 枚ずつスラブを置き、`domain`（外皮）を初期非表示にする
（`hidden_groups`）。凡例のクリックで外皮／各断面を切り替えられる。

### messi 側（同名ブランチ、v0.9.0 → v0.10.0、minor bump）

- `export_html` / `mirador`: `element_fields` の要素場を**着色モード**として自動追加（ベクトル場は `|U|`, `U_x/y/z` に展開）、
  **セル中心の矢印**（大きさで着色、スライダーで長さ ×0.1〜×10、深度テストなしで断面上に埋もれない）、
  `init_mode`（最初に出す場）、`hidden_groups`（初期非表示 elset）、`vector_field` / `vector_scale`
- `render_html`: `el_fields` / `vectors` / `vector_name` / `vector_scale` / `hidden_groups`。凡例の数値は桁で固定小数／指数を切替
- `messi.io.read_vtk`: VTK legacy（RECTILINEAR_GRID / STRUCTURED_POINTS / UNSTRUCTURED_GRID + CELL/POINT_DATA）リーダ。
  `messi mirador -i cavity-nc-1.vtk` で ykep の `FORMAT=VTK` 出力を場つきで表示できる
- テスト: `tests/test_viz.py` +6、`tests/test_io_vtk_legacy.py` +4。`docs/api_surface.txt` を 0.10.0 で更新、CHANGELOG 追記

## 動作確認（STA2 防止: ログ・スクリーンショットで確認）

| 項目 | コマンド / 内容 | 結果 |
|---|---|---|
| 例題再実行 | `ykep -j=examples/inp/cavity-nc-1.inp int -o=examples/inp/results`（`FORMAT=VTK+HTML` に変更） | 226 反復で収束（status-33 と同一）、`cavity-nc-1.html` 生成（外皮 + 断面 3 枚 = 2384 三角形、矢印 432、初期モード T）。YAML / log は `examples/inp/results/` |
| 後追い生成 | `ykep -j=examples/inp/cavity-nc-1.inp view -o=<dir> --slice=z=0.0125 --slice=x=0.05` | `VIEW: <dir>/cavity-nc-1.html` |
| VTK 経路 | `messi.Mesher.arrancar("cavity-nc-1.vtk").mirador(...)` | 場 `|U|,U_x,U_y,U_z,P,T`、T の範囲 [290.61, 309.39] が YAML の `temperature_range` と一致 |
| ブラウザ描画 | headless Chromium（three.js は npm tarball を unpkg の代わりに差し替え。本環境は CDN 不通） | JS エラーなし。T のカラーマップ、断面上の矢印が高温壁で上昇・低温壁で下降する循環を示す |

ブラウザ描画は本セッションの Playwright スクリプト（scratchpad、リポジトリ外）で確認した。
リポジトリにはスクリーンショットを入れていない（three.js は CDN 依存なので、表示にはネット接続が要る）。

## 追記（同セッション、レビュー反映）

1. **Abaqus レインボー**: messi 側でカラーマップを Abaqus 既定の 12 段レインボー（青→シアン→緑→黄→赤）に変更し、
   ビューアのセレクトで `Abaqus（12 段）/ Abaqus（連続）/ Turbo 風（従来）` を切替可能に（`colormap=` 引数）。
2. **自動 HTML 出力**: `.inp` の `*OUTPUT` に `FORMAT=` を書かなければ（`*OUTPUT` 自体が無くても）、messi が import
   できる環境では `<job>.html` を自動で書く（`OutputRequest.formats_explicit`。明示した FORMAT はそのまま尊重）。
3. **残差マップ**: `NaturalConvectionFDMProcess` が最終 SIMPLE 反復のセル別残差 `residual_fields`
   （`res_u / res_v / res_w / res_T`: |b − A x| / ‖b‖ の分布、`res_mass`: RC 面速度の連続不整合、`res_phi_<name>`）を返す
   （`assembly.compute_face_mass_residual_field` を新設し、既存のスカラー版はそのノルム）。`HeatTransferFDMProcess` は定常のみ
   `res_T`（疎行列系を組み直して評価）。`.inp` ランナーは残差マップを場に含め、`*ELEMENT OUTPUT` の `RES` で選択、
   mirador のモードに `res_*` が並ぶ。例題 `cavity-nc-1.inp` は `U, P, T, RES` に変更して再実行。
   テスト: `test_natural_convection.py` +1（マップのノルム = スカラー残差）、`test_heat_transfer_fdm.py` +1、
   `test_inp_runner.py` +1、`test_inp_parser.py` の HTML テストを拡張。
4. **操作パネルの畳み込み**: mirador 左上の操作パネルをタイトル行の「▾/▸」ボタンか `h` キーで畳めるようにした
   （messi `render_html` / `export_html` の `panel_collapsed=`、既定 False）。ykep 側は `MiradorExportInput.panel_collapsed`
   と `ykep view --collapse-panel` で渡す（True のときだけキーワードを渡すので、引数を知らない古い messi でも動く）。
   headless Chromium で「畳んで開く → `h` で展開 → ボタンで畳む」を確認（`shot-panel-collapsed.png` / `shot-panel-open.png`）。
   テスト: `test_post_mirador.py` +1、`test_inp_runner.py` の引数解釈テストを拡張、messi `test_viz.py` +1。
5. **CI 対応**: GitHub Actions の `test` ジョブ（messi 未導入）で `test_input_validation` が `MiradorUnavailableError` になっていた
   （入力検証の前に messi を import していた）。`export_mirador` で格子・場の検証と六面体構築を messi import より前に移し、
   messi が無くても形状不一致などは `ValueError` で返るようにした（`sys.modules["messi"] = None` で再現 → 修正後 18 passed / 8 skipped）。
   同ジョブの残り 9 件（`TestAMGSolverPhysics` / `TestNumbaSolverPhysics` の ImportError）は master でも同じく失敗している既存事象
   （pyamg / numba は `test-optional-deps` ジョブでのみ導入。そちらは success）。本 PR では触らず、下の TODO に残す。
6. **任意平面の断面（view cut）**: messi mirador に Abaqus の view cut 相当を追加した（同名ブランチ、v0.10.0 に含める）。
   `export_html(section=True)`（既定）でソリッド要素のコーナー結線を全セル分埋め込み、ビューアの「断面」チェック /
   `c` キーで任意平面 `n·x = d` のクリップ（three.js の `clippingPlanes`、外皮・ワイヤ・矢印・反則/整合漏れ・選択に適用）と、
   平面が横切る各セルの交差多角形（辺との交点を面内で角度順に並べて扇状三角形化）をセル値で着色した切り口を描く。
   法線は X/Y/Z か任意（3 成分）、位置はスライダー、「反転」で残す側を入れ替え。probe は切り口のセルも選択できる。
   凡例レンジは内部要素も含めて取る。`cut_plane=((nx,ny,nz), d)` / CLI `messi mirador --cut z=0.5` で最初から有効。
   ykep 側は `MiradorExportInput.cut_plane` と `ykep view --cut=<axis>=<pos> | <nx>,<ny>,<nz>,<d>`（指定時は中央スラブを
   自動挿入せず外皮も隠さない = 切った立体として見せる）。headless Chromium で 10×8×6 ブロック（T 場・2 elset）と
   `cavity-nc-1` の NPZ で確認: z 断面（切り口 160 三角形 = 10×8 セル × 2）、反転 + elset 非表示 + x 断面（48 = 8×3×2）、
   任意法線 (1,1,0.5)（三角形〜六角形の切り口）、`section=False`（クリップのみ）、`c` キーの on/off、切り口の probe
   （`shot-cut-on.png` / `shot-cut-flip.png` / `shot-cut-free.png` / `shot-cavity-cut.png`）。
   テスト: messi `test_viz.py` +6、`test_cli.py` +3、ykep `test_post_mirador.py` +2、`test_inp_runner.py` の引数解釈・view テストを拡張。
   CI の `test` ジョブ（messi 無し）で零法線テストが `MiradorUnavailableError` になったので、`cut_plane` の検証も
   messi import より前（`_normalize_cut_plane`）に移した（追記 5 と同じ型の修正。`sys.modules["messi"] = None` で再現 → `test_post_mirador.py` 単体で 7 passed / 6 skipped、messi ありで 13 passed）。

## 調査で分かったこと

- messi の `Mesher.load` は cwd に使用刻印 `.messi` を書く。ykep 側の `.gitignore` に追加した
- 矢印はセル中心（要素内部）にあるので、深度テスト有効だと断面スラブの上でも要素の面に隠れて見えなかった。
  messi 側で `depthTest: false` にして解決（外皮ごと見たいときはチェックで消す）
- messi の既存 `viz.py` / `viz_domain.py` には ruff の E501 / E741 が元から残っている（本変更では触っていない）

## テスト実行（本セッション環境: pyamg / numba / pypardiso 未導入）

```
python -m pytest tests/test_post_mirador.py tests/test_inp_runner.py tests/test_inp_parser.py -q → 54 passed
python contracts/validate_process_contracts.py → 契約違反なし（登録プロセス 27）
ruff check xkep_cae_fluid/ tests/ → All checks passed / ruff format --check → 全ファイル整形済み
python -m pytest tests/ -q -m "not slow and not external" --continue-on-collection-errors → 本文末尾
（messi）python -m pytest tests/ -q → 本文末尾
```

## 次にやること

表示まわり（messi 側の拡張が要るものは「messi」と付記）:

- [ ] 残差マップの**対数スケール表示**（messi）: `res_*` は桁がばらつくので線形カラーマップだと最大値の近傍しか
  色が付かない。凡例に log 切替を付け、0 以下は最小正値に丸める。`res_mass` は符号付きなので絶対値 + 符号の別表示
- [ ] `ykep view` に `--colormap=abaqus|abaqus-smooth|turbo` / `--init-mode=<場>` / `--title` を追加し、
  `MiradorExportInput` にも `colormap` を通す（いまは messi の既定に任せている）
- [ ] messi CLI `messi mirador` にも `--colormap` / `--collapse-panel` を出す（`docs/cli_surface.txt` の関所を通す）
- [ ] 操作パネルの開閉状態を `localStorage` に覚える（同じ HTML を開き直したとき前回の状態で開く。試験環境では
  `localStorage` が無効なことがあるので try/catch）
- [ ] `HeatTransferFDMProcess` の**過渡**でも `res_T` を返す（いまは定常のみ。陰解法の各ステップの最終残差を使う）
- [ ] 断面スラブの色に外皮の値が混ざらないよう、`domain` を隠したときの界面三角形の owner をスラブ側に固定する
  （いまは messi の `interface_faces` が「表示側」を owner にするので問題ないが、両方表示のときの規約を設計文書に明記）
- [x] 任意平面の切断 → 追記 6 の view cut で対応（`cut_plane` / `--cut`）
- [ ] view cut の切り口に**節点補間**（いまはセル値のフラット着色。セル中心値を節点へ平均して切り口内を補間すると滑らかになる）
- [ ] view cut の平面を**複数枚**（いまは 1 枚。three.js の `clippingPlanes` は複数可、切り口は平面ごとに作る）
- [ ] 時系列 `T_history` のフレーム切替（messi）
- [ ] 非構造格子（polyMesh 読込結果）を `MeshData.connectivity` から同じ経路で載せる
- [ ] 水槽 CAE（Phase 6）で `AquariumGeometryProcess` のマスクと連携した実例（水・ガラス・底床を elset 分け）
- [ ] 矢印の本数が多いとき（10^5 級）の間引きオプション（messi。`vector_stride=` で i/j/k 方向に飛ばす）
- [ ] CI: `test` ジョブ（pyamg / numba 無し）で `TestAMGSolverPhysics` / `TestNumbaSolverPhysics` が ImportError で落ち master が赤。
  `pytest.importorskip` で skip にするか、`-m` で `test-optional-deps` ジョブ側へ寄せる（status-32 以降ずっと赤）
- [ ] status-33 の残 TODO（`*DARCY` 実行、`HEAT TRANSFER=NONE`、SYMMETRY/SLIP の発散切り分け）は据え置き

## ファイル

- 追加: `xkep_cae_fluid/post/{__init__,mirador}.py`、`tests/test_post_mirador.py`、`docs/design/mirador-export.md`、`docs/status/status-34.md`
- 変更: `xkep_cae_fluid/inp/{case,builder,output,cli}.py`、`tests/test_inp_{runner,parser}.py`、`examples/inp/cavity-nc-1.inp`、
  `examples/inp/results/cavity-nc-1.{yaml,log}`、`.gitignore`、`README.md`、`docs/README.md`、`docs/design/README.md`、
  `docs/design/inp-format.md`、`docs/roadmap.md`、`docs/status/status-index.md`
- messi（同名ブランチ）: `src/messi/misc/viz.py`、`src/messi/mesh/mesh_base/viz_domain.py`、`src/messi/io/{vtk_legacy,__init__}.py`、
  `src/messi/mesh/__init__.py`、`tests/test_viz.py`、`tests/test_io_vtk_legacy.py`、`docs/api_surface.txt`、`pyproject.toml`、`CHANGELOG.md`、`README.md`

## 全体テスト

```
python -m pytest tests/ -q -m "not slow and not external" --continue-on-collection-errors
→ 564 passed / 10 failed / 18 deselected / 1 xfailed / 3 errors（1194 s）
```

失敗 10 件と収集エラー 3 件はすべて本セッション環境の任意依存の欠落によるもので、本変更とは無関係:
`tests/test_heat_transfer_fdm.py` の `TestAMGSolverPhysics`（5）/ `TestNumbaSolverPhysics`（4）は pyamg / numba 未導入
（status-32 / 33 と同じ）、`tests/test_nsb*.py` は pypardiso 未導入（`--continue-on-collection-errors` で残りを実行）。

messi 側（`python -m pytest tests/ -q`）: 失敗 20 件はすべて trimesh / triangle / scikit-learn / shapely
未導入（geomgen / pattern / mosaico 系）と root での刻印テストによるもので、main ブランチでも同一。
`tests/test_viz.py` / `tests/test_io_vtk_legacy.py` / `tests/test_api_surface.py` / `tests/test_cli_surface.py` は全て通過、
pyright（mesh スコープ）0 errors。
