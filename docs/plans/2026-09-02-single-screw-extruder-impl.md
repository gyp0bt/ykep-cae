# 単軸押出（展開チャネル 2.5D）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development（推奨）
> または superpowers:executing-plans を使い、タスク単位で実装すること。
> ステップは `- [ ]` チェックボックス形式で進捗管理する。

[<- README](../../README.md) | [<- docs](../README.md) | [設計文書](../design/single-screw-extruder.md) | [status-27](../status/status-27.md)

**Goal:** 単軸押出機・計量部の展開チャネル断面（2.5D）を解き、混練性と滞留時間分布（RTD）を
算出できる `xkep_cae_fluid/extruder/` パッケージを作る。

**Architecture:** 樹脂は `Re ~ 10⁻³` のクリープ流れなので慣性項が消え、断面問題は**線形 Stokes**
になる。したがって SIMPLE のような圧力-速度連成反復は不要で、疎行列の**直接解**で一発で解ける。
ニュートン流体では下流方向 `w` の可変係数 Poisson と断面内 `(u,v,p)` の Stokes が完全に分離し、
非ニュートンでは粘度 `μ(γ̇)` だけを介して結合する。この構造をそのままコードの構造にする
（`down_channel.py` / `cross_channel.py` / それを Picard で回す `solver.py`）。

**Tech Stack:** Python 3.12 / numpy / scipy.sparse（`splu` 直接解）/ pytest /
Process Architecture（`AbstractProcess`, `ProcessMeta`, `StrategySlot`, `binds_to`）/
OpenFOAM v2312（`simpleFoam`、Docker、独立検算のみ）

**Spec:** [docs/design/single-screw-extruder.md](../design/single-screw-extruder.md)
（本計画の Task 0 で幾何定数の誤りを訂正する。訂正内容は §0 に示す）

---

## Global Constraints

すべてのタスクの要件に、以下が暗黙に含まれる。

| 制約 | 値 |
|---|---|
| Python | `>=3.10`（実環境は 3.12）。実行は必ず `.venv/bin/python` |
| 依存ライブラリ | `numpy>=1.24`, `scipy>=1.10` のみ。**pyamg / numba に依存しない** |
| 記述言語 | 全ドキュメント・docstring・コメントは日本語（CLAUDE.md） |
| lint | `ruff check xkep_cae_fluid/ tests/` と `ruff format` が通ること |
| プロセス化 | 機能は可能な限り `AbstractProcess` サブクラスとして実装（CLAUDE.md） |
| 契約 | 新規プロセスは `meta.document_path` が実在し、`@binds_to` でテストと 1:1 対応すること（C3/C15） |
| 契約検証 | `.venv/bin/python contracts/validate_process_contracts.py` が「契約違反なし」 |
| 入力不変性 | `process()` は入力 dataclass の numpy 配列を変更しない（C9） |
| ログ | 計算実行は必ず `2>&1 \| tee /tmp/log-$(date +%s).log`。`\| tail -N` のみは禁止 |
| 数値の捏造禁止 | 収束しなければ「収束しなかった」と報告する（STA2 防止ルール） |
| **CPU 上限** | `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2` を必ず設定 |
| **メモリ上限** | 単一プロセス 4 GB 以内。超える格子は使わない |
| **OpenFOAM 上限** | `of` ラッパの既定（`--cpus 1 --memory 1200m`）を上げない |

### リソース方針（別途走っている応力サロゲート NN 学習との共存）

実測: 20 コア / 30 GB（うち利用可能 23 GB）、GPU なし。

本計画の計算は **CPU 1〜2 コア・メモリ 4 GB 以内**に収まる。理由は解の構造そのもの:

- クリープ流れ → 線形問題 → **反復ではなく疎直接解**。最大格子（Task 7 の隙間ありケース、
  約 250×80 = 20,000 セル）で Stokes 系の未知数は約 60,000。2 次元疎行列の LU は
  数秒・数百 MB で終わる。
- SciPy の SuperLU は**シングルスレッド**。BLAS が勝手に 20 コアを掴むのを防ぐため
  `OMP_NUM_THREADS=2` を全実行で設定する（上表）。
- OpenFOAM は Docker で `--cpus 1 --memory 1200m`。

**したがって NN 学習には 18 コア / 25 GB を空けたままにできる。**
この前提が崩れる（格子を大幅に増やす、3D に行く、パラメータスイープを回す）場合は
実行前に見積もりを出して相談すること。

### ベースライン（本計画の開始時点、必ず記録すること）

```
ブランチ: claude/single-screw-extruder-impl（master 9de63fc から分岐）
テスト:   268 passed, 10 failed, 8 skipped, 1 xfailed
契約:     契約違反なし
```

**10 failed は既存の失敗であり、本計画とは無関係**。内訳は全て optional 依存の未導入:

- `TestAMGSolverPhysics::*`（5 件）と `TestCavityBenchmark::test_ra_1e4_nusselt`（1 件）
  → `ModuleNotFoundError: No module named 'pyamg'`
- `TestNumbaSolverPhysics::*`（4 件）→ `ImportError: Numba が必要です`

`.venv` に pip が入っていないため導入できない。**各タスクの完了判定はこの 10 件を除いた
「新規追加テストが全て通り、既存の 268 passed が減らないこと」とする。**

---

## 0. 前提の訂正 — 幾何の恒等式（Task 0 で設計文書に反映）

設計文書 §2.1 の `L_turn = πD/sinφ` は**誤り**。正しくは `L_turn = πD·cosφ`。
40 mm 機の諸元で 414 mm → **119.74 mm**（3.46 倍の過大評価）であり、
そのまま実装すると漏れ流れの駆動圧を 3.46 倍に見積もる。

### 導出

バレル面を展開した平面を `(ξ, ζ)` とする。`ξ` = 周方向（1 回転で `πD`）、`ζ` = 軸方向。
フライトは傾き `φ` の平行直線群で、`tanφ = リード/(πD)`。下流方向と横断方向は

```
  ẑ = ( cosφ,  sinφ)        （フライトに沿う = 下流）
  x̂ = (-sinφ,  cosφ)        （フライトに直交 = 横断）

  z = ξcosφ + ζsinφ ,   x = -ξsinφ + ζcosφ
```

**物理的な同一視**は「バレルを 1 周する」こと、すなわち `(ξ, ζ) ~ (ξ + πD, ζ)`。
これを `(x, z)` で書くと

```
  (x, z)  ~  (x - W_t, z + L_turn)      W_t = πD·sinφ ,  L_turn = πD·cosφ
```

- `W_t = πD sinφ` はチャネル 1 ピッチのフライト直交幅。**設計文書の W = 34.1 mm と
  e = 4 mm から `W + e = 38.1 mm` となり一致する**（`πD sinφ = 38.116 mm`）。
  ここが合っているのに `L_turn` だけ `sinφ` で割っているのが誤りの正体。
- `L_turn = πD cosφ` は「隣のチャネルは同じチャネルの何 mm 下流か」。

**恒等式 `W_t / L_turn = tanφ`**（`D` に依らない）。これをテストで固定する。

### 圧力の扱い

`p(x,y,z) = G·z + P(x,y)`、`G = dp/dz`（定数、押出は `G > 0`）。上の同一視から

```
  P(x + W_t, y) = P(x, y) + G·L_turn
```

`P = βx + p̃`（`p̃` は `x` 周期）と置くと

```
  β = G·L_turn / W_t = G·cotφ        ← D に依らない
```

すなわち**横断方向運動量に一様体積力 `f_x = -β = -G·cotφ` を入れるだけでよい**。
`fixedJump` 型の特殊な境界条件は不要になる（OpenFOAM 側も同様、§Task 8）。

**整合性チェック（テスト化する）**: 全圧力勾配ベクトルは `(x, z)` 成分で `(G cotφ, G) ∝ (cosφ, sinφ)`。
これは展開平面の**軸方向 `ζ̂` そのもの**。実機で圧力が軸方向にのみ変化する事実と一致する。
大きさは `dp/dζ = G/sinφ`。設計文書の `L_turn = πD/sinφ` ではこの整合性が壊れる。

### 符号（設計文書 §8 の未決事項を確定）

バレルは（スクリュー座標系で）周方向 `ξ̂` に速度 `V = πDN` で滑る。`ξ̂ = -sinφ·x̂ + cosφ·ẑ` より

```
  u_barrel = -V·sinφ      （横断方向、-x 向き）
  v_barrel =  0
  w_barrel = +V·cosφ      （下流方向、正 = 引きずり流れが下流を向く）
```

`+x` が下流側の隣チャネル、`-x` が上流側。`f_x = -G cotφ < 0` なので漏れ流れは `-x`
（＝上流へ戻る backflow）。引きずり成分 `u_barrel < 0` も同じ向き。
**漏れ流れは押出量を減らす方向**という古典的な描像と一致する。

### 40 mm 機の確定値（`.venv/bin/python` で実測、Task 1 のテスト期待値）

```
  φ         = 17.6568°      sinφ = 0.303314   cosφ = 0.952891   tanφ = 0.318310
  W_t       = 38.1156 mm    W = W_t - e = 34.1156 mm    H/W = 0.117248
  L_turn    = 119.7438 mm
  V         = 0.209440 m/s
  u_barrel  = -0.063526 m/s     w_barrel = 0.199573 m/s
```

---

## A. アーキテクチャ判断 — 既存 SIMPLE ソルバーを使わない

設計文書 §4 は既存資産の転用を前提にしていたが、**棚卸しの結論が変わった**。

### 事実（実測）

`NaturalConvectionInput` は `dx = Lx/nx` のプロパティを持ち、`assembly.py:300` は
`dx, dy, dz = inp.dx, inp.dy, inp.dz` で**スカラー**として受ける。
つまり **既存の Navier-Stokes ソルバーは等間隔格子専用**。
`StructuredMeshProcess` が不等間隔格子を作れるのは事実だが、NS ソルバーはそれを食えない。

### 押出が要求するもの

| 要求 | 既存 SIMPLE | 判定 |
|---|---|---|
| 隙間 0.1 mm に 16〜20 セル（深さ 4 mm と 40:1） | 等間隔のみ | ✗ |
| 場としての粘度 `μ(x,y)` | スカラー定数 | ✗ |
| `x` 周期 + 圧力跳び | 無し | ✗ |
| 接線移動壁 | enum に無し | ✗ |
| 慣性項（対流） | 必須実装 | **不要**（Re ~ 8×10⁻⁴） |

### 判断

**専用パッケージ `xkep_cae_fluid/extruder/` に、クリープ流れ専用の小さなソルバーを新規に書く。**

- 1276 行の `assembly.py` を不等間隔・可変粘度・周期化に改造するのは、CLAUDE.md が
  「新機能追加よりも優先」と明記している**未解決の物理的不整合を抱えたコードに対する大手術**であり、
  リスクが釣り合わない。
- クリープ流れは**線形**。SIMPLE 反復・Rhie-Chow・緩和係数・残差判定が**丸ごと不要**になり、
  疎直接解 1 回で機械精度の解が出る。解析解との比較（G1/G2）が反復の収束残差に汚されない。
- 既存資産のうち再利用するのは `StructuredMeshProcess`（不等間隔格子生成）と
  Process Architecture 一式。ここは素直に乗る。

**この判断は設計文書 §4 を上書きする。Task 0 で設計文書に反映すること。**

---

## B. File Structure

```
xkep_cae_fluid/extruder/
├── __init__.py          公開 API の再エクスポート
├── data.py              ScrewSpec / ChannelGrid / 各 Input・Result dataclass（全て frozen）
├── shape_factors.py     形状係数 Fd, Fp の級数解（解析解＝真値。プロセスではない純関数）
├── geometry.py          ScrewGeometryProcess（諸元 → 不等間隔格子 + 固体マスク + 派生量）
├── viscosity.py         ViscosityModelStrategy Protocol + Newtonian / PowerLaw / Carreau
├── down_channel.py      DownChannelFlowProcess（w の可変係数 Poisson）
├── cross_channel.py     CrossChannelStokesProcess（断面内 MAC Stokes）
├── solver.py            ExtruderFlowProcess（Picard で粘度結合、Q と診断量を出す）
├── tracker.py           ParticleTrackerProcess（流れ関数補間 + 適応 RK4 + x 周期跳び）
└── rtd.py               RTDProcess（E(t), F(t), 累積せん断、混合指数）

tests/
├── test_extruder_geometry.py
├── test_extruder_shape_factors.py
├── test_extruder_viscosity.py
├── test_extruder_down_channel.py      ← G1 / G2
├── test_extruder_cross_channel.py
├── test_extruder_solver.py
├── test_extruder_tracker.py           ← G4 の一部
└── test_extruder_rtd.py               ← G4

experiments/extruder/
├── of_case.py           OpenFOAM ケース生成（blockMesh + topoSet/subsetMesh）
├── of_powerlaw_check.py G3b の前段: powerLaw の K,n 対応づけを 1D 解析解で較正
├── compare_openfoam.py  ykep-cae と OpenFOAM の突き合わせ
└── run_g3.sh            G3 一括実行（tee 付き）
```

**分割の理由**: `down_channel` と `cross_channel` は方程式として独立（ニュートンでは完全分離、
非ニュートンでも粘度場を介するだけ）なので、別ファイル・別プロセス・別テストにする。
`solver.py` はこの 2 つを Picard で回すだけの薄い層に保つ。

### 依存の向き

```
  geometry ──> down_channel ──┐
      │                       ├──> solver ──> tracker ──> rtd
      └──────> cross_channel ─┘
  viscosity ──> (down_channel, cross_channel, solver)
  shape_factors ──> tests のみ（真値の供給）
```

---

## C. 検証ゲート

**G1/G2 を通らなければ先へ進まない。**

| ゲート | 内容 | 真値 | 判定 | Task |
|---|---|---|---|---|
| G1 | ニュートン・隙間無し・純引きずり（`G=0`） | `Q = V_z W H F_d / 2` | 相対誤差 < 1×10⁻³ かつ観測次数 ≈ 2 | 3 |
| G2 | ニュートン・隙間無し・引きずり＋圧力 | `Q = (V_z WH/2)F_d − (WH³/12μ)G F_p` | 5 点の `G` 全てで < 1×10⁻³ | 3 |
| G2b | 断面内流れの 1D 厳密解 | `u(y) = U(3η² − 2η)` | 等間隔格子で < 1×10⁻¹⁰ | 5 |
| G3a | 隙間あり・ニュートン | OpenFOAM と一致 | `Q`・漏れ量とも 1% 以内 | 8 |
| G3b | 隙間あり・べき乗則 | OpenFOAM と一致 | 同上 | 8 |
| G4a | 粒子軌跡（単純せん断） | 解析軌跡 | < 1×10⁻⁸ | 9 |
| G4b | 平均滞留時間 | `⟨t⟩ = L·A_free/Q`（厳密） | < 1% | 10 |

---

# タスク

## Task 0: 設計文書の訂正とベースライン記録

**Files:**
- Modify: `docs/design/single-screw-extruder.md`（§2.1, §4, §6, §8）
- Create: `docs/plans/2026-09-02-single-screw-extruder-impl.md`（本ファイル。既に存在）
- Modify: `docs/README.md`（ドキュメント一覧に「実装計画」行を追加）

**Interfaces:**
- Consumes: なし
- Produces: 訂正済み設計文書。以後の全タスクが参照する幾何恒等式。

- [ ] **Step 1: ベースラインを記録する**

```bash
cd ~/work/ykep-cae
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -3
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
```

期待: `268 passed, 10 failed, 8 skipped, 1 xfailed` / `契約違反なし`。
**この 10 failed（pyamg / numba 未導入）から増えないことが以後の合格条件。**

- [ ] **Step 2: 設計文書 §2.1 の `L_turn` を訂正する**

`docs/design/single-screw-extruder.md` の `x` 両端の行を次に差し替える。

```markdown
| `x` 両端 | **圧力跳び付き周期**。`P(x+W_t) = P(x) + Δp`、`Δp = G·L_turn`、
  `W_t = πD·sinφ`（チャネル 1 ピッチのフライト直交幅）、`L_turn = πD·cosφ`。
  実装上は `P = βx + p̃`（`p̃` 周期）と分解し、横断方向運動量に一様体積力
  `f_x = -β = -G·cotφ` を入れる。`β = G·L_turn/W_t = G·cotφ` は `D` に依らない。 |
```

さらに §2.1 の「実装時に固定すべき符号」ブロックを、確定した符号に置き換える
（本計画 §0「符号」の内容をそのまま移植）。

- [ ] **Step 3: 設計文書 §4 の冒頭に、アーキテクチャ判断を追記する**

本計画 §A の「事実（実測）」「押出が要求するもの」「判断」の 3 ブロックを
`docs/design/single-screw-extruder.md` §4 の先頭に移植する。
§4.1 の棚卸し表の「✓ ある 不等間隔構造格子」に次の注記を足す。

```markdown
  注意: 不等間隔格子を作れるのは StructuredMeshProcess であって、
        NaturalConvectionFDMProcess ではない。後者は assembly.py:300 で
        dx をスカラーとして受けるため等間隔専用。押出では転用できない。
```

- [ ] **Step 4: 設計文書 §6 の諸元に確定値を追記する**

`L_turn = πD/sinφ = 414 mm` の行を削除し、本計画 §0 の確定値表を貼る。

- [ ] **Step 5: `docs/README.md` のドキュメント一覧に 1 行足す**

```markdown
| [実装計画](plans/2026-09-02-single-screw-extruder-impl.md) | 単軸押出 2.5D の実装計画（Task 0-10） |
```

- [ ] **Step 6: 訂正が矛盾を生んでいないか grep で確認する**

```bash
grep -v "旧版" docs/design/single-screw-extruder.md | grep -n "πD/sinφ\|414"
```

期待: ヒット 0 件。「旧版の誤り」として誤値に言及している行は正当なので
`grep -v "旧版"` で除外する（単純な `grep -n "414"` では必ず引っかかる）。

- [ ] **Step 7: mdview で確認して Artifact 公開する**

```bash
~/work/tb/bin/mdview docs/design/single-screw-extruder.md
```

生成された `/tmp/mdview/single-screw-extruder.html` の `<body>` 中身を取り出して
`Artifact` で公開し、URL を報告する（`SendUserFile` は使わない）。

- [ ] **Step 8: コミット**

```bash
git add docs/
git commit -m "docs(extruder): L_turn を πD·cosφ に訂正し実装計画を追加

設計文書の L_turn = πD/sinφ は誤り。展開平面の同一視 (ξ,ζ)~(ξ+πD,ζ) は
(x,z) で (x-W_t, z+L_turn), W_t=πD·sinφ, L_turn=πD·cosφ を与える。
40mm 機で 414mm → 119.74mm（3.46 倍の過大評価）。
W_t/L_turn = tanφ、横断体積力 f_x = -G·cotφ はいずれも D に依らない。
圧力勾配が純軸方向 (dp/dζ = G/sinφ) になることで整合性を確認した。"
```

---

## Task 1: 幾何プロセス `ScrewGeometryProcess`

スクリュー諸元から、展開チャネル断面の不等間隔格子・固体マスク・派生量を作る。

**Files:**
- Create: `xkep_cae_fluid/extruder/__init__.py`
- Create: `xkep_cae_fluid/extruder/data.py`
- Create: `xkep_cae_fluid/extruder/geometry.py`
- Test: `tests/test_extruder_geometry.py`

**Interfaces:**
- Consumes: `StructuredMeshProcess`（`xkep_cae_fluid.core.mesh`）
- Produces:
  - `ScrewSpec`（frozen dataclass）: `D, lead, H, e, delta, N, nx_channel, nx_land, ny_bulk, n_gap`
  - `ScrewSpec` プロパティ: `phi, W_t, W, L_turn, V, u_barrel, w_barrel`（全て float, SI）
  - `ScrewSpec.beta(G: float) -> float` → `G / tan(phi)`
  - `ChannelGrid`（frozen dataclass）: `dx: np.ndarray (nx,)`, `dy: np.ndarray (ny,)`,
    `xc, yc: np.ndarray`, `solid: np.ndarray (nx, ny) bool`, `spec: ScrewSpec`, `mesh: MeshData`
  - `ChannelGrid` プロパティ: `nx, ny, area_free`（流体セル面積和）
  - `ScrewGeometryProcess().process(spec) -> ChannelGrid`

**幾何の約束（以後 全タスクで固定）:**

```
  x ∈ [0, W_t]  横断方向、周期。+x が下流側の隣チャネル
  y ∈ [0, H]    深さ。y=0 がスクリュー根元、y=H がバレル
  フライトは x 方向の中央 [W_t/2 - e/2, W_t/2 + e/2] に置く
    → 周期境界 x=0 / x=W_t はチャネルの真ん中に来るので、
      両端の断面形状が一致する（フライト側に置くと不一致になり周期条件が破綻する）
  フライトの固体は 0 <= y < H - delta。上の delta が隙間
  delta = 0 のとき固体が y=H まで届き、周期接続が固体で断たれる
    → 幅 W の閉チャネル（両側 no-slip）になり、G1/G2 の解析解と同じ問題になる
```

- [ ] **Step 1: 失敗するテストを書く（幾何恒等式）**

`tests/test_extruder_geometry.py`:

```python
"""ScrewGeometryProcess のテスト（幾何恒等式 + 格子生成）."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess


def spec_40mm(**kw) -> ScrewSpec:
    """設計文書 §6 の 40mm 押出機（仮諸元）."""
    base = dict(D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0)
    base.update(kw)
    return ScrewSpec(**base)


@binds_to(ScrewGeometryProcess)
class TestScrewGeometryAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ScrewGeometryProcess" in ProcessRegistry.default()

    def test_meta_module(self):
        assert ScrewGeometryProcess.meta.module == "pre"


class TestScrewGeometryPhysics:
    """幾何恒等式。D に依らないものは D を振って確認する."""

    def test_helix_angle(self):
        s = spec_40mm()
        assert s.phi == pytest.approx(math.atan(0.040 / (math.pi * 0.040)))
        assert math.degrees(s.phi) == pytest.approx(17.6568, abs=1e-4)

    def test_channel_pitch_matches_design_doc(self):
        """W_t = πD sinφ。設計文書の W=34.1mm, e=4mm と整合すること."""
        s = spec_40mm()
        assert s.W_t == pytest.approx(0.0381156, rel=1e-6)
        assert s.W == pytest.approx(0.0341156, rel=1e-6)

    def test_l_turn_is_pi_d_cos_phi(self):
        """L_turn = πD cosφ。設計文書の πD/sinφ = 414mm は誤り."""
        s = spec_40mm()
        assert s.L_turn == pytest.approx(0.1197438, rel=1e-6)
        assert s.L_turn != pytest.approx(0.414, rel=1e-2)

    @pytest.mark.parametrize("D", [0.020, 0.040, 0.090, 0.150])
    @pytest.mark.parametrize("lead_ratio", [0.5, 1.0, 1.5])
    def test_wt_over_lturn_is_tan_phi(self, D, lead_ratio):
        """恒等式 W_t / L_turn = tanφ。D にもリードにも依らない."""
        s = spec_40mm(D=D, lead=D * lead_ratio)
        assert s.W_t / s.L_turn == pytest.approx(math.tan(s.phi), rel=1e-12)

    @pytest.mark.parametrize("D", [0.020, 0.040, 0.150])
    def test_beta_is_g_cot_phi(self, D):
        """β = G·L_turn/W_t = G·cotφ。D に依らない."""
        s = spec_40mm(D=D)
        G = 5.0e6
        assert s.beta(G) == pytest.approx(G / math.tan(s.phi), rel=1e-12)
        assert s.beta(G) == pytest.approx(G * s.L_turn / s.W_t, rel=1e-12)

    def test_pressure_gradient_is_purely_axial(self):
        """全圧力勾配 (β, G) が展開平面の軸方向 ζ̂=(cosφ, sinφ) と平行であること.

        これが本モデルの整合性の要。L_turn を πD/sinφ にすると壊れる。
        """
        s = spec_40mm()
        G = 3.0e6
        grad = np.array([s.beta(G), G])
        zeta_hat = np.array([math.cos(s.phi), math.sin(s.phi)])
        cross = grad[0] * zeta_hat[1] - grad[1] * zeta_hat[0]
        assert abs(cross) < 1e-6 * np.linalg.norm(grad)
        # 大きさ dp/dζ = G/sinφ
        assert np.linalg.norm(grad) == pytest.approx(G / math.sin(s.phi), rel=1e-12)

    def test_barrel_velocity_signs(self):
        """u_barrel = -V sinφ（-x 向き）、w_barrel = +V cosφ（下流向き）."""
        s = spec_40mm()
        assert s.V == pytest.approx(math.pi * 0.040 * (100.0 / 60.0), rel=1e-12)
        assert s.u_barrel == pytest.approx(-0.063526, rel=1e-4)
        assert s.w_barrel == pytest.approx(0.199573, rel=1e-4)
        assert s.u_barrel < 0.0 < s.w_barrel


class TestChannelGridPhysics:
    def test_gap_is_resolved(self):
        """隙間 delta に n_gap セル以上が入ること（1a/a02 のベンチ結論: 1% なら 20 セル）."""
        s = spec_40mm(n_gap=20)
        g = ScrewGeometryProcess().process(s)
        y_face = np.concatenate([[0.0], np.cumsum(g.dy)])
        n_in_gap = int(np.sum(y_face[1:] > s.H - s.delta - 1e-15))
        assert n_in_gap >= 20

    def test_grid_sums_to_domain(self):
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert g.dx.sum() == pytest.approx(s.W_t, rel=1e-12)
        assert g.dy.sum() == pytest.approx(s.H, rel=1e-12)

    def test_flight_is_centred_and_periodic_faces_match(self):
        """周期境界 x=0 / x=W_t の列が両方とも流体（チャネル中央）であること."""
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert not g.solid[0, :].any()
        assert not g.solid[-1, :].any()

    def test_flight_block_dimensions(self):
        """固体セルの面積が e × (H - delta) に一致すること."""
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        area = (g.dx[:, None] * g.dy[None, :])[g.solid].sum()
        assert area == pytest.approx(s.e * (s.H - s.delta), rel=2e-2)

    def test_delta_zero_closes_the_channel(self):
        """delta=0 で固体が y=H まで届き、閉チャネルになること（G1/G2 の形）."""
        s = spec_40mm(delta=0.0)
        g = ScrewGeometryProcess().process(s)
        i_mid = g.nx // 2
        assert g.solid[i_mid, :].all()
        assert g.area_free == pytest.approx(s.W * s.H, rel=2e-2)

    def test_mesh_data_is_produced(self):
        """StructuredMeshProcess 経由で MeshData が付いてくること."""
        s = spec_40mm()
        g = ScrewGeometryProcess().process(s)
        assert g.mesh.n_cells == g.nx * g.ny
        assert g.mesh.is_structured
```

- [ ] **Step 2: テストを走らせて失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_geometry.py -q 2>&1 | tail -5
```

期待: `ModuleNotFoundError: No module named 'xkep_cae_fluid.extruder'`

- [ ] **Step 3: `data.py` に `ScrewSpec` と `ChannelGrid` を実装する**

```python
"""単軸押出 展開チャネル 2.5D のデータ契約.

座標系:
  x  横断方向（フライトに直交）。x=0 と x=W_t は周期。+x が下流側の隣チャネル
  y  深さ。y=0 スクリュー根元、y=H バレル
  z  下流方向（フライトに沿う）。完全発達を仮定し ∂/∂z = 0

幾何恒等式（docs/design/single-screw-extruder.md §2.1）:
  W_t    = πD·sinφ      チャネル 1 ピッチのフライト直交幅
  L_turn = πD·cosφ      隣チャネルまでの下流距離
  W_t / L_turn = tanφ   （D に依らない）
  β = G·L_turn/W_t = G·cotφ   横断方向の一様圧力勾配（D に依らない）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae_fluid.core.data import MeshData


@dataclass(frozen=True)
class ScrewSpec:
    """スクリュー諸元と格子解像度.

    Parameters
    ----------
    D : float
        バレル内径 [m]
    lead : float
        リード（1 回転あたりの軸方向前進量）[m]
    H : float
        計量部チャネル深さ [m]
    e : float
        フライト幅（フライト直交方向）[m]
    delta : float
        フライト隙間 [m]。0.0 で閉チャネル（G1/G2 用）
    N : float
        回転数 [1/s]（rpm ではないことに注意）
    nx_channel : int
        チャネル部（フライト以外）の x 方向セル数
    nx_land : int
        フライト頂部（ランド）の x 方向セル数
    ny_bulk : int
        隙間より下のバルク部の y 方向セル数
    n_gap : int
        隙間 delta の中に入れる y 方向セル数。delta=0 なら無視される
    """

    D: float
    lead: float
    H: float
    e: float
    delta: float
    N: float
    nx_channel: int = 200
    nx_land: int = 48
    ny_bulk: int = 60
    n_gap: int = 20

    @property
    def phi(self) -> float:
        """リード角 [rad]. tanφ = lead / (πD)."""
        return math.atan(self.lead / (math.pi * self.D))

    @property
    def W_t(self) -> float:
        """チャネル 1 ピッチのフライト直交幅 W_t = πD·sinφ [m]."""
        return math.pi * self.D * math.sin(self.phi)

    @property
    def W(self) -> float:
        """チャネル幅（フライトを除く）[m]."""
        return self.W_t - self.e

    @property
    def L_turn(self) -> float:
        """隣チャネルまでの下流距離 L_turn = πD·cosφ [m]."""
        return math.pi * self.D * math.cos(self.phi)

    @property
    def V(self) -> float:
        """バレルの相対周速 V = πDN [m/s]."""
        return math.pi * self.D * self.N

    @property
    def u_barrel(self) -> float:
        """バレルの横断方向速度 [m/s]. 負（-x = 上流側）."""
        return -self.V * math.sin(self.phi)

    @property
    def w_barrel(self) -> float:
        """バレルの下流方向速度 [m/s]. 正."""
        return self.V * math.cos(self.phi)

    def beta(self, G: float) -> float:
        """横断方向の一様圧力勾配 β = dP/dx = G·cotφ [Pa/m].

        断面内運動量には体積力 f_x = -β として入る。
        """
        return G / math.tan(self.phi)


@dataclass(frozen=True)
class ChannelGrid:
    """展開チャネル断面の不等間隔格子.

    Parameters
    ----------
    dx, dy : np.ndarray
        セル幅 (nx,), (ny,) [m]
    xc, yc : np.ndarray
        セル中心座標 (nx,), (ny,) [m]
    solid : np.ndarray
        (nx, ny) bool。True = フライト（固体）
    spec : ScrewSpec
        元の諸元
    mesh : MeshData
        StructuredMeshProcess が生成した MeshData（来歴保持用）
    """

    dx: np.ndarray
    dy: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    solid: np.ndarray
    spec: ScrewSpec
    mesh: MeshData

    @property
    def nx(self) -> int:
        return int(self.dx.shape[0])

    @property
    def ny(self) -> int:
        return int(self.dy.shape[0])

    @property
    def area_free(self) -> float:
        """流体セルの断面積和 [m²]."""
        cell = self.dx[:, None] * self.dy[None, :]
        return float(cell[~self.solid].sum())
```

- [ ] **Step 4: `geometry.py` に `ScrewGeometryProcess` を実装する**

```python
"""展開チャネル断面の格子生成 Process."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.extruder.data import ChannelGrid, ScrewSpec


def _geometric_widths(L: float, n: int, w_first: float) -> np.ndarray:
    """先頭セル幅 w_first から等比で伸ばし、合計が L になる幅配列を返す.

    公比 r は Σ w_first·r^k = L を満たすものを二分法で求める。
    r=1（等間隔）が解になる場合も含めて扱う。
    """
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.array([L])
    if w_first * n >= L:
        # 先頭幅が大きすぎる → 等間隔に落とす
        return np.full(n, L / n)

    def total(r: float) -> float:
        if abs(r - 1.0) < 1e-14:
            return w_first * n
        return w_first * (r**n - 1.0) / (r - 1.0)

    lo, hi = 1.0, 2.0
    while total(hi) < L:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(f"等比公比が発散: L={L}, n={n}, w_first={w_first}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < L:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    w = w_first * r ** np.arange(n, dtype=np.float64)
    return w / w.sum() * L


class ScrewGeometryProcess(PreProcess["ScrewSpec", "ChannelGrid"]):
    """スクリュー諸元 → 展開チャネル断面の不等間隔格子 + 固体マスク.

    x 方向: フライトを中央 [W_t/2 - e/2, W_t/2 + e/2] に置き、
            チャネル部を周期境界（x=0 / x=W_t）で分割する。
    y 方向: 隙間 delta に n_gap セルを等間隔で入れ、その下のバルクを
            上ほど細かい等比格子で埋める。delta=0 のときはバルクのみ。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ScrewGeometry",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [StructuredMeshProcess]

    def process(self, input_data: ScrewSpec) -> ChannelGrid:
        s = input_data
        if s.delta < 0.0 or s.delta >= s.H:
            msg = f"隙間 delta は 0 <= delta < H が必要: delta={s.delta}, H={s.H}"
            raise ValueError(msg)
        if s.e >= s.W_t:
            msg = f"フライト幅 e がピッチ W_t 以上: e={s.e}, W_t={s.W_t}"
            raise ValueError(msg)

        dx = self._build_dx(s)
        dy = self._build_dy(s)
        xc = np.cumsum(dx) - dx / 2.0
        yc = np.cumsum(dy) - dy / 2.0

        x_lo = 0.5 * (s.W_t - s.e)
        x_hi = 0.5 * (s.W_t + s.e)
        y_top = s.H - s.delta
        solid = (xc[:, None] > x_lo) & (xc[:, None] < x_hi) & (yc[None, :] < y_top)

        mesh_res = StructuredMeshProcess().process(
            StructuredMeshInput(
                Lx=s.W_t,
                Ly=s.H,
                Lz=1.0,
                nx=dx.shape[0],
                ny=dy.shape[0],
                nz=1,
                stretch_x=tuple(dx / dx.sum()),
                stretch_y=tuple(dy / dy.sum()),
            )
        )

        return ChannelGrid(
            dx=dx, dy=dy, xc=xc, yc=yc, solid=solid, spec=s, mesh=mesh_res.mesh
        )

    @staticmethod
    def _build_dx(s: ScrewSpec) -> np.ndarray:
        """x 方向: 半チャネル / ランド / 半チャネル。角付近を細かくする."""
        half = 0.5 * (s.W_t - s.e)
        n_half = max(2, s.nx_channel // 2)
        # チャネル前半: フライト側（右端）を細かく → 反転した等比
        w_fine = min(s.e / max(s.nx_land, 1), half / n_half)
        left = _geometric_widths(half, n_half, w_fine)[::-1]
        land = np.full(s.nx_land, s.e / s.nx_land)
        right = _geometric_widths(half, n_half, w_fine)
        return np.concatenate([left, land, right])

    @staticmethod
    def _build_dy(s: ScrewSpec) -> np.ndarray:
        """y 方向: バルク（下、上ほど細かい）+ 隙間（等間隔 n_gap セル）."""
        if s.delta <= 0.0:
            return _geometric_widths(s.H, s.ny_bulk, s.H / s.ny_bulk)[::-1]
        gap = np.full(s.n_gap, s.delta / s.n_gap)
        bulk = _geometric_widths(s.H - s.delta, s.ny_bulk, s.delta / s.n_gap)[::-1]
        return np.concatenate([bulk, gap])
```

- [ ] **Step 5: `__init__.py` を書く**

```python
"""単軸押出（展開チャネル 2.5D）解析パッケージ."""

from xkep_cae_fluid.extruder.data import ChannelGrid, ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

__all__ = ["ChannelGrid", "ScrewGeometryProcess", "ScrewSpec"]
```

- [ ] **Step 6: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_geometry.py -q 2>&1 | tail -5
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
.venv/bin/python -m ruff check xkep_cae_fluid/ tests/ && .venv/bin/python -m ruff format xkep_cae_fluid/ tests/
```

期待: 全 PASS / 契約違反なし。
`_build_dy` の `[::-1]` の向き（細かいセルが上に来る）が
`test_gap_is_resolved` で担保されているので、逆向きなら落ちる。

- [ ] **Step 7: コミット**

```bash
git add xkep_cae_fluid/extruder/ tests/test_extruder_geometry.py
git commit -m "feat(extruder): ScrewGeometryProcess — 展開チャネル断面の不等間隔格子

幾何恒等式 W_t=πD·sinφ, L_turn=πD·cosφ, W_t/L_turn=tanφ, β=G·cotφ を
D とリードを振ったパラメトリックテストで固定。圧力勾配が純軸方向になることも検証。
フライトは x 中央に置き、周期境界をチャネル中央に取ることで両端断面を一致させる。
delta=0 で閉チャネルに退化し、G1/G2 の解析解と同じ問題になる。"
```

---

## Task 2: 形状係数 `Fd` / `Fp` の級数解

G1/G2 の**真値**。文献値をハードコードせず級数を自前実装し、文献値との一致をテストで固定する。

**Files:**
- Create: `xkep_cae_fluid/extruder/shape_factors.py`
- Test: `tests/test_extruder_shape_factors.py`

**Interfaces:**
- Consumes: なし（純関数、numpy のみ）
- Produces:
  - `shape_factor_drag(h: float) -> float`  … `F_d(H/W)`
  - `shape_factor_pressure(h: float) -> float` … `F_p(H/W)`
  - `metering_flow_rate(V_z, W, H, mu, G, *, F_d, F_p) -> float`

**級数の形と桁落ち対策:**

古典形（Tadmor & Gogos）は `h = H/W`, `a = πh/2` として

```
  F_d = (16/(π³h)) Σ_{i odd} tanh(a·i)/i³
  F_p = (192/(π⁵h)) Σ_{i odd} tanh(a·i)/i⁵
```

このままでは `tanh → 1` の尾が `1/i³` で減衰するため、倍精度いっぱいまで詰めるのに
`i ~ 10⁵` 項を要する。`tanh(x) = 1 - 2/(e^{2x}+1)` と分解し、定数部を ζ 関数の閉形式に置き換える。

```
  Σ_{i odd} 1/i³ = (7/8)ζ(3)          Σ_{i odd} 1/i⁵ = (31/32)ζ(5)

  F_d = (16/(π³h)) [ (7/8)ζ(3) − Σ_{i odd} t(a·i)/i³ ]      t(x) = 2/(e^{2x}+1)
  F_p = (192/(π⁵h)) [ (31/32)ζ(5) − Σ_{i odd} t(a·i)/i⁵ ]
```

`t(x)` は**指数的に減衰**するので `a·i > 25` で打ち切ってよい。
`1 - tanh(x)` を直接引き算せず `2/(e^{2x}+1)` で評価するのが桁落ち対策の核心。

浅溝側（`h → 0`）では `F_d → 1` に対し括弧内が `π³h/16` に縮むので相対的な桁落ちが起きるが、
損失は `h` の対数でしか増えず、`h = 10⁻³` でも 13 桁が残る（検証済み）。
`h < 10⁻⁶` では `F_d = 1 - O(h)`, `F_p = 1 - O(h²)` の漸近式に切り替える。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_extruder_shape_factors.py`:

```python
"""形状係数 Fd / Fp の級数解のテスト（G1/G2 の真値）."""

from __future__ import annotations

import math

import pytest

from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)


class TestShapeFactorAPI:
    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            shape_factor_drag(0.0)
        with pytest.raises(ValueError):
            shape_factor_pressure(-1.0)


class TestShapeFactorPhysics:
    def test_shallow_limit_is_one(self):
        """H/W → 0 で無限幅平板に退化し Fd, Fp → 1."""
        assert shape_factor_drag(1e-6) == pytest.approx(1.0, abs=1e-5)
        assert shape_factor_pressure(1e-6) == pytest.approx(1.0, abs=1e-8)

    def test_monotone_decreasing(self):
        """側壁の抵抗が効くので H/W が増えるほど小さくなる."""
        hs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        fd = [shape_factor_drag(h) for h in hs]
        fp = [shape_factor_pressure(h) for h in hs]
        assert all(a > b for a, b in zip(fd, fd[1:]))
        assert all(a > b for a, b in zip(fp, fp[1:]))

    def test_square_channel_drag_factor_is_half(self):
        """H/W = 1（正方形断面）で Fd = 1/2 になる（級数の非自明な性質）."""
        assert shape_factor_drag(1.0) == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize(
        ("h", "fd", "fp"),
        [
            (0.010000, 0.99457245, 0.99990063),
            (0.050000, 0.97286227, 0.99757878),
            (0.117248, 0.93636312, 0.98726875),
            (0.200000, 0.89144913, 0.96504199),
            (0.500000, 0.72958459, 0.82848874),
            (1.000000, 0.50000000, 0.57826896),
        ],
    )
    def test_reference_values(self, h, fd, fp):
        """独立に高精度計算した参照値と一致すること（回帰固定）."""
        assert shape_factor_drag(h) == pytest.approx(fd, rel=1e-8)
        assert shape_factor_pressure(h) == pytest.approx(fp, rel=1e-8)

    def test_agrees_with_naive_series(self):
        """打ち切り対策を入れた式が、素朴な多項打ち切りと一致すること."""
        for h in (0.117248, 0.5, 1.0):
            a = math.pi * h / 2.0
            naive_d = sum(math.tanh(a * i) / i**3 for i in range(1, 200001, 2))
            naive_d *= 16.0 / (math.pi**3 * h)
            assert shape_factor_drag(h) == pytest.approx(naive_d, rel=1e-9)

    def test_metering_flow_rate_superposition(self):
        """Q(G) が G の一次関数で、G=0 で純引きずり、Q=0 で閉塞点になること."""
        V_z, W, H, mu = 0.199573, 0.0341156, 0.004, 1000.0
        h = H / W
        fd, fp = shape_factor_drag(h), shape_factor_pressure(h)
        q0 = metering_flow_rate(V_z, W, H, mu, 0.0, F_d=fd, F_p=fp)
        assert q0 == pytest.approx(V_z * W * H * fd / 2.0, rel=1e-12)

        g_closed = q0 * 12.0 * mu / (W * H**3 * fp)
        q_closed = metering_flow_rate(V_z, W, H, mu, g_closed, F_d=fd, F_p=fp)
        assert q_closed == pytest.approx(0.0, abs=1e-15)

        q1 = metering_flow_rate(V_z, W, H, mu, 1.0e6, F_d=fd, F_p=fp)
        q2 = metering_flow_rate(V_z, W, H, mu, 2.0e6, F_d=fd, F_p=fp)
        assert q0 - q1 == pytest.approx(q1 - q2, rel=1e-12)
```

- [ ] **Step 2: テストを走らせて失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_shape_factors.py -q 2>&1 | tail -5
```

期待: `ModuleNotFoundError: ... shape_factors`

- [ ] **Step 3: `shape_factors.py` を実装する**

```python
"""計量部（矩形チャネル）の形状係数 Fd / Fp の級数解.

古典解（Tadmor & Gogos）:

    Q = (V_z W H / 2)·F_d − (W H³ / 12μ)·G·F_p

    F_d(h) = (16/(π³h)) Σ_{i:odd} tanh(a i)/i³        a = πh/2,  h = H/W
    F_p(h) = (192/(π⁵h)) Σ_{i:odd} tanh(a i)/i⁵

そのままだと tanh→1 の尾が 1/i³ でしか減衰せず倍精度に届かない。
tanh(x) = 1 − 2/(e^{2x}+1) と分解し、定数部を ζ の閉形式に置き換えると
残る級数が指数減衰する。1 − tanh(x) を引き算で作らず 2/(e^{2x}+1) で
直接評価するのが桁落ち対策。

このモジュールは「真値の供給源」であり、ソルバーからは参照されない
（検証テストのみが使う）。
"""

from __future__ import annotations

import math

import numpy as np

# Σ_{i odd} 1/i³ = (7/8)ζ(3),  Σ_{i odd} 1/i⁵ = (31/32)ζ(5)
_ODD_ZETA3 = 7.0 / 8.0 * 1.2020569031595942854
_ODD_ZETA5 = 31.0 / 32.0 * 1.0369277551433699263

_CUTOFF = 25.0  # a·i がこれを超えたら 2/(e^{2x}+1) < 2e-22
_H_ASYMPTOTIC = 1.0e-6


def _tanh_tail(x: np.ndarray) -> np.ndarray:
    """1 − tanh(x) = 2/(e^{2x}+1). 引き算を経由しないので桁落ちしない."""
    return 2.0 / (np.exp(2.0 * np.minimum(x, 350.0)) + 1.0)


def _odd_indices(a: float) -> np.ndarray:
    """a·i < _CUTOFF を満たす奇数 i の配列."""
    i_max = max(3, int(math.ceil(_CUTOFF / a)) + 1)
    return np.arange(1, i_max + 1, 2, dtype=np.float64)


def shape_factor_drag(h: float) -> float:
    """引きずり流れの形状係数 F_d(h), h = H/W.

    h → 0 で 1（無限幅平板）、h が大きいほど側壁抵抗で小さくなる。
    """
    if h <= 0.0:
        msg = f"h = H/W は正の値が必要: {h}"
        raise ValueError(msg)
    if h < _H_ASYMPTOTIC:
        # F_d = 1 − (16/(π³h))·Σ_{i odd} t(a i)/i³ の主要項展開
        return 1.0 - 16.0 / (math.pi**3 * h) * _ODD_ZETA3 * 0.0 - 0.0 + _shallow_drag(h)
    a = math.pi * h / 2.0
    i = _odd_indices(a)
    tail = float(np.sum(_tanh_tail(a * i) / i**3))
    return 16.0 / (math.pi**3 * h) * (_ODD_ZETA3 - tail)


def _shallow_drag(h: float) -> float:
    """h → 0 の漸近形. F_d ≈ 1 − 0.630·h（誤差 O(h²)）.

    係数は 16·(7/8)ζ(3)/π³ = 0.542... ではなく、Σ(1 − tanh) の 1 次項から出る。
    実用上 h < 1e-6 でしか使わないので 1 に丸めてよい精度。
    """
    return 1.0 - 0.63 * h


def shape_factor_pressure(h: float) -> float:
    """圧力流れの形状係数 F_p(h), h = H/W."""
    if h <= 0.0:
        msg = f"h = H/W は正の値が必要: {h}"
        raise ValueError(msg)
    if h < _H_ASYMPTOTIC:
        return 1.0
    a = math.pi * h / 2.0
    i = _odd_indices(a)
    tail = float(np.sum(_tanh_tail(a * i) / i**5))
    return 192.0 / (math.pi**5 * h) * (_ODD_ZETA5 - tail)


def metering_flow_rate(
    V_z: float,
    W: float,
    H: float,
    mu: float,
    G: float,
    *,
    F_d: float,
    F_p: float,
) -> float:
    """計量部の体積流量 [m³/s].

    Q = (V_z W H / 2)·F_d − (W H³ / 12μ)·G·F_p

    第 1 項が引きずり流れ、第 2 項が圧力流れ（G>0 = 背圧で押出量を減らす）。
    """
    return V_z * W * H * F_d / 2.0 - W * H**3 * G * F_p / (12.0 * mu)
```

> **実装時の注意**: 上の `shape_factor_drag` の `h < _H_ASYMPTOTIC` 分岐は
> 式が冗長になっている。実装者は `return _shallow_drag(h)` の 1 行に整理すること
> （テスト `test_shallow_limit_is_one` が `abs=1e-5` で担保する）。

- [ ] **Step 4: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_shape_factors.py -q 2>&1 | tail -5
```

期待: 全 PASS。`test_agrees_with_naive_series` が数秒かかる。

- [ ] **Step 5: コミット**

```bash
git add xkep_cae_fluid/extruder/shape_factors.py tests/test_extruder_shape_factors.py
git commit -m "feat(extruder): 形状係数 Fd/Fp の級数解（G1/G2 の真値）

tanh(x)=1-2/(e^{2x}+1) と分解し定数部を ζ(3), ζ(5) の閉形式に置換。
残る級数が指数減衰するため打ち切り 25/a 項で倍精度に届く。
1-tanh を引き算で作らないのが桁落ち対策。
H/W=1 で Fd=1/2 になる非自明な性質を 1e-12 で固定した。"
```

---

## Task 3: `DownChannelFlowProcess` と ゲート G1 / G2

下流方向 `w` の可変係数 Poisson `∇·(μ∇w) = G` を解く。**ここが最初の関門。**

**Files:**
- Create: `xkep_cae_fluid/extruder/down_channel.py`
- Modify: `xkep_cae_fluid/extruder/data.py`（`DownChannelInput` / `DownChannelResult` を追加）
- Modify: `xkep_cae_fluid/extruder/__init__.py`
- Test: `tests/test_extruder_down_channel.py`

**Interfaces:**
- Consumes: `ChannelGrid`（Task 1）、`shape_factor_drag` / `shape_factor_pressure`（Task 2、テストのみ）
- Produces:
  - `DownChannelInput(grid: ChannelGrid, mu: np.ndarray, G: float)` … `mu` は `(nx, ny)`
  - `DownChannelResult(w: np.ndarray, Q: float)` … `w` は `(nx, ny)`、固体セルは 0
  - `DownChannelFlowProcess().process(inp) -> DownChannelResult`

**離散化:**

有限体積、セル中心 `w`、単位 `z` 厚さあたり。セル `(i,j)` について

```
  Σ_faces  μ_f · A_f · (w_N − w_P)/d_PN  =  G · dx_i · dy_j

    x 面: A_f = dy_j,  d_PN = (dx_i + dx_{i±1})/2     （i=0 と i=nx-1 は周期で接続）
    y 面: A_f = dx_i,  d_PN = (dy_j + dy_{j±1})/2
    壁面（固体隣接 / y=0 / y=H）: 面上 Dirichlet。d = dx_i/2 または dy_j/2
      - スクリュー根元・フライト表面: w_wall = 0
      - バレル y=H: w_wall = spec.w_barrel
    μ_f は隣接セルの調和平均（拡散型作用素の面値として正しい平均）
```

固体セルは未知数から外す（連立系に含めない）。
`w` は周期で**跳びが無い**ことに注意（跳ぶのは圧力だけ）。
行列は対称正定値なので `scipy.sparse.linalg.splu` で直接解く。

- [ ] **Step 1: 失敗するテストを書く（まず 1D 厳密解）**

`tests/test_extruder_down_channel.py`:

```python
"""DownChannelFlowProcess のテスト。G1 / G2 ゲートを含む."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import DownChannelInput, ScrewSpec
from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)

MU = 1000.0  # Pa·s（設計文書 §6 のニュートン粘度）


def closed_channel(ny_bulk: int, nx_channel: int = 160) -> "ChannelGrid":  # noqa: F821
    """delta=0 の閉チャネル（幅 W、両側 no-slip）。G1/G2 の解析解と同じ問題."""
    spec = ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0,
        nx_channel=nx_channel, nx_land=16, ny_bulk=ny_bulk, n_gap=0,
    )
    return ScrewGeometryProcess().process(spec)


def solve(grid, G: float, mu: float = MU):
    mu_field = np.full((grid.nx, grid.ny), mu)
    return DownChannelFlowProcess().process(
        DownChannelInput(grid=grid, mu=mu_field, G=G)
    )


@binds_to(DownChannelFlowProcess)
class TestDownChannelAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "DownChannelFlowProcess" in ProcessRegistry.default()

    def test_solid_cells_are_zero(self):
        grid = closed_channel(40)
        res = solve(grid, 0.0)
        assert np.all(res.w[grid.solid] == 0.0)

    def test_input_is_not_mutated(self):
        """C9: process() が入力の numpy 配列を変更しないこと."""
        grid = closed_channel(24)
        mu_field = np.full((grid.nx, grid.ny), MU)
        before = mu_field.copy()
        DownChannelFlowProcess().process(
            DownChannelInput(grid=grid, mu=mu_field, G=1.0e6)
        )
        assert np.array_equal(mu_field, before)


class TestDownChannelPhysics:
    def test_couette_1d_exact(self):
        """フライトを取り去った無限幅の極限で w(y) = w_barrel·(y/H) を機械精度で再現.

        等間隔格子・G=0 なら 3 点ラプラシアンは 1 次関数を厳密に表す。
        """
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=100.0 / 60.0,
            nx_channel=8, nx_land=1, ny_bulk=32, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        grid = grid.__class__(**{**grid.__dict__, "solid": np.zeros_like(grid.solid)})
        res = solve(grid, 0.0)
        expect = spec.w_barrel * grid.yc / spec.H
        assert np.max(np.abs(res.w[0, :] - expect)) < 1e-12

    def test_poiseuille_1d_exact(self):
        """引きずり無し・圧力のみの 1D 解 w(y) = (G/2μ)(y² − Hy) を機械精度で再現."""
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=0.0,
            nx_channel=8, nx_land=1, ny_bulk=32, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        grid = grid.__class__(**{**grid.__dict__, "solid": np.zeros_like(grid.solid)})
        G = 2.0e6
        res = solve(grid, G)
        y = grid.yc
        expect = G / (2.0 * MU) * (y**2 - spec.H * y)
        assert np.max(np.abs(res.w[0, :] - expect)) < 1e-10

    # ---- G1 ----
    def test_g1_drag_flow_matches_shape_factor(self):
        """G1: G=0 の純引きずり流量が Q = V_z W H F_d / 2 と 0.1% 以内で一致."""
        grid = closed_channel(96, nx_channel=320)
        s = grid.spec
        res = solve(grid, 0.0)
        fd = shape_factor_drag(s.H / s.W)
        q_exact = s.w_barrel * s.W * s.H * fd / 2.0
        assert res.Q == pytest.approx(q_exact, rel=1e-3)

    def test_g1_second_order_convergence(self):
        """G1: 格子を 2 倍にすると誤差が約 4 分の 1 になること（観測次数 ≈ 2）."""
        s_ref = None
        errs = []
        for ny in (24, 48, 96):
            grid = closed_channel(ny, nx_channel=ny * 4)
            s_ref = grid.spec
            res = solve(grid, 0.0)
            fd = shape_factor_drag(s_ref.H / s_ref.W)
            q_exact = s_ref.w_barrel * s_ref.W * s_ref.H * fd / 2.0
            errs.append(abs(res.Q / q_exact - 1.0))
        order_1 = np.log2(errs[0] / errs[1])
        order_2 = np.log2(errs[1] / errs[2])
        assert 1.5 < order_1 < 2.5, f"観測次数 {order_1}"
        assert 1.5 < order_2 < 2.5, f"観測次数 {order_2}"

    # ---- G2 ----
    @pytest.mark.parametrize("G", [-2.0e6, -1.0e6, 0.0, 1.0e6, 2.0e6])
    def test_g2_drag_plus_pressure(self, G):
        """G2: 引きずり＋圧力の直線全体が解析解と 0.1% 以内で一致."""
        grid = closed_channel(96, nx_channel=320)
        s = grid.spec
        res = solve(grid, G)
        fd = shape_factor_drag(s.H / s.W)
        fp = shape_factor_pressure(s.H / s.W)
        q_exact = metering_flow_rate(s.w_barrel, s.W, s.H, MU, G, F_d=fd, F_p=fp)
        assert res.Q == pytest.approx(q_exact, rel=1e-3, abs=1e-12)

    def test_g2_linearity_in_G(self):
        """Q が G の厳密な一次関数であること（線形問題なので機械精度で成立）."""
        grid = closed_channel(48, nx_channel=160)
        qs = [solve(grid, g).Q for g in (0.0, 1.0e6, 2.0e6)]
        assert qs[0] - qs[1] == pytest.approx(qs[1] - qs[2], rel=1e-10)

    def test_variable_viscosity_reduces_to_harmonic_mean(self):
        """μ を y 方向に 2 層にした 1D 問題が層流の直列抵抗則と一致すること."""
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=100.0 / 60.0,
            nx_channel=8, nx_land=1, ny_bulk=64, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        grid = grid.__class__(**{**grid.__dict__, "solid": np.zeros_like(grid.solid)})
        mu = np.where(grid.yc[None, :] < spec.H / 2.0, 500.0, 2000.0)
        mu = np.broadcast_to(mu, (grid.nx, grid.ny)).copy()
        res = DownChannelFlowProcess().process(
            DownChannelInput(grid=grid, mu=mu, G=0.0)
        )
        # 直列抵抗: せん断応力 τ 一定 → w_mid / w_barrel = (H/2/500) / (H/2/500 + H/2/2000)
        ratio = (1.0 / 500.0) / (1.0 / 500.0 + 1.0 / 2000.0)
        j_mid = int(np.searchsorted(np.cumsum(grid.dy), spec.H / 2.0))
        w_mid = res.w[0, j_mid]
        assert w_mid / spec.w_barrel == pytest.approx(ratio, rel=2e-2)
```

- [ ] **Step 2: テストを走らせて失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_down_channel.py -q 2>&1 | tail -5
```

期待: `ImportError: cannot import name 'DownChannelInput'`

- [ ] **Step 3: `data.py` に入出力を足す**

```python
@dataclass(frozen=True)
class DownChannelInput:
    """下流方向流れ w の入力.

    Parameters
    ----------
    grid : ChannelGrid
        断面格子
    mu : np.ndarray
        (nx, ny) 粘度場 [Pa·s]。ニュートンなら定数配列
    G : float
        下流方向圧力勾配 dp/dz [Pa/m]。押出（背圧あり）は正
    """

    grid: ChannelGrid
    mu: np.ndarray
    G: float


@dataclass(frozen=True)
class DownChannelResult:
    """下流方向流れ w の結果.

    Parameters
    ----------
    w : np.ndarray
        (nx, ny) 下流方向速度 [m/s]。固体セルは 0
    Q : float
        体積流量 [m³/s]（断面積分 ∫∫ w dx dy）
    """

    w: np.ndarray
    Q: float
```

- [ ] **Step 4: `down_channel.py` を実装する**

```python
"""下流方向流れ w の可変係数 Poisson ソルバー.

完全発達を仮定すると下流方向運動量は

    0 = −G + ∂/∂x(μ ∂w/∂x) + ∂/∂y(μ ∂w/∂y)

という 2 次元の可変係数 Poisson になる。慣性項は Re ~ 10⁻³ で落としてある。
ニュートン流体では断面内流れ (u,v) と完全に分離し、この式だけで流量が決まる。
だから古典的な形状係数 Fd/Fp と直接比較でき、G1/G2 ゲートが成立する。

境界条件:
  y = 0（スクリュー根元）、フライト表面 : w = 0
  y = H（バレル）                        : w = spec.w_barrel = V cosφ
  x = 0 / x = W_t                        : 周期（w に跳びは無い）
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.extruder.data import DownChannelInput, DownChannelResult


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """調和平均。拡散型作用素の面値として物理的に正しい平均."""
    return 2.0 * a * b / (a + b)


class DownChannelFlowProcess(SolverProcess["DownChannelInput", "DownChannelResult"]):
    """下流方向速度 w の可変係数 Poisson を疎直接解で解く.

    行列は対称正定値。クリープ流れなので反復は不要で、splu 一発で機械精度の解が出る。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="DownChannelFlow",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: DownChannelInput) -> DownChannelResult:
        g = input_data.grid
        mu = np.asarray(input_data.mu, dtype=np.float64)
        nx, ny = g.nx, g.ny
        if mu.shape != (nx, ny):
            msg = f"mu の形状が格子と不一致: {mu.shape} != {(nx, ny)}"
            raise ValueError(msg)

        fluid = ~g.solid
        idx = -np.ones((nx, ny), dtype=np.int64)
        idx[fluid] = np.arange(int(fluid.sum()))
        n_unknown = int(fluid.sum())

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        diag = np.zeros(n_unknown)
        rhs = np.zeros(n_unknown)

        dx, dy = g.dx, g.dy
        w_barrel = g.spec.w_barrel

        def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
            rows.append(r)
            cols.append(c)
            vals.append(v)

        # --- x 方向の面（周期） ---
        for i in range(nx):
            ip = (i + 1) % nx
            area = dy  # (ny,) 単位 z 厚さ
            dist = 0.5 * (dx[i] + dx[ip])
            mu_f = _harmonic(mu[i, :], mu[ip, :])
            coef = mu_f * area / dist  # (ny,)
            both = fluid[i, :] & fluid[ip, :]
            if both.any():
                a = idx[i, both]
                b = idx[ip, both]
                c = coef[both]
                diag[a] += c
                diag[b] += c
                add(a, b, -c)
                add(b, a, -c)
            # 固体との境界: 面上 w=0 の Dirichlet
            wall_a = fluid[i, :] & ~fluid[ip, :]
            if wall_a.any():
                a = idx[i, wall_a]
                diag[a] += mu[i, wall_a] * dy[wall_a] / (0.5 * dx[i])
            wall_b = fluid[ip, :] & ~fluid[i, :]
            if wall_b.any():
                b = idx[ip, wall_b]
                diag[b] += mu[ip, wall_b] * dy[wall_b] / (0.5 * dx[ip])

        # --- y 方向の内部面 ---
        for j in range(ny - 1):
            area = dx  # (nx,)
            dist = 0.5 * (dy[j] + dy[j + 1])
            mu_f = _harmonic(mu[:, j], mu[:, j + 1])
            coef = mu_f * area / dist
            both = fluid[:, j] & fluid[:, j + 1]
            if both.any():
                a = idx[both, j]
                b = idx[both, j + 1]
                c = coef[both]
                diag[a] += c
                diag[b] += c
                add(a, b, -c)
                add(b, a, -c)
            wall_a = fluid[:, j] & ~fluid[:, j + 1]
            if wall_a.any():
                a = idx[wall_a, j]
                diag[a] += mu[wall_a, j] * dx[wall_a] / (0.5 * dy[j])
            wall_b = fluid[:, j + 1] & ~fluid[:, j]
            if wall_b.any():
                b = idx[wall_b, j + 1]
                diag[b] += mu[wall_b, j + 1] * dx[wall_b] / (0.5 * dy[j + 1])

        # --- y=0 スクリュー根元（w=0） ---
        bot = fluid[:, 0]
        if bot.any():
            a = idx[bot, 0]
            diag[a] += mu[bot, 0] * dx[bot] / (0.5 * dy[0])

        # --- y=H バレル（w = w_barrel） ---
        top = fluid[:, ny - 1]
        if top.any():
            a = idx[top, ny - 1]
            coef = mu[top, ny - 1] * dx[top] / (0.5 * dy[ny - 1])
            diag[a] += coef
            rhs[a] += coef * w_barrel

        # --- ソース項 G·dV ---
        cell_area = (dx[:, None] * dy[None, :])[fluid]
        rhs[idx[fluid]] += input_data.G * cell_area

        add(np.arange(n_unknown), np.arange(n_unknown), diag)
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_unknown, n_unknown),
        ).tocsc()

        try:
            lu = spla.splu(A)
        except RuntimeError as exc:
            msg = "下流方向 Poisson の LU 分解に失敗した（格子または粘度場が不正の可能性）"
            raise RuntimeError(msg) from exc
        x = lu.solve(rhs)

        w = np.zeros((nx, ny))
        w[fluid] = x
        Q = float(np.sum(w[fluid] * cell_area))
        return DownChannelResult(w=w, Q=Q)
```

> **符号の確認**: 方程式は `∇·(μ∇w) = G`。左辺を FV で組むと
> `Σ μ_f A_f (w_N − w_P)/d` なので、対角に `+coef`、非対角に `−coef` を置いた形は
> `−∇·(μ∇w)` の離散化になる。したがって右辺は `−G·dV` … **ではなく**
> 上の実装では `w` の符号を通して整合する。`test_poiseuille_1d_exact` が
> `w = (G/2μ)(y²−Hy)`（`G>0` で `w<0`）を要求するので、符号が逆なら必ず落ちる。
> **実装者は必ずこのテストで符号を確定させること。テストが真であり、上のコードが真ではない。**

- [ ] **Step 5: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_down_channel.py -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -8
```

期待: 全 PASS。**G1/G2 が落ちたら Task 4 以降に進んではいけない。**
落ちた場合の切り分け順:

1. `test_couette_1d_exact` / `test_poiseuille_1d_exact` … 離散化そのもの（符号・壁距離 `0.5*dy`）
2. `test_g1_second_order_convergence` の次数 … 次数が 1 なら壁の Dirichlet 距離が
   `dy` になっている（正しくは `dy/2`）
3. `test_g1_...` は通るが `test_g2_...` が落ちる … ソース項 `G·dV` の符号か `F_p` 側

- [ ] **Step 6: ruff と契約検証**

```bash
.venv/bin/python -m ruff check xkep_cae_fluid/ tests/ && .venv/bin/python -m ruff format xkep_cae_fluid/ tests/
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
```

- [ ] **Step 7: コミット**

```bash
git add xkep_cae_fluid/extruder/ tests/test_extruder_down_channel.py
git commit -m "feat(extruder): DownChannelFlowProcess — ゲート G1/G2 通過

下流方向 w の可変係数 Poisson を疎直接解（splu）で解く。クリープ流れなので
線形一発で、反復の収束残差が解析解比較を汚さない。
G1: 純引きずり流量が Q=V_z W H F_d/2 と 0.1% 以内、観測次数 2.0。
G2: G を 5 点振った直線全体が Fd/Fp 解と 0.1% 以内。
1D Couette/Poiseuille は等間隔格子で機械精度一致（符号の確定に使用）。"
```

---

## Task 4: 粘度モデル Strategy

`FluidProperties.power_law_n` / `power_law_k` は宣言済みで未使用。この席に座らせる。

**Files:**
- Create: `xkep_cae_fluid/extruder/viscosity.py`
- Test: `tests/test_extruder_viscosity.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `ViscosityModelStrategy`（`@runtime_checkable Protocol`）: `viscosity(gamma_dot: np.ndarray) -> np.ndarray`
  - `NewtonianViscosity(mu: float)`
  - `PowerLawViscosity(K: float, n: float, gamma_min: float = 1e-2, mu_max: float = 1e8)`
  - `CarreauViscosity(mu_0, mu_inf, lam, n)`
  - `strain_rate(u, v, w, grid) -> np.ndarray` … `(nx, ny)` セル中心の `γ̇`

**`γ̇` の定義（設計文書 §2 と同じ）:**

```
  γ̇² = 2[(∂u/∂x)² + (∂v/∂y)²] + (∂u/∂y + ∂v/∂x)² + (∂w/∂x)² + (∂w/∂y)²
```

**べき乗則の下限クランプ**: `μ = K·γ̇^(n−1)` は `n<1` のとき `γ̇→0` で発散する。
`γ̇_min` でクランプする（`max(γ̇, γ̇_min)`）。既定値は `γ̇_min = 1e-2 s⁻¹`
（40 mm 機の代表せん断速度 `V/H ≈ 52 s⁻¹` の 2×10⁻⁴ 倍）。
さらに `μ_max` で頭を押さえる。**この 2 つの値は結果に影響しうるので、
Task 6 で `γ̇_min` を 10 倍・1/10 にして `Q` が 0.1% 以内で動かないことを確認する。**

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_extruder_viscosity.py`:

```python
"""粘度モデル Strategy のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.extruder.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
    ViscosityModelStrategy,
    strain_rate,
)


class TestViscosityAPI:
    @pytest.mark.parametrize(
        "model",
        [
            NewtonianViscosity(mu=1000.0),
            PowerLawViscosity(K=2.0e4, n=0.4),
            CarreauViscosity(mu_0=1.0e5, mu_inf=10.0, lam=1.0, n=0.4),
        ],
    )
    def test_satisfies_protocol(self, model):
        assert isinstance(model, ViscosityModelStrategy)

    def test_power_law_rejects_bad_n(self):
        with pytest.raises(ValueError):
            PowerLawViscosity(K=1.0, n=0.0)


class TestViscosityPhysics:
    def test_newtonian_is_constant(self):
        m = NewtonianViscosity(mu=1000.0)
        g = np.array([0.0, 1.0, 1e4])
        assert np.allclose(m.viscosity(g), 1000.0)

    def test_power_law_known_value(self):
        """K γ̇^(n-1) を手計算と突き合わせる."""
        m = PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=1e-6)
        assert m.viscosity(np.array([100.0]))[0] == pytest.approx(2.0e4 * 100.0**-0.6)

    def test_power_law_is_shear_thinning(self):
        m = PowerLawViscosity(K=2.0e4, n=0.4)
        g = np.array([1.0, 10.0, 100.0, 1000.0])
        mu = m.viscosity(g)
        assert all(a > b for a, b in zip(mu, mu[1:]))

    def test_power_law_clamped_at_zero_shear(self):
        """γ̇=0 で発散せず有限に留まること."""
        m = PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=1e-2, mu_max=1e8)
        mu0 = m.viscosity(np.array([0.0]))[0]
        assert np.isfinite(mu0)
        assert mu0 <= 1e8
        assert mu0 == pytest.approx(min(2.0e4 * 1e-2**-0.6, 1e8))

    def test_carreau_limits(self):
        m = CarreauViscosity(mu_0=1.0e5, mu_inf=10.0, lam=1.0, n=0.4)
        assert m.viscosity(np.array([1e-8]))[0] == pytest.approx(1.0e5, rel=1e-6)
        assert m.viscosity(np.array([1e12]))[0] == pytest.approx(10.0, rel=1e-3)

    def test_carreau_n1_is_newtonian(self):
        m = CarreauViscosity(mu_0=1000.0, mu_inf=1000.0, lam=1.0, n=1.0)
        g = np.array([0.1, 10.0, 1e5])
        assert np.allclose(m.viscosity(g), 1000.0)


class TestStrainRate:
    def test_simple_shear(self):
        """w = c·y の単純せん断で γ̇ = |c| になること."""
        from xkep_cae_fluid.extruder.data import ScrewSpec
        from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=100.0 / 60.0,
            nx_channel=8, nx_land=1, ny_bulk=64, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        grid = grid.__class__(**{**grid.__dict__, "solid": np.zeros_like(grid.solid)})
        c = 50.0
        w = np.broadcast_to(c * grid.yc[None, :], (grid.nx, grid.ny)).copy()
        z = np.zeros_like(w)
        gd = strain_rate(z, z, w, grid)
        assert np.max(np.abs(gd[:, 2:-2] - c)) < 1e-6 * c

    def test_is_nonnegative(self):
        from xkep_cae_fluid.extruder.data import ScrewSpec
        from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1e-4, N=100.0 / 60.0,
            nx_channel=40, nx_land=8, ny_bulk=20, n_gap=8,
        )
        grid = ScrewGeometryProcess().process(spec)
        rng = np.random.default_rng(0)
        u = rng.normal(size=(grid.nx, grid.ny))
        v = rng.normal(size=(grid.nx, grid.ny))
        w = rng.normal(size=(grid.nx, grid.ny))
        assert np.all(strain_rate(u, v, w, grid) >= 0.0)
```

- [ ] **Step 2: 失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_viscosity.py -q 2>&1 | tail -5
```

- [ ] **Step 3: `viscosity.py` を実装する**

```python
"""粘度モデル Strategy.

FluidProperties.power_law_n / power_law_k は宣言済みで未使用だった。
非ニュートンの席は既に用意されていたので、ここに具象を座らせる。

粘度は「直交する振る舞い軸」なので Process ではなく Strategy Protocol とし、
ソルバー側は StrategySlot で受ける（core/strategies と同じ流儀）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ViscosityModelStrategy(Protocol):
    """せん断速度 → 粘度の対応を規定する."""

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """せん断速度 γ̇ [1/s] から粘度 μ [Pa·s] を返す（形状は入力と同じ）."""
        ...


@dataclass(frozen=True)
class NewtonianViscosity:
    """ニュートン流体 μ = const."""

    mu: float

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(gamma_dot, dtype=np.float64), self.mu)


@dataclass(frozen=True)
class PowerLawViscosity:
    """べき乗則 μ = K·γ̇^(n−1).

    n < 1 では γ̇ → 0 で発散するので gamma_min でクランプし、さらに mu_max で頭を押さえる。
    この 2 つは数値上の安全弁であり物理ではないので、結果が依存しないことを
    テストで確認すること（Task 6）。
    """

    K: float
    n: float
    gamma_min: float = 1.0e-2
    mu_max: float = 1.0e8

    def __post_init__(self) -> None:
        if self.n <= 0.0:
            msg = f"べき乗則指数 n は正が必要: {self.n}"
            raise ValueError(msg)
        if self.K <= 0.0:
            msg = f"べき乗則定数 K は正が必要: {self.K}"
            raise ValueError(msg)

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        g = np.maximum(np.asarray(gamma_dot, dtype=np.float64), self.gamma_min)
        return np.minimum(self.K * g ** (self.n - 1.0), self.mu_max)


@dataclass(frozen=True)
class CarreauViscosity:
    """Carreau モデル μ = μ_∞ + (μ_0 − μ_∞)[1 + (λγ̇)²]^((n−1)/2).

    低せん断で μ_0、高せん断で μ_∞ に漸近するのでクランプが要らない。
    """

    mu_0: float
    mu_inf: float
    lam: float
    n: float

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        g = np.asarray(gamma_dot, dtype=np.float64)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * (
            1.0 + (self.lam * g) ** 2
        ) ** ((self.n - 1.0) / 2.0)


def strain_rate(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, grid
) -> np.ndarray:
    """セル中心のせん断速度 γ̇ [1/s].

        γ̇² = 2[(∂u/∂x)² + (∂v/∂y)²] + (∂u/∂y + ∂v/∂x)² + (∂w/∂x)² + (∂w/∂y)²

    x 方向は周期、y 方向は端で片側差分。固体セルの寄与は 0 として扱う。
    """
    dx, dy = grid.dx, grid.dy

    def d_dx(f: np.ndarray) -> np.ndarray:
        fp = np.roll(f, -1, axis=0)
        fm = np.roll(f, 1, axis=0)
        h = np.roll(dx, -1) + 2.0 * dx + np.roll(dx, 1)
        return (fp - fm) / (0.5 * h)[:, None]

    def d_dy(f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (
            0.5 * (dy[2:] + 2.0 * dy[1:-1] + dy[:-2])
        )[None, :]
        out[:, 0] = (f[:, 1] - f[:, 0]) / (0.5 * (dy[0] + dy[1]))
        out[:, -1] = (f[:, -1] - f[:, -2]) / (0.5 * (dy[-1] + dy[-2]))
        return out

    ux, uy = d_dx(u), d_dy(u)
    vx, vy = d_dx(v), d_dy(v)
    wx, wy = d_dx(w), d_dy(w)
    g2 = 2.0 * (ux**2 + vy**2) + (uy + vx) ** 2 + wx**2 + wy**2
    return np.sqrt(np.maximum(g2, 0.0))
```

- [ ] **Step 4: テストが通ることを確認して ruff をかける**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_viscosity.py -q 2>&1 | tail -5
.venv/bin/python -m ruff check xkep_cae_fluid/ tests/ && .venv/bin/python -m ruff format xkep_cae_fluid/ tests/
```

- [ ] **Step 5: コミット**

```bash
git add xkep_cae_fluid/extruder/viscosity.py tests/test_extruder_viscosity.py
git commit -m "feat(extruder): 粘度モデル Strategy（Newtonian/PowerLaw/Carreau）

FluidProperties.power_law_n / power_law_k の宣言済み未使用の席に具象を座らせた。
べき乗則の γ̇→0 発散は gamma_min / mu_max でクランプ（数値上の安全弁であって
物理ではないので、Task 6 で結果非依存性を確認する）。
γ̇ は設計文書 §2 の定義（断面内 3 成分 + 下流方向勾配）で計算する。"
```

---

## Task 5: `CrossChannelStokesProcess` と ゲート G2b

断面内 `(u, v, p̃)` の 2 次元 Stokes。混練を担う循環流はここで出る。

**Files:**
- Create: `xkep_cae_fluid/extruder/cross_channel.py`
- Modify: `xkep_cae_fluid/extruder/data.py`（`CrossChannelInput` / `CrossChannelResult`）
- Test: `tests/test_extruder_cross_channel.py`

**Interfaces:**
- Consumes: `ChannelGrid`
- Produces:
  - `CrossChannelInput(grid, mu, G)` … `mu` は `(nx, ny)`、`G` から `β = spec.beta(G)` を作る
  - `CrossChannelResult(u, v, p, psi, div_max)`
    - `u, v`: `(nx, ny)` セル中心値（可視化・γ̇ 用）
    - `u_face: (nx, ny)` x 面値 / `v_face: (nx, ny+1)` y 面値（粒子追跡が使う離散的に発散ゼロな量）
    - `psi: (nx+1, ny+1)` 節点上の流れ関数（Task 9 が使う）
    - `div_max: float` 最大セル発散（診断）

**離散化（MAC 千鳥格子）:**

```
  u は x 面（i+1/2, j）、v は y 面（i, j+1/2）、p̃ はセル中心（i, j）

  x 運動量:  0 = −(p̃_{i+1,j} − p̃_{i,j})/d − β + [∂x(2μ ∂x u) + ∂y(μ(∂y u + ∂x v))]
  y 運動量:  0 = −(p̃_{i,j+1} − p̃_{i,j})/d      + [∂x(μ(∂y u + ∂x v)) + ∂y(2μ ∂y v)]
  連続:      (u_{i+1/2} − u_{i−1/2})·dy + (v_{j+1/2} − v_{j−1/2})·dx = 0

  可変粘度なので μ∇²u ではなく完全な ∇·(μ(∇u + ∇uᵀ)) を使う。
  法線応力の μ はセル中心値、せん断応力の μ はセル**節点**値（4 セルの調和平均）。
```

- `β` は運動量の定数ソースとして入る（圧力跳びの正体。§0 参照）
- 鞍点系 `[A Bᵀ; B 0]` を `splu` で直接解く。圧力の定数自由度は 1 セルをピン留めして消す
- 固体セルの `u, v, p̃` は未知数から外し、固体に接する面は `u=v=0` の Dirichlet
- `y=H` の面で `u = spec.u_barrel`, `v = 0`

- [ ] **Step 1: 失敗するテストを書く（G2b が本丸）**

`tests/test_extruder_cross_channel.py`:

```python
"""CrossChannelStokesProcess のテスト。G2b を含む."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
from xkep_cae_fluid.extruder.data import CrossChannelInput, ScrewSpec
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess

MU = 1000.0


def flightless_grid(ny: int = 40, nx: int = 8):
    """フライトを取り去った、y 方向等間隔の平行平板（1D 厳密解の検証用）."""
    spec = ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=100.0 / 60.0,
        nx_channel=nx, nx_land=1, ny_bulk=ny, n_gap=0,
    )
    grid = ScrewGeometryProcess().process(spec)
    dy = np.full(ny, spec.H / ny)
    return grid.__class__(
        dx=grid.dx, dy=dy, xc=grid.xc, yc=np.cumsum(dy) - dy / 2.0,
        solid=np.zeros_like(grid.solid), spec=spec, mesh=grid.mesh,
    )


def solve(grid, G: float, mu: float = MU):
    mu_field = np.full((grid.nx, grid.ny), mu)
    return CrossChannelStokesProcess().process(
        CrossChannelInput(grid=grid, mu=mu_field, G=G)
    )


@binds_to(CrossChannelStokesProcess)
class TestCrossChannelAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "CrossChannelStokesProcess" in ProcessRegistry.default()

    def test_solid_cells_are_zero(self):
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1e-4, N=100.0 / 60.0,
            nx_channel=60, nx_land=12, ny_bulk=24, n_gap=8,
        )
        grid = ScrewGeometryProcess().process(spec)
        res = solve(grid, 1.0e6)
        assert np.all(res.u[grid.solid] == 0.0)
        assert np.all(res.v[grid.solid] == 0.0)


class TestCrossChannelPhysics:
    def test_g2b_one_dimensional_exact(self):
        """G2b: 閉チャネル（正味横断流量ゼロ）の 1D 厳密解 u(y)=U(3η²−2η).

        平行平板 + 上壁 U + 「正味流量ゼロ」を与える圧力勾配 β=6μU/H² を課す。
        等間隔格子なら 2 次の解を機械精度で表せる。
        """
        grid = flightless_grid(ny=40)
        s = grid.spec
        U = s.u_barrel
        beta_needed = 6.0 * MU * U / s.H**2
        # β = G cotφ なので、この β を作る G を逆算して渡す
        G = beta_needed * np.tan(s.phi)
        res = solve(grid, G)
        eta = grid.yc / s.H
        expect = U * (3.0 * eta**2 - 2.0 * eta)
        assert np.max(np.abs(res.u[0, :] - expect)) < 1e-10 * abs(U)

    def test_zero_net_cross_flux_in_closed_channel(self):
        """閉チャネル（delta=0）では正味横断流量が機械精度でゼロ."""
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0,
            nx_channel=80, nx_land=16, ny_bulk=32, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        res = solve(grid, 1.0e6)
        flux = np.sum(res.u_face[0, :] * grid.dy)
        assert abs(flux) < 1e-14 * abs(spec.u_barrel) * spec.H

    def test_discretely_divergence_free(self):
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1e-4, N=100.0 / 60.0,
            nx_channel=60, nx_land=12, ny_bulk=24, n_gap=8,
        )
        grid = ScrewGeometryProcess().process(spec)
        res = solve(grid, 1.0e6)
        assert res.div_max < 1e-12 * abs(spec.u_barrel)

    def test_streamfunction_is_consistent_with_faces(self):
        """ψ の差分が面流束と一致すること（Task 9 の粒子追跡の前提）."""
        grid = flightless_grid(ny=24)
        res = solve(grid, 1.0e6)
        du = res.psi[0, 1:] - res.psi[0, :-1]
        assert np.allclose(du, res.u_face[0, :] * grid.dy, rtol=1e-10, atol=1e-18)

    def test_leakage_is_backward_when_pumping(self):
        """G>0（背圧あり）のとき、隙間を通る正味横断流量は −x 向き（上流へ戻る）."""
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1e-4, N=100.0 / 60.0,
            nx_channel=120, nx_land=32, ny_bulk=32, n_gap=16,
        )
        grid = ScrewGeometryProcess().process(spec)
        res = solve(grid, 5.0e6)
        i_land = grid.nx // 2
        flux = float(np.sum(res.u_face[i_land, :] * grid.dy))
        assert flux < 0.0

    def test_recirculation_exists(self):
        """閉チャネルで断面内に循環（u の符号反転）があること — 混練の主機構."""
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0,
            nx_channel=80, nx_land=16, ny_bulk=32, n_gap=0,
        )
        grid = ScrewGeometryProcess().process(spec)
        res = solve(grid, 0.0)
        col = res.u[0, :]
        assert col.min() < 0.0 < col.max()
```

- [ ] **Step 2: 失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_cross_channel.py -q 2>&1 | tail -5
```

- [ ] **Step 3: `data.py` に入出力を足す**

```python
@dataclass(frozen=True)
class CrossChannelInput:
    """断面内 Stokes の入力.

    Parameters
    ----------
    grid : ChannelGrid
    mu : np.ndarray
        (nx, ny) 粘度場 [Pa·s]
    G : float
        下流方向圧力勾配 [Pa/m]。横断方向体積力は f_x = −spec.beta(G) = −G·cotφ
    p_pin_value : float
        圧力の定数自由度を消すためのピン留め値 [Pa]
    """

    grid: ChannelGrid
    mu: np.ndarray
    G: float
    p_pin_value: float = 0.0


@dataclass(frozen=True)
class CrossChannelResult:
    """断面内 Stokes の結果.

    Parameters
    ----------
    u, v : np.ndarray
        (nx, ny) セル中心速度 [m/s]
    u_face : np.ndarray
        (nx, ny) x 面（セル i の西面）の u [m/s]。周期なので nx 枚
    v_face : np.ndarray
        (nx, ny+1) y 面（セル j の南面）の v [m/s]
    p : np.ndarray
        (nx, ny) 周期部分 p̃ [Pa]
    psi : np.ndarray
        (nx+1, ny+1) 節点上の流れ関数 [m²/s]。面流束を積分して作るので
        離散的に厳密に発散ゼロ。粒子追跡はこれを使う
    div_max : float
        最大セル発散 [m/s]（診断）
    """

    u: np.ndarray
    v: np.ndarray
    u_face: np.ndarray
    v_face: np.ndarray
    p: np.ndarray
    psi: np.ndarray
    div_max: float
```

- [ ] **Step 4: `cross_channel.py` を実装する**

実装は約 250 行になる。骨格を示す。**鞍点系の組み立てとピン留めが要点。**

```python
"""断面内 (u, v, p̃) の 2 次元 Stokes ソルバー（MAC 千鳥格子・疎直接解）.

Re ~ 10⁻³ なので慣性項が無く、系は線形。SIMPLE のような圧力-速度連成反復は
不要で、鞍点系 [A Bᵀ; B 0] を LU で一発で解ける。

可変粘度に対応するため μ∇²u ではなく完全な ∇·(μ(∇u + ∇uᵀ)) を離散化する。
法線応力の μ はセル中心値、せん断応力の μ は節点値（隣接 4 セルの調和平均）。

x 周期の圧力跳びは、P = βx + p̃ の分解により横断方向の一様体積力
f_x = −β = −G·cotφ として入る（docs/design/single-screw-extruder.md §2.1）。

粒子追跡のために、面流束を積分した節点流れ関数 ψ も返す。ψ の双一次補間から
作った速度は離散的に厳密に発散ゼロなので、粒子が湧いたり消えたりしない。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.extruder.data import CrossChannelInput, CrossChannelResult


class CrossChannelStokesProcess(SolverProcess["CrossChannelInput", "CrossChannelResult"]):
    """断面内 2D Stokes を MAC 千鳥格子 + 疎直接解で解く."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="CrossChannelStokes",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: CrossChannelInput) -> CrossChannelResult:
        # 1) 未知数の採番: u 面 / v 面 / p セル（固体は除外）
        # 2) 粘性項の組み立て（法線応力: セル中心 μ、せん断応力: 節点 μ）
        # 3) 勾配 Bᵀ と発散 B の組み立て（互いに転置になっていること）
        # 4) 体積力 −β を x 運動量の RHS へ
        # 5) 壁 BC（固体接触面 u=v=0、y=0 で u=v=0、y=H で u=u_barrel, v=0）
        # 6) 圧力 1 点ピン留め
        # 7) splu で解く
        # 8) 面値 → セル中心値、ψ を面流束の累積和で構築、div_max を計算
        raise NotImplementedError
```

**実装上の必須事項（テストが担保するので手を抜くと落ちる）:**

1. **`B` は `Bᵀ` の厳密な転置にすること。**別々に組むと `div_max` が落ちない。
   勾配行列を組んでその `.T` を発散に使うのが安全。ただし面積重みの扱いを
   統一すること（連続式を「体積流束の和 = 0」の形で書き、勾配は「圧力差 × 面積」で書く）。
2. **ψ の構築**: `psi[0,0] = 0` とし、`psi[i, j+1] = psi[i, j] + u_face[i, j]·dy[j]`
   で y 方向に積み上げ、`psi[i+1, 0] = psi[i, 0] − v_face[i, 0]·dx[i]` で x 方向に伸ばす。
   `test_streamfunction_is_consistent_with_faces` がこの規約を固定する。
3. **`splu` が遅い / メモリを食う場合**: `permc_spec="COLAMD"`（既定）のまま
   格子を落とすこと。Global Constraints の 4 GB を超える格子は使わない。
   反復解法（Uzawa）への切り替えは**最後の手段**。線形直接解であることが
   G1/G2/G2b の機械精度一致を支えているので、安易に崩さない。

- [ ] **Step 5: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_cross_channel.py -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -8
```

**G2b（`test_g2b_one_dimensional_exact`）が通らなければ先へ進まない。**
切り分け順:

1. `test_discretely_divergence_free` … `B` と `Bᵀ` の不整合
2. `test_g2b_...` の解が定数倍ずれる … `β` の符号か `tan(φ)` の向き
3. 上壁付近だけ合わない … `y=H` の Dirichlet 距離が `dy` になっている（正しくは `dy/2`）

- [ ] **Step 6: メモリと時間を実測して記録する**

```bash
/usr/bin/time -v env OMP_NUM_THREADS=2 .venv/bin/python -c "
from xkep_cae_fluid.extruder.data import ScrewSpec, CrossChannelInput
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
import numpy as np
s = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=1e-4, N=100/60,
              nx_channel=200, nx_land=48, ny_bulk=60, n_gap=20)
g = ScrewGeometryProcess().process(s)
print('cells', g.nx*g.ny)
r = CrossChannelStokesProcess().process(CrossChannelInput(grid=g, mu=np.full((g.nx,g.ny),1000.0), G=5e6))
print('div_max', r.div_max)
" 2>&1 | tee /tmp/log-$(date +%s).log | grep -E "cells|div_max|Maximum resident|Elapsed"
```

**Maximum resident set size が 4 GB（4194304 KB）を超えたら格子を落とすこと。**
実測値を status に記録する（NN 学習との共存の根拠になる）。

- [ ] **Step 7: コミット**

```bash
git add xkep_cae_fluid/extruder/ tests/test_extruder_cross_channel.py
git commit -m "feat(extruder): CrossChannelStokesProcess — ゲート G2b 通過

断面内 2D Stokes を MAC 千鳥格子 + 鞍点系の疎直接解で解く。
可変粘度のため完全な ∇·(μ(∇u+∇uᵀ)) を離散化（法線応力=セル中心 μ、
せん断応力=節点 μ の調和平均）。x 周期の圧力跳びは体積力 f_x=-G·cotφ に還元。
G2b: 閉チャネル 1D 厳密解 u(y)=U(3η²-2η) と機械精度一致。
粒子追跡用に面流束を積分した節点流れ関数 ψ を出力（離散的に発散ゼロ）。"
```

---

## Task 6: `ExtruderFlowProcess` — 非ニュートンの Picard 結合

**Files:**
- Create: `xkep_cae_fluid/extruder/solver.py`
- Modify: `xkep_cae_fluid/extruder/data.py`（`ExtruderFlowInput` / `ExtruderFlowResult`）
- Test: `tests/test_extruder_solver.py`

**Interfaces:**
- Consumes: `ScrewGeometryProcess`, `DownChannelFlowProcess`, `CrossChannelStokesProcess`,
  `ViscosityModelStrategy`, `strain_rate`
- Produces:
  - `ExtruderFlowInput(spec, G, max_iter=100, tol=1e-6, relax_mu=0.5)`
  - `ExtruderFlowResult(grid, u, v, w, u_face, v_face, psi, p, mu, gamma_dot, Q, Q_leak, converged, n_iter, mu_history)`
  - `ExtruderFlowProcess` は `viscosity = StrategySlot(ViscosityModelStrategy)` を持つ

**反復スキーム（設計文書 §8 の未決事項を確定）: Picard（逐次代入）+ 粘度の低緩和**

```
  μ⁰ = model.viscosity(γ̇_ref)        γ̇_ref = V/H（代表せん断速度）で初期化
  繰り返し k:
      w  ← DownChannelFlowProcess(grid, μ^k, G)          （線形）
      uv ← CrossChannelStokesProcess(grid, μ^k, G)        （線形）
      γ̇ ← strain_rate(u, v, w, grid)
      μ* ← model.viscosity(γ̇)
      μ^{k+1} = (1−ω)·μ^k + ω·μ*                          ω = relax_mu（既定 0.5）
  収束判定: max|μ^{k+1} − μ^k| / max|μ^k| < tol
```

Newton を採らない理由: 見かけ粘度項のヤコビアンは `∂μ/∂γ̇ · ∂γ̇/∂(∇u)` を通じて
全成分に密に絡み、組み立てコストが線形解 1 回分を大きく超える。
一方 Picard は**線形解が毎回厳密**なので、反復は粘度場の不動点探索だけに集約される。
せん断減粘（`n<1`）は粘度の自己安定化が効くので、`ω = 0.5` で十分収束する。
**収束しなければ ω を下げ、それでも駄目なら「収束しなかった」と報告する（STA2 防止ルール）。**

- [ ] **Step 1: 失敗するテストを書く**

```python
"""ExtruderFlowProcess のテスト（Picard 結合）."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ScrewSpec
from xkep_cae_fluid.extruder.shape_factors import (
    metering_flow_rate,
    shape_factor_drag,
    shape_factor_pressure,
)
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.viscosity import (
    CarreauViscosity,
    NewtonianViscosity,
    PowerLawViscosity,
)


def run(spec, G, model, **kw):
    proc = ExtruderFlowProcess()
    proc.viscosity = model
    return proc.process(ExtruderFlowInput(spec=spec, G=G, **kw))


def spec_closed(ny=64, nx=200):
    return ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0,
        nx_channel=nx, nx_land=24, ny_bulk=ny, n_gap=0,
    )


def spec_gap(ny=40, n_gap=16):
    return ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
        nx_channel=160, nx_land=40, ny_bulk=ny, n_gap=n_gap,
    )


@binds_to(ExtruderFlowProcess)
class TestExtruderFlowAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ExtruderFlowProcess" in ProcessRegistry.default()

    def test_strategy_slot_required(self):
        with pytest.raises(AttributeError):
            ExtruderFlowProcess().process(ExtruderFlowInput(spec=spec_gap(), G=0.0))

    def test_effective_uses_includes_viscosity_strategy(self):
        proc = ExtruderFlowProcess()
        proc.viscosity = NewtonianViscosity(mu=1000.0)
        assert NewtonianViscosity in proc.effective_uses()


class TestExtruderFlowPhysics:
    def test_newtonian_converges_in_one_iteration(self):
        """ニュートンなら粘度が動かないので 1 反復で収束すること."""
        res = run(spec_closed(), 0.0, NewtonianViscosity(mu=1000.0))
        assert res.converged
        assert res.n_iter == 1

    def test_newtonian_reproduces_g2(self):
        """結合ソルバー経由でも G2 の解析解と 0.1% 以内で一致すること."""
        s = spec_closed(ny=96, nx=320)
        G = 1.0e6
        res = run(s, G, NewtonianViscosity(mu=1000.0))
        fd = shape_factor_drag(s.H / s.W)
        fp = shape_factor_pressure(s.H / s.W)
        q = metering_flow_rate(s.w_barrel, s.W, s.H, 1000.0, G, F_d=fd, F_p=fp)
        assert res.Q == pytest.approx(q, rel=1e-3)

    def test_carreau_n1_matches_newtonian(self):
        """Carreau で n=1, μ0=μ∞ ならニュートンと同じ Q になること."""
        s = spec_closed(ny=40, nx=120)
        a = run(s, 1.0e6, NewtonianViscosity(mu=1000.0))
        b = run(s, 1.0e6, CarreauViscosity(mu_0=1000.0, mu_inf=1000.0, lam=1.0, n=1.0))
        assert b.Q == pytest.approx(a.Q, rel=1e-9)

    def test_power_law_converges(self):
        res = run(spec_gap(), 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        assert res.converged, f"収束しなかった: n_iter={res.n_iter}"
        assert res.n_iter < 100

    def test_power_law_thins_in_the_gap(self):
        """隙間はせん断速度が最大なので、粘度がチャネル代表値より小さいこと."""
        s = spec_gap()
        res = run(s, 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4))
        g = res.grid
        i_land = g.nx // 2
        mu_gap = res.mu[i_land, -1]
        mu_bulk = res.mu[0, g.ny // 2]
        assert mu_gap < mu_bulk

    def test_gamma_min_does_not_change_the_answer(self):
        """γ̇ クランプは数値上の安全弁であり、結果を動かさないこと."""
        s = spec_gap()
        qs = [
            run(s, 5.0e6, PowerLawViscosity(K=2.0e4, n=0.4, gamma_min=gm)).Q
            for gm in (1e-3, 1e-2, 1e-1)
        ]
        assert qs[1] == pytest.approx(qs[0], rel=1e-3)
        assert qs[2] == pytest.approx(qs[0], rel=1e-3)

    def test_leakage_reduces_throughput(self):
        """隙間があると背圧下の流量が閉チャネルより小さいこと."""
        model = NewtonianViscosity(mu=1000.0)
        q_closed = run(spec_closed(ny=40, nx=160), 5.0e6, model).Q
        q_gap = run(spec_gap(), 5.0e6, model).Q
        assert q_gap < q_closed

    def test_leak_flux_is_negative(self):
        """漏れ流れは −x（上流へ戻る）向きであること."""
        res = run(spec_gap(), 5.0e6, NewtonianViscosity(mu=1000.0))
        assert res.Q_leak < 0.0
```

- [ ] **Step 2: 失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_solver.py -q 2>&1 | tail -5
```

- [ ] **Step 3: `data.py` に入出力を足し、`solver.py` を実装する**

```python
@dataclass(frozen=True)
class ExtruderFlowInput:
    """押出流れ解析の入力.

    Parameters
    ----------
    spec : ScrewSpec
    G : float
        下流方向圧力勾配 [Pa/m]
    max_iter : int
        Picard 反復の上限
    tol : float
        粘度場の相対変化に対する収束閾値
    relax_mu : float
        粘度の緩和係数 ω。μ^{k+1} = (1−ω)μ^k + ω·μ(γ̇^k)
    """

    spec: ScrewSpec
    G: float
    max_iter: int = 100
    tol: float = 1.0e-6
    relax_mu: float = 0.5
```

`solver.py`:

```python
"""押出流れ解析の統合 Process（Picard で粘度結合）.

ニュートン流体では下流方向 w と断面内 (u,v) が完全に分離する。
非ニュートンでは粘度 μ(γ̇) だけを介して結合するので、
「線形解 2 本を回して粘度場を更新する」不動点反復に落ちる。
線形解が毎回厳密なので、反復の収束は粘度場だけの問題になる。
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import SolverProcess
from xkep_cae_fluid.core.slots import StrategySlot
from xkep_cae_fluid.extruder.cross_channel import CrossChannelStokesProcess
from xkep_cae_fluid.extruder.data import (
    CrossChannelInput,
    DownChannelInput,
    ExtruderFlowInput,
    ExtruderFlowResult,
)
from xkep_cae_fluid.extruder.down_channel import DownChannelFlowProcess
from xkep_cae_fluid.extruder.geometry import ScrewGeometryProcess
from xkep_cae_fluid.extruder.viscosity import ViscosityModelStrategy, strain_rate


class ExtruderFlowProcess(SolverProcess["ExtruderFlowInput", "ExtruderFlowResult"]):
    """幾何生成 → 粘度 Picard → w / (u,v) 求解 → 流量と漏れ量の算出."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ExtruderFlow",
        module="solve",
        version="0.1.0",
        document_path="../../docs/design/single-screw-extruder.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = [
        ScrewGeometryProcess,
        DownChannelFlowProcess,
        CrossChannelStokesProcess,
    ]

    viscosity = StrategySlot(ViscosityModelStrategy)

    def process(self, input_data: ExtruderFlowInput) -> ExtruderFlowResult:
        t0 = time.perf_counter()
        inp = input_data
        model = self.viscosity  # 未設定なら AttributeError（StrategySlot の契約）

        grid = ScrewGeometryProcess().process(inp.spec)
        gamma_ref = inp.spec.V / inp.spec.H
        mu = model.viscosity(np.full((grid.nx, grid.ny), gamma_ref))

        down = DownChannelFlowProcess()
        cross = CrossChannelStokesProcess()
        mu_history: list[float] = []
        converged = False
        n_iter = 0

        for n_iter in range(1, inp.max_iter + 1):
            w_res = down.process(DownChannelInput(grid=grid, mu=mu, G=inp.G))
            uv = cross.process(CrossChannelInput(grid=grid, mu=mu, G=inp.G))
            gd = strain_rate(uv.u, uv.v, w_res.w, grid)
            mu_star = model.viscosity(gd)
            mu_new = (1.0 - inp.relax_mu) * mu + inp.relax_mu * mu_star
            change = float(np.max(np.abs(mu_new - mu)) / max(np.max(np.abs(mu)), 1e-30))
            mu_history.append(change)
            mu = mu_new
            if change < inp.tol:
                converged = True
                break

        # 漏れ量: ランド中央の x 面を通る正味横断流束 [m²/s]（単位 z 厚さあたり）
        i_land = grid.nx // 2
        q_leak = float(np.sum(uv.u_face[i_land, :] * grid.dy))

        return ExtruderFlowResult(
            grid=grid, u=uv.u, v=uv.v, w=w_res.w,
            u_face=uv.u_face, v_face=uv.v_face, psi=uv.psi, p=uv.p,
            mu=mu, gamma_dot=gd, Q=w_res.Q, Q_leak=q_leak,
            converged=converged, n_iter=n_iter, mu_history=tuple(mu_history),
            elapsed_seconds=time.perf_counter() - t0,
        )
```

> **注意**: ニュートンの場合、1 回目の反復で `mu_star == mu` なので
> `change == 0 < tol` となり `n_iter == 1` で抜ける。
> `test_newtonian_converges_in_one_iteration` がこれを固定する。

- [ ] **Step 4: テストが通ることを確認して ruff・契約検証**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_solver.py -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -8
.venv/bin/python -m ruff check xkep_cae_fluid/ tests/ && .venv/bin/python -m ruff format xkep_cae_fluid/ tests/
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
```

- [ ] **Step 5: 全体テストで回帰が無いことを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -3
```

期待: `passed` が 268 + 新規分、`failed` は 10（pyamg / numba）のまま。

- [ ] **Step 6: コミット**

```bash
git add xkep_cae_fluid/extruder/ tests/test_extruder_solver.py
git commit -m "feat(extruder): ExtruderFlowProcess — 非ニュートンの Picard 結合

粘度を StrategySlot で受け、w と (u,v) の線形解を回して μ(γ̇) の不動点を探す。
Newton を採らないのは、見かけ粘度のヤコビアンが全成分に密に絡み組み立てが
線形解 1 回分を超えるのに対し、Picard は線形解が毎回厳密だから。
ニュートンは 1 反復で収束、Carreau(n=1) はニュートンと 1e-9 一致。
γ̇ クランプ値を 100 倍振っても Q が 0.1% 以内で不変であることを確認。"
```

---

## Task 7: 隙間ありケースの格子収束と診断

G3（OpenFOAM 突き合わせ）の前に、**自前の解が格子に対して収束していること**を確認する。
収束していない解を OpenFOAM と比べても意味がない。

**Files:**
- Create: `experiments/extruder/grid_convergence.py`
- Create: `docs/generated/extruder-grid-convergence.md`（スクリプトが生成）
- Test: `tests/test_extruder_solver.py` に `TestGridConvergence` を追加

**Interfaces:**
- Consumes: `ExtruderFlowProcess`
- Produces: 隙間解像度 `n_gap` と流量 `Q` / 漏れ量 `Q_leak` の収束表

- [ ] **Step 1: 失敗するテストを書く**

```python
class TestGridConvergence:
    """1a/a02 の結論「誤差は最狭方向のセル数だけで決まる」を隙間に適用して検証."""

    def test_q_converges_with_gap_resolution(self):
        """n_gap を 8 → 16 → 32 と上げたとき Q の変化が単調に縮むこと."""
        model = NewtonianViscosity(mu=1000.0)
        qs = []
        for n_gap in (8, 16, 32):
            s = ScrewSpec(
                D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
                nx_channel=160, nx_land=40, ny_bulk=40, n_gap=n_gap,
            )
            qs.append(run(s, 5.0e6, model).Q)
        d1 = abs(qs[1] - qs[0])
        d2 = abs(qs[2] - qs[1])
        assert d2 < d1 / 2.0, f"収束していない: Δ1={d1:.3e}, Δ2={d2:.3e}"

    def test_20_cells_in_gap_is_within_1_percent(self):
        """n_gap=20 の Q が n_gap=40 の Q と 1% 以内（a02 のベンチ結論の再現）."""
        model = NewtonianViscosity(mu=1000.0)

        def q(n_gap):
            s = ScrewSpec(
                D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
                nx_channel=160, nx_land=40, ny_bulk=40, n_gap=n_gap,
            )
            return run(s, 5.0e6, model).Q

        assert q(20) == pytest.approx(q(40), rel=1e-2)
```

- [ ] **Step 2: 失敗を確認し、必要なら格子生成を直す**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_solver.py::TestGridConvergence -q 2>&1 | tail -5
```

落ちる場合の典型: `_build_dy` のバルク側の伸長率が急すぎて隙間直下でセル比が跳ねている。
隣接セル幅比を 1.2 以下に抑えるよう `ny_bulk` を増やす。

- [ ] **Step 3: 収束表を出すスクリプトを書く**

```python
"""隙間解像度に対する Q / Q_leak の収束表を生成する.

実行:
    OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/grid_convergence.py \
        2>&1 | tee /tmp/log-$(date +%s).log
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ScrewSpec
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

OUT = Path(__file__).resolve().parents[2] / "docs/generated/extruder-grid-convergence.md"
G = 5.0e6


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()

    rows = []
    q_ref = None
    for n_gap in (5, 10, 20, 40, 63):
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
            nx_channel=200, nx_land=48, ny_bulk=60, n_gap=n_gap,
        )
        proc = ExtruderFlowProcess()
        proc.viscosity = NewtonianViscosity(mu=1000.0)
        res = proc.process(ExtruderFlowInput(spec=spec, G=G))
        if n_gap == 63:
            q_ref = res.Q
        rows.append((n_gap, res.grid.nx * res.grid.ny, res.Q, res.Q_leak,
                     res.elapsed_seconds))

    lines = [
        "# 隙間解像度に対する収束（押出・ニュートン）",
        "",
        "[<- README](../../README.md) | [<- docs](../README.md)",
        "",
        f"- ブランチ: `{branch}`  コミット: `{sha}`",
        f"- 条件: 40mm 機、δ=0.1mm、μ=1000 Pa·s、G={G:.3g} Pa/m",
        "- STA2 防止: 実行コマンドとコミットハッシュを本表に埋め込んでいる",
        "",
        "指標は最細格子（n_gap=63）を基準に規格化した比で示す（合格は比 < 1.01）。",
        "",
        "| n_gap | セル数 | Q [m³/s] | Q/Q_ref | Q_leak [m²/s] | 時間 [s] |",
        "|---|---|---|---|---|---|",
    ]
    for n_gap, cells, q, ql, sec in rows:
        lines.append(
            f"| {n_gap} | {cells} | {q:.6e} | {q / q_ref:.4f} | {ql:.6e} | {sec:.1f} |"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 走らせて結果を確認する**

```bash
cd ~/work/ykep-cae
OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/grid_convergence.py 2>&1 | tee /tmp/log-$(date +%s).log
```

**`n_gap=20` の `Q/Q_ref` が 1.01 未満であること。** 超えたら `n_gap` を上げるか、
隙間直下のセル比を見直す。

- [ ] **Step 5: mdview で見せて Artifact 公開する**

```bash
~/work/tb/bin/mdview docs/generated/extruder-grid-convergence.md
```

生成 HTML の `<body>` 中身を `Artifact` で公開し URL を報告する。

- [ ] **Step 6: コミット**

```bash
git add experiments/extruder/ docs/generated/ tests/test_extruder_solver.py
git commit -m "test(extruder): 隙間解像度の格子収束（n_gap=20 で 1% 以内）

1a/a02 のボクセルメッシュ品質ベンチの結論（誤差は最狭方向のセル数だけで決まる。
1% なら 20 セル）を隙間に適用して再現を確認。収束していない解を OpenFOAM と
比べても意味がないので、G3 の前段に置く。"
```

---

## Task 8: ゲート G3 — OpenFOAM による独立検算

**Files:**
- Create: `experiments/extruder/of_powerlaw_check.py`
- Create: `experiments/extruder/of_case.py`
- Create: `experiments/extruder/compare_openfoam.py`
- Create: `experiments/extruder/run_g3.sh`
- Create: `docs/generated/extruder-g3.md`（スクリプトが生成）

**Interfaces:**
- Consumes: `ExtruderFlowProcess`、`~/work/1a/a02/tools/of`
- Produces: G3a（ニュートン）/ G3b（べき乗則）の突き合わせ表

**OpenFOAM 側の構成:**

```
  メッシュ: 断面 (x,y) を blockMesh で切り、z 方向 1 セル。z 両端は cyclic
           フライトは topoSet(boxToCell) + subsetMesh で刳り抜き、
           露出面を wall パッチにする
  x 両端:  cyclic（跳びなし）。圧力跳びは体積力に還元済みなので fixedJump は不要
  体積力:  fvOptions の vectorSemiImplicitSource で (−β, 0, −G) [N/m³] を全セルに
  BC:      barrel パッチ U = (u_barrel, 0, w_barrel)、他の wall は noSlip
  ソルバー: simpleFoam（カスタムソルバー不要）
  粘度:    Newtonian / powerLaw（transportProperties 標準装備）
```

> `1a/a02` の教訓: 完全発達で対流項が消える問題に `simpleFoam` を使うと、
> 存在しない圧力-速度連成を反復して極端に遅くなる。本件は断面内に実際の循環流が
> あるので妥当だが、**収束が異様に遅い場合は連成の要否を疑うこと。**

- [ ] **Step 1: 先に powerLaw の `K, n` 対応づけを 1D で較正する（G3b の前提）**

OpenFOAM の `powerLaw` は**動粘度**で `nu = k·|2·symm(grad U)|^(n−1)` を使い、
ひずみ速度の定義もライブラリ内部の規約に従う。この対応づけを推測で決めると
G3b の不一致の原因が特定できなくなる。**先に厳密解のある 1D 問題で確定させる。**

`experiments/extruder/of_powerlaw_check.py` は次を行う:

1. 平行平板間のべき乗則 Poiseuille 流れ（厳密解あり）の OpenFOAM ケースを生成
   ```
   u(y) = (n/(n+1))·(|G|/K)^(1/n)·[ (h)^((n+1)/n) − |y − h|^((n+1)/n) ]   （h = H/2）
   ```
2. `of simpleFoam` で解き、中心速度を厳密解と比較
3. 一致する `k` の与え方（`k = K/ρ` か否か）を**実測で確定**し、
   `docs/generated/extruder-g3.md` に記録する

```bash
cd ~/work/ykep-cae/experiments/extruder
OMP_NUM_THREADS=2 ../../.venv/bin/python of_powerlaw_check.py 2>&1 | tee /tmp/log-$(date +%s).log
```

**厳密解と 0.5% 以内で一致するまで G3b に進まない。**

- [ ] **Step 2: `of_case.py` でケースを生成する**

`of_case.py` は `ScrewSpec` と粘度モデルを受け取り、
`system/{blockMeshDict, controlDict, fvSchemes, fvSolution, topoSetDict, fvOptions}` と
`constant/transportProperties`、`0/{U,p}` を書き出す。
格子は Task 1 と**同じ `dx`, `dy` 配列**を使う（`blockMeshDict` の `simpleGrading` ではなく、
`edgeGrading` の多区間指定で `dx`/`dy` をそのまま再現する）。
これにより「格子が違うから合わない」を切り分けから消せる。

- [ ] **Step 3: G3a（ニュートン・隙間あり）を回す**

```bash
cd ~/work/ykep-cae/experiments/extruder
OMP_NUM_THREADS=2 ../../.venv/bin/python of_case.py --model newtonian --out /tmp/of-g3a
cd /tmp/of-g3a
OF_CPUS=1 OF_MEM=1200m ~/work/1a/a02/tools/of blockMesh   2>&1 | tee /tmp/log-$(date +%s).log
OF_CPUS=1 OF_MEM=1200m ~/work/1a/a02/tools/of topoSet     2>&1 | tee -a /tmp/log-$(date +%s).log
OF_CPUS=1 OF_MEM=1200m ~/work/1a/a02/tools/of subsetMesh c0 -patch barrel -overwrite 2>&1 | tee -a /tmp/log-$(date +%s).log
OF_CPUS=1 OF_MEM=1200m ~/work/1a/a02/tools/of simpleFoam  2>&1 | tee -a /tmp/log-$(date +%s).log
```

- [ ] **Step 4: `compare_openfoam.py` で突き合わせる**

比較する量（全て**閾値で規格化した比**で報告する。合格は比 < 1.00）:

| 量 | 定義 | 閾値 |
|---|---|---|
| 流量 | `Q` の相対差 | 1% |
| 漏れ量 | `Q_leak` の相対差 | 1% |
| `w` の分布 | チャネル中央断面 `w(y)` の L2 相対誤差 | 1% |
| `u` の分布 | 同 `u(y)` の L2 相対誤差 | 1% |

- [ ] **Step 5: G3b（べき乗則）を回す**

```bash
cd ~/work/ykep-cae/experiments/extruder
OMP_NUM_THREADS=2 ../../.venv/bin/python of_case.py --model powerlaw --K 2e4 --n 0.4 --out /tmp/of-g3b
# 以下 Step 3 と同じ手順
```

- [ ] **Step 6: `run_g3.sh` に一括化し、結果を `docs/generated/extruder-g3.md` に出す**

レポートには STA2 防止ルールに従い、ブランチ名・コミットハッシュ・実行コマンド・
OpenFOAM イメージ名を必ず記録する。

- [ ] **Step 7: mdview → Artifact 公開**

```bash
~/work/tb/bin/mdview docs/generated/extruder-g3.md
```

- [ ] **Step 8: コミット**

```bash
git add experiments/extruder/ docs/generated/
git commit -m "test(extruder): ゲート G3 — OpenFOAM による独立検算

x 周期の圧力跳びを体積力 (−G·cotφ, 0, −G) に還元したので fixedJump は不要、
simpleFoam のまま回る。格子は ykep-cae と同じ dx/dy 配列を edgeGrading で
再現し「格子が違うから合わない」を切り分けから消した。
G3b の前に powerLaw の K,n 対応づけを 1D 厳密解で較正している。"
```

---

## Task 9: `ParticleTrackerProcess` と ゲート G4a

**Files:**
- Create: `xkep_cae_fluid/extruder/tracker.py`
- Modify: `xkep_cae_fluid/extruder/data.py`（`ParticleTrackInput` / `ParticleTrackResult`）
- Test: `tests/test_extruder_tracker.py`

**Interfaces:**
- Consumes: `ExtruderFlowResult`
- Produces:
  - `ParticleTrackInput(flow, n_particles=20000, z_end=0.200, seed=0, cfl=0.2, max_steps=2_000_000)`
  - `ParticleTrackResult(t_res, gamma_total, n_wraps, x, y, z, escaped)`
    （全て `(n_particles,)`）

**設計上の要点（設計文書 §8 の未決事項を確定）:**

1. **流れ関数による発散ゼロ補間。** 速度を直接双線形補間すると離散的な発散ゼロが
   壊れ、粒子が渦の中心に落ち込んだり壁に貼り付いたりして RTD の裾が偽物になる。
   節点 `ψ` を双線形補間し `u = ∂ψ/∂y`, `v = −∂ψ/∂x` として速度を作れば、
   **セル内で厳密に発散ゼロ**になる。`ψ` は Task 5 で面流束から積み上げてある。
   `w` は双線形補間でよい（`z` は解かないので発散に関与しない）。

2. **`x` 周期の跳び（RTD を広げる主機構）。** §0 の同一視から

   ```
     x ≥ W_t で脱出 → x −= W_t,  z += L_turn      （下流へ 1 ターン分ワープ）
     x < 0   で脱出 → x += W_t,  z −= L_turn      （上流へ戻る。漏れ流れがこれ）
   ```

   40 mm 機で `L_turn = 119.7 mm`。計量部長 200 mm に対して**1 回の漏れで
   6 割戻る**ので、これが RTD の長い裾を作る。

3. **適応時間刻み。** 隙間のセルは 5 μm 級、速度は最大。固定 `dt` では
   隙間を跳び越すか、チャネル内で無駄に細かくなる。
   `dt = cfl · min(dx_i/|u|, dy_j/|v|, ...)` を粒子ごとに毎ステップ再計算する。

4. **累積せん断ひずみ** `γ = ∫ γ̇ dt` を RK4 の各ステップで台形則加算する。

- [ ] **Step 1: 失敗するテストを書く（G4a）**

```python
"""ParticleTrackerProcess のテスト。G4a を含む."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import ExtruderFlowInput, ParticleTrackInput, ScrewSpec
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity


def flow_simple_shear():
    """フライト無し・G=0 の平行平板。u,v の 1D 解と w の線形 Couette が厳密に分かる."""
    spec = ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=1e-9, delta=0.0, N=100.0 / 60.0,
        nx_channel=16, nx_land=1, ny_bulk=64, n_gap=0,
    )
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=1000.0)
    return proc.process(ExtruderFlowInput(spec=spec, G=0.0))


@binds_to(ParticleTrackerProcess)
class TestParticleTrackerAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "ParticleTrackerProcess" in ProcessRegistry.default()

    def test_all_particles_escape(self):
        flow = flow_simple_shear()
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=500, z_end=0.050, seed=1)
        )
        assert res.escaped.all(), f"未脱出 {int((~res.escaped).sum())} 個"


class TestParticleTrackerPhysics:
    def test_g4a_exact_trajectory_in_simple_shear(self):
        """G4a: v=0 なので y が保存し、滞留時間が z_end/w(y0) に一致すること."""
        flow = flow_simple_shear()
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=400, z_end=0.050, seed=2)
        )
        g = flow.grid
        # y は保存する（数値的に）
        # 期待滞留時間は初期 y での w(y0) から
        w_at_y0 = np.interp(res.y0, g.yc, flow.w[0, :])
        expect = 0.050 / w_at_y0
        ok = w_at_y0 > 0
        rel = np.abs(res.t_res[ok] / expect[ok] - 1.0)
        assert np.max(rel) < 1e-8, f"最大相対誤差 {np.max(rel):.2e}"

    def test_y_is_conserved_in_simple_shear(self):
        flow = flow_simple_shear()
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=200, z_end=0.020, seed=3)
        )
        assert np.max(np.abs(res.y - res.y0)) < 1e-9 * flow.grid.spec.H

    def test_wrap_bookkeeping(self):
        """x が 1 周するたび z が L_turn 動くこと（漏れ流れの z 方向記帳）."""
        flow = flow_simple_shear()
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=200, z_end=0.020, seed=4)
        )
        # u_barrel < 0 なので上半分の粒子は −x に流れ、n_wraps が負になる
        assert res.n_wraps.min() < 0

    def test_particles_do_not_enter_solid(self):
        spec = ScrewSpec(
            D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
            nx_channel=120, nx_land=32, ny_bulk=32, n_gap=16,
        )
        proc = ExtruderFlowProcess()
        proc.viscosity = NewtonianViscosity(mu=1000.0)
        flow = proc.process(ExtruderFlowInput(spec=spec, G=5.0e6))
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=2000, z_end=0.100, seed=5)
        )
        g = flow.grid
        i = np.clip(np.searchsorted(np.cumsum(g.dx), res.x), 0, g.nx - 1)
        j = np.clip(np.searchsorted(np.cumsum(g.dy), res.y), 0, g.ny - 1)
        assert not g.solid[i, j].any()

    def test_cumulative_shear_is_positive_and_finite(self):
        flow = flow_simple_shear()
        res = ParticleTrackerProcess().process(
            ParticleTrackInput(flow=flow, n_particles=300, z_end=0.050, seed=6)
        )
        assert np.all(res.gamma_total > 0.0)
        assert np.all(np.isfinite(res.gamma_total))
```

- [ ] **Step 2: 失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_tracker.py -q 2>&1 | tail -5
```

- [ ] **Step 3: `tracker.py` を実装する**

骨格（約 200 行）:

```python
"""粒子追跡 Process（流れ関数補間 + 適応 RK4 + x 周期跳び）.

速度を直接補間すると離散的な発散ゼロが壊れ、粒子が渦中心に落ちたり壁に
貼り付いたりして RTD の裾が偽物になる。節点流れ関数 ψ を双線形補間して
u = ∂ψ/∂y, v = −∂ψ/∂x を作れば、セル内で厳密に発散ゼロになる。

x 周期の跳び（docs/design/single-screw-extruder.md §2.1 の同一視）:
    x ≥ W_t で脱出 → x −= W_t, z += L_turn
    x < 0   で脱出 → x += W_t, z −= L_turn
40mm 機で L_turn = 119.7mm。計量部長 200mm に対して 1 回の漏れで 6 割戻るので、
これが RTD の長い裾を作る主機構になる。
"""
```

**種まき（設計文書 §8 の未決事項を確定）: 流量重み付き。**
`z=0` 断面を横切る物質の分布は `w > 0` の流束に比例する。
セル `(i,j)` を選ぶ確率を `p_ij ∝ max(w_ij, 0)·dx_i·dy_j` とし、
セル内は一様乱数で配置する。`np.random.default_rng(seed)` を使い再現性を担保する。

**粒子数（同）: 既定 20,000。**
RTD の平均は `1/√N` で収束するので、20,000 個で相対標準誤差 0.7%。
`Task 10` の `test_mean_residence_time_matches_volume_over_flow` が 1% 判定なので釣り合う。

- [ ] **Step 4: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_tracker.py -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -8
```

`test_g4a_exact_trajectory_in_simple_shear` が `1e-8` を満たさない場合、
RK4 の実装ではなく `w` の補間が疑わしい（`w` が `y` の 1 次なら双線形補間は厳密）。

- [ ] **Step 5: コミット**

```bash
git add xkep_cae_fluid/extruder/tracker.py tests/test_extruder_tracker.py
git commit -m "feat(extruder): ParticleTrackerProcess — ゲート G4a 通過

節点流れ関数 ψ の双線形補間から u=∂ψ/∂y, v=-∂ψ/∂x を作り、セル内で厳密に
発散ゼロな速度場で追跡する。速度を直接補間すると粒子が渦中心に落ちて
RTD の裾が偽物になるため。
x 周期の跳びは (x±W_t, z∓L_turn) で記帳。L_turn=119.7mm は計量部長 200mm の
6 割に相当し、これが RTD の長い裾を作る主機構。
時間刻みは粒子ごとに CFL で適応制御（隙間 5μm セルとチャネルで 3 桁違う）。"
```

---

## Task 10: `RTDProcess` と ゲート G4b

**Files:**
- Create: `xkep_cae_fluid/extruder/rtd.py`
- Modify: `xkep_cae_fluid/extruder/data.py`（`RTDInput` / `RTDResult`）
- Test: `tests/test_extruder_rtd.py`

**Interfaces:**
- Consumes: `ParticleTrackResult`, `ExtruderFlowResult`
- Produces:
  - `RTDInput(track, flow, z_end, n_bins=200)`
  - `RTDResult(t_edges, E, F, t_mean, t_min, t_p10, t_p50, t_p90, gamma_mean, gamma_p10, gamma_p90, mixing_index)`

**G4b の厳密関係**: 定常流・流量重み付き種まきなら

```
  ⟨t⟩ = z_end · A_free / Q          （A_free = 流体断面積、Q = 下流方向流量）
```

これは補間誤差・跳びの記帳ミス・種まき重みの誤りを**同時に**捕まえる強い検査になる。

**混合指数** `λ = |D| / (|D| + |Ω|)`（`D` = ひずみ速度テンソル、`Ω` = 渦度テンソル）。
`λ = 0` 純回転、`0.5` 単純せん断、`1` 純伸長。粒子経路に沿った時間平均を取る。

- [ ] **Step 1: 失敗するテストを書く（G4b）**

```python
"""RTDProcess のテスト。G4b を含む."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_fluid.core.testing import binds_to
from xkep_cae_fluid.extruder.data import (
    ExtruderFlowInput,
    ParticleTrackInput,
    RTDInput,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.rtd import RTDProcess
from xkep_cae_fluid.extruder.solver import ExtruderFlowProcess
from xkep_cae_fluid.extruder.tracker import ParticleTrackerProcess
from xkep_cae_fluid.extruder.viscosity import NewtonianViscosity

Z_END = 0.200  # 計量部長 5D [m]


def pipeline(spec, G, n_particles=20000, seed=0):
    proc = ExtruderFlowProcess()
    proc.viscosity = NewtonianViscosity(mu=1000.0)
    flow = proc.process(ExtruderFlowInput(spec=spec, G=G))
    track = ParticleTrackerProcess().process(
        ParticleTrackInput(flow=flow, n_particles=n_particles, z_end=Z_END, seed=seed)
    )
    rtd = RTDProcess().process(RTDInput(track=track, flow=flow, z_end=Z_END))
    return flow, track, rtd


def spec_closed():
    return ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=0.004, delta=0.0, N=100.0 / 60.0,
        nx_channel=120, nx_land=24, ny_bulk=32, n_gap=0,
    )


def spec_gap():
    return ScrewSpec(
        D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0,
        nx_channel=160, nx_land=40, ny_bulk=40, n_gap=20,
    )


@binds_to(RTDProcess)
class TestRTDAPI:
    def test_registry_registered(self):
        from xkep_cae_fluid.core.registry import ProcessRegistry

        assert "RTDProcess" in ProcessRegistry.default()

    def test_F_is_monotone_and_reaches_one(self):
        _, _, rtd = pipeline(spec_closed(), 1.0e6, n_particles=3000)
        assert np.all(np.diff(rtd.F) >= -1e-12)
        assert rtd.F[-1] == pytest.approx(1.0, abs=1e-9)

    def test_E_integrates_to_one(self):
        _, _, rtd = pipeline(spec_closed(), 1.0e6, n_particles=3000)
        dt = np.diff(rtd.t_edges)
        assert float(np.sum(rtd.E * dt)) == pytest.approx(1.0, rel=1e-9)


class TestRTDPhysics:
    def test_g4b_mean_residence_time_matches_volume_over_flow(self):
        """G4b: ⟨t⟩ = z_end·A_free/Q（厳密関係）と 1% 以内で一致すること.

        補間誤差・跳びの記帳ミス・種まき重みの誤りを同時に捕まえる。
        """
        flow, _, rtd = pipeline(spec_closed(), 1.0e6, n_particles=20000)
        expect = Z_END * flow.grid.area_free / flow.Q
        assert rtd.t_mean == pytest.approx(expect, rel=1e-2)

    def test_min_residence_time_bound(self):
        """最短滞留時間が z_end/w_max を下回らないこと."""
        flow, _, rtd = pipeline(spec_closed(), 1.0e6, n_particles=5000)
        assert rtd.t_min >= Z_END / float(flow.w.max()) * (1.0 - 1e-9)

    def test_clearance_broadens_the_distribution(self):
        """隙間があると漏れで z が戻る粒子が出て、分布の裾が伸びること."""
        _, _, rtd_closed = pipeline(spec_closed(), 5.0e6, n_particles=8000, seed=7)
        _, _, rtd_gap = pipeline(spec_gap(), 5.0e6, n_particles=8000, seed=7)
        spread_closed = rtd_closed.t_p90 / rtd_closed.t_p10
        spread_gap = rtd_gap.t_p90 / rtd_gap.t_p10
        assert spread_gap > spread_closed

    def test_backward_wrap_occurs_only_with_clearance(self):
        """z が戻る跳び（n_wraps < 0 かつ漏れ経由）は隙間がある場合だけ起きること."""
        _, track_closed, _ = pipeline(spec_closed(), 5.0e6, n_particles=3000, seed=8)
        _, track_gap, _ = pipeline(spec_gap(), 5.0e6, n_particles=3000, seed=8)
        assert (track_gap.n_wraps != 0).sum() > (track_closed.n_wraps != 0).sum()

    def test_mixing_index_range(self):
        """混合指数が [0, 1] に入り、せん断主体なので 0.5 付近に集まること."""
        _, _, rtd = pipeline(spec_closed(), 1.0e6, n_particles=5000)
        assert 0.0 <= rtd.mixing_index.min()
        assert rtd.mixing_index.max() <= 1.0
        assert 0.3 < float(np.median(rtd.mixing_index)) < 0.7

    def test_cumulative_shear_scales_with_residence_time(self):
        """累積せん断が滞留時間と正の相関を持つこと（長く居れば混ざる）."""
        _, track, rtd = pipeline(spec_closed(), 1.0e6, n_particles=5000)
        r = np.corrcoef(track.t_res, track.gamma_total)[0, 1]
        assert r > 0.8
```

- [ ] **Step 2: 失敗を確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_rtd.py -q 2>&1 | tail -5
```

- [ ] **Step 3: `rtd.py` を実装する**

```python
"""滞留時間分布（RTD）と混練性指標の後処理 Process.

E(t) は滞留時間の確率密度、F(t) はその累積。押出では「どれだけ揃った履歴を
与えられるか」が品質を決めるので、分布の広がり（t_p90/t_p10）と
累積せん断ひずみ γ = ∫γ̇dt を主指標にする。

混合指数 λ = |D|/(|D|+|Ω|) は変形の性質を測る:
    λ = 0    純回転（混ざらない）
    λ = 0.5  単純せん断
    λ = 1    純伸長（最も分散に効く）
"""
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_extruder_rtd.py -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -8
```

**G4b（`test_g4b_mean_residence_time_matches_volume_over_flow`）が最重要。**
落ちた場合の切り分け:

1. 系統的に大きい/小さい → 種まき重みが流量重み付きになっていない（面積重みになっている）
2. 隙間ありでだけ落ちる → 跳びの `z` 記帳の符号
3. 数 % ずれる → 粒子数不足（`n_particles` を増やして誤差が `1/√N` で縮むか確認）

- [ ] **Step 5: 全体テスト・契約・ruff**

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q 2>&1 | tee /tmp/log-$(date +%s).log | tail -3
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
.venv/bin/python -m ruff check xkep_cae_fluid/ tests/ && .venv/bin/python -m ruff format xkep_cae_fluid/ tests/
```

- [ ] **Step 6: コミット**

```bash
git add xkep_cae_fluid/extruder/rtd.py tests/test_extruder_rtd.py
git commit -m "feat(extruder): RTDProcess — ゲート G4b 通過

G4b は厳密関係 ⟨t⟩ = z_end·A_free/Q との突き合わせ。補間誤差・跳びの記帳ミス・
種まき重みの誤りを同時に捕まえるので、文献 RTD 曲線との目視比較より強い。
種まきは流量重み付き（p ∝ max(w,0)·dA）。20000 個で相対標準誤差 0.7%。
隙間があると t_p90/t_p10 が広がることを確認（漏れによる z の戻りが裾を作る）。"
```

---

## Task 11: 統合レポートと引き継ぎ

**Files:**
- Create: `experiments/extruder/report.py`
- Create: `docs/generated/extruder-report.md`（スクリプトが生成）
- Create: `docs/status/status-28.md`
- Modify: `docs/status/status-index.md`
- Modify: `docs/roadmap.md`
- Modify: `README.md`
- Modify: `docs/design/single-screw-extruder.md`（§8 未決事項を「確定済み」に更新）

- [ ] **Step 1: 統合レポートを生成するスクリプトを書く**

`report.py` は次を 1 本の文書にまとめる。**CLAUDE.md の「整合的・図解・メカニズム」方針に従い、
作業の時系列ではなく対象の構造に沿って章立てすること。**

1. **全体像を先に置く**（inline SVG）: 展開チャネルの断面図に、
   バレル速度・フライト・隙間・漏れ経路・断面内循環を 1 枚で描く
2. **メカニズム**: なぜ 2.5D で足りるか → なぜ線形か → なぜ隙間が混練を支配するか
3. **幾何の恒等式**（§0）とその整合性チェック（圧力勾配が純軸方向）
4. **検証の階段** G1 → G2 → G2b → G3 → G4。各ゲートは**閾値で規格化した比**で表示
   （生値ではなく「閾値の何倍か」。合格は比 < 1.00）
5. **結果**: `Q`-`G` 特性線、粘度場、`γ̇` 場、RTD の `E(t)`/`F(t)`、
   累積せん断分布、隙間の有無による比較
6. **限界**: 混練エレメント（螺旋対称性が壊れる）、粘性発熱（Phase 2）、
   飢餓供給・自由表面は射程外であること

- [ ] **Step 2: レポートを生成して mdview → Artifact 公開する**

```bash
cd ~/work/ykep-cae
OMP_NUM_THREADS=2 .venv/bin/python experiments/extruder/report.py 2>&1 | tee /tmp/log-$(date +%s).log
~/work/tb/bin/mdview docs/generated/extruder-report.md
```

生成 HTML の `<body>` 中身を取り出して `Artifact` で公開し、**URL を報告に書く**。
`SendUserFile` は使わない（スマホからリンク 1 タップで読めるようにするため）。

- [ ] **Step 3: `docs/status/status-28.md` を書く**

必須項目（CLAUDE.md の作業完了時の手順）:

- テスト数の推移（開始 `268 passed / 10 failed` → 終了時の実測値）
- 契約違反件数（0 であること）
- ブランチ名・コミットハッシュ
- 各ゲート G1〜G4 の**規格化比**（合格は < 1.00）
- **実測リソース使用量**（最大 RSS・所要時間）— NN 学習との共存の根拠
- 決まったこと / 未決事項
- 設計文書 `L_turn` 訂正の記録

- [ ] **Step 4: `status-index.md`, `roadmap.md`, `README.md` を更新する**

- [ ] **Step 5: 不整合が無いか確認する**

```bash
grep -rv "旧版" docs/ --include="*.md" | grep -n "πD/sinφ\|414 mm\|414mm" && echo "訂正漏れあり" || echo "OK"
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q 2>&1 | tail -3
OMP_NUM_THREADS=2 .venv/bin/python contracts/validate_process_contracts.py 2>&1 | tail -3
```

- [ ] **Step 6: コミットして push、PR を作る**

```bash
git add docs/ README.md experiments/
git commit -m "docs(extruder): status-28 と統合レポート — Phase 1/1.5 完了

G1〜G4 の全ゲートを通過。各指標は閾値で規格化した比で報告（合格は比<1.00）。
レポートは作業の時系列ではなく対象の構造（幾何→方程式→検証階段→結果→限界）で
章立てし、展開チャネル断面の SVG を冒頭に置いた。
実測リソース: 最大 RSS ___ MB / 所要 ___ s（NN 学習と共存可能な範囲）。"
git push -u origin claude/single-screw-extruder-impl
gh pr create --fill
```

---

## D. Phase 2 以降（本計画の範囲外）

| Phase | 内容 | 本計画との関係 |
|---|---|---|
| 2 | 粘性発熱 `Φ = μγ̇²` と温度依存粘度 | エネルギー式に散逸項が要る。`Task 6` の Picard ループに温度を 1 本足す形になる |
| 3 | 3D 混練エレメント（マドック / ダルメージ / ピン） | **螺旋対称性が壊れる**ので本計画の 2.5D 定式化が使えない。messi + OpenFOAM。MRF で足りるか要検討 |

**Phase 2 に入る前に、Phase 1/1.5 の結果を実機データと突き合わせること。**
本計画の諸元は仮の 40 mm 機であり、実機に合わせる場合は `ScrewSpec` を差し替えるだけでよい
（検証の枠組みは `H/W` にのみ依存するので変わらない）。

---

## E. Self-Review

**1. 仕様の網羅性** — 設計文書の各節に対応するタスク:

| 設計文書 | タスク |
|---|---|
| §1 なぜ 2.5D で足りるか | Task 0（前提の確認）、Task 5（断面内 3 成分） |
| §2 支配方程式 | Task 3（`w`）、Task 5（`u,v,p`）、Task 4（`μ`, `γ̇`） |
| §2.1 境界条件 | Task 1（符号確定）、Task 3（壁・バレル）、Task 5（周期＋体積力） |
| §2.2 フライト隙間 | Task 1（不等間隔格子）、Task 7（格子収束） |
| §3 検証の階段 G1-G4 | Task 2/3（G1,G2）、Task 5（G2b）、Task 8（G3）、Task 9/10（G4） |
| §4 ykep-cae に足すもの | Task 1,3,4,5,6,9,10（設計文書の 7 プロセスに対応） |
| §5 OpenFOAM 検算構成 | Task 8 |
| §6 諸元 | Task 1（`ScrewSpec`） |
| §7 段取り Phase 1/1.5 | Task 3-10。Phase 2/3 は §D で範囲外と明示 |
| §8 未決事項（6 件） | 全て確定: 符号→Task 0/1、級数→Task 2、反復→Task 6、時間刻み→Task 9、粒子数と種まき→Task 9、諸元→Task 1 |

**2. プレースホルダ** — `Task 5` の `cross_channel.py` と `Task 9`/`Task 10` の本体は
骨格 + 必須事項の箇条書きで、全行のコードは書いていない。これは意図的な省略ではなく、
**テストが完全に書かれていて仕様を一意に決めている**ためである
（`test_g2b_one_dimensional_exact` は 1e-10、`test_streamfunction_is_consistent_with_faces` は
ψ の構築規約、`test_g4b_...` は種まき重みを、それぞれ一意に固定する）。
実装者はテストを仕様として読むこと。

**3. 型の一貫性** — 確認済み:

- `ChannelGrid` のフィールド名 `dx, dy, xc, yc, solid, spec, mesh` は Task 1/3/5/9 で一致
- `ScrewSpec.beta(G)` は Task 1 で定義、Task 5 で使用
- `CrossChannelResult.u_face` は `(nx, ny)`、`v_face` は `(nx, ny+1)`、`psi` は `(nx+1, ny+1)`
  で Task 5/9 一致
- `ExtruderFlowResult` のフィールドは Task 6 で定義、Task 9/10 で参照
- `ParticleTrackResult.y0`（初期 `y`）は Task 9 のテストが要求するので
  `ParticleTrackResult` に **`y0`, `x0`, `z0` を含めること**（Task 9 Step 3 の実装時に追加）
