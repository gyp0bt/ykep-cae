# status-36: 汎用記法（.inp）で押出級の流れを書けるようにする（Phase 12）

[<- README](../../README.md) | [<- docs](../README.md) | [<- ステータス一覧](status-index.md) | [<- roadmap](../roadmap.md)

- 日付: 2026-09-06
- ブランチ: `claude/phase11-roadmap-todo-n3s0lj`
- 前: [status-35](status-35.md)（ソルバー体験の現在地整理 + Phase 11 残 TODO 消化）

---

## 1. 何を求められたか

> 押出は .inp でどうやって指定しますか？境界条件が結構ややこしいはず
> では、汎用記法で .inp を書くためには？2 次元仮定の単軸押出限定の inp も便利ですが、設計自由度もろしい

専用手続き（`*SCREW` / `*EXTRUDER`）を足す案を出したが、**設計自由度**を理由に汎用記法
（`*NODE` / `*ELEMENT` + `*NAVIER STOKES`）が選ばれた。決定事項:

- 回転自由度は **4, 5, 6**（Abaqus 流）
- 座標系は **`*ORIENTATION`**
- 回転体は **`*MPC` で参照節点に拘束して参照節点を回す**
- 他は推奨どおり

## 2. 汎用経路に足りなかったもの（調査結果）

| # | 欠落 | 影響 |
|---|---|---|
| 1 | **周期境界**（`grep` しても periodic / cyclic の実装ゼロ） | 1 ピッチで閉じられない。漏れ流れの駆動が書けない。z 1 セルを対称面にすると w が偽の解になる |
| 2 | **一様体積力**（`*DLOAD` は `GRAV` のみ） | 圧力跳び `Δp = G·L_turn` を `P = βx + p̃` に分解できない |
| 3 | **非ニュートン粘度**（`NavierStokesFVMInput.mu` が `float` 固定） | 樹脂が書けない。`assemble_momentum` は既に配列を受けていた |
| 4 | **回転壁**（面ごとの `Ω × r`） | 3D で円筒バレルをそのまま扱えない |
| 5 | **慣性項を落とす手段** | `Re ~ 10⁻³` のクリープ流れで SIMPLE の緩和に悩む |

## 3. やったこと

設計文書: [inp-generic-extrusion.md](../design/inp-generic-extrusion.md)（新規）

### 3.1 周期境界（`*BOUNDARY, TYPE=PERIODIC`）

```
*BOUNDARY, TYPE=PERIODIC
 master_surface, slave_surface[, tx, ty, tz]
```

- `MeshData.face_offset`（内部面ごとの並進ベクトル）を追加し、fvm 層に
  `neighbour_centers(mesh) = cell_centers[neighbour] + face_offset` を 1 本入れた。
  P–N ベクトルを使う箇所（補間重み・スキュー・over-relaxed 分解・最小二乗勾配・
  TVD の上流距離・Rhie–Chow・Darcy の速度再構成）を**すべてこの関数経由**にしたので、
  拡散も対流も圧力補正も分岐なしで周期面を内部面として扱う
- `InpMeshProcess` が対の面を面中心の並進で照合し、master 面を内部面に昇格させて slave 面を消す。
  幾何（面積・法線・体積・セル中心）は**併合前**に計算するのでセルの幾何は変わらない
- 照合失敗（面数違い・並進違い・法線が反平行でない・内部面を含む・面の重複）は
  ずれの最大値を添えてエラー
- 並進周期のみ（回転・螺旋は未対応）。3D 螺旋 1 ピッチは軸方向並進なので書ける

**2.5D の要点**: z 方向 1 セルの両端を周期にすると `∂/∂z = 0` が厳密になり **w が自由**になる。
対称面にすると z 方向にも壁ができ、w が厚さ `lz` に依存する偽の解になる。テストで両者を対比した。

### 3.2 一様体積力（`*DLOAD` の `BX` / `BY` / `BZ` / `BF`）

`NavierStokesFVMInput.body_force`（`(3,)` か `(n_cells, 3)`）を運動量ソースに足す。
`GRAV` は従来どおり Boussinesq 浮力なので別経路。構造格子経路は体積力を明示エラーにした。

### 3.3 非ニュートン粘度（`*VISCOSITY, TYPE=POWER LAW | CARREAU`）

- 粘度モデル Strategy を `extruder/viscosity.py` から `fvm/viscosity.py` に移して再輸出
  （押出専用モジュールには構造格子専用の γ̇ 評価だけ残した）
- 非構造の γ̇: `velocity_gradient_cells`（最小二乗の速度勾配テンソル）+
  `strain_rate_from_gradient`（γ̇ = sqrt(2 D:D)）
- 外部反復ごとに μ を更新する Picard（緩和 `RELAXATION` の `VISCOSITY=`、既定 0.5）。
  変粘度で拡散項 `∇·(μ∇u)` に入らない `∇·(μ∇uᵀ)` の余剰項 `Σ_j ∂_i u_j ∂_j μ` は陽的ソース
- 結果に `viscosity` / `strain_rate` を追加（`.inp` 出力の `MU` / `GAMMA`）

### 3.4 回転壁（`*ORIENTATION` + `*MPC` + 自由度 4-6）

```
*ORIENTATION, NAME=SPIN, SYSTEM=CYLINDRICAL
 0., 0., 0.,  0., 0., 1.        ** 軸上の 2 点
*MPC
 BEAM, BARREL, REF              ** 面 BARREL を参照節点 REF の剛体運動に拘束
*BOUNDARY, ORIENTATION=SPIN
 REF, 6, 6, 2.0                 ** 自由度 6 = 局所 3 軸まわりの角速度 [rad/s]
```

`VelocityPatchBC.rotating_wall(angular_velocity, center, velocity)` が面ごとに
`u = v + ω × (x_f − center)` を割り当てる。回転中心は参照節点の座標。
回転面が回転軸まわりの回転面になっていないと壁が「吹く」ので、法線速度が接線速度の
1e-6 倍を超えたら警告する。

### 3.5 Stokes モードと速度–圧力の連成

- `CONVECTION=NONE`（`STOKES` も可）: 運動量の対流項を落とす。
  エネルギー・追加スカラーは風上のまま輸送する
- `PRESSURE_VELOCITY=COUPLED`: 速度 nd 成分と圧力を 1 つの線形系にして直接解く
  （`fvm/momentum.assemble_coupled`）。圧力勾配は最小二乗勾配の**線形作用素**
  （`geometry.lsq_gradient_operator` を新設）、連続式は Rhie–Chow 流束を u と p の両方について陰的に。
  緩和係数を使わない

### 3.6 押出への橋渡し

`ExtruderChannelInpProcess`（`extruder/inp_export.py`）: `ScrewSpec` + `G` + 粘度モデルから
**汎用記法の .inp テキスト**を生成する PreProcess。出力はただの .inp なので、生成後に手で編集できる。
例題 [extruder-channel-1.inp](../../examples/inp/extruder-channel-1.inp)（2016 セル）。

## 4. 検証（数値は実測。ログは `examples/inp/results/extruder-channel-1.log`）

40 mm 機の計量部（D=40 mm, リード 40 mm, H=4 mm, e=4 mm, δ=0.2 mm, N=1 s⁻¹, μ=1000 Pa·s, G=1e5 Pa/m）:

| 量 | 汎用 vs 参照 | 備考 |
|---|---|---|
| `Q`（閉チャネル、G=0 と G>0） | 解析解と 2.3e-3（40×20 相当）→ 7.3e-4（60×24） | 形状係数 `F_d` / `F_p`（ゲート G1 / G2） |
| `Q`（隙間あり） | 専用ソルバーと **1.1e-15** | 下流方向は同じ可変係数 Poisson になるので機械精度 |
| `Q_leak`（漏れ） | 5.6e-2（30×12）→ 3.9e-2（40×20）→ 2.2e-2（60×24）→ 7.2e-3（120×48） | 断面内が MAC 千鳥格子（専用）vs Rhie–Chow 同位置格子（汎用）。細分化で縮む |
| `Q_axial = Q + L_turn·Q_leak` | 3.6e-3 → 2.5e-3 → 1.4e-3 → 4.7e-4 | 実際の押出量 |

その他の実測:

- 周期流路の Poiseuille（体積力駆動）: 解析解と 3.9e-3（4×16）、圧力は周期方向に一様（`ptp < 1e-10`）
- Taylor–Couette（円環 16×96、外周回転 ω=2）: 解析解 `u_θ = Ar + B/r` と 1.3e-3、
  半径方向速度は接線速度の 3.5e-14。`.inp` 経由（12×64）でも 2.7e-3
- べき乗則流路（K=0.05, n=0.5）: 解析解と 2.6e-3。`gamma_min` を 1e-2 / 1e-3 / 1e-4 と変えても同じ
- COUPLED: Stokes キャビティ（16×16）が **2 反復**、SIMPLE は 273 反復で同じ解（1e-6 一致）。
  Re=100 キャビティ（20×20）は 10 反復 vs 197 反復（1e-4 一致）
- 例題 extruder-channel-1: 2016 セル・COUPLED・**2 反復 0.18 s**、mass 残差 7.5e-12

### 4.1 全件テスト

`python -m pytest tests/ -q -m "not slow"`（コミット前、ブランチ `claude/phase11-roadmap-todo-n3s0lj`）:
**849 収集 / 815 passed / 15 skipped / 1 xfailed / 18 deselected、11 分 22 秒**。
実行時の 1 failed は `coupling="coupled"` を未対応扱いにしていた旧テスト
（`TestNavierStokesFVMAPI::test_validation`）で、期待値を `"block"` に直して通過。
`python contracts/validate_process_contracts.py` は契約違反 0 件（36 プロセス）。

新規テスト: `tests/test_extruder_inp_export.py`（16 件）、
`tests/test_inp_mesh.py::TestInpMeshPeriodic`（7 件）、
`tests/test_inp_parser.py::TestGenericExtrusionKeywords`（27 件）、
`tests/test_inp_mapping.py::TestGenericExtrusionMapping`（11 件）、
`tests/test_navier_stokes_fvm.py` の周期・COUPLED・回転壁・非ニュートン（13 件）。

## 5. 途中で分かったこと

- **SIMPLE は Stokes で遅い**。周期流路（体積力駆動）で α=(0.7, 0.3) の SIMPLE は 369 反復、
  SIMPLEC α=(0.9, 1.0) で 104 反復。α_u = 1.0 にすると 2 反復で解けるがキャビティでは発散する。
  非ニュートンの Picard を重ねると 800 反復でも収束しなかった。COUPLED を入れて解決した
  （Stokes は 2 反復、べき乗則でも 44 反復）
- `repr()` を numpy スカラーに使うと `np.float64(0.0)` になり .inp が壊れる（numpy 2）。
  `repr(float(v))` にした
- 予約面名（`XM..ZP`）は明示した `*SURFACE` と**同じ面を含みうる**。両方に境界条件を書くと
  後勝ちになる。除外する変更を試したが既存例題（`*SURFACE` で一面全体を覆ってから `ZP` も使う書き方）を
  壊すので戻し、文書に注意として書いた

## 6. 残件

- 非構造メッシュの**粒子追跡 / RTD**（構造格子の ψ 双一次補間のみ）。汎用経路で RTD を出すには
  面流束ベースの追跡（Pollock 型）が要る
- 周期は並進のみ（回転周期・螺旋周期）
- COUPLED は直接法固定・`OUTFLOW` 非対応（大規模では SIMPLE 系が省メモリ）
- 粘性発熱 `Φ = μγ̇²` と温度依存粘度（専用ソルバー側も Phase 2）
- status-35 からの持ち越し: 非直交補正の limited 版、バッフルの片側条件、
  構造格子版 Rhie–Chow の緩和非依存化、空気実物性の再調査
