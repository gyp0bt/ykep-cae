# xkep-cae-fluid

FDM（差分法）・FVM（有限体積法）による流体ソルバー基盤。
[xkep-cae](https://github.com/gyp0bt/xkep-cae) と共通の Process Architecture を基盤とし、
流体解析特有の離散化・乱流モデル・圧力-速度連成をモジュール化する。

## xkep-cae との関係

| 項目 | xkep-cae | xkep-cae-fluid |
|------|----------|----------------|
| 手法 | FEM（有限要素法） | FDM / FVM |
| 対象 | 構造解析（撚線曲げ揺動等） | 流体解析（非圧縮性NS等） |
| 共通基盤 | Process Architecture | Process Architecture |
| Strategy | Penalty, Friction, ContactForce | Convection, Turbulence, PV-Coupling |

## 現在の状態

**884 テスト**（本環境の 4 並列で slow 除外の全件実行: **838 passed / 15 skipped / 1 xfailed、2 分 28 秒**。`slow` のみは 29 passed / 1 failed・4 分 25 秒で、落ちる 1 件は空気実物性の SIMPLE 連成という[既存の未解決問題](docs/status/status-37.md)（本ブランチの変更前から同じく失敗する）） -- 2026-09-06 [非構造メッシュの粒子追跡と滞留時間分布](docs/status/status-37.md)（Phase 12 完了）: 汎用記法で書いた .inp から**混練性と RTD** を出せるようにした。セル中心速度を補間すると離散的な発散ゼロが壊れて粒子が渦心に落ち込むので、[**面流束から**セル内の速度場を再構成する](docs/design/particle-tracking-fvm.md) — セルの全ての面について流束を厳密に再現するアフィン場 `u = a_c + B_c(x − x_c)` を最小ノルムで閉じると、直交六面体では **Pollock（1988）そのもの**、四面体では **RT0** になり、`∇·u = tr(B) = Σq_f/V` が離散連続式のぶんだけ恒等的にゼロになる（セル形状は問わない）。面平面までの到達時刻で刻んで false position で面に落とし隣接セルへ渡す。周期面は並進を持ち回るので `x + shift_total` が「巻き戻さない座標」になり、押出の ζ がそのまま出る。`ParticleTrackFVMProcess` + `ResidenceTimeProcess`、NS 結果に γ̇ と混合指数 λ を常時追加（`*OUTPUT` の `GAMMA` / `LAMBDA`）。厳密関係 **⟨t⟩ = length·V/Σw** が周期 Poiseuille で 1e-12 一致。例題 [extruder-channel-1](examples/inp/extruder-channel-1.inp)（2016 セル）で構造格子トラッカー（ψ 双一次補間、ゲート G4a/G4b/G5 通過済み）と ⟨t⟩ 6.5e-3・t_p10/p50/p90 が 1.7e-3/1.0e-3/2.0e-3・λ 2.4e-5 で一致（[ログ](examples/inp/results/extruder-channel-1-rtd.log)）。あわせて**後方互換を全撤去**（`RegistryProxy`・`ProcessMeta.deprecated` 機構・粘度モデルと重み付き統計の再輸出シム）し、**全件テストを 14 分 26 秒 → 2 分 28 秒**にした（流れ場を各テストで解き直していたのを共有化、格子収束など 6 件を `slow` へ、`pytest-xdist`）。前: 2026-09-06 [汎用記法（.inp）で押出級の流れを書く](docs/status/status-36.md)（Phase 12）: 押出を専用キーワードではなく**汎用記法**（`*NODE`/`*ELEMENT` + `*NAVIER STOKES`）で書けるようにした。**周期境界** [`*BOUNDARY, TYPE=PERIODIC`](docs/design/inp-generic-extrusion.md)（`MeshData.face_offset` を 1 本足し、fvm 層の `neighbour_centers` を通すだけで拡散・対流・圧力補正・Rhie–Chow が分岐なしで通る。`InpMeshProcess` が対の面を並進で照合して内部面に併合）、**一様体積力** `*DLOAD, BX/BY/BZ/BF`（圧力跳び `Δp = G·L_turn` を `P = βx + p̃` に分解）、**非ニュートン粘度** `*VISCOSITY, TYPE=POWER LAW｜CARREAU`（粘度モデルを [`fvm/viscosity.py`](docs/design/fvm-layer.md) へ、非構造の γ̇ は最小二乗の速度勾配から、Picard 緩和は `RELAXATION` の `VISCOSITY=`）、**回転壁** `*ORIENTATION` + `*MPC` + 参照節点の自由度 4-6（Taylor–Couette が解析解と 1.3e-3）、**Stokes** `CONVECTION=NONE` と**速度–圧力の連成** `PRESSURE_VELOCITY=COUPLED`（`assemble_coupled` + `lsq_gradient_operator`。Stokes キャビティが 273 → **2 反復**、Re=100 が 197 → 10 反復で同じ解）。`ExtruderChannelInpProcess` が諸元から汎用 .inp を生成し、例題 [extruder-channel-1](examples/inp/extruder-channel-1.inp)（2016 セル、2 反復 0.18 s）が専用 2.5D ソルバーと押出量 `Q` で **1.1e-15**、形状係数（ゲート G1/G2）と 2.3e-3、`Q_axial` 1.4e-3 で一致。前: 2026-09-06 [ソルバー体験の現在地整理](docs/status/status-35.md)（入口 / 2 経路の機能差 / 境界条件の書き方 / 出力 / 収束の実測 / ギャップ順位）+ Phase 11 残 TODO の消化: 非構造 NS [`NavierStokesFVMProcess`](docs/design/navier-stokes-fvm.md) に**圧力補正の非直交補正**（`NONORTHOGONAL_CORRECTORS`、既定 2。せん断 31°/45° の Stokes 的キャビティで α=(0.8,0.5) が発散 → 28 反復、収束解は不変）、**適応緩和** `ADAPTIVE`（規則を [`fvm/relaxation.py`](docs/design/fvm-layer.md) に切り出して構造格子版と共有、最小残差の 5 倍超の停滞検出と α_p ≤ 1 − α_u を追加。cavity-nc-2 75 → 62 反復）、**Rhie–Chow を緩和前の a_P で**（Majumdar。収束解の α_u 依存 2% → 3e-8）、反復ログと発散検出。[`InpMeshProcess`](docs/design/unstructured-inp-mesh.md) が **2 次要素**（C3D10/C3D15/C3D20/CPS6/CPS8、頂点のみ）と**角錐 C3D5** を受理、**内部面の `*SURFACE` をバッフル**（厚さゼロの壁、両側の境界面に分割。境界条件の target なら ykep が自動で非構造経路へ）に。例題 [channel-baffle-1](examples/inp/channel-baffle-1.inp)（26 反復、隙間で流速 1.8 倍）。`HEAT TRANSFER=NONE` で構造格子版もエネルギー方程式を解かない（`solve_energy`）。`*HEAT TRANSFER` で `TYPE=WALL` = 断熱。CI の `test` ジョブで AMG / Numba テストを skip（master が赤だった件）。前: 2026-09-06 Phase 11 の残件: 非構造 NS [`NavierStokesFVMProcess`](docs/design/navier-stokes-fvm.md) に TVD 遅延補正（蓋駆動キャビティ Re=100 で u_min −0.211、Ghia −0.2109）/ BDF2 / PISO（Issa の H(u) 再評価、分離誤差 6% → 1.5% → 0.8%）/ 対流流出 OUTFLOW / 内部吐出・吸入セル `InternalCellBC`（`.inp` は要素集合 target の `*BOUNDARY`）/ 追加スカラー `ScalarSpec`、軸平行な対称面を陰的にして cavity-nc-2 が 165 → 75 反復、[`InpMeshProcess`](docs/design/unstructured-inp-mesh.md) が四面体 / 楔 / 三角形と種別混在に対応（fvm 層の勾配にスキュー補正、四面体の線形場 1e-7、mirador も C3D4 / C3D6）、[Darcy](docs/design/darcy-flow-fvm.md) の Forchheimer（Picard）と比貯留の非定常、`CorrectedDiffusionScheme` を fvm 層の包みに、`core/data` の未使用スキーマ削除。前: 2026-09-05 非構造格子対応（本体側）: [`HeatTransferFVMProcess`](docs/design/heat-transfer-fvm.md)（構造格子 FDM と 1e-8 一致）、[`NavierStokesFVMProcess`](docs/design/navier-stokes-fvm.md)（面リストの SIMPLE/SIMPLEC + Rhie–Chow、Boussinesq、Brinkman 抵抗、固体マスク、エネルギー。Poiseuille / Brinkman 流路 / 蓋駆動キャビティ Re=100 / 差分加熱キャビティ Ra=10³ で検証）、fvm 層の非直交補正（over-relaxed 分解 + 境界接線補正 + 遅延補正反復、最小二乗勾配）、`ykep --mesh=auto|structured|unstructured`（箱格子でなければ `*NAVIER STOKES` / `*HEAT TRANSFER` / `*DARCY` を `InpMeshProcess` + FVM 版で解く）、例題 [plate-ht-2](examples/inp/plate-ht-2.inp)（せん断平板）/ [cavity-nc-2](examples/inp/cavity-nc-2.inp)（平行四辺形キャビティ、当時 165 反復で収束）、`nsb/` をコミット 1647839 時点のスナップショットとして切り離し。前: 2026-09-05 ソルバー層の分離と非構造格子: [棚卸しと計画](docs/plans/2026-09-05-solver-layering.md)、[面ベース FVM 共通低レイヤー `xkep_cae_fluid.fvm`](docs/design/fvm-layer.md)（パッチ境界条件・面演算・係数組み立て・線形ソルバー Strategy）、`MeshData` に境界面・パッチ・セル種別（構造格子 6 パッチ、polyMesh の節点順序付き接続）、[`InpMeshProcess`](docs/design/unstructured-inp-mesh.md)（`.inp` の任意六面体 / 四辺形 → 面ベース非構造メッシュ、`*SURFACE` → パッチ）、`ScalarTransportFVMProcess`（パイロット、構造格子 FDM と 1e-8 一致）、[`DarcyFlowProcess` + `*DARCY`](docs/design/darcy-flow-fvm.md)（非構造 NPZ / VTK / HTML 出力、例題 darcy-1: せん断メッシュ + 低透過率ブロック、流入 = 流出）、mirador の `mesh=` 入力。`NaturalConvectionFDM` の過渡 dt 差し替えで `internal_face_bcs` が落ちる回帰を修正。前: 2026-09-05 3D レンダリング: [messi mirador 連携](docs/design/mirador-export.md)（`MiradorExportProcess`、構造格子 → C3D8 + 断面スラブ elset + セル場、速度矢印、任意平面の断面 view cut（`--cut=z=0.5`、切り口をセル値で着色）、`*OUTPUT, FIELD, FORMAT=HTML` / `ykep -j=<job> view --slice=x=0.05`。残差マップ `res_u/res_v/res_w/res_T/res_mass` を場として出力、`FORMAT=` 未指定なら messi のある環境で HTML 自動出力。messi 側は v0.10.0 で要素場カラーマップ（Abaqus レインボー既定）・矢印・`.vtk` リーダを追加 / [status-34](docs/status/status-34.md)）。前: ykep .inp 入力フォーマット（[Abaqus 風キーワード構文](docs/design/inp-format.md)、`*PARAMETER`/`*CONTROLS`/`*GRID`、中立表現 `CaseDefinition`、`ykep -j=<job>.inp int` コマンド、`*NAVIER STOKES` → NaturalConvectionFDM / `*HEAT TRANSFER` → HeatTransferFDM、NPZ/YAML/VTK 出力。例題 Ra=1000 キャビティは 226 反復で収束し Nu=1.169 / [status-33](docs/status/status-33.md)）。前: nsb を xkep_cae_fluid から切り離し（[`data`/`assembly` のコピー方式 + 同期スクリプト](nsb/README.md)、numpy/scipy/pypardiso だけで単体持ち出し可）+ 高速化見積り実測（LU 分解が 70〜81%、for ループ削減は 0%、JAX は autodiff 目的のみ）+ 疎 LU を PARDISO 前提に + 前処理 LU の遅延更新（144×96 で 40 s → 17 s、MKL スレッド分割が鍵 / [status-32](docs/status/status-32.md)）。前: Brinkman 流路の[座標マスク境界条件 + 質量流入 + 領域内マニホールド + 随伴設計感度](docs/design/brinkman-flow-fvm.md)（4 辺任意配置、流量固定で inlet 探索、紙面垂直方向のヘッダ、位置・径の勾配を陰関数定理で、冷却流路設計の前段 / [status-31](docs/status/status-31.md)）。前: 収束破綻の再現と機構切り分け（[status-30](docs/status/status-30.md)、[nsb/](nsb/README.md)） | 契約違反 **0件**（38プロセス） | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

前: [単軸押出解析 Phase 1/1.5 + G5 文献照合](docs/design/single-screw-extruder.md)（展開チャネル 2.5D、ゲート G1〜G5 全通過、OpenFOAM 検算・Pinto–Tadmor RTD 照合済み / [status-29](docs/status/status-29.md) / [図解レポート](docs/reports/extruder/README.md)）

## パッケージ構成

```
xkep_cae_fluid/
+-- core/              # プロセスアーキテクチャ基盤（xkep-cae共通設計）
|   +-- base.py        # AbstractProcess + ProcessMeta + ProcessMetaclass
|   +-- registry.py    # ProcessRegistry
|   +-- slots.py       # StrategySlot
|   +-- categories.py  # PreProcess / SolverProcess / PostProcess / VerifyProcess / BatchProcess
|   +-- data.py        # MeshData（面リスト・境界パッチ・セル種別）
|   +-- mesh.py        # StructuredMeshProcess（不等間隔直交格子生成）
|   +-- mesh_reader.py # PolyMeshReaderProcess（OpenFOAM polyMesh 読込）
|   +-- runner.py      # ProcessRunner
|   +-- diagnostics.py # 実行診断
|   +-- benchmark.py   # BenchmarkRunnerProcess
|   +-- tree.py        # ProcessTree（依存グラフ）
|   +-- testing.py     # binds_to（テスト紐付け）
|   +-- strategies/    # Strategy Protocol 定義 + 具象スキーム（拡散/対流/TVD/非直交補正）
|   +-- docs/          # コアモジュール設計文書
+-- fvm/               # 面ベース FVM 共通低レイヤー（方程式ファミリー非依存、非直交補正、Phase 11）
|   +-- boundary.py    # PatchBC（Dirichlet/Neumann/Robin/ゼロ勾配）+ resolve_boundary（パッチ名 → 境界面配列）
|   +-- geometry.py    # 面補間重み・調和平均・面質量流束・Green-Gauss 勾配
|   +-- assembly.py    # 拡散・1 次風上対流・時間項・ソース項の係数行列（体積積分形）
|   +-- linear.py      # DirectSolver / BiCGSTABSolver / AMGSolver（LinearSolverStrategy 実装）
|   +-- momentum.py    # 運動量・圧力連成カーネル（速度境界、Rhie–Chow、圧力補正 + 非直交補正）
|   +-- relaxation.py  # 緩和係数の適応的調整（構造格子版 / 非構造版で共有）
|   +-- viscosity.py   # 粘度モデル Strategy（Newtonian / べき乗則 / Carreau）+ 非構造のせん断速度 γ̇（Phase 12）
+-- darcy/             # Darcy 流れ（*DARCY 方程式ファミリー、面ベース FVM、非構造六面体メッシュ可）
+-- incompressible/    # 非圧縮 NS（面ベース FVM、SIMPLE/SIMPLEC/PISO/COUPLED + Rhie–Chow、Stokes、体積力、非ニュートン粘度、回転壁、周期境界、Boussinesq、Brinkman、非構造メッシュ可）
|   +-- data.py        # DarcyFlowInput / Result / DarcyPatchBC（PRESSURE / VELOCITY / WALL）
|   +-- solver.py      # DarcyFlowProcess（圧力ポアソン + 面流束からのセル速度再構成）
+-- natural_convection/ # 3次元自然対流解析 (FDM + SIMPLE法)
|   +-- data.py        # NaturalConvectionInput / Result / FluidBoundarySpec
|   +-- assembly.py    # 疎行列アセンブリ（運動量・圧力補正・エネルギー）
|   +-- solver.py      # NaturalConvectionFDMProcess (SIMPLE/SIMPLEC/PISO + TVD + BDF2)
+-- scalar_transport/  # 汎用スカラー輸送 (Phase 6.1a 水槽CAE基盤)
|   +-- data.py        # ScalarFieldSpec / ScalarBoundarySpec / Input / Result
|   +-- assembly.py    # 疎行列アセンブリ（対流-拡散-ソース、Dirichlet/Neumann/Robin BC）
|   +-- solver.py      # ScalarTransportProcess (陰的Euler + BiCGSTAB+ILU)
|   +-- fvm.py         # ScalarTransportFVMProcess（MeshData 上の面ベース版。構造格子で FDM と一致）
+-- brinkman_flow/     # 2D Brinkman 補正 Navier-Stokes (FVM, Newton–Krylov) — 収束破綻の再現実験
|   +-- data.py        # BrinkmanFlowInput / Result / SolverSettings / ThicknessSpec / BoundaryPatch（座標マスク・質量流入）
|   +-- geometry.py    # UTurnThicknessProcess（flat / uturn 厚さ場）
|   +-- assembly.py    # 同位置 FVM 残差（1次/2次風上+Venkatakrishnan）+ 1次風上ヤコビアン + Rhie-Chow + 4 辺の座標マスク境界
|   +-- solver.py      # BrinkmanFlowFVMProcess（Newton + GMRES/LU(J1) + 擬似時間 + 陰的緩和）
+-- aquarium/          # 水槽設計 CAE ドメイン（Phase 6.2 / 6.3）
|   +-- geometry.py    # AquariumGeometryProcess（90×30×45 cm + 底床/ガラス/水マスク + z-refinement）
|   +-- heater.py      # HeaterProcess（定熱流束 + 定温ヒステリシス）
|   +-- filter.py      # AquariumFilterProcess + InternalFaceBC（外部フィルター循環, Q[L/h]）
+-- extruder/          # 単軸押出 2.5D 断面解析（Phase 7、status-28/29）
|   +-- geometry.py    # ScrewGeometryProcess（展開チャネル + 隙間の等比格子）
|   +-- shape_factors.py  # 形状係数 Fd/Fp の級数解（ゲート G1/G2 の真値）
|   +-- down_channel.py   # DownChannelFlowProcess（w: 可変係数 Poisson）
|   +-- cross_channel.py  # CrossChannelStokesProcess（u,v,p: MAC Stokes 鞍点系）
|   +-- viscosity.py   # 構造格子の Green-Gauss γ̇（粘度モデル本体は fvm/viscosity.py から再輸出）
|   +-- solver.py      # ExtruderFlowProcess（Picard 結合、Q_axial = Q + L_turn·Q_leak）
|   +-- inp_export.py  # ExtruderChannelInpProcess（諸元 → 汎用記法の .inp、Phase 12）
|   +-- tracker.py     # ParticleTrackerProcess（ψ 双一次補間、RK4、ζ 座標）
|   +-- rtd.py         # RTDProcess（流束重み付き RTD、パーセンタイル、累積せん断）
+-- post/              # 後処理（非構造メッシュ共通）
|   +-- mirador.py     # MiradorExportProcess（messi mirador 3D レンダリング、status-34）
|   +-- tracking.py    # ParticleTrackFVMProcess（面流束から再構成した Pollock 型の粒子追跡、status-37）
|   +-- rtd.py         # ResidenceTimeProcess（滞留時間分布・経路積分スカラー）
|   +-- statistics.py  # 流束重み付きの分位点・経験分布
+-- inp/               # ykep .inp 入力フォーマット（Abaqus 風キーワード構文、status-33）
|   +-- parameters.py  # *PARAMETER の安全な式評価 + <expr> 置換
|   +-- parser.py      # InpKeywordParseProcess（*INCLUDE / コメント / 継続行 / KeywordBlock 列）
|   +-- case.py        # CaseDefinition（ソルバー非依存の中立表現）
|   +-- builder.py     # InpCaseBuildProcess（意味付け、*GRID 拡張）
|   +-- grid.py        # StructuredGridRecoveryProcess（*NODE/*ELEMENT → 直交構造格子、*SURFACE → 領域面）
|   +-- mesh.py        # InpMeshProcess（*NODE/*ELEMENT → 面ベース非構造 MeshData、六面体 / 楔 / 四面体 / 角錐、2 次要素は頂点のみ、*SURFACE → 境界パッチ、内部面はバッフル、*BOUNDARY TYPE=PERIODIC で周期面を内部面に併合）
|   +-- mapping.py     # InpToNaturalConvectionProcess / InpToHeatTransferProcess / InpToDarcyProcess（*CONTROLS 含む）
|   +-- output.py      # InpOutputWriterProcess（NPZ / YAML サマリ / VTK RECTILINEAR or UNSTRUCTURED）
|   +-- runner.py      # InpCaseRunnerProcess（方程式ファミリーで振り分け）
|   +-- cli.py         # ykep コマンド（ykep -j=<job>.inp int）
+-- heat_transfer/     # 3次元非定常伝熱解析 (FDM) + fvm.py（面ベース FVM 版、非構造メッシュ可）
|   +-- data.py        # HeatTransferInput / HeatTransferResult / BoundarySpec (Robin対応)
|   +-- solver.py      # HeatTransferFDMProcess (ヤコビ/GS/疎行列/AMG/Numba)
|   +-- solver_vectorized.py  # NumPy ベクトル化ヤコビ法
|   +-- solver_sparse.py      # SciPy 疎行列ソルバー (直接解法/BiCGSTAB/AMG)
|   +-- solver_numba.py       # Numba JIT 高速化ガウスザイデル法
|   +-- multilayer.py  # MultilayerBuilderProcess (多層シート物性値ビルダー)
|   +-- visualize.py   # TemperatureMapProcess (温度マップ/CJK/ミラーリング)
+-- examples/          # 実行例
|   +-- multilayer_sheet_temperature.py  # 4層多層シート温度マップ
|   +-- multilayer_robin_analysis.py     # MultilayerBuilder+FDM+Robin BC 連携例
|   +-- benchmark_solver_methods.py      # ソルバー手法別ベンチマーク
|   +-- aquarium_heater_natural_convection.py  # Geometry+Heater+NC 3 段（Phase 6.2b）
|   +-- aquarium_filter_circulation.py         # Geometry+Heater+Filter+NC 4 段（Phase 6.3b）
|   +-- inp/           # .inp 例題（cavity-nc-1/2: Ra=1000 キャビティ（箱格子 / 平行四辺形）、plate-ht-1/2: 平板伝熱、darcy-1: せん断メッシュの Darcy 流れ、channel-baffle-1: 薄板バッフル流路、extruder-channel-1: 単軸押出の展開チャネル 2.5D（汎用記法））+ results/
+-- experiments/brinkman_uturn/  # Brinkman U ターン収束性スイープ（sweep.py / diagnose_u2.py / diagnose_local_dtau.py / results / logs）
+-- nsb/               # 手元構成ミラー（core / solver / utils / geo / adjoint + data / assembly のコピー。xkep_cae_fluid 非依存で単体持ち出し可）+ theory.md（数理ノート）+ ルート main.py
+-- experiments/nsb/   # nsb パラメータスタディの results / logs
+-- tests/             # テスト
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ドキュメント総覧](docs/README.md) | ドキュメント一覧 + xkep-cae との関係 |
| [Process Architecture](docs/process-architecture.md) | 共通アーキテクチャ設計仕様 |
| [データスキーマ](docs/data-schemas.md) | MeshData の仕様（ファミリー別 Input / Result は各設計文書） |
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [水槽設計ロードマップ](docs/roadmap-aquarium.md) | Phase 6 持続的水槽設計 CAE 詳細計画 |
| [設計文書一覧](docs/design/README.md) | 設計仕様書リンク集（コロケーション方式） |
| [.inp 入力フォーマット](docs/design/inp-format.md) | Abaqus 風キーワード構文と `ykep -j=<job>.inp int` コマンド |
| [汎用記法で押出級を書く](docs/design/inp-generic-extrusion.md) | 周期境界・体積力・非ニュートン粘度・回転壁・Stokes / COUPLED（Phase 12） |
| [3D レンダリング（messi mirador）](docs/design/mirador-export.md) | 解析結果を messi の three.js ビューアで表示（断面スラブ・任意平面の view cut・速度矢印、`FORMAT=HTML` / `ykep view`） |
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |

## インストール

```bash
pip install -e ".[dev]"
```

## .inp で実行（ykep コマンド）

```bash
ykep -j=examples/inp/cavity-nc-1.inp int            # Abaqus 風: -j=<job>[.inp] と int（対話ログ）
ykep -j=examples/inp/plate-ht-1 int -o=out          # 出力先指定（<job>.npz / .yaml / .vtk / .log）
ykep -j=case.inp --check                            # 解析せず読込・格子復元・マッピングのみ検証
ykep -j=examples/inp/cavity-nc-1 view -o=out --slice=x=0.05   # 解析せず NPZ → <job>.html（messi mirador 3D ビューア）
ykep -j=examples/inp/cavity-nc-1 view -o=out --cut=y=0.05     # 任意平面の断面（view cut）を y=0.05 で有効にして開く
ykep -j=examples/inp/darcy-1 int -o=out                       # *DARCY: 箱格子でない六面体メッシュも InpMeshProcess で解く
ykep -j=examples/inp/channel-baffle-1 int -o=out              # 内部面の *SURFACE を WALL にすると厚さゼロのバッフル（非構造経路）
ykep -j=examples/inp/extruder-channel-1 int -o=out            # 単軸押出の展開チャネル 2.5D を汎用記法で（周期境界 + 体積力 + Stokes + COUPLED）
python examples/extruder_generic_rtd.py                       # その .inp から滞留時間分布・累積せん断ひずみ・混合指数（面流束ベースの粒子追跡）
```

ソルバー体験の現在地（何を書いて何を打つか、構造格子 / 非構造の機能差、境界条件の書き方、出力、収束の実測、
残るギャップ）は [status-35](docs/status/status-35.md) の第 1 節にまとめてある。

## 3D レンダリング（messi mirador 連携）

[messi](https://github.com/gyp0bt/messi)（v0.10.0 以降）を入れると、構造格子の結果を three.js の
自己完結 HTML に書き出してブラウザで回せる（`*OUTPUT, FIELD, FORMAT=VTK+HTML` か `ykep ... view`、
Python からは `MiradorExportProcess`）。外皮 + 断面スラブ（elset 切替）+ 速度矢印、場ごとのカラーマップ（Abaqus レインボー既定）、残差マップ `res_*`、
probe で値表示。任意平面の断面（view cut、`c` キー / `ykep view --cut=z=0.5`）は切り口をセル値で着色し、
法線・位置・反転をパネルで動かせる。操作パネルは `h` キーで畳める（`ykep view --collapse-panel` で畳んだ状態から）。
`FORMAT=` を書かなければ messi のある環境では HTML が自動で出る。非構造格子（`*DARCY`）の結果は `MeshData` の六面体を
そのまま描く（`MiradorExportInput.mesh`、`ykep view --cut` 可）。詳細は [設計文書](docs/design/mirador-export.md)。

```bash
pip install -e ../messi     # 任意依存（未導入なら FORMAT=HTML は警告してスキップ）
```

キーワード一覧は [設計文書](docs/design/inp-format.md) を参照。

## テスト実行

```bash
pytest tests/ -q -m "not slow" -n 4      # 通常（本環境で 2 分 28 秒）
pytest tests/ -q -m "slow" -n 4          # 格子収束・長時間の検証（4 分 25 秒）
```

重い押出テストは同じ流れ場を `functools.cache` で使い回すので、`-n` で並列にするときは
**ファイル単位で 1 ワーカーに固める**（`pyproject.toml` の `addopts` に `--dist loadfile` を入れてある）。

## Lint / Format

```bash
ruff check xkep_cae_fluid/ tests/
ruff format xkep_cae_fluid/ tests/
```

## ライセンス

[MIT License](LICENSE)

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は [docs/status/](docs/status/status-index.md) を参照。
