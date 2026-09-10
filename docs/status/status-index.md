# ステータスインデックス

[<- README](../../README.md)

| # | 日付 | テスト数 | 概要 |
|---|------|---------|------|
| 1 | 2026-03-31 | 16 | 初期移植: xkep-cae から Process Architecture 移植 |
| 2 | 2026-03-31 | 25 | 3次元非定常伝熱解析 (FDM) HeatTransferFDMProcess 実装 |
| 3 | 2026-03-31 | 31 | ソルバー高速化 + 可視化PostProcess + 多層シート温度マップ |
| 4 | 2026-03-31 | 39 | Robin BC + 多層ビルダー + CJK対応 + ミラーリング表示 |
| 5 | 2026-03-31 | 49 | status-4 TODO消化: 非定常Robin BC・冷却フィン・連携例・scipy復旧 |
| 6 | 2026-03-31 | 59 | status-5 TODO消化: 疎行列ソルバー・フィンアレイ2D/3D・CI整備・Phase 2設計 |
| 7 | 2026-03-31 | 88 | status-6 TODO消化: StructuredMeshProcess + PyAMG + Numba JIT |
| 8 | 2026-03-31 | 124 | status-7 TODO全消化: 離散化スキーム + MeshData対応 + polyMesh読込 |
| 9 | 2026-04-01 | 138 | 3D自然対流ソルバー (SIMPLE法 + Boussinesq) + 固体-流体練成 |
| 10 | 2026-04-01 | 176 | status-9 TODO全消化: TVD/Rhie-Chow/非直交補正/AMGキャッシュ/バイナリpolyMesh |
| 11 | 2026-04-01 | 180 | 自然対流調査: q_vol追加 + パラメトリックスタディ(24ケース) + 設計指針 |
| 12 | 2026-04-01 | 180 | ソ��バー安定性改善: adaptive dt + RC energy + 収束判定修正 |
| 13 | 2026-04-02 | 197 | SIMPLEC連成 + BDF2���間積分 + Poiseuille検証 + 保守的緩和テスト |
| 14 | 2026-04-02 | 214 | PISO連成 + TVD対流スキーム統合 + 対流流出BC |
| 15 | 2026-04-02 | 214 | 空気実物性 収束評価 + PISO速度緩和修正 |
| 16 | 2026-04-02 | 221 | AMG圧力ソルバー + 面ベース質量残差修正 + 適応的緩和 |
| 17 | 2026-04-03 | 224 | CG+AMG圧力ソルバー + Ra=1e4ベンチマーク修正 + 長時間安定性検証 |
| 18 | 2026-04-07 | 224 | 1D過渡ジュール電熱 Gauss-Seidel 検算スクリプト |
| 19 | 2026-04-08 | 224 | 1D FDMソルバー 輻射実装 + 断熱BC修正 + LineAreaヘルパー |
| 20 | 2026-04-20 | 224 | 持続的水槽設計CAE Phase 6 ロードマップ策定（90×30×45 水草水槽） |
| 21 | 2026-04-20 | 232 | 汎用スカラー輸送 `ScalarTransportProcess` 新設（Phase 6.1a） |
| 22 | 2026-04-20 | 240 | NaturalConvection に `extra_scalars` を統合（Phase 6.1b、温度+トレーサー同時輸送） |
| 23 | 2026-04-21 | 254 | `AquariumGeometryProcess` 新設（Phase 6.2a、90×30×45 cm 水槽 + 底床/ガラス/水マスク） |
| 24 | 2026-04-21 | 267 | `HeaterProcess` + 水槽ヒーター自然対流デモ（Phase 6.2b、Geometry+Heater+NC 3 段連携） |
| 25 | 2026-04-21 | 286 | `AquariumFilterProcess` + `InternalFaceBC`（Phase 6.3a、外部フィルター循環 INLET/OUTLET BC） |
| 26 | 2026-04-23 | 286 | `examples/aquarium_filter_circulation.py`（Phase 6.3b、Geometry+Heater+Filter+NC 4 段連携デモ） |
| 27 | 2026-09-02 | 286 | 単軸押出解析 設計策定（展開チャネル 2.5D、RTD 目的、実装未着手） |
| 28 | 2026-09-03 | 438 | 単軸押出解析 Phase 1/1.5 実装（`extruder/` 6 プロセス、ゲート G1〜G4 全通過、OpenFOAM G3 検算） |
| 29 | 2026-09-04 | 460 | ゲート G5 文献 RTD 照合（Pinto–Tadmor 1970 再導出、浅溝極限で収束。Phase 2 前提を実機データから差し替え） |
| 30 | 2026-09-04 | 502 | 2D Brinkman 補正 NS (FVM, Newton–Krylov) 新設 + U ターン/平板 収束破綻の再現実験と機構切り分け（局所/大域 Δτ 比較、手元構成ミラー `nsb/` を含む） |
| 31 | 2026-09-04 | 516 | Brinkman 流路の座標マスク境界条件 + 質量流入 + 領域内マニホールド + 位置・径の随伴設計感度（冷却流路設計の前段） |
| 32 | 2026-09-04 | 531 | nsb を xkep_cae_fluid から切り離し（`data`/`assembly` のコピー方式 + 同期スクリプト）+ 高速化見積り実測（LU 分解が 70〜81%）+ PARDISO 化（後方互換なし、分解/三角解のスレッド分割、`KMP_BLOCKTIME=0`）+ 前処理 LU の遅延更新（144×96: 40 s → 17 s） |
| 33 | 2026-09-05 | 598 | ykep .inp 入力フォーマット（Abaqus 風キーワード構文、`*PARAMETER`/`*CONTROLS`/`*GRID`、`CaseDefinition` 中立表現、`ykep -j=<job>.inp int` CLI）+ NS/伝熱マッピング + 例題 2 本（Ra=1000 キャビティ Nu=1.169） |
| 34 | 2026-09-05 | 615 | 3D レンダリング: messi mirador 連携 `MiradorExportProcess`（断面スラブ + 速度矢印 + 残差マップ + 任意平面の view cut `--cut`、`FORMAT=HTML` 自動出力 / `ykep view`）+ messi v0.10.0（要素場カラーマップ Abaqus レインボー・矢印・`.vtk` リーダ・操作パネル畳み込み・断面 view cut） |
| 35 | 2026-09-06 | 775 | ソルバー体験の現在地整理 + Phase 11 残 TODO 消化（圧力補正の非直交補正・適応緩和の共有規則・Rhie–Chow の緩和非依存化・2 次要素/角錐/バッフルの `InpMeshProcess`・`solve_energy`・CI skip） |
| 36 | 2026-09-06 | 826 | 汎用記法（.inp）で押出級の流れを書く（Phase 12）: 周期境界 `*BOUNDARY, TYPE=PERIODIC`（`MeshData.face_offset` + fvm 層の `neighbour_centers`）+ 一様体積力 `*DLOAD, BX/BY/BZ/BF` + 非ニュートン粘度 `*VISCOSITY, TYPE=POWER LAW｜CARREAU`（`fvm/viscosity.py`、非構造 γ̇ の Picard）+ 回転壁 `*ORIENTATION`/`*MPC`/自由度 4-6 + Stokes `CONVECTION=NONE` と連成 `PRESSURE_VELOCITY=COUPLED` + `ExtruderChannelInpProcess` と例題 extruder-channel-1（専用 2.5D ソルバーと Q が機械精度一致、Q_axial 1.4e-3） |
| 37 | 2026-09-06 | 838 | 非構造メッシュの粒子追跡と滞留時間分布（Phase 12 完了）: 面流束から再構成したセル内アフィン場（直交六面体で Pollock、四面体で RT0）を辿る `ParticleTrackFVMProcess` + `ResidenceTimeProcess`、NS 結果に γ̇ と混合指数 λ を常時追加、厳密関係 ⟨t⟩ = length·V/Σw が周期 Poiseuille で 1e-12 一致、構造格子トラッカーと t_p10/p50/p90 が 1e-3 台で一致。**後方互換を全撤去**（RegistryProxy / deprecated 機構 / 再輸出シム）+ **全件テスト 14 分 26 秒 → 2 分 28 秒**（流れ場の共有・6 件を slow へ・pytest-xdist） |
| 38 | 2026-09-07 | 897 | nsb の線形ソルバーに SIMPLE 型ブロック前処理（運動量 ILU + Schur 補元 smoothed aggregation AMG、`nsb/precond.py`、pyamg 必須）を `linear_solver="jfnk_simple"` / `"dc_simple"` として追加。Fluent との差（疎直接 LU vs AMG、三角解 1 スレッド）の整理、部品の切り分け（GS・運動量 AMG は高 CFL で発散、Ruge–Stüben 199 反復 → SA 43 反復）、4 コア実測 で 288×192 が PARDISO 229.6 s → 86.3 s（`gmres_tol=1e-2`、**2.66×**、1 Newton 反復 6.4 s → 3.3 s）、144×96 が 18.6 s → 11.3 s。運動量 ILU の零ピボット（drop_tol 1e-2 で Newton 途中に発生）は締めて組み直すリトライで回避 |
| 39 | 2026-09-07 | 923 | nsb 線形ソルバーの高速化 第 2 段: 自作 FGMRES（scipy gmres は指定 rtol より深く解いていた。JFNK の FD matvec の非線形性 1e-4〜1e-3 と Givens 推定の食い違いを同定）+ SA 階層の再利用（集約固定・Galerkin 再構築 667 → 10 ms）+ V サイクル直呼び + 残差評価の numba 化（5.8 → 0.5 ms）+ pyamg 乱数の固定で決定化。20 コア実測（288×192）で GMRES 1 反復 26.6 → 16.7 ms、組立 863 → 265 ms、総時間は status-38 のコード 73.8 s → 26〜43 s（Newton 反復数 22〜36 回の経路差、PARDISO 55 s）。次の律速は前処理適用 × GMRES 反復数（高 CFL で 45〜56）と SER の経路敏感さ |
| 40 | 2026-09-08 | 917 | nsb の制御則を一長一短の切替だけに絞る（静止場発進・LU 直接/defect correction・運動量 Jacobi・CFL backtracking・速度下限なし・numpy 残差・SA 毎回構築を廃止）。収束判定の参照は常に Stokes 解の完全 NS 残差、初期場は `u0/v0/p0` か Stokes 解、SER は古典形 `cfl_init·|R_ref|/|R_init|` で出発。粗格子 144×96 の解を注入した 288×192 が 36 → 16 Newton（42.3 s → 15.1 s + 3.7 s、解の差 5e-6） |
| 41 | 2026-09-08 | 927 | 入れ子反復の粒度の目安と SER 制御の良いところ。`nsb.nested.solve_nested`（双一次 / 注入の 2× 補間、粗い順に解いて次段の初期場に）。目安: 最粗格子 72×48 から 2 倍ずつ全段・各段 tol 1e-4（粗格子側は細格子の 7〜10%、段を増やしても増えない。1e-6 まで解くのは損）。細格子側の Newton は初期場の質で決まる下限（288×192: 13、576×384: 20〜30）。288×192 が 44.5 → 18.5 s、576×384 が 310 → 195 s。SER は乗法形のまま `cfl_init` 0.5 → 0.25 + 線形解が実質失敗（真の残差比 0.3 超）したステップの棄却 `reject_lin_ratio`。成長率・減少率・上限・前処理更新則・古典形目標則・GMRES 余力則は掃引で全て劣る。発散は「GMRES 200 反復で解けなかったごみ修正量を採用」が原因で、CFL の上限は線形ソルバーの余力で決まる。uturn 144×96 U=1/2 の発散を解消 |
| 42 | 2026-09-08 | 960 | **nsbp**: nsb の離散化をそのまま PETSc（petsc4py）で解く別パッケージ。DMDA の MPI 分割、FD カラーリングの厳密ヤコビアン（75 色、13 点疎パターン）、FGMRES + ASM(重なり 2)/ILU(2)、定常残差で駆動する SER、Venkatakrishnan リミターの凍結（厳密 Newton が分岐で 1e-4 に停滞するのを止める）。flat 288×192: 8 ランク 5.2 s（nsb 44.5 s）、入れ子 4.3 s（nsb 18.5 s）、576×384 入れ子 39.1 s（nsb 195 s）、uturn 144×96 U=1: 1.2 s（nsb 13.2 s）。解の差 1e-5〜2e-4。Schur fieldsplit + hypre は uturn で発散、bjacobi は並列で KSP 6 倍、Euclid は 17 倍遅い、幾何 MG は発散。PETSc は MPICH 同梱で手ビルド（システム OpenMPI はシングルトン起動がハング）。全件テストで既存の失敗 18 件を検出（941 passed / 18 failed / 1 xfailed。失敗 18 は全て既存: `test_inp_runner` 9・`test_post_mirador` 8（いずれも `VizMixin.export_html()` の `vector_field` 引数）・`test_natural_convection::TestAMGPressureSolver::test_adaptive_relaxation` 1。nsbp は全通過） |
| 43 | 2026-09-09 | 995 | **nsbm**: nsb の初期場を学習で出す（72×48 固定、8 ファミリ × 4 壁ポートの θ サンプラー、4000 件の並列生成、教師あり UNet）。結論は否定的: 学習初期場は既定の制御則で Stokes 発進に 41 勝 299 敗（中央値 19 vs 13）、kNN 補間と同程度。機構は (a) 閉塞セルの速度が Brinkman 抗力で残差比 1000（マスクで解消）、(b) 反復数は SER の CFL 梯子で決まり予測場の残差比 2〜8 では出発 CFL が上がらない、(c) 予測場は Newton の吸引域の外（減衰なし 1 歩は 6 割で棄却）。`cfl_init` 4 の方が効く（13 → 7、未収束 3%）。効くのは遅い裾と未収束 60 件中 8 件の救済 |
| 44 | 2026-09-09 | 1007 | **nsbm 第 2 段**: 残差損失 + cfl_init ヘッド（gyp さん指定）、Stokes 床 + 補正（messi-bb の助言）、局所 Galerkin、cfl_init の掃引と選択器。場の精度は床 + 補正で R² u 0.91 → 0.97・p per-instance 0.53 → 0.97 に上がるが初期残差比は 2 のままで反復は 149 勝 173 敗。Newton 射影 1 歩を足すと 207 勝 138 敗、閉塞のない uniform / quad / sin2d では中央値 13 → 7（閉塞系は壁際の残差で負け）。残差損失は残差項を下げても実反復数は増える（cfl ヘッドだけで 13 → 10）。局所 Galerkin は残差比 0.6 でも五分、残差を減らした場ほど破綻。効くのは cfl_init: 固定 0.25 平均 20.6 → 早期やり直し規則（cfl 4、10 反復で残差比 > 0.3 なら 0.25）15.9、選択器 16.8、オラクル 10.8 |
| 45 | 2026-09-09 | 1007 | **nsb 蛇行流路（messi trama）の収束不良**を高 Re・階段・前処理の 3 観点で可視化（コード修正なし）。κ₂(J1+τ) は 1.6e12 → 2.1e12 でほぼ不変（両端とも閉塞領域）、変わるのは RC 結合 d ∝ 1/Re → 厳密 J1 歩の圧力成分 60 → 8.4e6、真の残差比 1.16 → 81。階段は無関係（直線流路は 0〜45° どれも健全）、引き金は蛇行の U ターン。SIMPLE + JFNK は Givens 0.077 / 真 30、LU(J1) は 3.3e-3 だが Newton 一歩で残差 5060 倍。内部/壁ポートで κ は同等、食い違いは内部が 30〜50 倍。τ の 2 重カウントを発見。**改修**: 色分け FD の厳密ヤコビアン（`jacobian="fd"`）+ τ 修正で 0.0015 kg/s が 80 反復未達 → 22 反復収束。0.15 は直接解法 20 走行で全滅（cfl 大は出口 sink の非線形性、cfl 小は時間発展の劣化版）。継続法は Re_h 50〜150 で停滞 → 次はリミター凍結 / 物理時間の非定常計算。[図解レポート](../reports/nsb-trama-convergence.md) |
| 46 | 2026-09-10 | 1007 | **nsb 蛇行流路にリミター凍結・隙間の摩擦則・物理時間の非定常を実装して反復数を測る。0.15 kg/s は未収束**。凍結（残差比 1e-3 かつ ψ 安定で凍結、解凍は patience 3）で継続法が 1 段先まで登る（内部ポート 0.005、壁ポート 0.015 が初めて収束）が次の段は凍結閾値より上で停滞。摩擦則（Blasius 型、抗力 1.3 倍）は収束性を悪化。非定常（`nsb/unsteady.py`、後退 Euler + Δt 後退）は細格子で Δt 7.8e-5 → 5 ms に育てて過渡追跡中。落とし穴: fd ヤコビアンは SIMPLE と組まない、ラインサーチは逆効果。**status-45 の τ 修正が既存テスト 7 件を壊していた**（擬似時間残差 SER の偽収束）→ 既定を定常残差 SER に切替。[図解レポート §10](../reports/nsb-trama-convergence.md) |
| 47 | 2026-09-10 | 1007 | **OpenFOAM による独立検算**（ゲート G3 の Docker ラッパ・`foam_io` を転用、`experiments/nsb/trama_of_*.py`）。nsb の式（2D 非圧縮 NS + Brinkman 抗力 12μ/h²）は `simpleFoam` + DarcyForchheimer 多孔質源 d = 12/h² と 1 対 1。**0.15 kg/s は OpenFOAM でも定常で解けない**（8000 反復で p 残差 8.4e-2 に平坦）。理由は「定常解が存在しない」。決めるのは N = L_drag/w = ρuh²/(12μw) = 渦が抗力で消える移動距離 ÷ 流路幅（0.15 kg/s で 13.3）。境目は N 1.33 収束 / 2.22 停滞で、**nsb の継続法の境目（0.015 収束 / 0.025 停滞）と一致**。Reynolds 数ではない（流量そのままで隙間を 1/12 にすると Re_h 1449・Re_w 157895 のまま N 1.11 で 233 反復収束）。ただし N だけでもなく Re_w ≳ 2000 も要る。乱流モデルでは定常化しない（必要な渦粘性 4.2e-4 は隙間乱流の目安の 22 倍）。0.15 kg/s の答えは非定常の時間平均で、入口の必要圧力ヘッド **25.5 ± 2.3 kPa**、速度変動は平均流速の 45%。費用は OpenFOAM 615 s vs nsb 8.3 時間（1 ステップ 0.20 s vs 25 s、原因は死んだ閉塞セル 73% と毎ステップの直接 LU）。[図解レポート §11](../reports/nsb-trama-convergence.md) |
| 48 | 2026-09-10 | 1010 | **nsb / nsbp: 厚み h ≤ `h_solid` のセルを壁として解かない**。閉塞域を Brinkman 抗力の栓で表すのをやめ、面ごとの種別 `wall_x` / `wall_y` で内部にも no-slip 壁を置き、固体セルの行を残差 0 + ヤコビアン単位行にして δ = 0 に固定する（圧力基準に繋がらない孤立塊は刈り込み）。**効くのは大きさよりコントラスト**: 同じ大きさのまま抗力コントラストを 1e4 → 1e2 に下げるだけで GMRES が 2441 → 1303 と半減し壁セル版（1299）と並ぶ。uturn 144×96 U=1 で `jfnk` 60 反復未収束 39.5 s → 28 反復 7.9 s、nsbp 12.2 → 4.2 s、入れ子 97.0 → 67.1 s。物理は Δp 1.0% / u_max 0.04% 差。h_blocked を上げる代替案は物理を壊す（Δp −24%、u_max −35%）。**OpenFOAM 検算**: 定常 0.005 kg/s で流路幅 1 つぶん離れれば速度 L2 **1.8%** / 圧力 **0.07%**（栓モデル同士は 8% で頭打ちだった）。**0.15 kg/s の非定常**は 1 ステップ 46.7 s → **2.07 s**（22 倍）で物理時間 4.17 s を 2.53 時間。時間平均の積分量は合う（平均流速 0.06%、圧力 span 3.4%、必要ヘッドの振れ幅 9.7% vs 9.1%）が場は合わない（L2 58%）。原因は**ポートの与え方が抗力長で下流に運ばれる**こと: 変動の立ち上がりが 400 mm ≈ ターン 3 つぶんずれ、差は減衰長 597 mm ＝ L_drag 459 mm の 1.30 倍で消える。境目（N ≈ 2）と status-47 の結論は不変。nsbp は壁セルでも勝てない（1 プロセスの nsb 5.7 s vs 8 ランク 22.3 s。ILU(2) はコントラストを消しても 200 反復上限に張り付く）。[図解レポート §12](../reports/nsb-trama-convergence.md) |
