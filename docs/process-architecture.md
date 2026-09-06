# Process Architecture 設計仕様

[<- README](../README.md) | [<- docs](README.md)

## 概要

xkep-cae / xkep-cae-fluid 共通のソフトウェアアーキテクチャ。
全ての計算ロジックを **Process** として契約化し、依存関係を明示的に管理する。

## 設計原則

1. **全てはProcess**: 計算ロジックは `AbstractProcess` のサブクラスとして実装
2. **依存は宣言的**: `uses` クラス変数で依存プロセスを明示宣言
3. **Strategyで直交分離**: 離散化手法・乱流モデル等の振る舞い軸は Strategy Protocol + StrategySlot で注入
4. **設計文書は必須**: `ProcessMeta.document_path` でドキュメントを紐付け（存在チェックあり）
5. **テストは1:1**: `@binds_to` で各プロセスに1つのテストクラスを紐付け
6. **不変データ**: プロセス間のデータ受け渡しは `frozen dataclass` で不変性を保証

## クラス階層

```
AbstractProcess[TIn, TOut]  (ABC + ProcessMetaclass)
+-- PreProcess              前処理
+-- SolverProcess           求解
+-- PostProcess             後処理
+-- VerifyProcess           検証
+-- BatchProcess            バッチ実行
```

## ProcessMeta

各具象プロセスは `ProcessMeta` を定義する必要がある:

```python
class MyProcess(SolverProcess[MyInput, MyOutput]):
    meta = ProcessMeta(
        name="MyProcess",
        module="solve",
        version="0.1.0",
        document_path="docs/my_process.md",
        stability="experimental",   # experimental / stable / frozen
        support_tier="ci-required", # ci-required / compat-only / dev-only
    )
    uses = [DependencyProcess]

    def process(self, input_data: MyInput) -> MyOutput:
        ...
```

## ProcessMetaclass の自動ラップ

`ProcessMetaclass` は `process()` メソッドを自動的にラップし、以下を透過的に実現する:

1. **実行トレース**: `ProcessExecutionLog` に呼び出し記録
2. **プロファイリング**: `_profile_data` に実行時間を蓄積

## StrategySlot

Strategy Pattern を型安全に実装するディスクリプタ:

```python
class MySolver(SolverProcess[...]):
    convection = StrategySlot(ConvectionSchemeStrategy)
    turbulence = StrategySlot(TurbulenceModelStrategy, required=False)

    def __init__(self, strategies):
        self.convection = strategies.convection
        self.turbulence = strategies.turbulence
```

`effective_uses()` が静的 `uses` + StrategySlot の動的依存を統合する。

## ProcessRegistry

全プロセスは `__init_subclass__` で自動登録される:

- `ProcessRegistry.default()`: グローバルシングルトン
- `registry.filter_by_category("SolverProcess")`: カテゴリ絞り込み
- `registry.filter_by_stability("stable")`: 安定性フィルタ
- `registry.dependants_of("MyProcess")`: 逆依存検索
- `registry.isolate()`: テスト用スナップショット

## ProcessRunner

`process()` を統一的に実行する管理クラス:

```python
runner = ProcessRunner(ExecutionContext(
    dry_run=False,
    profile=True,
    validate_deps=True,
    checksum_inputs=True,
))
result = runner.run(my_process, input_data)
```

- `validate_deps`: uses 依存がレジストリに存在するか検証
- `checksum_inputs`: 入力 numpy 配列の不変性を checksum で検証
- `dry_run`: パイプライン構成の検証（実行はスキップ）

## BenchmarkRunner

STA2防止のため、プロセス実行の全記録をYAMLマニフェストに自動保存:

- 環境情報（git commit, branch, dirty, Python/NumPy バージョン）
- Config パラメータ（frozen dataclass を再帰シリアライズ）
- 結果サマリー（ユーザー定義の抽出関数で取得）

## 契約検証（contracts/validate_process_contracts.py）

| コード | 検証内容 |
|--------|---------|
| C3 | テスト未紐付けプロセスの検出 |
| C5 | process() 内の未宣言依存（AST解析） |
| C7 | process() のメタクラスラップ漏れ |
| C9 | frozen dataclass numpy 配列の不変性チェック |
| C15 | document_path で指定されたドキュメントの実在検証 |

## データ型（流体向け）

### MeshData

構造化格子・非構造化格子の両方に対応:

```python
@dataclass(frozen=True)
class MeshData:
    node_coords: np.ndarray      # (n_nodes, ndim)
    connectivity: np.ndarray     # (n_cells, max_nodes_per_cell)
    cell_volumes: np.ndarray     # (n_cells,)
    face_areas: np.ndarray       # (n_faces,)
    face_normals: np.ndarray     # (n_faces, ndim)
    face_owner: np.ndarray       # (n_faces,) FVM用オーナーセル
    face_neighbour: np.ndarray   # (n_internal_faces,) FVM用隣接セル
    dimensions: tuple | None     # 構造化格子の場合 (nx, ny, nz)
```

### FlowFieldData

```python
@dataclass(frozen=True)
class FlowFieldData:
    velocity: np.ndarray         # (n_cells, ndim)
    pressure: np.ndarray         # (n_cells,)
    temperature: np.ndarray      # (n_cells,) optional
    turbulent_ke: np.ndarray     # (n_cells,) optional
    turbulent_epsilon: np.ndarray # (n_cells,) optional
```

### SolverInputData

```python
@dataclass(frozen=True)
class SolverInputData:
    mesh: MeshData
    boundary: BoundaryData
    fluid: FluidProperties
    coupling_method: str = "SIMPLE"
    relax_velocity: float = 0.7
    relax_pressure: float = 0.3
    ...
```

## Strategy Protocol（流体向け）

| Protocol | 責務 | 実装候補 |
|----------|------|---------|
| ConvectionSchemeStrategy | 対流項離散化 | Upwind, QUICK, TVD, WENO |
| DiffusionSchemeStrategy | 拡散項離散化 | 中心差分, 非直交補正付き |
| TimeIntegrationStrategy | 時間積分 | 定常, 陰的Euler, BDF2 |
| TurbulenceModelStrategy | 乱流モデル | Laminar, k-epsilon, k-omega SST |
| PressureVelocityCouplingStrategy | 圧力-速度連成 | SIMPLE, SIMPLEC, PISO |
| LinearSolverStrategy | 線形ソルバー | 直接法, GMRES, BiCGSTAB, AMG |
