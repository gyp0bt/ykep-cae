# データスキーマ仕様

[<- README](../README.md) | [<- docs](README.md)

## 概要

xkep-cae-fluid のプロセス間データ受け渡しに使用する `frozen dataclass` の仕様。
全データ型は `xkep_cae_fluid.core.data` に定義される。

## 一覧

| データ型 | カテゴリ | 用途 |
|---------|---------|------|
| MeshData | メッシュ | 構造化/非構造化メッシュの節点・セル・面情報（`core/data.py`） |
| `fvm.PatchBC` / `BoundaryFaces` | 境界条件 | パッチ名 → Dirichlet / Neumann / Robin / ゼロ勾配（[fvm-layer.md](design/fvm-layer.md)） |
| 各ファミリーの Input / Result | ソルバー入出力 | `HeatTransferFVMInput`、`NavierStokesFVMInput`（`FlowPatchBC`、`ScalarSpec`、`InternalCellBC`）、`DarcyFlowInput`、`NaturalConvectionInput` 等（各設計文書） |

旧来の汎用スキーマ `BoundaryData` / `FluidProperties` / `FlowFieldData` / `SolverInputData` /
`SolverResultData` / `VerifyInput` / `VerifyResult` はどのプロセスも使っていなかったので
2026-09-06 に削除した（Phase 11）。

## MeshData

FDM（構造化格子）と FVM（非構造化格子）の両方に対応する。

### 構造化格子の場合

```python
mesh = MeshData(
    node_coords=coords,          # (nx*ny*nz, 3)
    connectivity=conn,           # (n_cells, 8) for hexahedra
    cell_volumes=volumes,        # (n_cells,)
    dimensions=(nx, ny, nz),     # 構造化を示すフラグ
)
assert mesh.is_structured == True
```

### 非構造化格子（FVM）の場合

```python
mesh = MeshData(
    node_coords=coords,
    connectivity=conn,
    cell_volumes=volumes,
    face_areas=areas,            # (n_faces,)
    face_normals=normals,        # (n_faces, 3)
    face_centers=centers,        # (n_faces, 3)
    cell_centers=cell_centers,   # (n_cells, 3)
    face_owner=owner,            # (n_faces,) 各面のオーナーセル
    face_neighbour=neighbour,    # (n_internal_faces,) 内部面の隣接セル
)
assert mesh.is_structured == False
```

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `n_nodes` | int | 節点数 |
| `n_cells` | int | セル数 |
| `ndim` | int | 空間次元（2 or 3） |
| `is_structured` | bool | 構造化格子か |
