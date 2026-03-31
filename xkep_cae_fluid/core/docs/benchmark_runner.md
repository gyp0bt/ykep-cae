# BenchmarkRunner -- 実行マニフェスト自動記録

[<- README](../../../README.md)

## 概要

STA2防止（担当者間再現性ルール）のため、プロセス実行時の全パラメータ・
環境情報・結果サマリーを自動記録する仕組み。

## 目的

1. **パラメータ自動記録**: frozen dataclass の全フィールドをYAMLシリアライズ
2. **環境記録**: git commit/branch/dirty、Python/NumPy バージョン
3. **結果紐付け**: ソルバー結果サマリー + statusファイルリンク
4. **再現手順生成**: 同一結果を得るためのコマンド列を自動出力

## アーキテクチャ

```
RunManifest (frozen dataclass)
+-- environment: EnvironmentInfo  <- git/Python自動取得
+-- config_params: dict           <- dataclass -> dict 再帰シリアライズ
+-- results_summary: dict         <- スカラー結果抽出
+-- process_name: str
+-- elapsed_seconds: float
+-- timestamp: str

BenchmarkRunnerProcess (BatchProcess)
+-- input: BenchmarkRunInput[TIn]
|   +-- process: AbstractProcess[TIn, TOut]
|   +-- config: TIn (frozen dataclass)
|   +-- status_file: str | None
|   +-- result_extractors: dict[str, Callable]
+-- output: BenchmarkRunResult[TOut]
|   +-- result: TOut
|   +-- manifest: RunManifest
+-- YAML出力: docs/benchmarks/{process}_{timestamp}.yaml
```
