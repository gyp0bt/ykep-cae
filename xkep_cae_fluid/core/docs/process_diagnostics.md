# Process 実行診断

[<- README](../../../README.md)

## 概要

全 `AbstractProcess.process()` 呼び出しを自動記録し、
プロセス使用状況のレポートを生成する診断モジュール。

## 機能

1. **実行ログ**: `ProcessExecutionLog` シングルトンが全プロセス呼び出しを記録
2. **呼び出し元追跡**: `inspect.stack()` で呼び出し元ファイル・関数・行番号を自動検知
3. **deprecated エラー**: `ProcessMeta.deprecated=True` のプロセス実行時に `DeprecatedProcessError` を送出
4. **レポート生成**: `docs/generated/process_usage_report.md` にプロセス使用レポートを自動出力

## アーキテクチャ

```
ProcessMetaclass.traced_process()
  -> ProcessExecutionLog.record_start()   # inspect.stack() で呼び出し元検知
  -> original process()                    # 実プロセス実行
  -> ProcessExecutionLog.record_end()      # 実行時間 + 警告記録
  -> atexit: write_report()               # セッション終了時にレポート出力
```
