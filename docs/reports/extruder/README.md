# 単軸押出 図解レポート

[<- docs](../../README.md) | [<- 設計文書](../../design/single-screw-extruder.md)

`xkep_cae_fluid/extruder/` の計算結果を、押出初心者向けに図解したページ 2 本。
どちらも Claude Code の Artifact として公開済み（スマホからリンク 1 回で開ける）。
HTML の実体もここに置いてあるので、リンクが切れてもブラウザで直接開ける。

| ページ | 公開 URL | 実体 | 内容 |
|---|---|---|---|
| 🌀 **押出機の中の8秒** | https://claude.ai/code/artifact/4ed6f1d4-6fa9-4a17-8fdc-d93af00724d4 | [extruder-primer.html](extruder-primer.html) | 押出の基本のキ。スクリューの展開、引きずり流れと圧力流れ、循環、隙間の役割、滞留時間分布、混練性まで 10 節・SVG 12 枚 |
| 🔥 **押出断面のフィールド** | https://claude.ai/code/artifact/bbc1809e-ca05-4e34-bc57-1ff6ea01fb34 | [extruder-fields.html](extruder-fields.html) | 40mm 機・計量部断面の実解。軸方向速度・断面内速度・粘度・せん断速度・粘性発熱の 5 面 + 隙間拡大 2 面 + 流線 63 本。ニュートン vs べき乗則の比較 |

## 再生成

`extruder-fields.html` は [build_fields.py](build_fields.py) が
`ExtruderFlowProcess` を回して生成する（ラスタは base64 PNG、オーバーレイは SVG）。

```bash
cd ~/work/ykep-cae
OMP_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python docs/reports/extruder/build_fields.py
```

出力はスクリプトと同じディレクトリに書かれる。約 450 KB、実行 1〜2 分。

`extruder-primer.html` は手書き（データは設計文書 §6 の諸元と RTD 結果を埋め込み）。
数値を更新したいときはファイル内の値を直接編集する。

## 公開の更新

同じファイルパスで `Artifact` ツールに publish し直せば同じ URL が更新される。
Artifact の URL は `docs/design/single-screw-extruder.md` §9 にも記載。
