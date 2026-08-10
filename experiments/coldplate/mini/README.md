# mini/ — 写経再構築用の最小核

[← coldplate README](../README.md)

`darcy.py` 本体 (~700 行) から可視化・ポート設計・粒度一般化・ウォームスタートを
剥がし、**物理と離散 adjoint だけ** を残したリファレンス実装。

| ファイル | 行数 | 役割 |
|---|---|---|
| `darcy_core.py` | ~360 | 物理コア (幾何・adjoint・φ 閉包・流れ・熱・目的・Adam) |
| `test_darcy_core.py` | ~140 | 検定器 10 件 — 再構築版がこれを通れば正しい |

## 写経の推奨順

1. **`test_darcy_core.py` を先に読む** — 何が保証されるべきか (保存則・勾配の
   FD 一致・閉包の単調性) を仕様として頭に入れる。検定は実装詳細に依存しない
   ように書いてあるので、自分の実装スタイルで書き直しても通せる
2. **`SparseSolve`** (~20 行) — 離散 adjoint の全て。backward の
   `λ = A⁻ᵀḡ, ∂J/∂A_kl = −λ_k·x_l` が「設計変数が何千個でも勾配コストは
   前進解 1 回分」の源泉。ソルバ外側の連鎖律 (φ→K→行列成分) は torch の
   autograd が自動で繋ぐので、手書きの随伴導出はここだけ
3. **φ の 3 閉包** (`permeability` / `interlayer_u` / `ergun_beta`, ~60 行) —
   物理的主張のほぼ全部。特に `interlayer_u` の「フィン抵抗を挟む」構造が
   旧モデルの一桁楽観を排除した本丸
4. **`solve_flow` / `solve_heat`** — FVM の調和平均コンダクタンスと 2N×2N の
   2 層連成。Forchheimer の緩和付き Picard (裸の固定点は減衰振動する) も含む
5. **`objective` / `optimize`** — smooth-max + 分散罰則 + γ·ΔP の重み付き和と
   素朴な Adam ループ (100 反復ごとのロジットクランプ = 飽和勾配死の予防)

## 検定

```bash
cd experiments/coldplate/mini
OMP_NUM_THREADS=4 ../../../.venv/bin/python -m pytest test_darcy_core.py -q  # 10 passed (数秒)
```

再構築版を検定するときは `import darcy_core as dc` を自分のモジュールに
差し替えるだけでよい。

## 本体との差分 (mini に無いもの)

- 旧 logK モデル (`pin_fin=False` 分岐) — mini はピンフィン一本
- ポート位置設計 (`optimize_ports`)・任意設計粒度 (`design_shape`)・
  ウォームスタート (`xi0`)
- 可視化 (`panel` / `cell_speed`)・γ スイープ等の実行スクリプト群
- `evaluate()` の指標辞書 (mini は objective のスカラーのみ)

物理と勾配は本体と同一 (同じ式・同じ許容誤差のテスト)。
