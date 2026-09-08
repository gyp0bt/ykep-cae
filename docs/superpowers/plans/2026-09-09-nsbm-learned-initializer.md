# nsbm（学習初期場）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 形状パラメータ θ から nsb（72×48）の収束解を UNet で近似し、初期場に入れて Newton 反復数を減らす。Stokes 発進・kNN 補間と比較して価値を判定する。

**Architecture:** `nsbm/` に torch 依存の別パッケージ（families → dataset → model → train → evaluate の直線パイプライン）。nsb は変更しない。データは multiprocessing で並列生成し npz シャードに保存、学習は CPU torch、評価は `solve_steady` を 3 通りの初期場で回して `n_iter` を比較する。

**Tech Stack:** numpy, scipy.ndimage, nsb, torch 2.14 (CPU), multiprocessing(spawn), pytest

**Spec:** `docs/superpowers/specs/2026-09-09-nsbm-learned-initializer-design.md`

## Global Constraints

- 格子 `nx, ny = 72, 48`、`lx, ly = 0.7, 0.4`、nsb 既定物性（rho 1000, mu 1e-3, mu_b 1e-3）
- ソルバー設定は `NSBSettings()` 既定、収束判定は変えない
- 計算実行は `2>&1 | tee` でログを残す。数 GB 級の走行は `~/.claude/hooks/memcap` 下で
- ワーカーは numpy import 前に `OMP_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`
- 文書は日本語、README へのバックリンク
- lint: `ruff check nsbm/ tests/ experiments/nsbm/ && ruff format nsbm/ tests/ experiments/nsbm/`

---

## ファイル構成

| ファイル | 責務 |
|---|---|
| `nsbm/__init__.py` | 公開 API の再輸出 |
| `nsbm/families.py` | `Theta`（θ の記述、JSON 化可）、`sample_theta(seed)`、`build_h(theta)`、`build_bc(theta)`、`build_input(theta)`、連結性チェック |
| `nsbm/features.py` | `make_x(theta, h)`（8 ch 入力画像）、`normalize_y / denormalize_y`（スケール `u_in`, `p_ref`） |
| `nsbm/dataset.py` | `solve_sample(seed)`（1 件生成）、`generate(seeds, n_workers, out_dir)`（並列 + シャード保存）、`load_shards(dir)` |
| `nsbm/model.py` | `UNet(in_ch=8, out_ch=3)` |
| `nsbm/train.py` | `train(data, out_dir, epochs, ...)`、`split_by_family(metas, seed)` |
| `nsbm/evaluate.py` | `evaluate(model, data, split, ...)`: Stokes / kNN / UNet の `n_iter` 比較、YAML 出力 |
| `experiments/nsbm/{gen,train,eval}.py` | CLI 入口 |
| `tests/test_nsbm_families.py`, `tests/test_nsbm_features.py`, `tests/test_nsbm_dataset.py`, `tests/test_nsbm_model.py` | テスト |

---

### Task 1: families — θ のサンプルと h 場 / BC の構築

**Files:** Create `nsbm/__init__.py`, `nsbm/families.py`, Test `tests/test_nsbm_families.py`

**Interfaces（Produces）:**
```python
FAMILIES = ("uniform", "sin2d", "quad", "grf", "uturn", "serpentine", "pins", "blobs")
@dataclass(frozen=True)
class Port:      # inlet / outlet
    wall: str    # "west" | "east" | "north" | "south"
    s0: float    # 壁に沿った区間 [m]（west/east は y、north/south は x）
    s1: float
@dataclass(frozen=True)
class Theta:
    seed: int; family: str; h0: float; u_in: float; inlet: Port; outlet: Port; params: dict[str, float]
    def to_dict(self) -> dict; @staticmethod def from_dict(d) -> "Theta"
def sample_theta(seed: int, families: Sequence[str] = FAMILIES) -> Theta   # 連結性を満たすまで再抽選
def build_h(theta: Theta) -> np.ndarray            # (72, 48), > 0
def build_bc(theta: Theta) -> BC
def build_input(theta: Theta, settings: NSBSettings | None = None, init=None) -> NSBInput
def port_cells(port: Port) -> tuple[np.ndarray, np.ndarray]   # 境界に接するセルの (i, j) index 配列
def connected(h: np.ndarray, inlet: Port, outlet: Port, h_blocked: float) -> bool
```

- [ ] **Step 1: 失敗するテストを書く** `tests/test_nsbm_families.py`

```python
import numpy as np
import pytest
from nsbm.families import FAMILIES, Theta, build_bc, build_h, build_input, connected, port_cells, sample_theta

@pytest.mark.parametrize("family", FAMILIES)
def test_build_h_shape_positive(family):
    th = sample_theta(seed=11, families=[family])
    h = build_h(th)
    assert h.shape == (72, 48) and np.all(h > 0) and np.isfinite(h).all()
    assert th.family == family

@pytest.mark.parametrize("family", ["uturn", "serpentine", "pins", "blobs"])
def test_blocked_families_connected(family):
    for seed in range(5):
        th = sample_theta(seed=seed, families=[family])
        h = build_h(th)
        assert connected(h, th.inlet, th.outlet, th.h0 / 100)
        assert (h <= th.h0 / 100 * 1.001).any()   # 閉塞セルが存在する

def test_ports_do_not_overlap_same_wall():
    for seed in range(50):
        th = sample_theta(seed)
        if th.inlet.wall == th.outlet.wall:
            a, b = th.inlet, th.outlet
            assert a.s1 <= b.s0 or b.s1 <= a.s0

def test_seed_reproducible_and_roundtrip():
    a, b = sample_theta(3), sample_theta(3)
    assert a == b
    assert Theta.from_dict(a.to_dict()) == a

def test_port_cells_on_boundary():
    i, j = port_cells(Port("east", 0.1, 0.2))
    assert np.all(i == 71) and len(j) > 0

def test_build_input_solves():
    from nsb import solve_steady
    th = sample_theta(seed=1, families=["uniform"])
    inp = build_input(th)
    assert inp.nx == 72 and inp.ny == 48
    res = solve_steady(inp, log=None)
    assert res.converged
```

- [ ] **Step 2: 失敗確認** `pytest tests/test_nsbm_families.py -x -q` → ImportError
- [ ] **Step 3: 実装** `nsbm/families.py`（要点）
  - `NX, NY, LX, LY = 72, 48, 0.7, 0.4`、セル中心 `xc, yc`、`X, Y = meshgrid(indexing="ij")`
  - `sample_theta`: `rng = np.random.default_rng(seed)`; family を一様選択; `h0 = exp(U(log 3e-4, log 3e-3))`; `u_in = exp(U(log 0.1, log 2))`; ポートは `_sample_port(rng)`（壁一様、長さ U(0.05, 0.15)、位置一様）、同壁重なりは最大 20 回再抽選; family 固有 params を `_sample_params(family, rng)`; uturn は inlet/outlet を west に強制。閉塞系は `build_h` → `connected` で判定し失敗なら `seed` 固定のまま rng を進めて再抽選（最大 50 回、超えたら RuntimeError）
  - `build_h` は family ごとの純関数 `_h_uniform, _h_sin2d, _h_quad, _h_grf, _h_uturn, _h_serpentine, _h_pins, _h_blobs`（spec §3 の式）。閉塞系は最後に `_open_ports(h, theta)` でポート帯 1 セル + 奥行き 2 セルを h0 にする
  - `_h_grf`: `rng.standard_normal((NX, NY))` を `scipy.ndimage.gaussian_filter(sigma=corr/dx)` → std で割る → `h0·exp(σ G)`
  - `_h_serpentine`: 往復数 n、流路高さ hc、間隔で n 本の水平帯 + 端で交互に連結する縦帯、inlet 帯から最初の水平帯まで縦帯で接続
  - `_h_pins`: 六方/正方格子の円板で閉塞、`_h_blobs`: `_h_grf` の G を分位点で閾値化（開口率 r）
  - `connected`: `open = h > h_blocked * 1.5`; `scipy.ndimage.label(open)`; inlet セルのラベル集合 ∩ outlet セルのラベル集合 が非空
  - `build_bc`: `port_mask(port)` が `west_span/east_span(…, LX)/south_span/north_span(…, LY)` を返す; `BC(patches=(BC.velocity_inlet(mask_in, u_in), BC.pressure_outlet(mask_out)))`
- [ ] **Step 4: パス確認** `pytest tests/test_nsbm_families.py -q`
- [ ] **Step 5: ruff + commit** `git add nsbm tests/test_nsbm_families.py && git commit -m "nsbm: families — θ のサンプルと 8 ファミリの h 場・BC 構築（連結性チェック）"`

---

### Task 2: features — 入力画像と正規化

**Files:** Create `nsbm/features.py`, Test `tests/test_nsbm_features.py`

**Interfaces:**
```python
IN_CH = 8
def make_x(theta: Theta, h: np.ndarray) -> np.ndarray             # (8, 72, 48) float32
def p_ref(theta: Theta) -> float                                     # 12 mu u_in LX / h0^2 + rho u_in^2
def normalize_y(theta, u, v, p) -> np.ndarray                        # (3, 72, 48) float32
def denormalize_y(theta, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]   # float64 (72,48)×3
```

- [ ] **Step 1: テスト**

```python
def test_roundtrip():
    th = sample_theta(5); h = build_h(th)
    u, v, p = rng.normal(size=(3, 72, 48))
    y = normalize_y(th, u, v, p); u2, v2, p2 = denormalize_y(th, y)
    assert np.allclose(u, u2, atol=1e-5) and np.allclose(p, p2, rtol=1e-5, atol=1e-5 * p_ref(th))

def test_make_x_channels():
    x = make_x(th, h)
    assert x.shape == (8, 72, 48) and x.dtype == np.float32
    assert np.allclose(x[0], np.log(h / th.h0))
    i, j = port_cells(th.inlet); assert np.hypot(x[3], x[4])[i, j].max() == pytest.approx(th.u_in)
    assert x[5].sum() == len(port_cells(th.outlet)[0])
```

- [ ] **Step 2〜5:** 失敗確認 → 実装（inlet 法線: west (+1,0)、east (−1,0)、south (0,+1)、north (0,−1)）→ パス → commit

---

### Task 3: dataset — 1 件生成と並列生成・シャード I/O

**Files:** Create `nsbm/dataset.py`, `experiments/nsbm/gen.py`, Test `tests/test_nsbm_dataset.py`; `.gitignore` に `experiments/nsbm/data/`, `experiments/nsbm/runs/`

**Interfaces:**
```python
@dataclass
class Sample: theta: Theta; x: np.ndarray; y: np.ndarray; u: np.ndarray; v: np.ndarray; p: np.ndarray; n_iter: int; converged: bool; n_gmres_total: int; residual_ref: float; elapsed: float
def solve_sample(seed: int) -> Sample
def generate(seeds: Sequence[int], out_dir: Path, n_workers: int, shard_size: int = 256, log=print) -> list[Path]
def save_shard(path, samples) / def load_shards(dir) -> list[Sample]
```

- テスト: `solve_sample(1)` が `Sample` を返し `x.shape == (8,72,48)`, `y.shape == (3,72,48)`; `save_shard`→`load_shards` 往復で theta/配列一致（tmp_path）
- `generate`: `multiprocessing.get_context("spawn").Pool(n_workers, initializer=_init_worker)`; `_init_worker` が `threading` 系環境変数を設定（保険。主は gen.py の先頭）し `numba.set_num_threads(1)`; `imap_unordered(solve_sample, seeds, chunksize=1)`; 進捗ログ（件数・family・n_iter・converged・elapsed）; 256 件ごとに `save_shard`
- `experiments/nsbm/gen.py`: 先頭で env 設定 → `argparse(--n 2000 --workers 16 --out experiments/nsbm/data --seed0 0)`
- commit

---

### Task 4: model — UNet

**Files:** Create `nsbm/model.py`, Test `tests/test_nsbm_model.py`（`pytest.importorskip("torch")`）

```python
class UNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=3, widths=(32, 64, 128, 256)): ...
    def forward(self, x: Tensor) -> Tensor   # (B, 8, 72, 48) → (B, 3, 72, 48)
```
- Block = Conv3×3 → GroupNorm(8) → GELU ×2；down は MaxPool2；up は bilinear upsample + skip concat
- テスト: 形状、`loss.backward()` が通る、パラメータ数 1M〜5M
- commit

---

### Task 5: train — 学習ループ

**Files:** Create `nsbm/train.py`, `experiments/nsbm/train.py`

```python
def split_by_family(samples: list[Sample], seed=0, frac=(0.8, 0.1, 0.1)) -> dict[str, list[int]]   # family 層化、converged のみ
def to_tensors(samples, idx) -> tuple[Tensor, Tensor]
def train(samples, out_dir, epochs=200, batch=32, lr=1e-3, seed=0, threads=None, log=print) -> Path  # best.pt を返す
```
- AdamW(weight_decay 1e-4)、CosineAnnealingLR、MSE、各 epoch の train/val loss を `history.csv`、val 最良で `best.pt` 保存（`state_dict` + widths + split の seed 番号）
- テスト: `test_nsbm_model.py` に「合成 8 件で 2 epoch 回り best.pt ができる」を追加
- commit

---

### Task 6: evaluate — 3 方式の Newton 反復数比較

**Files:** Create `nsbm/evaluate.py`, `experiments/nsbm/eval.py`

```python
def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, k=4) -> np.ndarray   # 距離逆数重み
def run_with_init(theta, init: tuple | None) -> dict   # n_iter, converged, r0_ratio, cfl0, elapsed
def evaluate(model, samples, split, k=4, log=print) -> list[dict]   # 各テスト θ について 3 方式の結果
def summarize(rows) -> dict   # 全体/ファミリ別の中央値・四分位・勝敗
```
- Stokes 発進は再計算せず `Sample.n_iter` を使う（同じ設定・同じ Stokes 発進）。ただし `r0_ratio` は `steady_residual_history[0]/residual_ref` が Sample に無いので、評価時に Stokes も再走行して揃える（1 件数秒、テスト 200 件 × 3 方式を並列化: dataset の Pool を再利用し、モデル推論は親で済ませてから初期場を渡す）
- 出力 `experiments/nsbm/results/eval-<run>.yaml` と `rows.csv`
- commit

---

### Task 7: 実走（生成 → 学習 → 評価）と報告

- [ ] `nohup ~/.claude/hooks/memcap -- python experiments/nsbm/gen.py --n 2000 --workers 16 2>&1 | tee experiments/nsbm/logs/gen-$(date +%s).log`
- [ ] 学習 `python experiments/nsbm/train.py --data experiments/nsbm/data --out experiments/nsbm/runs/unet-a 2>&1 | tee ...`
- [ ] 評価 `python experiments/nsbm/eval.py --run experiments/nsbm/runs/unet-a 2>&1 | tee ...`
- [ ] 報告 `docs/status/status-43.md`（全体像 → メカニズム → 表 → 図）、`nsbm/README.md`、README/status-index/roadmap 更新、mdview → Artifact
- [ ] commit / push
