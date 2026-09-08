#!/usr/bin/env bash
# 空きコアに合わせてワーカー数を決めながら、seed をチャンクごとに生成する（他セッションのジョブと共存するため）.
#   bash experiments/nsbm/gen_chunks.sh <n_total> <chunk> [out_dir]
set -u
N=${1:-4000}; CHUNK=${2:-250}; OUT=${3:-experiments/nsbm/data}
NCPU=$(nproc)
for ((s0 = 0; s0 < N; s0 += CHUNK)); do
  # 自分以外のプロセスの CPU 使用（コア数換算）を ps で見積もり、残りをワーカーに充てる（4〜16）
  OTHER=$(ps -eo pcpu,args --no-headers | grep -v "nsbm/gen.py" | awk '{s+=$1} END {printf "%d", (s+50)/100}')
  W=$(( NCPU - OTHER )); [ "$W" -lt 4 ] && W=4; [ "$W" -gt 16 ] && W=16
  echo "=== chunk seed0=$s0 n=$CHUNK workers=$W (other load ~${OTHER} cores) $(date +%H:%M:%S)"
  python experiments/nsbm/gen.py --n "$CHUNK" --seed0 "$s0" --workers "$W" --out "$OUT" --shard "$CHUNK"
done
echo "=== all done $(date +%H:%M:%S)"
