#!/usr/bin/env bash
# ゲート G3 一括実行: 1D 較正 → G3a（ニュートン） → G3b（べき乗則） → 比較 → レポート
#
#   cd ~/work/ykep-cae
#   OMP_NUM_THREADS=2 experiments/extruder/run_g3.sh [work_dir]
#
# OpenFOAM は ~/work/1a/a02/tools/of（Docker、OF_CPUS=1 / OF_MEM=1200m 既定）。
# G3b は G3a の収束解から始める（べき乗則を U=0 から始めると nuMax クランプで
# 擬似時間刻みが潰れ、収束に数万反復かかる。1D で実測）。
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-/tmp/of-g3}
PY="$REPO/.venv/bin/python"
OF=~/work/1a/a02/tools/of
export PYTHONPATH="$REPO" OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$WORK" "$REPO/logs"
LOG="$REPO/logs/g3-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "== branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
echo "== image ${OF_IMG:-opencfd/openfoam-run:2312}  work $WORK"

echo "== [1/4] 1D powerLaw calibration"
rm -rf "$WORK/pl1d"
"$PY" "$REPO/experiments/extruder/of_powerlaw_check.py" --out "$WORK/pl1d"

echo "== [2/4] G3a newtonian"
rm -rf "$WORK/g3a"
"$PY" "$REPO/experiments/extruder/of_case.py" --model newtonian --out "$WORK/g3a"
( cd "$WORK/g3a"
  $OF blockMesh > log.blockMesh 2>&1
  $OF topoSet > log.topoSet 2>&1
  $OF subsetMesh c0 -patch screw -overwrite > log.subsetMesh 2>&1
  $OF checkMesh > log.checkMesh 2>&1
  time $OF simpleFoam > log.simpleFoam 2>&1
  $OF postProcess -func writeCellCentres -latestTime > log.post 2>&1 )
"$PY" "$REPO/experiments/extruder/compare_openfoam.py" --case "$WORK/g3a" --model newtonian

echo "== [3/4] G3b powerLaw (K=2e4, n=0.4), initialised from G3a"
rm -rf "$WORK/g3b"
"$PY" "$REPO/experiments/extruder/of_case.py" --model powerlaw --K 2e4 --n 0.4 --out "$WORK/g3b"
LATEST=$(ls -d "$WORK"/g3a/[0-9]* | sort -t/ -k1 -n | awk -F/ '{print $NF}' | sort -n | tail -1)
cp -r "$WORK/g3a/constant/polyMesh" "$WORK/g3b/constant/"
cp "$WORK/g3a/$LATEST/U" "$WORK/g3a/$LATEST/p" "$WORK/g3a/$LATEST/phi" "$WORK/g3b/0/"
( cd "$WORK/g3b"
  time $OF simpleFoam > log.simpleFoam 2>&1
  $OF postProcess -func writeCellCentres -latestTime > log.post 2>&1 )
"$PY" "$REPO/experiments/extruder/compare_openfoam.py" --case "$WORK/g3b" --model powerlaw --K 2e4 --n 0.4

echo "== [4/4] report"
"$PY" "$REPO/experiments/extruder/g3_report.py" --work "$WORK" --out "$REPO/docs/reports/extruder/g3-openfoam.md"
echo "== done. log: $LOG"
