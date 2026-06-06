#!/usr/bin/env bash
# Recall-vs-QPS: measure (warm+cold, all index types) per dataset, then plot.
# Assumes this script sits in the same dir as recall_qps_diagram.py / plot_recall_qps.py.
# Live output to terminal AND .log, even under sudo.
set -uo pipefail          # NOT -e: one failed (dataset,mode) shouldn't kill the rest

# Resolve the directory THIS script lives in (= where the .py files are), then cd into it.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

DATASETS=("gist" "deep")
MODES=("warm" "cold")
LOG_DIR="$DIR/logs"
RESULTS_DIR="$DIR/results_recall_qps"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

fail=0
failed=()

# === 1. MEASURE (needs root: pg restart, drop_caches, compile) ===
for ds in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    log="$LOG_DIR/measure_${ds}_${mode}_${TS}.log"
    echo "=== MEASURE  dataset=$ds  mode=$mode  ->  $log"
    if sudo stdbuf -oL -eL python3 -u recall_qps_diagram.py \
         --dataset "$ds" --mode "$mode" --index all 2>&1 | stdbuf -oL tee "$log"; then
      echo "    OK: measure $ds $mode"
    else
      echo "    FAILED: measure $ds $mode (see $log)"
      fail=1
      failed+=("measure:$ds:$mode")
    fi
  done
done

# === 2. PLOT (no sudo: just reads JSON, writes figures/) ===
for ds in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    log="$LOG_DIR/plot_${ds}_${mode}_${TS}.log"
    echo "=== PLOT     dataset=$ds  mode=$mode  ->  $log"
    if stdbuf -oL -eL python3 -u plot_recall_qps.py \
         --dataset "$ds" --mode "$mode" --results-dir "$RESULTS_DIR" 2>&1 | stdbuf -oL tee "$log"; then
      echo "    OK: plot $ds $mode"
    else
      echo "    FAILED: plot $ds $mode (see $log)"
      fail=1
      failed+=("plot:$ds:$mode")
    fi
  done
done

echo ""
echo "========================================"
if [ "$fail" -eq 0 ]; then
  echo "ALL DONE — no failures"
else
  echo "DONE WITH FAILURES:"
  printf '  - %s\n' "${failed[@]}"
fi
echo "Logs:    $LOG_DIR"
echo "Results: $RESULTS_DIR"
echo "Figures: $DIR/figures"
echo "========================================"
