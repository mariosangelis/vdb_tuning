#!/bin/bash
set -e

# Usage: sudo bash tune_and_compute_recall_plots.sh <dataset>
# Example: sudo bash tune_and_compute_recall_plots.sh gist
#          sudo bash tune_and_compute_recall_plots.sh deep

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    echo "  Available datasets: gist, deep, sift"
    exit 1
fi

DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== ${DATASET^^} DATASET: TUNE AND COMPUTE RECALL/QPS ==="
echo "Started: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Step 1: Compute exact results (if not already cached)
echo ''
echo '=========================================='
echo "Step 1: Computing exact results for ${DATASET}"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/exact_results/compute_exact_results.py" --dataset "$DATASET"

# Step 2: Tune IVFFlat
echo ''
echo '=========================================='
echo "Step 2: Tuning IVFFlat for ${DATASET}"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/tuning/tune_ivfflat.py" --dataset "$DATASET"

# Step 3: Tune HNSW
echo ''
echo '=========================================='
echo "Step 3: Tuning HNSW for ${DATASET}"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/tuning/tune_hnsw.py" --dataset "$DATASET" --recall 0.9

# Step 4: Build Parallel IVFFlat index and save snapshot
echo ''
echo '=========================================='
echo "Step 4: Building Parallel IVFFlat for ${DATASET}"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/tuning/tune_parallel_ivfflat.py" --dataset "$DATASET"

# Step 5: Recall vs QPS - IVFFlat warm
echo ''
echo '=========================================='
echo "Step 5: Recall vs QPS - IVFFlat (warm)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode warm --index ivfflat

# Step 6: Recall vs QPS - HNSW warm
echo ''
echo '=========================================='
echo "Step 6: Recall vs QPS - HNSW (warm)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode warm --index hnsw

# Step 7: Recall vs QPS - Parallel IVFFlat warm
echo ''
echo '=========================================='
echo "Step 7: Recall vs QPS - Parallel IVFFlat (warm)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode warm --index parallel_ivfflat

# Step 8: Cold measurements - IVFFlat
echo ''
echo '=========================================='
echo "Step 8: Recall vs QPS - IVFFlat (cold)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode cold --index ivfflat

# Step 9: Cold measurements - HNSW
echo ''
echo '=========================================='
echo "Step 9: Recall vs QPS - HNSW (cold)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode cold --index hnsw

# Step 10: Cold measurements - Parallel IVFFlat
echo ''
echo '=========================================='
echo "Step 10: Recall vs QPS - Parallel IVFFlat (cold)"
echo '=========================================='
sudo python3 "$SCRIPT_DIR/recall_qps_plots/recall_qps_diagram.py" --dataset "$DATASET" --mode cold --index parallel_ivfflat

# Step 11: Generate plots
echo ''
echo '=========================================='
echo "Step 11: Generating plots"
echo '=========================================='
python3 "$SCRIPT_DIR/recall_qps_plots/plot_recall_qps.py" --dataset "$DATASET" --mode warm
python3 "$SCRIPT_DIR/recall_qps_plots/plot_recall_qps.py" --dataset "$DATASET" --mode cold

echo ''
echo '=========================================='
echo 'ALL DONE!'
echo "Finished: $(date)"
echo '=========================================='
