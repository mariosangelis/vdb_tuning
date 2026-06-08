#!/bin/bash
# Run throughput frontier experiments
# Usage: sudo ./run_frontier.sh 20:50 [--warm] [--dense] [--index ivfflat|hnsw|parallel_ivfflat] [--hnsw-m M] [--hnsw-efc EFC] [--snapshot]
#
# Examples:
#   sudo ./run_frontier.sh 20:50 --warm --index ivfflat --ivfflat-lists 8192 --snapshot
#   sudo ./run_frontier.sh 20:50 --warm --index hnsw --hnsw-m 16 --hnsw-efc 128 --snapshot
#   sudo ./run_frontier.sh 20:50 --warm --index parallel_ivfflat --ivfflat-lists 8192 --snapshot

set -e

if [ -z "$1" ]; then
    echo "Usage: sudo ./run_frontier.sh <SB:CG> [FLAGS]"
    echo "  e.g.: sudo ./run_frontier.sh 20:50 --warm --snapshot"
    echo "        sudo ./run_frontier.sh 20:50 --warm --index hnsw --hnsw-m 16 --hnsw-efc 128 --snapshot"
    echo "        sudo ./run_frontier.sh 20:50 --warm --index parallel_ivfflat --snapshot"
    echo ""
    echo "Flags:"
    echo "  --warm              Prewarm DB once, no restart between measurements"
    echo "  --dense             Use dense grid (14 fractions instead of 8)"
    echo "  --index TYPE        Index type: ivfflat (default), hnsw, parallel_ivfflat"
    echo "  --hnsw-m M          HNSW M parameter (default: from tune)"
    echo "  --hnsw-efc EFC      HNSW ef_construction (default: from tune)"
    echo "  --ivfflat-lists N   IVFFlat lists parameter (default: from tune)"
    echo "  --snapshot          Skip ensure_index (index already loaded from snapshot)"
    exit 1
fi

MEMORY="$1"
SB=$(echo $MEMORY | cut -d: -f1)
CG=$(echo $MEMORY | cut -d: -f2)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p $LOG_DIR

# Extract index name for log file naming
INDEX_NAME="ivfflat"
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "--index" ]; then
        INDEX_NAME="$arg"
    fi
    prev_arg="$arg"
done

LOGFILE="${LOG_DIR}/frontier_${INDEX_NAME}_${SB}GB_sb_${CG}GB_cg.log"
EXTRA_FLAGS="${@:2}"

echo "========================================"
echo "Running frontier: ${INDEX_NAME}, shared_buffers=${SB}GB, cgroup=${CG}GB"
echo "Flags: ${EXTRA_FLAGS}"
echo "Log file: ${LOGFILE}"
echo "========================================"

FRONTIER_DIR="${SCRIPT_DIR}/frontier_analysis"

echo "[Step 1/2] find_saturation_points.py --memory ${MEMORY} ${EXTRA_FLAGS}" | tee $LOGFILE
python3 -u ${FRONTIER_DIR}/find_saturation_points.py --memory ${MEMORY} ${EXTRA_FLAGS} 2>&1 | tee -a $LOGFILE

echo "" | tee -a $LOGFILE
echo "[Step 2/2] build_throughput_frontier.py --memory ${MEMORY} ${EXTRA_FLAGS}" | tee -a $LOGFILE
python3 -u ${FRONTIER_DIR}/build_throughput_frontier.py --memory ${MEMORY} ${EXTRA_FLAGS} 2>&1 | tee -a $LOGFILE

echo "" | tee -a $LOGFILE
echo "DONE! Index: ${INDEX_NAME}, Memory: ${SB}GB/${CG}GB" | tee -a $LOGFILE
