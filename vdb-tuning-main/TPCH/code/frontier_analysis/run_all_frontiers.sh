#!/bin/bash
# Master script: Run ALL 3 frontiers overnight (IVFFlat, HNSW, Parallel IVFFlat)
# Uses pre-built PGDATA snapshots instead of building indexes from scratch.
#
# Usage: sudo nohup ./run_all_frontiers.sh 10:15 > /tmp/all_frontiers.log 2>&1 &
#
# This script:
# 1. Loads IVFFlat snapshot, compiles default pgvector, runs full frontier
# 2. Loads HNSW snapshot, runs full frontier (reuses OLAP saturation)
# 3. Loads Parallel IVFFlat snapshot, compiles parallel pgvector, runs full frontier (reuses OLAP saturation)

set -e

if [ -z "$1" ]; then
    echo "Usage: sudo nohup ./run_all_frontiers.sh <SB:CG> > /tmp/all_frontiers.log 2>&1 &"
    echo "  e.g.: sudo ./run_all_frontiers.sh 10:15"
    exit 1
fi

MEMORY="$1"
SB=$(echo $MEMORY | cut -d: -f1)
CG=$(echo $MEMORY | cut -d: -f2)

# Check for --extra parameter
EXTRA_FLAG=""
if [ "$2" = "--extra" ] && [ -n "$3" ]; then
    EXTRA_FLAG="--extra $3"
    echo "Extra points specified: $3"
    echo "DEBUG: EXTRA_FLAG = "${EXTRA_FLAG}""
fi

# Auto-detect paths from config.yaml using yq or python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_YAML="$(cd "$SCRIPT_DIR/../../.." && pwd)/config.yaml"

if command -v yq &> /dev/null; then
    PG_SRC=$(yq '.paths.pg_src' "$CONFIG_YAML")
    PG_INSTALL=$(yq '.paths.pg_install' "$CONFIG_YAML")
    PGDATA_DIR=$(yq '.paths.pg_data' "$CONFIG_YAML")
    SNAPSHOTS_DIR=$(yq '.paths.snapshot_dir' "$CONFIG_YAML")
    PGVECTOR_DEFAULT=$(yq '.paths.pgvector_default' "$CONFIG_YAML")
    PGVECTOR_PARALLEL=$(yq '.paths.pgvector_parallel' "$CONFIG_YAML")
else
    # Fallback: use python to parse yaml
    eval $(python3 -c "
import yaml
with open('$CONFIG_YAML') as f:
    c = yaml.safe_load(f)['paths']
print(f\"PG_SRC={c['pg_src']}\")
print(f\"PG_INSTALL={c['pg_install']}\")
print(f\"PGDATA_DIR={c['pg_data']}\")
print(f\"SNAPSHOTS_DIR={c['snapshot_dir']}\")
print(f\"PGVECTOR_DEFAULT={c['pgvector_default']}\")
print(f\"PGVECTOR_PARALLEL={c['pgvector_parallel']}\")
")
fi

export PATH=${PG_INSTALL}/bin:$PATH

# Read NUMA config from experiment_config.json
PG_NUMA_NODES=$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/experiment_config.json'))['hardware']['postgres_numa_nodes'])")

# Snapshot files for GIST dataset
SNAPSHOT_IVFFLAT=${SNAPSHOTS_DIR}/gist_ivfflat_lists8192_pgdata.tar.gz
SNAPSHOT_HNSW=${SNAPSHOTS_DIR}/gist_hnsw_efc256_m32_pgdata.tar.gz
SNAPSHOT_PARALLEL=${SNAPSHOTS_DIR}/gist_parallel_ivfflat_lists8192_pgdata.tar.gz

# Verify snapshots exist
for snap in "$SNAPSHOT_IVFFLAT" "$SNAPSHOT_HNSW" "$SNAPSHOT_PARALLEL"; do
    if [ ! -f "$snap" ]; then
        echo "ERROR: Snapshot not found: $snap"
        exit 1
    fi
done

echo "========================================================================"
echo "MASTER FRONTIER RUNNER - All 3 indexes (using snapshots)"
echo "Memory config: shared_buffers=${SB}GB, cgroup=${CG}GB"
echo "Snapshots:"
echo "  IVFFlat:  $SNAPSHOT_IVFFLAT"
echo "  HNSW:     $SNAPSHOT_HNSW"
echo "  Parallel: $SNAPSHOT_PARALLEL"
echo "Started at: $(date)"
echo "========================================================================"

# ============================================================
# Function: Load a snapshot into PGDATA
# ============================================================
load_snapshot() {
    local SNAPSHOT_PATH="$1"
    local LABEL="$2"

    # Stop postgres if running
    su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && pg_ctl -D ${PGDATA_DIR} stop -m fast" 2>/dev/null || true
    sleep 2

    # Clear existing PGDATA contents (keep the directory, same as recall_qps_diagram.py)
    echo "    Clearing old PGDATA contents..."
    rm -rf ${PGDATA_DIR}/*

    # Extract snapshot INTO pgdata directory (tar contents are pgdata files, no wrapper dir)
    echo "    Extracting snapshot (this may take a few minutes)..."
    mkdir -p ${PGDATA_DIR}
    # Use pigz if available for faster decompression
    if command -v pigz &> /dev/null; then
        pigz -dc "${SNAPSHOT_PATH}" | tar -xf - -C ${PGDATA_DIR}
    else
        tar -xzf "${SNAPSHOT_PATH}" -C ${PGDATA_DIR}
    fi

    # Fix ownership and permissions (same as recall_qps_diagram.py)
    chown -R postgres:postgres ${PGDATA_DIR}
    chmod 700 ${PGDATA_DIR}

    echo "    Snapshot loaded successfully."
}

# ============================================================
# Function: Start postgres with NUMA pinning
# ============================================================
start_postgres() {
    echo "    Starting PostgreSQL (NUMA nodes: ${PG_NUMA_NODES})..."
    su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && numactl --cpunodebind=${PG_NUMA_NODES} --membind=${PG_NUMA_NODES} pg_ctl -D ${PGDATA_DIR} -l /tmp/logfile start"
    sleep 3

    # Wait for postgres to be ready
    for i in $(seq 1 30); do
        if su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && pg_isready" > /dev/null 2>&1; then
            echo "    PostgreSQL is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: PostgreSQL did not start in time!"
    exit 1
}

# ########################################################################
# # PHASE 1: IVFFlat
# ########################################################################
# echo ""
# echo "========================================================================"
# echo "PHASE 1/3: IVFFlat"
# echo "========================================================================"

# #Compile default pgvector
# echo "[1a] Compiling default pgvector..."
# cd ${PGVECTOR_DEFAULT}
# make clean > /dev/null 2>&1 || true
# make -j$(nproc) > /dev/null 2>&1
# make install > /dev/null 2>&1
# echo "  Done."

# # Load IVFFlat snapshot
# echo "[1b] Loading IVFFlat snapshot..."
# load_snapshot "$SNAPSHOT_IVFFLAT" "IVFFlat (lists=8192)"
# start_postgres
# #create_tpch_indexes

# # Run IVFFlat frontier
# echo "[1c] Running IVFFlat frontier..."
# cd ${SCRIPT_DIR}/frontier_analysis
# ./run_frontier.sh ${MEMORY} --warm --index ivfflat --ivfflat-lists 8192 --snapshot ${EXTRA_FLAG}

# echo ""
# echo "PHASE 1 COMPLETE: IVFFlat frontier done at $(date)"

# # ########################################################################
# # # PHASE 2: HNSW (reuse OLAP saturation from phase 1)
# # ########################################################################
# echo ""
# echo "========================================================================"
# echo "PHASE 2/3: HNSW"
# echo "========================================================================"

# # pgvector default is already compiled (supports HNSW)
# # Load HNSW snapshot
# echo "[2a] Loading HNSW snapshot..."
# load_snapshot "$SNAPSHOT_HNSW" "HNSW (m=32, efc=256)"
# start_postgres
# #create_tpch_indexes

# # Run HNSW frontier
# echo "[2b] Running HNSW frontier..."
# cd ${SCRIPT_DIR}/frontier_analysis
# ./run_frontier.sh ${MEMORY} --warm --index hnsw --hnsw-m 32 --hnsw-efc 256 --snapshot ${EXTRA_FLAG}

# # echo ""
# # echo "PHASE 2 COMPLETE: HNSW frontier done at $(date)"

########################################################################
# PHASE 3: Parallel IVFFlat
#######################################################################
echo ""
echo "========================================================================"
echo "PHASE 3/3: Parallel IVFFlat"
echo "========================================================================"

# # Compile parallel pgvector
echo "[3a] Compiling parallel pgvector..."
cd ${PGVECTOR_PARALLEL}
make clean > /dev/null 2>&1 || true
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1
echo "  Done."

# Load Parallel IVFFlat snapshot
echo "[3b] Loading Parallel IVFFlat snapshot..."
load_snapshot "$SNAPSHOT_PARALLEL" "Parallel IVFFlat (lists=8192)"
start_postgres
#create_tpch_indexes

# Run parallel IVFFlat frontier
echo "[3c] Running Parallel IVFFlat frontier..."
cd ${SCRIPT_DIR}/frontier_analysis
./run_frontier.sh ${MEMORY} --warm --index parallel_ivfflat --ivfflat-lists 8192 --snapshot ${EXTRA_FLAG}

echo ""
echo "PHASE 3 COMPLETE: Parallel IVFFlat frontier done at $(date)"

# ########################################################################
# ALL DONE
########################################################################
echo ""
echo "========================================================================"
echo "ALL 3 FRONTIERS COMPLETE!"
echo "Finished at: $(date)"
echo "Results in:"
echo "  IVFFlat:          results_frontier_warm_constrained_${SB}GB_sb_${CG}GB_cg/"
echo "  HNSW:             results_frontier_hnsw_warm_constrained_${SB}GB_sb_${CG}GB_cg/"
echo "  Parallel IVFFlat: results_frontier_pivfflat_warm_constrained_${SB}GB_sb_${CG}GB_cg/"
echo "========================================================================"
