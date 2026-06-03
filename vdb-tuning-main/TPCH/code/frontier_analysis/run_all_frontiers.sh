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

SCRIPT_DIR=/users/mangelis/vdb_tuning/vdb-tuning-main/TPCH/code
PG_DIR=/users/mangelis/vdb_tuning/postgres/postgres
PGVECTOR_DEFAULT=${PG_DIR}/pgvector
PGVECTOR_PARALLEL=${PG_DIR}/pgvector_modified_final
SNAPSHOTS_DIR=/mnt/nvme/saved_indexes
PGDATA_DIR=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH

# Snapshot files for GIST dataset
SNAPSHOT_IVFFLAT=${SNAPSHOTS_DIR}/gist_ivfflat_lists8192_pgdata.tar.gz
SNAPSHOT_HNSW=${SNAPSHOTS_DIR}/gist_hnsw_efc128_m16_pgdata.tar.gz
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

    echo "  Loading snapshot: ${LABEL}"
    echo "    Source: ${SNAPSHOT_PATH}"

    # Stop postgres if running
    su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && pg_ctl -D ${PGDATA_DIR} stop -m fast" 2>/dev/null || true
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
# Function: Create TPC-H secondary indexes (benchbase-style)
# These are missing from the snapshots (only PKs were created by dbgen)
# ============================================================
create_tpch_indexes() {
    echo "    Creating TPC-H secondary indexes (if not exist)..."
    su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && psql -h /tmp -d mydb" <<'EOF'
CREATE INDEX IF NOT EXISTS l_ok ON lineitem (l_orderkey ASC);
CREATE INDEX IF NOT EXISTS l_pk ON lineitem (l_partkey ASC);
CREATE INDEX IF NOT EXISTS l_sk ON lineitem (l_suppkey ASC);
CREATE INDEX IF NOT EXISTS l_sd ON lineitem (l_shipdate ASC);
CREATE INDEX IF NOT EXISTS l_cd ON lineitem (l_commitdate ASC);
CREATE INDEX IF NOT EXISTS l_rd ON lineitem (l_receiptdate ASC);
CREATE INDEX IF NOT EXISTS l_pk_sk ON lineitem (l_partkey, l_suppkey);
CREATE INDEX IF NOT EXISTS l_sk_pk ON lineitem (l_suppkey, l_partkey);
CREATE INDEX IF NOT EXISTS ps_pk ON partsupp (ps_partkey ASC);
CREATE INDEX IF NOT EXISTS ps_sk ON partsupp (ps_suppkey ASC);
CREATE INDEX IF NOT EXISTS o_ck ON orders (o_custkey ASC);
CREATE INDEX IF NOT EXISTS o_od ON orders (o_orderdate ASC);
CREATE INDEX IF NOT EXISTS c_nk ON customer (c_nationkey ASC);
CREATE INDEX IF NOT EXISTS s_nk ON supplier (s_nationkey ASC);
CREATE INDEX IF NOT EXISTS n_rk ON nation (n_regionkey ASC);
ANALYZE;
EOF
    echo "    TPC-H indexes created and ANALYZE complete."
}

# ============================================================
# Function: Start postgres with NUMA pinning
# ============================================================
start_postgres() {
    echo "    Starting PostgreSQL..."
    su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && numactl --cpunodebind=0 --membind=0 pg_ctl -D ${PGDATA_DIR} -l /tmp/logfile start"
    sleep 3

    # Wait for postgres to be ready
    for i in $(seq 1 30); do
        if su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && pg_isready" > /dev/null 2>&1; then
            echo "    PostgreSQL is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: PostgreSQL did not start in time!"
    exit 1
}

########################################################################
# PHASE 1: IVFFlat
########################################################################
echo ""
echo "========================================================================"
echo "PHASE 1/3: IVFFlat"
echo "========================================================================"

# Compile default pgvector
echo "[1a] Compiling default pgvector..."
cd ${PGVECTOR_DEFAULT}
make clean > /dev/null 2>&1 || true
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1
echo "  Done."

# Load IVFFlat snapshot
echo "[1b] Loading IVFFlat snapshot..."
load_snapshot "$SNAPSHOT_IVFFLAT" "IVFFlat (lists=8192)"
start_postgres
create_tpch_indexes

# Run IVFFlat frontier
echo "[1c] Running IVFFlat frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index ivfflat --ivfflat-lists 8192 --snapshot

echo ""
echo "PHASE 1 COMPLETE: IVFFlat frontier done at $(date)"

########################################################################
# PHASE 2: HNSW (reuse OLAP saturation from phase 1)
########################################################################
echo ""
echo "========================================================================"
echo "PHASE 2/3: HNSW"
echo "========================================================================"

# pgvector default is already compiled (supports HNSW)
# Load HNSW snapshot
echo "[2a] Loading HNSW snapshot..."
load_snapshot "$SNAPSHOT_HNSW" "HNSW (m=16, efc=128)"
start_postgres
create_tpch_indexes

# Run HNSW frontier
echo "[2b] Running HNSW frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index hnsw --hnsw-m 16 --hnsw-efc 128 --snapshot

echo ""
echo "PHASE 2 COMPLETE: HNSW frontier done at $(date)"

########################################################################
# PHASE 3: Parallel IVFFlat
########################################################################
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
create_tpch_indexes

# Run parallel IVFFlat frontier
echo "[3c] Running Parallel IVFFlat frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index parallel_ivfflat --ivfflat-lists 8192 --snapshot

echo ""
echo "PHASE 3 COMPLETE: Parallel IVFFlat frontier done at $(date)"

########################################################################
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
