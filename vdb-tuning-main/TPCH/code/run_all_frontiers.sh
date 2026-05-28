#!/bin/bash
# Master script: Run ALL 3 frontiers overnight (IVFFlat, HNSW, Parallel IVFFlat)
# Usage: sudo nohup ./run_all_frontiers.sh 10:15 > /tmp/all_frontiers.log 2>&1 &
#
# This script:
# 1. Compiles default pgvector, builds IVFFlat index, runs full frontier
# 2. Drops IVFFlat, builds HNSW index, runs full frontier (reuses OLAP saturation)
# 3. Compiles parallel pgvector, drops HNSW, builds parallel IVFFlat, runs full frontier (reuses OLAP saturation)

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
export PATH=/usr/local/pgsql/bin:$PATH

echo "========================================================================"
echo "MASTER FRONTIER RUNNER - All 3 indexes"
echo "Memory config: shared_buffers=${SB}GB, cgroup=${CG}GB"
echo "Started at: $(date)"
echo "========================================================================"

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

# Restart postgres to pick up new extension
su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && pg_ctl -D /mnt/nvme/pgdata restart -l /tmp/logfile" > /dev/null 2>&1
sleep 3

# Run IVFFlat frontier (includes OLAP saturation + ANN saturation + grid + frontier)
echo "[1b] Running IVFFlat frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index ivfflat --ivfflat-lists 8192

echo ""
echo "PHASE 1 COMPLETE: IVFFlat frontier done at $(date)"

########################################################################
# PHASE 2: HNSW (reuse OLAP saturation from phase 1)
########################################################################
echo ""
echo "========================================================================"
echo "PHASE 2/3: HNSW"
echo "========================================================================"

# pgvector is already the default (supports both IVFFlat and HNSW)
# Just run with --index hnsw; the script handles dropping old index + building new one
echo "[2a] Running HNSW frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index hnsw --hnsw-m 16 --hnsw-efc 128

echo ""
echo "PHASE 2 COMPLETE: HNSW frontier done at $(date)"

########################################################################
# PHASE 3: Parallel IVFFlat
########################################################################
echo ""
echo "========================================================================"
echo "PHASE 3/3: Parallel IVFFlat"
echo "========================================================================"

# Compile parallel pgvector
echo "[3a] Compiling parallel pgvector..."
cd ${PGVECTOR_PARALLEL}
make clean > /dev/null 2>&1 || true
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1
echo "  Done."

# Restart postgres to pick up parallel extension
su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && pg_ctl -D /mnt/nvme/pgdata restart -l /tmp/logfile" > /dev/null 2>&1
sleep 3

# Run parallel IVFFlat frontier
echo "[3b] Running Parallel IVFFlat frontier..."
cd ${SCRIPT_DIR}
./run_frontier.sh ${MEMORY} --warm --index parallel_ivfflat --ivfflat-lists 8192

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
