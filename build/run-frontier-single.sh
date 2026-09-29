#!/usr/bin/env bash
# =============================================================================
# run-frontier-single.sh — throughput frontier for ONE index + ONE memory config
# =============================================================================
# Runs INSIDE the container. Mirrors a single phase of
# vdb-tuning-main/TPCH/code/frontier_analysis/run_all_frontiers.sh:
# compile the right pgvector, load the index snapshot, start PG with NUMA
# binding, then run_frontier.sh (saturation + grid) in warm mode.
#
# Non-IVFFlat indexes copy OLAP saturation from the IVFFlat results dir.
# --olap-first measures it there first (IVFFlat snapshot, OLAP only), as the
# paper did, so HNSW / parallel can run without a full IVFFlat frontier.
#
# --saturation-only stops after step 1 (saturation points), with no grid.
# Per-point duration: MEASUREMENT_SEC env var (default 300).
#
# Usage (from host):
#   sudo docker exec -e PGPASSWORD=secret -e MEASUREMENT_SEC=600 vdb-tuning \
#     bash -lc '/vdb_tuning/build/run-frontier-single.sh 10:15 hnsw --olap-first [--saturation-only]'
# =============================================================================
set -euo pipefail

MEM="$1"; INDEX="$2"; shift 2
OLAP_FIRST=""; SAT_ONLY=""; POINTS=""; POINTS_NEXT=""
for a in "$@"; do
    case "$a" in
        --olap-first) OLAP_FIRST="--olap-first" ;;
        --saturation-only) SAT_ONLY=1 ;;
        --points) POINTS_NEXT=1 ;;
        --ef-search|--probes|--recall|--variant) VAR_NEXT="$a" ;;  # opt-in run variants, passed through
        *) if [ -n "${POINTS_NEXT}" ]; then POINTS="$a"; POINTS_NEXT=""; \
           elif [ -n "${VAR_NEXT:-}" ]; then VAR_ARGS="${VAR_ARGS:-} ${VAR_NEXT} $a"; VAR_NEXT=""; \
           else echo "unknown flag: $a" >&2; exit 1; fi ;;
    esac
done
export MEASUREMENT_SEC="${MEASUREMENT_SEC:-300}"

PG_INSTALL=/usr/local/pgsql
PGDATA=/mnt/nvme/pgdata
SNAP=/mnt/nvme/saved_indexes
FA=/vdb_tuning/vdb-tuning-main/TPCH/code/frontier_analysis
PGV_DEFAULT=/vdb_tuning/postgres/postgres/pgvector
PGV_PARALLEL=/vdb_tuning/postgres/postgres/pgvector_modified_final
NUMA=$(python3 -c "import json; print(json.load(open('${FA}/../experiment_config.json'))['hardware']['postgres_numa_nodes'])")
export PATH=${PG_INSTALL}/bin:$PATH

log() { echo "[$(date +%F' '%T)] $*"; }

compile_pgvector() {
    log "Compiling + installing pgvector from $1"
    cd "$1"
    make clean > /dev/null 2>&1 || true
    make -j"$(nproc)" PG_CONFIG=${PG_INSTALL}/bin/pg_config > /dev/null
    make install PG_CONFIG=${PG_INSTALL}/bin/pg_config > /dev/null
}

load_snapshot() {
    log "Loading snapshot $1"
    su - postgres -c "${PG_INSTALL}/bin/pg_ctl -D ${PGDATA} stop -m fast" 2>/dev/null || true
    sleep 2
    rm -rf "${PGDATA:?}"/*
    pigz -dc "$1" | tar -xf - -C "${PGDATA}"
    chown -R postgres:postgres "${PGDATA}"
    chmod 700 "${PGDATA}"
}

start_pg() {
    log "Starting PostgreSQL (NUMA nodes: ${NUMA})"
    su - postgres -c "numactl --cpunodebind=${NUMA} --membind=${NUMA} ${PG_INSTALL}/bin/pg_ctl -D ${PGDATA} -l /tmp/logfile start"
    for _ in $(seq 1 30); do
        su - postgres -c "${PG_INSTALL}/bin/pg_isready" > /dev/null 2>&1 && return 0
        sleep 2
    done
    log "ERROR: PostgreSQL did not start"; exit 1
}

case "${INDEX}" in
    ivfflat)          SNAPSHOT=${SNAP}/gist_ivfflat_lists8192_pgdata.tar.gz;          PGV=${PGV_DEFAULT};  ARGS="--ivfflat-lists 8192" ;;
    hnsw)             SNAPSHOT=${SNAP}/gist_hnsw_efc256_m32_pgdata.tar.gz;            PGV=${PGV_DEFAULT};  ARGS="--hnsw-m 32 --hnsw-efc 256" ;;
    hnsw_m8)          INDEX=hnsw; SNAPSHOT=${SNAP}/gist_hnsw_efc256_m8_pgdata.tar.gz; PGV=${PGV_DEFAULT};  ARGS="--hnsw-m 8 --hnsw-efc 256" ;;
    parallel_ivfflat) SNAPSHOT=${SNAP}/gist_parallel_ivfflat_lists8192_pgdata.tar.gz; PGV=${PGV_PARALLEL}; ARGS="--ivfflat-lists 8192" ;;
    *) echo "unknown index: ${INDEX}" >&2; exit 1 ;;
esac
ARGS="${ARGS}${VAR_ARGS:-}"

log "=== Frontier: index=${INDEX} mem=${MEM} MEASUREMENT_SEC=${MEASUREMENT_SEC} NUMA=${NUMA} olap_first=${OLAP_FIRST:-no} saturation_only=${SAT_ONLY:-no}"

compile_pgvector "${PGV}"

if [ "${OLAP_FIRST}" = "--olap-first" ]; then
    log "=== OLAP-only saturation pass (IVFFlat snapshot)"
    load_snapshot "${SNAP}/gist_ivfflat_lists8192_pgdata.tar.gz"
    start_pg
    cd "${FA}"
    python3 -u find_saturation_points.py --memory "${MEM}" --warm --index ivfflat \
        --ivfflat-lists 8192 --snapshot --olap-only
fi

log "=== ${INDEX} frontier"
load_snapshot "${SNAPSHOT}"
start_pg
cd "${FA}"
if [ -n "${SAT_ONLY}" ]; then
    python3 -u find_saturation_points.py --memory "${MEM}" --warm --index "${INDEX}" ${ARGS} --snapshot
elif [ -n "${POINTS}" ]; then
    # Selected points only. Requires saturation JSONs to already exist in the
    # results dir (tau_max / alpha_max are read, not re-measured). Writes nothing.
    log "=== selected points: ${POINTS} (MEASUREMENT_SEC=${MEASUREMENT_SEC})"
    python3 -u build_throughput_frontier.py --memory "${MEM}" --warm --index "${INDEX}" \
        ${ARGS} --snapshot --single "${POINTS}"
else
    ./run_frontier.sh "${MEM}" --warm --index "${INDEX}" ${ARGS} --snapshot
fi

log "=== DONE: index=${INDEX} mem=${MEM}"
