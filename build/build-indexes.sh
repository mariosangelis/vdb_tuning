#!/usr/bin/env bash
# Build the vector index on gist_data. Default: IVFFlat with lists=1000
# (matches the experiment scripts' CREATE INDEX gist_data_embedding_idx).
# Optionally HNSW. Run AFTER load-gist.sh. Idempotent. Architecture-agnostic.
#
# Usage:
#   build-indexes.sh                      # ivfflat, lists=1000
#   build-indexes.sh --type hnsw          # hnsw, m=16 ef_construction=128
#   build-indexes.sh --type ivfflat --lists 2000
#   build-indexes.sh --force              # drop & rebuild
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TYPE="ivfflat"
LISTS=1000
M=16
EFC=128
FORCE=0
TABLE="gist_data"
IDX="gist_data_embedding_idx"

while [ $# -gt 0 ]; do
    case "$1" in
        --type)  TYPE="$2"; shift 2 ;;
        --lists) LISTS="$2"; shift 2 ;;
        --m)     M="$2"; shift 2 ;;
        --ef-construction) EFC="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) die "unknown arg: $1" ;;
    esac
done

psql_tcp() { PGPASSWORD="${PG_PASSWORD}" "${PGBIN}/psql" -h 127.0.0.1 -U postgres -d "${DB_NAME}" "$@"; }

log "======================================================"
log "Vector index on ${TABLE}: type=${TYPE} lists=${LISTS} m=${M} efc=${EFC}"
log "======================================================"

rows="$(psql_tcp -tAqc "SELECT count(*) FROM ${TABLE};" 2>/dev/null || echo 0)"
[ "${rows}" -gt 0 ] || die "${TABLE} is empty or missing; run load-gist.sh first."

# Existing index?
exists="$(psql_tcp -tAqc "SELECT 1 FROM pg_indexes WHERE tablename='${TABLE}' AND indexname='${IDX}';" 2>/dev/null || echo)"
if [ -n "${exists}" ]; then
    if [ "${FORCE}" -eq 0 ]; then
        log "Index ${IDX} already exists. Skipping (use --force to rebuild)."
        exit 0
    fi
    log "Dropping existing ${IDX} (--force)..."
    psql_tcp -c "DROP INDEX IF EXISTS ${IDX};"
fi

# Raise memory + parallelism for the build.
psql_tcp -c "SET maintenance_work_mem='15GB';" >/dev/null 2>&1 || true

log "Building index (this can take many minutes for 1M vectors)..."
case "${TYPE}" in
    ivfflat)
        psql_tcp -v ON_ERROR_STOP=1 -c \
            "CREATE INDEX ${IDX} ON ${TABLE} USING ivfflat (embedding vector_l2_ops) WITH (lists = ${LISTS});"
        ;;
    hnsw)
        psql_tcp -v ON_ERROR_STOP=1 -c \
            "CREATE INDEX ${IDX} ON ${TABLE} USING hnsw (embedding vector_l2_ops) WITH (m = ${M}, ef_construction = ${EFC});"
        ;;
    *) die "unknown --type ${TYPE} (use ivfflat|hnsw)" ;;
esac

psql_tcp -c "ANALYZE ${TABLE};" >/dev/null
log "Index ${IDX} (${TYPE}) built. Verifying:"
psql_tcp -c "\\di+ ${IDX}"
log "Done."
