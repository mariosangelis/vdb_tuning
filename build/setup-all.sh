#!/usr/bin/env bash
# One-command setup, run INSIDE the container.
# Runs, in order and idempotently:
#   install-prerequisites -> build-postgres -> load-tpch -> load-gist -> build-indexes
# Each step is resumable; re-running skips work already done.
#
# Usage:
#   setup-all.sh                 # full flow (ivfflat lists=1000)
#   setup-all.sh --index hnsw    # build HNSW instead
#   setup-all.sh --skip-tpch     # skip TPC-H (vectors only)
#   setup-all.sh --force         # force every step to redo
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

INDEX_TYPE="ivfflat"
SKIP_TPCH=0
FORCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --index) INDEX_TYPE="$2"; shift 2 ;;
        --skip-tpch) SKIP_TPCH=1; shift ;;
        --force) FORCE="--force"; shift ;;
        *) die "unknown arg: $1" ;;
    esac
done

LOG_DIR="${REPO_ROOT}/build/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S).log"

# Mirror all output to a timestamped log.
exec > >(tee -a "${RUN_LOG}") 2>&1

log "######################################################"
log "vdb_tuning setup-all  (log: ${RUN_LOG})"
log "  config      = ${CONFIG_FILE}"
log "  index type  = ${INDEX_TYPE}"
log "  skip tpch   = ${SKIP_TPCH}"
log "  force       = ${FORCE:-no}"
log "######################################################"

step() { log ""; log ">>> STEP: $*"; }

step "install prerequisites"
bash "${COMMON_DIR}/install-prerequisites.sh"

step "build postgres + pgvector"
bash "${COMMON_DIR}/build-postgres.sh" ${FORCE}

if [ "${SKIP_TPCH}" -eq 0 ]; then
    step "load TPC-H SF${TPCH_SCALE}"
    bash "${COMMON_DIR}/load-tpch.sh" ${FORCE}
else
    log "Skipping TPC-H (--skip-tpch)."
fi

step "download + load GIST-960"
bash "${COMMON_DIR}/load-gist.sh" ${FORCE}

step "build vector index (${INDEX_TYPE})"
bash "${COMMON_DIR}/build-indexes.sh" --type "${INDEX_TYPE}" ${FORCE}

log ""
log "######################################################"
log "SETUP COMPLETE."
log "  DB: psql -h 127.0.0.1 -U postgres -d ${DB_NAME} (password: ${PG_PASSWORD})"
log "  Next: run experiments under vdb-tuning-main/ (see README)."
log "  Full log: ${RUN_LOG}"
log "######################################################"
