#!/usr/bin/env bash
# =============================================================================
# tune.sh — TUNING PHASE (build + tune each index type, save params + snapshots)
# =============================================================================
# Run AFTER install.sh. For the chosen dataset (default: gist) this:
#   - tune_ivfflat.py          -> best (lists, probes) + snapshot
#   - tune_hnsw.py             -> best (m, ef_construction, ef_search) + snapshot
#   - tune_parallel_ivfflat.py -> parallel index (lists from ivfflat) + snapshot
#
# Ground-truth exact results ship with the repo (exact_results_gist/...), so they
# are reused automatically; no recompute needed.
#
# Outputs:
#   results_from_tune_scripts/tune_{ivfflat,hnsw}_results_{dataset}.json
#   /mnt/nvme/saved_indexes/{dataset}_{ivfflat,hnsw,parallel_ivfflat}_*_pgdata.tar.gz
#
# Runs inside the container (via docker exec) with PGPASSWORD set. Idempotent:
# tune scripts reuse an existing snapshot if present.
#
# Usage (from the host):
#   ./build/tune.sh                       # gist, all three index types
#   ./build/tune.sh --dataset gist --only ivfflat,hnsw
#   ./build/tune.sh --recall 0.9
# =============================================================================
set -euo pipefail

DATASET="gist"
RECALL="0.9"
ONLY="ivfflat,hnsw,parallel"     # comma list: ivfflat,hnsw,parallel
CN="vdb-tuning"
DOCKER="sudo docker"

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --recall)  RECALL="$2"; shift 2 ;;
        --only)    ONLY="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

LOGDIR=/vdb_tuning/build/logs
TUNE_DIR=/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/tuning
PY=/opt/vdb-venv/bin/python

runtune() {   # runtune <script> <logname> <args...>
    local script="$1" logname="$2"; shift 2
    echo -e "\n\033[1;34m==>\033[0m tuning: ${script} $*"
    ${DOCKER} exec -e PGPASSWORD=secret "${CN}" bash -lc \
        "mkdir -p ${LOGDIR}; cd ${TUNE_DIR} && PGPASSWORD=secret ${PY} -u ${script} $* 2>&1 | tee ${LOGDIR}/${logname}"
}

echo "Tuning phase: dataset=${DATASET} recall=${RECALL} only=${ONLY}"

case ",${ONLY}," in *,ivfflat,*)
    runtune tune_ivfflat.py "tune_ivfflat_${DATASET}.log" --dataset "${DATASET}" --recall "${RECALL}" --warm ;;
esac
case ",${ONLY}," in *,hnsw,*)
    runtune tune_hnsw.py "tune_hnsw_${DATASET}.log" --dataset "${DATASET}" --recall "${RECALL}" --warm ;;
esac
case ",${ONLY}," in *,parallel,*)
    # parallel reads best lists from tune_ivfflat results; run it after ivfflat.
    runtune tune_parallel_ivfflat.py "tune_parallel_${DATASET}.log" --dataset "${DATASET}" ;;
esac

echo -e "\n\033[1;32m==>\033[0m TUNING COMPLETE for ${DATASET}"
${DOCKER} exec "${CN}" bash -lc 'ls -la /mnt/nvme/saved_indexes/ 2>/dev/null'
