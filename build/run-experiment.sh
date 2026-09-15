#!/usr/bin/env bash
# =============================================================================
# run-experiment.sh — run one experiment CATEGORY against the tuned DB
# =============================================================================
# Run AFTER install.sh (and tune.sh for anything needing snapshots).
# Thin, fixed wrappers around the repo's experiment scripts, launched inside the
# container with PGPASSWORD set and from the correct directory.
#
# Categories:
#   recall-qps     Recall vs QPS sweep for an index type, + plot.
#                  Needs tune.sh snapshots. Args: --index <ivfflat|hnsw|parallel_ivfflat|all>
#                                                  --mode <warm|cold>
#   parallel-vs-default
#                  Parallel pgvector (NUM_WORKERS 4/8/12) vs default IVFFlat.
#                  Reuses the ivfflat snapshot. Args: --mode <warm|cold>
#   topk-opt       Top-K executor optimization (execExpr.c flag on/off) across K.
#                  (Rebuilds PostgreSQL per config — long.)
#   tpch-interference
#                  TPC-H <-> ANN frontier experiments (cgroup + core pinning).
#
# Usage (from the host):
#   ./build/run-experiment.sh recall-qps --index ivfflat --mode warm
#   ./build/run-experiment.sh parallel-vs-default --mode warm
#   ./build/run-experiment.sh topk-opt
#   ./build/run-experiment.sh tpch-interference
# =============================================================================
set -euo pipefail

CN="vdb-tuning"
DOCKER="sudo docker"
PY=/opt/vdb-venv/bin/python
VMAIN=/vdb_tuning/vdb-tuning-main
LOGDIR=/vdb_tuning/build/logs
DATASET="gist"

[ $# -ge 1 ] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
CATEGORY="$1"; shift

# defaults
INDEX="ivfflat"; MODE="warm"; MEM="10:15"   # MEM = shared_buffers_GB:cgroup_GB (tpch-interference)
while [ $# -gt 0 ]; do
    case "$1" in
        --index)   INDEX="$2"; shift 2 ;;
        --mode)    MODE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --mem)     MEM="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

dexec() {  # run a command string inside the container with PGPASSWORD + logging
    local logname="$1"; shift
    ${DOCKER} exec -e PGPASSWORD=secret "${CN}" bash -lc \
        "mkdir -p ${LOGDIR}; PGPASSWORD=secret $* 2>&1 | tee ${LOGDIR}/${logname}"
}

case "${CATEGORY}" in
  recall-qps)
    DIR="${VMAIN}/compute_and_compare_recall/recall_qps_plots"
    echo "==> recall-qps: dataset=${DATASET} index=${INDEX} mode=${MODE}"
    dexec "recall_qps_${DATASET}_${INDEX}_${MODE}.log" \
        "cd ${DIR} && ${PY} -u recall_qps_diagram.py --dataset ${DATASET} --mode ${MODE} --index ${INDEX}"
    echo "==> plotting"
    dexec "plot_recall_qps_${DATASET}_${MODE}.log" \
        "cd ${DIR} && ${PY} -u plot_recall_qps.py --dataset ${DATASET} --mode ${MODE} --results-dir results_recall_qps"
    echo "    figure: ${DIR}/figures/recall_qps_${DATASET}_${MODE}.png"
    ;;

  parallel-vs-default)
    DIR="${VMAIN}/parallel_implementation_vs_default"
    echo "==> parallel-vs-default: dataset=${DATASET} mode=${MODE}"
    dexec "parallel_vs_default_${DATASET}_${MODE}.log" \
        "cd ${DIR} && ${PY} -u run_comparison_workers.py --dataset ${DATASET} --mode ${MODE}"
    echo "==> plotting"
    dexec "plot_parallel_vs_default_${DATASET}_${MODE}.log" \
        "cd ${DIR} && ${PY} -u plot_comparison_workers.py --dataset ${DATASET} --mode ${MODE} || true"
    ;;

  topk-opt)
    DIR="${VMAIN}/avoid_reduntant_computations_optimization"
    echo "==> topk-opt: (rebuilds PostgreSQL per config; long)"
    echo "    See ${DIR} scripts; running compute_optimization_results_for_different_K.py"
    dexec "topk_opt_${DATASET}.log" \
        "cd ${DIR} && ${PY} -u compute_optimization_results_for_different_K.py ivfflat with ${DATASET} warm"
    ;;

  tpch-interference)
    DIR="${VMAIN}/TPCH/code/frontier_analysis"
    echo "==> tpch-interference: frontier analysis (cgroup + core pinning)"
    # Auto-fit the CPU core split to THIS machine (avoids the committed 64-core layout).
    echo "    Auto-partitioning cores for this host..."
    ${DOCKER} exec "${CN}" bash -lc "/vdb_tuning/build/partition-cores.sh"
    # run_all_frontiers.sh needs a memory config SB:CG (shared_buffers:cgroup GB).
    echo "    NOTE: run tune-host-os.sh on the HOST first for benchmark-grade isolation."
    echo "    Using memory config MEM=${MEM} (override with --mem SB:CG)."
    dexec "tpch_interference_${DATASET}_${MEM}.log" \
        "cd ${DIR} && bash run_all_frontiers.sh ${MEM}"
    ;;

  *)
    echo "unknown category: ${CATEGORY}" >&2
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac

echo -e "\n\033[1;32m==>\033[0m ${CATEGORY} done. Logs in ${LOGDIR} (host: ~/vdb_tuning/build/logs)"
