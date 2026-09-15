#!/usr/bin/env bash
# =============================================================================
# apply-fixes.sh — patch repo files so the experiments run on this environment
# =============================================================================
# The upstream repo was written for a CloudLab x86 host. These edits make it
# run correctly in our container on x86_64 OR aarch64. All edits are idempotent
# (safe to run repeatedly) and each keeps a .orig backup on first change.
#
# Fixes applied:
#   1. config.yaml            : setup_nvme true -> false  (never reformat a disk)
#   2. experiment_config.json : postgres_numa_nodes -> "0" (this box has 1 NUMA node)
#   3. tune_ivfflat.py        : maintenance_work_mem 400GB -> 15GB, workers 64 -> 16
#   4. tune_parallel_ivfflat.py: same memory/worker fix
#   5. pgvector_modified_final/src/vector.c : guard <immintrin.h> for x86 only
#      (so the parallel fork compiles on ARM; harmless on x86)
#
# NOTE: the postgres password (secret) + TCP md5 auth are handled by
# build-postgres.sh; experiment scripts are launched with PGPASSWORD=secret
# by run-experiment.sh, so no per-script DB_CONFIG edits are needed.
#
# Run from the host (operates on the bind-mounted repo) or inside the container.
# =============================================================================
set -euo pipefail

# Resolve repo root whether we're on the host or in the container.
if [ -d /vdb_tuning ] && [ -f /vdb_tuning/config.yaml ]; then
    REPO=/vdb_tuning                                    # inside container
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # on host
fi
VMAIN="${REPO}/vdb-tuning-main"
TUNE="${VMAIN}/compute_and_compare_recall/tuning"

backup() { [ -f "$1.orig" ] || cp "$1" "$1.orig"; }
say() { echo "    $*"; }

echo "==> apply-fixes: REPO=${REPO}"

# --- 1. config.yaml: setup_nvme true -> false --------------------------------
CFG="${REPO}/config.yaml"
if grep -qE "^\s*setup_nvme:\s*true" "${CFG}" 2>/dev/null; then
    backup "${CFG}"
    sed -i 's/^\(\s*setup_nvme:\s*\)true/\1false/' "${CFG}"
    say "config.yaml: setup_nvme -> false"
else
    say "config.yaml: setup_nvme already false (ok)"
fi

# --- 1b. config.yaml: point source paths at the container repo ---------------
# Some bash experiment scripts (e.g. TPCH run_all_frontiers.sh) read this
# top-level config.yaml DIRECTLY (they do not honour $VDB_CONFIG), so its
# pg_src / pgvector_* paths must resolve inside the container. Rewrite any
# CloudLab-style prefix (/users/<name>/vdb_tuning) to the container mount
# (/vdb_tuning). Python scripts are unaffected (they use config.container.yaml).
if grep -qE "^\s*(pg_src|pgvector_default|pgvector_parallel):\s*/users/" "${CFG}" 2>/dev/null; then
    backup "${CFG}"
    sed -i -E 's#(^\s*(pg_src|pgvector_default|pgvector_parallel):\s*)/users/[^/]+/vdb_tuning#\1/vdb_tuning#' "${CFG}"
    say "config.yaml: source paths -> /vdb_tuning (for bash experiment scripts)"
else
    say "config.yaml: source paths already container-local (ok)"
fi

# --- 2. experiment_config.json: NUMA nodes -> "0" ----------------------------
EC="${VMAIN}/TPCH/code/experiment_config.json"
if [ -f "${EC}" ] && grep -qE '"postgres_numa_nodes":\s*"0-[0-9]+"' "${EC}"; then
    backup "${EC}"
    sed -i 's/"postgres_numa_nodes":\s*"0-[0-9]*"/"postgres_numa_nodes": "0"/' "${EC}"
    say "experiment_config.json: postgres_numa_nodes -> \"0\""
else
    say "experiment_config.json: NUMA already single-node or file absent (ok)"
fi

# --- 3 & 4. tune_ivfflat.py / tune_parallel_ivfflat.py: memory + workers ------
fix_mem() {
    local f="$1"
    [ -f "$f" ] || { say "$(basename "$f"): not found (skip)"; return; }
    if grep -qE "maintenance_work_mem = '400GB'|max_parallel_[a-z_]+ = '64'" "$f"; then
        backup "$f"
        # Use python for precise, quote-safe replacement.
        python3 - "$f" <<'PYEOF'
import sys
p=sys.argv[1]
lines=open(p).read().split("\n")
out=[]
for l in lines:
    if "maintenance_work_mem = '400GB'" in l:
        out.append(l.replace("'400GB'", "'15GB'"))
    elif "max_parallel_maintenance_workers = '64'" in l:
        out.append(l.replace("'64'", "'16'"))
    elif "max_parallel_workers_per_gather = '64'" in l:
        out.append(l.replace("'64'", "'16'"))
    else:
        out.append(l)
open(p,"w").write("\n".join(out))
PYEOF
        say "$(basename "$f"): maintenance_work_mem 400GB->15GB, workers 64->16"
    else
        say "$(basename "$f"): memory/workers already fixed (ok)"
    fi
}
fix_mem "${TUNE}/tune_ivfflat.py"
fix_mem "${TUNE}/tune_parallel_ivfflat.py"

# --- 5. vector.c: guard immintrin.h for x86 only -----------------------------
VC="${REPO}/postgres/postgres/pgvector_modified_final/src/vector.c"
if [ -f "${VC}" ] && grep -qE '^#include <immintrin.h>' "${VC}"; then
    backup "${VC}"
    python3 - "${VC}" <<'PYEOF'
import sys
p=sys.argv[1]
s=open(p).read()
old="#include <immintrin.h>\n"
new=("#if defined(__x86_64__) || defined(_M_AMD64)\n"
     "#include <immintrin.h>\n"
     "#endif\n")
idx=s.find(old)
if idx!=-1:
    s=s[:idx]+new+s[idx+len(old):]
    open(p,"w").write(s)
    print("    vector.c: guarded <immintrin.h> for x86")
else:
    print("    vector.c: include not found in expected form (skip)")
PYEOF
else
    say "vector.c: already guarded or file absent (ok)"
fi

echo "==> apply-fixes: done"
