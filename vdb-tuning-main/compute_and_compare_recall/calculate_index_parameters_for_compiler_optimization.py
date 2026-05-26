#!/usr/bin/env python3
"""
Calculate optimal probes/ef_search for all K values at recall >= target.
Used by the compiler optimization experiment.

Uses EXPLAIN (ANALYZE, FORMAT JSON) for pure PostgreSQL execution timing.
Does NOT drop existing indexes if one of the correct type already exists.
Loads NUMA config from experiment_config.json for consistent PostgreSQL pinning.

Usage:
    python3 calculate_index_parameters_for_compiler_optimization.py ivfflat 0.9
    python3 calculate_index_parameters_for_compiler_optimization.py hnsw 0.95
    python3 calculate_index_parameters_for_compiler_optimization.py ivfflat 0.9 --lists 8192
    python3 calculate_index_parameters_for_compiler_optimization.py hnsw 0.9 --m 16 --efc 128
"""

import psycopg2
import time
import subprocess
import json
import sys
import os
import shutil
from pathlib import Path

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

WORKLOAD_PATH = str(PROJECT_DIR / "sampled_vectors_gist.txt")
EXACT_RESULTS_DIR = SCRIPT_DIR / "exact_results_gist"
RESULTS_DIR = SCRIPT_DIR / "results_from_calculate_index_parameters_for_compiler_optimization"

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

# === EXPERIMENT PARAMETERS ===
K_VALUES = [1, 10, 100, 200, 400, 800, 1000]

# Tuning parameter search lists
IVFFLAT_PROBES_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500]
HNSW_EF_SEARCH_LIST = [1, 10, 50] + list(range(100, 1050, 50))  # 100, 150, 200, ..., 1000

# === INDEX DEFINITIONS (loaded from tuning results) ===
def _load_tuned_index_defs():
    """Load optimal build params from tune_ivfflat_results.json and tune_hnsw_results.json."""
    defs = {
        "ivfflat": {
            "name": "gist_data_embedding_idx",
            "create_sql": "CREATE INDEX gist_data_embedding_idx ON gist_data USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000)",
        },
        "hnsw": {
            "name": "gist_data_embedding_hnsw_idx",
            "create_sql": "CREATE INDEX gist_data_embedding_hnsw_idx ON gist_data USING hnsw (embedding vector_l2_ops)",
        },
    }

    # Try to load IVFFlat tuning results
    ivf_tune_file = SCRIPT_DIR / "results_from_tune_scripts/tune_ivfflat_results.json"
    if ivf_tune_file.exists():
        with open(ivf_tune_file, 'r') as f:
            ivf_tune = json.load(f)
        if "best_configuration" in ivf_tune:
            best_lists = ivf_tune["best_configuration"]["lists"]
            defs["ivfflat"]["create_sql"] = (
                f"CREATE INDEX gist_data_embedding_idx ON gist_data "
                f"USING ivfflat (embedding vector_l2_ops) WITH (lists = {best_lists})"
            )
            print(f"  IVFFlat: using tuned lists={best_lists} (from tune_ivfflat_results.json)")
        else:
            print("  IVFFlat: no best_configuration in tuning results, using default lists=1000")
    else:
        print("  IVFFlat: tune_ivfflat_results.json not found, using default lists=1000")

    # Try to load HNSW tuning results
    hnsw_tune_file = SCRIPT_DIR / "results_from_tune_scripts/tune_hnsw_results.json"
    if hnsw_tune_file.exists():
        with open(hnsw_tune_file, 'r') as f:
            hnsw_tune = json.load(f)
        if "best_configuration" in hnsw_tune:
            best_m = hnsw_tune["best_configuration"]["m"]
            ef_construction = hnsw_tune["best_configuration"].get("ef_construction", 200)
            defs["hnsw"]["create_sql"] = (
                f"CREATE INDEX gist_data_embedding_hnsw_idx ON gist_data "
                f"USING hnsw (embedding vector_l2_ops) WITH (m = {best_m}, ef_construction = {ef_construction})"
            )
            print(f"  HNSW: using tuned m={best_m}, ef_construction={ef_construction} (from tune_hnsw_results.json)")
        else:
            print("  HNSW: no best_configuration in tuning results, using defaults")
    else:
        print("  HNSW: tune_hnsw_results.json not found, using defaults")

    return defs

INDEX_DEFS = _load_tuned_index_defs()

# PostgreSQL paths
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"

# === LOAD HARDWARE CONFIG FOR NUMA PINNING ===
EXPERIMENT_CONFIG_PATH = PROJECT_DIR / "TPCH" / "code" / "experiment_config.json"
if EXPERIMENT_CONFIG_PATH.exists():
    with open(EXPERIMENT_CONFIG_PATH, "r") as _f:
        _hw_cfg = json.load(_f).get("hardware", {})
    PG_NUMA_NODES = _hw_cfg.get("postgres_numa_nodes", "0")
    print(f"  Loaded NUMA config: postgres_numa_nodes={PG_NUMA_NODES}")
else:
    PG_NUMA_NODES = "0"
    print("  WARNING: experiment_config.json not found, using NUMA node 0")


def wait_for_postgres(max_attempts=60):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            print("PostgreSQL is ready.")
            return True
        except psycopg2.OperationalError:
            if attempt % 10 == 0:
                print(f"Waiting for PostgreSQL... attempt {attempt + 1}")
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def restart_postgres():
    print("Restarting PostgreSQL (with NUMA pinning)...")
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={PG_NUMA_NODES} --membind={PG_NUMA_NODES} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def check_existing_index(index_type):
    """Check if an index of the correct type (ivfflat/hnsw) already exists on gist_data.
    Returns the index name if found, None otherwise."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT i.relname as index_name, am.amname as index_type
        FROM pg_index ix
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_am am ON am.oid = i.relam
        WHERE t.relname = 'gist_data'
        AND am.amname = %s
    """, (index_type,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0]
    return None


def ensure_index(index_type):
    """Ensure an index of the correct type exists. Does NOT drop existing matching index."""
    existing = check_existing_index(index_type)
    if existing:
        print(f"  Index of type '{index_type}' already exists: '{existing}'. Keeping it.")
        return

    # No matching index exists - need to create one
    idx_def = INDEX_DEFS[index_type]
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop any indexes of OTHER types (e.g., drop hnsw if we need ivfflat)
    other_type = "hnsw" if index_type == "ivfflat" else "ivfflat"
    other_existing = check_existing_index(other_type)
    if other_existing:
        print(f"  Dropping conflicting {other_type} index '{other_existing}'...")
        cur.execute(f"DROP INDEX IF EXISTS {other_existing}")

    print(f"  Creating index '{idx_def['name']}' (this may take a while)...")
    start = time.time()
    cur.execute(idx_def["create_sql"])
    elapsed = time.time() - start
    print(f"  Index created in {elapsed:.1f}s")

    cur.close()
    conn.close()


def compute_recall(workload, exact_results, k, param_name, param_value):
    """Run all queries with given param, using EXPLAIN (ANALYZE, FORMAT JSON) for timing.
    Returns (avg_recall, avg_time_ms) where avg_time_ms is pure PostgreSQL execution time."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if param_name == "ivfflat.probes":
        cur.execute(f"SET ivfflat.probes = {param_value}")
    elif param_name == "hnsw.ef_search":
        cur.execute(f"SET hnsw.ef_search = {param_value}")

    cur.execute("SET enable_seqscan = off")

    total_recall = 0.0
    total_exec_time_ms = 0.0

    for i in range(len(workload)):
        exact_ids = set(exact_results[i][:k])
        vec_str = workload[i]

        query = f"SELECT id FROM gist_data ORDER BY embedding <-> '{vec_str}'::vector LIMIT {k}"

        # Run EXPLAIN ANALYZE to get pure PostgreSQL execution time
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        explain_json = cur.fetchone()[0]
        execution_time_ms = explain_json[0]["Execution Time"]
        total_exec_time_ms += execution_time_ms

        # Run the actual query to get result IDs for recall computation
        cur.execute(query)
        result_ids = set(row[0] for row in cur.fetchall())
        hits = len(result_ids & exact_ids)
        total_recall += hits / k

    avg_recall = total_recall / len(workload)
    avg_time_ms = total_exec_time_ms / len(workload)

    cur.close()
    conn.close()
    return avg_recall, avg_time_ms


def tune_for_k(index_type, k, workload, exact_results, param_list, target_recall=0.9):
    param_name = "ivfflat.probes" if index_type == "ivfflat" else "hnsw.ef_search"
    param_label = "probes" if index_type == "ivfflat" else "ef_search"

    print(f"  K={k}:", end=" ")

    best_param = None
    best_recall = 0.0
    best_time = 0.0
    all_results = []

    for param_value in param_list:
        if index_type == "hnsw" and param_value < k:
            continue

        avg_recall, avg_time_ms = compute_recall(
            workload, exact_results, k, param_name, param_value
        )

        all_results.append({
            param_label: param_value,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2)
        })

        if avg_recall >= target_recall and best_param is None:
            best_param = param_value
            best_recall = avg_recall
            best_time = avg_time_ms
            print(f"{param_label}={param_value}, recall={avg_recall:.4f}, time={avg_time_ms:.2f}ms (EXPLAIN ANALYZE)")
            break

    if best_param is None:
        if all_results:
            best_entry = max(all_results, key=lambda x: x["recall"])
            best_param = best_entry[param_label]
            best_recall = best_entry["recall"]
            best_time = best_entry["avg_time_ms"]
        print(f"WARNING: max recall={best_recall:.4f} with {param_label}={best_param}")

    return {
        "k": k,
        "index_type": index_type,
        "optimal_" + param_label: best_param,
        "recall": best_recall,
        "avg_time_ms": best_time,
        "target_recall": target_recall,
        "timing_method": "EXPLAIN_ANALYZE",
        "all_tested": all_results
    }


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("ivfflat", "hnsw"):
        print("Usage: python3 calculate_index_parameters_for_compiler_optimization.py <ivfflat|hnsw> <target_recall> [--lists N] [--m M] [--efc EFC]")
        print("  target_recall: e.g. 0.9 or 0.95")
        print("  --lists N: manually set IVFFlat lists (overrides tuning JSON)")
        print("  --m M: manually set HNSW m parameter (overrides tuning JSON)")
        print("  --efc EFC: manually set HNSW ef_construction (overrides tuning JSON)")
        sys.exit(1)

    index_type = sys.argv[1]
    TARGET_RECALL = float(sys.argv[2])

    # Parse optional overrides
    manual_lists = None
    manual_m = None
    manual_efc = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--lists" and i + 1 < len(sys.argv):
            manual_lists = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--m" and i + 1 < len(sys.argv):
            manual_m = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--efc" and i + 1 < len(sys.argv):
            manual_efc = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    param_list = IVFFLAT_PROBES_LIST if index_type == "ivfflat" else HNSW_EF_SEARCH_LIST
    param_label = "probes" if index_type == "ivfflat" else "ef_search"

    print(f"\n{'='*60}")
    print(f"Index type: {index_type.upper()}")
    print(f"Target recall: {TARGET_RECALL}")
    print(f"K values: {K_VALUES}")
    print(f"Timing method: EXPLAIN (ANALYZE, FORMAT JSON)")
    print(f"NUMA pinning: nodes={PG_NUMA_NODES}")
    print(f"{'='*60}")

    # Load workload
    print(f"\nLoading workload from {WORKLOAD_PATH}...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Ensure correct index exists (without dropping if it's already the right type)
    print(f"\nChecking index...")
    if index_type == "hnsw" and (manual_m is not None or manual_efc is not None):
        # Manual HNSW override: check if hnsw index exists, if so keep it
        existing = check_existing_index("hnsw")
        if existing:
            print(f"  HNSW index already exists: '{existing}'. Keeping it (manual override specified: m={manual_m}, efc={manual_efc}).")
            print(f"  NOTE: If you want to rebuild with new params, drop the index manually first.")
        else:
            # Build with manual params
            m_val = manual_m if manual_m else 32
            efc_val = manual_efc if manual_efc else 128
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            cur = conn.cursor()
            # Drop conflicting ivfflat index if exists
            ivf_existing = check_existing_index("ivfflat")
            if ivf_existing:
                print(f"  Dropping conflicting ivfflat index '{ivf_existing}'...")
                cur.execute(f"DROP INDEX IF EXISTS {ivf_existing}")
            print(f"  Creating HNSW index with m={m_val}, ef_construction={efc_val}...")
            start = time.time()
            cur.execute(f"CREATE INDEX gist_data_hnsw_idx ON gist_data USING hnsw (embedding vector_l2_ops) WITH (m = {m_val}, ef_construction = {efc_val})")
            print(f"  Index created in {time.time() - start:.1f}s")
            cur.close()
            conn.close()
    elif index_type == "ivfflat" and manual_lists is not None:
        # Manual IVFFlat override: check if ivfflat index exists, if so keep it
        existing = check_existing_index("ivfflat")
        if existing:
            print(f"  IVFFlat index already exists: '{existing}'. Keeping it (manual override specified: lists={manual_lists}).")
            print(f"  NOTE: If you want to rebuild with new params, drop the index manually first.")
        else:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            cur = conn.cursor()
            # Drop conflicting hnsw index if exists
            hnsw_existing = check_existing_index("hnsw")
            if hnsw_existing:
                print(f"  Dropping conflicting hnsw index '{hnsw_existing}'...")
                cur.execute(f"DROP INDEX IF EXISTS {hnsw_existing}")
            print(f"  Creating IVFFlat index with lists={manual_lists}...")
            start = time.time()
            cur.execute(f"CREATE INDEX gist_data_embedding_idx ON gist_data USING ivfflat (embedding vector_l2_ops) WITH (lists = {manual_lists})")
            print(f"  Index created in {time.time() - start:.1f}s")
            cur.close()
            conn.close()
    else:
        ensure_index(index_type)

    # Restart PostgreSQL with cache clearing and NUMA pinning
    restart_postgres()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tune for each K
    all_optimal = {}
    for k in K_VALUES:
        exact_file = EXACT_RESULTS_DIR / f"exact_search_results_1000_queries_gist_K_{k}.json"
        if not exact_file.exists():
            print(f"  K={k}: SKIPPED (no exact results file: {exact_file})")
            continue

        with open(exact_file, 'r') as f:
            exact_results = json.load(f)

        result = tune_for_k(index_type, k, workload, exact_results, param_list, TARGET_RECALL)
        all_optimal[str(k)] = result

    # Save results
    output_file = RESULTS_DIR / f"optimal_{index_type}_params_recall_{TARGET_RECALL}.json"
    with open(output_file, 'w') as f:
        json.dump(all_optimal, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - Optimal {param_label} for {index_type.upper()} (recall >= {TARGET_RECALL})")
    print(f"Timing: EXPLAIN (ANALYZE, FORMAT JSON) - pure PostgreSQL execution time")
    print(f"{'='*60}")
    print(f"{'K':<8} {param_label:<12} {'Recall':<10} {'Time (ms)':<12}")
    print("-" * 42)
    for k in K_VALUES:
        if str(k) in all_optimal:
            r = all_optimal[str(k)]
            opt_val = r.get(f"optimal_{param_label}", "N/A")
            print(f"{k:<8} {str(opt_val):<12} {r['recall']:<10.4f} {r['avg_time_ms']:<12.2f}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
