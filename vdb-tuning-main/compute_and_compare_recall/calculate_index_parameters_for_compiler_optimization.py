#!/usr/bin/env python3
"""
Calculate optimal probes/ef_search for all K values at recall >= 0.9.
Used by the compiler optimization experiment.

Creates the required index if it does not exist (drops conflicting indexes).

Usage:
    python3 calculate_index_parameters_for_compiler_optimization.py ivfflat 0.9
    python3 calculate_index_parameters_for_compiler_optimization.py hnsw 0.95
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
HNSW_EF_SEARCH_LIST = list(range(100, 1050, 50))  # 100, 150, 200, ..., 1000

# === INDEX DEFINITIONS ===
INDEX_DEFS = {
    "ivfflat": {
        "name": "gist_data_embedding_idx",
        "create_sql": "CREATE INDEX gist_data_embedding_idx ON gist_data USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000)",
    },
    "hnsw": {
        "name": "gist_data_embedding_hnsw_idx",
        "create_sql": "CREATE INDEX gist_data_embedding_hnsw_idx ON gist_data USING hnsw (embedding vector_l2_ops)",
    },
}

# PostgreSQL paths
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"


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
    print("Restarting PostgreSQL...")
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, check=True
    )
    time.sleep(3)
    wait_for_postgres()


def ensure_index(index_type):
    idx_def = INDEX_DEFS[index_type]
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data' AND indexname = %s",
        (idx_def["name"],)
    )
    if cur.fetchone():
        print(f"Index '{idx_def['name']}' already exists.")
        cur.close()
        conn.close()
        return

    # Drop existing indexes
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data'")
    for (idx_name,) in cur.fetchall():
        print(f"Dropping existing index '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    print(f"Creating index '{idx_def['name']}' (this may take a while)...")
    start = time.time()
    cur.execute(idx_def["create_sql"])
    elapsed = time.time() - start
    print(f"Index created in {elapsed:.1f}s")

    cur.close()
    conn.close()


def compute_recall(workload, exact_results, k, param_name, param_value):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if param_name == "ivfflat.probes":
        cur.execute(f"SET ivfflat.probes = {param_value}")
    elif param_name == "hnsw.ef_search":
        cur.execute(f"SET hnsw.ef_search = {param_value}")

    total_recall = 0.0
    total_time = 0.0

    for i in range(len(workload)):
        exact_ids = set(exact_results[i][:k])
        vec_str = workload[i]

        start = time.time()
        cur.execute(f"SELECT id FROM gist_data ORDER BY embedding <-> '{vec_str}'::vector LIMIT {k}")
        elapsed = time.time() - start
        total_time += elapsed

        result_ids = set(row[0] for row in cur.fetchall())
        hits = len(result_ids & exact_ids)
        total_recall += hits / k

    avg_recall = total_recall / len(workload)
    avg_time_ms = (total_time / len(workload)) * 1000

    cur.close()
    conn.close()
    return avg_recall, avg_time_ms


def tune_for_k(index_type, k, workload, exact_results, param_list):
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

        if avg_recall >= TARGET_RECALL and best_param is None:
            best_param = param_value
            best_recall = avg_recall
            best_time = avg_time_ms
            print(f"{param_label}={param_value}, recall={avg_recall:.4f}")
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
        "target_recall": TARGET_RECALL,
        "all_tested": all_results
    }


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("ivfflat", "hnsw"):
        print("Usage: python3 calculate_index_parameters_for_compiler_optimization.py <ivfflat|hnsw> <target_recall>")
        print("  target_recall: e.g. 0.9 or 0.95")
        sys.exit(1)

    index_type = sys.argv[1]
    TARGET_RECALL = float(sys.argv[2])
    param_list = IVFFLAT_PROBES_LIST if index_type == "ivfflat" else HNSW_EF_SEARCH_LIST
    param_label = "probes" if index_type == "ivfflat" else "ef_search"

    print(f"Index type: {index_type.upper()}")
    print(f"Target recall: {TARGET_RECALL}")
    print(f"K values: {K_VALUES}")

    # Load workload
    print(f"Loading workload from {WORKLOAD_PATH}...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Ensure index exists
    ensure_index(index_type)

    # Restart PostgreSQL
    restart_postgres()

    # Tune for each K
    all_optimal = {}
    for k in K_VALUES:
        exact_file = EXACT_RESULTS_DIR / f"exact_search_results_1000_queries_gist_K_{k}.json"
        if not exact_file.exists():
            print(f"  K={k}: SKIPPED (no exact results file)")
            continue

        with open(exact_file, 'r') as f:
            exact_results = json.load(f)

        result = tune_for_k(index_type, k, workload, exact_results, param_list)
        all_optimal[str(k)] = result

    # Save results
    output_file = SCRIPT_DIR / f"optimal_{index_type}_params_recall_{TARGET_RECALL}.json"
    with open(output_file, 'w') as f:
        json.dump(all_optimal, f, indent=2)

    # Print summary
    print(f"\nSUMMARY - Optimal {param_label} for {index_type.upper()} (recall >= {TARGET_RECALL})")
    print(f"{'K':<8} {param_label:<12} {'Recall':<10} {'Time (ms)':<10}")
    print("-" * 40)
    for k in K_VALUES:
        if str(k) in all_optimal:
            r = all_optimal[str(k)]
            opt_val = r.get(f"optimal_{param_label}", "N/A")
            print(f"{k:<8} {str(opt_val):<12} {r['recall']:<10.4f} {r['avg_time_ms']:<10.2f}")

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
