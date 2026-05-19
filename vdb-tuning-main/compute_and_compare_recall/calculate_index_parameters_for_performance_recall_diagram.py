#!/usr/bin/env python3
"""
Calculate optimal probes/ef_search for a single K at multiple recall targets.
Used for performance-vs-recall diagrams.

Creates the required index if it does not exist (drops conflicting indexes).

Usage:
    python3 calculate_index_parameters_for_performance_recall_diagram.py ivfflat 100 0.5,0.6,0.7,0.8,0.9
    python3 calculate_index_parameters_for_performance_recall_diagram.py hnsw 100 0.5,0.6,0.7,0.8,0.9
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


def tune_for_target(index_type, k, workload, exact_results, param_list, target_recall):
    """Find minimum parameter value that achieves target recall for a given K."""
    param_name = "ivfflat.probes" if index_type == "ivfflat" else "hnsw.ef_search"
    param_label = "probes" if index_type == "ivfflat" else "ef_search"

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
            break

    if best_param is None:
        if all_results:
            best_entry = max(all_results, key=lambda x: x["recall"])
            best_param = best_entry[param_label]
            best_recall = best_entry["recall"]
            best_time = best_entry["avg_time_ms"]

    return {
        "target_recall": target_recall,
        "optimal_" + param_label: best_param,
        "actual_recall": best_recall,
        "avg_time_ms": best_time,
        "all_tested": all_results
    }


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("ivfflat", "hnsw"):
        print("Usage: python3 calculate_index_parameters_for_performance_recall_diagram.py <ivfflat|hnsw> <K> <recall_targets>")
        print("  K: single integer (e.g. 100)")
        print("  recall_targets: comma-separated list (e.g. 0.5,0.6,0.7,0.8,0.9)")
        print("  Examples:")
        print("    python3 calculate_index_parameters_for_performance_recall_diagram.py ivfflat 100 0.5,0.6,0.7,0.8,0.9")
        print("    python3 calculate_index_parameters_for_performance_recall_diagram.py hnsw 100 0.5,0.6,0.7,0.8,0.9")
        sys.exit(1)

    index_type = sys.argv[1]
    k = int(sys.argv[2])
    target_recalls = sorted([float(x) for x in sys.argv[3].split(',')])
    param_list = IVFFLAT_PROBES_LIST if index_type == "ivfflat" else HNSW_EF_SEARCH_LIST
    param_label = "probes" if index_type == "ivfflat" else "ef_search"

    print(f"Index type: {index_type.upper()}")
    print(f"K: {k}")
    print(f"Target recalls: {target_recalls}")

    # Load workload
    print(f"\nLoading workload from {WORKLOAD_PATH}...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Load exact results for this K
    exact_file = EXACT_RESULTS_DIR / f"exact_search_results_1000_queries_gist_K_{k}.json"
    if not exact_file.exists():
        print(f"ERROR: Exact results file not found: {exact_file}")
        sys.exit(1)

    with open(exact_file, 'r') as f:
        exact_results = json.load(f)
    print(f"Loaded exact results for K={k}")

    # Ensure index exists
    ensure_index(index_type)

    # Restart PostgreSQL
    restart_postgres()

    # Tune for each target recall
    results = {
        "index_type": index_type,
        "k": k,
        "results": []
    }

    print(f"\n{'='*60}")
    print(f"Tuning {index_type.upper()} for K={k}")
    print(f"{'='*60}")
    print(f"{'Target':<10} {param_label:<12} {'Actual Recall':<15} {'Time (ms)':<10}")
    print("-" * 47)

    for target in target_recalls:
        result = tune_for_target(index_type, k, workload, exact_results, param_list, target)
        results["results"].append(result)

        opt_val = result.get(f"optimal_{param_label}", "N/A")
        actual = result["actual_recall"]
        t_ms = result["avg_time_ms"]
        status = "OK" if actual >= target else "BELOW TARGET"
        print(f"{target:<10} {str(opt_val):<12} {actual:<15.4f} {t_ms:<10.2f} {status}")

    # Save results
    output_file = SCRIPT_DIR / f"performance_recall_{index_type}_K_{k}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
