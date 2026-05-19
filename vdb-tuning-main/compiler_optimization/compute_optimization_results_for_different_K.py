#!/usr/bin/env python3
"""
Compute execution times for different K values using the optimal probe/ef_search
settings from the recall tuning results.

Usage:
    python3 compute_optimization_results_for_different_K.py ivfflat with
    python3 compute_optimization_results_for_different_K.py ivfflat without
    python3 compute_optimization_results_for_different_K.py hnsw with
    python3 compute_optimization_results_for_different_K.py hnsw without
"""

import psycopg2
import time
import subprocess
import json
import shutil
import os
import sys
from pathlib import Path

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RECALL_DIR = PROJECT_DIR / "compute_and_compare_recall"

# === CONFIG ===
K_VALUES = [1, 10, 100, 200, 400, 800, 1000]
RUNS = 1

# Workload: sampled_vectors_gist.txt (entries are pgvector-format strings)
WORKLOAD_PATH = str(PROJECT_DIR / "sampled_vectors_gist.txt")

PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"

# PostgreSQL source for compiler optimization toggle
POSTGRES_SRC_DIR = PROJECT_DIR.parent / "postgres" / "postgres"
EXECEXPR_PATH = POSTGRES_SRC_DIR / "src" / "backend" / "executor" / "execExpr.c"


def wait_for_postgres(max_attempts=30):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                dbname="mydb", user="postgres", host="localhost", port="5432"
            )
            conn.close()
            print("PostgreSQL is ready.")
            return True
        except psycopg2.OperationalError:
            print(f"Waiting for PostgreSQL... attempt {attempt + 1}")
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def restart_postgres():
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind=0-7 --membind=0-7 {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True,
        check=True
    )
    time.sleep(3)
    wait_for_postgres()


def set_compiler_optimization(enable):
    """Toggle enable_topk_search_optimizations in execExpr.c and recompile PostgreSQL."""
    value_str = "true" if enable else "false"
    print(f"Setting enable_topk_search_optimizations = {value_str} in {EXECEXPR_PATH}")



    with open(EXECEXPR_PATH, 'r') as f:
        src = f.read()

    # Replace the flag value
    import re
    new_src = re.sub(
        r'bool enable_topk_search_optimizations = (true|false);',
        f'bool enable_topk_search_optimizations = {value_str};',
        src
    )

    if new_src == src:
        print("  Flag already set correctly, skipping recompile.")
        return

    with open(EXECEXPR_PATH, 'w') as f:
        f.write(new_src)

    # Verify
    with open(EXECEXPR_PATH, 'r') as f:
        verify = f.read()
    match = re.search(r'bool enable_topk_search_optimizations = (true|false);', verify)
    print(f"  Verified: {match.group() if match else 'NOT FOUND'}")

    # Recompile PostgreSQL
    print("  Recompiling PostgreSQL (make clean && make && make install)...")
    result = subprocess.run(
        f"cd {POSTGRES_SRC_DIR} && make clean -j16 > /dev/null 2>&1 && make -j16 > /dev/null 2>&1 && make install > /dev/null 2>&1",
        shell=True
    )
    if result.returncode != 0:
        # Retry with output visible
        print("  WARNING: Build may have issues. Retrying with output...")
        subprocess.run(
            f"cd {POSTGRES_SRC_DIR} && make -j16 && make install",
            shell=True, check=True
        )
    print("  Recompilation complete.")


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


def ensure_index(index_type):
    """Check if the required index exists, create it if not."""
    idx_def = INDEX_DEFS[index_type]

    conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
    conn.autocommit = True
    cur = conn.cursor()

    # Check if the index already exists
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'gist_data' AND indexname = %s",
        (idx_def["name"],)
    )
    result = cur.fetchone()

    if result:
        print(f"Index '{idx_def['name']}' already exists.")
        cur.close()
        conn.close()
        return

    # Drop any other indexes on gist_data first
    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data'"
    )
    existing = cur.fetchall()
    for (idx_name,) in existing:
        print(f"Dropping existing index '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Create the index
    print(f"Index '{idx_def['name']}' not found. Creating it (this may take a while)...")
    start = time.time()
    cur.execute(idx_def["create_sql"])
    elapsed = time.time() - start
    print(f"Index created in {elapsed:.1f}s")

    cur.close()
    conn.close()


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("ivfflat", "hnsw") or sys.argv[2] not in ("with", "without"):
        print("Usage: python3 compute_optimization_results_for_different_K.py <ivfflat|hnsw> <with|without>")
        print("  with    = with compiler optimization")
        print("  without = without compiler optimization")
        sys.exit(1)

    index_type = sys.argv[1]
    optimization = sys.argv[2]

    # Load optimal params from recall tuning
    optimal_params_file = RECALL_DIR / f"optimal_{index_type}_params.json"
    if not optimal_params_file.exists():
        print(f"ERROR: {optimal_params_file} not found.")
        print(f"Run tune_recall.py {index_type} first.")
        sys.exit(1)

    with open(optimal_params_file, 'r') as f:
        optimal_params = json.load(f)

    param_label = "probes" if index_type == "ivfflat" else "ef_search"
    set_cmd = "SET ivfflat.probes = {}" if index_type == "ivfflat" else "SET hnsw.ef_search = {}"

    print(f"Index type: {index_type.upper()}")
    print(f"Optimal params loaded from: {optimal_params_file}")
    for k in K_VALUES:
        k_str = str(k)
        if k_str in optimal_params:
            val = optimal_params[k_str].get(f"optimal_{param_label}", "N/A")
            print(f"  K={k}: {param_label}={val}")
        else:
            print(f"  K={k}: NOT FOUND in optimal params, will skip")

    # Load workload
    print(f"\nLoading workload from {WORKLOAD_PATH}...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Output file
    output_file = SCRIPT_DIR / "data" / f"results_{optimization}_optimization_{index_type.upper()}.txt"
    os.makedirs(SCRIPT_DIR / "data", exist_ok=True)

    # Set compiler optimization flag and recompile if needed
    set_compiler_optimization(enable=(optimization == "with"))

    # Ensure the required index exists (create if needed)
    ensure_index(index_type)

    execution_times = []

    for K in K_VALUES:
        k_str = str(K)
        if k_str not in optimal_params:
            print(f"\nSkipping K={K} (no optimal params)")
            execution_times.append(0.0)
            continue

        optimal_value = optimal_params[k_str][f"optimal_{param_label}"]
        print(f"\nRunning K={K} with {param_label}={optimal_value}")

        for run in range(RUNS):
            restart_postgres()

            conn = psycopg2.connect(
                dbname="mydb", user="postgres", host="localhost", port="5432"
            )
            cur = conn.cursor()

            # Set the optimal tuning parameter
            cur.execute(set_cmd.format(optimal_value))

            total_exec_time = 0.0

            for i in range(len(workload)):
                vec_str = workload[i]
                cur.execute(
                    f"EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM gist_data ORDER BY embedding <-> '{vec_str}'::vector LIMIT {K}"
                )
                explain_result = cur.fetchall()

                for row in explain_result:
                    line = row[0].strip()
                    if 'Execution Time:' in line:
                        parts = line.split()
                        elapsed_time = float(parts[2]) / 1000.0
                        total_exec_time += elapsed_time

            execution_times.append(total_exec_time)
            print(f"  K={K}, Total Exec Time = {total_exec_time:.4f}s")

            cur.close()
            conn.close()

    # Write results
    with open(output_file, "w") as f:
        for k, exec_time in zip(K_VALUES, execution_times):
            f.write(f"K={k}, exec_time={exec_time:.6f}\n")

    print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()
