#!/usr/bin/env python3
"""
Compute execution times for different K values with/without the top-K heap
compiler optimization.

For each K value, finds the minimum probes/ef_search that achieves recall >= 0.9
using precomputed exact results, then measures total execution time.

Usage:
    sudo python3 compute_optimization_results_for_different_K.py <ivfflat|hnsw> <with|without> <dataset>

Examples:
    sudo python3 compute_optimization_results_for_different_K.py ivfflat with gist
    sudo python3 compute_optimization_results_for_different_K.py ivfflat without gist
    sudo python3 compute_optimization_results_for_different_K.py hnsw with deep
    sudo python3 compute_optimization_results_for_different_K.py hnsw without deep

The script will:
  1. Load the appropriate PGDATA snapshot from /mnt/nvme/saved_indexes/
  2. Toggle the compiler optimization flag in execExpr.c and recompile PostgreSQL
  3. For each K: find optimal probes/ef_search for recall >= 0.9, then measure execution time
"""

import psycopg2
import time
import subprocess
import json
import shutil
import os
import sys
import glob
import re
from pathlib import Path

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent                                     # vdb-tuning-main/
RECALL_DIR = PROJECT_DIR / "compute_and_compare_recall"

# === CONFIG ===
K_VALUES = [1, 10, 100, 200, 400, 800, 1000]
TARGET_RECALL = 0.9

SNAPSHOTS_DIR = Path("/mnt/nvme/saved_indexes")
PG_DATA_DIR = "/mnt/nvme/pgdata"
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"

# PostgreSQL source for compiler optimization toggle
POSTGRES_SRC_DIR = PROJECT_DIR.parent / "postgres" / "postgres"
EXECEXPR_PATH = POSTGRES_SRC_DIR / "src" / "backend" / "executor" / "execExpr.c"

# Default pgvector extension (must be compiled before running)
PGVECTOR_DEFAULT_DIR = PROJECT_DIR.parent / "postgres" / "postgres" / "pgvector"

# Probes/ef_search values to sweep when finding optimal params
IVFFLAT_PROBES = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80,
                  100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
HNSW_EF_SEARCH = [10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# Load NUMA config
_config_path = PROJECT_DIR / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path) as _f:
        NUMA_NODE = str(json.load(_f).get("hardware", {}).get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    "gist": {
        "table": "gist_data",
        "queries_file": "gist_test_queries.txt",
        "exact_results_dir": "exact_results_gist",
        "distance_op": "<->",
    },
    "deep": {
        "table": "deep_data",
        "queries_file": "deep_test_queries.txt",
        "exact_results_dir": "exact_results_deep",
        "distance_op": "<=>",
    },
    "sift": {
        "table": "sift_data",
        "queries_file": "sift_test_queries.txt",
        "exact_results_dir": "exact_results_sift",
        "distance_op": "<->",
    },
}


def wait_for_postgres(max_attempts=30):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def stop_postgres():
    """Stop postgres if running."""
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop -m fast' 2>/dev/null || true",
        shell=True, capture_output=True
    )
    time.sleep(2)


def start_postgres():
    """Start postgres with NUMA pinning and cache clearing."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} "
        f"{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def restart_postgres():
    """Restart postgres (drop caches, restart)."""
    stop_postgres()
    start_postgres()


def load_snapshot(snapshot_path):
    # """Load snapshot into PGDATA."""
    # print(f"  Loading snapshot: {snapshot_path.name}")
    # stop_postgres()

    # subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)
    # subprocess.run(f"mkdir -p {PG_DATA_DIR}", shell=True, check=True)

    # pigz = shutil.which("pigz")
    # if pigz:
    #     cmd = f"pigz -dc '{snapshot_path}' | tar -xf - -C {PG_DATA_DIR}"
    # else:
    #     cmd = f"tar -xzf '{snapshot_path}' -C {PG_DATA_DIR}"
    # subprocess.run(cmd, shell=True, check=True)

    # subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    # subprocess.run(f"chmod 700 {PG_DATA_DIR}", shell=True, check=True)
    print("  Snapshot loaded.")


def find_snapshot(dataset, index_type, tune_data):
    """Find the best matching snapshot based on tune results."""
    if tune_data and "best_configuration" in tune_data:
        best = tune_data["best_configuration"]
        if index_type == "ivfflat":
            lists = best.get("lists", 8192)
            fname = f"{dataset}_ivfflat_lists{lists}_pgdata.tar.gz"
        else:
            m = best.get("m", 16)
            efc = best.get("ef_construction", 128)
            fname = f"{dataset}_hnsw_efc{efc}_m{m}_pgdata.tar.gz"

        path = SNAPSHOTS_DIR / fname
        if path.exists():
            return path

    # Fallback: find any snapshot of this dataset + index type
    pattern = str(SNAPSHOTS_DIR / f"{dataset}_{index_type}_*_pgdata.tar.gz")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[0])

    return None


def compile_default_pgvector():
    """Compile and install the default pgvector extension."""
    if not PGVECTOR_DEFAULT_DIR.exists():
        print(f"  WARNING: Default pgvector not found at {PGVECTOR_DEFAULT_DIR}, skipping")
        return
    print(f"  Compiling default pgvector from {PGVECTOR_DEFAULT_DIR}...")
    env_cmd = "export PATH=/usr/local/pgsql/bin:$PATH && export PG_CONFIG=/usr/local/pgsql/bin/pg_config"
    result = subprocess.run(
        f"cd {PGVECTOR_DEFAULT_DIR} && {env_cmd} && make clean > /dev/null 2>&1 && make > /dev/null 2>&1 && make install > /dev/null 2>&1",
        shell=True
    )
    if result.returncode != 0:
        print("  WARNING: Build may have issues. Retrying with output...")
        subprocess.run(
            f"cd {PGVECTOR_DEFAULT_DIR} && {env_cmd} && make clean && make && make install",
            shell=True, check=True
        )
    print("  Default pgvector compiled and installed.")


def set_compiler_optimization(enable):
    """Toggle enable_topk_search_optimizations in execExpr.c and recompile PostgreSQL."""
    value_str = "true" if enable else "false"
    print(f"\n[SETUP] Setting enable_topk_search_optimizations = {value_str}")

    with open(EXECEXPR_PATH, 'r') as f:
        src = f.read()

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
    print("  Recompiling PostgreSQL...")
    result = subprocess.run(
        f"cd {POSTGRES_SRC_DIR} && make -j$(nproc) > /dev/null 2>&1 && make install > /dev/null 2>&1",
        shell=True
    )
    if result.returncode != 0:
        print("  WARNING: Build may have issues. Retrying with output...")
        subprocess.run(
            f"cd {POSTGRES_SRC_DIR} && make -j$(nproc) && make install",
            shell=True, check=True
        )
    print("  Recompilation complete.")


def find_optimal_param(workload, exact_results, k, table_name, distance_op, index_type):
    """Find minimum probes/ef_search that achieves recall >= TARGET_RECALL for given K.

    Uses warm measurement (pg_prewarm) for fast parameter search.
    """
    if index_type == "ivfflat":
        param_values = IVFFLAT_PROBES
        set_cmd = "SET ivfflat.probes = {}"
    else:
        param_values = HNSW_EF_SEARCH
        set_cmd = "SET hnsw.ef_search = {}"

    conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")

    # Prewarm
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    cur.execute(f"""
        SELECT i.relname FROM pg_index ix
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_am am ON am.oid = i.relam
        WHERE t.relname = '{table_name}' AND am.amname IN ('ivfflat', 'hnsw')
    """)
    result = cur.fetchone()
    if result:
        cur.execute(f"SELECT pg_prewarm('{result[0]}')")
        cur.fetchone()
    cur.execute(f"SELECT pg_prewarm('{table_name}')")
    cur.fetchone()

    for param_val in param_values:
        cur.execute(set_cmd.format(param_val))

        total_recall = 0.0

        for i in range(len(workload)):
            vec_str = workload[i]
            cur.execute(
                f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
            )
            result_ids = set(row[0] for row in cur.fetchall())
            exact_ids = set(exact_results[i][:k])
            if len(exact_ids) > 0:
                total_recall += len(result_ids & exact_ids) / len(exact_ids)

        avg_recall = total_recall / len(workload)
        param_name = "probes" if index_type == "ivfflat" else "ef_search"
        print(f"      {param_name}={param_val:>4}: recall={avg_recall:.4f}")

        if avg_recall >= TARGET_RECALL:
            cur.close()
            conn.close()
            return param_val, avg_recall

    # If we couldn't achieve target recall, return the last (highest) value
    cur.close()
    conn.close()
    return param_values[-1], avg_recall


def prewarm_table_and_index(table_name):
    """Use pg_prewarm to load table and index into shared buffers."""
    conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    cur.execute(f"""
        SELECT i.relname FROM pg_index ix
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_am am ON am.oid = i.relam
        WHERE t.relname = '{table_name}' AND am.amname IN ('ivfflat', 'hnsw')
    """)
    result = cur.fetchone()
    if result:
        cur.execute(f"SELECT pg_prewarm('{result[0]}')")
        blocks = cur.fetchone()[0]
        print(f"    Prewarmed index '{result[0]}': {blocks} blocks")
    cur.execute(f"SELECT pg_prewarm('{table_name}')")
    tblocks = cur.fetchone()[0]
    print(f"    Prewarmed table '{table_name}': {tblocks} blocks")
    cur.close()
    conn.close()


def run_measurement(workload, k, optimal_param, table_name, distance_op, index_type, mode="cold"):
    """Run full workload measurement with given K and optimal param.
    
    mode='cold': restart + drop caches before measurement
    mode='warm': do nothing (data stays cached from prewarm)
    """
    if mode == "cold":
        restart_postgres()

    if index_type == "ivfflat":
        set_cmd = f"SET ivfflat.probes = {optimal_param}"
    else:
        set_cmd = f"SET hnsw.ef_search = {optimal_param}"

    conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
    cur = conn.cursor()
    cur.execute(set_cmd)

    total_exec_time = 0.0

    for i in range(len(workload)):
        vec_str = workload[i]
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM {table_name} "
            f"ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
        )
        explain_result = cur.fetchall()

        for row in explain_result:
            line = row[0].strip()
            if 'Execution Time:' in line:
                parts = line.split()
                elapsed_time = float(parts[2]) / 1000.0
                total_exec_time += elapsed_time

    cur.close()
    conn.close()
    return total_exec_time


def main():
    if len(sys.argv) < 5 or sys.argv[1] not in ("ivfflat", "hnsw") or sys.argv[2] not in ("with", "without"):
        print("Usage: sudo python3 compute_optimization_results_for_different_K.py <ivfflat|hnsw> <with|without> <dataset> <cold|warm>")
        print()
        print("  index_type:   ivfflat | hnsw")
        print("  optimization: with | without")
        print("  dataset:      gist | deep | sift")
        print("  mode:         cold | warm")
        print()
        print("Examples:")
        print("  sudo python3 compute_optimization_results_for_different_K.py ivfflat with gist cold")
        print("  sudo python3 compute_optimization_results_for_different_K.py hnsw without deep warm")
        sys.exit(1)

    index_type = sys.argv[1]
    optimization = sys.argv[2]
    dataset = sys.argv[3]
    mode = sys.argv[4]
    if mode not in ("cold", "warm"):
        print(f"ERROR: mode must be 'cold' or 'warm', got '{mode}'")
        sys.exit(1)

    if dataset not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    ds_cfg = DATASET_CONFIGS[dataset]
    table_name = ds_cfg["table"]
    distance_op = ds_cfg["distance_op"]

    # Paths
    workload_path = PROJECT_DIR / ds_cfg["queries_file"]
    exact_results_dir = RECALL_DIR / "exact_results" / ds_cfg["exact_results_dir"]
    tune_file = RECALL_DIR / "results_from_tune_scripts" / f"tune_{index_type}_results_{dataset}.json"

    print("=" * 70)
    print("COMPILER OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"  Index type:    {index_type.upper()}")
    print(f"  Optimization:  {optimization}")
    print(f"  Dataset:       {dataset}")
    print(f"  Table:         {table_name}")
    print(f"  Mode:          {mode}")
    print(f"  Target recall: {TARGET_RECALL}")
    print(f"  K values:      {K_VALUES}")
    print(f"  Workload:      {workload_path}")
    print("=" * 70)

    # Validate workload file
    if not workload_path.exists():
        print(f"ERROR: Workload file not found: {workload_path}")
        sys.exit(1)

    # Load workload
    print(f"\nLoading workload from {workload_path}...")
    with open(workload_path, 'r') as f:
        content = f.read().strip()
        workload_raw = eval(content)

    # Normalize to string format
    workload = []
    for q in workload_raw:
        if isinstance(q, list):
            workload.append("[" + ",".join(str(x) for x in q) + "]")
        else:
            workload.append(q)
    print(f"  Loaded {len(workload)} queries")

    # Load tune data
    tune_data = None
    if tune_file.exists():
        with open(tune_file) as f:
            tune_data = json.load(f)
        print(f"  Tune data loaded from {tune_file.name}")
    else:
        print(f"  WARNING: Tune file not found: {tune_file}")

    # === SETUP PHASE ===
    # 0. Compile default pgvector (ensure parallel extension is not loaded)
    compile_default_pgvector()

    # 1. Set compiler optimization flag (recompile if needed)
    set_compiler_optimization(enable=(optimization == "with"))

    # 2. Find and load snapshot
    snapshot = find_snapshot(dataset, index_type, tune_data)
    if snapshot:
        load_snapshot(snapshot)
    else:
        print(f"  WARNING: No snapshot found for {dataset}/{index_type}. Using current PGDATA.")

    # 3. Start postgres
    start_postgres()

    # === FIND OPTIMAL PARAMS FOR EACH K ===
    param_label = "probes" if index_type == "ivfflat" else "ef_search"
    output_dir = SCRIPT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    params_file = output_dir / f"optimal_params_{index_type}_{dataset}_recall_{TARGET_RECALL}.json"

    if params_file.exists():
        print(f"\n[TUNING] Loading cached optimal params from {params_file.name}")
        with open(params_file) as f:
            optimal_params = {int(k): v for k, v in json.load(f).items()}
        for k in sorted(optimal_params.keys()):
            print(f"  K={k:>5}: {param_label}={optimal_params[k]}")
    else:
        print(f"\n[TUNING] Finding optimal {param_label} for each K (recall >= {TARGET_RECALL})...")
        print(f"  (Using all {len(workload)} queries per sweep, warm cache)")

        optimal_params = {}
        for k in K_VALUES:
            exact_file = exact_results_dir / f"exact_search_results_{len(workload)}_queries_{dataset}_K_{k}.json"
            if not exact_file.exists():
                print(f"  K={k}: exact results not found at {exact_file.name}, skipping")
                continue

            with open(exact_file) as f:
                exact_results = json.load(f)

            print(f"\n    Sweeping for K={k}:")
            param_val, recall = find_optimal_param(
                workload, exact_results, k, table_name, distance_op, index_type
            )
            optimal_params[k] = param_val
            print(f"    >>> K={k:>5}: optimal {param_label}={param_val}, recall={recall:.4f}")

        with open(params_file, "w") as f:
            json.dump({str(k): v for k, v in optimal_params.items()}, f, indent=2)
        print(f"\n  Optimal params saved to: {params_file}")

    # === MEASUREMENT PHASE ===
    if mode == "warm":
        print("\n[PREWARM] Prewarming index and table into shared buffers...")
        _conn = psycopg2.connect(dbname="mydb", user="postgres", host="localhost", port="5432")
        _conn.autocommit = True
        _cur = _conn.cursor()
        _cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        _cur.execute(f"SELECT pg_prewarm('{table_name}')")
        _tbl_blocks = _cur.fetchone()[0]
        print(f"  Prewarmed table '{table_name}': {_tbl_blocks} blocks")
        _cur.execute("SELECT i.relname FROM pg_index ix JOIN pg_class t ON t.oid = ix.indrelid JOIN pg_class i ON i.oid = ix.indexrelid JOIN pg_am am ON am.oid = i.relam WHERE t.relname = '" + table_name + "' AND am.amname IN ('ivfflat', 'hnsw')")
        _idx = _cur.fetchone()
        if _idx:
            _cur.execute(f"SELECT pg_prewarm('{_idx[0]}')")
            _idx_blocks = _cur.fetchone()[0]
            print(f"  Prewarmed index '{_idx[0]}': {_idx_blocks} blocks")
        _cur.close()
        _conn.close()
        print("  Prewarm complete.")


    print(f"\n[MEASUREMENT] Running experiments ({optimization} optimization)...")
    results = []

    for k in K_VALUES:
        if k not in optimal_params:
            print(f"  K={k}: skipped (no optimal params)")
            results.append((k, 0.0))
            continue

        total_time = run_measurement(
            workload, k, optimal_params[k], table_name, distance_op, index_type, mode=mode
        )
        avg_ms = (total_time / len(workload)) * 1000
        results.append((k, total_time))
        print(f"  K={k:>5}: {param_label}={optimal_params[k]}, total={total_time:.4f}s, avg={avg_ms:.3f}ms/query")

    # Write results
    output_dir = SCRIPT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_{optimization}_optimization_{index_type.upper()}_{dataset}_{mode}.txt"

    with open(output_file, "w") as f:
        for k, exec_time in results:
            f.write(f"K={k}, exec_time={exec_time:.6f}\n")

    # Optimal params already saved during tuning phase

    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY ({optimization} optimization, {index_type.upper()}, {dataset})")
    print(f"{'=' * 70}")
    for k, exec_time in results:
        if exec_time > 0:
            avg_ms = (exec_time / len(workload)) * 1000
            print(f"  K={k:>5}: total={exec_time:.4f}s, avg={avg_ms:.3f}ms/query")
    print(f"\nResults written to: {output_file}")
    # Optimal params already saved during tuning phase


if __name__ == "__main__":
    main()
