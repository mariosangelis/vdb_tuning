#!/usr/bin/env python3
"""
Compare Default IVFFlat vs Parallel IVFFlat with varying worker counts.

For K=100, runs the GIST workload at multiple recall targets (using probes from
performance_recall_ivfflat_K_100.json) and measures execution time for:
  1. Default pgvector (~/vdb_tuning/postgres/postgres/pgvector)
  2. Parallel pgvector with NUM_WORKERS = 4, 8, 12
     (~/vdb_tuning/postgres/postgres/pgvector_modified_final)

For each worker count:
  - sed-replace #define NUM_WORKERS in ivfscan.c
  - Compile pgvector extension (make clean && make && make install)
  - Restart PostgreSQL (drop caches)
  - Drop and recreate the IVFFlat index
  - For each recall target: cold restart, set probes, run 1000 queries, measure time

Usage:
    sudo python3 run_comparison_workers.py
    sudo python3 run_comparison_workers.py --skip-default   # skip default, only run parallel variants
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
RECALL_DIR = PROJECT_DIR / "compute_and_compare_recall"

# === LOAD GLOBAL EXPERIMENT CONFIG ===
CONFIG_PATH = PROJECT_DIR / "TPCH" / "code" / "experiment_config.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
HW_CFG = CONFIG["hardware"]
PG_NUMA_NODES = HW_CFG["postgres_numa_nodes"]

# pgvector directories
PGVECTOR_DEFAULT = Path("/users/mangelis/vdb_tuning/postgres/postgres/pgvector")
PGVECTOR_PARALLEL = Path("/users/mangelis/vdb_tuning/postgres/postgres/pgvector_modified_final")

# The ivfscan.c file where NUM_WORKERS is defined
IVFSCAN_PATH = PGVECTOR_PARALLEL / "src" / "ivfscan.c"

# Worker counts to test
WORKER_COUNTS = [4, 8, 12]

# Workload
WORKLOAD_PATH = str(PROJECT_DIR / "sampled_vectors_gist.txt")

# Recall params file
RECALL_PARAMS_FILE = RECALL_DIR / "results_from_calculate_index_parameters_for_performance_recall_diagram" / "performance_recall_ivfflat_K_100.json"

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

K = 100

# Index definition
INDEX_NAME = "gist_data_embedding_idx"
# Load tuned lists
import json as _json2
_tune_file2 = Path(__file__).resolve().parent.parent / "compute_and_compare_recall" / "results_from_tune_scripts/tune_ivfflat_results.json"
_lists2 = 8192
if _tune_file2.exists():
    with open(_tune_file2) as _f2:
        _t2 = _json2.load(_f2)
    if "best_configuration" in _t2:
        _lists2 = _t2["best_configuration"]["lists"]
CREATE_INDEX_SQL = f"CREATE INDEX gist_data_embedding_idx ON gist_data USING ivfflat (embedding vector_l2_ops) WITH (lists = {_lists2})"

# PostgreSQL paths
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"


def wait_for_postgres(max_attempts=60, quiet=False):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            if not quiet and attempt % 10 == 0:
                print(f"  Waiting for PostgreSQL... attempt {attempt + 1}")
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def restart_postgres(quiet=False):
    """Restart PostgreSQL with cache clearing, pinned to NUMA node from config."""
    if not quiet:
        print(f"  Restarting PostgreSQL (clearing caches, NUMA={PG_NUMA_NODES})...")
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={PG_NUMA_NODES} --membind={PG_NUMA_NODES} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres(quiet=quiet)
    if not quiet:
        print("  PostgreSQL is ready.")


def set_num_workers(n):
    """Change #define NUM_WORKERS in ivfscan.c to specified value."""
    print(f"  Setting NUM_WORKERS = {n} in {IVFSCAN_PATH}...")
    result = subprocess.run(
        f"sed -i 's/#define NUM_WORKERS [0-9]*/#define NUM_WORKERS {n}/' {IVFSCAN_PATH}",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Failed to set NUM_WORKERS: {result.stderr}")

    # Verify
    verify = subprocess.run(
        f"grep '#define NUM_WORKERS' {IVFSCAN_PATH}",
        shell=True, capture_output=True, text=True
    )
    print(f"  Verified: {verify.stdout.strip()}")


def compile_pgvector(pgvector_dir):
    """Compile and install pgvector from given directory."""
    print(f"  Compiling pgvector from {pgvector_dir}...")
    env_cmd = "export PATH=/usr/local/pgsql/bin:$PATH && export PG_CONFIG=/usr/local/pgsql/bin/pg_config"
    result = subprocess.run(
        f"cd {pgvector_dir} && {env_cmd} && make clean > /dev/null 2>&1 && make CFLAGS='-O2 -fno-omit-frame-pointer -g' > /dev/null 2>&1 && make install > /dev/null 2>&1",
        shell=True
    )
    if result.returncode != 0:
        print("  WARNING: Build may have issues. Retrying with output...")
        subprocess.run(
            f"cd {pgvector_dir} && {env_cmd} && make clean && make CFLAGS='-O2 -fno-omit-frame-pointer -g' && make install",
            shell=True, check=True
        )
    print("  pgvector compiled and installed.")


def drop_and_create_index():
    """Drop all indexes on gist_data and create IVFFlat index."""
    print("  Dropping existing indexes and creating IVFFlat index...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop all indexes on gist_data
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data'")
    for (idx_name,) in cur.fetchall():
        print(f"    Dropping '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Create index
    print(f"    Creating '{INDEX_NAME}' (this may take a while)...")
    start = time.time()
    cur.execute(CREATE_INDEX_SQL)
    elapsed = time.time() - start
    print(f"    Index created in {elapsed:.1f}s")

    cur.close()
    conn.close()


def run_workload(workload, probes):
    """Run all queries with given probes setting, return total execution time."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SET ivfflat.probes = {probes}")

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

    cur.close()
    conn.close()
    return total_exec_time


def run_implementation(name, pgvector_dir, workload, recall_configs, num_workers=None, create_index=True):
    """Run full benchmark for one pgvector implementation.

    If num_workers is specified, set NUM_WORKERS before compiling.
    If create_index is False, skip index drop/create (reuse existing).
    """
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    if num_workers:
        print(f"  Workers: {num_workers}")
    print(f"{'='*60}")

    # Step 0: Set NUM_WORKERS if parallel
    if num_workers is not None:
        set_num_workers(num_workers)

    # Step 1: Compile pgvector
    compile_pgvector(pgvector_dir)

    # Step 2: Restart PostgreSQL after recompiling
    restart_postgres()

    # Step 3: Drop and create index (only if requested)
    if create_index:
        drop_and_create_index()
        # Restart PostgreSQL after creating index (clear caches)
        restart_postgres()

    # Step 5: Run workload for each recall target
    results = []
    print(f"\n  {'Target':<10} {'Probes':<10} {'Total Time (s)':<15} {'Avg Time (ms)':<15}")
    print(f"  {'-'*50}")

    for config in recall_configs:
        target = config["target_recall"]
        probes = config["optimal_probes"]

        # Restart and clear caches before each run (cold run, no output)
        restart_postgres(quiet=True)

        total_time = run_workload(workload, probes)
        avg_time_ms = (total_time / len(workload)) * 1000

        results.append({
            "target_recall": target,
            "probes": probes,
            "total_time_s": round(total_time, 4),
            "avg_time_ms": round(avg_time_ms, 2)
        })

        print(f"  {target:<10} {probes:<10} {total_time:<15.4f} {avg_time_ms:<15.2f}")

    return results


def main():
    skip_default = "--skip-default" in sys.argv
    no_rebuild = "--no-rebuild" in sys.argv

    print("Parallel IVFFlat vs Default IVFFlat Comparison (Multiple Worker Counts)")
    print(f"K = {K}")
    print(f"Worker counts to test: {WORKER_COUNTS}")
    print(f"Workload: {WORKLOAD_PATH}")
    if skip_default:
        print("SKIPPING default implementation (--skip-default)")

    # Load recall params
    if not RECALL_PARAMS_FILE.exists():
        print(f"ERROR: {RECALL_PARAMS_FILE} not found.")
        print("Run calculate_index_parameters_for_performance_recall_diagram.py first.")
        sys.exit(1)

    with open(RECALL_PARAMS_FILE, 'r') as f:
        recall_data = json.load(f)

    recall_configs = recall_data["results"]
    print(f"Recall targets: {[r['target_recall'] for r in recall_configs]}")
    print(f"Probes:         {[r['optimal_probes'] for r in recall_configs]}")

    # Load workload
    print(f"\nLoading workload...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Output file (save incrementally)
    output_file = SCRIPT_DIR / "comparison_results_workers_K_100.json"

    # Load existing results if any (for resuming)
    output = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            output = json.load(f)
        print(f"Loaded existing results from {output_file}")

    output["k"] = K
    output["workload_size"] = len(workload)
    output["worker_counts_tested"] = WORKER_COUNTS

    # Run default implementation (only once)
    if not skip_default and "default" not in output:
        default_results = run_implementation(
            "Default IVFFlat (pgvector)",
            PGVECTOR_DEFAULT,
            workload,
            recall_configs
        )
        output["default"] = default_results
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nDefault results saved to {output_file}")
    elif "default" in output:
        print("\nDefault results already exist, skipping.")
        default_results = output["default"]
    else:
        default_results = output.get("default", None)

    # Run parallel implementation for each worker count
    if "parallel_by_workers" not in output:
        output["parallel_by_workers"] = {}

    # For parallel: create index only on first worker count, then reuse
    first_parallel = not no_rebuild  # if --no-rebuild, never create index
    for num_workers in WORKER_COUNTS:
        key = str(num_workers)
        if key in output["parallel_by_workers"]:
            print(f"\nParallel (workers={num_workers}) already done, skipping.")
            if not no_rebuild:
                first_parallel = False
            continue




        parallel_results = run_implementation(
            f"Parallel IVFFlat (workers={num_workers})",
            PGVECTOR_PARALLEL,
            workload,
            recall_configs,
            num_workers=num_workers,
            create_index=(first_parallel and not no_rebuild)
        )
        if not no_rebuild:
            first_parallel = False
        output["parallel_by_workers"][key] = parallel_results

        # Save incrementally after each worker count
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults for workers={num_workers} saved to {output_file}")

    # Print comparison summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY (K={K})")
    print(f"{'='*80}")

    header = f"{'Target':<10} {'Probes':<8} {'Default (ms)':<14}"
    for w in WORKER_COUNTS:
        header += f" {'W=' + str(w) + ' (ms)':<14}"
    print(header)
    print("-" * (32 + 14 * len(WORKER_COUNTS)))

    if default_results:
        for i, d in enumerate(default_results):
            target = d["target_recall"]
            probes = d["probes"]
            d_time = d["avg_time_ms"]
            line = f"{target:<10} {probes:<8} {d_time:<14.2f}"

            for w in WORKER_COUNTS:
                key = str(w)
                if key in output["parallel_by_workers"]:
                    p_time = output["parallel_by_workers"][key][i]["avg_time_ms"]
                    speedup = d_time / p_time if p_time > 0 else 0
                    line += f" {p_time:<7.2f}({speedup:.1f}x) "
                else:
                    line += f" {'N/A':<14}"
            print(line)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
