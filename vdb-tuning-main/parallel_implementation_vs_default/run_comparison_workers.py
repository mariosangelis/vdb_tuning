#!/usr/bin/env python3
"""
Compare Default IVFFlat vs Parallel IVFFlat with varying worker counts.

For K=100, runs the workload at multiple recall targets and measures execution time for:
  1. Default pgvector (~/vdb_tuning/postgres/postgres/pgvector)
  2. Parallel pgvector with NUM_WORKERS = 4, 8, 12
     (~/vdb_tuning/postgres/postgres/pgvector_modified_final)

Probe values for each recall target are loaded from the dataset-specific
recall_qps results file (results_recall_qps/recall_qps_{dataset}_ivfflat_cold.json).

Supports --mode cold (restart + drop caches between each probe) and
--mode warm (prewarm index+table, no restart between probes).

Usage:
    sudo python3 run_comparison_workers.py --dataset gist --mode cold
    sudo python3 run_comparison_workers.py --dataset gist --mode warm
    sudo python3 run_comparison_workers.py --dataset deep --mode cold --skip-default
    sudo python3 run_comparison_workers.py --dataset gist --mode warm --no-rebuild
"""

import psycopg2
import time
import subprocess
import json
import sys
import os
import shutil
import argparse
from pathlib import Path

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RECALL_DIR = PROJECT_DIR / "compute_and_compare_recall"

# Central machine-specific config (paths) from config.yaml at the repo root.
sys.path.insert(0, str(PROJECT_DIR.parent))  # config_loader lives at ~/vdb_tuning (repo root)
from config_loader import get_path  # noqa: E402

# === LOAD GLOBAL EXPERIMENT CONFIG ===
CONFIG_PATH = PROJECT_DIR / "TPCH" / "code" / "experiment_config.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
HW_CFG = CONFIG["hardware"]
PG_NUMA_NODES = HW_CFG["postgres_numa_nodes"]

# pgvector directories (from config.yaml)
PGVECTOR_DEFAULT = Path(get_path("pgvector_default"))
PGVECTOR_PARALLEL = Path(get_path("pgvector_parallel"))

# The ivfscan.c file where NUM_WORKERS is defined
IVFSCAN_PATH = PGVECTOR_PARALLEL / "src" / "ivfscan.c"

# Snapshot directory (from config.yaml)
SNAPSHOT_DIR = Path(get_path("snapshot_dir"))

# Worker counts to test
WORKER_COUNTS = [4, 8, 12]

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    "gist": {
        "table": "gist_data",
        "queries_file": "gist_test_queries.txt",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
    "deep": {
        "table": "deep_data",
        "queries_file": "deep_test_queries.txt",
        "distance_op": "<=>",
        "index_ops": "vector_cosine_ops",
    },
    "sift": {
        "table": "sift_data",
        "queries_file": "sift_test_queries.txt",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
}

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

K = 100

# PostgreSQL paths (from config.yaml)
PG_INSTALL = get_path("pg_install")
PG_CTL_PATH = shutil.which("pg_ctl") or f"{PG_INSTALL}/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or get_path("pg_data")


def get_snapshot_path(dataset, index_type, params):
    """Get path for a PGDATA snapshot tar.gz. params is a dict like {'lists': 1024}."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    fname = f"{dataset}_{index_type}_{param_str}_pgdata.tar.gz"
    return SNAPSHOT_DIR / fname


def load_pgdata_snapshot(snapshot_path):
    """Stop postgres, restore PGDATA from tar, restart postgres."""
    print(f"    Loading PGDATA snapshot from {snapshot_path}...")

    # Stop postgres
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop'",
        shell=True, capture_output=True
    )
    time.sleep(2)

    # Clear PGDATA and extract
    subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)

    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"pigz -dc {snapshot_path} | tar -xf - -C {PG_DATA_DIR}"
    else:
        cmd = f"tar -xzf {snapshot_path} -C {PG_DATA_DIR}"

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    # Fix permissions
    subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    subprocess.run(f"chmod 700 {PG_DATA_DIR}", shell=True, check=True)

    print(f"    Snapshot restored in {elapsed:.1f}s")

    # Start postgres pinned to NUMA node
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={PG_NUMA_NODES} --membind={PG_NUMA_NODES} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


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


def prewarm_table_and_index(table_name):
    """Use pg_prewarm to load entire table and index into shared buffers."""
    print(f"  Prewarming table and index for '{table_name}'...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")

        # Prewarm the vector index
        cur.execute(f"""
            SELECT i.relname FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = '{table_name}' AND am.amname IN ('ivfflat', 'hnsw')
        """)
        result = cur.fetchone()
        if result:
            idx_name = result[0]
            cur.execute(f"SELECT pg_prewarm('{idx_name}')")
            blocks = cur.fetchone()[0]
            print(f"    Prewarmed index '{idx_name}': {blocks} blocks")

        # Prewarm the table
        cur.execute(f"SELECT pg_prewarm('{table_name}')")
        tblocks = cur.fetchone()[0]
        print(f"    Prewarmed table '{table_name}': {tblocks} blocks")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"    Prewarm error: {e}")


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
    env_cmd = f"export PATH={PG_INSTALL}/bin:$PATH && export PG_CONFIG={PG_INSTALL}/bin/pg_config"
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


def drop_and_create_index(table_name, index_ops, lists):
    """Drop all indexes on table and create IVFFlat index."""
    index_name = f"{table_name}_embedding_idx"
    create_sql = f"CREATE INDEX {index_name} ON {table_name} USING ivfflat (embedding {index_ops}) WITH (lists = {lists})"

    print(f"  Dropping existing indexes and creating IVFFlat index (lists={lists})...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop all indexes on table
    cur.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
    for (idx_name,) in cur.fetchall():
        print(f"    Dropping '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Create index
    print(f"    Creating '{index_name}' (this may take a while)...")
    start = time.time()
    cur.execute(create_sql)
    elapsed = time.time() - start
    print(f"    Index created in {elapsed:.1f}s")

    cur.close()
    conn.close()


def setup_index(dataset, lists, table_name, index_ops, index_type="ivfflat"):
    """Try to load snapshot for the index; if not available, rebuild from scratch."""
    snapshot_path = get_snapshot_path(dataset, index_type, {"lists": lists})

    if snapshot_path.exists():
        print(f"  Found saved snapshot: {snapshot_path.name}")
        load_pgdata_snapshot(snapshot_path)
        return True
    else:
        print(f"  No snapshot found at {snapshot_path.name}, rebuilding index...")
        drop_and_create_index(table_name, index_ops, lists)
        return True


def run_workload(workload, probes, table_name, distance_op):
    """Run all queries with given probes setting, return list of per-query execution times (seconds)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SET ivfflat.probes = {probes}")

    query_times = []

    for i in range(len(workload)):
        vec_str = workload[i]
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {K}"
        )
        explain_result = cur.fetchall()

        for row in explain_result:
            line = row[0].strip()
            if 'Execution Time:' in line:
                parts = line.split()
                elapsed_time = float(parts[2]) / 1000.0  # ms -> seconds
                query_times.append(elapsed_time)

    cur.close()
    conn.close()
    return query_times


def run_implementation(name, pgvector_dir, workload, recall_configs, dataset, lists,
                       table_name, index_ops, distance_op, mode="cold",
                       index_type="ivfflat", num_workers=None, no_rebuild=False,
                       num_iterations=1):
    """Run full benchmark for one pgvector implementation.

    mode: "cold" = restart + drop caches before each probe measurement
          "warm" = prewarm once after index setup, no restart between probes
    num_iterations: number of times to repeat the full workload per probe setting
    """
    print(f"\n{'='*60}")
    print(f"Running: {name} (mode={mode}, iterations={num_iterations})")
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

    # Step 3: Set up index (load snapshot or rebuild), unless --no-rebuild
    if not no_rebuild:
        setup_index(dataset, lists, table_name, index_ops, index_type=index_type)
        # Restart after index setup
        restart_postgres()

    # Step 4: For warm mode, prewarm once
    if mode == "warm":
        prewarm_table_and_index(table_name)

    # Step 5: Run workload for each recall target
    results = []
    print(f"\n  {'Target':<10} {'Probes':<10} {'Total Time (s)':<15} {'Avg Time (ms)':<15}")
    print(f"  {'-'*50}")

    for config in recall_configs:
        target = config["target_recall"]
        actual_recall = config.get("actual_recall", target)
        probes = config["probes"]

        all_query_times_ms = []

        for iteration in range(num_iterations):
            if mode == "cold":
                # Cold: restart and clear caches before each iteration
                restart_postgres(quiet=True)
            elif mode == "warm" and iteration > 0:
                # Warm: re-prewarm before subsequent iterations to ensure consistent state
                prewarm_table_and_index(table_name)

            query_times = run_workload(workload, probes, table_name, distance_op)
            query_times_ms_iter = [round(t * 1000, 3) for t in query_times]
            all_query_times_ms.extend(query_times_ms_iter)

            if num_iterations > 1:
                iter_avg = sum(query_times_ms_iter) / len(query_times_ms_iter) if query_times_ms_iter else 0
                print(f"    iter {iteration+1}/{num_iterations}: avg={iter_avg:.2f}ms")

        total_time = sum(all_query_times_ms) / 1000  # back to seconds
        avg_time_ms = sum(all_query_times_ms) / len(all_query_times_ms) if all_query_times_ms else 0
        sorted_times = sorted(all_query_times_ms)
        median_time_ms = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_time_ms = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0

        results.append({
            "target_recall": target,
            "actual_recall": actual_recall,
            "probes": probes,
            "num_iterations": num_iterations,
            "total_time_s": round(total_time, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "median_time_ms": round(median_time_ms, 2),
            "p95_time_ms": round(p95_time_ms, 2),
            "p99_time_ms": round(p99_time_ms, 2),
            "query_times_ms": all_query_times_ms,
        })

        print(f"  {target:<10} {probes:<10} {total_time:<15.4f} {avg_time_ms:<15.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Default vs Parallel IVFFlat (multiple worker counts)")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to use (gist, deep, sift)")
    parser.add_argument("--mode", required=True, choices=["cold", "warm"],
                        help="cold = restart+drop caches between probes; warm = prewarm, no restart")
    parser.add_argument("--skip-default", action="store_true",
                        help="Skip the default pgvector run, only test parallel variants")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Don't rebuild/restore the index, reuse existing one")
    parser.add_argument("--num-iterations", type=int, default=1,
                        help="Number of times to repeat the workload per probe setting (default: 1)")
    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode
    ds_cfg = DATASET_CONFIGS[dataset]
    table_name = ds_cfg["table"]
    distance_op = ds_cfg["distance_op"]
    index_ops = ds_cfg["index_ops"]
    queries_file = ds_cfg["queries_file"]

    # Load tuned lists value
    tune_file = RECALL_DIR / f"results_from_tune_scripts/tune_ivfflat_results_{dataset}.json"
    if not tune_file.exists():
        print(f"ERROR: {tune_file} not found. Run tuning first.")
        sys.exit(1)

    with open(tune_file) as f:
        tune_data = json.load(f)
    if "best_configuration" not in tune_data:
        print("ERROR: tune results missing best_configuration.")
        sys.exit(1)
    lists = tune_data["best_configuration"]["lists"]

    # Load recall targets and probe values from recall_qps results (try matching mode first, fallback to other)
    recall_qps_file = RECALL_DIR / "recall_qps_plots" / "results_recall_qps" / f"recall_qps_{dataset}_ivfflat_{mode}.json"
    if not recall_qps_file.exists():
        # Fallback: try the other mode
        fallback_mode = "cold" if mode == "warm" else "warm"
        recall_qps_file = RECALL_DIR / "recall_qps_plots" / "results_recall_qps" / f"recall_qps_{dataset}_ivfflat_{fallback_mode}.json"
    if not recall_qps_file.exists():
        print(f"ERROR: No recall_qps_{dataset}_ivfflat_{{warm,cold}}.json found.")
        print("Run recall_qps_diagram.py first.")
        sys.exit(1)

    with open(recall_qps_file, 'r') as f:
        recall_qps_data = json.load(f)

    recall_configs = recall_qps_data["ivfflat"]["recall_qps_points"]

    print("Parallel IVFFlat vs Default IVFFlat Comparison (Multiple Worker Counts)")
    print(f"Dataset: {dataset}")
    print(f"Table: {table_name}")
    print(f"Mode: {mode}")
    print(f"K = {K}")
    print(f"Lists = {lists}")
    num_iterations = args.num_iterations
    print(f"Worker counts to test: {WORKER_COUNTS}")
    print(f"Iterations per probe: {num_iterations}")
    print(f"Probe source: {recall_qps_file.name}")
    print(f"Recall targets: {[r['target_recall'] for r in recall_configs]}")
    print(f"Probes:         {[r['probes'] for r in recall_configs]}")
    if args.skip_default:
        print("SKIPPING default implementation (--skip-default)")

    # Load workload (test queries)
    workload_path = PROJECT_DIR / queries_file
    print(f"\nLoading queries from {workload_path}...")
    with open(workload_path, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} queries")

    # Output file (save incrementally) - includes mode in filename
    output_file = SCRIPT_DIR / f"comparison_results_workers_{dataset}_{mode}_K_{K}.json"

    # Load existing results if any (for resuming)
    output = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            output = json.load(f)
        print(f"Loaded existing results from {output_file}")

    output["dataset"] = dataset
    output["mode"] = mode
    output["k"] = K
    output["lists"] = lists
    output["workload_size"] = len(workload)
    output["num_iterations"] = num_iterations
    output["worker_counts_tested"] = WORKER_COUNTS

    # Run default implementation (only once)
    if not args.skip_default and "default" not in output:
        default_results = run_implementation(
            f"Default IVFFlat (pgvector) - {dataset}",
            PGVECTOR_DEFAULT,
            workload,
            recall_configs,
            dataset=dataset,
            lists=lists,
            table_name=table_name,
            index_ops=index_ops,
            distance_op=distance_op,
            mode=mode,
            index_type="ivfflat",
            no_rebuild=args.no_rebuild,
            num_iterations=num_iterations
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

    # For parallel: load snapshot on first worker count, then reuse for rest
    first_parallel = True
    for num_workers in WORKER_COUNTS:
        key = str(num_workers)
        if key in output["parallel_by_workers"]:
            print(f"\nParallel (workers={num_workers}) already done, skipping.")
            first_parallel = False
            continue

        # Only set up index (load snapshot or rebuild) for the first parallel run
        # Subsequent worker counts reuse the same index
        should_rebuild = first_parallel and not args.no_rebuild

        parallel_results = run_implementation(
            f"Parallel IVFFlat (workers={num_workers}) - {dataset}",
            PGVECTOR_PARALLEL,
            workload,
            recall_configs,
            dataset=dataset,
            lists=lists,
            table_name=table_name,
            index_ops=index_ops,
            distance_op=distance_op,
            mode=mode,
            index_type="parallel_ivfflat",
            num_workers=num_workers,
            no_rebuild=not should_rebuild,
            num_iterations=num_iterations
        )
        first_parallel = False
        output["parallel_by_workers"][key] = parallel_results

        # Save incrementally after each worker count
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults for workers={num_workers} saved to {output_file}")

    # Print comparison summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY (dataset={dataset}, mode={mode}, K={K})")
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
