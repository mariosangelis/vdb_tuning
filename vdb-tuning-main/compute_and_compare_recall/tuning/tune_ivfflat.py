#!/usr/bin/env python3
"""
Tune IVFFlat index: find optimal (lists, probes) for a target recall.

Supports both GIST and SIFT datasets via --dataset parameter.

Tests different numbers of centroids (lists) and for each, finds the minimum
number of probes needed to achieve the target recall. Reports execution time
for each configuration so we can pick the fastest one.

Usage:
    sudo python3 tune_ivfflat.py --dataset sift --recall 0.9
    sudo python3 tune_ivfflat.py --dataset gist --recall 0.95 --k 100

Output:
    ../results_from_tune_scripts/tune_ivfflat_results_{dataset}.json
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
from datetime import datetime

# Try importing h5py for SIFT dataset support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent                    # tuning/
RECALL_DIR = SCRIPT_DIR.parent                                  # compute_and_compare_recall/
PROJECT_ROOT = RECALL_DIR.parent                                # vdb-tuning-main/

# Central machine-specific config (paths) from config.yaml at the repo root.
sys.path.insert(0, str(PROJECT_ROOT.parent))  # config_loader lives at ~/vdb_tuning (repo root)
from config_loader import get_path  # noqa: E402

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

# PostgreSQL paths (from config.yaml; PG_CTL/PGDATA still auto-detected first)
PG_INSTALL = get_path("pg_install")
PG_CTL_PATH = shutil.which("pg_ctl") or f"{PG_INSTALL}/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or get_path("pg_data")
SNAPSHOT_DIR = Path(get_path("snapshot_dir"))

# === TUNING PARAMETERS ===
DEFAULT_LISTS_TO_TEST = [2048, 4096, 8192]

# Probes to search through (ascending order)
PROBES_TO_TEST = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80,
                  100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# Default K
DEFAULT_K = 100

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    "gist": {
        "table": "gist_data",
        "workload_type": "text",
        "workload_path": "gist_test_queries.txt",
        "exact_results_dir": "exact_results_gist",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
        "lists_to_test": [8192],
    },
    "sift": {
        "table": "sift_data",
        "workload_type": "text",
        "workload_path": "sift_test_queries.txt",
        "exact_results_dir": "exact_results_sift",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
        "lists_to_test": [2048, 4096, 8192],
    },
    "deep": {
        "table": "deep_data",
        "workload_type": "text",
        "workload_path": "deep_test_queries.txt",
        "exact_results_dir": "exact_results_deep",
        "distance_op": "<=>",
        "index_ops": "vector_cosine_ops",
        "lists_to_test": [16384, 32768],
    },
}


def wait_for_postgres(max_attempts=60):
    """Wait for PostgreSQL to become ready."""
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


# Load NUMA config from experiment_config.json if available
_config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path, "r") as _f:
        _hw_cfg = json.load(_f).get("hardware", {})
    NUMA_NODE = str(_hw_cfg.get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"

# Default pgvector source path (from config.yaml)
PGVECTOR_DEFAULT_DIR = Path(get_path("pgvector_default"))


def compile_default_pgvector():
    """Compile and install the default pgvector extension, then restart postgres."""
    if not PGVECTOR_DEFAULT_DIR.exists():
        print(f"  WARNING: Default pgvector not found at {PGVECTOR_DEFAULT_DIR}, skipping compile")
        return
    print(f"  Compiling default pgvector from {PGVECTOR_DEFAULT_DIR}...")
    env_prefix = f"export PATH={PG_INSTALL}/bin:$PATH"
    result = subprocess.run(
        f"cd {PGVECTOR_DEFAULT_DIR} && {env_prefix} && make clean > /dev/null 2>&1; make -j$(nproc) && make install",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Compile ERROR: {result.stderr[:300]}")
        sys.exit(1)
    print("  Default pgvector compiled and installed.")
    # Restart to load new .so
    restart_postgres_no_numa()


def restart_postgres():
    """Restart PostgreSQL with cache clearing, pinned to NUMA node."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()

def restart_postgres_no_numa():
    """Restart PostgreSQL with cache clearing, no NUMA pinning."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def get_snapshot_path(dataset, index_type, params):
    """Get path for a PGDATA snapshot tar.gz."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    fname = f"{dataset}_{index_type}_{param_str}_pgdata.tar.gz"
    return SNAPSHOT_DIR / fname


def save_pgdata_snapshot(snapshot_path):
    """Stop postgres, save PGDATA as tar.gz, restart postgres."""
    print(f"    Saving PGDATA snapshot to {snapshot_path}...")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop'",
        shell=True, capture_output=True
    )
    time.sleep(2)

    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"tar -cf - -C {PG_DATA_DIR} . | pigz > {snapshot_path}"
    else:
        cmd = f"tar -czf {snapshot_path} -C {PG_DATA_DIR} ."

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    size_mb = snapshot_path.stat().st_size / (1024 * 1024)
    print(f"    Snapshot saved: {size_mb:.0f} MB in {elapsed:.1f}s")

    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def load_pgdata_snapshot(snapshot_path):
    """Stop postgres, restore PGDATA from tar, restart postgres."""
    print(f"    Loading PGDATA snapshot from {snapshot_path}...")

    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop'",
        shell=True, capture_output=True
    )
    time.sleep(2)

    subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)

    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"pigz -dc {snapshot_path} | tar -xf - -C {PG_DATA_DIR}"
    else:
        cmd = f"tar -xzf {snapshot_path} -C {PG_DATA_DIR}"

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    subprocess.run(f"chmod 700 {PG_DATA_DIR}", shell=True, check=True)

    print(f"    Snapshot restored in {elapsed:.1f}s")

    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def drop_all_indexes(table_name):
    """Drop all indexes on the specified table."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
    for (idx_name,) in cur.fetchall():
        print(f"    Dropping index '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
    cur.close()
    conn.close()


def create_ivfflat_index(table_name, lists, index_ops="vector_l2_ops"):
    """Create IVFFlat index with specified number of lists."""
    print(f"    Creating IVFFlat index with lists={lists}, ops={index_ops}...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SET maintenance_work_mem = '15GB'")
    cur.execute("SET max_parallel_maintenance_workers = '16'")
    cur.execute("SET max_parallel_workers_per_gather = '16'")
    cur.execute(f"DROP INDEX IF EXISTS {table_name}_ivfflat_idx")

    start = time.time()
    cur.execute(
        f"CREATE INDEX {table_name}_ivfflat_idx ON {table_name} "
        f"USING ivfflat (embedding {index_ops}) WITH (lists = {lists})"
    )
    build_time = time.time() - start
    print(f"    Index built in {build_time:.1f}s")

    cur.close()
    conn.close()
    return build_time


def compute_recall_and_time(table_name, workload, exact_results, k, probes, distance_op="<->"):
    """Run all queries with given probes, return (avg_recall, avg_time_ms, total_time_s)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SET ivfflat.probes = {probes}")
    cur.execute("SET enable_seqscan = off")

    total_recall = 0.0
    total_time = 0.0

    for i in range(len(workload)):
        exact_ids = set(exact_results[i][:k])
        vec_str = workload[i]

        start = time.time()
        cur.execute(
            f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
        )
        rows = cur.fetchall()
        elapsed = time.time() - start
        total_time += elapsed

        result_ids = set(row[0] for row in rows)
        hits = len(result_ids & exact_ids)
        total_recall += hits / k

    cur.close()
    conn.close()

    avg_recall = total_recall / len(workload)
    avg_time_ms = (total_time / len(workload)) * 1000
    return avg_recall, avg_time_ms, total_time


def prewarm_table_and_index(table_name):
    """Use pg_prewarm to load entire table and index into shared buffers."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
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
            idx_name = result[0]
            cur.execute(f"SELECT pg_prewarm('{idx_name}')")
            blocks = cur.fetchone()[0]
            print(f"      Prewarmed index '{idx_name}': {blocks} blocks")
        cur.execute(f"SELECT pg_prewarm('{table_name}')")
        tblocks = cur.fetchone()[0]
        print(f"      Prewarmed table '{table_name}': {tblocks} blocks")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"      Prewarm error: {e}")


def find_optimal_probes(table_name, workload, exact_results, k, target_recall, distance_op="<->"):
    """Find minimum probes that achieve target recall (warm measurements)."""
    results = []

    print(f"      (prewarming with pg_prewarm...)")
    prewarm_table_and_index(table_name)

    for probes in PROBES_TO_TEST:
        avg_recall, avg_time_ms, total_time = compute_recall_and_time(
            table_name, workload, exact_results, k, probes, distance_op
        )

        results.append({
            "probes": probes,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "total_time_s": round(total_time, 3),
        })

        print(f"      probes={probes:>4}: recall={avg_recall:.4f}, avg_time={avg_time_ms:.2f}ms")

        if avg_recall >= target_recall:
            break

    optimal = None
    for r in results:
        if r["recall"] >= target_recall:
            optimal = r
            break

    return results, optimal


def load_workload_from_text(workload_path):
    """Load workload from text file."""
    print(f"  Loading workload from: {workload_path}")
    with open(workload_path, 'r') as f:
        workload = eval(f.read())
    return workload


def load_workload_from_hdf5(hdf5_path, key='test'):
    """Load workload from HDF5 file and convert to pgvector format strings."""
    if not HAS_H5PY:
        raise RuntimeError("h5py is required. Install with: pip install h5py")

    print(f"  Loading workload from HDF5: {hdf5_path} (key='{key}')")
    with h5py.File(hdf5_path, 'r') as f:
        if key not in f:
            raise ValueError(f"Key '{key}' not found. Available: {list(f.keys())}")
        vectors = f[key][:]
        workload = ['[' + ','.join(map(str, row)) + ']' for row in vectors]

    return workload


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tune IVFFlat index for target recall on GIST or SIFT datasets"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["gist", "sift", "deep"],
                        help="Dataset to use (gist, sift, or deep)")
    parser.add_argument("--recall", type=float, default=0.9, help="Target recall (default: 0.9)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of nearest neighbors (default: {DEFAULT_K})")
    parser.add_argument("--warm", action="store_true",
                        help="Use warm measurements (no restart between probe sweeps)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = args.dataset
    target_recall = args.recall
    k = args.k

    # Step 0: Ensure default pgvector is compiled and installed
    compile_default_pgvector()

    config = DATASET_CONFIGS[dataset]
    table_name = config["table"]
    distance_op = config.get("distance_op", "<->")
    index_ops = config.get("index_ops", "vector_l2_ops")
    lists_to_test = config.get("lists_to_test", DEFAULT_LISTS_TO_TEST)

    print(f"{'='*70}")
    print(f"IVFFlat Tuning for {dataset.upper()} dataset")
    print(f"{'='*70}")
    print(f"Target recall: {target_recall}")
    print(f"K: {k}")
    print(f"Table: {table_name}")
    print(f"Distance op: {distance_op}")
    print(f"Index ops: {index_ops}")
    print(f"Lists to test: {lists_to_test}")

    # Workload path: in project root
    workload_path = PROJECT_ROOT / config["workload_path"]
    if not workload_path.exists():
        print(f"ERROR: Workload file not found: {workload_path}")
        sys.exit(1)
    print(f"Workload file: {workload_path}")

    # Load workload
    print("\nLoading workload...")
    if config["workload_type"] == "text":
        workload = load_workload_from_text(workload_path)
    elif config["workload_type"] == "hdf5":
        workload = load_workload_from_hdf5(workload_path, config["hdf5_key"])
    else:
        print(f"ERROR: Unknown workload type: {config['workload_type']}")
        sys.exit(1)
    print(f"  Loaded {len(workload)} queries")

    # Exact results path: compute_and_compare_recall/exact_results/exact_results_{dataset}/
    exact_results_dir = RECALL_DIR / "exact_results" / config["exact_results_dir"]
    exact_file = exact_results_dir / f"exact_search_results_{len(workload)}_queries_{dataset}_K_{k}.json"

    if exact_file.exists():
        print(f"\nLoading existing exact results from: {exact_file}")
        with open(exact_file, 'r') as f:
            exact_results = json.load(f)
        print(f"  Loaded exact results for K={k}")
    else:
        print(f"\nExact results not found at: {exact_file}")
        print("Computing exact results via sequential scan (this may take a while)...")

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            print(f"  Table {table_name} contains {count} vectors")
        except Exception as e:
            print(f"ERROR: Cannot access table {table_name}: {e}")
            sys.exit(1)

        # Compute exact results
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")
        exact_results = []
        for i, vec_str in enumerate(workload):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    Processing query {i+1}/{len(workload)}...")
            cur.execute(
                f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
            )
            exact_results.append([row[0] for row in cur.fetchall()])
        cur.close()
        conn.close()

        # Save
        exact_results_dir.mkdir(parents=True, exist_ok=True)
        with open(exact_file, 'w') as f:
            json.dump(exact_results, f)
        print(f"  Exact results saved to: {exact_file}")

    # Results directory: compute_and_compare_recall/results_from_tune_scripts/
    results_dir = RECALL_DIR / "results_from_tune_scripts"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"tune_ivfflat_results_{dataset}.json"

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "table": table_name,
        "target_recall": target_recall,
        "k": k,
        "workload_size": len(workload),
        "configurations": []
    }

    print(f"\n{'='*70}")
    print(f"Testing {len(lists_to_test)} configurations")
    print(f"{'='*70}")


    for lists in lists_to_test:
        print(f"\n--- Configuration: lists={lists} ---")

        snapshot_path = get_snapshot_path(dataset, "ivfflat", {"lists": lists})
        if snapshot_path.exists():
            print(f"  Found cached snapshot: {snapshot_path.name}")
            load_pgdata_snapshot(snapshot_path)
            build_time = 0.0
        else:
            drop_all_indexes(table_name)
            build_time = create_ivfflat_index(table_name, lists, index_ops)
            save_pgdata_snapshot(snapshot_path)

        if not args.warm:
            restart_postgres()

        print(f"  Searching for probes to achieve recall >= {target_recall}...")
        probe_results, optimal = find_optimal_probes(
            table_name, workload, exact_results, k, target_recall, distance_op
        )

        config_result = {
            "lists": lists,
            "index_build_time_s": round(build_time, 2),
            "target_recall": target_recall,
            "optimal_probes": optimal["probes"] if optimal else None,
            "achieved_recall": optimal["recall"] if optimal else None,
            "avg_query_time_ms": optimal["avg_time_ms"] if optimal else None,
            "total_query_time_s": optimal["total_time_s"] if optimal else None,
            "all_probe_results": probe_results,
        }
        all_results["configurations"].append(config_result)

        if optimal:
            print(f"  RESULT: lists={lists}, probes={optimal['probes']}, "
                  f"recall={optimal['recall']:.4f}, avg_time={optimal['avg_time_ms']:.2f}ms")
        else:
            print(f"  WARNING: Could not achieve recall={target_recall} with any probes tested!")

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Find best configuration
    valid_configs = [c for c in all_results["configurations"] if c["optimal_probes"] is not None]
    if valid_configs:
        best = min(valid_configs, key=lambda c: c["avg_query_time_ms"])
        all_results["best_configuration"] = {
            "lists": best["lists"],
            "probes": best["optimal_probes"],
            "recall": best["achieved_recall"],
            "avg_query_time_ms": best["avg_query_time_ms"],
            "index_build_time_s": best["index_build_time_s"],
        }

        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION for recall >= {target_recall}:")
        print(f"  Dataset: {dataset}")
        print(f"  lists={best['lists']}, probes={best['optimal_probes']}")
        print(f"  recall={best['achieved_recall']:.4f}")
        print(f"  avg_query_time={best['avg_query_time_ms']:.2f}ms")
        print(f"  index_build_time={best['index_build_time_s']:.1f}s")
        print(f"{'='*70}")

    # Print comparison table
    print(f"\n{'Lists':<8} {'Probes':<8} {'Recall':<10} {'Avg Time (ms)':<15} {'Build Time (s)':<15}")
    print("-" * 66)
    for c in all_results["configurations"]:
        if c["optimal_probes"]:
            print(f"{c['lists']:<8} {c['optimal_probes']:<8} {c['achieved_recall']:<10.4f} "
                  f"{c['avg_query_time_ms']:<15.2f} {c['index_build_time_s']:<15.1f}")
        else:
            print(f"{c['lists']:<8} {'N/A':<8} {'< target':<10} {'N/A':<15} {c['index_build_time_s']:<15.1f}")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
