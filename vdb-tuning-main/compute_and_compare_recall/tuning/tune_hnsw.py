#!/usr/bin/env python3
"""
Tune HNSW index: find optimal (m, ef_construction, ef_search) for a target recall.

Supports GIST, SIFT, and DEEP datasets via --dataset parameter.

Tests different values of m (max connections per layer) and ef_construction,
and for each index configuration, finds the minimum ef_search needed to achieve
the target recall. Reports execution time for each configuration so we can pick
the fastest one.

Usage:
    sudo python3 tune_hnsw.py --dataset sift --recall 0.9
    sudo python3 tune_hnsw.py --dataset gist --recall 0.95 --k 100
    sudo python3 tune_hnsw.py --dataset deep --recall 0.9

Output:
    results_from_tune_scripts/tune_hnsw_results_{dataset}.json - Full results for all configurations
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

# PostgreSQL paths
PG_INSTALL = get_path("pg_install")
PG_CTL_PATH = shutil.which("pg_ctl") or f"{PG_INSTALL}/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or get_path("pg_data")
# Load NUMA config from experiment_config.json
import json as _json
_config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path, "r") as _f:
        NUMA_NODE = str(_json.load(_f).get("hardware", {}).get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"
SNAPSHOT_DIR = Path(get_path("snapshot_dir"))

# === TUNING PARAMETERS ===
M_VALUES_TO_TEST = [8,16,32,64]
EF_CONSTRUCTION_VALUES = [128,256]

# ef_search values to test (ascending order)
EF_SEARCH_TO_TEST = [10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# Default K
DEFAULT_K = 100

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    "gist": {
        "table": "gist_data",
        "workload_type": "text",  # text file with vector strings
        "workload_path": "gist_test_queries.txt",
        "exact_results_dir": "exact_results_gist",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
    "sift": {
        "table": "sift_data",
        "workload_type": "text",
        "workload_path": "sampled_vectors_sift.txt",
        "exact_results_dir": "exact_results_sift",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
    "deep": {
        "table": "deep_data",
        "workload_type": "text",  # text file with vector strings
        "workload_path": "deep_test_queries.txt",
        "exact_results_dir": "exact_results_deep",
        "distance_op": "<=>",
        "index_ops": "vector_cosine_ops",
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


def restart_postgres():
    """Restart PostgreSQL with cache clearing."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def get_snapshot_path(dataset, index_type, params):
    """Get path for a PGDATA snapshot tar.gz. params is a dict like {'lists': 1024} or {'m': 16, 'efc': 128}."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    filename = f"{dataset}_{index_type}_{param_str}_pgdata.tar.gz"
    return SNAPSHOT_DIR / filename


def save_pgdata_snapshot(snapshot_path):
    """Stop postgres, copy PGDATA directory, restart postgres."""
    print(f"    Saving PGDATA snapshot to {snapshot_path}...")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Stop postgres
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop'",
        shell=True, capture_output=True
    )
    time.sleep(2)

    # Create tarball with pigz (parallel gzip)
    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"tar -cf - -C {PG_DATA_DIR} . | pigz > {snapshot_path}"
    else:
        cmd = f"tar -czf {snapshot_path} -C {PG_DATA_DIR} ."

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    size_mb = snapshot_path.stat().st_size / (1024*1024)
    print(f"    Snapshot saved: {size_mb:.0f} MB in {elapsed:.1f}s")

    # Restart postgres
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


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

    # Start postgres
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


def create_hnsw_index(table_name, m, ef_construction, index_ops="vector_l2_ops"):
    """Create HNSW index with specified m and ef_construction."""
    print(f"    Creating HNSW index with m={m}, ef_construction={ef_construction}, ops={index_ops}...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop existing index if any
    cur.execute(f"DROP INDEX IF EXISTS {table_name}_hnsw_idx")

    start = time.time()
    cur.execute(
        f"CREATE INDEX {table_name}_hnsw_idx ON {table_name} "
        f"USING hnsw (embedding {index_ops}) WITH (m = {m}, ef_construction = {ef_construction})"
    )
    build_time = time.time() - start
    print(f"    Index built in {build_time:.1f}s")

    # Get index size
    cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{table_name}_hnsw_idx'))")
    index_size = cur.fetchone()[0]
    print(f"    Index size: {index_size}")

    cur.close()
    conn.close()
    return build_time, index_size


def compute_recall_and_time(table_name, workload, exact_results, k, ef_search, distance_op="<->"):
    """Run all queries with given ef_search, return (avg_recall, avg_time_ms, total_time_s)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SET hnsw.ef_search = {ef_search}")
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
        # Find vector index
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


def find_optimal_ef_search(table_name, workload, exact_results, k, target_recall, distance_op="<->"):
    """Find minimum ef_search that achieves target recall (warm measurements)."""
    results = []

    # Warmup: use pg_prewarm to load everything into shared buffers
    print(f"      (prewarming with pg_prewarm...)")
    prewarm_table_and_index(table_name)

    for ef_search in EF_SEARCH_TO_TEST:
        avg_recall, avg_time_ms, total_time = compute_recall_and_time(
            table_name, workload, exact_results, k, ef_search, distance_op
        )

        results.append({
            "ef_search": ef_search,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "total_time_s": round(total_time, 3),
        })

        print(f"      ef_search={ef_search:>4}: recall={avg_recall:.4f}, avg_time={avg_time_ms:.2f}ms")

        # Stop once we've found the target
        if avg_recall >= target_recall:
            break

    # Find the first ef_search value that meets the target
    optimal = None
    for r in results:
        if r["recall"] >= target_recall:
            optimal = r
            break

    return results, optimal


def load_workload_from_text(workload_path):
    """Load workload from text file (GIST/DEEP format)."""
    print(f"  Loading workload from: {workload_path}")
    with open(workload_path, 'r') as f:
        workload = eval(f.read())
    return workload


def load_workload_from_hdf5(hdf5_path, key='test'):
    """Load workload from HDF5 file and convert to pgvector format strings."""
    if not HAS_H5PY:
        raise RuntimeError(
            "h5py is required to load SIFT dataset. Install it with: pip install h5py"
        )

    print(f"  Loading workload from HDF5: {hdf5_path} (key='{key}')")
    with h5py.File(hdf5_path, 'r') as f:
        if key not in f:
            raise ValueError(f"Key '{key}' not found in HDF5 file. Available keys: {list(f.keys())}")

        vectors = f[key][:]
        # Convert each row to pgvector format: '[v1,v2,v3,...]'
        workload = ['[' + ','.join(map(str, row)) + ']' for row in vectors]

    return workload


def compute_exact_results(table_name, workload, k, distance_op="<->"):
    """Compute exact results using sequential scan (brute force)."""
    print(f"  Computing exact results via sequential scan (K={k})...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Force sequential scan
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")

    exact_results = []
    total_queries = len(workload)

    for i, vec_str in enumerate(workload):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Processing query {i+1}/{total_queries}...")

        cur.execute(
            f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
        )
        rows = cur.fetchall()
        result_ids = [row[0] for row in rows]
        exact_results.append(result_ids)

    cur.close()
    conn.close()

    return exact_results


def save_exact_results(exact_results, output_path, dataset, workload_size, k):
    """Save exact results to JSON file."""
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"exact_search_results_{workload_size}_queries_{dataset}_K_{k}.json"

    with open(filename, 'w') as f:
        json.dump(exact_results, f)

    print(f"  Exact results saved to: {filename}")
    return filename


def load_exact_results(exact_results_path):
    """Load exact results from JSON file."""
    with open(exact_results_path, 'r') as f:
        return json.load(f)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune HNSW index for target recall on GIST, SIFT, or DEEP datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gist", "sift", "deep"],
        help="Dataset to use (gist, sift, or deep)"
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=0.9,
        help="Target recall (default: 0.9)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of nearest neighbors (default: {DEFAULT_K})"
    )

    parser.add_argument(
        "--warm",
        action="store_true",
        help="Use warm measurements (no cold restart between configs, just warmup pass)"
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    dataset = args.dataset
    target_recall = args.recall
    k = args.k

    # Get dataset configuration
    if dataset not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    config = DATASET_CONFIGS[dataset]
    table_name = config["table"]
    distance_op = config.get("distance_op", "<->")
    index_ops = config.get("index_ops", "vector_l2_ops")

    print(f"{'='*70}")
    print(f"HNSW Tuning for {dataset.upper()} dataset")
    print(f"{'='*70}")
    print(f"Target recall: {target_recall}")
    print(f"K: {k}")
    print(f"Table: {table_name}")
    print(f"Distance op: {distance_op}")
    print(f"Index ops: {index_ops}")
    print(f"M values to test: {M_VALUES_TO_TEST}")
    print(f"ef_construction values to test: {EF_CONSTRUCTION_VALUES}")

    # Determine workload path
    workload_filename = config["workload_path"]
    workload_path = PROJECT_ROOT / workload_filename

    if not workload_path.exists():
        print(f"ERROR: Workload file not found: {workload_path}")
        sys.exit(1)

    print(f"Workload file: {workload_path}")

    # Load workload based on type
    print("\nLoading workload...")
    try:
        if config["workload_type"] == "text":
            workload = load_workload_from_text(workload_path)
        elif config["workload_type"] == "hdf5":
            workload = load_workload_from_hdf5(workload_path, config["hdf5_key"])
        else:
            print(f"ERROR: Unknown workload type: {config['workload_type']}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR loading workload: {e}")
        sys.exit(1)

    print(f"  Loaded {len(workload)} queries")

    # Determine exact results directory
    exact_results_dir_name = config["exact_results_dir"]
    exact_results_dir = RECALL_DIR / "exact_results" / exact_results_dir_name

    # Check if exact results exist
    exact_file = exact_results_dir / f"exact_search_results_{len(workload)}_queries_{dataset}_K_{k}.json"

    if exact_file.exists():
        print(f"\nLoading existing exact results from: {exact_file}")
        exact_results = load_exact_results(exact_file)
        print(f"  Loaded exact results for K={k}")
    else:
        print(f"\nExact results not found at: {exact_file}")
        print("Computing exact results via sequential scan (this may take a while)...")

        # Verify table exists
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
        exact_results = compute_exact_results(table_name, workload, k, distance_op)

        # Save exact results
        save_exact_results(exact_results, exact_results_dir, dataset, len(workload), k)

    # Create results directory
    results_dir = RECALL_DIR / "results_from_tune_scripts"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"tune_hnsw_results_{dataset}.json"

    # Results storage
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
    print(f"Testing HNSW configurations")
    print(f"{'='*70}")

    # Test all combinations of m and ef_construction
    for m in M_VALUES_TO_TEST:
        for ef_construction in EF_CONSTRUCTION_VALUES:
            # HNSW requires ef_construction >= 2*m
            if ef_construction < 2 * m:
                print(f"\nSkipping m={m}, ef_construction={ef_construction} (ef_construction must be >= 2*m)")
                continue

            print(f"\n--- Configuration: m={m}, ef_construction={ef_construction} ---")

            # Compute actual ef_construction (HNSW requirement: ef_construction >= 2*m)
            actual_ef = max(ef_construction, 2*m)

            # Check if we have a cached snapshot for this config
            snapshot_path = get_snapshot_path(dataset, "hnsw", {"m": m, "efc": actual_ef})
            if snapshot_path.exists():
                print(f"  Found cached snapshot: {snapshot_path.name}")
                load_pgdata_snapshot(snapshot_path)
                build_time = 0.0
                # Get index size from live DB
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{table_name}_hnsw_idx'))")
                index_size = cur.fetchone()[0]
                cur.close()
                conn.close()
            else:
                # Drop existing indexes
                drop_all_indexes(table_name)
                # Create index with this configuration
                build_time, index_size = create_hnsw_index(table_name, m, ef_construction, index_ops)
                # Save snapshot for future use
                save_pgdata_snapshot(snapshot_path)

            # Restart to clear caches after index build (skip if --warm)
            if not args.warm:
                restart_postgres()

            # Find optimal ef_search for target recall
            print(f"  Searching for ef_search to achieve recall >= {target_recall}...")
            ef_search_results, optimal = find_optimal_ef_search(
                table_name, workload, exact_results, k, target_recall, distance_op
            )

            config_result = {
                "m": m,
                "ef_construction": ef_construction,
                "index_build_time_s": round(build_time, 2),
                "index_size": index_size,
                "target_recall": target_recall,
                "optimal_ef_search": optimal["ef_search"] if optimal else None,
                "achieved_recall": optimal["recall"] if optimal else None,
                "avg_query_time_ms": optimal["avg_time_ms"] if optimal else None,
                "total_query_time_s": optimal["total_time_s"] if optimal else None,
                "all_ef_search_results": ef_search_results,
            }
            all_results["configurations"].append(config_result)

            if optimal:
                print(f"  RESULT: m={m}, ef_construction={ef_construction}, "
                      f"ef_search={optimal['ef_search']}, "
                      f"recall={optimal['recall']:.4f}, avg_time={optimal['avg_time_ms']:.2f}ms")
            else:
                print(f"  WARNING: Could not achieve recall={target_recall} with any ef_search tested!")

            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Find the best configuration (fastest to achieve target recall)
    valid_configs = [c for c in all_results["configurations"] if c["optimal_ef_search"] is not None]
    if valid_configs:
        best = min(valid_configs, key=lambda c: c["avg_query_time_ms"])
        all_results["best_configuration"] = {
            "m": best["m"],
            "ef_construction": best["ef_construction"],
            "ef_search": best["optimal_ef_search"],
            "recall": best["achieved_recall"],
            "avg_query_time_ms": best["avg_query_time_ms"],
            "index_build_time_s": best["index_build_time_s"],
            "index_size": best["index_size"],
        }

        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION for recall >= {target_recall}:")
        print(f"  Dataset: {dataset}")
        print(f"  m={best['m']}, ef_construction={best['ef_construction']}, "
              f"ef_search={best['optimal_ef_search']}")
        print(f"  recall={best['achieved_recall']:.4f}")
        print(f"  avg_query_time={best['avg_query_time_ms']:.2f}ms")
        print(f"  index_build_time={best['index_build_time_s']:.1f}s")
        print(f"  index_size={best['index_size']}")
        print(f"{'='*70}")

    # Print comparison table
    print(f"\n{'M':<6} {'ef_const':<10} {'ef_search':<12} {'Recall':<10} {'Avg Time (ms)':<15} "
          f"{'Build Time (s)':<15} {'Index Size':<15}")
    print("-" * 93)
    for c in all_results["configurations"]:
        if c["optimal_ef_search"]:
            print(f"{c['m']:<6} {c['ef_construction']:<10} {c['optimal_ef_search']:<12} "
                  f"{c['achieved_recall']:<10.4f} {c['avg_query_time_ms']:<15.2f} "
                  f"{c['index_build_time_s']:<15.1f} {c['index_size']:<15}")
        else:
            print(f"{c['m']:<6} {c['ef_construction']:<10} {'N/A':<12} {'< target':<10} "
                  f"{'N/A':<15} {c['index_build_time_s']:<15.1f} {c['index_size']:<15}")

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
