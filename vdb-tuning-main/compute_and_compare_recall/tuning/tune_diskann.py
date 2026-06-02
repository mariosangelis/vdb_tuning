#!/usr/bin/env python3
"""
Tune DiskANN (pgvectorscale) index: find optimal build and query parameters.

Tests different num_neighbors and storage_layout configurations,
and for each, sweeps query_search_list_size to find the recall/speed tradeoff.

Usage:
    sudo python3 tune_diskann.py --dataset sift --recall 0.9
    sudo python3 tune_diskann.py --dataset sift --recall 0.95 --k 100

Output:
    results_from_tune_scripts/tune_diskann_results_{dataset}.json
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

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent                    # tuning/
RECALL_DIR = SCRIPT_DIR.parent                                  # compute_and_compare_recall/
PROJECT_ROOT = RECALL_DIR.parent                                # vdb-tuning-main/

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

# PostgreSQL paths
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"

# Load NUMA config
import json as _json
_config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path, "r") as _f:
        NUMA_NODE = str(_json.load(_f).get("hardware", {}).get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"

SNAPSHOT_DIR = Path("/mnt/nvme/saved_indexes")

# === TUNING PARAMETERS ===
# Build-time: graph structure variations
NUM_NEIGHBORS_TO_TEST = [32, 50, 64]
STORAGE_LAYOUTS = ["plain", "memory_optimized"]

# Runtime: query_search_list_size sweep (equivalent to ef_search/probes)
QUERY_SEARCH_LIST_SIZES = [10, 20, 40, 60, 80, 100, 150, 200, 300, 500]

# Runtime: query_rescore values (only relevant for memory_optimized)
QUERY_RESCORE_VALUES = [0, 25, 50, 75, 100]

# Default K
DEFAULT_K = 100

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    "sift": {
        "table": "sift_data",
        "workload_type": "text",
        "workload_path": "sift_test_queries.txt",
        "exact_results_dir": "exact_results_sift",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
    "gist": {
        "table": "gist_data",
        "workload_type": "text",
        "workload_path": "gist_test_queries.txt",
        "exact_results_dir": "exact_results_gist",
        "distance_op": "<->",
        "index_ops": "vector_l2_ops",
    },
}


def wait_for_postgres(max_attempts=60):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready")


def restart_postgres():
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} "
        f"{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, check=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def get_snapshot_path(table_name, num_neighbors, storage_layout):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"{table_name}_diskann_n{num_neighbors}_{storage_layout}_pgdata.tar.gz"


def save_pgdata_snapshot(snapshot_path):
    print(f"    Saving snapshot: {snapshot_path.name}")
    subprocess.run(
        f"tar -czf {snapshot_path} -C {PG_DATA_DIR} .",
        shell=True, check=True
    )


def load_pgdata_snapshot(snapshot_path):
    print(f"    Loading snapshot: {snapshot_path.name}")
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop' 2>/dev/null",
        shell=True, capture_output=True
    )
    time.sleep(2)
    subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)
    subprocess.run(
        f"tar -xzf {snapshot_path} -C {PG_DATA_DIR}",
        shell=True, check=True
    )
    subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    restart_postgres()


def drop_all_indexes(table_name):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"""
        SELECT indexname FROM pg_indexes
        WHERE tablename = '{table_name}' AND indexname != '{table_name}_pkey'
    """)
    indexes = cur.fetchall()
    for (idx_name,) in indexes:
        print(f"    Dropping index: {idx_name}")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
    cur.close()
    conn.close()


def create_diskann_index(table_name, num_neighbors, storage_layout, index_ops):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    idx_name = f"{table_name}_diskann_idx"
    cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    print(f"    Building DiskANN index: num_neighbors={num_neighbors}, storage={storage_layout}")
    start = time.time()
    cur.execute(
        f"CREATE INDEX {idx_name} ON {table_name} "
        f"USING diskann (embedding {index_ops}) "
        f"WITH (num_neighbors = {num_neighbors}, storage_layout = {storage_layout})"
    )
    build_time = time.time() - start
    print(f"    Index built in {build_time:.1f}s")

    cur.close()
    conn.close()
    return build_time


def load_workload(dataset_cfg):
    """Load workload vectors from text file."""
    workload_path = PROJECT_ROOT / dataset_cfg["workload_path"]
    if not workload_path.exists():
        workload_path = SCRIPT_DIR / dataset_cfg["workload_path"]
    if not workload_path.exists():
        workload_path = SCRIPT_DIR.parent / dataset_cfg["workload_path"]

    print(f"  Loading workload from: {workload_path}")
    with open(workload_path, 'r') as f:
        workload = eval(f.read())
    print(f"  Loaded {len(workload)} query vectors")
    return workload


def compute_exact_results(table_name, workload, k, distance_op):
    """Compute exact (brute-force) results using sequential scan."""
    print(f"  Computing exact results (sequential scan, {len(workload)} queries)...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_seqscan = on")

    exact_results = []
    for i, vec in enumerate(workload):
        vec_str = vec if isinstance(vec, str) else "[" + ",".join(str(x) for x in vec) + "]"
        cur.execute(
            f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} "
            f"(SELECT '{vec_str}'::vector) LIMIT {k}"
        )
        ids = [row[0] for row in cur.fetchall()]
        exact_results.append(ids)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(workload)}")

    cur.close()
    conn.close()
    return exact_results


def save_exact_results(exact_results, path):
    with open(path, 'w') as f:
        json.dump(exact_results, f)


def load_exact_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_recall_and_time(table_name, workload, exact_results, k,
                            query_search_list_size, query_rescore, distance_op):
    """Run queries with given parameters and compute recall + avg time."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SET diskann.query_search_list_size = {query_search_list_size}")
    cur.execute(f"SET diskann.query_rescore = {query_rescore}")

    total_recall = 0.0
    total_time = 0.0

    for i, vec in enumerate(workload):
        vec_str = vec if isinstance(vec, str) else "[" + ",".join(str(x) for x in vec) + "]"

        start = time.time()
        cur.execute(
            f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} "
            f"(SELECT '{vec_str}'::vector) LIMIT {k}"
        )
        ids = [row[0] for row in cur.fetchall()]
        elapsed = time.time() - start
        total_time += elapsed

        # Compute recall
        exact_set = set(exact_results[i][:k])
        if len(exact_set) > 0:
            recall = len(set(ids) & exact_set) / len(exact_set)
            total_recall += recall

    cur.close()
    conn.close()

    n = len(workload)
    avg_recall = total_recall / n
    avg_time_ms = (total_time / n) * 1000
    return avg_recall, avg_time_ms, total_time


def prewarm_table_and_index(table_name):
    """Use pg_prewarm to load table and indexes into shared buffers."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT pg_prewarm('{table_name}')")
        # Prewarm all indexes on the table
        cur.execute(f"""
            SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'
        """)
        for (idx_name,) in cur.fetchall():
            try:
                cur.execute(f"SELECT pg_prewarm('{idx_name}')")
            except:
                pass
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"    Warning: pg_prewarm failed: {e}")


def find_optimal_search_list(table_name, workload, exact_results, k, target_recall,
                             storage_layout, distance_op):
    """Sweep query_search_list_size to find minimum that achieves target recall."""
    results = []

    # Warmup
    print(f"      (prewarming with pg_prewarm...)")
    prewarm_table_and_index(table_name)

    # Determine rescore value based on storage layout
    rescore = 50 if storage_layout == "memory_optimized" else 0

    for qsl in QUERY_SEARCH_LIST_SIZES:
        avg_recall, avg_time_ms, total_time = compute_recall_and_time(
            table_name, workload, exact_results, k, qsl, rescore, distance_op
        )

        results.append({
            "query_search_list_size": qsl,
            "query_rescore": rescore,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "total_time_s": round(total_time, 3),
        })

        print(f"      qsl={qsl:>4}, rescore={rescore}: recall={avg_recall:.4f}, avg_time={avg_time_ms:.2f}ms")

        # Stop once we've found the target
        if avg_recall >= target_recall:
            break

    # Find optimal
    optimal = None
    for r in results:
        if r["recall"] >= target_recall:
            optimal = r
            break

    return results, optimal


def main():
    parser = argparse.ArgumentParser(description="Tune DiskANN index parameters")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--recall", type=float, default=0.9, help="Target recall (default: 0.9)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K results (default: 100)")
    args = parser.parse_args()

    dataset = args.dataset
    target_recall = args.recall
    k = args.k
    cfg = DATASET_CONFIGS[dataset]
    table_name = cfg["table"]
    distance_op = cfg["distance_op"]
    index_ops = cfg["index_ops"]

    print(f"=== DiskANN Tuning: {dataset} dataset ===")
    print(f"  Target recall: {target_recall}")
    print(f"  K: {k}")
    print(f"  Table: {table_name}")
    print(f"  num_neighbors to test: {NUM_NEIGHBORS_TO_TEST}")
    print(f"  storage_layouts: {STORAGE_LAYOUTS}")
    print(f"  query_search_list_sizes: {QUERY_SEARCH_LIST_SIZES}")

    # Load workload
    workload = load_workload(cfg)

    # Load or compute exact results
    exact_dir = RECALL_DIR / "exact_results" / cfg["exact_results_dir"]
    exact_path = exact_dir / f"exact_top{k}.json"
    if exact_path.exists():
        print(f"  Loading cached exact results from: {exact_path}")
        exact_results = load_exact_results(exact_path)
    else:
        exact_dir.mkdir(parents=True, exist_ok=True)
        exact_results = compute_exact_results(table_name, workload, k, distance_op)
        save_exact_results(exact_results, exact_path)
        print(f"  Saved exact results to: {exact_path}")

    # Results storage
    all_results = []
    results_dir = RECALL_DIR / "results_from_tune_scripts"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Test each configuration
    for num_neighbors in NUM_NEIGHBORS_TO_TEST:
        for storage_layout in STORAGE_LAYOUTS:
            config_name = f"n{num_neighbors}_{storage_layout}"
            print(f"\n--- Configuration: num_neighbors={num_neighbors}, storage={storage_layout} ---")

            snapshot_path = get_snapshot_path(table_name, num_neighbors, storage_layout)

            if snapshot_path.exists():
                print(f"  Found cached snapshot, loading...")
                load_pgdata_snapshot(snapshot_path)
            else:
                print(f"  Building new index...")
                restart_postgres()
                drop_all_indexes(table_name)
                build_time = create_diskann_index(table_name, num_neighbors, storage_layout, index_ops)
                # Save snapshot
                subprocess.run(
                    f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop'",
                    shell=True, check=True, capture_output=True
                )
                time.sleep(2)
                save_pgdata_snapshot(snapshot_path)
                restart_postgres()

            # Find optimal query_search_list_size
            print(f"  Sweeping query_search_list_size (target recall={target_recall})...")
            sweep_results, optimal = find_optimal_search_list(
                table_name, workload, exact_results, k, target_recall,
                storage_layout, distance_op
            )

            config_result = {
                "num_neighbors": num_neighbors,
                "storage_layout": storage_layout,
                "config_name": config_name,
                "sweep_results": sweep_results,
                "optimal": optimal,
                "snapshot_path": str(snapshot_path),
            }
            all_results.append(config_result)

            if optimal:
                print(f"  OPTIMAL: qsl={optimal['query_search_list_size']}, "
                      f"recall={optimal['recall']:.4f}, time={optimal['avg_time_ms']:.2f}ms")
            else:
                print(f"  WARNING: Could not achieve target recall {target_recall}")

    # Find best overall configuration
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (target recall >= {target_recall})")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'QSL':>5} {'Recall':>8} {'Time(ms)':>10}")
    print(f"{'-'*60}")

    best = None
    for r in all_results:
        opt = r["optimal"]
        if opt:
            print(f"{r['config_name']:<25} {opt['query_search_list_size']:>5} "
                  f"{opt['recall']:>8.4f} {opt['avg_time_ms']:>10.2f}")
            if best is None or opt["avg_time_ms"] < best["avg_time_ms"]:
                best = {**opt, "config": r["config_name"],
                        "num_neighbors": r["num_neighbors"],
                        "storage_layout": r["storage_layout"]}
        else:
            print(f"{r['config_name']:<25}  {'N/A':>5} {'N/A':>8} {'N/A':>10}")

    if best:
        print(f"\nBEST CONFIGURATION:")
        print(f"  num_neighbors={best['num_neighbors']}, storage={best['storage_layout']}")
        print(f"  query_search_list_size={best['query_search_list_size']}")
        print(f"  recall={best['recall']:.4f}, avg_time={best['avg_time_ms']:.2f}ms")

    # Save results
    output = {
        "dataset": dataset,
        "target_recall": target_recall,
        "k": k,
        "table": table_name,
        "timestamp": datetime.now().isoformat(),
        "configurations": all_results,
        "best": best,
    }
    output_path = results_dir / f"tune_diskann_results_{dataset}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
