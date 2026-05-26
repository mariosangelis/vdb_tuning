#!/usr/bin/env python3
"""
Tune IVFFlat index: find optimal (lists, probes) for a target recall.

Tests different numbers of centroids (lists) and for each, finds the minimum
number of probes needed to achieve the target recall. Reports execution time
for each configuration so we can pick the fastest one.

Usage:
    sudo python3 tune_ivfflat.py --recall 0.9
    sudo python3 tune_ivfflat.py --recall 0.95 --k 100

Output:
    tune_ivfflat_results.json - Full results for all configurations
"""

import psycopg2
import time
import subprocess
import json
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RECALL_DIR = PROJECT_DIR / "compute_and_compare_recall"

# Workload and exact results
WORKLOAD_PATH = str(PROJECT_DIR / "sampled_vectors_gist.txt")
EXACT_RESULTS_DIR = RECALL_DIR / "exact_results_gist"

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

# === TUNING PARAMETERS ===
LISTS_TO_TEST = [1024, 2048, 4096, 8192]

# Probes to search through (ascending order)
PROBES_TO_TEST = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80,
                  100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# Default K
DEFAULT_K = 100


def wait_for_postgres(max_attempts=60):
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
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def drop_all_indexes():
    """Drop all indexes on gist_data."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data'")
    for (idx_name,) in cur.fetchall():
        print(f"    Dropping index '{idx_name}'...")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
    cur.close()
    conn.close()


def create_ivfflat_index(lists):
    """Create IVFFlat index with specified number of lists (centroids)."""
    print(f"    Creating IVFFlat index with lists={lists}...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    start = time.time()
    cur.execute(
        f"CREATE INDEX gist_data_ivfflat_idx ON gist_data "
        f"USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists})"
    )
    build_time = time.time() - start
    print(f"    Index built in {build_time:.1f}s")

    cur.close()
    conn.close()
    return build_time


def compute_recall_and_time(workload, exact_results, k, probes):
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
            f"SELECT id FROM gist_data ORDER BY embedding <-> '{vec_str}'::vector LIMIT {k}"
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


def find_optimal_probes(workload, exact_results, k, target_recall):
    """Binary-search-like approach: find minimum probes that achieve target recall."""
    results = []

    for probes in PROBES_TO_TEST:
        # Cold restart before measurement
        restart_postgres()

        avg_recall, avg_time_ms, total_time = compute_recall_and_time(
            workload, exact_results, k, probes
        )

        results.append({
            "probes": probes,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "total_time_s": round(total_time, 3),
        })

        print(f"      probes={probes:>4}: recall={avg_recall:.4f}, avg_time={avg_time_ms:.2f}ms")

        # Stop once we've found the target (and test one more for confidence)
        if avg_recall >= target_recall:
            # Test one more to confirm
            break

    # Find the first probes value that meets the target
    optimal = None
    for r in results:
        if r["recall"] >= target_recall:
            optimal = r
            break

    return results, optimal


def main():
    # Parse arguments
    target_recall = 0.9
    k = DEFAULT_K

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--recall" and i < len(sys.argv) - 1:
            target_recall = float(sys.argv[i + 1])
        elif arg == "--k" and i < len(sys.argv) - 1:
            k = int(sys.argv[i + 1])

    print(f"IVFFlat Tuning: target_recall={target_recall}, K={k}")
    print(f"Lists to test: {LISTS_TO_TEST}")
    print(f"Workload: {WORKLOAD_PATH}")

    # Load workload
    print("\nLoading workload...")
    with open(WORKLOAD_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"  Loaded {len(workload)} queries")

    # Load exact results
    exact_file = EXACT_RESULTS_DIR / f"exact_search_results_{len(workload)}_queries_gist_K_{k}.json"
    if not exact_file.exists():
        print(f"ERROR: Exact results not found: {exact_file}")
        print("Run exact search first to generate ground truth.")
        sys.exit(1)

    with open(exact_file, 'r') as f:
        exact_results = json.load(f)
    print(f"  Loaded exact results for K={k}")

    # Output file
    output_file = SCRIPT_DIR / "results_from_tune_scripts" / "tune_ivfflat_results.json"

    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "target_recall": target_recall,
        "k": k,
        "workload_size": len(workload),
        "configurations": []
    }

    print(f"\n{'='*70}")
    print(f"Testing {len(LISTS_TO_TEST)} configurations")
    print(f"{'='*70}")

    for lists in LISTS_TO_TEST:
        print(f"\n--- Configuration: lists={lists} ---")

        # Drop existing indexes
        drop_all_indexes()

        # Create index with this number of lists
        build_time = create_ivfflat_index(lists)

        # Restart to clear caches after index build
        restart_postgres()

        # Find optimal probes for target recall
        print(f"  Searching for probes to achieve recall >= {target_recall}...")
        probe_results, optimal = find_optimal_probes(
            workload, exact_results, k, target_recall
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

        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Find the best configuration (fastest to achieve target recall)
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
        print(f"  lists={best['lists']}, probes={best['optimal_probes']}")
        print(f"  recall={best['achieved_recall']:.4f}")
        print(f"  avg_query_time={best['avg_query_time_ms']:.2f}ms")
        print(f"  index_build_time={best['index_build_time_s']:.1f}s")
        print(f"{'='*70}")

    # Print comparison table
    print(f"\n{'Lists':<8} {'Probes':<8} {'Recall':<10} {'Avg Time (ms)':<15} {'Build Time (s)':<15}")
    print("-" * 56)
    for c in all_results["configurations"]:
        if c["optimal_probes"]:
            print(f"{c['lists']:<8} {c['optimal_probes']:<8} {c['achieved_recall']:<10.4f} "
                  f"{c['avg_query_time_ms']:<15.2f} {c['index_build_time_s']:<15.1f}")
        else:
            print(f"{c['lists']:<8} {'N/A':<8} {'< target':<10} {'N/A':<15} {c['index_build_time_s']:<15.1f}")

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
