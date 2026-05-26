#!/usr/bin/env python3
"""
Tune HNSW index: find optimal (m, ef_search) for a target recall.

Tests different values of m (max connections) and for each, finds the minimum
ef_search needed to achieve the target recall. Reports execution time
for each configuration so we can pick the fastest one.

Usage:
    sudo python3 tune_hnsw.py --recall 0.9
    sudo python3 tune_hnsw.py --recall 0.95 --k 100

Output:
    tune_hnsw_results.json - Full results for all configurations
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
M_VALUES_TO_TEST = [8, 16, 32, 64]
EF_CONSTRUCTION_VALUES = [64, 128, 256, 512]
# Note: pgvector requires ef_construction >= 2*m, so values below 2*m are skipped

# ef_search values to sweep (must be >= K)
EF_SEARCH_TO_TEST = [100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

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


def create_hnsw_index(m, ef_construction):
    """Create HNSW index with specified m and ef_construction."""
    # pgvector requires ef_construction >= 2*m
    actual_ef = max(ef_construction, 2 * m)
    print(f"    Creating HNSW index with m={m}, ef_construction={actual_ef}...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    start = time.time()
    cur.execute(
        f"CREATE INDEX gist_data_hnsw_idx ON gist_data "
        f"USING hnsw (embedding vector_l2_ops) WITH (m = {m}, ef_construction = {actual_ef})"
    )
    build_time = time.time() - start
    print(f"    Index built in {build_time:.1f}s")

    # Get index size
    cur.execute("SELECT pg_relation_size('gist_data_hnsw_idx')")
    index_size_bytes = cur.fetchone()[0]
    index_size_mb = round(index_size_bytes / (1024 * 1024), 2)
    print(f"    Index size: {index_size_mb} MB")

    cur.close()
    conn.close()
    return build_time, index_size_mb, actual_ef


def compute_recall_and_time(workload, exact_results, k, ef_search):
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


def find_optimal_ef_search(workload, exact_results, k, target_recall):
    """Find minimum ef_search that achieves target recall."""
    results = []

    for ef_search in EF_SEARCH_TO_TEST:
        # Cold restart before measurement
        restart_postgres()

        avg_recall, avg_time_ms, total_time = compute_recall_and_time(
            workload, exact_results, k, ef_search
        )

        results.append({
            "ef_search": ef_search,
            "recall": round(avg_recall, 4),
            "avg_time_ms": round(avg_time_ms, 2),
            "total_time_s": round(total_time, 3),
        })

        print(f"      ef_search={ef_search:>4}: recall={avg_recall:.4f}, avg_time={avg_time_ms:.2f}ms")

        # Stop once we've achieved the target
        if avg_recall >= target_recall:
            break

    # Find the first ef_search value that meets the target
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

    print(f"HNSW Tuning: target_recall={target_recall}, K={k}")
    print(f"m values to test: {M_VALUES_TO_TEST}")
    print(f"ef_construction values to test: {EF_CONSTRUCTION_VALUES}")
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
    output_file = SCRIPT_DIR / "results_from_tune_scripts" / "tune_hnsw_results.json"

    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "target_recall": target_recall,
        "k": k,
        "workload_size": len(workload),
        "m_values": M_VALUES_TO_TEST,
        "ef_construction_values": EF_CONSTRUCTION_VALUES,
        "configurations": []
    }

    # Count valid combinations (ef_construction must be >= 2*m)
    configs_to_test = []
    for m in M_VALUES_TO_TEST:
        for ef_c in EF_CONSTRUCTION_VALUES:
            if ef_c >= 2 * m:
                configs_to_test.append((m, ef_c))

    print(f"\n{'='*70}")
    print(f"Testing {len(configs_to_test)} configurations (m x ef_construction where ef_c >= 2*m)")
    print(f"{'='*70}")

    for m, ef_c in configs_to_test:
        print(f"\n--- Configuration: m={m}, ef_construction={ef_c} ---")

        # Drop existing indexes
        drop_all_indexes()

        # Create index with this m and ef_construction
        build_time, index_size_mb, actual_ef = create_hnsw_index(m, ef_c)

        # Restart to clear caches after index build
        restart_postgres()

        # Find optimal ef_search for target recall
        print(f"  Searching for ef_search to achieve recall >= {target_recall}...")
        ef_results, optimal = find_optimal_ef_search(
            workload, exact_results, k, target_recall
        )

        config_result = {
            "m": m,
            "ef_construction": actual_ef,
            "index_build_time_s": round(build_time, 2),
            "index_size_mb": index_size_mb,
            "target_recall": target_recall,
            "optimal_ef_search": optimal["ef_search"] if optimal else None,
            "achieved_recall": optimal["recall"] if optimal else None,
            "avg_query_time_ms": optimal["avg_time_ms"] if optimal else None,
            "total_query_time_s": optimal["total_time_s"] if optimal else None,
            "all_ef_search_results": ef_results,
        }
        all_results["configurations"].append(config_result)

        if optimal:
            print(f"  RESULT: m={m}, ef_c={actual_ef}, ef_search={optimal['ef_search']}, "
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
            "index_size_mb": best["index_size_mb"],
        }

        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION for recall >= {target_recall}:")
        print(f"  m={best['m']}, ef_construction={best['ef_construction']}, ef_search={best['optimal_ef_search']}")
        print(f"  recall={best['achieved_recall']:.4f}")
        print(f"  avg_query_time={best['avg_query_time_ms']:.2f}ms")
        print(f"  index_build_time={best['index_build_time_s']:.1f}s")
        print(f"  index_size={best['index_size_mb']:.1f} MB")
        print(f"{'='*70}")

    # Print comparison table
    print(f"\n{'m':<6} {'ef_c':<8} {'ef_search':<12} {'Recall':<10} {'Avg Time (ms)':<15} {'Build (s)':<12} {'Size (MB)':<10}")
    print("-" * 73)
    for c in all_results["configurations"]:
        if c["optimal_ef_search"]:
            print(f"{c['m']:<6} {c['ef_construction']:<8} {c['optimal_ef_search']:<12} {c['achieved_recall']:<10.4f} "
                  f"{c['avg_query_time_ms']:<15.2f} {c['index_build_time_s']:<12.1f} {c['index_size_mb']:<10.1f}")
        else:
            print(f"{c['m']:<6} {c['ef_construction']:<8} {'N/A':<12} {'< target':<10} "
                  f"{'N/A':<15} {c['index_build_time_s']:<12.1f} {c['index_size_mb']:<10.1f}")

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
