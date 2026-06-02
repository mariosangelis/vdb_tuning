#!/usr/bin/env python3
"""
Compute exact (brute-force) search results for recall evaluation.

Usage:
    sudo python3 compute_exact_results.py --dataset gist --k 100
    sudo python3 compute_exact_results.py --dataset sift --k 100

This runs sequential scan (no index) queries for each test vector and saves
the top-K IDs as ground truth for recall computation.

Input:  {dataset}_test_queries.txt  (in PROJECT_DIR)
Output: exact_results_{dataset}/exact_search_results_1000_queries_{dataset}_K_{k}.json
"""

import psycopg2
import json
import time
import sys
import os
from pathlib import Path

# === AUTO-DETECT PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent                    # exact_results/
RECALL_DIR = SCRIPT_DIR.parent                                  # compute_and_compare_recall/
PROJECT_ROOT = RECALL_DIR.parent                                # vdb-tuning-main/

# === DATASET CONFIGS ===
DATASETS = {
    "gist": {
        "table": "gist_data",
        "queries_file": "gist_test_queries.txt",
        "distance_op": "<->",
    },
    "sift": {
        "table": "sift_data",
        "queries_file": "sift_test_queries.txt",
        "distance_op": "<->",
    },
    "deep": {
        "table": "deep_data",
        "queries_file": "deep_test_queries.txt",
        "distance_op": "<=>",
    },
}

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432",
}


def load_queries(queries_path):
    """Load query vectors from text file (Python list of lists or list of strings)."""
    with open(queries_path, 'r') as f:
        queries = eval(f.read())

    # Convert to vector string format if needed
    workload = []
    for q in queries:
        if isinstance(q, list):
            vec_str = "[" + ",".join(str(x) for x in q) + "]"
        else:
            vec_str = q  # already a string like "[0.1,0.2,...]"
        workload.append(vec_str)

    return workload


def compute_exact_results(table_name, workload, k, distance_op="<->"):
    """Run brute-force sequential scan for each query, return list of ID lists."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = on;")
    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_bitmapscan = off;")

    results = []
    total = len(workload)
    start_time = time.time()

    for i, vec_str in enumerate(workload):
        q_start = time.time()
        cur.execute(
            f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}"
        )
        ids = [row[0] for row in cur.fetchall()]
        results.append(ids)
        q_elapsed = time.time() - q_start

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_per_query = elapsed / (i + 1)
            remaining = avg_per_query * (total - i - 1)
            print(f"  [{i+1}/{total}] last={q_elapsed:.2f}s, avg={avg_per_query:.2f}s, ETA={remaining/60:.1f}min")

    cur.close()
    conn.close()

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg per query: {total_time/total:.2f}s")

    return results


def main():
    # Parse arguments
    dataset = None
    k = 100

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--k" and i + 1 < len(args):
            k = int(args[i + 1])
            i += 2
        else:
            i += 1

    if dataset is None or dataset not in DATASETS:
        print(f"Usage: sudo python3 {sys.argv[0]} --dataset <gist|sift> [--k 100]")
        print(f"  Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)

    cfg = DATASETS[dataset]
    table_name = cfg["table"]
    queries_file = PROJECT_ROOT / cfg["queries_file"]
    output_dir = SCRIPT_DIR / f"exact_results_{dataset}"

    print(f"{'='*60}")
    print(f"Computing Exact Search Results (Brute Force)")
    print(f"{'='*60}")
    print(f"  Dataset:      {dataset}")
    print(f"  Table:        {table_name}")
    print(f"  K:            {k}")
    print(f"  Queries file: {queries_file}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*60}")

    # Check queries file exists
    if not queries_file.exists():
        print(f"\nERROR: Queries file not found: {queries_file}")
        print(f"  Expected file: {cfg['queries_file']} in {PROJECT_DIR}")
        sys.exit(1)

    # Load queries
    print(f"\nLoading queries from {queries_file}...")
    workload = load_queries(queries_file)
    print(f"  Loaded {len(workload)} query vectors")

    # Verify table exists and has data
    print(f"\nVerifying table '{table_name}'...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT count(*) FROM {table_name}")
        count = cur.fetchone()[0]
        print(f"  Table has {count:,} rows")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  ERROR: Cannot access table '{table_name}': {e}")
        sys.exit(1)

    # Compute exact results
    distance_op = cfg.get("distance_op", "<->")
    print(f"\nComputing exact top-{k} for {len(workload)} queries (sequential scan)...")
    print(f"  Distance operator: {distance_op}")
    print(f"  This will take a while for large tables. Be patient.\n")

    results = compute_exact_results(table_name, workload, k, distance_op)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"exact_search_results_{len(workload)}_queries_{dataset}_K_{k}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"\nResults saved to: {output_file}")
    print(f"  {len(results)} queries x {k} results each")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
