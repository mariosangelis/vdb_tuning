#!/usr/bin/env python3
"""
Tune Parallel IVFFlat: Reads the IVFFlat tuning results JSON to get the best
lists parameter, compiles the parallel pgvector extension, builds the index,
and saves the PGDATA snapshot as tar.gz.

This "cheats" by reusing the lists value from the default IVFFlat tuning
(same number of centroids, same recall characteristics) but builds with
the parallel extension for faster index construction.

Usage:
    sudo python3 tune_parallel_ivfflat.py --dataset sift
    sudo python3 tune_parallel_ivfflat.py --dataset gist
    sudo python3 tune_parallel_ivfflat.py --dataset sift --lists 4096  # override

Output:
    /mnt/nvme/saved_indexes/{dataset}_parallel_ivfflat_lists{N}_pgdata.tar.gz
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

# Central machine-specific config (paths) from config.yaml at the repo root.
sys.path.insert(0, str(PROJECT_ROOT.parent))  # config_loader lives at ~/vdb_tuning (repo root)
from config_loader import get_path  # noqa: E402

# === DB CONFIG ===
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432",
}

# PostgreSQL paths
PG_INSTALL = get_path("pg_install")
PG_CTL_PATH = shutil.which("pg_ctl") or f"{PG_INSTALL}/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or get_path("pg_data")
SNAPSHOT_DIR = Path(get_path("snapshot_dir"))

# Load NUMA config from experiment_config.json if available
_config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if not _config_path.exists():
    # Try from PROJECT_DIR
    _config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path, "r") as _f:
        _hw_cfg = json.load(_f).get("hardware", {})
    NUMA_NODE = str(_hw_cfg.get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"

# Parallel pgvector source path (one level above PROJECT_DIR)
PGVECTOR_PARALLEL_DIR = Path(get_path("pgvector_parallel"))

# === DATASET CONFIGS ===
DATASETS = {
    "gist": {
        "table": "gist_data",
        "ops_class": "vector_l2_ops",
        "index_name": "gist_data_embedding_parallel_ivfflat_idx",
    },
    "sift": {
        "table": "sift_data",
        "ops_class": "vector_l2_ops",
        "index_name": "sift_data_embedding_parallel_ivfflat_idx",
    },
    "deep": {
        "table": "deep_data",
        "ops_class": "vector_cosine_ops",
        "index_name": "deep_data_embedding_parallel_ivfflat_idx",
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
    raise Exception("PostgreSQL did not become ready in time.")


def stop_postgres():
    """Stop PostgreSQL."""
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop -m fast'",
        shell=True, capture_output=True,
    )
    time.sleep(2)


def restart_postgres():
    """Restart PostgreSQL with cache clearing, pinned to NUMA node."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True,
    )
    time.sleep(3)
    wait_for_postgres()


def restart_postgres_no_numa():
    """Restart PostgreSQL with cache clearing, pinned to NUMA node."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    subprocess.run(
        f"su - postgres -c ' {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, capture_output=True
    )
    time.sleep(3)
    wait_for_postgres()


def compile_parallel_pgvector():
    """Compile and install the parallel pgvector extension."""
    if not PGVECTOR_PARALLEL_DIR.exists():
        print(f"ERROR: Parallel pgvector source not found at {PGVECTOR_PARALLEL_DIR}")
        sys.exit(1)

    print(f"  Compiling parallel pgvector from {PGVECTOR_PARALLEL_DIR}...")
    env_prefix = f"export PATH={PG_INSTALL}/bin:$PATH"
    result = subprocess.run(
        f"cd {PGVECTOR_PARALLEL_DIR} && {env_prefix} && make clean > /dev/null 2>&1; make -j$(nproc) && make install",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Compile ERROR: {result.stderr[:500]}")
        sys.exit(1)
    print("  Compiled and installed successfully.")

    # Restart postgres to load new shared library
    print("  Restarting PostgreSQL to load new extension binary...")
    restart_postgres_no_numa()
    print("  PostgreSQL restarted with parallel extension.")


def load_ivfflat_tuning_results(dataset):
    """Load IVFFlat tuning results to get the best lists parameter."""
    # Look in results_from_tune_scripts directory relative to SCRIPT_DIR
    tune_dir = RECALL_DIR / "results_from_tune_scripts"
    fname = f"tune_ivfflat_results_{dataset}.json"

    fpath = tune_dir / fname
    if not fpath.exists():
        # Try from compute_and_compare_recall
        alt_dir = RECALL_DIR / "results_from_tune_scripts"
        fpath = alt_dir / fname
    if not fpath.exists():
        # Try absolute well-known path
        fpath = Path(f"/users/mangelis/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/results_from_tune_scripts/{fname}")
    if not fpath.exists():
        return None

    print(f"  Loading tuning results from: {fpath}")
    with open(fpath, "r") as f:
        data = json.load(f)

    return data


def get_best_lists(tuning_data):
    """Extract best lists parameter from tuning results."""
    if tuning_data is None:
        return None

    if "best_configuration" in tuning_data:
        return tuning_data["best_configuration"]["lists"]

    # Fallback: pick the config with best recall that achieved target
    configs = tuning_data.get("configurations", [])
    target = tuning_data.get("target_recall", 0.9)
    achieved = [c for c in configs if c.get("achieved_recall") and c["achieved_recall"] >= target]

    if achieved:
        # Pick fastest among those that achieved target
        best = min(achieved, key=lambda c: c.get("avg_query_time_ms", float("inf")) or float("inf"))
        return best["lists"]

    # If none achieved target, pick highest recall
    if configs:
        best = max(configs, key=lambda c: c.get("achieved_recall", 0) or 0)
        return best["lists"]

    return None


def drop_existing_vector_indexes(table_name):
    """Drop any existing vector indexes on the table."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"""
        SELECT i.relname FROM pg_index ix
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_am am ON am.oid = i.relam
        WHERE t.relname = '{table_name}'
        AND am.amname IN ('ivfflat', 'hnsw')
    """)
    for row in cur.fetchall():
        idx_name = row[0]
        print(f"  Dropping existing index: {idx_name}")
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
    cur.close()
    conn.close()


def build_index(table_name, ops_class, index_name, lists):
    """Build the IVFFlat index with parallel pgvector."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Set maintenance_work_mem high for large list counts
    cur.execute("SET maintenance_work_mem = '15GB'")
    #cur.execute("SET shared_buffers = '230000000kB'")
    cur.execute("SET max_parallel_maintenance_workers = '16'")
    cur.execute("SET max_parallel_workers_per_gather = '16'")
    # Drop existing index if any
    cur.execute(f"DROP INDEX IF EXISTS {table_name}_ivfflat_idx")

    create_sql = (
        f"CREATE INDEX {index_name} ON {table_name} "
        f"USING ivfflat (embedding {ops_class}) WITH (lists = {lists})"
    )

    print(f"  Building index: {create_sql}")
    start = time.time()
    cur.execute(create_sql)
    elapsed = time.time() - start
    print(f"  Index built in {elapsed:.1f}s")

    # Get index size
    cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{index_name}'))")
    size = cur.fetchone()[0]
    print(f"  Index size: {size}")

    cur.close()
    conn.close()

    return elapsed


def save_snapshot(dataset, lists):
    """Save PGDATA as tar.gz snapshot."""
    snapshot_name = f"{dataset}_parallel_ivfflat_lists{lists}_pgdata.tar.gz"
    snapshot_path = SNAPSHOT_DIR / snapshot_name

    print(f"\n  Saving snapshot: {snapshot_path}")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Force checkpoint to flush all dirty buffers to disk before stopping
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CHECKPOINT")
        cur.close()
        conn.close()
        print("  Checkpoint completed.")
    except Exception as e:
        print(f"  Checkpoint warning: {e}")

    # Stop postgres
    stop_postgres()

    # Create tarball with pigz if available
    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"tar -cf - -C {PG_DATA_DIR} . | pigz > {snapshot_path}"
    else:
        cmd = f"tar -czf {snapshot_path} -C {PG_DATA_DIR} ."

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    size_mb = snapshot_path.stat().st_size / (1024 * 1024)
    print(f"  Snapshot saved: {size_mb:.0f} MB in {elapsed:.1f}s")

    # Restart postgres
    restart_postgres()

    return str(snapshot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build Parallel IVFFlat index (using lists from tune_ivfflat results) and save snapshot"
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS.keys()),
        help="Dataset to use (gist, sift, deep)"
    )
    parser.add_argument(
        "--lists", type=int, default=None,
        help="Override lists parameter (default: read from tune_ivfflat_results_{dataset}.json)"
    )
    args = parser.parse_args()

    dataset = args.dataset
    cfg = DATASETS[dataset]

    print(f"{'='*60}")
    print(f"  Build Parallel IVFFlat Index")
    print(f"  Dataset: {dataset}")
    print(f"  NUMA node: {NUMA_NODE}")
    print(f"{'='*60}")

    # Step 1: Get lists parameter
    if args.lists:
        lists = args.lists
        print(f"\n[1/4] Using override lists={lists}")
    else:
        print(f"\n[1/4] Loading IVFFlat tuning results...")
        tuning_data = load_ivfflat_tuning_results(dataset)
        lists = get_best_lists(tuning_data)
        if lists is None:
            print("  ERROR: Could not determine best lists from tuning results.")
            print("  Use --lists to specify manually.")
            sys.exit(1)
        print(f"  Best lists from tuning: {lists}")

    # Step 2: Compile parallel pgvector
    print(f"\n[2/4] Compiling parallel pgvector extension...")
    compile_parallel_pgvector()

    # Step 3: Drop existing indexes and build
    print(f"\n[3/4] Building parallel IVFFlat index (lists={lists})...")
    drop_existing_vector_indexes(cfg["table"])
    build_time = build_index(cfg["table"], cfg["ops_class"], cfg["index_name"], lists)

    # Step 4: Save snapshot
    print(f"\n[4/4] Saving PGDATA snapshot...")
    snapshot_path = save_snapshot(dataset, lists)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Dataset:       {dataset}")
    print(f"  Lists:         {lists}")
    print(f"  Build time:    {build_time:.1f}s")
    print(f"  Snapshot:      {snapshot_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
