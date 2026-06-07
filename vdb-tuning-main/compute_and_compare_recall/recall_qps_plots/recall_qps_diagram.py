#!/usr/bin/env python3
"""
Recall vs QPS Diagram: Measures throughput at various recall targets for IVFFlat, HNSW,
and Parallel IVFFlat indexes.

For each index type:
  1. Reads tuning JSON to find the best build parameters (lists, m, ef_construction)
  2. Loads the cached PGDATA snapshot from /mnt/nvme/saved_indexes/ instead of rebuilding
  3. Recompiles the appropriate pgvector extension (default vs parallel)
  4. Sweeps probes/ef_search to find the parameter that achieves each recall target
  5. Measures QPS (queries per second) at each recall point

For parallel_ivfflat specifically, tests multiple NUM_WORKERS values (4, 8, 12) and
saves a separate result file for each worker count.

Supports --warm (prewarm index before measuring) and --cold (restart + drop caches).

Usage:
    sudo python3 recall_qps_diagram.py --dataset gist --mode warm
    sudo python3 recall_qps_diagram.py --dataset gist --mode cold
    sudo python3 recall_qps_diagram.py --dataset gist --mode warm --index ivfflat
    sudo python3 recall_qps_diagram.py --dataset deep --mode warm --index all
    sudo python3 recall_qps_diagram.py --dataset gist --mode warm --index parallel_ivfflat

Output:
    results_recall_qps/recall_qps_{dataset}_{index}_{mode}.json
    results_recall_qps/recall_qps_{dataset}_parallel_ivfflat_w{N}_{mode}.json  (for each worker count)
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
SCRIPT_DIR = Path(__file__).resolve().parent                    # recall_qps_plots/
RECALL_DIR = SCRIPT_DIR.parent                                  # compute_and_compare_recall/
PROJECT_ROOT = RECALL_DIR.parent                                # vdb-tuning-main/

# Central machine-specific config (paths) from config.yaml at the repo root.
sys.path.insert(0, str(PROJECT_ROOT.parent))  # config_loader lives at ~/vdb_tuning
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
    _config_path = PROJECT_ROOT / "TPCH" / "code" / "experiment_config.json"
if _config_path.exists():
    with open(_config_path, "r") as _f:
        _hw_cfg = json.load(_f).get("hardware", {})
    NUMA_NODE = str(_hw_cfg.get("postgres_numa_nodes", "0"))
else:
    NUMA_NODE = "0"

# pgvector source directories (one level above PROJECT_DIR)
PGVECTOR_DEFAULT_DIR = Path(get_path("pgvector_default"))
PGVECTOR_PARALLEL_DIR = Path(get_path("pgvector_parallel"))

# The ivfscan.c file where NUM_WORKERS is defined
IVFSCAN_PATH = PGVECTOR_PARALLEL_DIR / "src" / "ivfscan.c"

# === PARALLEL WORKER COUNTS TO TEST ===
PARALLEL_WORKER_COUNTS = [4, 8, 12]

# === DATASET CONFIGS ===
DATASETS = {
    "gist": {
        "table": "gist_data",
        "workload_file": "gist_test_queries.txt",
        "exact_results_dir": "exact_results_gist",
        "distance_op": "<->",
        "ops_class": "vector_l2_ops",
        "k": 100,
    },
    "sift": {
        "table": "sift_data",
        "workload_file": "sift_test_queries.txt",
        "exact_results_dir": "exact_results_sift",
        "distance_op": "<->",
        "ops_class": "vector_l2_ops",
        "k": 100,
    },
    "deep": {
        "table": "deep_data",
        "workload_file": "deep_test_queries.txt",
        "exact_results_dir": "exact_results_deep",
        "distance_op": "<=>",
        "ops_class": "vector_cosine_ops",
        "k": 100,
    },
}

# === RECALL TARGETS ===
RECALL_TARGETS = [0.7, 0.8, 0.9, 0.95, 0.96, 0.98]

# === PROBE/EF_SEARCH SWEEP VALUES ===
IVFFLAT_PROBES_LIST = [30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500, 700, 1000]
HNSW_EF_SEARCH_LIST = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000]

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
def start_postgres():
    """Start PostgreSQL pinned to NUMA node."""
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={NUMA_NODE} --membind={NUMA_NODE} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True,
    )
    time.sleep(3)
    wait_for_postgres()
def restart_postgres_cold():
    """Cold restart: stop, drop caches, start."""
    print("  Cold restart (drop caches)...")
    stop_postgres()
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)
    start_postgres()
def prewarm_index(table_name):
    """Prewarm the index using pg_prewarm if available."""
    print("  Prewarming index...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        # Create extension if not exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        # Find the index name
        cur.execute(f"""
            SELECT i.relname FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = '{table_name}'
            AND am.amname IN ('ivfflat', 'hnsw')
        """)
        result = cur.fetchone()
        if result:
            idx_name = result[0]
            cur.execute(f"SELECT pg_prewarm('{idx_name}')")
            blocks = cur.fetchone()[0]
            print(f"    Prewarmed index '{idx_name}': {blocks} blocks")
            # Also prewarm the table
            cur.execute(f"SELECT pg_prewarm('{table_name}')")
            tblocks = cur.fetchone()[0]
            print(f"    Prewarmed table '{table_name}': {tblocks} blocks")
        else:
            print("    WARNING: No vector index found to prewarm!")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"    Prewarm error: {e}")
def find_snapshot(dataset, index_type, params):
    """Find a cached PGDATA snapshot. Tries both tar.gz and directory formats."""
    # Build possible names
    if index_type == "parallel_ivfflat":
        # Parallel IVFFlat has its own snapshot: {dataset}_parallel_ivfflat_lists{N}_pgdata[.tar.gz]
        lists = params["lists"]
        base_name = f"{dataset}_parallel_ivfflat_lists{lists}_pgdata"
    elif index_type == "ivfflat":
        # IVFFlat naming: {dataset}_ivfflat_lists{N}_pgdata[.tar.gz]
        lists = params["lists"]
        base_name = f"{dataset}_ivfflat_lists{lists}_pgdata"
    elif index_type == "hnsw":
        # HNSW naming: {dataset}_hnsw_efc{N}_m{N}_pgdata[.tar.gz]
        m = params["m"]
        efc = params["ef_construction"]
        base_name = f"{dataset}_hnsw_efc{efc}_m{m}_pgdata"
    else:
        return None

    # Check tar.gz first, then directory
    tar_path = SNAPSHOT_DIR / f"{base_name}.tar.gz"
    dir_path = SNAPSHOT_DIR / base_name

    if tar_path.exists():
        return tar_path
    elif dir_path.exists():
        return dir_path
    else:
        # Try alternate naming conventions
        if index_type == "hnsw":
            alt_name = f"{dataset}_hnsw_m{m}_efc{efc}_pgdata"
            if (SNAPSHOT_DIR / f"{alt_name}.tar.gz").exists():
                return SNAPSHOT_DIR / f"{alt_name}.tar.gz"
            if (SNAPSHOT_DIR / alt_name).exists():
                return SNAPSHOT_DIR / alt_name
        return None
def load_snapshot(snapshot_path):
    """Load a PGDATA snapshot (supports both tar.gz and directory formats)."""
    print(f"  Loading snapshot: {snapshot_path.name}...")
    stop_postgres()

    # Clear existing PGDATA
    subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)

    start = time.time()
    if str(snapshot_path).endswith(".tar.gz"):
        # tar.gz format
        pigz = shutil.which("pigz")
        if pigz:
            cmd = f"pigz -dc {snapshot_path} | tar -xf - -C {PG_DATA_DIR}"
        else:
            cmd = f"tar -xzf {snapshot_path} -C {PG_DATA_DIR}"
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Directory format - use cp -a
        subprocess.run(f"cp -a {snapshot_path}/. {PG_DATA_DIR}/", shell=True, check=True)

    elapsed = time.time() - start
    print(f"    Loaded in {elapsed:.1f}s")

    # Fix permissions
    subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    subprocess.run(f"chmod 700 {PG_DATA_DIR}", shell=True, check=True)

    start_postgres()
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
    print(f"    Verified: {verify.stdout.strip()}")
def compile_pgvector(variant="default"):
    """Compile and install pgvector extension (default or parallel), then restart postgres."""
    if variant == "default":
        src_dir = PGVECTOR_DEFAULT_DIR
    else:
        src_dir = PGVECTOR_PARALLEL_DIR

    if not src_dir.exists():
        print(f"  ERROR: pgvector source not found at {src_dir}")
        sys.exit(1)

    print(f"  Compiling pgvector ({variant}) from {src_dir}...")
    env_prefix = f"export PATH={PG_INSTALL}/bin:$PATH"
    result = subprocess.run(
        f"cd {src_dir} && {env_prefix} && make clean > /dev/null 2>&1; make -j$(nproc) && make install",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    Compile ERROR: {result.stderr[:500]}")
        sys.exit(1)
    print(f"    Compiled and installed successfully.")

    # Restart postgres to load new .so
    print(f"    Restarting PostgreSQL to load new extension binary...")
    stop_postgres()
    start_postgres()
def load_tuning_results(dataset, index_type):
    """Load tuning results JSON to find best build parameters."""
    tune_dir = RECALL_DIR / "results_from_tune_scripts"

    if index_type in ("ivfflat", "parallel_ivfflat"):
        # For parallel_ivfflat, we use the same index structure as ivfflat
        fname = f"tune_ivfflat_results_{dataset}.json"
        fallback = "tune_ivfflat_results.json"
    else:
        fname = f"tune_hnsw_results_{dataset}.json"
        fallback = "tune_hnsw_results.json"

    fpath = tune_dir / fname
    if not fpath.exists():
        fpath = tune_dir / fallback
    if not fpath.exists():
        print(f"  ERROR: Tuning results not found: {fname} or {fallback}")
        print(f"         Looked in: {tune_dir}")
        return None

    with open(fpath, "r") as f:
        data = json.load(f)

    return data
def get_best_params(tuning_data, index_type):
    """Extract best build parameters from tuning results."""
    if tuning_data is None:
        return None

    # Check for best_configuration key
    if "best_configuration" in tuning_data:
        best = tuning_data["best_configuration"]
        if index_type in ("ivfflat", "parallel_ivfflat"):
            return {"lists": best["lists"]}
        else:
            return {"m": best.get("m", 16), "ef_construction": best.get("ef_construction", 200)}

    # Fallback: pick configuration with fastest query time that achieved target recall
    configs = tuning_data.get("configurations", [])
    achieved = [c for c in configs if c.get("achieved_recall") and c["achieved_recall"] >= tuning_data.get("target_recall", 0.9)]

    if not achieved:
        # Just pick the one with highest recall
        achieved = sorted(configs, key=lambda c: c.get("achieved_recall", 0) or 0, reverse=True)

    if achieved:
        best = min(achieved, key=lambda c: c.get("avg_query_time_ms", float("inf")) or float("inf"))
        if index_type in ("ivfflat", "parallel_ivfflat"):
            return {"lists": best["lists"]}
        else:
            return {"m": best["m"], "ef_construction": best["ef_construction"]}

    return None
def load_workload(dataset_cfg):
    """Load query vectors from workload file."""
    workload_path = PROJECT_ROOT / dataset_cfg["workload_file"]
    if not workload_path.exists():
        # Try SCRIPT_DIR parent
        workload_path = PROJECT_ROOT / dataset_cfg["workload_file"]
    if not workload_path.exists():
        # Try several common locations
        for candidate in [
            SCRIPT_DIR / dataset_cfg["workload_file"],
            Path("/users/mangelis/vdb_tuning/vdb-tuning-main") / dataset_cfg["workload_file"],
        ]:
            if candidate.exists():
                workload_path = candidate
                break

    if not workload_path.exists():
        print(f"  ERROR: Workload file not found: {dataset_cfg['workload_file']}")
        print(f"         Searched in: {PROJECT_DIR}, {SCRIPT_DIR.parent}")
        sys.exit(1)

    print(f"  Loading workload from {workload_path}...")
    with open(workload_path, "r") as f:
        content = f.read().strip()
        # Try JSON first, then eval
        try:
            workload = json.loads(content)
        except json.JSONDecodeError:
            workload = eval(content)

    # Normalize to string format
    result = []
    for q in workload:
        if isinstance(q, list):
            result.append("[" + ",".join(str(x) for x in q) + "]")
        else:
            result.append(q)

    print(f"    Loaded {len(result)} query vectors")
    return result
def load_exact_results(dataset, k):
    """Load precomputed exact (brute-force) results."""
    exact_dir = RECALL_DIR / "exact_results" / DATASETS[dataset]["exact_results_dir"]
    # Try various naming patterns
    candidates = [
        exact_dir / f"exact_search_results_1000_queries_{dataset}_K_{k}.json",
        exact_dir / f"exact_search_results_{dataset}_K_{k}.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r") as f:
                results = json.load(f)
            print(f"  Loaded exact results: {path.name} ({len(results)} queries)")
            return results

    print(f"  ERROR: Exact results not found in {exact_dir}")
    print(f"         Expected pattern: exact_search_results_*_{dataset}_K_{k}.json")
    sys.exit(1)
def compute_recall_and_qps(workload, exact_results, table_name, distance_op, k, param_name, param_value, mode="warm"):
    """
    Run ALL queries in workload, compute recall and QPS.
    Prewarming is handled externally via pg_prewarm (for warm mode).
    Returns (recall, qps, avg_latency_ms, query_times_ms).
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Set the search parameter
    if param_name == "ivfflat.probes":
        cur.execute(f"SET ivfflat.probes = {param_value}")
    elif param_name == "hnsw.ef_search":
        cur.execute(f"SET hnsw.ef_search = {param_value}")

    cur.execute("SET enable_seqscan = off")
    cur.execute("SET max_parallel_workers_per_gather = 200")

    # Measurement phase: run all queries, compute recall and timing per query
    total_recall = 0.0
    num_queries = len(workload)
    query_times = []

    for i in range(num_queries):
        vec_str = workload[i]

        q_start = time.time()
        cur.execute(f"SELECT id FROM {table_name} ORDER BY embedding {distance_op} '{vec_str}'::vector LIMIT {k}")
        result_ids = set(row[0] for row in cur.fetchall())
        q_elapsed = time.time() - q_start
        query_times.append(round(q_elapsed * 1000, 3))  # ms

        # Compute recall against exact results
        if i < len(exact_results):
            exact_ids = set(exact_results[i][:k])
            hits = len(result_ids & exact_ids)
            total_recall += hits / k

    total_time = sum(query_times) / 1000  # back to seconds
    avg_recall = total_recall / min(num_queries, len(exact_results))
    qps = num_queries / total_time if total_time > 0 else 0
    avg_latency_ms = total_time / num_queries * 1000 if num_queries > 0 else 0

    cur.close()
    conn.close()

    return avg_recall, qps, avg_latency_ms, query_times
def sweep_recall_qps(workload, exact_results, table_name, distance_op, k, index_type, mode, num_workers=4, num_iterations=1):
    """
    Sweep probes/ef_search values and measure recall + QPS at each point.
    Returns list of (param_value, recall, qps, avg_latency_ms) tuples.
    num_iterations: number of times to repeat the workload per param setting.
    """
    if index_type in ("ivfflat", "parallel_ivfflat"):
        param_name = "ivfflat.probes"
        # Parallel IVFFlat requires probes >= number of workers
        if index_type == "parallel_ivfflat":
            param_list = [p for p in IVFFLAT_PROBES_LIST if p >= num_workers]
        else:
            param_list = IVFFLAT_PROBES_LIST
        param_label = "probes"
    else:
        param_name = "hnsw.ef_search"
        param_list = HNSW_EF_SEARCH_LIST
        param_label = "ef_search"

    sweep_results = []

    for pval in param_list:
        all_query_times_ms = []
        all_recalls = []

        for iteration in range(num_iterations):
            # In cold mode, restart postgres and drop caches before each iteration
            if mode == "cold":
                restart_postgres_cold()
            elif mode == "warm" and iteration > 0:
                # Re-prewarm for consistent state on subsequent iterations
                prewarm_index(table_name)

            recall, qps, latency, query_times_ms = compute_recall_and_qps(
                workload, exact_results, table_name, distance_op, k, param_name, pval, mode
            )
            all_query_times_ms.extend(query_times_ms)
            all_recalls.append(recall)

            if num_iterations > 1:
                iter_avg = sum(query_times_ms) / len(query_times_ms) if query_times_ms else 0
                print(f"      iter {iteration+1}/{num_iterations}: recall={recall:.4f}, avg={iter_avg:.2f}ms")

        # Compute aggregated stats across all iterations
        avg_recall = sum(all_recalls) / len(all_recalls)
        total_time_s = sum(all_query_times_ms) / 1000
        num_queries_total = len(all_query_times_ms)
        avg_latency = sum(all_query_times_ms) / num_queries_total if num_queries_total else 0
        agg_qps = num_queries_total / total_time_s if total_time_s > 0 else 0

        sorted_times = sorted(all_query_times_ms)
        median_ms = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95_ms = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_ms = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0

        sweep_results.append({
            param_label: pval,
            "recall": round(avg_recall, 4),
            "qps": round(agg_qps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "median_latency_ms": round(median_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "p99_latency_ms": round(p99_ms, 2),
            "num_iterations": num_iterations,
            "query_times_ms": all_query_times_ms,
        })

        print(f"    {param_label}={pval:>5}: recall={avg_recall:.4f}, QPS={agg_qps:.1f}, avg={avg_latency:.2f}ms, median={median_ms:.2f}ms, p95={p95_ms:.2f}ms")

        # Early stop if we've exceeded max recall target
        if avg_recall >= max(RECALL_TARGETS) + 0.01:
            break

    return sweep_results
def find_recall_qps_points(sweep_results, index_type):
    """
    From sweep results, find the QPS at each recall target.
    For each target, pick the smallest param that achieves >= target recall.
    """
    if index_type in ("ivfflat", "parallel_ivfflat"):
        param_label = "probes"
    else:
        param_label = "ef_search"

    points = []
    for target in RECALL_TARGETS:
        # Find first param that achieves target
        achieved = [r for r in sweep_results if r["recall"] >= target]
        if achieved:
            best = achieved[0]  # first (smallest param) that achieves target
            points.append({
                "target_recall": target,
                "actual_recall": best["recall"],
                param_label: best[param_label],
                "qps": best["qps"],
                "avg_latency_ms": best["avg_latency_ms"],
            })
        else:
            # Could not achieve this recall
            if sweep_results:
                best = max(sweep_results, key=lambda r: r["recall"])
                points.append({
                    "target_recall": target,
                    "actual_recall": best["recall"],
                    param_label: best[param_label],
                    "qps": best["qps"],
                    "avg_latency_ms": best["avg_latency_ms"],
                    "note": "max achievable",
                })
            else:
                points.append({
                    "target_recall": target,
                    "actual_recall": 0,
                    "qps": 0,
                    "avg_latency_ms": 0,
                    "note": "no data",
                })

    return points
def run_for_index(dataset, index_type, mode, dataset_cfg, workload, exact_results, num_workers=None, skip_snapshot=False, num_iterations=1):
    """Run the full recall-vs-QPS measurement for one index type.
    
    If num_workers is specified (for parallel_ivfflat), sets NUM_WORKERS before compiling.
    """
    worker_label = f" (workers={num_workers})" if num_workers else ""
    print(f"\n{'='*70}")
    print(f"  INDEX TYPE: {index_type.upper()}{worker_label}")
    print(f"  Dataset: {dataset}, Mode: {mode}")
    print(f"{'='*70}")

    # Step 1: Compile appropriate pgvector extension
    if index_type == "parallel_ivfflat":
        if num_workers is not None:
            set_num_workers(num_workers)
        compile_pgvector("parallel")
    else:
        compile_pgvector("default")

    # Step 2: Load tuning results to find best build params
    tuning_data = load_tuning_results(dataset, index_type)
    best_params = get_best_params(tuning_data, index_type)

    if best_params is None:
        print(f"  ERROR: Could not determine best params for {index_type}")
        return None

    if index_type in ("ivfflat", "parallel_ivfflat"):
        print(f"  Best params: lists={best_params['lists']}")
    else:
        print(f"  Best params: m={best_params['m']}, ef_construction={best_params['ef_construction']}")

    # Step 3: Find and load snapshot
    snapshot_path = find_snapshot(dataset, index_type, best_params)
    if snapshot_path is None:
        print(f"  ERROR: No cached snapshot found for {index_type} with params {best_params}")
        print(f"         Expected in: {SNAPSHOT_DIR}")
        print(f"         Available snapshots:")
        if SNAPSHOT_DIR.exists():
            for f in sorted(SNAPSHOT_DIR.iterdir()):
                print(f"           {f.name}")
        return None

    if not skip_snapshot:
        load_snapshot(snapshot_path)

    # Step 4: Prepare for measurement (warm or cold)
    table_name = dataset_cfg["table"]
    distance_op = dataset_cfg["distance_op"]
    k = dataset_cfg["k"]

    if mode == "warm":
        prewarm_index(table_name)
    else:
        restart_postgres_cold()

    # Step 5: Sweep probes/ef_search
    print(f"\n  Sweeping parameters (measuring recall + QPS, iterations={num_iterations})...")
    sweep_results = sweep_recall_qps(
        workload, exact_results, table_name, distance_op, k, index_type, mode, num_workers=(num_workers or 4), num_iterations=num_iterations
    )

    # Step 6: Extract recall-vs-QPS points
    points = find_recall_qps_points(sweep_results, index_type)

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {index_type.upper()}{worker_label} ({mode})")
    print(f"  {'='*50}")
    print(f"  {'Target':>8} {'Actual':>8} {'QPS':>10} {'Latency':>10}")
    print(f"  {'-'*40}")
    for pt in points:
        print(f"  {pt['target_recall']:>8.2f} {pt['actual_recall']:>8.4f} {pt['qps']:>10.1f} {pt['avg_latency_ms']:>8.2f}ms")

    result_data = {
        "index_type": index_type,
        "dataset": dataset,
        "mode": mode,
        "build_params": best_params,
        "snapshot_used": str(snapshot_path),
        "k": k,
        "recall_targets": RECALL_TARGETS,
        "recall_qps_points": points,
        "full_sweep": sweep_results,
        "timestamp": datetime.now().isoformat(),
    }

    if num_workers is not None:
        result_data["num_workers"] = num_workers

    return result_data
def main():
    parser = argparse.ArgumentParser(description="Recall vs QPS diagram measurement")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Dataset to use (gist, sift, deep)")
    parser.add_argument("--mode", required=True, choices=["warm", "cold"],
                        help="Measurement mode: warm (prewarm) or cold (restart+drop caches)")
    parser.add_argument("--index", default="all",
                        choices=["ivfflat", "hnsw", "parallel_ivfflat", "all"],
                        help="Which index type(s) to test")
    parser.add_argument("--k", type=int, default=None,
                        help="Override K value (default: from dataset config)")
    parser.add_argument("--workers", type=int, nargs='+', default=None,
                        help="Override worker counts for parallel_ivfflat (default: 4 8 12)")
    parser.add_argument("--num-iterations", type=int, default=1,
                        help="Number of times to repeat the workload per param setting (default: 1)")
    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode
    num_iterations = args.num_iterations
    index_types = ["ivfflat", "hnsw", "parallel_ivfflat"] if args.index == "all" else [args.index]
    worker_counts = args.workers if args.workers else PARALLEL_WORKER_COUNTS

    dataset_cfg = DATASETS[dataset]
    if args.k:
        dataset_cfg["k"] = args.k
    k = dataset_cfg["k"]

    print(f"{'#'*70}")
    print(f"  Recall vs QPS Diagram")
    print(f"  Dataset: {dataset}, K: {k}, Mode: {mode}")
    print(f"  Index types: {index_types}")
    if "parallel_ivfflat" in index_types:
        print(f"  Parallel worker counts: {worker_counts}")
    print(f"  Iterations per param: {num_iterations}")
    print(f"  Recall targets: {RECALL_TARGETS}")
    print(f"{'#'*70}")

    # Load workload
    workload = load_workload(dataset_cfg)

    # Load exact results
    exact_results = load_exact_results(dataset, k)

    # Results output directory
    results_dir = SCRIPT_DIR / "results_recall_qps"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run for each index type
    all_results = {}
    for idx_type in index_types:
        if idx_type == "parallel_ivfflat":
            # Load snapshot only once for the first worker count
            for i, nw in enumerate(worker_counts):
                result_key = f"parallel_ivfflat_w{nw}"
                result = run_for_index(dataset, idx_type, mode, dataset_cfg, workload, exact_results, num_workers=nw, skip_snapshot=(i > 0), num_iterations=num_iterations)
                if result:

                    all_results[result_key] = result

                    # Save individual file for this worker count
                    output_file = results_dir / f"recall_qps_{dataset}_parallel_ivfflat_w{nw}_{mode}.json"
                    with open(output_file, "w") as f:
                        json.dump({result_key: result}, f, indent=2)
                    print(f"\n  Saved: {output_file}")
        else:
            result = run_for_index(dataset, idx_type, mode, dataset_cfg, workload, exact_results, num_iterations=num_iterations)
            if result:
                all_results[idx_type] = result

                # Save individual file
                output_file = results_dir / f"recall_qps_{dataset}_{idx_type}_{mode}.json"
                with open(output_file, "w") as f:
                    json.dump({idx_type: result}, f, indent=2)
                print(f"\n  Saved: {output_file}")

    # Save combined results file
    if args.index == "all":
        combined_file = results_dir / f"recall_qps_{dataset}_all_{mode}.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined results saved: {combined_file}")

    print(f"\n{'#'*70}")
    print(f"  ALL RESULTS SAVED TO: {results_dir}")
    print(f"{'#'*70}")

    # Print summary comparison table
    if len(all_results) > 1:
        print(f"\n  COMPARISON TABLE (QPS at each recall target):")
        header = f"  {'Recall':>8}"
        for key in all_results:
            print_key = key if len(key) <= 18 else key[:18]
            header += f"  {print_key:>18}"
        print(header)
        print(f"  {'-'*8}" + f"  {'-'*18}" * len(all_results))

        for target in RECALL_TARGETS:
            line = f"  {target:>8.2f}"
            for key, data in all_results.items():
                pts = [p for p in data["recall_qps_points"] if p["target_recall"] == target]
                if pts:
                    line += f"  {pts[0]['qps']:>10.1f} QPS"
                else:
                    line += f"  {'N/A':>18}"
            print(line)

    print("\nDone!")
if __name__ == "__main__":
    main()
