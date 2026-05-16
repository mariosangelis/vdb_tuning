#!/usr/bin/env python3
"""
Automated benchmark: Measure OLAP (TPC-H) interference on ANN (TopK) performance.

This script automatically runs experiments across multiple memory configurations,
with multiple iterations per configuration for statistical reliability.

Memory Configurations:
  - No Pressure: 40GB shared_buffers, 64GB cgroup
  - Relaxed: 16GB shared_buffers, 20GB cgroup
  - Medium: 12GB shared_buffers, 16GB cgroup
  - Tight: 8GB shared_buffers, 12GB cgroup
"""

import psycopg2
import threading
import time
import subprocess
import json
import re
import random
import os
import statistics
from pathlib import Path
import psutil
from datetime import datetime

def vec_to_str(vec):
    """Convert Python list to PostgreSQL vector string format."""
    return '[' + ','.join(str(x) for x in vec) + ']'

# ============== CONFIGURATION ==============
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "secret",
    "host": "localhost",
    "port": "5432"
}

# Paths
BENCHBASE_DIR = "/users/mangelis/vdb-tuning-main/benchbase/target/benchbase-postgres"
WORKLOAD_VECTORS_PATH = "../gist_test_queries.txt"
RESULTS_DIR = "/users/mangelis/vdb-tuning-main/TPCH"

# CPU pinning
POSTGRES_CORES = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
BENCHBASE_CORES = "24,25,26,27,28,29,30,31"
TOPK_CORES = [24, 25, 26, 27, 28, 29, 30, 31]

# PostgreSQL paths
PG_CTL_PATH = "/usr/local/pgsql/bin/pg_ctl"
PG_DATA_DIR = "/mnt/nvme/pgdata"
PG_CONF_PATH = "/mnt/nvme/pgdata/postgresql.conf"
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"

# Experiment parameters
K = 100  # TopK parameter
NUM_TOPK_WORKERS = 1  # Fixed number of ANN workers
TPCH_WORKERS_LIST = [6, 5, 4, 3, 2, 1]  # Variable TPC-H concurrency
NUM_ITERATIONS = 3  # Number of iterations per configuration

# Index configuration (modify as needed)
INDEX_TYPE = "IVFFLAT"  # or "IVFFLAT", "HNSW"
IO_DEPTH = 0  # or 0

# Memory configurations: (name, shared_buffers_gb, cgroup_gb)
MEMORY_CONFIGS = [
    ("tight", 8, 12),
]

# Query template for GIST dataset - using L2 distance
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %s"


def get_results_filename(index_type, io_depth, config_name, shared_buf_gb, cgroup_gb):
    """Generate descriptive results filename."""
    return (f"OLAP_IMPACT_mixed_workload_tpch_and_topk_{index_type}_io_{io_depth}_"
            f"MULTIPLE_ITERATIONS_NO_HYPERTHREADING_16_cores_for_postgres_"
            f"{cgroup_gb}GB_RAM_{shared_buf_gb}_shared_buffers_GIST.txt")


def set_shared_buffers(size_gb):
    """Modify shared_buffers in postgresql.conf."""
    print(f"Setting shared_buffers to {size_gb}GB...")

    # Use round numbers in kB (easier to read)
    size_kb = size_gb * 1000000  # e.g., 40GB -> 40000000kB

    # Read current config
    with open(PG_CONF_PATH, 'r') as f:
        config = f.read()

    # Replace shared_buffers line - capture everything after = until end of line
    import re
    new_config = re.sub(
        r'(shared_buffers\s*=\s*)[^\n]+',
        f'\\g<1>{size_kb}kB',
        config
    )

    # Write back
    with open(PG_CONF_PATH, 'w') as f:
        f.write(new_config)

    # Verify the change
    with open(PG_CONF_PATH, 'r') as f:
        verify = f.read()
    match = re.search(r'shared_buffers\s*=\s*[^\n]+', verify)
    print(f"shared_buffers now: {match.group() if match else 'NOT FOUND'}")


def set_cgroup_memory(size_gb):
    """Set cgroup memory limit."""
    print(f"Setting cgroup memory to {size_gb}GB...")
    size_bytes = size_gb * 1024 * 1024 * 1024

    with open(CGROUP_MEM_PATH, 'w') as f:
        f.write(str(size_bytes))

    # Verify
    with open(CGROUP_MEM_PATH, 'r') as f:
        actual = int(f.read().strip())

    print(f"cgroup memory set to {actual / (1024**3):.1f}GB")


def wait_for_postgres(max_attempts=60):
    """Wait for PostgreSQL to be ready."""
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            print("PostgreSQL is ready.")
            return True
        except psycopg2.OperationalError:
            print(f"Waiting for PostgreSQL... attempt {attempt + 1}")
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def restart_postgres():
    """Restart PostgreSQL with CPU pinning."""
    # Drop caches
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

    # Restart PostgreSQL pinned to NUMA 0
    subprocess.run(
        f"su - postgres -c 'taskset -c {POSTGRES_CORES} {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True,
        check=True
    )

    time.sleep(3)
    wait_for_postgres()

    # Move all postgres processes into cgroup
    subprocess.run(
        "for pid in $(pgrep -u postgres); do echo $pid > /sys/fs/cgroup/postgres_test/cgroup.procs; done",
        shell=True, check=False
    )


def create_tpch_config(num_terminals):
    """Create a TPC-H config file with specified number of terminals."""
    config_content = f'''<?xml version="1.0"?>
<parameters>
    <type>POSTGRES</type>
    <driver>org.postgresql.Driver</driver>
    <url>jdbc:postgresql://localhost:5432/{DB_CONFIG["dbname"]}?sslmode=disable&amp;ApplicationName=tpch&amp;reWriteBatchedInserts=true</url>
    <username>{DB_CONFIG["user"]}</username>
    <password>{DB_CONFIG["password"]}</password>
    <reconnectOnConnectionFailure>true</reconnectOnConnectionFailure>
    <isolation>TRANSACTION_SERIALIZABLE</isolation>
    <batchsize>1024</batchsize>

    <scalefactor>10</scalefactor>

    <terminals>1</terminals>
    <works>
        <work>
            <serial>true</serial>
            <rate>unlimited</rate>
            <weights>1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1</weights>
        </work>
    </works>

    <transactiontypes>
        <transactiontype><name>Q1</name><id>1</id></transactiontype>
        <transactiontype><name>Q2</name><id>2</id></transactiontype>
        <transactiontype><name>Q3</name><id>3</id></transactiontype>
        <transactiontype><name>Q4</name><id>4</id></transactiontype>
        <transactiontype><name>Q5</name><id>5</id></transactiontype>
        <transactiontype><name>Q6</name><id>6</id></transactiontype>
        <transactiontype><name>Q7</name><id>7</id></transactiontype>
        <transactiontype><name>Q8</name><id>8</id></transactiontype>
        <transactiontype><name>Q9</name><id>9</id></transactiontype>
        <transactiontype><name>Q10</name><id>10</id></transactiontype>
        <transactiontype><name>Q11</name><id>11</id></transactiontype>
        <transactiontype><name>Q12</name><id>12</id></transactiontype>
        <transactiontype><name>Q13</name><id>13</id></transactiontype>
        <transactiontype><name>Q14</name><id>14</id></transactiontype>
        <transactiontype><name>Q15</name><id>15</id></transactiontype>
        <transactiontype><name>Q16</name><id>16</id></transactiontype>
        <transactiontype><name>Q17</name><id>17</id></transactiontype>
        <transactiontype><name>Q18</name><id>18</id></transactiontype>
        <transactiontype><name>Q19</name><id>19</id></transactiontype>
        <transactiontype><name>Q20</name><id>20</id></transactiontype>
        <transactiontype><name>Q21</name><id>21</id></transactiontype>
        <transactiontype><name>Q22</name><id>22</id></transactiontype>
    </transactiontypes>
</parameters>'''

    config_path = f"/tmp/tpch_config_{num_terminals}_terminals.xml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path


def run_topk_queries(thread_id, workload, topk_latencies, stop_event):
    """Worker thread that runs TopK queries continuously."""
    local_random = random.Random(thread_id + 42)

    # Pin thread to specific core
    core_id = TOPK_CORES[thread_id % len(TOPK_CORES)]
    psutil.Process().cpu_affinity([core_id])

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        while not stop_event.is_set():
            index = local_random.randint(0, len(workload) - 1)
            vec = workload[index]
            vec_str = vec_to_str(vec)

            # Run query with EXPLAIN ANALYZE to get execution time
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {TOPK_QUERY}" % (vec_str, K))
            explain_result = cur.fetchall()

            # Parse execution time
            for row in explain_result:
                line = row[0].strip()
                if 'Execution Time:' in line:
                    elapsed_ms = float(line.split()[2])
                    topk_latencies.append(elapsed_ms)
                    break
    finally:
        cur.close()
        conn.close()


def calculate_percentiles(latencies):
    """Calculate latency percentiles."""
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0, "count": 0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "p50": sorted_latencies[int(n * 0.50)],
        "p95": sorted_latencies[int(n * 0.95)],
        "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        "avg": statistics.mean(sorted_latencies),
        "min": sorted_latencies[0],
        "max": sorted_latencies[-1],
        "count": n
    }


def run_single_experiment(num_tpch_workers, workload):
    """Run a single experiment with given TPC-H concurrency."""
    topk_latencies = []
    stop_event = threading.Event()

    print(f"\n{'='*60}")
    print(f"Running experiment: {num_tpch_workers} TPC-H workers")
    print(f"{'='*60}")

    # Restart PostgreSQL with clean state
    restart_postgres()

    # Start TopK worker threads
    threads = []
    for thread_id in range(NUM_TOPK_WORKERS):
        t = threading.Thread(target=run_topk_queries, args=(thread_id, workload, topk_latencies, stop_event))
        t.start()
        threads.append(t)

    tpch_summary = None

    # Create config with 1 terminal (we spawn multiple processes instead)
    config_path = create_tpch_config(1)

    # Start multiple BenchBase processes (one per worker)
    print(f"Starting TPC-H with {num_tpch_workers} workers...")
    benchbase_procs = []
    for i in range(num_tpch_workers):
        proc = subprocess.Popen([
            "taskset", "-c", BENCHBASE_CORES,
            "java", "-jar", "benchbase.jar", "-b", "tpch",
            "-c", config_path,
            "--create=false", "--load=false", "--execute=true"
        ], cwd=BENCHBASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64"})
        benchbase_procs.append(proc)
        print(f"  Started BenchBase process {i+1} (PID: {proc.pid})")

    # Wait for all to complete
    results_list = []
    for i, proc in enumerate(benchbase_procs):
        stdout, stderr = proc.communicate()
        results_list.append((stdout, stderr, proc.returncode))
        print(f"  BenchBase process {i+1} finished")

    # Use first successful result for summary
    output = results_list[0][0] + results_list[0][1]
    match = re.search(r"tpch_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.summary\.json", output)
    if match:
        timestamp = match.group(1)
        results_dir = Path(BENCHBASE_DIR) / "results"
        summary_file = results_dir / f"tpch_{timestamp}.summary.json"

        try:
            with open(summary_file, "r") as f:
                content = f.read()
                # Handle case where file has multiple JSON objects
                # Just parse the first one
                decoder = json.JSONDecoder()
                tpch_summary, _ = decoder.raw_decode(content)
        except Exception as e:
            print(f"Warning: Error reading summary file: {e}")
            tpch_summary = None

        if tpch_summary:
            tpch_execution_time = tpch_summary["Elapsed Time (nanoseconds)"] / 1_000_000_000
        else:
            tpch_execution_time = 0
    else:
        tpch_execution_time = 0

    # Stop TopK threads
    stop_event.set()
    for t in threads:
        t.join()

    # Calculate results
    latency_stats = calculate_percentiles(topk_latencies)
    topk_throughput = latency_stats["count"] / tpch_execution_time if tpch_execution_time > 0 else 0
    topk_avg_exec_time = latency_stats["avg"] / 1000.0 if latency_stats["count"] > 0 else 0

    return {
        "tpch_workers": num_tpch_workers,
        "topk_workers": NUM_TOPK_WORKERS,
        "summary": tpch_summary,
        "topk_query_count": latency_stats["count"],
        "topk_throughput": topk_throughput,
        "topk_avg_exec_time_sec": topk_avg_exec_time,
        "tpch_execution_time": tpch_execution_time,
        "topk_latency_p50_ms": latency_stats["p50"],
        "topk_latency_p95_ms": latency_stats["p95"],
        "topk_latency_p99_ms": latency_stats["p99"],
    }


def aggregate_results(results_list):
    """Aggregate results from multiple iterations."""
    if not results_list:
        return None

    # Calculate means and std devs
    aggregated = {
        "tpch_workers": results_list[0]["tpch_workers"],
        "topk_workers": results_list[0]["topk_workers"],
        "num_iterations": len(results_list),

        # Means
        "topk_throughput_mean": statistics.mean(r["topk_throughput"] for r in results_list),
        "topk_throughput_std": statistics.stdev(r["topk_throughput"] for r in results_list) if len(results_list) > 1 else 0,

        "tpch_throughput_mean": statistics.mean(r["summary"]["Throughput (requests/second)"] for r in results_list if r["summary"]),
        "tpch_throughput_std": statistics.stdev(r["summary"]["Throughput (requests/second)"] for r in results_list if r["summary"]) if len(results_list) > 1 else 0,

        "topk_latency_p50_mean": statistics.mean(r["topk_latency_p50_ms"] for r in results_list),
        "topk_latency_p50_std": statistics.stdev(r["topk_latency_p50_ms"] for r in results_list) if len(results_list) > 1 else 0,

        "topk_latency_p95_mean": statistics.mean(r["topk_latency_p95_ms"] for r in results_list),
        "topk_latency_p99_mean": statistics.mean(r["topk_latency_p99_ms"] for r in results_list),

        "tpch_execution_time_mean": statistics.mean(r["tpch_execution_time"] for r in results_list),
        "topk_query_count_mean": statistics.mean(r["topk_query_count"] for r in results_list),

        # Raw results for reference
        "raw_results": results_list,
    }

    return aggregated


def run_memory_config(config_name, shared_buf_gb, cgroup_gb, workload):
    """Run all experiments for a single memory configuration."""
    print(f"\n{'#'*70}")
    print(f"# MEMORY CONFIG: {config_name}")
    print(f"# shared_buffers: {shared_buf_gb}GB, cgroup: {cgroup_gb}GB")
    print(f"{'#'*70}")

    # Configure memory settings
    set_shared_buffers(shared_buf_gb)
    set_cgroup_memory(cgroup_gb)

    # Results for this configuration
    config_results = []

    for num_tpch_workers in TPCH_WORKERS_LIST:
        iteration_results = []

        for iteration in range(NUM_ITERATIONS):
            print(f"\n>>> Iteration {iteration + 1}/{NUM_ITERATIONS} for {num_tpch_workers} TPC-H workers")
            result = run_single_experiment(num_tpch_workers, workload)
            iteration_results.append(result)

            # Print intermediate result
            print(f"  TopK throughput: {result['topk_throughput']:.2f} qps")
            print(f"  TopK p50 latency: {result['topk_latency_p50_ms']:.2f} ms")
            if result['summary']:
                print(f"  TPC-H throughput: {result['summary']['Throughput (requests/second)']:.4f} tps")

        # Aggregate results for this worker count
        aggregated = aggregate_results(iteration_results)
        config_results.append(aggregated)

        print(f"\n=== AGGREGATED for {num_tpch_workers} workers ===")
        print(f"  TopK throughput: {aggregated['topk_throughput_mean']:.2f} +/- {aggregated['topk_throughput_std']:.2f} qps")
        print(f"  TPC-H throughput: {aggregated['tpch_throughput_mean']:.4f} +/- {aggregated['tpch_throughput_std']:.4f} tps")

    return config_results


def main():
    print(f"Starting automated benchmark at {datetime.now()}")
    print(f"Index type: {INDEX_TYPE}, io_depth: {IO_DEPTH}")
    print(f"Iterations per config: {NUM_ITERATIONS}")
    print(f"TPC-H worker counts: {TPCH_WORKERS_LIST}")

    # Load workload vectors
    print("\nLoading workload vectors...")
    with open(WORKLOAD_VECTORS_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} vectors")

    all_results = {}

    for config_name, shared_buf_gb, cgroup_gb in MEMORY_CONFIGS:
        # Run experiments for this configuration
        config_results = run_memory_config(config_name, shared_buf_gb, cgroup_gb, workload)

        # Generate filename and save results
        filename = get_results_filename(INDEX_TYPE, IO_DEPTH, config_name, shared_buf_gb, cgroup_gb)
        filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, 'w') as f:
            for result in config_results:
                f.write(json.dumps(result) + "\n")

        print(f"\nSaved results to: {filepath}")
        all_results[config_name] = config_results

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for config_name, results in all_results.items():
        print(f"\n{config_name.upper()}:")
        print(f"{'Workers':<10} {'TopK QPS':<20} {'TPC-H TPS':<20}")
        print("-" * 50)
        for r in results:
            print(f"{r['tpch_workers']:<10} "
                  f"{r['topk_throughput_mean']:>8.2f} +/- {r['topk_throughput_std']:<8.2f} "
                  f"{r['tpch_throughput_mean']:>8.4f} +/- {r['tpch_throughput_std']:<8.4f}")

    print(f"\nBenchmark completed at {datetime.now()}")


if __name__ == "__main__":
    main()
