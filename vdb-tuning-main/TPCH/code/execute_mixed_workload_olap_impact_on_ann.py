#!/usr/bin/env python3
"""
Measure OLAP (TPC-H) interference on ANN (TopK) performance.

This script runs a fixed ANN workload while varying TPC-H concurrency
to measure how OLAP queries impact vector search latency and throughput.

CPU Pinning (Wisconsin machine - 32 cores, 2 NUMA nodes):
  - PostgreSQL: cores 0-15 (NUMA 0)
  - BenchBase (TPC-H) + TopK threads: cores 24-31 (NUMA 1)
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
RESULTS_FILE = "OLAP_IMPACT_mixed_workload_tpch_and_topk_IVFFLAT_PARALLEL_io_0_NO_HYPERTHREADING_16_cores_for_postgres_20GB_RAM_16_shared_buffers_GIST.txt"

# CPU pinning
POSTGRES_CORES = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
BENCHBASE_CORES = "24,25,26,27,28,29,30,31"
TOPK_CORES = [24, 25, 26, 27, 28, 29, 30, 31]

# PostgreSQL paths
PG_CTL_PATH = "/usr/local/pgsql/bin/pg_ctl"
PG_DATA_DIR = "/mnt/nvme/pgdata"

# Experiment parameters
K = 100  # TopK parameter
NUM_TOPK_WORKERS = 1  # Fixed number of ANN workers
TPCH_WORKERS_LIST = [6,5,4,3,2,1]  # Variable TPC-H concurrency
BASELINE_DURATION_SEC = 120  # Duration for baseline (no TPC-H) run

# Query template for GIST dataset - using L2 distance
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %s"

# ============== GLOBAL STATE ==============
query_counter = threading.Lock()
topk_latencies = []  # Store individual latencies for percentile calculation
topk_query_count = 0
total_exec_time_topk = 0.0
stop_event = threading.Event()

# Load workload vectors
with open(WORKLOAD_VECTORS_PATH, 'r') as f:
    workload = eval(f.read())

results = []

# ============== HELPER FUNCTIONS ==============

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
    subprocess.run("for pid in $(pgrep -u postgres); do echo $pid > /sys/fs/cgroup/postgres_test/cgroup.procs; done", shell=True, check=False)

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

def run_topk_queries(thread_id):
    """Worker thread that runs TopK queries continuously."""
    global topk_query_count, total_exec_time_topk, topk_latencies

    local_random = random.Random(thread_id + 42)

    # Pin thread to specific core
    core_id = TOPK_CORES[thread_id % len(TOPK_CORES)]
    psutil.Process().cpu_affinity([core_id])

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        count=0
        while not stop_event.is_set():
            index = local_random.randint(0, len(workload) - 1)
            vec = workload[index]
            print("Executed topk Query ",count)
            count+=1


            # Run query with EXPLAIN ANALYZE to get execution time
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {TOPK_QUERY}" % (vec_to_str(vec), K))
            explain_result = cur.fetchall()

            # Parse execution time
            for row in explain_result:
                line = row[0].strip()
                if 'Execution Time:' in line:
                    elapsed_ms = float(line.split()[2])
                    elapsed_sec = elapsed_ms / 1000.0
                    with query_counter:
                        topk_latencies.append(elapsed_ms)
                        total_exec_time_topk += elapsed_sec
                        topk_query_count += 1
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

# ============== MAIN EXPERIMENT LOOP ==============

for num_tpch_workers in TPCH_WORKERS_LIST:
    # Reset global counters
    topk_latencies = []
    topk_query_count = 0
    total_exec_time_topk = 0.0
    tpch_execution_time = 0.0
    stop_event.clear()

    print(f"\n{'='*60}")
    print(f"Running experiment: {num_tpch_workers} TPC-H workers, {NUM_TOPK_WORKERS} TopK workers")
    print(f"{'='*60}")

    # Restart PostgreSQL with clean state
    restart_postgres()

    # Start TopK worker threads
    threads = []
    for thread_id in range(NUM_TOPK_WORKERS):
        t = threading.Thread(target=run_topk_queries, args=(thread_id,))
        t.start()
        threads.append(t)

    tpch_summary = None

    if num_tpch_workers == 0:
        # Baseline: run TopK only for fixed duration
        print(f"Running baseline (TopK only) for {BASELINE_DURATION_SEC} seconds...")
        time.sleep(BASELINE_DURATION_SEC)
        tpch_execution_time = BASELINE_DURATION_SEC
    else:
        # Create config with specified terminals
        config_path = create_tpch_config(num_tpch_workers)

        # Run multiple BenchBase processes in parallel (1 terminal each)
        print(f"Starting {num_tpch_workers} TPC-H processes in parallel...")
        
        # Create config with 1 terminal (serial=true)
        config_path = create_tpch_config(1)
        
        # Start all BenchBase processes
        benchbase_procs = []
        for i in range(num_tpch_workers):
            proc = subprocess.Popen([
                "taskset", "-c", BENCHBASE_CORES,
                "java", "-jar", "benchbase.jar", "-b", "tpch",
                "-c", config_path,
                "--create=false", "--load=false", "--execute=true"
            ], cwd=BENCHBASE_DIR, env={**os.environ, "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            benchbase_procs.append(proc)
            print(f"  Started BenchBase process {i+1} (PID: {proc.pid})")
        
        # Wait for all to complete and collect results
        results_list = []
        for i, proc in enumerate(benchbase_procs):
            stdout, stderr = proc.communicate()
            results_list.append((stdout, stderr, proc.returncode))
            print(f"  BenchBase process {i+1} finished")
        
        # Use first successful result for summary
        result = type("Result", (), {"stdout": results_list[0][0], "stderr": results_list[0][1]})()
        
        print("BENCHBASE FINISHED")
        
        # Parse TPC-H results
        output = result.stdout + result.stderr
        match = re.search(r"tpch_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.summary\.json", output)
        if match:
            timestamp = match.group(1)
            results_dir = Path(BENCHBASE_DIR) / "results"
            summary_file = results_dir / f"tpch_{timestamp}.summary.json"

            with open(summary_file, "r") as f:
                tpch_summary = json.load(f)

            tpch_execution_time = tpch_summary["Elapsed Time (nanoseconds)"] / 1_000_000_000

    # Stop TopK threads
    stop_event.set()
    for t in threads:
        t.join()

    # Calculate results
    latency_stats = calculate_percentiles(topk_latencies)
    
    if topk_query_count > 0:
        topk_throughput = topk_query_count / tpch_execution_time
        topk_avg_time = total_exec_time_topk / topk_query_count
    else:
        topk_throughput = 0
        topk_avg_time = 0

    print("BENCHMARK FINISHED")

    # Record result (same format as original script)
    result_entry = {
        "tpch_workers": num_tpch_workers,
        "topk_workers": NUM_TOPK_WORKERS,
        "summary": tpch_summary,
        "topk_query_count": topk_query_count,
        "topk_throughput": topk_throughput,
        "topk_avg_exec_time_sec": topk_avg_time,
        "tpch_execution_time": tpch_execution_time,
        "topk_latency_p50_ms": latency_stats["p50"],
        "topk_latency_p95_ms": latency_stats["p95"],
        "topk_latency_p99_ms": latency_stats["p99"],
    }
    
    results.append(result_entry)
    
    # Save to file immediately (same style as original)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result_entry) + "\n")

    print(result_entry)
    print(f"Run with {num_tpch_workers} TPC-H workers complete: TopK throughput {topk_throughput:.2f} qps, p50 {latency_stats['p50']:.2f}ms")

# Print summary table
print("\n" + "="*80)
print("SUMMARY: OLAP Impact on ANN Performance (GIST dataset)")
print("="*80)
print(f"{' TPC-H Workers':<15} {'TopK QPS':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12}")
print("-"*80)

for r in results:
    print(f"{r['tpch_workers']:<15} {r['topk_throughput']:>8.1f}    {r['topk_latency_p50_ms']:>8.2f}    {r['topk_latency_p95_ms']:>8.2f}    {r['topk_latency_p99_ms']:>8.2f}")

print("="*80)
print(f"Results saved to {RESULTS_FILE}")
