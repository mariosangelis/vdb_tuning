import psycopg2
import threading
import time
import subprocess
import json
import re
from pathlib import Path

query_counter = threading.Lock()
topk_query_count = 0
total_exec_time_topk = 0.0
topk_shared_hits=0
topk_shared_reads=0

import random
import os
import psutil
def vec_to_str(vec):
    """Convert Python list to PostgreSQL vector string format."""
    return "[" + ",".join(str(x) for x in vec) + "]"


METRICS_PATH = "/tmp/ivfflat_metrics.json"
# Load workload vectors
with open('../gist_test_queries.txt', 'r') as f:
    workload = eval(f.read())

K = 100
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %s"
results = []

topk_workers_list=[0,1,2,3,4,5,6,7,8,9,10]
#topk_workers_list=[10,10,10,10]


for num_workers in topk_workers_list:
    topk_query_count = 0
    total_exec_time_topk = 0.0
    tpch_execution_time= 0.0
    topk_shared_hits=0
    topk_shared_reads=0


    timing_json = {
        "get_scan_lists_time": 0.0,
        "get_scan_items_time": 0.0,
        "get_next_tuple_time":0.0,
        "read_buffer_extended_overhead": 0.0,
        "calculate_distances_from_page_overhead": 0.0,
        "process_tuples_overhead": 0.0,
        "sort_overhead": 0.0,
        "total_time": 0.0,
        "calculations": 0.0
    }

    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

    cpu_list_postgres= "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    subprocess.run(
        f"su - postgres -c 'taskset -c {cpu_list_postgres} /usr/local/pgsql/bin/pg_ctl -D /mnt/nvme/pgdata -l /tmp/logfile restart'",
        shell=True,
        check=True
    )


    # Move all postgres processes into cgroup
    subprocess.run("for pid in $(pgrep -u postgres); do echo $pid > /sys/fs/cgroup/postgres_test/cgroup.procs; done", shell=True, check=True)

    subprocess.run(
        f"rm -f {METRICS_PATH}",
        shell=True,
        check=True
    )

    stop_event = threading.Event()  # stop signal shared across threads

    def run_topk_queries(thread_id):

        local_random = random.Random(thread_id + 42)

        # Pin thread to a NUMA 1 CPU (cycling through its cores)
        testing_cores = [24,25,26,27,28,29,30,31]
        core_id = testing_cores[thread_id % len(testing_cores)]
        psutil.Process().cpu_affinity([core_id])

        global topk_query_count, total_exec_time_topk,topk_shared_hits,topk_shared_reads
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="secret",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        #interval = 1.0 / per_thread_qps
        next_scheduled = time.time()
        i=0
        try:
            while not stop_event.is_set():
                index = local_random.randint(0, len(workload) - 1)
                vec = workload[index]

                found_main_buffers = False
                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {TOPK_QUERY}" % (vec_to_str(vec), K))
                explain_result = cur.fetchall()

                elapsed = 0
                for row in explain_result:
                    line = row[0].strip()
                    if 'Execution Time:' in line:
                        elapsed = float(line.split()[2]) / 1000.0
                        with query_counter:
                            total_exec_time_topk += elapsed

                with query_counter:
                    topk_query_count += 1
        finally:
            cur.close()
            conn.close()

    for attempt in range(1000):
        try:
            conn = psycopg2.connect(
                dbname="mydb",
                user="postgres",
                password="secret",
                host="localhost",
                port="5432"
            )
            conn.close()
            print("✅ PostgreSQL is ready.")
            break
        except psycopg2.OperationalError:
            print(f"⏳ Waiting for PostgreSQL to accept connections... attempt {attempt + 1}")
            time.sleep(2)
    else:
        raise Exception("❌ PostgreSQL did not become ready in time.")

    
    # Start token refill thread for current QPS
    stop_event.clear()

    threads = []
    for thread_id in range(num_workers):
        t = threading.Thread(target=run_topk_queries, args=(thread_id,))
        t.start()
        threads.append(t)

    result = subprocess.run([
        "taskset", "-c", "24,25,26,27,28,29,30,31",
        "java", "-jar", "benchbase.jar", "-b", "tpch",
        "-c", "config/postgres/sample_tpch_config.xml",
        "--create=false", "--load=false", "--execute=true"
    ], cwd="/users/mangelis/vdb-tuning-main/benchbase/target/benchbase-postgres",
        capture_output=True, text=True, check=True)

    print("BENSHBASE FINSIHED")
    # Stop TopK threads
    stop_event.set()
    for t in threads:
        t.join()

    # Step 2: Extract the result directory and timestamp from the output
    output = result.stdout + result.stderr
    match = re.search(r"tpch_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.summary\.json", output)
    if not match:
        raise RuntimeError("Summary file name not found in output")

    timestamp = match.group(1)
    results_dir = Path("/users/mangelis/vdb-tuning-main/benchbase/target/benchbase-postgres/results")
    summary_file = results_dir / f"tpch_{timestamp}.summary.json"

    # Step 3: Load and parse the summary file
    with open(summary_file, "r") as f:
        summary = json.load(f)

    # Step 4: Extract and convert execution time
    elapsed_ns = summary["Elapsed Time (nanoseconds)"]
    tpch_execution_time = elapsed_ns / 1_000_000_000

    if topk_query_count > 0:
        topk_throughput = topk_query_count / tpch_execution_time
        topk_avg_time = total_exec_time_topk / topk_query_count
    else:
        topk_throughput = 0
        topk_avg_time = 0

    print("BENCHMARK FINISHED")

    # Record result
    results.append({
        "summary": summary,
        "topk_query_count": topk_query_count,
        "topk_throughput": topk_throughput,
        "topk_avg_exec_time_sec": topk_avg_time,
        "tpch_execution_time": tpch_execution_time,
	    "topk_shared_hits": topk_shared_hits,
        "topk_shared_reads": topk_shared_reads,
    })
    with open("mixed_workload_tpch_and_topk_IVFFLAT_PARALLEL_io_1000_NO_HYPERTHREADING_16_cores_for_postgres_12_GB_shared_buffers_16GB_RAM_GIST.txt", "a") as f:
      f.write(json.dumps(results[-1]) + "\n")

    print(results[-1])
    print(f"✅ Run with {topk_throughput} TopK QPS complete: TPCH {tpch_execution_time:.2f} seconds")
 