#!/usr/bin/env python3
"""
Automated benchmark: Measure ANN (TopK) interference on OLAP (TPC-H) performance.

This script runs experiments across multiple memory configurations,
varying the number of ANN workers while keeping OLAP worker at 1.

TPC-H queries are executed directly via psycopg2 (no BenchBase overhead).

Memory Configurations:
  - No Pressure: 40GB shared_buffers, 60GB cgroup
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
import shutil
from pathlib import Path
import psutil
from datetime import datetime

def vec_to_str(vec):
    """Convert Python list to PostgreSQL vector string format."""
    return '[' + ','.join(str(x) for x in vec) + ']'

# ============== AUTO-DETECT PATHS ==============
SCRIPT_DIR = Path(__file__).resolve().parent           # .../TPCH/code/
TPCH_DIR = SCRIPT_DIR.parent                           # .../TPCH/
PROJECT_DIR = TPCH_DIR.parent                          # .../vdb-tuning-main/

# ============== CONFIGURATION ==============
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

# Paths — derived from script location and home directory
WORKLOAD_VECTORS_PATH = str(PROJECT_DIR / "gist_test_queries.txt")
RESULTS_DIR = str(TPCH_DIR / "results")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU pinning — adjust these for your machine
NUM_PG_CORES = 30
POSTGRES_CORES = "0-29"
BENCHBASE_CORES = "71-80"
TOPK_CORES = list(range(60, 70))

# PostgreSQL paths — auto-detect
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/home/postgres/pgdata"
PG_CONF_PATH = os.path.join(PG_DATA_DIR, "postgresql.conf")

# Cgroup
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"

# ============== TPC-H QUERIES (same SQL as BenchBase, fixed parameters) ==============
TPCH_QUERIES = [
    ("Q1", "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_base_price, SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, AVG(l_discount) AS avg_disc, COUNT(*) AS count_order FROM lineitem WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90 DAY' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus"),
    ("Q2", "SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment FROM part, supplier, partsupp, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15 AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE' AND ps_supplycost = (SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE') ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100"),
    ("Q3", "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15' GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10"),
    ("Q4", "SELECT o_orderpriority, COUNT(*) AS order_count FROM orders WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority"),
    ("Q5", "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM customer, orders, lineitem, supplier, nation, region WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1994-01-01' + INTERVAL '1' YEAR GROUP BY n_name ORDER BY revenue DESC"),
    ("Q6", "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01 AND l_quantity < 24"),
    ("Q7", "SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue FROM (SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, EXTRACT(YEAR FROM l_shipdate) AS l_year, l_extendedprice * (1 - l_discount) AS volume FROM supplier, lineitem, orders, customer, nation n1, nation n2 WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31') AS shipping GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year"),
    ("Q8", "SELECT o_year, SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share FROM (SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey AND r_name = 'AMERICA' AND s_nationkey = n2.n_nationkey AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31' AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations GROUP BY o_year ORDER BY o_year"),
    ("Q9", "SELECT nation, o_year, SUM(amount) AS sum_profit FROM (SELECT n_name AS nation, EXTRACT(YEAR FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC"),
    ("Q10", "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name, c_address, c_phone, c_comment FROM customer, orders, lineitem, nation WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1993-10-01' + INTERVAL '3' MONTH AND l_returnflag = 'R' AND c_nationkey = n_nationkey GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment ORDER BY revenue DESC LIMIT 20"),
    ("Q11", "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY' GROUP BY ps_partkey HAVING SUM(ps_supplycost * ps_availqty) > (SELECT SUM(ps_supplycost * ps_availqty) * 0.00001 FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY') ORDER BY value DESC"),
    ("Q12", "SELECT l_shipmode, SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1994-01-01' + INTERVAL '1' YEAR GROUP BY l_shipmode ORDER BY l_shipmode"),
    ("Q13", "SELECT c_count, COUNT(*) AS custdist FROM (SELECT c_custkey, COUNT(o_orderkey) AS c_count FROM customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' GROUP BY c_custkey) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC"),
    ("Q14", "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-09-01' + INTERVAL '1' MONTH"),
    ("Q15", None),  # Special: handled in run_tpch_direct()
    ("Q16", "SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt FROM partsupp, part WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%%' AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9) AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%%Customer%%Complaints%%') GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size"),
    ("Q17", "SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX' AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)"),
    ("Q18", "SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity) FROM customer, orders, lineitem WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300) AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100"),
    ("Q19", "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 1 AND l_quantity <= 1 + 10 AND p_size BETWEEN 1 AND 5 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND l_quantity >= 10 AND l_quantity <= 10 + 10 AND p_size BETWEEN 1 AND 10 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND l_quantity >= 20 AND l_quantity <= 20 + 10 AND p_size BETWEEN 1 AND 15 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')"),
    ("Q20", "SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%%') AND ps_availqty > (SELECT 0.5 * SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR)) AND s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name"),
    ("Q21", "SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA' GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100"),
    ("Q22", "SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal FROM (SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17') AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')) AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale GROUP BY cntrycode ORDER BY cntrycode"),
]

Q15_CREATE_VIEW = "CREATE VIEW revenue0 (supplier_no, total_revenue) AS SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount)) FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH GROUP BY l_suppkey"
Q15_QUERY = "SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier, revenue0 WHERE s_suppkey = supplier_no AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0) ORDER BY s_suppkey"
Q15_DROP_VIEW = "DROP VIEW revenue0"


def run_tpch_direct():
    """Run all 22 TPC-H queries directly via psycopg2. Returns total wall-clock time and per-query details."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    query_times = []
    total_start = time.time()

    # Enable sequential scans for TPC-H (disabled globally for vector search index usage)
    cur.execute("SET enable_seqscan = on;")

    for qname, sql in TPCH_QUERIES:
        if qname == "Q15":
            start = time.time()
            cur.execute(Q15_CREATE_VIEW)
            cur.execute(Q15_QUERY)
            cur.fetchall()
            cur.execute(Q15_DROP_VIEW)
            elapsed = time.time() - start
        else:
            start = time.time()
            cur.execute(sql)
            cur.fetchall()
            elapsed = time.time() - start

        query_times.append({"query": qname, "time_sec": round(elapsed, 3)})
        print(f"    {qname}: {elapsed:.3f}s")

    total_time = time.time() - total_start

    cur.close()
    conn.close()

    return total_time, query_times


# Experiment parameters
K = 100  # TopK parameter
TOPK_WORKERS_LIST = [ 5,10]  # Variable ANN workers
NUM_TPCH_WORKERS = 1  # Fixed number of OLAP workers

# Index configuration - set these before running
INDEX_TYPE = "IVFFLAT"  # "IVFFLAT" or "HNSW"
PARALLEL_INDEX = False  # True for Parallel IVFFlat, False for Default
IO_DEPTH = 1000  # 0 or 1000

# Memory configurations: (name, shared_buffers_gb, cgroup_gb)
# Uncomment the configuration(s) you want to run
MEMORY_CONFIGS = [
    #("no_pressure", 40, 60),
    #("relaxed", 16, 20),
    #("medium", 12, 20),
    ("tight", 12 , 20),
]

# Query template for GIST dataset - using L2 distance
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %s"

# Thread-safe counters
query_counter = threading.Lock()


def get_results_filename(index_type, parallel, io_depth, shared_buf_gb, cgroup_gb):
    """Generate descriptive results filename."""
    parallel_str = "_PARALLEL" if parallel else ""
    return (f"mixed_workload_tpch_and_topk_{index_type}{parallel_str}_io_{io_depth}_"
            f"{NUM_PG_CORES}_cores_for_postgres_{shared_buf_gb}GB_shared_buffers_"
            f"{cgroup_gb}GB_RAM_GIST.txt")


def set_shared_buffers(size_gb):
    """Modify shared_buffers in postgresql.conf."""
    print(f"Setting shared_buffers to {size_gb}GB...")
    size_kb = size_gb * 1000000

    with open(PG_CONF_PATH, 'r') as f:
        config = f.read()

    new_config = re.sub(
        r'(shared_buffers\s*=\s*)[^\n]+',
        f'\\g<1>{size_kb}kB',
        config
    )

    with open(PG_CONF_PATH, 'w') as f:
        f.write(new_config)

    with open(PG_CONF_PATH, 'r') as f:
        verify = f.read()
    match = re.search(r'shared_buffers\s*=\s*[^\n]+', verify)
    print(f"  shared_buffers now: {match.group() if match else 'NOT FOUND'}")


def set_cgroup_memory(size_gb):
    """Set cgroup memory limit."""
    print(f"Setting cgroup memory to {size_gb}GB...")
    size_bytes = size_gb * 1024 * 1024 * 1024

    with open(CGROUP_MEM_PATH, 'w') as f:
        f.write(str(size_bytes))

    with open(CGROUP_MEM_PATH, 'r') as f:
        actual = int(f.read().strip())

    print(f"  cgroup memory set to {actual / (1024**3):.1f}GB")


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


def run_topk_queries(thread_id, workload, stop_event, results_dict):
    """Worker thread that runs TopK queries continuously."""
    local_random = random.Random(thread_id + 42)

    # Pin thread to specific core
    core_id = TOPK_CORES[thread_id % len(TOPK_CORES)]
    psutil.Process().cpu_affinity([core_id])

    local_count = 0
    local_exec_time = 0.0

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
                    local_exec_time += elapsed_ms / 1000.0  # Convert to seconds
                    local_count += 1
                    break
    finally:
        cur.close()
        conn.close()

        # Store results
        with query_counter:
            results_dict['count'] += local_count
            results_dict['exec_time'] += local_exec_time


def run_single_experiment(num_topk_workers, workload):
    """Run a single experiment with given number of ANN workers."""
    results_dict = {'count': 0, 'exec_time': 0.0}
    stop_event = threading.Event()

    print(f"\n{'='*60}")
    print(f"Running experiment: {num_topk_workers} ANN workers, {NUM_TPCH_WORKERS} OLAP worker")
    print(f"{'='*60}")

    # Restart PostgreSQL with clean state
    restart_postgres()

    # Start TopK worker threads (only if num_topk_workers > 0)
    threads = []
    if num_topk_workers > 0:
        for thread_id in range(num_topk_workers):
            t = threading.Thread(target=run_topk_queries,
                               args=(thread_id, workload, stop_event, results_dict))
            t.start()
            threads.append(t)

    # Run TPC-H queries directly (no BenchBase)
    print(f"Starting TPC-H (direct psycopg2, 22 queries)...")
    tpch_execution_time, query_times = run_tpch_direct()
    print(f"TPC-H finished in {tpch_execution_time:.2f}s")

    # Stop TopK threads
    stop_event.set()
    for t in threads:
        t.join()

    # Calculate TopK metrics
    topk_query_count = results_dict['count']
    total_exec_time_topk = results_dict['exec_time']

    if topk_query_count > 0 and tpch_execution_time > 0:
        topk_throughput = topk_query_count / tpch_execution_time
        topk_avg_time = total_exec_time_topk / topk_query_count
    else:
        topk_throughput = 0
        topk_avg_time = 0

    return {
        "tpch_query_times": query_times,
        "topk_query_count": topk_query_count,
        "topk_throughput": topk_throughput,
        "topk_avg_exec_time_sec": topk_avg_time,
        "tpch_execution_time": tpch_execution_time,
        "topk_shared_hits": 0,
        "topk_shared_reads": 0,
    }


def run_memory_config(config_name, shared_buf_gb, cgroup_gb, workload):
    """Run all experiments for a single memory configuration."""
    print(f"\n{'#'*70}")
    print(f"# MEMORY CONFIG: {config_name}")
    print(f"# shared_buffers: {shared_buf_gb}GB, cgroup: {cgroup_gb}GB")
    print(f"# Index: {INDEX_TYPE}, Parallel: {PARALLEL_INDEX}, io_depth: {IO_DEPTH}")
    print(f"{'#'*70}")

    # Configure memory settings
    set_shared_buffers(shared_buf_gb)
    set_cgroup_memory(cgroup_gb)

    # Generate output filename
    filename = get_results_filename(INDEX_TYPE, PARALLEL_INDEX, IO_DEPTH, shared_buf_gb, cgroup_gb)
    filepath = os.path.join(RESULTS_DIR, filename)

    print(f"Results will be saved to: {filepath}")

    # Run experiments for each worker count
    for num_workers in TOPK_WORKERS_LIST:
        result = run_single_experiment(num_workers, workload)

        # Append result to file
        with open(filepath, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Print summary
        print(f"\n>>> Result for {num_workers} ANN workers:")
        print(f"    TopK throughput: {result['topk_throughput']:.2f} qps")
        print(f"    TopK queries: {result['topk_query_count']}")
        print(f"    TPC-H execution time: {result['tpch_execution_time']:.2f} s")
        if result['tpch_execution_time'] > 0:
            print(f"    TPC-H throughput: {22.0 / result['tpch_execution_time']:.4f} qps")

    print(f"\nCompleted {config_name} configuration. Results saved to: {filepath}")


def main():
    print(f"Starting automated ANN impact benchmark at {datetime.now()}")
    print(f"\n--- Detected Configuration ---")
    print(f"  PG_CTL_PATH:    {PG_CTL_PATH}")
    print(f"  PG_DATA_DIR:    {PG_DATA_DIR}")
    print(f"  RESULTS_DIR:    {RESULTS_DIR}")
    print(f"  VECTORS_PATH:   {WORKLOAD_VECTORS_PATH}")
    print(f"  Total cores:    {os.cpu_count()}")
    print(f"  PG cores:       {POSTGRES_CORES} ({NUM_PG_CORES} cores)")
    print(f"  TopK cores:     {TOPK_CORES[0]}-{TOPK_CORES[-1]} ({len(TOPK_CORES)} available)")
    print(f"  TPC-H:          Direct psycopg2 (no BenchBase)")
    print(f"------------------------------\n")
    print(f"Index type: {INDEX_TYPE}, Parallel: {PARALLEL_INDEX}, io_depth: {IO_DEPTH}")
    print(f"ANN worker counts: {TOPK_WORKERS_LIST}")
    print(f"TPC-H workers: {NUM_TPCH_WORKERS} (fixed)")

    # Load workload vectors
    print("\nLoading workload vectors...")
    with open(WORKLOAD_VECTORS_PATH, 'r') as f:
        workload = eval(f.read())
    print(f"Loaded {len(workload)} vectors")

    # Run experiments for each memory configuration
    for config_name, shared_buf_gb, cgroup_gb in MEMORY_CONFIGS:
        run_memory_config(config_name, shared_buf_gb, cgroup_gb, workload)

    print(f"\n{'='*70}")
    print(f"All experiments completed at {datetime.now()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
