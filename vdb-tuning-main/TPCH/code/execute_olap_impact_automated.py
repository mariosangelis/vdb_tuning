#!/usr/bin/env python3
"""
Automated benchmark: Measure OLAP (TPC-H) interference on ANN (TopK) performance.

This is the reverse of execute_ann_impact_automated.py:
  - Fixed: 1 ANN worker (TopK vector search)
  - Variable: OLAP workers (0, 1, 2, 3, 4, 5, 6 concurrent TPC-H sessions)

For each memory configuration, we sweep OLAP concurrency and measure:
  - ANN throughput (qps) and latency (p50/p95/p99)
  - OLAP throughput (tps = total completed queries / wall-clock time)
  - Buffer statistics (shared_hit, shared_read) per TPC-H query

The output produces data for throughput frontier plots:
  X-axis = Vector Search Throughput (qps)
  Y-axis = Analytical Throughput (tps)
  Labels = number of OLAP workers

Memory Configurations (same as ANN impact script):
  - No Pressure: 40GB shared_buffers, 60GB cgroup
  - Relaxed:     16GB shared_buffers, 20GB cgroup
  - Medium:      12GB shared_buffers, 16GB cgroup
  - Tight:        8GB shared_buffers, 12GB cgroup

CPU Pinning (Wisconsin machine - AMD EPYC 7543, 2 sockets):
  - PostgreSQL:  cores 0-31  (socket 0, NUMA 0-7)
  - TopK + OLAP: cores 32-47 (socket 1, NUMA 8-11)
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
import statistics
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

# Paths — derived from script location
WORKLOAD_VECTORS_PATH = str(PROJECT_DIR / "gist_test_queries.txt")
RESULTS_DIR = str(TPCH_DIR / "results_olap_impact_to_ann/")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU pinning — Wisconsin machine: 2 sockets, 32 cores each
NUM_PG_CORES = 32
POSTGRES_CORES = "0-31"
# TopK and OLAP client threads on socket 1
TOPK_CORES = list(range(32, 48))
OLAP_CORES = list(range(32, 48))

# PostgreSQL paths — auto-detect
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"
PG_CONF_PATH = os.path.join(PG_DATA_DIR, "postgresql.conf")

# Cgroup
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"

# ============== EXPERIMENT PARAMETERS ==============
K = 100  # TopK parameter
NUM_TOPK_WORKERS = 1                        # Fixed: 1 ANN worker
OLAP_WORKERS_LIST = [0, 1, 2, 3, 4, 5, 6]  # Variable: TPC-H concurrency
BASELINE_DURATION_SEC = 180                  # Duration for baseline (0 OLAP workers) run

# Index configuration - set these before running
INDEX_TYPE = "IVFFLAT"   # "IVFFLAT" or "HNSW"
PARALLEL_INDEX = False   # True for Parallel IVFFlat, False for Default
IO_DEPTH = 0          # 0 or 1000

# Memory configurations: (name, shared_buffers_gb, cgroup_gb)
MEMORY_CONFIGS = [
    ("no_pressure", 40, 60),
    ("relaxed", 16, 20),
    ("medium", 12, 16),
    ("tight", 8, 12),
]

# Query template for GIST dataset - using L2 distance
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %s"

# Thread-safe lock
global_lock = threading.Lock()


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
    ("Q15", None),  # Special: handled separately
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


# ============== HELPER FUNCTIONS ==============

def parse_explain_buffers(explain_lines):
    """Parse EXPLAIN (ANALYZE, BUFFERS) output to extract buffer stats."""
    total_shared_hit = 0
    total_shared_read = 0
    total_shared_written = 0
    exec_time_ms = 0

    for row in explain_lines:
        line = row[0] if isinstance(row, tuple) else row
        line = line.strip()

        if 'Buffers:' in line and 'shared' in line:
            hit_match = re.search(r'shared hit=(\d+)', line)
            read_match = re.search(r'read=(\d+)', line)
            written_match = re.search(r'written=(\d+)', line)
            if hit_match:
                total_shared_hit = max(total_shared_hit, int(hit_match.group(1)))
            if read_match:
                total_shared_read = max(total_shared_read, int(read_match.group(1)))
            if written_match:
                total_shared_written = max(total_shared_written, int(written_match.group(1)))

        if 'Execution Time:' in line:
            exec_time_ms = float(line.split()[2])

    return {
        "shared_hit": total_shared_hit,
        "shared_read": total_shared_read,
        "shared_written": total_shared_written,
        "exec_time_ms": exec_time_ms,
    }


def get_results_filename(index_type, parallel, io_depth, shared_buf_gb, cgroup_gb):
    """Generate descriptive results filename."""
    parallel_str = "_PARALLEL" if parallel else ""
    return (f"olap_impact_on_ann_{index_type}{parallel_str}_io_{io_depth}_"
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
    """Restart PostgreSQL with CPU pinning to socket 0."""
    # Drop caches
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

    # Restart PostgreSQL pinned to socket 0 (NUMA 0-7)
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind=0-7 --membind=0-7 {PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
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


# ============== WORKER FUNCTIONS ==============

def run_topk_queries(thread_id, workload, stop_event, results_dict):
    """Worker thread that runs TopK queries continuously.

    Runs until stop_event is set, recording throughput and latency.
    """
    local_random = random.Random(thread_id + 42)

    # Pin thread to specific core on socket 1
    core_id = TOPK_CORES[thread_id % len(TOPK_CORES)]
    psutil.Process().cpu_affinity([core_id])

    local_count = 0
    local_exec_time = 0.0
    local_latencies = []

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        while not stop_event.is_set():
            index = local_random.randint(0, len(workload) - 1)
            vec = workload[index]
            vec_str = vec_to_str(vec)

            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {TOPK_QUERY}" % (vec_str, K))
            explain_result = cur.fetchall()

            for row in explain_result:
                line = row[0].strip()
                if 'Execution Time:' in line:
                    elapsed_ms = float(line.split()[2])
                    local_exec_time += elapsed_ms / 1000.0
                    local_count += 1
                    local_latencies.append(elapsed_ms)
                    break
    finally:
        cur.close()
        conn.close()

        with global_lock:
            results_dict['count'] += local_count
            results_dict['exec_time'] += local_exec_time
            results_dict['latencies'].extend(local_latencies)


def run_tpch_worker(worker_id, stop_event, worker_results):
    """OLAP worker thread: runs all 22 TPC-H queries sequentially, looping until stop_event.

    Each worker opens its own connection and continuously runs the full
    TPC-H query suite. We track per-query times and buffer stats.
    """
    # Pin to socket 1
    core_id = OLAP_CORES[worker_id % len(OLAP_CORES)]
    psutil.Process().cpu_affinity([core_id])

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Enable sequential scans for TPC-H (disabled globally for vector search index usage)
    cur.execute("SET enable_seqscan = on;")

    rounds_completed = 0
    per_query_times = []  # List of dicts per round

    try:
        while not stop_event.is_set():
            round_queries = []
            for qname, sql in TPCH_QUERIES:
                if stop_event.is_set():
                    break

                if qname == "Q15":
                    start = time.time()
                    # Q15 needs a temp view — each worker needs unique name to avoid conflicts
                    view_name = f"revenue0_w{worker_id}"
                    create_view = Q15_CREATE_VIEW.replace("revenue0", view_name)
                    query = Q15_QUERY.replace("revenue0", view_name)
                    drop_view = Q15_DROP_VIEW.replace("revenue0", view_name)

                    cur.execute(create_view)
                    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {query}")
                    explain_result = cur.fetchall()
                    cur.execute(drop_view)
                    elapsed = time.time() - start
                    buffers = parse_explain_buffers(explain_result)
                else:
                    start = time.time()
                    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}")
                    explain_result = cur.fetchall()
                    elapsed = time.time() - start
                    buffers = parse_explain_buffers(explain_result)

                round_queries.append({
                    "query": qname,
                    "time_sec": round(elapsed, 3),
                    "shared_hit": buffers["shared_hit"],
                    "shared_read": buffers["shared_read"],
                    "shared_written": buffers["shared_written"],
                })

            if len(round_queries) == 22:
                rounds_completed += 1
                per_query_times.append(round_queries)
                round_time = sum(q["time_sec"] for q in round_queries)
                print(f"    OLAP worker {worker_id}: completed round {rounds_completed} in {round_time:.1f}s")

    finally:
        cur.close()
        conn.close()

        with global_lock:
            worker_results.append({
                "worker_id": worker_id,
                "rounds_completed": rounds_completed,
                "per_query_times": per_query_times,
            })


def calculate_percentiles(latencies):
    """Calculate latency percentiles from a list of latencies (ms)."""
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
        "count": n,
    }


# ============== EXPERIMENT LOGIC ==============

def run_single_experiment(num_olap_workers, workload):
    """Run a single experiment with a given number of OLAP workers and 1 ANN worker.

    Protocol:
      1. Restart PostgreSQL (clean state, drop caches)
      2. Start 1 ANN (TopK) worker thread
      3. Start N OLAP (TPC-H) worker threads
      4. Wait for all OLAP workers to complete at least 1 full round of 22 queries
      5. After first round completes for ALL workers, stop everything
      6. Record ANN throughput/latency and OLAP throughput

    For num_olap_workers == 0 (baseline):
      - Run TopK only for BASELINE_DURATION_SEC seconds
    """
    topk_results = {'count': 0, 'exec_time': 0.0, 'latencies': []}
    olap_results = []
    stop_event = threading.Event()

    print(f"\n{'='*60}")
    print(f"Running experiment: {num_olap_workers} OLAP workers, {NUM_TOPK_WORKERS} ANN worker")
    print(f"{'='*60}")

    # Restart PostgreSQL with clean state
    restart_postgres()

    # Start TopK worker thread
    topk_threads = []
    for tid in range(NUM_TOPK_WORKERS):
        t = threading.Thread(target=run_topk_queries,
                             args=(tid, workload, stop_event, topk_results))
        t.start()
        topk_threads.append(t)

    wall_start = time.time()

    if num_olap_workers == 0:
        # Baseline: run TopK only for a fixed duration
        print(f"Running baseline (TopK only) for {BASELINE_DURATION_SEC} seconds...")
        time.sleep(BASELINE_DURATION_SEC)
        wall_time = BASELINE_DURATION_SEC
    else:
        # Start OLAP worker threads — each runs TPC-H queries in a loop
        olap_threads = []
        for wid in range(num_olap_workers):
            t = threading.Thread(target=run_tpch_worker,
                                 args=(wid, stop_event, olap_results))
            t.start()
            olap_threads.append(t)

        # Wait for all OLAP workers to complete at least 1 full round
        # We poll until all have finished
        for t in olap_threads:
            t.join()

        wall_time = time.time() - wall_start

    # Stop TopK threads
    stop_event.set()
    for t in topk_threads:
        t.join()

    # Calculate TopK metrics
    topk_query_count = topk_results['count']
    total_exec_time_topk = topk_results['exec_time']
    latency_stats = calculate_percentiles(topk_results['latencies'])

    if topk_query_count > 0 and wall_time > 0:
        topk_throughput = topk_query_count / wall_time
        topk_avg_time = total_exec_time_topk / topk_query_count
    else:
        topk_throughput = 0
        topk_avg_time = 0

    # Calculate OLAP metrics
    total_olap_queries = 0
    olap_round_times = []
    all_per_query = []

    for wr in olap_results:
        total_olap_queries += wr["rounds_completed"] * 22
        for round_data in wr["per_query_times"]:
            round_time = sum(q["time_sec"] for q in round_data)
            olap_round_times.append(round_time)
            all_per_query.append(round_data)

    if wall_time > 0 and num_olap_workers > 0:
        olap_throughput = total_olap_queries / wall_time
    else:
        olap_throughput = 0

    # Aggregate per-query buffer stats (from first round of first worker for comparability)
    first_round_queries = None
    if all_per_query:
        first_round_queries = all_per_query[0]

    return {
        "olap_workers": num_olap_workers,
        "topk_workers": NUM_TOPK_WORKERS,
        "wall_time_sec": round(wall_time, 2),
        # ANN metrics
        "topk_query_count": topk_query_count,
        "topk_throughput": round(topk_throughput, 4),
        "topk_avg_exec_time_sec": round(topk_avg_time, 6),
        "topk_latency_p50_ms": round(latency_stats["p50"], 2),
        "topk_latency_p95_ms": round(latency_stats["p95"], 2),
        "topk_latency_p99_ms": round(latency_stats["p99"], 2),
        "topk_latency_avg_ms": round(latency_stats["avg"], 2),
        # OLAP metrics
        "olap_total_queries": total_olap_queries,
        "olap_throughput_qps": round(olap_throughput, 4),
        "olap_throughput_tps": round(olap_throughput / 22, 6) if olap_throughput > 0 else 0,
        "olap_avg_round_time_sec": round(statistics.mean(olap_round_times), 2) if olap_round_times else 0,
        # Per-worker details
        "olap_worker_results": [
            {"worker_id": wr["worker_id"], "rounds_completed": wr["rounds_completed"]}
            for wr in olap_results
        ],
        # Per-query breakdown (first round, first worker)
        "tpch_query_times": first_round_queries,
    }


def run_memory_config(config_name, shared_buf_gb, cgroup_gb, workload):
    """Run all experiments for a single memory configuration."""
    print(f"\n{'#'*70}")
    print(f"# MEMORY CONFIG: {config_name}")
    print(f"# shared_buffers: {shared_buf_gb}GB, cgroup: {cgroup_gb}GB")
    print(f"# Index: {INDEX_TYPE}, Parallel: {PARALLEL_INDEX}, io_depth: {IO_DEPTH}")
    print(f"# Direction: OLAP impact on ANN (varying OLAP workers, fixed 1 ANN)")
    print(f"{'#'*70}")

    # Configure memory settings
    set_shared_buffers(shared_buf_gb)
    set_cgroup_memory(cgroup_gb)

    # Generate output filename
    filename = get_results_filename(INDEX_TYPE, PARALLEL_INDEX, IO_DEPTH, shared_buf_gb, cgroup_gb)
    filepath = os.path.join(RESULTS_DIR, filename)

    print(f"Results will be saved to: {filepath}")

    # Run experiments for each OLAP worker count
    for num_workers in OLAP_WORKERS_LIST:
        result = run_single_experiment(num_workers, workload)

        # Append result to file (one JSON per line)
        with open(filepath, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Print summary
        print(f"\n>>> Result for {num_workers} OLAP workers:")
        print(f"    Wall time:         {result['wall_time_sec']:.2f} s")
        print(f"    TopK throughput:   {result['topk_throughput']:.2f} qps")
        print(f"    TopK p50 latency:  {result['topk_latency_p50_ms']:.2f} ms")
        print(f"    TopK p95 latency:  {result['topk_latency_p95_ms']:.2f} ms")
        print(f"    OLAP throughput:   {result['olap_throughput_tps']:.4f} tps (transactions/sec)")
        print(f"    OLAP total queries:{result['olap_total_queries']}")
        if result['olap_avg_round_time_sec'] > 0:
            print(f"    OLAP avg round:    {result['olap_avg_round_time_sec']:.2f} s")

    print(f"\nCompleted {config_name} configuration. Results saved to: {filepath}")


# ============== MAIN ==============

def main():
    print(f"{'='*70}")
    print(f" OLAP Impact on ANN - Automated Benchmark")
    print(f" Started at {datetime.now()}")
    print(f"{'='*70}")

    print(f"\n--- Detected Configuration ---")
    print(f"  PG_CTL_PATH:    {PG_CTL_PATH}")
    print(f"  PG_DATA_DIR:    {PG_DATA_DIR}")
    print(f"  RESULTS_DIR:    {RESULTS_DIR}")
    print(f"  VECTORS_PATH:   {WORKLOAD_VECTORS_PATH}")
    print(f"  Total cores:    {os.cpu_count()}")
    print(f"  PG cores:       {POSTGRES_CORES} ({NUM_PG_CORES} cores, socket 0)")
    print(f"  TopK cores:     {TOPK_CORES[0]}-{TOPK_CORES[-1]} ({len(TOPK_CORES)} available, socket 1)")
    print(f"  OLAP cores:     {OLAP_CORES[0]}-{OLAP_CORES[-1]} ({len(OLAP_CORES)} available, socket 1)")
    print(f"  TPC-H:          Direct psycopg2 (no BenchBase)")
    print(f"------------------------------\n")
    print(f"Index type: {INDEX_TYPE}, Parallel: {PARALLEL_INDEX}, io_depth: {IO_DEPTH}")
    print(f"OLAP worker counts: {OLAP_WORKERS_LIST}")
    print(f"TopK workers:       {NUM_TOPK_WORKERS} (fixed)")
    print(f"Baseline duration:  {BASELINE_DURATION_SEC}s (for 0 OLAP workers)")

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
