#!/usr/bin/env python3
"""
Step 1 of Throughput Frontier: Find Saturation Points.

Following the methodology of "How Good is My HTAP System?" (SIGMOD'22, Section 3.3),
this script finds:
  - X^ANN: Maximum ANN throughput when running ANN alone (no OLAP)
  - X^OLAP: Maximum OLAP throughput when running OLAP alone (no ANN)

These saturation points define the bounds of the throughput frontier.

The script runs each workload in isolation with increasing number of workers
until throughput stops improving (saturates).

Usage:
    sudo python3 find_saturation_points.py [--ann-only] [--olap-only]

Output: saturation_points.json
"""

import psycopg2
import threading
import time
import subprocess
import json
import re
import random
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ============== AUTO-DETECT PATHS ==============
# ============== DETERMINISTIC QUERY ASSIGNMENT ==============
import itertools

class DeterministicQueryAssigner:
    """Pre-generates deterministic sequence of query indices."""
    def __init__(self, seed, workload_size, sequence_length=1000000):
        import random as _rand
        rng = _rand.Random(seed)
        self.query_sequence = [rng.randint(0, workload_size - 1) for _ in range(sequence_length)]
        self.counter = itertools.count()
        self.workload_size = workload_size
        self.seed = seed

    def get_next_query_index(self):
        position = next(self.counter)
        idx = self.query_sequence[position % len(self.query_sequence)]
        # if position < 5:
        #     print(f"  [DeterministicAssigner seed={self.seed}] position={position} -> query_idx={idx}")
        return idx

    def reset(self):
        self.counter = itertools.count()

ann_query_assigner = None
olap_query_assigner = None


SCRIPT_DIR = Path(__file__).resolve().parent                    # frontier_analysis/
TPCH_DIR = SCRIPT_DIR.parent.parent                             # TPCH/
PROJECT_DIR = TPCH_DIR.parent                                    # vdb-tuning-main/

# Central machine-specific config (paths) from config.yaml at the repo root.
sys.path.insert(0, str(PROJECT_DIR.parent))  # config_loader lives at ~/vdb_tuning (repo root)
from config_loader import get_path  # noqa: E402

# ============== LOAD EXPERIMENT CONFIG ==============
CONFIG_PATH = SCRIPT_DIR.parent / "experiment_config.json"      # code/experiment_config.json
with open(CONFIG_PATH, 'r') as _cf:
    CONFIG = json.load(_cf)

IDX_CFG = CONFIG["index_config"]
HW_CFG = CONFIG["hardware"]

# ============== CONFIGURATION ==============
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "host": "localhost",
    "port": "5432"
}

# Paths (from config.yaml)
PG_INSTALL = get_path("pg_install")
WORKLOAD_VECTORS_PATH = str(PROJECT_DIR / IDX_CFG["workload_vectors_file"])
TUNE_RESULTS_DIR = PROJECT_DIR / "compute_and_compare_recall" / "results_from_tune_scripts"

RESULTS_DIR = str(TPCH_DIR / "results_throughput_frontier")
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU pinning (from experiment_config.json)
POSTGRES_CORES = HW_CFG["postgres_cores"]
PG_NUMA_NODES = HW_CFG["postgres_numa_nodes"]
TOPK_CORES = HW_CFG["topk_producers_cores"]
OLAP_CORES = HW_CFG["olap_producers_cores"]

# PostgreSQL paths (from config.yaml)
PG_CTL_PATH = shutil.which("pg_ctl") or f"{PG_INSTALL}/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or get_path("pg_data")
PG_CONF_PATH = os.path.join(PG_DATA_DIR, "postgresql.conf")
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"

# Memory constraint config (set via --memory flag)
MEMORY_CONFIG = None

# Tune configuration (loaded from tune results)
TUNE_CFG = None

# Opt-in run variants, set only by --ef-search / --probes / --recall / --variant (see main).
# Without these flags nothing changes: same results dir, same JSON keys.
KNOB = {}          # search-knob override, e.g. {"ef_search": 250}
RECALL_NOTE = None # measured recall at that knob, recorded in the JSONs
VARIANT = None     # e.g. "noopt" = PostgreSQL built without the redundant-computation optimization
RUN_SUFFIX = ""    # appended to the results dir, e.g. "_ef250", "_noopt"

# Warm mode: prewarm DB once, no restart between measurements
WARM_MODE = False  # Will be {"shared_buffers_gb": X, "cgroup_gb": Y} if --memory used

# Multi-index support: override index type and params via CLI
INDEX_TYPE_OVERRIDE = None  # Set by --index flag (ivfflat, hnsw, parallel_ivfflat)
INDEX_PARAMS = {}  # Manual params: {"m": 16, "ef_construction": 128} or {"lists": 8192}

# Snapshot mode: skip ensure_index (index already loaded from PGDATA snapshot)
SNAPSHOT_MODE = False

# ============== EXPERIMENT PARAMETERS ==============
K = IDX_CFG["k"]

# Worker counts to test for saturation (stop when < 5% improvement)
ANN_WORKERS_TO_TEST = [20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
#ANN_WORKERS_TO_TEST = [30, 50, 55, 60, 65, 70, 75, 80, 85, 90]
ANN_WORKERS_PARALLEL_IVFFLAT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
OLAP_WORKERS_TO_TEST = [5, 8, 10, 12, 14, 16, 18, 20, 22, 24]
SATURATION_THRESHOLD = 0.01  # Stop if throughput improves less than 0.05%

# Timing
MEASUREMENT_SEC = int(os.environ.get("MEASUREMENT_SEC", "300"))  # per-point window (s); override via env, default 5 min

# TopK query template
TOPK_QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> '%s'::vector LIMIT %d"

# ============== TPC-H QUERIES ==============
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
    ("Q15_create", "CREATE OR REPLACE VIEW revenue0 (supplier_no, total_revenue) AS SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount)) FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH GROUP BY l_suppkey"),
    ("Q15", "SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier, revenue0 WHERE s_suppkey = supplier_no AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0) ORDER BY s_suppkey"),
    ("Q16", "SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt FROM partsupp, part WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%%' AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9) AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%%Customer%%Complaints%%') GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size"),
    ("Q17", "SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX' AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)"),
    ("Q18", "SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity) FROM customer, orders, lineitem WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300) AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100"),
    ("Q19", "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 1 AND l_quantity <= 1 + 10 AND p_size BETWEEN 1 AND 5 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND l_quantity >= 10 AND l_quantity <= 10 + 10 AND p_size BETWEEN 1 AND 10 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND l_quantity >= 20 AND l_quantity <= 20 + 10 AND p_size BETWEEN 1 AND 15 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')"),
    ("Q20", "SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%%') AND ps_availqty > (SELECT 0.5 * SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR)) AND s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name"),
    ("Q21", "SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA' GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100"),
    ("Q22", "SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal FROM (SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17') AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')) AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale GROUP BY cntrycode ORDER BY cntrycode"),
]



# ============== MEMORY CONFIGURATION ==============

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

    # Verify
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

    # Verify
    with open(CGROUP_MEM_PATH, 'r') as f:
        actual = int(f.read().strip())
    print(f"  cgroup memory set to {actual / (1024**3):.1f}GB")


def move_postgres_to_cgroup():
    """Move all postgres processes into the cgroup."""
    subprocess.run(
        "for pid in $(pgrep -u postgres); do echo $pid > /sys/fs/cgroup/postgres_test/cgroup.procs; done",
        shell=True, check=False
    )


# ============== UTILITY FUNCTIONS ==============



# ============== PREWARM ==============

def prewarm_database():
    """Prewarm ALL tables and indexes into shared buffers using pg_prewarm."""
    print("  Prewarming ALL relations into shared buffers...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Get ALL user tables and indexes sorted by size (largest first)
    cur.execute("""
        SELECT c.relname, c.relkind, pg_relation_size(c.oid) as size_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        AND c.relkind IN ('r', 'i')
        ORDER BY pg_relation_size(c.oid) DESC
    """)
    relations = cur.fetchall()

    total_pages = 0
    total_relations = 0
    for relname, relkind, size_bytes in relations:
        size_mb = size_bytes / (1024 * 1024)
        try:
            cur.execute(f"SELECT pg_prewarm('{relname}', 'buffer')")
            pages = cur.fetchone()[0]
            total_pages += pages
            total_relations += 1
            kind = "TBL" if relkind == "r" else "IDX"
            print(f"    [{kind}] {relname}: {pages} pages ({size_mb:.0f} MB)")
        except Exception as e:
            print(f"    {relname}: skipped ({e})")

    cur.close()
    conn.close()
    print(f"  Prewarm complete: {total_relations} relations, {total_pages} pages (~{total_pages * 8 / 1024:.0f} MB)")


# ============== INDEX MANAGEMENT ==============

def load_tune_config():
    """Load tuning parameters from tune results JSON.
    Tries both generic and dataset-specific filenames (e.g., tune_ivfflat_results.json
    and tune_ivfflat_results_gist.json).
    """
    index_type = (INDEX_TYPE_OVERRIDE or "ivfflat").upper()
    if index_type == "PARALLEL_IVFFLAT":
        index_type = "IVFFLAT"
    if index_type == "IVFFLAT":
        base_name = "tune_ivfflat_results"
    elif index_type == "HNSW":
        base_name = "tune_hnsw_results"
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    # Try multiple filename patterns
    candidates = [
        TUNE_RESULTS_DIR / f"{base_name}.json",
        TUNE_RESULTS_DIR / f"{base_name}_gist.json",
        TUNE_RESULTS_DIR / f"{base_name}_deep.json",
    ]

    tune_file = None
    for c in candidates:
        if c.exists():
            tune_file = c
            break

    if tune_file is None:
        raise FileNotFoundError(
            f"Tune results not found. Tried: {[str(c) for c in candidates]}\n"
            f"Run the tuning script first to generate index parameters.")

    with open(tune_file, "r") as f:
        tune_data = json.load(f)

    best = tune_data["best_configuration"]
    print(f"  Loaded tune config from: {tune_file}")
    print(f"  Best config: {best}")
    if KNOB:
        best = {**best, **KNOB}
        print(f"  Search knob override (opt-in): {KNOB}")
    return best


def pgvector_num_workers():
    """NUM_WORKERS #define of the parallel IVFFlat fork (None for other indexes)."""
    if INDEX_TYPE_OVERRIDE != "parallel_ivfflat":
        return None
    src = Path(get_path("pgvector_parallel")) / "src" / "ivfscan.c"
    return int(re.search(r"^#define\s+NUM_WORKERS\s+(\d+)", src.read_text(), re.M).group(1))


def compile_pgvector(variant="default"):
    """Compile and install pgvector extension (default or parallel), then restart postgres."""
    if variant == "parallel":
        src_dir = get_path("pgvector_parallel")
    else:
        src_dir = get_path("pgvector_default")

    print(f"  Compiling pgvector ({variant}) from {src_dir}...")
    env_prefix = f"export PATH={PG_INSTALL}/bin:$PATH"
    result = subprocess.run(
        f"cd {src_dir} && {env_prefix} && make clean > /dev/null 2>&1; make -j$(nproc) > /dev/null 2>&1 && make install > /dev/null 2>&1",
        shell=True,
    )
    if result.returncode != 0:
        subprocess.run(
            f"cd {src_dir} && {env_prefix} && make clean && make -j$(nproc) && make install",
            shell=True, check=True,
        )
    print(f"  pgvector ({variant}) compiled and installed.")


def load_snapshot(snapshot_path):
    """Stop postgres, restore PGDATA from snapshot, restart postgres."""
    print(f"  Loading snapshot: {Path(snapshot_path).name}...")

    # Stop postgres
    subprocess.run(
        f"su - postgres -c '{PG_CTL_PATH} -D {PG_DATA_DIR} stop -m fast'",
        shell=True, capture_output=True,
    )
    time.sleep(2)

    # Clear and restore
    subprocess.run(f"rm -rf {PG_DATA_DIR}/*", shell=True, check=True)

    pigz = shutil.which("pigz")
    if pigz:
        cmd = f"pigz -dc {snapshot_path} | tar -xf - -C {PG_DATA_DIR}"
    else:
        cmd = f"tar -xzf {snapshot_path} -C {PG_DATA_DIR}"

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - start

    subprocess.run(f"chown -R postgres:postgres {PG_DATA_DIR}", shell=True, check=True)
    subprocess.run(f"chmod 700 {PG_DATA_DIR}", shell=True, check=True)
    print(f"  Snapshot loaded in {elapsed:.1f}s")

    # Restart postgres with NUMA binding
    subprocess.run(
        f"su - postgres -c 'numactl --cpunodebind={PG_NUMA_NODES} --membind={PG_NUMA_NODES} "
        f"{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile start'",
        shell=True, capture_output=True,
    )
    time.sleep(3)
    wait_for_postgres()


def ensure_index():
    """Always compile the correct pgvector extension and load the correct snapshot."""
    # Determine index type
    if INDEX_TYPE_OVERRIDE:
        if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
            index_type = "IVFFLAT"
        else:
            index_type = INDEX_TYPE_OVERRIDE.upper()
    else:
        index_type = (INDEX_TYPE_OVERRIDE or "ivfflat").upper()
    if index_type == "PARALLEL_IVFFLAT":
        index_type = "IVFFLAT"

    # Load tune config
    tune_cfg = load_tune_config()
    if INDEX_PARAMS:
        tune_cfg.update(INDEX_PARAMS)
        print(f"  Overriding tune config with CLI params: {INDEX_PARAMS}")

    # Step 1: Compile the correct pgvector extension
    if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
        compile_pgvector("parallel")
    else:
        compile_pgvector("default")

    # Step 2: Find and load the correct snapshot
    snapshot_dir = Path(get_path("snapshot_dir"))
    if index_type == "IVFFLAT":
        lists = tune_cfg["lists"]
        if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
            snapshot_name = f"gist_parallel_ivfflat_lists{lists}_pgdata.tar.gz"
        else:
            snapshot_name = f"gist_ivfflat_lists{lists}_pgdata.tar.gz"
    elif index_type == "HNSW":
        m = tune_cfg["m"]
        efc = tune_cfg["ef_construction"]
        snapshot_name = f"gist_hnsw_efc{efc}_m{m}_pgdata.tar.gz"
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    snapshot_path = snapshot_dir / snapshot_name
    if not snapshot_path.exists():
        print(f"  ERROR: Snapshot not found: {snapshot_path}")
        print(f"  Available snapshots:")
        for f in sorted(snapshot_dir.iterdir()):
            print(f"    {f.name}")
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    load_snapshot(str(snapshot_path))

    return tune_cfg


def wait_for_postgres(max_attempts=60):
    """Wait for PostgreSQL to be ready."""
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(2)
    raise Exception("PostgreSQL did not become ready in time.")


def restart_postgres():
    """Restart PostgreSQL with CPU/NUMA pinning and clear OS caches."""
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

    subprocess.run(
        f"su - postgres -c '"
        f"numactl --cpunodebind={PG_NUMA_NODES} --membind={PG_NUMA_NODES} "
        f"{PG_CTL_PATH} -D {PG_DATA_DIR} -l /tmp/logfile restart'",
        shell=True, check=True, capture_output=True
    )

    time.sleep(3)
    wait_for_postgres()

    # If memory-constrained, move postgres processes into cgroup
    if MEMORY_CONFIG is not None:
        move_postgres_to_cgroup()


def load_workload():
    """Load vector workload from file (handles JSON array of arrays)."""
    with open(WORKLOAD_VECTORS_PATH, 'r') as f:
        content = f.read().strip()

    data = json.loads(content)

    # If it's a list of lists (e.g., [[vec1], [vec2], ...])
    if isinstance(data[0], list):
        vectors = data
    else:
        vectors = [data]

    print(f"Loaded {len(vectors)} workload vectors from {WORKLOAD_VECTORS_PATH}")
    return vectors


def vec_to_str(vec):
    """Convert Python list to PostgreSQL vector string."""
    return '[' + ','.join(str(x) for x in vec) + ']'


# ============== ANN WORKER ==============

def ann_worker(thread_id, workload, stop_event, count_lock, results):
    """Closed-loop ANN worker: continuously executes TopK queries."""
    # Using global deterministic query assigner
    core_id = TOPK_CORES[thread_id % len(TOPK_CORES)]

    try:
        os.sched_setaffinity(0, {core_id})
    except (OSError, AttributeError):
        pass

    local_count = 0
    local_latencies = []

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        # Disable seqscan so index is used
        cur.execute("SET enable_seqscan = off;")
        # if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
        #     cur.execute("SET max_parallel_workers_per_gather = 200;")
        if TUNE_CFG and INDEX_TYPE_OVERRIDE in ("ivfflat", "parallel_ivfflat"):
            #print("probes are  ", TUNE_CFG['probes'])
            cur.execute(f"SET ivfflat.probes = {TUNE_CFG['probes']};")
        elif TUNE_CFG and INDEX_TYPE_OVERRIDE == "hnsw":
            #print("ef_search is  ", TUNE_CFG['ef_search'])
            cur.execute(f"SET hnsw.ef_search = {TUNE_CFG.get('ef_search', 100)};")

        while not stop_event.is_set():
            idx = ann_query_assigner.get_next_query_index()
            vec_str = vec_to_str(workload[idx])

            start = time.time()
            cur.execute(TOPK_QUERY % (vec_str, K))
            cur.fetchall()
            elapsed = time.time() - start

            # Only count if completed before stop event
            if not stop_event.is_set():
                local_count += 1
                local_latencies.append(elapsed)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"  ANN worker {thread_id} error: {e}")

    with count_lock:
        results['ann_count'] += local_count
        results['ann_latencies'].extend(local_latencies)


# ============== OLAP WORKER ==============

def olap_worker(thread_id, stop_event, count_lock, results):
    """
    Closed-loop OLAP worker: continuously executes TPC-H queries one at a time.
    Uses olap_query_assigner for deterministic query order (same as ANN worker pattern).
    """
    core_id = OLAP_CORES[thread_id % len(OLAP_CORES)]

    try:
        os.sched_setaffinity(0, {core_id})
    except (OSError, AttributeError):
        pass

    local_query_count = 0
    local_latencies = []

    # Build executable query list (handle Q15 specially)
    executable_queries = []
    for qname, sql in TPCH_QUERIES:
        if qname == "Q15_create" or sql is None:
            continue
        executable_queries.append((qname, sql))

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        # Enable seqscan for OLAP
        cur.execute("SET enable_seqscan = on;")
        cur.execute("SET work_mem = '100MB';")
        cur.execute(f"SET statement_timeout = {MEASUREMENT_SEC * 1000}")

        while not stop_event.is_set():
            qi = olap_query_assigner.get_next_query_index()
            qname, sql = executable_queries[qi]

            start = time.time()

            # Handle Q15: create view, run query, drop view
            if qname == "Q15":
                try:
                    cur.execute("CREATE OR REPLACE VIEW revenue0 (supplier_no, total_revenue) AS "
                                "SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount)) "
                                "FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' "
                                "AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH "
                                "GROUP BY l_suppkey")
                    cur.execute(sql)
                    cur.fetchall()
                    cur.execute("DROP VIEW IF EXISTS revenue0")
                except Exception:
                    try:
                        cur.execute("DROP VIEW IF EXISTS revenue0")
                    except:
                        pass
                    continue
            else:
                try:
                    cur.execute(sql)
                    cur.fetchall()
                except Exception:
                    continue

            elapsed = time.time() - start

            if not stop_event.is_set():
                local_query_count += 1
                local_latencies.append(elapsed)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"  OLAP worker {thread_id} error: {e}")

    with count_lock:
        results['olap_query_count'] += local_query_count
        results['olap_latencies'].extend(local_latencies)


# ============== MEASUREMENT FUNCTION ==============

def measure_throughput(num_ann_workers, num_olap_workers, workload,
                       measurement_sec=MEASUREMENT_SEC):
    """
    Restart DB (cold), run workload for fixed measurement period, return throughput.
    Both ANN and OLAP workers run as closed-loop clients.
    No warmup — measures cold-start performance.
    """
    print(f"\n  Measuring: {num_ann_workers} ANN workers, {num_olap_workers} OLAP workers")
    print(f"  Duration: {measurement_sec}s (cold start, no warmup)")

    # Restart with clean state (clears OS page cache + restarts postgres)
    if not WARM_MODE:
        restart_postgres()

    stop_event = threading.Event()
    count_lock = threading.Lock()
    measure_results = {'ann_count': 0, 'ann_latencies': [], 'olap_latencies': [], 'olap_query_count': 0}

    # Initialize deterministic query assigners
    global ann_query_assigner, olap_query_assigner

    ann_query_assigner = DeterministicQueryAssigner(
        seed=42,
        workload_size=len(workload),
        sequence_length=1000000
    )

    executable_queries = []
    for qname, sql in TPCH_QUERIES:
        if qname == "Q15_create" or sql is None:
            continue
        executable_queries.append((qname, sql))

    olap_query_assigner = DeterministicQueryAssigner(
        seed=100,
        workload_size=len(executable_queries),
        sequence_length=1000000
    )

    ann_query_assigner.reset()
    olap_query_assigner.reset()

    threads = []

    for i in range(num_ann_workers):
        t = threading.Thread(target=ann_worker, args=(i, workload, stop_event, count_lock, measure_results))
        t.start()
        threads.append(t)

    for i in range(num_olap_workers):
        t = threading.Thread(target=olap_worker, args=(i, stop_event, count_lock, measure_results))
        t.start()
        threads.append(t)

    print(f"  Running for {measurement_sec}s...")
    measurement_start = time.time()

    time.sleep(measurement_sec)

    # Stop all threads
    stop_event.set()
    for t in threads:
        t.join(timeout=360)  # Wait at most 6 min for thread to finish current query

    # --- Compute metrics ---
    ann_throughput = measure_results['ann_count'] / measurement_sec
    olap_throughput = measure_results['olap_query_count'] / measurement_sec  # queries/sec

    ann_latencies = measure_results['ann_latencies']
    olap_latencies = measure_results['olap_latencies']  # per-query latencies (seconds)

    ann_avg_latency = (sum(ann_latencies) / len(ann_latencies) * 1000) if ann_latencies else 0
    olap_avg_latency = (sum(olap_latencies) / len(olap_latencies) * 1000) if olap_latencies else 0

    # Percentiles
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    result = {
        "ann_workers": num_ann_workers,
        "olap_workers": num_olap_workers,
        "ann_throughput_qps": round(ann_throughput, 2),
        "olap_throughput_qps": round(olap_throughput, 4),
        "ann_total_queries": measure_results['ann_count'],
        "olap_total_queries": measure_results['olap_query_count'],
        "ann_avg_latency_ms": round(ann_avg_latency, 2),
        "ann_p50_latency_ms": round(percentile(ann_latencies, 50) * 1000, 2),
        "ann_p99_latency_ms": round(percentile(ann_latencies, 99) * 1000, 2),
        "olap_avg_latency_ms": round(olap_avg_latency, 2),
        "olap_p50_latency_ms": round(percentile(olap_latencies, 50) * 1000, 2),
        "olap_p99_latency_ms": round(percentile(olap_latencies, 99) * 1000, 2),
        "measurement_duration_sec": measurement_sec,
    }

    print(f"  Results: ANN={ann_throughput:.2f} qps, OLAP={olap_throughput:.4f} qps")
    return result


# ============== MAIN ==============

def find_ann_saturation(workload):
    """Find maximum ANN throughput (no OLAP running).
    Resumes from existing measurements if saturation_points_ann.json exists.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Finding ANN Saturation (X^ANN)")
    print("Running ANN alone with increasing workers...")
    print("=" * 70)

    results = []
    max_throughput = 0
    saturation_workers = 0

    # Load existing measurements to skip already-computed points
    ann_file = os.path.join(RESULTS_DIR, "saturation_points_ann.json")
    existing_workers = set()
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            existing_data = json.load(f)
        if 'ann_saturation' in existing_data:
            for m in existing_data['ann_saturation'].get('measurements', []):
                w = m.get('ann_workers', m.get('num_ann_workers'))
                if w is not None:
                    existing_workers.add(w)
                    results.append(m)
                    qps = m['ann_throughput_qps']
                    if qps > max_throughput:
                        max_throughput = qps
                        saturation_workers = w
            if existing_workers:
                print(f"  Resuming: found existing measurements for workers {sorted(existing_workers)}")
                print(f"  Current best: {max_throughput:.2f} qps at {saturation_workers} workers")

    # Use separate range for parallel_ivfflat
    if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
        ann_workers_list = ANN_WORKERS_PARALLEL_IVFFLAT
    else:
        ann_workers_list = ANN_WORKERS_TO_TEST

    for num_workers in ann_workers_list:
        # Skip already-computed points
        if num_workers in existing_workers:
            print(f"  >> {num_workers} workers: SKIPPED (already measured)")
            continue

        result = measure_throughput(
            num_ann_workers=num_workers,
            num_olap_workers=0,
            workload=workload
        )
        results.append(result)

        qps = result['ann_throughput_qps']
        print(f"  >> {num_workers} workers: {qps:.2f} qps")

        if qps > max_throughput:
            # Check if improvement is less than threshold (saturated)
            if max_throughput > 0:
                improvement = (qps - max_throughput) / max_throughput
                if improvement < SATURATION_THRESHOLD:
                    print(f"     (improvement {improvement*100:.1f}% < {SATURATION_THRESHOLD*100:.0f}% threshold, saturated)")
                    max_throughput = qps
                    saturation_workers = num_workers
                    break
            max_throughput = qps
            saturation_workers = num_workers
        # else:
        #     # Throughput decreased — already past saturation
        #     print(f"     (throughput decreased, saturated)")
        #     break

    print(f"\n  *** ANN Saturation: X^ANN = {max_throughput:.2f} qps "
          f"(at {saturation_workers} workers) ***\n")

    return {
        "X_ANN": max_throughput,
        "saturation_workers": saturation_workers,
        "measurements": results
    }


def find_olap_saturation(workload):
    """Find maximum OLAP throughput (no ANN running)."""
    print("\n" + "=" * 70)
    print("PHASE 2: Finding OLAP Saturation (X^OLAP)")
    print("Running OLAP alone with increasing workers...")
    print("=" * 70)

    results = []
    max_throughput = 0
    saturation_workers = 0

    for num_workers in OLAP_WORKERS_TO_TEST:
        result = measure_throughput(
            num_ann_workers=0,
            num_olap_workers=num_workers,
            workload=workload
        )
        results.append(result)

        qps = result["olap_throughput_qps"]

        print(f"  >> {num_workers} workers: {qps:.4f} qps")

        if qps > max_throughput:
            # Check if improvement is less than threshold (saturated)
            if max_throughput > 0:
                improvement = (qps - max_throughput) / max_throughput
                if improvement < SATURATION_THRESHOLD:
                    print(f"     (improvement {improvement*100:.1f}% < {SATURATION_THRESHOLD*100:.0f}% threshold, saturated)")
                    max_throughput = qps
                    saturation_workers = num_workers
                    break
            max_throughput = qps
            saturation_workers = num_workers
        # else:
        #     # Throughput decreased — already past saturation
        #     print(f"     (throughput decreased, saturated)")
        #     break

    print(f"\n  *** OLAP Saturation: X^OLAP = {max_throughput:.4f} qps "
          f"(at {saturation_workers} workers) ***\n")

    return {
        "X_OLAP": max_throughput,
        "saturation_workers": saturation_workers,
        "measurements": results
    }


def main():
    global MEMORY_CONFIG, RESULTS_DIR, TUNE_CFG, WARM_MODE, INDEX_TYPE_OVERRIDE, INDEX_PARAMS, SNAPSHOT_MODE

    run_ann = True
    run_olap = True

    if "--ann-only" in sys.argv:
        run_olap = False
    if "--olap-only" in sys.argv:
        run_ann = False

    # Parse --memory SB:CG flag (e.g., --memory 12:16 = 12GB shared_buffers, 16GB cgroup)
    if "--memory" in sys.argv:
        idx = sys.argv.index("--memory")
        if idx + 1 >= len(sys.argv):
            print("ERROR: --memory requires argument in format SB:CG (e.g., 12:16)")
            sys.exit(1)
        mem_arg = sys.argv[idx + 1]
        try:
            sb_gb, cg_gb = mem_arg.split(":")
            sb_gb = int(sb_gb)
            cg_gb = int(cg_gb)
        except ValueError:
            print(f"ERROR: Invalid --memory format '{mem_arg}'. Use SB:CG (e.g., 12:16)")
            sys.exit(1)

        MEMORY_CONFIG = {"shared_buffers_gb": sb_gb, "cgroup_gb": cg_gb}
        RESULTS_DIR = str(TPCH_DIR / f"results_frontier_constrained_{sb_gb}GB_sb_{cg_gb}GB_cg")

    # Parse --warm flag
    if "--warm" in sys.argv:
        WARM_MODE = True
        # Modify results dir to include 'warm' prefix
        RESULTS_DIR = RESULTS_DIR.replace("results_frontier_", "results_frontier_warm_")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"WARM MODE: No restart between measurements. Results: {RESULTS_DIR}")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        print(f"MEMORY CONSTRAINED MODE: shared_buffers={sb_gb}GB, cgroup={cg_gb}GB")
        print(f"Results directory: {RESULTS_DIR}")

        # Apply memory configuration
        set_shared_buffers(sb_gb)
        set_cgroup_memory(cg_gb)


    # Parse --index flag (ivfflat, hnsw, parallel_ivfflat)
    if "--index" in sys.argv:
        idx_i = sys.argv.index("--index")
        if idx_i + 1 >= len(sys.argv):
            print("ERROR: --index requires argument (ivfflat, hnsw, parallel_ivfflat)")
            sys.exit(1)
        INDEX_TYPE_OVERRIDE = sys.argv[idx_i + 1].lower()
        valid_indexes = ["ivfflat", "hnsw", "parallel_ivfflat"]
        if INDEX_TYPE_OVERRIDE not in valid_indexes:
            print(f"ERROR: --index must be one of {valid_indexes}, got '{INDEX_TYPE_OVERRIDE}'")
            sys.exit(1)

    # Parse HNSW manual params
    if "--hnsw-m" in sys.argv:
        idx_m = sys.argv.index("--hnsw-m")
        INDEX_PARAMS["m"] = int(sys.argv[idx_m + 1])
    if "--hnsw-efc" in sys.argv:
        idx_e = sys.argv.index("--hnsw-efc")
        INDEX_PARAMS["ef_construction"] = int(sys.argv[idx_e + 1])

    # Parse IVFFlat manual params
    if "--ivfflat-lists" in sys.argv:
        idx_l = sys.argv.index("--ivfflat-lists")
        INDEX_PARAMS["lists"] = int(sys.argv[idx_l + 1])

    # Opt-in run variants (e.g. ~0.967-recall or no-optimization runs); nothing changes without these flags.
    global KNOB, RECALL_NOTE, VARIANT, RUN_SUFFIX
    if "--ef-search" in sys.argv:
        KNOB["ef_search"] = int(sys.argv[sys.argv.index("--ef-search") + 1])
        RUN_SUFFIX += f"_ef{KNOB['ef_search']}"
    if "--probes" in sys.argv:
        KNOB["probes"] = int(sys.argv[sys.argv.index("--probes") + 1])
        RUN_SUFFIX += f"_p{KNOB['probes']}"
    if "--recall" in sys.argv:
        RECALL_NOTE = float(sys.argv[sys.argv.index("--recall") + 1])
    if "--variant" in sys.argv:
        VARIANT = sys.argv[sys.argv.index("--variant") + 1]
        assert re.fullmatch(r"[a-z0-9]+", VARIANT), "--variant must be lowercase letters/digits, e.g. noopt"
        RUN_SUFFIX += f"_{VARIANT}"

    # Parse --snapshot flag (skip ensure_index, index loaded from snapshot)
    if "--snapshot" in sys.argv:
        SNAPSHOT_MODE = True
        print("SNAPSHOT MODE: Skipping ensure_index (index already in PGDATA snapshot)")

    # Modify results dir to include index name (only for non-default indexes)
    if INDEX_TYPE_OVERRIDE and INDEX_TYPE_OVERRIDE != "ivfflat":
        idx_short = INDEX_TYPE_OVERRIDE.replace("parallel_ivfflat", f"pivfflat_w{pgvector_num_workers()}")
        # Insert index name into results dir path
        RESULTS_DIR = RESULTS_DIR.replace("results_frontier_", f"results_frontier_{idx_short}_")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"INDEX: {INDEX_TYPE_OVERRIDE.upper()} (results: {RESULTS_DIR})")
        if INDEX_PARAMS:
            print(f"  Manual params: {INDEX_PARAMS}")

        print("=" * 70)
    if RUN_SUFFIX:
        RESULTS_DIR += RUN_SUFFIX
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"RUN VARIANT (opt-in): knob={KNOB} recall={RECALL_NOTE} variant={VARIANT}. Results: {RESULTS_DIR}")
    print("THROUGHPUT FRONTIER - Step 1: Find Saturation Points")
    print(f"Index: {INDEX_TYPE_OVERRIDE or "ivfflat"}, K: {K}, "
          f"Measurement: {MEASUREMENT_SEC}s per data point")
    print(f"Measurement: {MEASUREMENT_SEC}s per data point (cold start, no warmup)")
    print(f"Postgres cores: {POSTGRES_CORES}, NUMA: {PG_NUMA_NODES}")
    print(f"Client cores (TopK): {TOPK_CORES[:4]}...")
    print(f"Client cores (OLAP): {OLAP_CORES[:4]}...")
    if MEMORY_CONFIG:
        print(f"Memory: shared_buffers={MEMORY_CONFIG['shared_buffers_gb']}GB, cgroup={MEMORY_CONFIG['cgroup_gb']}GB")
    else:
        print(f"No memory constraint (unconstrained cgroup)")
    print("=" * 70)

    # Ensure correct index exists (from tune config)
    if SNAPSHOT_MODE:
        print("\nSnapshot mode: skipping index creation (already in PGDATA).")
        # Still load tune config for probes/ef_search settings
        TUNE_CFG = load_tune_config()
        if INDEX_PARAMS:
            TUNE_CFG.update(INDEX_PARAMS)
    else:
        print("\nEnsuring correct index from tune config...")
        TUNE_CFG = ensure_index()

    # If warm mode, do one cold restart then prewarm
    if WARM_MODE:
        print("\nWARM MODE: Performing initial restart and prewarm...")
        restart_postgres()
        prewarm_database()
        print("Database is now in steady state. No more restarts.")

    # Load workload vectors
    workload = load_workload()

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "index_type": INDEX_TYPE_OVERRIDE or "ivfflat",
            "parallel_index": INDEX_TYPE_OVERRIDE == "parallel_ivfflat",
            "io_depth": 1000,
            "k": K,
            "measurement_sec": MEASUREMENT_SEC,
            "postgres_cores": POSTGRES_CORES,
            "numa_nodes": PG_NUMA_NODES,
            "num_workers": pgvector_num_workers(),
            "memory_config": MEMORY_CONFIG,
        }
    }
    if KNOB:
        output["config"]["search_knob"] = KNOB
    if RECALL_NOTE is not None:
        output["config"]["recall"] = RECALL_NOTE
    if VARIANT:
        output["config"]["variant"] = VARIANT

    # Check if saturation results already exist (skip if found)
    ann_file = os.path.join(RESULTS_DIR, "saturation_points_ann.json")
    olap_file = os.path.join(RESULTS_DIR, "saturation_points_olap.json")

    # For non-IVFFlat indexes:
    #   - OLAP saturation: ALWAYS copy from IVFFlat (index-independent)
    #   - ANN saturation: ALWAYS measure (index-specific)
    if INDEX_TYPE_OVERRIDE and INDEX_TYPE_OVERRIDE != "ivfflat":
        # Get default (ivfflat) results directory
        idx_short = INDEX_TYPE_OVERRIDE.replace("parallel_ivfflat", f"pivfflat_w{pgvector_num_workers()}")
        default_results = RESULTS_DIR.replace(f"_{idx_short}_", "_")
        if RUN_SUFFIX and default_results.endswith(RUN_SUFFIX):
            default_results = default_results[:-len(RUN_SUFFIX)]  # OLAP-only saturation: copy from the default IVFFlat dir

        # Always copy OLAP from IVFFlat
        default_olap = os.path.join(default_results, "saturation_points_olap.json")
        if os.path.exists(default_olap):
            import shutil as _shutil
            _shutil.copy2(default_olap, olap_file)
            print(f"  Copied OLAP saturation from IVFFlat run: {default_olap}")
            print(f"  -> {olap_file}")
        else:
            print(f"  WARNING: No OLAP saturation found at {default_olap}")
            print(f"  Run IVFFlat frontier first!")

        # ANN: measure only if file doesn't exist (never copy from IVFFlat)
        run_olap = False
        if os.path.exists(ann_file) and "--ann-only" not in sys.argv:
            print(f"  ANN saturation already exists: {ann_file} (skipping)")
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            if 'ann_saturation' in ann_data:
                print(f"  X^ANN  = {ann_data['ann_saturation']['X_ANN']:.2f} qps")
            run_ann = False
        else:
            run_ann = True
            print(f"  Will measure ANN saturation for {INDEX_TYPE_OVERRIDE}...")

    else:
        # For IVFFlat: check if files exist and skip individually
        if os.path.exists(olap_file):
            print(f"\n  OLAP saturation already exists: {olap_file} (skipping)")
            with open(olap_file, 'r') as f:
                olap_data = json.load(f)
            if 'olap_saturation' in olap_data:
                print(f"  X^OLAP = {olap_data['olap_saturation']['X_OLAP']:.4f} qps")
            run_olap = False

        if os.path.exists(ann_file) and "--ann-only" not in sys.argv:
            print(f"  ANN saturation already exists: {ann_file} (skipping)")
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            if 'ann_saturation' in ann_data:
                print(f"  X^ANN  = {ann_data['ann_saturation']['X_ANN']:.2f} qps")
            run_ann = False

    if not run_olap and not run_ann:
        print(f"  Both saturation files exist. Nothing to measure.")
        return

    # Run OLAP saturation FIRST (so we quickly see approximate throughput)
    if run_olap:
        olap_result = find_olap_saturation(workload)
        output["olap_saturation"] = olap_result

    # Run ANN saturation
    if run_ann:
        ann_result = find_ann_saturation(workload)
        output["ann_saturation"] = ann_result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if run_ann:
        print(f"  X^ANN  = {ann_result['X_ANN']:.2f} qps "
              f"(saturates at {ann_result['saturation_workers']} workers)")
    if run_olap:
        print(f"  X^OLAP = {olap_result['X_OLAP']:.4f} qps "
              f"(saturates at {olap_result['saturation_workers']} workers)")
    print("=" * 70)

    # Save results as separate files (matching build_throughput_frontier.py expectations)
    if run_ann:
        ann_file = os.path.join(RESULTS_DIR, "saturation_points_ann.json")
        ann_output = {k: v for k, v in output.items() if k != "olap_saturation"}
        with open(ann_file, "w") as f:
            json.dump(ann_output, f, indent=2)
        print(f"\nANN results saved to: {ann_file}")

    if run_olap:
        olap_file = os.path.join(RESULTS_DIR, "saturation_points_olap.json")
        olap_output = {k: v for k, v in output.items() if k != "ann_saturation"}
        with open(olap_file, "w") as f:
            json.dump(olap_output, f, indent=2)
        print(f"\nOLAP results saved to: {olap_file}")

    # Also save combined file for reference
    combined_file = os.path.join(RESULTS_DIR, "saturation_points.json")
    with open(combined_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Combined results saved to: {combined_file}")


if __name__ == "__main__":
    main()
