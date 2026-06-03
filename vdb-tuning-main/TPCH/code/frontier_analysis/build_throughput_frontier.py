#!/usr/bin/env python3
"""
Step 2 of Throughput Frontier: Build Grid Graph & Extract Frontier.

Following the methodology of "How Good is My HTAP System?" (SIGMOD'22, Section 3.3),
this script:
  1. Reads saturation points from Step 1 (X^ANN, X^OLAP, tau_max, alpha_max)
  2. Builds the grid graph:
     - Fixed-ANN lines: fix ANN workers at fractions of tau_max, vary OLAP workers
     - Fixed-OLAP lines: fix OLAP workers at fractions of alpha_max, vary ANN workers
  3. Extracts the throughput frontier (Pareto-optimal boundary)
  4. Computes metrics: AUC, proportional line, bounding box ratio

Usage:
    sudo python3 build_throughput_frontier.py [--skip-measurements] [--fixed-ann-only] [--fixed-olap-only]

    --skip-measurements: Only compute frontier from existing grid_graph.json
    --fixed-ann-only:    Only run fixed-ANN lines
    --fixed-olap-only:   Only run fixed-OLAP lines

Output:
    grid_graph.json          - All measurement points
    throughput_frontier.json  - Frontier points + metrics (AUC, etc.)
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
import numpy as np
from pathlib import Path
from datetime import datetime

# ============== AUTO-DETECT PATHS ==============
SCRIPT_DIR = Path(__file__).resolve().parent                    # frontier_analysis/
TPCH_DIR = SCRIPT_DIR.parent.parent                             # TPCH/
PROJECT_DIR = TPCH_DIR.parent                                    # vdb-tuning-main/

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

# Paths
WORKLOAD_VECTORS_PATH = str(PROJECT_DIR / IDX_CFG["workload_vectors_file"])
TUNE_RESULTS_DIR = PROJECT_DIR / "compute_and_compare_recall" / "results_from_tune_scripts"

RESULTS_DIR = str(TPCH_DIR / "results_throughput_frontier")
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU pinning
POSTGRES_CORES = HW_CFG["postgres_cores"]
PG_NUMA_NODES = HW_CFG["postgres_numa_nodes"]
TOPK_CORES = HW_CFG["topk_producers_cores"]
OLAP_CORES = HW_CFG["olap_producers_cores"]

# PostgreSQL paths
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
_pgdata_result = subprocess.run(
    "su - postgres -c 'echo $PGDATA'", shell=True, capture_output=True, text=True
)
PG_DATA_DIR = _pgdata_result.stdout.strip() or "/mnt/nvme/pgdata"
PG_CONF_PATH = os.path.join(PG_DATA_DIR, "postgresql.conf")
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"

# Memory constraint config (set via --memory flag)
MEMORY_CONFIG = None

# Tune configuration (loaded from tune results)
TUNE_CFG = None

# Warm mode: prewarm DB once, no restart between measurements
WARM_MODE = False  # Will be {"shared_buffers_gb": X, "cgroup_gb": Y} if --memory used

# Multi-index support: override index type and params via CLI
INDEX_TYPE_OVERRIDE = None  # Set by --index flag (ivfflat, hnsw, parallel_ivfflat)
INDEX_PARAMS = {}  # Manual params: {"m": 16, "ef_construction": 128} or {"lists": 8192}

# Snapshot mode: skip ensure_index (index already loaded from PGDATA snapshot)
SNAPSHOT_MODE = False

# ============== EXPERIMENT PARAMETERS ==============
K = IDX_CFG["k"]

# Measurement duration per data point (same as saturation script)
MEASUREMENT_SEC = 300  # 5 minutes per point, cold start

# Grid graph fractions (following paper: 0, 0.1, 0.2, 0.5, 0.8, 1.0 of max)
#GRID_FRACTIONS = [0.0, 0.01, 0.02, 0.2, 0.4, 0.6, 0.8, 1.0]
GRID_FRACTIONS = [0.0, 0.1, 0.3, 0.6, 1.0]
DENSE_GRID_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    index_type = IDX_CFG["index_type"].upper()
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
    return best


def ensure_index():
    """Ensure the correct index exists based on tune config or CLI override. Build if needed."""
    # Determine index type: CLI override > config
    if INDEX_TYPE_OVERRIDE:
        if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
            index_type = "IVFFLAT"  # Same index, different query strategy
        else:
            index_type = INDEX_TYPE_OVERRIDE.upper()
    else:
        index_type = IDX_CFG["index_type"].upper()

    # Load tune config, then override with manual CLI params
    tune_cfg = load_tune_config()
    if INDEX_PARAMS:
        tune_cfg.update(INDEX_PARAMS)
        print(f"  Overriding tune config with CLI params: {INDEX_PARAMS}")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        SELECT am.amname, ic.reloptions
        FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_class tc ON tc.oid = i.indrelid
        JOIN pg_am am ON am.oid = ic.relam
        WHERE tc.relname = 'gist_data'
        AND ic.relname != 'gist_data_pkey'
    """)
    existing = cur.fetchall()

    needs_rebuild = True

    if existing:
        current_am = existing[0][0]
        current_opts = existing[0][1] or []

        if index_type == "IVFFLAT" and current_am == "ivfflat":
            for opt in current_opts:
                if opt.startswith("lists=") and int(opt.split("=")[1]) == tune_cfg["lists"]:
                    needs_rebuild = False
                    print(f"  Index OK: ivfflat with lists={tune_cfg['lists']} already exists")
                    break
        elif index_type == "HNSW" and current_am == "hnsw":
            for opt in current_opts:
                if opt.startswith("m=") and int(opt.split("=")[1]) == tune_cfg["m"]:
                    needs_rebuild = False
                    print(f"  Index OK: hnsw with m={tune_cfg['m']} already exists")
                    break

    if needs_rebuild:
        print(f"  Building {index_type} index from tune config...")
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'gist_data' AND indexname != 'gist_data_pkey'")
        for (idx_name,) in cur.fetchall():
            print(f"    Dropping index: {idx_name}")
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

        if index_type == "IVFFLAT":
            lists = tune_cfg["lists"]
            print(f"    Creating IVFFlat index with lists={lists}...")
            cur.execute(
                f"CREATE INDEX gist_data_embedding_idx ON gist_data "
                f"USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists})"
            )
        elif index_type == "HNSW":
            m = tune_cfg["m"]
            ef_construction = tune_cfg["ef_construction"]
            print(f"    Creating HNSW index with m={m}, ef_construction={ef_construction}...")
            cur.execute(
                f"CREATE INDEX gist_data_embedding_idx ON gist_data "
                f"USING hnsw (embedding vector_l2_ops) WITH (m = {m}, ef_construction = {ef_construction})"
            )
        print(f"    Index created successfully.")

    cur.close()
    conn.close()
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
    """Load vector workload from file."""
    with open(WORKLOAD_VECTORS_PATH, 'r') as f:
        content = f.read().strip()

    data = json.loads(content)

    if isinstance(data[0], list):
        vectors = data
    else:
        vectors = [data]

    print(f"Loaded {len(vectors)} workload vectors from {WORKLOAD_VECTORS_PATH}")
    return vectors


def vec_to_str(vec):
    """Convert Python list to PostgreSQL vector string."""
    return '[' + ','.join(str(x) for x in vec) + ']'


def load_saturation_points():
    """Load saturation points from Step 1 results."""
    ann_path = os.path.join(RESULTS_DIR, "saturation_points_ann.json")
    olap_path = os.path.join(RESULTS_DIR, "saturation_points_olap.json")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ANN saturation file not found: {ann_path}\n"
                                f"Run find_saturation_points.py --ann-only first.")
    if not os.path.exists(olap_path):
        raise FileNotFoundError(f"OLAP saturation file not found: {olap_path}\n"
                                f"Run find_saturation_points.py --olap-only first.")

    with open(ann_path, 'r') as f:
        ann_data = json.load(f)
    with open(olap_path, 'r') as f:
        olap_data = json.load(f)

    tau_max = ann_data["ann_saturation"]["saturation_workers"]
    x_ann = ann_data["ann_saturation"]["X_ANN"]
    alpha_max = olap_data["olap_saturation"]["saturation_workers"]
    x_olap = olap_data["olap_saturation"]["X_OLAP"]

    print(f"Saturation points loaded:")
    print(f"  X^ANN  = {x_ann:.2f} qps (tau_max = {tau_max} workers)")
    print(f"  X^OLAP = {x_olap:.4f} tps (alpha_max = {alpha_max} workers)")

    return {
        "tau_max": tau_max,
        "alpha_max": alpha_max,
        "X_ANN": x_ann,
        "X_OLAP": x_olap,
        "ann_saturation_data": ann_data,
        "olap_saturation_data": olap_data,
    }


# ============== ANN WORKER ==============

def ann_worker(thread_id, workload, stop_event, count_lock, results):
    """Closed-loop ANN worker: continuously executes TopK queries."""
    local_random = random.Random(thread_id + 42)
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

        #cur.execute("SET enable_seqscan = off;")
        # if INDEX_TYPE_OVERRIDE == "parallel_ivfflat":
        #     cur.execute("SET max_parallel_workers_per_gather = 200;")
        if TUNE_CFG and (IDX_CFG["index_type"].upper() == "IVFFLAT" or INDEX_TYPE_OVERRIDE in ("ivfflat", "parallel_ivfflat")):
            print("PROBEEEEEEEEEEEEEEEEESSSSS ARE  ", TUNE_CFG['probes'])
            cur.execute(f"SET ivfflat.probes = {TUNE_CFG['probes']};")
        elif TUNE_CFG and (IDX_CFG["index_type"].upper() == "HNSW" or INDEX_TYPE_OVERRIDE == "hnsw"):
            cur.execute(f"SET hnsw.ef_search = {TUNE_CFG.get('ef_search', 100)};")

        while not stop_event.is_set():
            idx = local_random.randint(0, len(workload) - 1)
            vec_str = vec_to_str(workload[idx])

            start = time.time()
            cur.execute(TOPK_QUERY % (vec_str, K))
            cur.fetchall()
            elapsed = time.time() - start

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
    Closed-loop OLAP worker: continuously runs full TPC-H batches.
    Each completed batch of all 22 queries counts as 1 OLAP "transaction".
    """
    core_id = OLAP_CORES[thread_id % len(OLAP_CORES)]

    try:
        os.sched_setaffinity(0, {core_id})
    except (OSError, AttributeError):
        pass

    local_random = random.Random(thread_id + 100)
    local_batch_count = 0
    local_query_count = 0
    local_batch_latencies = []
    local_query_count = 0

    # Build executable query list
    executable_queries = []
    for qname, sql in TPCH_QUERIES:
        if qname == "Q15_create" or sql is None:
            continue
        executable_queries.append((qname, sql))

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("SET enable_seqscan = on;")

        while not stop_event.is_set():
            batch_start = time.time()
            batch_order = list(range(len(executable_queries)))
            local_random.shuffle(batch_order)

            batch_completed = True
            for qi in batch_order:
                if stop_event.is_set():
                    batch_completed = False
                    break

                qname, sql = executable_queries[qi]

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
                        local_query_count += 1
                    except Exception:
                        try:
                            cur.execute("DROP VIEW IF EXISTS revenue0")
                        except:
                            pass
                else:
                    try:
                        cur.execute(sql)
                        cur.fetchall()
                        local_query_count += 1
                    except Exception:
                        pass

            if batch_completed:
                batch_elapsed = time.time() - batch_start
                local_batch_count += 1
                local_batch_latencies.append(batch_elapsed)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"  OLAP worker {thread_id} error: {e}")

    with count_lock:
        results['olap_count'] += local_batch_count
        results['olap_latencies'].extend(local_batch_latencies)
        results['olap_query_count'] += local_query_count


# ============== MEASUREMENT FUNCTION ==============

def measure_throughput(num_ann_workers, num_olap_workers, workload,
                       measurement_sec=MEASUREMENT_SEC):
    """
    Restart DB (cold), run mixed workload for fixed measurement period, return throughput.
    """
    print(f"\n  Measuring: {num_ann_workers} ANN workers, {num_olap_workers} OLAP workers "
          f"({measurement_sec}s, {"warm" if WARM_MODE else "cold start"})")

    if not WARM_MODE:
        restart_postgres()

    stop_event = threading.Event()
    count_lock = threading.Lock()
    measure_results = {'ann_count': 0, 'ann_latencies': [], 'olap_count': 0, 'olap_latencies': [], 'olap_query_count': 0}
    threads = []

    for i in range(num_ann_workers):
        t = threading.Thread(target=ann_worker, args=(i, workload, stop_event, count_lock, measure_results))
        t.start()
        threads.append(t)

    for i in range(num_olap_workers):
        t = threading.Thread(target=olap_worker, args=(i, stop_event, count_lock, measure_results))
        t.start()
        threads.append(t)

    time.sleep(measurement_sec)

    stop_event.set()
    for t in threads:
        t.join()

    # Compute metrics
    ann_throughput = measure_results['ann_count'] / measurement_sec
    olap_throughput = measure_results['olap_count'] / measurement_sec
    olap_query_throughput = measure_results['olap_query_count'] / measurement_sec

    ann_latencies = measure_results['ann_latencies']
    olap_latencies = measure_results['olap_latencies']

    ann_avg_latency = (sum(ann_latencies) / len(ann_latencies) * 1000) if ann_latencies else 0
    olap_avg_batch_time = (sum(olap_latencies) / len(olap_latencies)) if olap_latencies else 0

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
        "olap_throughput_tps": round(olap_throughput, 4),
        "olap_throughput_qps": round(olap_query_throughput, 4),
        "ann_total_queries": measure_results['ann_count'],
        "olap_total_queries": measure_results['olap_query_count'],
        "olap_total_batches": measure_results['olap_count'],
        "ann_avg_latency_ms": round(ann_avg_latency, 2),
        "ann_p50_latency_ms": round(percentile(ann_latencies, 50) * 1000, 2),
        "ann_p99_latency_ms": round(percentile(ann_latencies, 99) * 1000, 2),
        "olap_avg_batch_time_sec": round(olap_avg_batch_time, 3),
        "measurement_duration_sec": measurement_sec,
    }

    print(f"  Results: ANN={ann_throughput:.2f} qps, OLAP={olap_query_throughput:.2f} qps ({olap_throughput:.4f} batches/s)")
    return result


# ============== GRID GRAPH CONSTRUCTION ==============

def compute_grid_points(tau_max, alpha_max):
    """
    Compute the (ann_workers, olap_workers) pairs for the grid graph.

    Returns:
        fixed_ann_lines: dict mapping fixed_ann_count -> list of olap_counts to test
        fixed_olap_lines: dict mapping fixed_olap_count -> list of ann_counts to test
    """
    # Always include low worker counts (1,2,3) for both ANN and OLAP
    ALWAYS_ANN = [1, 2, 3]
    ALWAYS_OLAP = [1, 2, 3]

    # Fixed-ANN lines: fix ANN at fractions of tau_max, vary OLAP
    ann_levels = sorted(set(
        [max(0, round(f * tau_max)) for f in GRID_FRACTIONS] + ALWAYS_ANN
    ))
    olap_sweep = sorted(set(
        [max(0, round(f * alpha_max)) for f in GRID_FRACTIONS] + ALWAYS_OLAP
    ))

    # Fixed-OLAP lines: fix OLAP at fractions of alpha_max, vary ANN
    olap_levels = sorted(set(
        [max(0, round(f * alpha_max)) for f in GRID_FRACTIONS] + ALWAYS_OLAP
    ))
    ann_sweep = sorted(set(
        [max(0, round(f * tau_max)) for f in GRID_FRACTIONS] + ALWAYS_ANN
    ))

    fixed_ann_lines = {}
    for ann_count in ann_levels:
        if ann_count == 0:
            # ANN=0 is just OLAP-only (already measured in saturation)
            continue
        fixed_ann_lines[ann_count] = olap_sweep

    fixed_olap_lines = {}
    for olap_count in olap_levels:
        if olap_count == 0:
            # OLAP=0 is just ANN-only (already measured in saturation)
            continue
        fixed_olap_lines[olap_count] = ann_sweep

    return fixed_ann_lines, fixed_olap_lines, ann_levels, olap_levels


def run_grid_measurements(workload, saturation, run_fixed_ann=True, run_fixed_olap=True):
    """Run all grid graph measurements."""
    tau_max = saturation["tau_max"]
    alpha_max = saturation["alpha_max"]

    fixed_ann_lines, fixed_olap_lines, ann_levels, olap_levels = compute_grid_points(tau_max, alpha_max)

    # Load existing progress if available
    grid_file = os.path.join(RESULTS_DIR, "grid_graph.json")
    if os.path.exists(grid_file):
        with open(grid_file, 'r') as f:
            grid_data = json.load(f)
        measurements = grid_data.get("measurements", [])
        print(f"Resuming: {len(measurements)} existing measurements found.")
    else:
        measurements = []
        grid_data = {
            "timestamp_start": datetime.now().isoformat(),
            "config": {
                "index_type": IDX_CFG["index_type"],
                "parallel_index": IDX_CFG["parallel_index"],
                "io_depth": IDX_CFG["io_depth"],
                "k": K,
                "measurement_sec": MEASUREMENT_SEC,
                "postgres_cores": POSTGRES_CORES,
                "numa_nodes": PG_NUMA_NODES,
                "tau_max": tau_max,
                "alpha_max": alpha_max,
                "X_ANN": saturation["X_ANN"],
                "X_OLAP": saturation["X_OLAP"],
            },
            "measurements": measurements,
        }

    # Build set of already-completed measurement keys
    completed = set()
    for m in measurements:
        completed.add((m["ann_workers"], m["olap_workers"]))

    # Count total measurements needed
    total_new = 0
    if run_fixed_ann:
        for ann_count, olap_list in fixed_ann_lines.items():
            for olap_count in olap_list:
                if (ann_count, olap_count) not in completed:
                    total_new += 1
    if run_fixed_olap:
        for olap_count, ann_list in fixed_olap_lines.items():
            for ann_count in ann_list:
                if (ann_count, olap_count) not in completed:
                    total_new += 1

    print(f"\nGrid graph plan:")
    print(f"  ANN levels (fixed-ANN lines): {list(fixed_ann_lines.keys())}")
    print(f"  OLAP levels (fixed-OLAP lines): {list(fixed_olap_lines.keys())}")
    print(f"  Total new measurements needed: {total_new}")
    print(f"  Estimated time: {total_new * MEASUREMENT_SEC / 60:.0f} minutes "
          f"({total_new * MEASUREMENT_SEC / 3600:.1f} hours)")

    progress = 0

    # Run fixed-ANN lines
    if run_fixed_ann:
        for ann_count in sorted(fixed_ann_lines.keys()):
            olap_list = fixed_ann_lines[ann_count]
            print(f"\n{'='*70}")
            print(f"FIXED-ANN LINE: ANN = {ann_count} workers")
            print(f"  Sweeping OLAP: {olap_list}")
            print(f"{'='*70}")

            for olap_count in olap_list:
                if (ann_count, olap_count) in completed:
                    print(f"  Skipping ({ann_count}, {olap_count}) — already measured")
                    continue

                progress += 1
                print(f"\n  [{progress}/{total_new}] Measuring ({ann_count} ANN, {olap_count} OLAP)...")

                result = measure_throughput(ann_count, olap_count, workload)
                result["line_type"] = "fixed_ann"
                result["fixed_value"] = ann_count

                measurements.append(result)
                completed.add((ann_count, olap_count))

                # Save after each measurement (crash-safe)
                grid_data["measurements"] = measurements
                grid_data["timestamp_last"] = datetime.now().isoformat()
                with open(grid_file, 'w') as f:
                    json.dump(grid_data, f, indent=2)

    # Run fixed-OLAP lines
    if run_fixed_olap:
        for olap_count in sorted(fixed_olap_lines.keys()):
            ann_list = fixed_olap_lines[olap_count]
            print(f"\n{'='*70}")
            print(f"FIXED-OLAP LINE: OLAP = {olap_count} workers")
            print(f"  Sweeping ANN: {ann_list}")
            print(f"{'='*70}")

            for ann_count in ann_list:
                if (ann_count, olap_count) in completed:
                    print(f"  Skipping ({ann_count}, {olap_count}) — already measured")
                    continue

                progress += 1
                print(f"\n  [{progress}/{total_new}] Measuring ({ann_count} ANN, {olap_count} OLAP)...")

                result = measure_throughput(ann_count, olap_count, workload)
                result["line_type"] = "fixed_olap"
                result["fixed_value"] = olap_count

                measurements.append(result)
                completed.add((ann_count, olap_count))

                # Save after each measurement (crash-safe)
                grid_data["measurements"] = measurements
                grid_data["timestamp_last"] = datetime.now().isoformat()
                with open(grid_file, 'w') as f:
                    json.dump(grid_data, f, indent=2)

    print(f"\n\nAll grid measurements complete. Total: {len(measurements)} data points.")
    return grid_data


# ============== FRONTIER EXTRACTION ==============

def extract_frontier(grid_data, saturation):
    """
    Extract the throughput frontier from the grid graph data.

    The frontier is the set of Pareto-optimal points: no other measured point
    has both higher ANN throughput AND higher OLAP throughput.

    Also computes:
    - AUC (area under frontier curve)
    - Bounding box area (X^ANN * X^OLAP)
    - AUC ratio (AUC / bounding box area)
    """
    X_ANN = saturation["X_ANN"]
    X_OLAP = saturation["X_OLAP"]

    # Collect all (ann_throughput, olap_throughput) points
    points = []
    for m in grid_data["measurements"]:
        points.append((m["ann_throughput_qps"], m["olap_throughput_tps"]))

    # Add the saturation endpoints
    points.append((X_ANN, 0.0))  # Pure ANN
    points.append((0.0, X_OLAP))  # Pure OLAP

    # Remove duplicates and sort by ANN throughput descending
    points = list(set(points))
    points.sort(key=lambda p: p[0], reverse=True)

    # Extract Pareto frontier (non-dominated points)
    # A point (x, y) is on the frontier if no other point has both x' >= x AND y' >= y
    frontier = []
    max_olap_seen = -1

    for ann_tp, olap_tp in points:
        if olap_tp > max_olap_seen:
            frontier.append((ann_tp, olap_tp))
            max_olap_seen = olap_tp

    # Sort frontier by ANN throughput ascending (for plotting/AUC)
    frontier.sort(key=lambda p: p[0])

    # Compute AUC using trapezoidal rule
    # The frontier goes from (0, X^OLAP) to (X^ANN, 0)
    auc = 0.0
    for i in range(len(frontier) - 1):
        x1, y1 = frontier[i]
        x2, y2 = frontier[i + 1]
        # Trapezoidal area
        auc += (x2 - x1) * (y1 + y2) / 2.0

    # Bounding box area
    bounding_box_area = X_ANN * X_OLAP

    # AUC ratio (1.0 = perfect isolation, i.e., frontier is the bounding box boundary)
    auc_ratio = auc / bounding_box_area if bounding_box_area > 0 else 0

    # Proportional line: connects (X^ANN, 0) to (0, X^OLAP)
    # Points on this line satisfy: ann/X^ANN + olap/X^OLAP = 1
    # Area under proportional line = 0.5 * X^ANN * X^OLAP (triangle)
    proportional_area = 0.5 * bounding_box_area
    above_proportional = auc > proportional_area

    frontier_result = {
        "timestamp": datetime.now().isoformat(),
        "saturation_points": {
            "X_ANN": X_ANN,
            "X_OLAP": X_OLAP,
            "tau_max": saturation["tau_max"],
            "alpha_max": saturation["alpha_max"],
        },
        "frontier_points": [{"ann_throughput_qps": p[0], "olap_throughput_tps": p[1]} for p in frontier],
        "metrics": {
            "auc": round(auc, 4),
            "bounding_box_area": round(bounding_box_area, 4),
            "auc_ratio": round(auc_ratio, 4),
            "proportional_line_area": round(proportional_area, 4),
            "above_proportional_line": above_proportional,
            "num_frontier_points": len(frontier),
            "num_total_measurements": len(grid_data["measurements"]),
        },
        "interpretation": {
            "auc_ratio_meaning": (
                "AUC/BoundingBox ratio: 1.0 = perfect isolation (frontier at bounding box), "
                "0.5 = proportional line (linear tradeoff), "
                "<0.5 = negative interference (below proportional line)"
            ),
            "system_classification": (
                "above_proportional" if above_proportional else "below_proportional"
            ),
        },
        "all_points": [
            {"ann_workers": m["ann_workers"], "olap_workers": m["olap_workers"],
             "ann_throughput_qps": m["ann_throughput_qps"], "olap_throughput_tps": m["olap_throughput_tps"]}
            for m in grid_data["measurements"]
        ],
    }

    return frontier_result


# ============== MAIN ==============

def main():
    global MEMORY_CONFIG, RESULTS_DIR, TUNE_CFG, WARM_MODE, GRID_FRACTIONS, INDEX_TYPE_OVERRIDE, INDEX_PARAMS

    run_fixed_ann = True
    run_fixed_olap = True
    skip_measurements = False
    run_extra = False
    single_point = None

    if "--skip-measurements" in sys.argv:
        skip_measurements = True
    if "--fixed-ann-only" in sys.argv:
        run_fixed_olap = False
    if "--fixed-olap-only" in sys.argv:
        run_fixed_ann = False
    if "--extra" in sys.argv:
        run_extra = True
        skip_measurements = True  # Don't run grid, only extra points
    if "--single" in sys.argv:
        idx = sys.argv.index("--single")
        ann_w, olap_w = sys.argv[idx + 1].split(",")
        single_point = (int(ann_w), int(olap_w))

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

    # Parse --dense flag
    if "--dense" in sys.argv:
        GRID_FRACTIONS = DENSE_GRID_FRACTIONS
        print(f"DENSE MODE: Using {len(GRID_FRACTIONS)} grid fractions (~280 points)")
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

    # Parse --snapshot flag (skip ensure_index, index loaded from snapshot)
    if "--snapshot" in sys.argv:
        SNAPSHOT_MODE = True
        print("SNAPSHOT MODE: Skipping ensure_index (index already in PGDATA snapshot)")

    # Modify results dir to include index name (only for non-default indexes)
    if INDEX_TYPE_OVERRIDE and INDEX_TYPE_OVERRIDE != "ivfflat":
        idx_short = INDEX_TYPE_OVERRIDE.replace("parallel_ivfflat", "pivfflat")
        RESULTS_DIR = RESULTS_DIR.replace("results_frontier_", f"results_frontier_{idx_short}_")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"INDEX: {INDEX_TYPE_OVERRIDE.upper()} (results: {RESULTS_DIR})")
        if INDEX_PARAMS:
            print(f"  Manual params: {INDEX_PARAMS}")

        print("=" * 70)
    print("THROUGHPUT FRONTIER - Step 2: Build Grid Graph & Extract Frontier")
    print(f"Index: {IDX_CFG['index_type']}, Parallel: {IDX_CFG['parallel_index']}, "
          f"io_depth: {IDX_CFG['io_depth']}, K: {K}")
    print(f"Measurement: {MEASUREMENT_SEC}s per data point ({"warm" if WARM_MODE else "cold start"})")
    print(f"Grid fractions: {GRID_FRACTIONS}")
    print(f"Postgres cores: {POSTGRES_CORES}, NUMA: {PG_NUMA_NODES}")
    print(f"Client cores (TopK): {TOPK_CORES[:4]}...")
    print(f"Client cores (OLAP): {OLAP_CORES[:4]}...")
    print("=" * 70)

    # Load saturation points from Step 1
    saturation = load_saturation_points()

    # Single-point measurement mode (quick test)
    if single_point:
        ann_w, olap_w = single_point
        print(f"\n{'='*70}")
        print(f"SINGLE POINT MODE: Measuring ({ann_w} ANN, {olap_w} OLAP)")
        print(f"{'='*70}")

        # Load workload
        workload = load_workload()

        # Measure
        result = measure_throughput(ann_w, olap_w, workload, measurement_sec=MEASUREMENT_SEC)
        print(f"\n{'='*70}")
        print(f"RESULT: ANN={result['ann_throughput_qps']:.2f} qps, "
              f"OLAP={result['olap_throughput_qps']:.4f} qps")
        print(f"{'='*70}")
        import json as _json
        print(_json.dumps(result, indent=2))
        return

    if run_extra:
        # Run extra fill-in points to refine frontier
        EXTRA_POINTS = [
            (1, 10), (2, 10), (3, 10),
            (1, 8), (2, 8), (3, 8),
        ]

        # Ensure correct index exists (from tune config)
        if SNAPSHOT_MODE:
            print("\nSnapshot mode: skipping index creation (already in PGDATA).")
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

        grid_file = os.path.join(RESULTS_DIR, "grid_graph.json")
        with open(grid_file, 'r') as f:
            grid_data = json.load(f)
        completed = set((m["ann_workers"], m["olap_workers"]) for m in grid_data["measurements"])


        print(f"\nRunning extra measurements to fill frontier gaps...")
        for ann_count, olap_count in EXTRA_POINTS:
            if (ann_count, olap_count) in completed:
                print(f"  Skipping ({ann_count}, {olap_count}) - already measured")
                continue
            result = measure_throughput(ann_count, olap_count, workload)
            result["line_type"] = "extra_fill"
            result["fixed_value"] = 0
            grid_data["measurements"].append(result)
            completed.add((ann_count, olap_count))
            grid_data["timestamp_last"] = datetime.now().isoformat()
            with open(grid_file, 'w') as f:
                json.dump(grid_data, f, indent=2)

        print(f"Extra measurements done. Total: {len(grid_data['measurements'])} points.")

    elif not skip_measurements:
        # Ensure correct index exists (from tune config)
        if SNAPSHOT_MODE:
            print("\nSnapshot mode: skipping index creation (already in PGDATA).")
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

        # Run grid measurements
        grid_data = run_grid_measurements(workload, saturation, run_fixed_ann, run_fixed_olap)

    else:
        # Load existing grid data
        grid_file = os.path.join(RESULTS_DIR, "grid_graph.json")
        if not os.path.exists(grid_file):
            print(f"ERROR: {grid_file} not found. Run measurements first.")
            sys.exit(1)
        with open(grid_file, 'r') as f:
            grid_data = json.load(f)
        print(f"Loaded {len(grid_data['measurements'])} existing measurements.")

    # Extract frontier and compute metrics
    print("\n" + "=" * 70)
    print("EXTRACTING THROUGHPUT FRONTIER")
    print("=" * 70)

    frontier_result = extract_frontier(grid_data, saturation)

    # Save frontier
    frontier_file = os.path.join(RESULTS_DIR, "throughput_frontier.json")
    with open(frontier_file, 'w') as f:
        json.dump(frontier_result, f, indent=2)

    # Print summary
    metrics = frontier_result["metrics"]
    sat = frontier_result["saturation_points"]

    print(f"\n{'='*70}")
    print("THROUGHPUT FRONTIER SUMMARY")
    print(f"{'='*70}")
    print(f"  X^ANN  = {sat['X_ANN']:.2f} qps (tau_max = {sat['tau_max']} workers)")
    print(f"  X^OLAP = {sat['X_OLAP']:.4f} tps (alpha_max = {sat['alpha_max']} workers)")
    print(f"  ")
    print(f"  Bounding box area:      {metrics['bounding_box_area']:.4f}")
    print(f"  Proportional line area: {metrics['proportional_line_area']:.4f}")
    print(f"  Frontier AUC:           {metrics['auc']:.4f}")
    print(f"  AUC ratio:              {metrics['auc_ratio']:.4f}")
    print(f"  Above proportional:     {metrics['above_proportional_line']}")
    print(f"  Frontier points:        {metrics['num_frontier_points']}")
    print(f"  Total measurements:     {metrics['num_total_measurements']}")
    print(f"{'='*70}")

    if metrics['auc_ratio'] > 0.7:
        print("  => GOOD isolation: Frontier close to bounding box.")
        print("     The system can serve both workloads with minimal interference.")
    elif metrics['auc_ratio'] > 0.5:
        print("  => MODERATE isolation: Frontier above proportional line.")
        print("     Some interference, but workloads can partially coexist.")
    elif metrics['auc_ratio'] > 0.3:
        print("  => LINEAR tradeoff: Frontier near proportional line.")
        print("     Increasing one workload proportionally decreases the other.")
    else:
        print("  => POOR isolation: Frontier below proportional line.")
        print("     Significant negative interference between workloads.")

    print(f"\nResults saved to:")
    print(f"  Grid graph: {os.path.join(RESULTS_DIR, 'grid_graph.json')}")
    print(f"  Frontier:   {frontier_file}")


if __name__ == "__main__":
    main()
