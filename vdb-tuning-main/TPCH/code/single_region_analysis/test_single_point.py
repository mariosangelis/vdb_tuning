#!/usr/bin/env python3
"""
Self-contained measurement tool with full setup:
  1. Compiles the appropriate pgvector version (default or parallel)
  2. Loads the requested snapshot (hnsw / ivfflat / parallel_ivfflat)
  3. Sets shared_buffers + cgroup memory limit
  4. Starts postgres with NUMA pinning
  5. Creates TPC-H secondary indexes
  6. Prewarms all relations
  7. Runs all ANN x OLAP worker combinations and reports throughput

Usage:
    sudo python3 test_single_point.py --index hnsw --memory 18:25 --ann 1,2 --olap 10,20 [--duration 120]
    sudo python3 test_single_point.py --index parallel_ivfflat --memory 20:30 --ann 1,2,4 --olap 5,10,20 --duration 300

Required flags:
    --index     hnsw | ivfflat | parallel_ivfflat
    --memory    SB:CG (shared_buffers in GB : cgroup in GB)
    --ann       comma-separated list of ANN worker counts
    --olap      comma-separated list of OLAP worker counts

Optional:
    --duration  seconds per combination (default: 120)
    --no-setup  skip snapshot loading/compile/restart (just measure against running postgres)
"""
import sys, os, time, threading, random, json, re, subprocess, shutil

# ========== PATHS ==========
PG_DIR = "/users/mangelis/vdb_tuning/postgres/postgres"
PGVECTOR_DEFAULT = os.path.join(PG_DIR, "pgvector")
PGVECTOR_PARALLEL = os.path.join(PG_DIR, "pgvector_modified_final")
SNAPSHOTS_DIR = "/mnt/nvme/saved_indexes"
PGDATA_DIR = "/mnt/nvme/pgdata"
PG_CTL_PATH = shutil.which("pg_ctl") or "/usr/local/pgsql/bin/pg_ctl"
PG_CONF_PATH = os.path.join(PGDATA_DIR, "postgresql.conf")
CGROUP_MEM_PATH = "/sys/fs/cgroup/postgres_test/memory.max"
WORKLOAD_FILE = "/users/mangelis/vdb_tuning/vdb-tuning-main/gist_test_queries.txt"

SNAPSHOT_MAP = {
    "ivfflat": os.path.join(SNAPSHOTS_DIR, "gist_ivfflat_lists8192_pgdata.tar.gz"),
    "hnsw": os.path.join(SNAPSHOTS_DIR, "gist_hnsw_efc128_m16_pgdata.tar.gz"),
    "parallel_ivfflat": os.path.join(SNAPSHOTS_DIR, "gist_parallel_ivfflat_lists8192_pgdata.tar.gz"),
}

DB_HOST = "/tmp"
DB_NAME = "mydb"
DB_USER = "postgres"
K = 100
MEASUREMENT_SEC = 120
PG_NUMA_NODE = "0"
PRODUCER_CORES = list(range(32, 64))  # NUMA node 1 cores for ANN/OLAP producers
USE_NUMA = True  # Set to False with --no-numa flag

# ========== TPC-H QUERIES ==========
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

# Q15 requires creating/dropping the revenue0 view
Q15_CREATE_VIEW = "CREATE OR REPLACE VIEW revenue0 (supplier_no, total_revenue) AS SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount)) FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH GROUP BY l_suppkey"
Q15_DROP_VIEW = "DROP VIEW IF EXISTS revenue0"

# Build executable query list (skip Q15_create)
EXECUTABLE_QUERIES = [(name, sql) for name, sql in TPCH_QUERIES if name != "Q15_create"]

# TPC-H secondary indexes (same as run_all_frontiers.sh)
TPCH_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS l_ok ON lineitem (l_orderkey ASC);
CREATE INDEX IF NOT EXISTS l_pk ON lineitem (l_partkey ASC);
CREATE INDEX IF NOT EXISTS l_sk ON lineitem (l_suppkey ASC);
CREATE INDEX IF NOT EXISTS l_sd ON lineitem (l_shipdate ASC);
CREATE INDEX IF NOT EXISTS l_cd ON lineitem (l_commitdate ASC);
CREATE INDEX IF NOT EXISTS l_rd ON lineitem (l_receiptdate ASC);
CREATE INDEX IF NOT EXISTS l_pk_sk ON lineitem (l_partkey, l_suppkey);
CREATE INDEX IF NOT EXISTS l_sk_pk ON lineitem (l_suppkey, l_partkey);
CREATE INDEX IF NOT EXISTS ps_pk ON partsupp (ps_partkey ASC);
CREATE INDEX IF NOT EXISTS ps_sk ON partsupp (ps_suppkey ASC);
CREATE INDEX IF NOT EXISTS o_ck ON orders (o_custkey ASC);
CREATE INDEX IF NOT EXISTS o_od ON orders (o_orderdate ASC);
CREATE INDEX IF NOT EXISTS c_nk ON customer (c_nationkey ASC);
CREATE INDEX IF NOT EXISTS s_nk ON supplier (s_nationkey ASC);
CREATE INDEX IF NOT EXISTS n_rk ON nation (n_regionkey ASC);
ANALYZE;
"""


# ========== SETUP FUNCTIONS ==========

def run_cmd(cmd, check=True):
    """Run a shell command, print it, return output."""
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"    FAILED (rc={r.returncode}): {r.stderr[:500]}")
        if check:
            sys.exit(1)
    return r


def compile_pgvector(index_type):
    """Compile the right pgvector version."""
    if index_type == "parallel_ivfflat":
        pgvector_dir = PGVECTOR_PARALLEL
        label = "parallel pgvector"
    else:
        pgvector_dir = PGVECTOR_DEFAULT
        label = "default pgvector"

    print(f"\n[SETUP] Compiling {label}...")
    env = "export PATH=/usr/local/pgsql/bin:$PATH"
    run_cmd(f"cd {pgvector_dir} && {env} && make clean > /dev/null 2>&1 || true", check=False)
    run_cmd(f"cd {pgvector_dir} && {env} && make -j$(nproc) > /dev/null 2>&1")
    run_cmd(f"cd {pgvector_dir} && {env} && make install > /dev/null 2>&1")
    print(f"  Done.")


def stop_postgres():
    """Stop postgres if running."""
    print("\n[SETUP] Stopping postgres...")
    run_cmd(
        f"su - postgres -c 'export PATH=/usr/local/pgsql/bin:$PATH && "
        f"pg_ctl -D {PGDATA_DIR} stop -m fast' 2>/dev/null || true",
        check=False
    )
    time.sleep(2)


def load_snapshot(index_type):
    """Load snapshot into PGDATA."""
    snap_path = SNAPSHOT_MAP[index_type]
    if not os.path.exists(snap_path):
        print(f"ERROR: Snapshot not found: {snap_path}")
        sys.exit(1)

    print(f"\n[SETUP] Loading snapshot: {index_type}")
    print(f"  Source: {snap_path}")

    # Clear PGDATA
    run_cmd(f"rm -rf {PGDATA_DIR}/*")
    run_cmd(f"mkdir -p {PGDATA_DIR}")

    # Extract (use pigz if available)
    if shutil.which("pigz"):
        run_cmd(f"pigz -dc '{snap_path}' | tar -xf - -C {PGDATA_DIR}")
    else:
        run_cmd(f"tar -xzf '{snap_path}' -C {PGDATA_DIR}")

    # Fix ownership
    run_cmd(f"chown -R postgres:postgres {PGDATA_DIR}")
    run_cmd(f"chmod 700 {PGDATA_DIR}")
    print(f"  Snapshot loaded.")


def set_shared_buffers(size_gb):
    """Modify shared_buffers in postgresql.conf."""
    print(f"\n[SETUP] Setting shared_buffers to {size_gb}GB...")
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
    print(f"\n[SETUP] Setting cgroup memory to {size_gb}GB...")
    size_bytes = size_gb * 1024 * 1024 * 1024

    with open(CGROUP_MEM_PATH, 'w') as f:
        f.write(str(size_bytes))

    # Verify
    with open(CGROUP_MEM_PATH, 'r') as f:
        actual = int(f.read().strip())
    print(f"  cgroup memory set to {actual / (1024**3):.1f}GB")


def move_postgres_to_cgroup():
    """Move all postgres processes into the cgroup."""
    print("  Moving postgres processes into cgroup...")
    subprocess.run(
        "for pid in $(pgrep -u postgres); do echo $pid > /sys/fs/cgroup/postgres_test/cgroup.procs; done",
        shell=True, check=False
    )


def start_postgres():
    """Start postgres with optional NUMA pinning."""
    # Drop caches first
    run_cmd("sync; echo 3 > /proc/sys/vm/drop_caches")

    if USE_NUMA:
        print(f"\n[SETUP] Starting PostgreSQL (NUMA node {PG_NUMA_NODE})...")
        run_cmd(
            f"su - postgres -c 'export PATH=/usr/local/pgsql/bin:$PATH && "
            f"numactl --cpunodebind={PG_NUMA_NODE} --membind={PG_NUMA_NODE} "
            f"pg_ctl -D {PGDATA_DIR} -l /tmp/logfile start'"
        )
    else:
        print(f"\n[SETUP] Starting PostgreSQL (no NUMA pinning, all cores)...")
        run_cmd(
            f"su - postgres -c 'export PATH=/usr/local/pgsql/bin:$PATH && "
            f"pg_ctl -D {PGDATA_DIR} -l /tmp/logfile start'"
        )
    time.sleep(3)

    # Wait for postgres ready
    for i in range(30):
        r = subprocess.run(
            "su - postgres -c 'export PATH=/usr/local/pgsql/bin:$PATH && pg_isready'",
            shell=True, capture_output=True
        )
        if r.returncode == 0:
            print("  PostgreSQL is ready.")
            return
        time.sleep(2)
    print("ERROR: PostgreSQL did not start in time!")
    sys.exit(1)


def create_tpch_indexes():
    """Create TPC-H secondary indexes."""
    print("\n[SETUP] Creating TPC-H secondary indexes...")
    import psycopg2
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER)
    conn.autocommit = True
    cur = conn.cursor()
    for stmt in TPCH_INDEXES_SQL.strip().split(';'):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt + ';')
            except Exception as e:
                print(f"    Warning: {e}")
    cur.close()
    conn.close()
    print("  Done.")


def prewarm_database():
    """Prewarm ALL tables and indexes into shared buffers."""
    print("\n[SETUP] Prewarming ALL relations into shared buffers...")
    import psycopg2
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER)
    conn.autocommit = True
    cur = conn.cursor()

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


# ========== WORKLOAD ==========

def load_workload_vectors():
    with open(WORKLOAD_FILE, 'r') as f:
        data = json.loads(f.read().strip())
    if isinstance(data[0], list):
        vectors = data
    else:
        vectors = [data]
    print(f"  Loaded {len(vectors)} workload vectors")
    return vectors


# ========== MEASUREMENT WORKERS ==========

def ann_worker(worker_id, workload, stop_event, lock, results):
    import psycopg2
    # Pin to NUMA node 1 (skip if --no-numa)
    if USE_NUMA:
        core_id = PRODUCER_CORES[worker_id % len(PRODUCER_CORES)]
        try:
            os.sched_setaffinity(0, {core_id})
        except (OSError, AttributeError):
            pass
    try:
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET ivfflat.probes = 150;")
        #cur.execute("SET max_parallel_workers_per_gather = 200;")
    except Exception as e:
        print(f"  ANN worker {worker_id} connect failed: {e}")
        return

    local_random = random.Random(worker_id)
    while not stop_event.is_set():
        vec = local_random.choice(workload)
        vec_str = "[" + ",".join(str(x) for x in vec) + "]"
        query = f"SELECT id FROM gist_data ORDER BY embedding <-> '{vec_str}'::vector LIMIT {K};"
        try:
            t0 = time.time()
            cur.execute(query)
            cur.fetchall()
            lat = time.time() - t0
            with lock:
                results['ann_count'] += 1
                results['ann_latencies'].append(lat)
        except:
            if not stop_event.is_set():
                time.sleep(0.1)
    conn.close()


def olap_worker(worker_id, stop_event, lock, results):
    import psycopg2
    # Pin to NUMA node 1 (skip if --no-numa)
    if USE_NUMA:
        core_id = PRODUCER_CORES[worker_id % len(PRODUCER_CORES)]
        try:
            os.sched_setaffinity(0, {core_id})
        except (OSError, AttributeError):
            pass
    try:
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET enable_seqscan = on;")
        cur.execute("SET work_mem = '100MB';")
        cur.execute("SET statement_timeout = '300s';")
    except Exception as e:
        print(f"  OLAP worker {worker_id} connect failed: {e}")
        return

    local_random = random.Random(worker_id + 1000)
    query_order = list(range(len(EXECUTABLE_QUERIES)))
    local_random.shuffle(query_order)
    qi = 0

    while not stop_event.is_set():
        idx = query_order[qi % len(query_order)]
        qi += 1
        qname, sql = EXECUTABLE_QUERIES[idx]
        try:
            if qname == "Q15":
                cur.execute(Q15_CREATE_VIEW)
                cur.execute(sql)
                cur.fetchall()
                cur.execute(Q15_DROP_VIEW)
            else:
                cur.execute(sql)
                cur.fetchall()
            with lock:
                results['olap_query_count'] += 1
        except:
            try:
                cur.execute(Q15_DROP_VIEW)
            except:
                pass
            if not stop_event.is_set():
                time.sleep(0.1)
    conn.close()


# ========== MEASURE ==========

def measure(num_ann, num_olap, workload, duration):
    print(f"\n  Measuring: {num_ann} ANN, {num_olap} OLAP ({duration}s)...")

    stop_event = threading.Event()
    lock = threading.Lock()
    results = {'ann_count': 0, 'ann_latencies': [], 'olap_query_count': 0}
    threads = []

    for i in range(num_ann):
        t = threading.Thread(target=ann_worker, args=(i, workload, stop_event, lock, results))
        t.start()
        threads.append(t)
    for i in range(num_olap):
        t = threading.Thread(target=olap_worker, args=(i, stop_event, lock, results))
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=60)

    ann_qps = results['ann_count'] / duration
    olap_qps = results['olap_query_count'] / duration
    ann_lats = results['ann_latencies']
    avg_lat = (sum(ann_lats) / len(ann_lats) * 1000) if ann_lats else 0

    print(f"  => ANN={ann_qps:.2f} qps (lat={avg_lat:.1f}ms), OLAP={olap_qps:.4f} qps ({results['olap_query_count']} queries)")
    return {'ann_w': num_ann, 'olap_w': num_olap, 'ann_qps': round(ann_qps, 2),
            'olap_qps': round(olap_qps, 4), 'ann_lat_ms': round(avg_lat, 1),
            'ann_total': results['ann_count'], 'olap_total': results['olap_query_count']}


# ========== MAIN ==========

if __name__ == '__main__':
    import psycopg2

    # Parse arguments
    index_type = None
    sb_gb = None
    cg_gb = None
    ann_list = [1, 2]
    olap_list = [10, 20]
    duration = MEASUREMENT_SEC
    no_setup = False

    if "--index" in sys.argv:
        idx = sys.argv.index("--index")
        index_type = sys.argv[idx + 1]
    if "--memory" in sys.argv:
        idx = sys.argv.index("--memory")
        parts = sys.argv[idx + 1].split(":")
        sb_gb = int(parts[0])
        cg_gb = int(parts[1])
    if "--ann" in sys.argv:
        idx = sys.argv.index("--ann")
        ann_list = [int(x) for x in sys.argv[idx + 1].split(",")]
    if "--olap" in sys.argv:
        idx = sys.argv.index("--olap")
        olap_list = [int(x) for x in sys.argv[idx + 1].split(",")]
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        duration = int(sys.argv[idx + 1])
    if "--no-setup" in sys.argv:
        no_setup = True
    if "--no-numa" in sys.argv:
        USE_NUMA = False

    # Validate
    if not no_setup:
        if index_type is None:
            print("ERROR: --index required (hnsw | ivfflat | parallel_ivfflat)")
            sys.exit(1)
        if index_type not in SNAPSHOT_MAP:
            print(f"ERROR: Unknown index type '{index_type}'. Use: hnsw, ivfflat, parallel_ivfflat")
            sys.exit(1)
        if sb_gb is None or cg_gb is None:
            print("ERROR: --memory SB:CG required (e.g., --memory 18:25)")
            sys.exit(1)

    print("=" * 70)
    print("MEASUREMENT TOOL (with full setup)")
    print("=" * 70)
    if not no_setup:
        print(f"  Index type:   {index_type}")
        print(f"  Memory:       shared_buffers={sb_gb}GB, cgroup={cg_gb}GB")
    else:
        print(f"  Mode:         --no-setup (measuring against running postgres)")
    print(f"  ANN workers:  {ann_list}")
    print(f"  OLAP workers: {olap_list}")
    print(f"  Duration:     {duration}s per combo")
    print(f"  Combos:       {len(ann_list) * len(olap_list)}")
    print(f"  Total time:   ~{len(ann_list) * len(olap_list) * duration / 60:.0f} min")
    print(f"  OLAP queries: {len(EXECUTABLE_QUERIES)}")
    print("=" * 70)

    # ===== SETUP PHASE =====
    if not no_setup:
        # 1. Compile pgvector
        compile_pgvector(index_type)

        # 2. Stop postgres, load snapshot
        stop_postgres()
        load_snapshot(index_type)

        # 3. Set shared_buffers in postgresql.conf
        set_shared_buffers(sb_gb)

        # 4. Set cgroup memory limit
        set_cgroup_memory(cg_gb)

        # 5. Start postgres
        start_postgres()

        # 6. Move to cgroup
        move_postgres_to_cgroup()

        # 7. Create TPC-H indexes
        create_tpch_indexes()

        # 8. Prewarm
        prewarm_database()

        print("\n" + "=" * 70)
        print("SETUP COMPLETE - Starting measurements")
        print("=" * 70)

    # ===== MEASUREMENT PHASE =====
    print("\nLoading workload vectors...")
    workload = load_workload_vectors()

    all_results = []
    for ann in ann_list:
        for olap in olap_list:
            r = measure(ann, olap, workload, duration)
            all_results.append(r)

            # Save incrementally after each measurement
            output = {
                "index_type": index_type,
                "shared_buffers_gb": sb_gb,
                "cgroup_gb": cg_gb,
                "duration_sec": duration,
                "results": all_results
            }
            numa_suffix = "" if USE_NUMA else "_nonuma"
            outfile = f"/tmp/measurement_{index_type}_{sb_gb}sb_{cg_gb}cg{numa_suffix}.json"
            with open(outfile, "w") as f:
                json.dump(output, f, indent=2)

    # ===== SUMMARY =====
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    if not no_setup:
        print(f"  Index: {index_type}, SB={sb_gb}GB, CG={cg_gb}GB")
    print("=" * 70)
    print(f"{'ANN_w':>5} {'OLAP_w':>6} {'ANN qps':>9} {'OLAP qps':>10} {'ANN lat(ms)':>12}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['ann_w']:>5} {r['olap_w']:>6} {r['ann_qps']:>9.2f} {r['olap_qps']:>10.4f} {r['ann_lat_ms']:>12.1f}")
    print("=" * 70)

    # Save results to JSON
    output = {
        'index_type': index_type,
        'shared_buffers_gb': sb_gb,
        'cgroup_gb': cg_gb,
        'duration_sec': duration,
        'results': all_results
    }
    numa_suffix = "" if USE_NUMA else "_nonuma"
    outfile = f"/tmp/measurement_{index_type}_{sb_gb}sb_{cg_gb}cg{numa_suffix}.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {outfile}")
