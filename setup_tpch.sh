#!/bin/bash
# Generate and load TPC-H data (scale factor from config).
# Usage: sudo ./setup_tpch.sh   (run AFTER setup_postgres.sh)
# Machine-specific paths/flags come from setup_config.sh.
set -e

# ---- Load central config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SETUP_CONFIG:-${SCRIPT_DIR}/setup_config.sh}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    echo "       Create it (see setup_config.sh) or set SETUP_CONFIG=/path/to/setup_config.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
echo "Loaded config: ${CONFIG_FILE}"

PGBIN=${PG_INSTALL}/bin
DBGEN_PARENT="$(dirname "${DBGEN_DIR}")"

echo "========================================"
echo "TPC-H SF${TPCH_SCALE} setup"
echo "  DBGEN_DIR = ${DBGEN_DIR}"
echo "  DB_NAME   = ${DB_NAME}"
echo "========================================"

echo "[1/5] Cloning and compiling dbgen..."
mkdir -p "${DBGEN_PARENT}"
cd "${DBGEN_PARENT}"
if [ ! -d "${DBGEN_DIR}" ]; then
    git clone https://github.com/electrum/tpch-dbgen.git "$(basename "${DBGEN_DIR}")"
fi
cd "${DBGEN_DIR}"
make clean > /dev/null 2>&1 || true
make -j"$(nproc)"
echo "  Done."

echo "[2/5] Generating SF${TPCH_SCALE} data..."
./dbgen -s "${TPCH_SCALE}" -f
# Remove trailing delimiter that dbgen adds
sed -i 's/|$//' *.tbl
echo "  Generated:"
ls -lh *.tbl

echo "[3/5] Creating TPC-H tables and loading data..."
chmod 644 "${DBGEN_DIR}"/*.tbl

cat > /tmp/create_tpch.sql << 'SQLEOF'
CREATE TABLE IF NOT EXISTS region (r_regionkey INTEGER PRIMARY KEY, r_name CHAR(25) NOT NULL, r_comment VARCHAR(152));
CREATE TABLE IF NOT EXISTS nation (n_nationkey INTEGER PRIMARY KEY, n_name CHAR(25) NOT NULL, n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey), n_comment VARCHAR(152));
CREATE TABLE IF NOT EXISTS supplier (s_suppkey INTEGER PRIMARY KEY, s_name CHAR(25) NOT NULL, s_address VARCHAR(40) NOT NULL, s_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey), s_phone CHAR(15) NOT NULL, s_acctbal DECIMAL(15,2) NOT NULL, s_comment VARCHAR(101));
CREATE TABLE IF NOT EXISTS part (p_partkey INTEGER PRIMARY KEY, p_name VARCHAR(55) NOT NULL, p_mfgr CHAR(25) NOT NULL, p_brand CHAR(10) NOT NULL, p_type VARCHAR(25) NOT NULL, p_size INTEGER NOT NULL, p_container CHAR(10) NOT NULL, p_retailprice DECIMAL(15,2) NOT NULL, p_comment VARCHAR(23));
CREATE TABLE IF NOT EXISTS partsupp (ps_partkey INTEGER NOT NULL REFERENCES part(p_partkey), ps_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey), ps_availqty INTEGER NOT NULL, ps_supplycost DECIMAL(15,2) NOT NULL, ps_comment VARCHAR(199), PRIMARY KEY (ps_partkey, ps_suppkey));
CREATE TABLE IF NOT EXISTS customer (c_custkey INTEGER PRIMARY KEY, c_name VARCHAR(25) NOT NULL, c_address VARCHAR(40) NOT NULL, c_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey), c_phone CHAR(15) NOT NULL, c_acctbal DECIMAL(15,2) NOT NULL, c_mktsegment CHAR(10) NOT NULL, c_comment VARCHAR(117));
CREATE TABLE IF NOT EXISTS orders (o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER NOT NULL REFERENCES customer(c_custkey), o_orderstatus CHAR(1) NOT NULL, o_totalprice DECIMAL(15,2) NOT NULL, o_orderdate DATE NOT NULL, o_orderpriority CHAR(15) NOT NULL, o_clerk CHAR(15) NOT NULL, o_shippriority INTEGER NOT NULL, o_comment VARCHAR(79));
CREATE TABLE IF NOT EXISTS lineitem (l_orderkey INTEGER NOT NULL REFERENCES orders(o_orderkey), l_partkey INTEGER NOT NULL REFERENCES part(p_partkey), l_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey), l_linenumber INTEGER NOT NULL, l_quantity DECIMAL(15,2) NOT NULL, l_extendedprice DECIMAL(15,2) NOT NULL, l_discount DECIMAL(15,2) NOT NULL, l_tax DECIMAL(15,2) NOT NULL, l_returnflag CHAR(1) NOT NULL, l_linestatus CHAR(1) NOT NULL, l_shipdate DATE NOT NULL, l_commitdate DATE NOT NULL, l_receiptdate DATE NOT NULL, l_shipinstruct CHAR(25) NOT NULL, l_shipmode CHAR(10) NOT NULL, l_comment VARCHAR(44), PRIMARY KEY (l_orderkey, l_linenumber));
SQLEOF

su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -f /tmp/create_tpch.sql"
echo "  Tables created."

echo "[4/5] Loading data with COPY..."
cat > /tmp/load_tpch.sql << SQLEOF
\copy region FROM '${DBGEN_DIR}/region.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy nation FROM '${DBGEN_DIR}/nation.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy supplier FROM '${DBGEN_DIR}/supplier.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy part FROM '${DBGEN_DIR}/part.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy partsupp FROM '${DBGEN_DIR}/partsupp.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy customer FROM '${DBGEN_DIR}/customer.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy orders FROM '${DBGEN_DIR}/orders.tbl' WITH (DELIMITER '|', FORMAT csv);
\copy lineitem FROM '${DBGEN_DIR}/lineitem.tbl' WITH (DELIMITER '|', FORMAT csv);
ANALYZE;
SQLEOF

su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -f /tmp/load_tpch.sql"

echo "[5/5] Creating TPC-H secondary indexes (benchbase-style)..."
cat > /tmp/index_tpch.sql << 'SQLEOF'
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
SQLEOF

su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -f /tmp/index_tpch.sql"
echo "  Secondary indexes created and ANALYZE complete."

echo ""
echo "Verifying:"
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -c \"SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size FROM pg_tables WHERE schemaname='public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;\""

echo ""
echo "TPC-H SF${TPCH_SCALE} loaded successfully!"
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -t -c 'SELECT count(*) FROM lineitem;'"
echo "lineitem rows above"
