#!/usr/bin/env bash
# Generate TPC-H (scale from config) with dbgen, create tables, load, index.
# Run AFTER build-postgres.sh. Idempotent: skips loading if lineitem already
# has the expected row count (use --force to reload). Architecture-agnostic.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -E bash "$0" "$@"
fi
export PATH="${PGBIN}:${PATH}"

log "======================================================"
log "TPC-H SF${TPCH_SCALE} setup   (dbgen=${DBGEN_DIR})"
log "======================================================"

# Skip if already loaded.
if [ "${FORCE}" -eq 0 ]; then
    existing="$(su - postgres -c "export PATH=${PGBIN}:\$PATH && psql -tAqc \"SELECT to_regclass('public.lineitem') IS NOT NULL AND (SELECT count(*) FROM lineitem) > 0;\" ${DB_NAME}" 2>/dev/null || echo f)"
    if [ "${existing}" = "t" ]; then
        cnt="$(su - postgres -c "export PATH=${PGBIN}:\$PATH && psql -tAqc 'SELECT count(*) FROM lineitem;' ${DB_NAME}")"
        log "lineitem already loaded (${cnt} rows). Skipping (use --force to reload)."
        exit 0
    fi
fi

# ---- 1. dbgen ----
DBGEN_PARENT="$(dirname "${DBGEN_DIR}")"
log "[1/5] Building dbgen..."
mkdir -p "${DBGEN_PARENT}"
if [ ! -d "${DBGEN_DIR}" ]; then
    git clone https://github.com/electrum/tpch-dbgen.git "${DBGEN_DIR}"
fi
cd "${DBGEN_DIR}"
make clean >/dev/null 2>&1 || true
make -j"$(nproc)"

# ---- 2. generate ----
log "[2/5] Generating SF${TPCH_SCALE} .tbl files..."
rm -f ./*.tbl
./dbgen -s "${TPCH_SCALE}" -f
sed -i 's/|$//' ./*.tbl          # strip dbgen trailing delimiter
chmod 644 ./*.tbl
ls -lh ./*.tbl

# ---- 3. tables ----
log "[3/5] Creating tables..."
cat > /tmp/create_tpch.sql <<'SQLEOF'
CREATE TABLE IF NOT EXISTS region (r_regionkey INTEGER PRIMARY KEY, r_name CHAR(25) NOT NULL, r_comment VARCHAR(152));
CREATE TABLE IF NOT EXISTS nation (n_nationkey INTEGER PRIMARY KEY, n_name CHAR(25) NOT NULL, n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey), n_comment VARCHAR(152));
CREATE TABLE IF NOT EXISTS supplier (s_suppkey INTEGER PRIMARY KEY, s_name CHAR(25) NOT NULL, s_address VARCHAR(40) NOT NULL, s_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey), s_phone CHAR(15) NOT NULL, s_acctbal DECIMAL(15,2) NOT NULL, s_comment VARCHAR(101));
CREATE TABLE IF NOT EXISTS part (p_partkey INTEGER PRIMARY KEY, p_name VARCHAR(55) NOT NULL, p_mfgr CHAR(25) NOT NULL, p_brand CHAR(10) NOT NULL, p_type VARCHAR(25) NOT NULL, p_size INTEGER NOT NULL, p_container CHAR(10) NOT NULL, p_retailprice DECIMAL(15,2) NOT NULL, p_comment VARCHAR(23));
CREATE TABLE IF NOT EXISTS partsupp (ps_partkey INTEGER NOT NULL REFERENCES part(p_partkey), ps_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey), ps_availqty INTEGER NOT NULL, ps_supplycost DECIMAL(15,2) NOT NULL, ps_comment VARCHAR(199), PRIMARY KEY (ps_partkey, ps_suppkey));
CREATE TABLE IF NOT EXISTS customer (c_custkey INTEGER PRIMARY KEY, c_name VARCHAR(25) NOT NULL, c_address VARCHAR(40) NOT NULL, c_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey), c_phone CHAR(15) NOT NULL, c_acctbal DECIMAL(15,2) NOT NULL, c_mktsegment CHAR(10) NOT NULL, c_comment VARCHAR(117));
CREATE TABLE IF NOT EXISTS orders (o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER NOT NULL REFERENCES customer(c_custkey), o_orderstatus CHAR(1) NOT NULL, o_totalprice DECIMAL(15,2) NOT NULL, o_orderdate DATE NOT NULL, o_orderpriority CHAR(15) NOT NULL, o_clerk CHAR(15) NOT NULL, o_shippriority INTEGER NOT NULL, o_comment VARCHAR(79));
CREATE TABLE IF NOT EXISTS lineitem (l_orderkey INTEGER NOT NULL REFERENCES orders(o_orderkey), l_partkey INTEGER NOT NULL REFERENCES part(p_partkey), l_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey), l_linenumber INTEGER NOT NULL, l_quantity DECIMAL(15,2) NOT NULL, l_extendedprice DECIMAL(15,2) NOT NULL, l_discount DECIMAL(15,2) NOT NULL, l_tax DECIMAL(15,2) NOT NULL, l_returnflag CHAR(1) NOT NULL, l_linestatus CHAR(1) NOT NULL, l_shipdate DATE NOT NULL, l_commitdate DATE NOT NULL, l_receiptdate DATE NOT NULL, l_shipinstruct CHAR(25) NOT NULL, l_shipmode CHAR(10) NOT NULL, l_comment VARCHAR(44), PRIMARY KEY (l_orderkey, l_linenumber));
SQLEOF
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -v ON_ERROR_STOP=1 -f /tmp/create_tpch.sql"

# ---- 4. load ----
log "[4/5] Loading data with \\copy..."
cat > /tmp/load_tpch.sql <<SQLEOF
\\copy region FROM '${DBGEN_DIR}/region.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy nation FROM '${DBGEN_DIR}/nation.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy supplier FROM '${DBGEN_DIR}/supplier.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy part FROM '${DBGEN_DIR}/part.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy partsupp FROM '${DBGEN_DIR}/partsupp.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy customer FROM '${DBGEN_DIR}/customer.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy orders FROM '${DBGEN_DIR}/orders.tbl' WITH (DELIMITER '|', FORMAT csv);
\\copy lineitem FROM '${DBGEN_DIR}/lineitem.tbl' WITH (DELIMITER '|', FORMAT csv);
ANALYZE;
SQLEOF
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -v ON_ERROR_STOP=1 -f /tmp/load_tpch.sql"

# ---- 5. secondary indexes ----
log "[5/5] Creating secondary indexes..."
cat > /tmp/index_tpch.sql <<'SQLEOF'
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
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -v ON_ERROR_STOP=1 -f /tmp/index_tpch.sql"

cnt="$(su - postgres -c "export PATH=${PGBIN}:\$PATH && psql -tAqc 'SELECT count(*) FROM lineitem;' ${DB_NAME}")"
log "TPC-H SF${TPCH_SCALE} loaded. lineitem rows: ${cnt}"
