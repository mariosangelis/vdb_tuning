#!/bin/bash
# Automated PostgreSQL + pgvector setup.
# Usage: sudo ./setup_postgres.sh
# Machine-specific paths/flags come from setup_config.sh (edit that, not this).
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

echo "========================================"
echo "PostgreSQL + pgvector Setup"
echo "Started: $(date)"
echo "  PG_SRC      = ${PG_SRC}"
echo "  PG_INSTALL  = ${PG_INSTALL}"
echo "  PG_DATA     = ${PG_DATA}"
echo "  SETUP_NVME  = ${SETUP_NVME}"
echo "========================================"

# Step 1: Storage
if [ "${SETUP_NVME}" = "true" ]; then
    echo "[1/8] Setting up NVMe (${NVME_DEV} -> ${NVME_MOUNT})..."
    if mountpoint -q "${NVME_MOUNT}" 2>/dev/null; then
        echo "  Already mounted."
    else
        mkfs.ext4 -F "${NVME_DEV}"
        mkdir -p "${NVME_MOUNT}"
        mount "${NVME_DEV}" "${NVME_MOUNT}"
        grep -q "$(basename "${NVME_DEV}")" /etc/fstab || \
            echo "${NVME_DEV} ${NVME_MOUNT} ext4 defaults,noatime 0 2" >> /etc/fstab
    fi
    echo "  NVMe: $(df -h "${NVME_MOUNT}" | tail -1 | awk '{print $2}')"
else
    echo "[1/8] Skipping NVMe setup (SETUP_NVME=false); using existing storage for PG_DATA."
fi

# Step 2: Build deps
echo "[2/8] Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq libreadline-dev zlib1g-dev flex bison build-essential numactl python3-psycopg2 > /dev/null 2>&1
echo "  Done."

# Step 3: Compile PostgreSQL
echo "[3/8] Compiling PostgreSQL..."
cd "${PG_SRC}"
make clean > /dev/null 2>&1 || true
./configure --prefix="${PG_INSTALL}" ${PG_CONFIGURE_FLAGS} > /dev/null 2>&1
make -j"$(nproc)" > /dev/null 2>&1
make install > /dev/null 2>&1
echo "  Installed to ${PG_INSTALL}"

# Build contrib modules: pg_prewarm, pg_buffercache
for mod in pg_prewarm pg_buffercache; do
    echo "  Building ${mod}..."
    cd "${PG_SRC}/contrib/${mod}"
    make clean > /dev/null 2>&1 || true
    make -j"$(nproc)" > /dev/null 2>&1
    make install > /dev/null 2>&1
done

# Step 4: postgres user & initdb
echo "[4/8] Setting up postgres user and initdb..."
export PATH=${PG_INSTALL}/bin:$PATH
id -u postgres > /dev/null 2>&1 || useradd -m -s /bin/bash postgres
mkdir -p "${PG_DATA}"
chown postgres:postgres "${PG_DATA}"
chmod 700 "${PG_DATA}"
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && initdb -D ${PG_DATA}" > /dev/null 2>&1
echo "  initdb at ${PG_DATA}"

# Step 5: Compile pgvector (default)
echo "[5/8] Compiling pgvector..."
cd "${PGVECTOR_DEFAULT}"
make clean > /dev/null 2>&1 || true
make -j"$(nproc)" PG_CONFIG="${PG_INSTALL}/bin/pg_config" > /dev/null 2>&1
make install PG_CONFIG="${PG_INSTALL}/bin/pg_config" > /dev/null 2>&1
echo "  pgvector installed."

# Step 6: Copy postgresql.conf from repo + trust pg_hba
echo "[6/8] Copying postgresql.conf from repo..."
cp "${PG_CONF_SRC}" "${PG_DATA}/postgresql.conf"
chown postgres:postgres "${PG_DATA}/postgresql.conf"

cat > "${PG_DATA}/pg_hba.conf" << 'HBA'
local   all   all                 trust
host    all   all   127.0.0.1/32  trust
host    all   all   ::1/128       trust
HBA
chown postgres:postgres "${PG_DATA}/pg_hba.conf"
echo "  postgresql.conf copied, pg_hba.conf set to trust."

# Step 7: Start postgres, create db, extensions
echo "[7/8] Starting PostgreSQL and creating database '${DB_NAME}'..."
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && pg_ctl -D ${PG_DATA} -l /tmp/logfile start" > /dev/null 2>&1
sleep 3
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && createdb ${DB_NAME}"
for ext in vector pg_prewarm pg_buffercache; do
    su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && psql ${DB_NAME} -c 'CREATE EXTENSION IF NOT EXISTS ${ext};'"
done
echo "  Database '${DB_NAME}' with extensions: vector, pg_prewarm, pg_buffercache."

# Step 8: cgroup
echo "[8/8] Setting up cgroup..."
mkdir -p /sys/fs/cgroup/postgres_test
echo "  cgroup created."

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo "PostgreSQL data at ${PG_DATA}"
echo "Next: run setup_tpch.sh to load TPC-H data, then update TPCH/code/experiment_config.json for cores."
