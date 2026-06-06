#!/bin/bash
# Automated PostgreSQL + pgvector setup.
# Usage: sudo ./setup_postgres.sh
# Machine-specific paths/flags come from config.yaml (edit that, not this).
set -e

# ---- Load central config (config.yaml) ----
# Locate config.yaml: $VDB_CONFIG override, else repo root (vdb-tuning-main/),
# else walk up from this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${VDB_CONFIG}" ]; then
    CONFIG_FILE="${VDB_CONFIG}"
elif [ -f "${SCRIPT_DIR}/vdb-tuning-main/config.yaml" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/vdb-tuning-main/config.yaml"
elif [ -f "${SCRIPT_DIR}/config.yaml" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
else
    # walk up looking for vdb-tuning-main/config.yaml or config.yaml
    d="${SCRIPT_DIR}"
    while [ "${d}" != "/" ]; do
        if [ -f "${d}/vdb-tuning-main/config.yaml" ]; then CONFIG_FILE="${d}/vdb-tuning-main/config.yaml"; break; fi
        if [ -f "${d}/config.yaml" ]; then CONFIG_FILE="${d}/config.yaml"; break; fi
        d="$(dirname "${d}")"
    done
fi
if [ -z "${CONFIG_FILE}" ] || [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config.yaml not found. Set VDB_CONFIG=/path/to/config.yaml"
    exit 1
fi

# Minimal dependency-free YAML reader for flat 'key: value' lines under a section.
# Usage: yget <section> <key>   (e.g. yget paths pg_src)
yget() {
    local section="$1" key="$2"
    awk -v sec="$section" -v key="$key" '
        /^[a-zA-Z0-9_]+:/ { cur=$1; sub(":","",cur) }
        $0 ~ "^[[:space:]]+" key ":" {
            if (cur==sec) {
                sub("^[[:space:]]*" key ":[[:space:]]*", "")
                gsub(/^["'"'"']|["'"'"']$/, "")   # strip surrounding quotes
                sub(/[[:space:]]*#.*$/, "")        # strip trailing comments
                gsub(/[[:space:]]+$/, "")          # strip trailing ws
                print; exit
            }
        }
    ' "${CONFIG_FILE}"
}

# Map config.yaml -> shell vars used below.
PG_SRC="$(yget paths pg_src)"
PGVECTOR_DEFAULT="$(yget paths pgvector_default)"
PGVECTOR_PARALLEL="$(yget paths pgvector_parallel)"
PG_INSTALL="$(yget paths pg_install)"
PG_DATA="$(yget paths pg_data)"
PG_CONF_SRC="$(yget paths pg_conf_src)"
DB_NAME="$(yget database name)"
PG_CONFIGURE_FLAGS="$(yget build configure_flags)"
SETUP_NVME="$(yget storage setup_nvme)"
NVME_DEV="$(yget storage nvme_dev)"
NVME_MOUNT="$(yget storage nvme_mount)"
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
# chmod 700 "${PG_DATA}"
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && initdb -D ${PG_DATA}" > /dev/null 2>&1
echo "  initdb at ${PG_DATA}"

# Persist PGDATA + PATH in the postgres user's .bashrc so that
# `su - postgres -c 'echo $PGDATA'` and psql/pg_ctl work without manual exports.
# Idempotent: only append if not already present. Paths come from config.
PG_BASHRC="$(eval echo ~postgres)/.bashrc"
grep -q "export PGDATA=${PG_DATA}" "${PG_BASHRC}" 2>/dev/null || \
    echo "export PGDATA=${PG_DATA}" >> "${PG_BASHRC}"
grep -q "export PATH=${PG_INSTALL}/bin" "${PG_BASHRC}" 2>/dev/null || \
    echo "export PATH=${PG_INSTALL}/bin:\$PATH" >> "${PG_BASHRC}"
chown postgres:postgres "${PG_BASHRC}"
echo "  postgres ~/.bashrc set: PGDATA=${PG_DATA}, PATH+=${PG_INSTALL}/bin"

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
