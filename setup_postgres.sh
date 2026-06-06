#!/bin/bash
# Automated PostgreSQL + pgvector setup for clnode265
# Usage: sudo ./setup_postgres.sh
set -e

PG_SRC=/users/mangelis/vdb_tuning/postgres/postgres
PGVECTOR_DEFAULT=${PG_SRC}/pgvector
PGVECTOR_PARALLEL=${PG_SRC}/pgvector_modified_final
PG_INSTALL=/usr/local/pgsql
PG_DATA=/mnt/nvme/pgdata
NVME_DEV=/dev/nvme0n1
PG_CONF_SRC=/users/mangelis/vdb_tuning/postgres_configurations/postgresql.conf

echo "========================================"
echo "PostgreSQL + pgvector Setup"
echo "Started: $(date)"
echo "========================================"

# Step 1: NVMe
echo "[1/8] Setting up NVMe..."
if mountpoint -q /mnt/nvme 2>/dev/null; then
    echo "  Already mounted."
else
    mkfs.ext4 -F ${NVME_DEV}
    mkdir -p /mnt/nvme
    mount ${NVME_DEV} /mnt/nvme
    grep -q nvme0n1 /etc/fstab || echo "${NVME_DEV} /mnt/nvme ext4 defaults,noatime 0 2" >> /etc/fstab
fi
echo "  NVMe: $(df -h /mnt/nvme | tail -1 | awk '{print $2}')"

# Step 2: Build deps
echo "[2/8] Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq libreadline-dev zlib1g-dev flex bison build-essential numactl python3-psycopg2 > /dev/null 2>&1
echo "  Done."

# Step 3: Compile PostgreSQL
echo "[3/8] Compiling PostgreSQL..."
cd ${PG_SRC}
make clean > /dev/null 2>&1 || true
./configure --prefix=${PG_INSTALL} --without-icu > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1
echo "  Installed to ${PG_INSTALL}"

# Build pg_prewarm
echo "  Building pg_prewarm..."
cd ${PG_SRC}/contrib/pg_prewarm
make clean > /dev/null 2>&1 || true
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1

# Step 4: postgres user & initdb
echo "[4/8] Setting up postgres user and initdb..."
export PATH=${PG_INSTALL}/bin:$PATH
id -u postgres > /dev/null 2>&1 || useradd -m -s /bin/bash postgres
mkdir -p ${PG_DATA}
chown postgres:postgres ${PG_DATA}
chmod 700 ${PG_DATA}
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && initdb -D ${PG_DATA}" > /dev/null 2>&1
echo "  initdb at ${PG_DATA}"

# Step 5: Compile pgvector (default)
echo "[5/8] Compiling pgvector..."
cd ${PGVECTOR_DEFAULT}
make clean > /dev/null 2>&1 || true
make -j$(nproc) PG_CONFIG=${PG_INSTALL}/bin/pg_config > /dev/null 2>&1
make install PG_CONFIG=${PG_INSTALL}/bin/pg_config > /dev/null 2>&1
echo "  pgvector installed."

# Step 6: Copy postgresql.conf from repo
echo "[6/8] Copying postgresql.conf from repo..."
cp ${PG_CONF_SRC} ${PG_DATA}/postgresql.conf
chown postgres:postgres ${PG_DATA}/postgresql.conf

# Also set pg_hba.conf for trust auth
cat > ${PG_DATA}/pg_hba.conf << 'HBA'
local   all   all                 trust
host    all   all   127.0.0.1/32  trust
host    all   all   ::1/128       trust
HBA
chown postgres:postgres ${PG_DATA}/pg_hba.conf
echo "  postgresql.conf copied, pg_hba.conf set to trust."

# Step 7: Start postgres, create db, extensions
echo "[7/8] Starting PostgreSQL and creating database..."
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && pg_ctl -D ${PG_DATA} -l /tmp/logfile start" > /dev/null 2>&1
sleep 3
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && createdb mydb"
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && psql mydb -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
su - postgres -c "export PATH=${PG_INSTALL}/bin:\$PATH && psql mydb -c 'CREATE EXTENSION IF NOT EXISTS pg_prewarm;'"
echo "  Database 'mydb' with extensions: vector, pg_prewarm."

# Step 8: cgroup
echo "[8/8] Setting up cgroup..."
mkdir -p /sys/fs/cgroup/postgres_test
echo "  cgroup created."

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo "PostgreSQL running on NVMe at ${PG_DATA}"
echo "Next: load TPC-H data + vector data, update experiment_config.json for 36+36 cores"
