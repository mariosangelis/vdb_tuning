#!/usr/bin/env bash
# Build PostgreSQL 18 + contrib + pgvector, init the cluster, configure auth,
# start the server, and create the database + extensions.
#
# Idempotent: skips recompiles / re-inits that are already done (use --force to
# rebuild). Architecture-agnostic: no -march/AVX/NEON flags are forced; PG's
# configure and pgvector's Makefile auto-detect the CPU/ISA.
#
# Run INSIDE the container as a user with sudo (the image's postgres user works),
# or as root.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FORCE=0
PGVECTOR_VARIANT="parallel"   # parallel (default) | default
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --pgvector=default)  PGVECTOR_VARIANT="default" ;;
        --pgvector=parallel) PGVECTOR_VARIANT="parallel" ;;
        *) die "unknown arg: $arg" ;;
    esac
done

# Must run privileged (we useradd, write to /usr/local, manage cgroups).
if [ "$(id -u)" -ne 0 ]; then
    exec sudo -E PG_PASSWORD="${PG_PASSWORD}" bash "$0" "$@"
fi

ARCH="$(uname -m)"
log "======================================================"
log "PostgreSQL + pgvector build  (arch=${ARCH})"
log "  PG_SRC     = ${PG_SRC}"
log "  PG_INSTALL = ${PG_INSTALL}"
log "  PG_DATA    = ${PG_DATA}"
log "  pgvector   = ${PGVECTOR_VARIANT}"
log "  config     = ${CONFIG_FILE}"
log "======================================================"

NPROC="$(nproc)"

# ---- 1. Compile PostgreSQL (skip if already installed unless --force) ----
if [ "${FORCE}" -eq 0 ] && [ -x "${PG_INSTALL}/bin/postgres" ]; then
    log "[1/7] PostgreSQL already installed at ${PG_INSTALL} (use --force to rebuild). Skipping."
else
    log "[1/7] Compiling PostgreSQL (this takes a few minutes)..."
    cd "${PG_SRC}"
    make clean >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    ./configure --prefix="${PG_INSTALL}" ${PG_CONFIGURE_FLAGS}
    make -j"${NPROC}"
    make install
    log "      Installed to ${PG_INSTALL}"
fi

# ---- 2. contrib modules ----
log "[2/7] Building contrib modules (pg_prewarm, pg_buffercache)..."
for mod in pg_prewarm pg_buffercache; do
    if [ "${FORCE}" -eq 0 ] && [ -f "${PG_INSTALL}/lib/${mod}.so" ]; then
        log "      ${mod} present. Skipping."
        continue
    fi
    cd "${PG_SRC}/contrib/${mod}"
    make clean >/dev/null 2>&1 || true
    make -j"${NPROC}"
    make install
done

# ---- 3. pgvector ----
if [ "${PGVECTOR_VARIANT}" = "parallel" ]; then
    PGVECTOR_DIR="${PGVECTOR_PARALLEL}"
else
    PGVECTOR_DIR="${PGVECTOR_DEFAULT}"
fi
[ -d "${PGVECTOR_DIR}" ] || die "pgvector dir not found: ${PGVECTOR_DIR}"
log "[3/7] Compiling pgvector (${PGVECTOR_VARIANT}) from ${PGVECTOR_DIR}..."
cd "${PGVECTOR_DIR}"
make clean PG_CONFIG="${PG_INSTALL}/bin/pg_config" >/dev/null 2>&1 || true
# OPTFLAGS="" avoids pgvector's Makefile injecting -march=native, which is not
# portable and is x86-centric; the compiler still uses sane arch defaults.
make -j"${NPROC}" OPTFLAGS="" PG_CONFIG="${PG_INSTALL}/bin/pg_config"
make install PG_CONFIG="${PG_INSTALL}/bin/pg_config"
log "      pgvector installed."

export PATH="${PG_INSTALL}/bin:${PATH}"

# ---- 4. postgres user + initdb ----
log "[4/7] Ensuring postgres user and data directory..."
id -u postgres >/dev/null 2>&1 || useradd -m -s /bin/bash postgres
mkdir -p "${PG_DATA}"
chown -R postgres:postgres "${PG_DATA}"

if [ "${FORCE}" -eq 0 ] && [ -f "${PG_DATA}/PG_VERSION" ]; then
    log "      Cluster already initialized at ${PG_DATA}. Skipping initdb."
else
    if [ -f "${PG_DATA}/PG_VERSION" ]; then
        log "      --force: stopping any server and wiping ${PG_DATA}..."
        su - postgres -c "export PATH=${PGBIN}:\$PATH && pg_ctl -D ${PG_DATA} stop -m immediate" >/dev/null 2>&1 || true
        rm -rf "${PG_DATA:?}/"*
    fi
    log "      Running initdb..."
    su - postgres -c "export PATH=${PGBIN}:\$PATH && initdb -D ${PG_DATA} --auth-local=trust --auth-host=md5"
fi

# Persist PGDATA + PATH for the postgres user.
PG_BASHRC="$(eval echo ~postgres)/.bashrc"
grep -q "export PGDATA=${PG_DATA}" "${PG_BASHRC}" 2>/dev/null || echo "export PGDATA=${PG_DATA}" >> "${PG_BASHRC}"
grep -q "export PATH=${PG_INSTALL}/bin" "${PG_BASHRC}" 2>/dev/null || echo "export PATH=${PG_INSTALL}/bin:\$PATH" >> "${PG_BASHRC}"
chown postgres:postgres "${PG_BASHRC}"

# ---- 5. Config + auth (postgresql.conf, listen, pg_hba md5 for TCP) ----
log "[5/7] Applying postgresql.conf and TCP md5 auth..."
[ -f "${PG_CONF_SRC}" ] && cp "${PG_CONF_SRC}" "${PG_DATA}/postgresql.conf"
# Ensure the server listens on localhost TCP (loaders connect to 127.0.0.1:5432).
if ! grep -qE "^[[:space:]]*listen_addresses" "${PG_DATA}/postgresql.conf"; then
    echo "listen_addresses = 'localhost'" >> "${PG_DATA}/postgresql.conf"
fi
# pg_hba: local trust (already), TCP md5 for the loaders.
HBA="${PG_DATA}/pg_hba.conf"
if ! grep -qE "^host\s+all\s+all\s+127\.0\.0\.1/32\s+md5" "${HBA}" 2>/dev/null; then
    {
        echo "# added by build-postgres.sh: TCP md5 for Python loaders"
        echo "host    all    all    127.0.0.1/32    md5"
        echo "host    all    all    ::1/128         md5"
    } >> "${HBA}"
fi
chown postgres:postgres "${PG_DATA}/postgresql.conf" "${HBA}"

# ---- 6. Start server, set password, create db + extensions ----
log "[6/7] Starting PostgreSQL and creating database '${DB_NAME}'..."
if su - postgres -c "export PATH=${PGBIN}:\$PATH && pg_ctl -D ${PG_DATA} status" >/dev/null 2>&1; then
    log "      Server already running. Reloading config..."
    su - postgres -c "export PATH=${PGBIN}:\$PATH && pg_ctl -D ${PG_DATA} reload" >/dev/null 2>&1 || true
else
    su - postgres -c "export PATH=${PGBIN}:\$PATH && pg_ctl -D ${PG_DATA} -l /tmp/pg_logfile -w start" || {
        log "      pg_ctl start failed; last log lines:"; tail -20 /tmp/pg_logfile || true; die "postgres failed to start";
    }
fi
sleep 2

# Set the postgres role password (used by user=postgres/pw=secret TCP loaders).
su - postgres -c "export PATH=${PGBIN}:\$PATH && psql -c \"ALTER USER postgres WITH PASSWORD '${PG_PASSWORD}';\"" >/dev/null

# Create db if missing.
if ! su - postgres -c "export PATH=${PGBIN}:\$PATH && psql -lqt" | cut -d'|' -f1 | grep -qw "${DB_NAME}"; then
    su - postgres -c "export PATH=${PGBIN}:\$PATH && createdb ${DB_NAME}"
fi

for ext in vector pg_prewarm pg_buffercache; do
    su - postgres -c "export PATH=${PGBIN}:\$PATH && psql ${DB_NAME} -c 'CREATE EXTENSION IF NOT EXISTS ${ext};'" >/dev/null
done
log "      Database '${DB_NAME}' ready with extensions: vector, pg_prewarm, pg_buffercache."

# ---- 7. cgroup for the memory-limit experiments ----
log "[7/7] Ensuring cgroup /sys/fs/cgroup/postgres_test..."
mkdir -p /sys/fs/cgroup/postgres_test 2>/dev/null || log "      (could not create cgroup; needs privileged container)"

# Verify TCP auth works end to end.
log "Verifying TCP connection as postgres/${PG_PASSWORD}..."
if PGPASSWORD="${PG_PASSWORD}" "${PGBIN}/psql" -h 127.0.0.1 -U postgres -d "${DB_NAME}" -tAc "select 'ok', version();" ; then
    log "TCP auth verified."
else
    die "TCP auth failed — the Python loaders will not be able to connect."
fi

log "======================================================"
log "PostgreSQL build COMPLETE."
log "  version: $(${PGBIN}/postgres --version)"
log "  connect: psql -h 127.0.0.1 -U postgres -d ${DB_NAME}  (password: ${PG_PASSWORD})"
log "======================================================"
