#!/usr/bin/env bash
# Download GIST-960 (1M x 960-dim) and load it into table 'gist_data'
# (the table name the experiment scripts query). Run AFTER build-postgres.sh.
# Idempotent: skips download if the .hdf5 exists and skips load if gist_data
# already has 1,000,000 rows (use --force to reload). Architecture-agnostic.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

DATA_DIR="${REPO_ROOT}/vdb-tuning-main/data"
HDF5="${DATA_DIR}/gist-960-euclidean.hdf5"
LOADER="${DATA_DIR}/load_hdf5_to_postgres.py"
URL="http://ann-benchmarks.com/gist-960-euclidean.hdf5"
TABLE="gist_data"
EXPECTED_ROWS=1000000

VENV="/opt/vdb-venv"; [ -x "${VENV}/bin/python" ] || VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
[ -x "${PY}" ] || die "python venv not found; run install-prerequisites.sh first"
[ -f "${LOADER}" ] || die "loader not found: ${LOADER}"

log "======================================================"
log "GIST-960 load into table '${TABLE}'"
log "======================================================"

# Skip if already loaded.
if [ "${FORCE}" -eq 0 ]; then
    rows="$(PGPASSWORD="${PG_PASSWORD}" "${PGBIN}/psql" -h 127.0.0.1 -U postgres -d "${DB_NAME}" -tAqc \
        "SELECT count(*) FROM ${TABLE};" 2>/dev/null || echo 0)"
    if [ "${rows}" = "${EXPECTED_ROWS}" ]; then
        log "${TABLE} already has ${rows} rows. Skipping (use --force to reload)."
        exit 0
    fi
fi

# ---- 1. download ----
mkdir -p "${DATA_DIR}"
if [ -f "${HDF5}" ]; then
    log "[1/2] Dataset present: ${HDF5} ($(du -h "${HDF5}" | cut -f1))"
else
    log "[1/2] Downloading GIST-960 (~1GB) from ${URL}..."
    wget -O "${HDF5}.part" "${URL}"
    mv "${HDF5}.part" "${HDF5}"
    log "      Downloaded: $(du -h "${HDF5}" | cut -f1)"
fi

# ---- 2. load ----
log "[2/2] Loading vectors into '${TABLE}' via COPY BINARY..."
# The loader reads DB_CONFIG (user=postgres/pw=secret/host=localhost:5432) from
# the script itself; we ensured that password + TCP md5 in build-postgres.sh.
cd "${DATA_DIR}"
"${PY}" "${LOADER}" --file "$(basename "${HDF5}")" --table "${TABLE}"

rows="$(PGPASSWORD="${PG_PASSWORD}" "${PGBIN}/psql" -h 127.0.0.1 -U postgres -d "${DB_NAME}" -tAqc \
    "SELECT count(*) FROM ${TABLE};")"
log "GIST loaded. ${TABLE} rows: ${rows}"
[ "${rows}" = "${EXPECTED_ROWS}" ] || log "WARNING: expected ${EXPECTED_ROWS} rows, got ${rows}."
