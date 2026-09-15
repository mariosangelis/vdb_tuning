#!/usr/bin/env bash
# Shared helpers for the vdb_tuning build scripts.
# Source this file: . "$(dirname "$0")/_common.sh"
#
# Provides:
#   - CONFIG_FILE resolution (honours $VDB_CONFIG, else repo config.yaml)
#   - yget <section> <key>   flat YAML reader (same logic as the repo scripts)
#   - log / die helpers
set -euo pipefail

log() { echo -e "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Resolve the build/ directory this library lives in, and the repo root.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"

# Locate config: $VDB_CONFIG override, else build/config.container.yaml,
# else repo-root config.yaml.
if [ -n "${VDB_CONFIG:-}" ] && [ -f "${VDB_CONFIG}" ]; then
    CONFIG_FILE="${VDB_CONFIG}"
elif [ -f "${COMMON_DIR}/config.container.yaml" ]; then
    CONFIG_FILE="${COMMON_DIR}/config.container.yaml"
elif [ -f "${REPO_ROOT}/config.yaml" ]; then
    CONFIG_FILE="${REPO_ROOT}/config.yaml"
else
    die "config.yaml not found. Set VDB_CONFIG=/path/to/config.yaml"
fi

# Minimal dependency-free reader for flat 'key: value' under a section.
# Usage: yget paths pg_src
yget() {
    local section="$1" key="$2"
    awk -v sec="$section" -v key="$key" '
        /^[a-zA-Z0-9_]+:/ { cur=$1; sub(":","",cur) }
        $0 ~ "^[[:space:]]+" key ":" {
            if (cur==sec) {
                sub("^[[:space:]]*" key ":[[:space:]]*", "")
                gsub(/^["\x27]|["\x27]$/, "")
                sub(/[[:space:]]*#.*$/, "")
                gsub(/[[:space:]]+$/, "")
                print; exit
            }
        }
    ' "${CONFIG_FILE}"
}

# Common config values (available to all scripts that source this).
PG_SRC="$(yget paths pg_src)"
PGVECTOR_DEFAULT="$(yget paths pgvector_default)"
PGVECTOR_PARALLEL="$(yget paths pgvector_parallel)"
PG_INSTALL="$(yget paths pg_install)"
PG_DATA="$(yget paths pg_data)"
PG_CONF_SRC="$(yget paths pg_conf_src)"
SNAPSHOT_DIR="$(yget paths snapshot_dir)"
DBGEN_DIR="$(yget paths dbgen_dir)"
DB_NAME="$(yget database name)"
PG_CONFIGURE_FLAGS="$(yget build configure_flags)"
TPCH_SCALE="$(yget tpch scale)"
PGBIN="${PG_INSTALL}/bin"

# Postgres superuser password used by the Python loaders (user=postgres/pw=secret/tcp).
PG_PASSWORD="${PG_PASSWORD:-secret}"

# Run a psql command as the postgres OS user with PATH set.
pg_psql() {
    su - postgres -c "export PATH=${PGBIN}:\$PATH && psql $*"
}

export CONFIG_FILE REPO_ROOT COMMON_DIR
export PG_SRC PGVECTOR_DEFAULT PGVECTOR_PARALLEL PG_INSTALL PG_DATA PG_CONF_SRC
export SNAPSHOT_DIR DBGEN_DIR DB_NAME PG_CONFIGURE_FLAGS TPCH_SCALE PGBIN PG_PASSWORD
