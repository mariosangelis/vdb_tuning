#!/usr/bin/env bash
# =============================================================================
# vdb_tuning — ONE-SHOT INSTALL (host + container + build + data)
# =============================================================================
# Runs the entire base setup end to end, on a fresh machine, from the HOST:
#   1. Install Docker CE + compose plugin (arch-aware: x86_64 / aarch64)
#   2. Create the shared data dir (/mnt/nvme) and the persistent pgsql dir
#   3. Build the Ubuntu image (locales, pigz, pyyaml, build deps baked in)
#   4. Start the container (48G /dev/shm, privileged, bind-mounts)
#   5. Apply repo fixes needed to run on this environment
#   6. Inside the container: build PostgreSQL (from repo source) + pgvector,
#      load TPC-H SF10, download + load GIST-960, build the base IVFFlat index
#
# Idempotent: re-running skips work already done. Works on Amazon Linux 2023
# and Ubuntu/CloudLab. Nothing here is machine-specific except what config
# files declare.
#
# Usage (from the repo, on the host):
#   ./build/install.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../vdb_tuning/build
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
sub()  { echo -e "    $*"; }

DOCKER="sudo docker"
CN="vdb-tuning"    # container name (matches docker-compose.yml)
NVME_DIR="/mnt/nvme"
# Space needed: ~33GB PGDATA + up to ~45GB snapshots + build headroom.
MIN_FREE_GB="${MIN_FREE_GB:-150}"

# -----------------------------------------------------------------------------
log "[0/6] Preflight: architecture, disk space, and shm sizing"
# -----------------------------------------------------------------------------
sub "Architecture: $(uname -m)   (ubuntu:24.04 image is multi-arch; nothing to do)"

# Where will /mnt/nvme live? If the operator already mounted a dedicated disk at
# /mnt/nvme, we use it as-is. Otherwise it is a directory on the root filesystem.
if mountpoint -q "${NVME_DIR}" 2>/dev/null; then
    sub "${NVME_DIR} is a dedicated mount (using it as-is)."
    CHECK_PATH="${NVME_DIR}"
else
    sub "${NVME_DIR} is a directory on the root filesystem (no dedicated disk mounted)."
    sub "  -> If you WANT a dedicated data disk, Ctrl-C now, mount it at ${NVME_DIR}, and re-run."
    CHECK_PATH="$(dirname "${NVME_DIR}")"   # root fs
fi

# Disk space check.
FREE_GB="$(df -BG --output=avail "${CHECK_PATH}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "${FREE_GB}" ] && [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    echo "ERROR: only ${FREE_GB} GB free at ${CHECK_PATH}; need >= ${MIN_FREE_GB} GB." >&2
    echo "       (33GB PGDATA + ~45GB snapshots + build). Free space or mount a bigger disk at ${NVME_DIR}." >&2
    exit 1
fi
sub "Disk: ${FREE_GB:-?} GB free at ${CHECK_PATH} (need >= ${MIN_FREE_GB} GB) — OK"

# Auto-size /dev/shm for PostgreSQL parallel index builds. Use ~75% of host RAM,
# capped at 48G, min 4G. Rewrite the shm_size line in docker-compose.yml so this
# works on VMs of any RAM size (a 61GB box gets ~45G; a 16GB box gets ~12G).
RAM_GB="$(free -g | awk '/^Mem:/{print $2}')"
SHM_GB=$(( RAM_GB * 3 / 4 ))
[ "${SHM_GB}" -gt 48 ] && SHM_GB=48
[ "${SHM_GB}" -lt 4 ]  && SHM_GB=4
sub "Host RAM: ${RAM_GB} GB  ->  setting /dev/shm to ${SHM_GB}g in docker-compose.yml"
sed -i "s/^\( *shm_size:\).*/\1 \"${SHM_GB}gb\"/" "${SCRIPT_DIR}/docker-compose.yml"

# -----------------------------------------------------------------------------
log "[1/6] Installing Docker CE + compose plugin on the host"
# -----------------------------------------------------------------------------
bash "${SCRIPT_DIR}/install-docker-host.sh"

# -----------------------------------------------------------------------------
log "[2/6] Preparing host directories (bind-mount targets)"
# -----------------------------------------------------------------------------
# /mnt/nvme  : PGDATA + snapshots (data survives container recreate)
# /mnt/nvme/pgsql : the compiled PostgreSQL install (survives container recreate)
sudo mkdir -p /mnt/nvme /mnt/nvme/pgsql
sudo chmod 777 /mnt/nvme
sub "Created /mnt/nvme and /mnt/nvme/pgsql"

# -----------------------------------------------------------------------------
log "[3/6] Building the container image"
# -----------------------------------------------------------------------------
${DOCKER} compose build

# -----------------------------------------------------------------------------
log "[4/6] Starting the container"
# -----------------------------------------------------------------------------
${DOCKER} compose up -d
sleep 3
${DOCKER} exec "${CN}" bash -lc 'df -h /dev/shm | tail -1'

# -----------------------------------------------------------------------------
log "[5/6] Applying environment fixes to repo files"
# -----------------------------------------------------------------------------
bash "${SCRIPT_DIR}/apply-fixes.sh"

# -----------------------------------------------------------------------------
log "[6/6] Building Postgres + pgvector, loading data (inside container)"
# -----------------------------------------------------------------------------
# setup-all.sh runs: prerequisites -> build-postgres -> load-tpch -> load-gist
#                    -> build-indexes  (all idempotent, logged)
${DOCKER} exec "${CN}" bash -lc '/vdb_tuning/build/setup-all.sh'

log "INSTALL COMPLETE"
cat <<EOF

  Container:  ${CN}   (docker exec -it ${CN} bash)
  Postgres:   PGPASSWORD=secret psql -h 127.0.0.1 -U postgres -d mydb
  Data:       gist_data (1M vectors) + TPC-H SF10, base IVFFlat index built

  Next:
    ./build/tune.sh            # build + tune all index types (ivfflat/hnsw/parallel)
    ./build/run-experiment.sh  # run a specific experiment category (see --help)
EOF
