#!/usr/bin/env bash
# Verify/install build + Python prerequisites INSIDE the container.
# Idempotent: safe to re-run. Architecture-agnostic.
#
# In the provided Docker image these are already installed; this script exists
# so the flow also works if you run it in a plain Ubuntu container/host.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

log "Prerequisites: checking system packages..."

PKGS=(build-essential gcc make libreadline-dev zlib1g-dev libicu-dev pkg-config
      flex bison libxml2-dev libssl-dev python3 python3-pip python3-venv
      python3-dev numactl git wget curl ca-certificates)

missing=()
for p in "${PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done

if [ "${#missing[@]}" -gt 0 ]; then
    log "Installing missing packages: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${missing[@]}"
else
    log "All system packages present."
fi

# Python venv: prefer the image's /opt/vdb-venv; else create one at REPO_ROOT/.venv.
VENV="${VENV:-/opt/vdb-venv}"
if [ ! -x "${VENV}/bin/python" ]; then
    VENV="${REPO_ROOT}/.venv"
    if [ ! -x "${VENV}/bin/python" ]; then
        log "Creating Python venv at ${VENV}..."
        python3 -m venv "${VENV}"
    fi
fi

log "Ensuring Python packages in ${VENV}..."
"${VENV}/bin/pip" install --no-cache-dir --upgrade pip >/dev/null
"${VENV}/bin/pip" install --no-cache-dir \
    psycopg2-binary "psycopg[binary]" pgvector h5py numpy psutil matplotlib pandas

log "Prerequisites OK. venv=${VENV}"
