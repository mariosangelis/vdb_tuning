#!/usr/bin/env bash
# Install Docker CE on the HOST and prepare the shared data directory.
# Runs on the host (not in the container). Works on:
#   - Amazon Linux 2023 / Fedora-like  (dnf)
#   - Ubuntu / Debian                  (apt-get, e.g. CloudLab)
# Architecture-agnostic (x86_64 and arm64).
set -euo pipefail

NVME_DIR="${NVME_DIR:-/mnt/nvme}"

echo "==> Detecting host OS..."
if [ -r /etc/os-release ]; then
    . /etc/os-release
else
    echo "ERROR: /etc/os-release not found; unsupported host." >&2
    exit 1
fi
echo "    ID=${ID:-?}  VERSION_ID=${VERSION_ID:-?}  ARCH=$(uname -m)"

install_docker_dnf() {
    echo "==> Installing Docker via dnf (Amazon Linux 2023 / Fedora)..."
    if [ "${ID:-}" = "amzn" ]; then
        # Amazon Linux 2023 ships Docker in its own repos.
        sudo dnf install -y docker
    else
        # Fedora: use Docker's official repo.
        sudo dnf -y install dnf-plugins-core || true
        sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo || true
        sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi
}

install_docker_apt() {
    echo "==> Installing Docker via apt-get (Ubuntu/Debian)..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    local distro="ubuntu"
    [ "${ID:-}" = "debian" ] && distro="debian"
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
        curl -fsSL "https://download.docker.com/linux/${distro}/gpg" \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
    fi
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/${distro} $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

if command -v docker >/dev/null 2>&1; then
    echo "==> Docker already installed: $(docker --version)"
else
    case "${ID:-}" in
        amzn|fedora|rhel|centos|rocky|almalinux) install_docker_dnf ;;
        ubuntu|debian)                           install_docker_apt ;;
        *)
            if command -v dnf >/dev/null 2>&1; then install_docker_dnf
            elif command -v apt-get >/dev/null 2>&1; then install_docker_apt
            else echo "ERROR: no supported package manager (dnf/apt-get)." >&2; exit 1
            fi
            ;;
    esac
fi

echo "==> Enabling and starting Docker service..."
sudo systemctl enable --now docker

echo "==> Adding user '${USER}' to the docker group (takes effect on next login)..."
sudo usermod -aG docker "${USER}" || true

echo "==> Creating shared data directory ${NVME_DIR} (bind-mounted into the container)..."
sudo mkdir -p "${NVME_DIR}"
# Container's postgres user is uid/gid 999 or 1000 depending on base image;
# build-postgres.sh chowns PGDATA inside the container, so just make it writable.
sudo chmod 777 "${NVME_DIR}"

echo "==> Verifying Docker..."
if sudo docker run --rm hello-world >/dev/null 2>&1; then
    echo "    Docker works."
else
    echo "    WARNING: 'docker run hello-world' failed; check 'sudo systemctl status docker'." >&2
fi

cat <<EOF

==================================================================
Docker install complete.

NOTE: your user was added to the 'docker' group but that only takes
effect after you log out and back in (or run: newgrp docker).
Until then, prefix docker commands with sudo.

Next:
  cd /path/to/vdb_tuning/build
  sudo docker compose up -d --build      # build image + start container
  sudo docker exec -it vdb-tuning bash    # enter container
  # inside the container:
  /vdb_tuning/build/setup-all.sh
==================================================================
EOF
