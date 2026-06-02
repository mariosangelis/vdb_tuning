#!/bin/bash
# Install pgvectorscale (StreamingDiskANN) on PostgreSQL 18
# This script handles the full installation from scratch.
# Run as root: sudo bash install_pgvectorscale_pg18.sh
set -e

echo '=== Installing pgvectorscale (StreamingDiskANN) on PG18 ==='

# Config
CARGO_USER="mangelis"  # user that has cargo installed
CARGO_BIN="/users/${CARGO_USER}/.cargo/bin"
PG_CONFIG="/usr/local/pgsql/bin/pg_config"
PGRX_HOME="/root/.pgrx"
PGRX_VERSION="0.16.1"

# Step 1: Install Rust (if not already installed)
echo 'Step 1: Checking Rust installation...'
if [ ! -f "${CARGO_BIN}/cargo" ]; then
    echo '  Installing Rust for user...'
    su - ${CARGO_USER} -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
fi
echo "  Rust OK: $(su - ${CARGO_USER} -c 'source ~/.cargo/env && rustc --version')"

# Step 2: Install build dependencies
echo 'Step 2: Installing build dependencies...'
apt-get update -qq
apt-get install -y -qq pkg-config libssl-dev libclang-dev clang

# Step 3: Install cargo-pgrx (matching pgvectorscale's pgrx version)
echo "Step 3: Installing cargo-pgrx ${PGRX_VERSION}..."
export PATH="${CARGO_BIN}:${PATH}"
cargo install cargo-pgrx --version ${PGRX_VERSION} --locked

# Step 4: Create PGRX_HOME with config
echo 'Step 4: Setting up PGRX_HOME...'
mkdir -p ${PGRX_HOME}
cat > ${PGRX_HOME}/config.toml << EOF
[configs]
pg18 = "${PG_CONFIG}"
EOF

# Step 5: Clone pgvectorscale
echo 'Step 5: Cloning pgvectorscale...'
cd /tmp
rm -rf pgvectorscale
git clone https://github.com/timescale/pgvectorscale.git

# Step 6: Build and install
echo 'Step 6: Building and installing pgvectorscale (this takes ~5 min)...'
export PGRX_HOME="${PGRX_HOME}"
export PATH="${CARGO_BIN}:/usr/local/pgsql/bin:${PATH}"
cd /tmp/pgvectorscale/pgvectorscale
cargo pgrx install --pg-config=${PG_CONFIG} --release

echo ''
echo '=== pgvectorscale installed successfully! ==='
echo ''
echo 'Next steps (as postgres user):'
echo '  psql mydb -c "CREATE EXTENSION vectorscale;"'
echo '  psql mydb -c "CREATE INDEX sift_data_diskann_idx ON sift_data USING diskann (embedding vector_l2_ops);"'
echo ''
echo 'Query tuning:'
echo '  SET diskann.query_search_list_size = 100;  -- higher = better recall, slower'
echo '  SET diskann.query_rescore = 50;            -- rescore top candidates with full vectors'
