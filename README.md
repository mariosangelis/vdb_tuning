# VDB Tuning: OLAP/ANN Workload Interference in PostgreSQL with pgvector

This repository contains the experimental framework for analyzing interference between OLAP (TPC-H) and ANN (vector similarity search) workloads in PostgreSQL with a modified pgvector extension.

## Repository Contents

```
vdb_tuning/
├── postgres/postgres/          # PostgreSQL 18 source (REL_18_STABLE)
├── pgvector_modified_final/    # Modified pgvector extension (parallel IVFFlat)
├── vdb-tuning-main/
│   ├── TPCH/
│   │   ├── code/               # Experiment scripts
│   │   │   ├── execute_ann_impact_automated.py   # Main experiment: ANN impact on OLAP
│   │   │   ├── run_tpch_queries.py               # Standalone TPC-H runner
│   │   │   └── plot_paper_style_figures.py        # Plotting
│   │   └── results/            # Experiment output (JSON lines)
│   ├── data/
│   │   ├── gist-960-euclidean.hdf5   # GIST-960 dataset (not in git, ~1GB)
│   │   └── load_hdf5_to_postgres.py  # Vector data loader
│   ├── gist_test_queries.txt   # Pre-extracted query vectors (not in git)
│   └── benchbase_configs/      # BenchBase XML configs (optional)
└── README.md
```

## Target Machine Requirements

- **OS**: Ubuntu 24.04 (or any Linux with cgroup v2)
- **CPU**: 16+ physical cores (experiment uses 16 for PostgreSQL)
- **RAM**: 64GB+ (experiments test cgroup limits from 12GB to 60GB)
- **Storage**: NVMe SSD recommended (500GB+)
- **No hyperthreading** (disable SMT for consistent results)

The experiments were validated on:
- CloudLab Wisconsin: AMD EPYC 7543, 2x32 cores, 250GB RAM, 1.5TB NVMe, Ubuntu 24.04

---

## 1. Install Prerequisites

### Ubuntu 24.04

```bash
sudo apt-get update
sudo apt-get install -y build-essential libreadline-dev zlib1g-dev libicu-dev \
    pkg-config flex bison libxml2-dev libssl-dev \
    python3-pip python3-dev numactl

pip3 install psycopg2-binary h5py numpy psutil "psycopg[binary]" pgvector --break-system-packages
```
---

## 2. Prepare Storage and Create postgres User

Mount the NVMe drive and create the postgres system user:

```bash
# Format and mount NVMe (skip if already mounted)
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1 /mnt/nvme

# Create postgres user and data directory
sudo useradd -m -s /bin/bash postgres
sudo mkdir -p /mnt/nvme/pgdata
sudo chown postgres:postgres /mnt/nvme/pgdata
```

**Note:** The mount is not persistent across reboots. Add to `/etc/fstab` if needed:
```bash
echo '/dev/nvme0n1 /mnt/nvme ext4 defaults 0 0' | sudo tee -a /etc/fstab
```

---

## 3. Compile and Install PostgreSQL 18

```bash
cd ~/vdb_tuning/postgres/postgres

# Clean any stale build artifacts (important if source was copied from another arch)
make distclean 2>/dev/null

# Configure with debug flags
./configure --enable-debug --enable-cassert CFLAGS="-O1 -fno-omit-frame-pointer"

# Build and install
make -j$(nproc)
sudo make install

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/pgsql/bin:$PATH
```

Default install prefix is `/usr/local/pgsql`. `sudo` is required for `make install`.

### 3.1 Install pg_buffercache Extension (optional, for monitoring)

```bash
cd ~/vdb_tuning/postgres/postgres/contrib/pg_buffercache
make
sudo make install
```

---

## 4. Compile and Install pgvector 

The default pgvector extension
```bash
cd ~/vdb_tuning/pgvector
export PATH=/usr/local/pgsql/bin:$PATH

make clean
make -j$(nproc)
sudo make install
```

The Makefile applies these optimization flags automatically:
- `-march=native` (uses AVX2/AVX-512 on x86, SVE/NEON on ARM)
- `-ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math`

---

## 5. Initialize and Start PostgreSQL

```bash
# Switch to postgres user
sudo su - postgres -s /bin/bash

# Set environment
export PGDATA=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH

# Initialize the database
initdb -D $PGDATA --encoding=UTF8 --locale=en_US.UTF-8

# Start the server
pg_ctl -D $PGDATA -l /tmp/logfile start

# Create database and enable extensions
createdb mydb
psql -d mydb -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d mydb -c "CREATE EXTENSION IF NOT EXISTS pg_buffercache;"

# Verify
psql -d mydb -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
# Expected: 0.8.0 or similar
```


## 6. Load Data

### 6.1 Option A: pg_restore from Dump (Fastest)

If you have a `pg_dump` directory from another machine:

```bash
# As root, fix permissions
chmod -R a+r /tmp/mydb_dump
chmod a+rx /tmp/mydb_dump

# As postgres user
export PATH=/usr/local/pgsql/bin:$PATH
pg_restore -j 8 -d mydb /tmp/mydb_dump
```

This restores all tables, data, and indexes. Takes ~10-15 minutes.

### 6.2 Option B: Load from Scratch

#### Load TPC-H Data (SF10)

Use BenchBase or dbgen to generate and load TPC-H scale factor 10 data:

```bash
# Using dbgen
git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen && make
./dbgen -s 10

# Then COPY each .tbl file into PostgreSQL (create schema first)
```

Or use BenchBase (requires Java 23):
```bash
# Install Java
sudo apt install -y openjdk-23-jdk   # Ubuntu
# or: sudo dnf install -y java-23-amazon-corretto-devel  # Amazon Linux

# Clone and build BenchBase
git clone --depth 1 https://github.com/cmu-db/benchbase.git
cd benchbase && ./mvnw clean package -P postgres -DskipTests
cd target && tar xvzf benchbase-postgres.tgz && cd benchbase-postgres

# Load TPC-H data
java -jar benchbase.jar -b tpch -c config/postgres/tpch_mydb_sf10.xml \
    --create=true --load=true --execute=false
```

#### Load GIST-960 Vectors

Download the GIST dataset and load into PostgreSQL:

```bash
cd ~/vdb_tuning/vdb-tuning-main/data

# Download GIST-960 dataset (~1GB)
wget http://ann-benchmarks.com/gist-960-euclidean.hdf5

# Load vectors into PostgreSQL (creates 'gist_data' table with ~1M vectors)
python3 load_hdf5_to_postgres.py --file gist-960-euclidean.hdf5 --table gist_data

# Save query vectors for experiments
python3 load_hdf5_to_postgres.py --file gist-960-euclidean.hdf5 --table gist_data \
    --save-queries ../gist_test_queries.txt --num-queries 1000
```

### 6.3 Create Vector Index

```sql
-- Connect to mydb as postgres
SET maintenance_work_mem = '15GB';

-- IVFFlat index (1000 lists, ~2-3 min build time)
CREATE INDEX gist_data_embedding_idx ON gist_data
    USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000);

-- OR: HNSW index (M=16, ef_construction=200, ~20-30 min build time)
CREATE INDEX gist_data_embedding_idx ON gist_data
    USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);
```

**Note:** Only one index at a time. Drop the existing one before creating another:
```sql
DROP INDEX IF EXISTS gist_data_embedding_idx;
```

### 6.4 Verify Data

```sql
SELECT count(*) FROM lineitem;       -- Expected: ~59,986,052 (SF10)
SELECT count(*) FROM gist_data;      -- Expected: ~1,000,000
\di+ gist_data_embedding_idx         -- Should show the index
```

---

## 8. Prepare for Experiments

### 8.1 Disable Hyperthreading

```bash
echo off | sudo tee /sys/devices/system/cpu/smt/control
# Verify: nproc should show half the original count
```

### 8.2 Set Up cgroup v2

The experiments use cgroup v2 to control PostgreSQL memory:

```bash
# Enable memory controller (may already be enabled)
echo "+memory" | sudo tee /sys/fs/cgroup/cgroup.subtree_control

# Create the experiment cgroup
sudo mkdir -p /sys/fs/cgroup/postgres_test

# Verify
cat /sys/fs/cgroup/postgres_test/memory.max
# Should show "max" (unlimited by default)
```

**Note:** The cgroup is not persistent across reboots. Re-run after each reboot.

### 8.3 Configure CPU Pinning

The experiment script (`execute_ann_impact_automated.py`) has these CPU settings at the top:

```python
NUM_PG_CORES = 16
POSTGRES_CORES = "0-15"         # PostgreSQL pinned to cores 0-15
TOPK_CORES = list(range(16, 26))  # ANN workers on cores 16-25
```

Adjust these for your machine. Key rules:
- Keep PostgreSQL and TopK on the **same NUMA socket** (check with `numactl --hardware`)
- Use physical cores only (disable HT first)
- Leave some cores unused to avoid interference with OS

The script uses `numactl --cpunodebind=0-3 --membind=0-3` to start PostgreSQL with NUMA-aware memory binding. Adjust the NUMA node numbers for your topology.

---

## 9. Running Experiments

### 9.1 Main Experiment: ANN Impact on OLAP

This measures how increasing ANN (TopK) concurrency degrades TPC-H performance under different memory pressures:

```bash
cd ~/vdb_tuning/vdb-tuning-main/TPCH/code

# Must run as root (needs cgroup + pg_ctl + cache dropping)
sudo python3 execute_ann_impact_automated.py
```

**Configuration** (edit the script before running):

```python
# Memory configurations: (name, shared_buffers_gb, cgroup_gb)
MEMORY_CONFIGS = [
    ("no_pressure", 20, 50),   # Everything fits in memory
    ("relaxed", 12, 30),       # Moderate pressure
    ("medium", 8, 20),         # Significant pressure
    ("tight", 4, 12),          # Heavy disk I/O
]

# ANN worker counts to test
TOPK_WORKERS_LIST = [0, 5, 10]

# Index type (must match the created index)
INDEX_TYPE = "IVFFLAT"  # or "HNSW"
```

**What it does for each (memory_config, ann_workers) combination:**
1. Sets `shared_buffers` in postgresql.conf
2. Sets cgroup `memory.max`
3. Drops OS page caches (`echo 3 > /proc/sys/vm/drop_caches`)
4. Restarts PostgreSQL with NUMA-aware CPU pinning
5. Moves PostgreSQL processes into the cgroup
6. Starts N ANN worker threads running continuous TopK vector queries
7. Runs all 22 TPC-H queries with `EXPLAIN (ANALYZE, BUFFERS)` to capture buffer statistics
8. Records per-query timing, shared_hit, shared_read counts
9. Saves results as JSON lines

### 9.2 Output Format

Results are saved as JSON lines (one JSON object per experiment run):

```json
{
  "tpch_query_times": [
    {"query": "Q1", "time_sec": 13.2, "shared_hit": 228, "shared_read": 1124552, "shared_written": 0},
    {"query": "Q2", "time_sec": 2.6, "shared_hit": 1586154, "shared_read": 233687, "shared_written": 0},
    ...
  ],
  "topk_query_count": 14612,
  "topk_throughput": 59.26,
  "topk_avg_exec_time_sec": 0.083,
  "tpch_execution_time": 246.59
}
```

Results files are named: `mixed_workload_tpch_and_topk_{INDEX}_{io}_{cores}_cores_for_postgres_{shared_buf}GB_shared_buffers_{cgroup}GB_RAM_GIST.txt`

---

## 10. Understanding the Results

### Memory Pressure Effect (0 ANN workers)

As the cgroup limit decreases, PostgreSQL's buffer pool can hold less data, forcing more disk I/O:

| Config | shared_buffers | cgroup | TPC-H Time | Buffer Hit Rate |
|--------|---------------|--------|-----------|----------------|
| no_pressure | 20GB | 50GB | ~115s | >99% (after Q1) |
| relaxed | 12GB | 30GB | ~130s | 50-99% |
| medium | 8GB | 20GB | ~200s | 30-90% |
| tight | 4GB | 12GB | ~315s | 2-90% |

### ANN Interference Effect

ANN (TopK) workers compete for the same shared buffer pool, evicting TPC-H data pages:

| Config | 0 ANN | 5 ANN | 10 ANN | Interference |
|--------|-------|-------|--------|-------------|
| no_pressure | 115s | 126s | 143s | 1.1-1.2x (minimal) |
| relaxed | 130s | 247s | 261s | 1.9-2.0x |
| medium | 199s | 410s | 412s | 2.1x |
| tight | 314s | 533s | 590s | 1.7-1.9x |

Key finding: Under no_pressure, the 40GB buffer pool absorbs both workloads with negligible interference. Under memory pressure, ANN index scans fill the buffer pool with GIST vector pages, evicting TPC-H data and causing dramatic OLAP degradation.

### Buffer Statistics

The `shared_hit` and `shared_read` counts per query reveal the mechanism:
- `shared_hit` = page found in shared_buffers (fast, ~ns)
- `shared_read` = page read from OS/disk (slow, ~us-ms)

Example: Q12 under relaxed config:
- 0 ANN workers: 8K reads (data cached from prior queries)
- 5 ANN workers: 509K reads (ANN evicted the cached pages)
- 64x increase in disk reads due to ANN interference

---

## 11. Troubleshooting

### PostgreSQL won't start with large shared_buffers

Increase system shared memory limits:
```bash
sudo sysctl -w kernel.shmmax=42949672960    # 40GB
sudo sysctl -w kernel.shmall=10485760
```

### OOM kills under tight cgroup

If PostgreSQL gets killed under tight cgroup configs, check:
```bash
dmesg | grep -i oom
cat /sys/fs/cgroup/postgres_test/memory.events
```

Increase the cgroup limit or reduce `shared_buffers`.

### Q1 is slow (~70s) even under no_pressure

Check that `enable_seqscan` is not globally `off` in postgresql.conf, or that the experiment script sets `SET enable_seqscan = on;` before running TPC-H queries. Without sequential scans, PostgreSQL uses bitmap index scans which are much slower for full-table queries.

### Permission denied errors

The experiment script must run as root to:
- Modify postgresql.conf
- Drop OS caches
- Set cgroup memory limits
- Restart PostgreSQL via `su - postgres`
