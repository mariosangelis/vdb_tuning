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

python3 -m venv .venv
source .venv/bin/activate
pip3 install psycopg2-binary h5py numpy psutil "psycopg[binary]" pgvector --break-system-packages
```
---

## 2. Prepare Storage and Create postgres User

Check your disk layout with `df -h` and `lsblk` to decide where to store PostgreSQL data.

**Option A: Dedicated NVMe drive** (e.g., CloudLab machines with a separate data disk)

```bash
# Only if you have a separate, unformatted NVMe device (e.g., /dev/nvme1n1)
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
# Optional: make persistent
echo '/dev/nvme1n1 /mnt/nvme ext4 defaults 0 0' | sudo tee -a /etc/fstab
```

**Option B: Single root disk** (e.g., AWS EC2 with one NVMe as root `/`)

```bash
# The root filesystem already uses the NVMe — just create the directory
sudo mkdir -p /mnt/nvme
```

Then create the postgres user and data directory:

```bash
sudo useradd -m -s /bin/bash postgres
# sgs-intel-02: /local/home/postgres/pgdata
sudo mkdir -p /mnt/nvme/pgdata 
sudo chown postgres:postgres /mnt/nvme/pgdata
```

---

## 3. Compile and Install PostgreSQL 18 
This Postgres version includes our patches for Compiler Optmization !!!

```bash
unset CC CXX
cd ~/vdb_tuning/postgres/postgres
rm -f config.cache
# Clean any stale build artifacts (important if source was copied from another arch)
make distclean 2>/dev/null

# Configure with debug flags
./configure --enable-debug --enable-cassert CFLAGS="-O1 -fno-omit-frame-pointer"

# Build and install
make -j$(nproc)
sudo make install

# Add to PATH (add to ~/.bashrc for persistence)
echo "export PATH=/usr/local/pgsql/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
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

The modified pgvector extension that includes the parallel IVFFLAT implementation
```bash
cd ~/vdb_tuning/postgres/postgres/pgvector_modified_final

export PATH=/usr/local/pgsql/bin:$PATH
export PG_CONFIG=/usr/local/pgsql/bin/pg_config
make clean
make CFLAGS='-O2 -fno-omit-frame-pointer -g'
sudo make install PG_CONFIG=/usr/local/pgsql/bin/pg_config
```
### 4.1 (skip) Default pgvector
The default pgvector extension
```bash
cd ~/vdb_tuning/postgres/postgres/pgvector

export PATH=/usr/local/pgsql/bin:$PATH
export PG_CONFIG=/usr/local/pgsql/bin/pg_config
make clean
make CFLAGS='-O2 -fno-omit-frame-pointer -g'
sudo make install PG_CONFIG=/usr/local/pgsql/bin/pg_config
```

The Makefile applies these optimization flags automatically:
- `-march=native` (uses AVX2/AVX-512 on x86, SVE/NEON on ARM)
- `-ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math`


---

## 5. Initialize and Start PostgreSQL

```bash
sudo bash 

# Switch to postgres user
sudo su - postgres -s /bin/bash


# Set environment
export PGDATA=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH
## Set enviroment on sgs-intel-02:
echo "export PGDATA=/local/home/postgres/pgdata" >> ~/.bashrc
echo "export PATH=/usr/local/pgsql/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

# Initialize the database
initdb -D $PGDATA --encoding=UTF8 --locale=en_US.UTF-8
```

**Important:** Before starting the server, copy the pre-configured postgresql.conf:

```bash
# Exit back to your regular user (exit the postgres shell)
exit

# Copy the experiment configuration files (overwrites the default postgresql.conf)
sudo cp ~/vdb_tuning/postgres_configurations/postgresql.conf /mnt/nvme/pgdata/postgresql.conf
sudo chown postgres:postgres /mnt/nvme/pgdata/postgresql.conf
## sgs-intel-02
sudo cp postgres_configurations/postgresql.conf /local/home/postgres/pgdata/postgresql.conf
sudo chown postgres:postgres /local/home/postgres/pgdata/postgresql.conf
```

The `postgresql.conf` in the repo has tuned settings for the experiments (shared_buffers,
work_mem, effective_io_concurrency, enable_seqscan=off for vector queries, etc.).
A `postgresql_default.conf` is also provided for reference.

Now start the server and create the database:

```bash
# Switch back to postgres user
sudo su - postgres -s /bin/bash
export PGDATA=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH

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

The database dump contains all TPC-H SF10 tables and GIST-960 vector data (~4.8GB compressed).
Download it from MediaFire and restore with `pg_restore`.

### 6.1 Download the Database Dump

```bash
# Create the dump directory
mkdir -p /tmp/tpch_sf10
cd /tmp/tpch_sf10

# Download all dump files from MediaFire
# (uses the premium/direct links which redirect automatically)
wget -O 4685.dat.gz "https://www.mediafire.com/file_premium/59gidu10ljsv8et/4685.dat.gz/file"
wget -O 4686.dat.gz "https://www.mediafire.com/file_premium/va2ljstm5fpbmun/4686.dat.gz/file"
wget -O 4687.dat.gz "https://www.mediafire.com/file_premium/15fjmnipz05xndf/4687.dat.gz/file"
wget -O 4688.dat.gz "https://www.mediafire.com/file_premium/402k0wcnt4ew758/4688.dat.gz/file"
wget -O 4689.dat.gz "https://www.mediafire.com/file_premium/bnxogj4y6ghviu3/4689.dat.gz/file"
wget -O 4690.dat.gz "https://www.mediafire.com/file_premium/qje6m3xr4gxupqz/4690.dat.gz/file"
wget -O 4691.dat.gz "https://www.mediafire.com/file_premium/y1h3sarinsfa79n/4691.dat.gz/file"
wget -O 4692.dat.gz "https://www.mediafire.com/file_premium/k3qawdhhg4aknd1/4692.dat.gz/file"
wget -O 4693.dat.gz "https://www.mediafire.com/file_premium/3617jfx3vt731mx/4693.dat.gz/file"
wget -O toc.dat    "https://www.mediafire.com/file_premium/af566ev4qyhjxn6/toc.dat/file"

# Verify all files are present (should be 10 files, ~4.8GB total)
ls -la /tmp/tpch_sf10/
```

### 6.2 Restore the Database

```bash
# Fix permissions so postgres user can read the dump
chmod -R a+r /tmp/tpch_sf10
chmod a+rx /tmp/tpch_sf10

# Restore as postgres user (parallel with 8 jobs)
export PATH=/usr/local/pgsql/bin:\$PATH; pg_restore -j 8 -d mydb /tmp/tpch_sf10
# or
sudo su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH; pg_restore -j 8 -d mydb /tmp/tpch_sf10"

# check
psql -d mydb -c "\dt+"
psql -d mydb -c "SELECT count(*) FROM gist_data;" # gist 1M
psql -d mydb -c "SELECT count(*) FROM lineitem;" # if sf10: ~60M
psql -d mydb -c "\d gist_data" # is there an index
```

This restores all TPC-H tables (lineitem, orders, customer, etc.), the GIST data table (`gist_data` with ~1M 960-dimensional vectors), and an IVFFLAT index with 1000 clusters generated for GIST dataset. Takes ~10-15 minutes.

### 6.3 Create Vector Index [Optional]

Optional since tph_sf10 folder already includes an IVFFLAT index with 1000 centroids.


```sql
sudo bash
sudo su - postgres -s /bin/bash
export PGDATA=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH

psql -d mydb

-- IVFFlat index (1000 lists, ~2-3 min build time)
CREATE INDEX gist_data_embedding_idx ON gist_data USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000);

-- OR: HNSW index (M=16, ef_construction=200, ~20-30 min build time)
CREATE INDEX gist_data_embedding_idx ON gist_data USING hnsw (embedding vector_l2_ops);
```

**Note:** Only one index at a time. Drop the existing one before creating another:
```sql
DROP INDEX IF EXISTS gist_data_embedding_idx;
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

Under `vdb-tuning-main/TPCH/code/*`: Each experiment script (`execute_ann_impact_automated.py, execute_olap_impact_automated.py`) has these CPU settings at the top:

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

## 9. Experiment 1: ANN Impact on OLAP

This measures how increasing ANN (TopK) concurrency degrades TPC-H performance under
different memory pressures. Fixed: 1 OLAP worker. Variable: 0-10 ANN workers.

### 9.1 Running the Experiment

The script must be run **3 times** — once per index/IO configuration.

```bash
cd ~/vdb_tuning/vdb-tuning-main/TPCH/code
```

Before each run, edit the two configuration variables at the top of `execute_ann_impact_automated.py`:

```python
INDEX_TYPE = "IVFFLAT"   # or "HNSW"
IO_DEPTH = 1000          # 0 or 1000
```

The script automatically sets `effective_io_concurrency` in `postgresql.conf` to match
`IO_DEPTH` before each PostgreSQL restart, so you only need to change these two variables.

#### Run 1: IVFFlat with io_depth=1000

The database dump already includes an IVFFlat index. Set:

```python
INDEX_TYPE = "IVFFLAT"
IO_DEPTH = 1000
```

```bash
sudo python3 execute_ann_impact_automated.py
```

#### Run 2: IVFFlat with io_depth=0

Change only `IO_DEPTH`:

```python
INDEX_TYPE = "IVFFLAT"
IO_DEPTH = 0
```

```bash
sudo python3 execute_ann_impact_automated.py
```

#### Run 3: HNSW with io_depth=1000

First, switch the index from IVFFlat to HNSW:

```bash
sudo su - postgres -s /bin/bash
export PGDATA=/mnt/nvme/pgdata
export PATH=/usr/local/pgsql/bin:$PATH

psql -d mydb
-- Drop the IVFFlat index
DROP INDEX IF EXISTS gist_data_embedding_idx;

-- Create HNSW index (M=16 takes ~20-30 min)
CREATE INDEX gist_data_embedding_idx ON gist_data USING hnsw (embedding vector_l2_ops);
```

Then set:

```python
INDEX_TYPE = "HNSW"
IO_DEPTH = 1000
```

```bash
sudo python3 execute_ann_impact_automated.py
```

### 9.2 What the Script Does

For each `(memory_config, ann_workers)` combination:

1. Sets `shared_buffers` in postgresql.conf
2. Sets `effective_io_concurrency` in postgresql.conf (matching `IO_DEPTH`)
3. Sets cgroup `memory.max`
4. Drops OS page caches (`echo 3 > /proc/sys/vm/drop_caches`)
5. Restarts PostgreSQL with NUMA-aware CPU pinning
6. Moves PostgreSQL processes into the cgroup
7. Starts N ANN worker threads running continuous TopK vector queries
8. Runs all 22 TPC-H queries with `EXPLAIN (ANALYZE, BUFFERS)` to capture buffer statistics
9. Records per-query timing, shared_hit, shared_read counts
10. Saves results as JSON lines to `results_ann_impact_to_olap/`

### 9.3 Generating Plots

Once all 3 runs are complete, generate the paper-style figures:

```bash
cd ~/vdb_tuning/vdb-tuning-main/TPCH/figures
python3 generate_plot_for_ann_to_olap_impact.py
```

The script reads the JSON-lines result files from `results_ann_impact_to_olap/` and
produces figures in `figure_plots/`, including:

- **ANN impact on OLAP frontier** — OLAP execution time vs vector search throughput (4 memory panels)
- **TPC-H time vs ANN workers** — bar charts per memory config
- **Buffer pool misses** — shared_read counts vs ANN workers
- **Per-query breakdown** — individual TPC-H query times with 0 vs N ANN workers
- **Throughput frontier** — OLAP throughput (tps) vs ANN throughput (qps)

Figures are saved as both PDF and PNG.

---

## 10. Experiment 2: OLAP Impact on ANN

This measures the opposite direction — how increasing OLAP (TPC-H) concurrency degrades
ANN (TopK) performance. Fixed: 1 ANN worker. Variable: 0-6 OLAP workers.

### 10.1 Running the Experiment

The script must be run **3 times** — same index/IO combinations as Experiment 1.

```bash
cd ~/vdb_tuning/vdb-tuning-main/TPCH/code
```

Before each run, edit the two configuration variables at the top of `execute_olap_impact_automated.py`:

```python
INDEX_TYPE = "IVFFLAT"   # or "HNSW"
IO_DEPTH = 1000          # 0 or 1000
```

#### Run 1: IVFFlat with io_depth=1000

```python
INDEX_TYPE = "IVFFLAT"
IO_DEPTH = 1000
```

```bash
sudo python3 execute_olap_impact_automated.py
```

#### Run 2: IVFFlat with io_depth=0

```python
INDEX_TYPE = "IVFFLAT"
IO_DEPTH = 0
```

```bash
sudo python3 execute_olap_impact_automated.py
```

#### Run 3: HNSW with io_depth=1000

Make sure the HNSW index is already created (see Run 3 in section 9.1). Then:

```python
INDEX_TYPE = "HNSW"
IO_DEPTH = 1000
```

```bash
sudo python3 execute_olap_impact_automated.py
```

### 10.2 What the Script Does

For each `(memory_config, olap_workers)` combination:

1. Restarts PostgreSQL (clean state, drop caches, set shared_buffers/effective_io_concurrency/cgroup)
2. Starts 1 ANN (TopK) worker thread running continuous vector queries
3. Starts N OLAP worker threads, each running 1 full round of all 22 TPC-H queries
4. Waits until all OLAP workers finish their round, then stops the ANN worker
5. Records ANN throughput/latency (p50/p95/p99) and OLAP throughput (tps)
6. For `0 OLAP workers` (baseline), TopK runs solo for 180 seconds
7. Saves results as JSON lines to `results_olap_impact_to_ann/`

Each data point represents a different OLAP concurrency level, producing the throughput
frontier plots (OLAP throughput vs Vector Search throughput).

### 10.3 Generating Plots

Once all 3 runs are complete, generate the throughput frontier figures:

```bash
cd ~/vdb_tuning/vdb-tuning-main/TPCH/figures
python3 generate_plot_for_olap_to_ann_impact.py
```

The script reads the JSON-lines result files from `results_olap_impact_to_ann/` and
produces a 2x2 grid throughput frontier plot in `figure_plots/`:

- **Throughput frontier grid** — OLAP throughput (tps) vs Vector Search throughput (qps), with each point labeled by OLAP worker count (1-6), across 4 memory pressure panels

Figures are saved as both PDF and PNG.

---

## 11. Results Directory Structure

After running all experiments, the results are stored under:

```
vdb-tuning-main/TPCH/
├── results_ann_impact_to_olap/    # Experiment 1: ANN→OLAP interference
│   ├── mixed_workload_..._IVFFLAT_io_1000_..._GIST.txt
│   ├── mixed_workload_..._IVFFLAT_io_0_..._GIST.txt
│   └── mixed_workload_..._HNSW_io_1000_..._GIST.txt
├── results_olap_impact_to_ann/    # Experiment 2: OLAP→ANN interference
│   ├── olap_impact_on_ann_IVFFLAT_io_1000_..._GIST.txt
│   ├── olap_impact_on_ann_IVFFLAT_io_0_..._GIST.txt
│   └── olap_impact_on_ann_HNSW_io_1000_..._GIST.txt
└── figures/
    ├── generate_plot_for_ann_to_olap_impact.py    # Plots for Experiment 1
    ├── generate_plot_for_olap_to_ann_impact.py    # Plots for Experiment 2
    └── figure_plots/                              # Generated figures
```

---
