# Experiment Setup

System's Group Machines:
- `sgs-intel-*`: 2x16 core, SSD (sata)
- `sgs-gpu-*`: 2x16 core, SSD (nvme)

# Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install psycopg2-binary h5py numpy psutil "psycopg[binary]" pgvector

# PostgreSQL + TPCH
sudo ./setup_postgres.sh
sudo ./setup_tpch.sh

# ANN ------
cd /local/home/vmageirakos/projects/vdb_tuning/vdb-tuning-main/data
# Gist :
wget http://ann-benchmarks.com/gist-960-euclidean.hdf5

python3 load_hdf5_to_postgres.py --file gist-960-euclidean.hdf5 --table gist_data

# Deep10M :
# TODO:
wget http://ann-benchmarks.com/deep-image-96-angular.hdf5
python3 load_hdf5_to_postgres.py --file deep-image-96-angular.hdf5 --table deep_data

```


# Build/Tune indexes
Builds a bunch of indexes, finds the "best" one at Target Recall (e.g. 90%)
- [] Q: does it matter if hnsw is build on the parallel postgres?
```bash
## Assume: Data already in the database
# Build/Tune Index
cd /local/home/vmageirakos/projects/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/tuning
./tune_indexes.sh

# Individually:
## IVFFlat
sudo python3 tune_ivfflat.py --dataset gist --warm 2>&1 | tee build_ivfflat_gist.txt
sudo python3 tune_ivfflat.py --dataset deep --warm 2>&1 | tee build_ivfflat_deep.txt
## Parallel IVFFlat
sudo python3 tune_parallel_ivfflat.py --dataset gist 2>&1 | tee build_ivfflat_parallel_gist.txt
sudo python3 tune_parallel_ivfflat.py --dataset deep 2>&1 | tee build_ivfflat_parallel_deep.txt
## HNSW
sudo python3 tune_hnsw.py --dataset gist --warm 2>&1 | tee build_hnsw_gist.txt
sudo python3 tune_hnsw.py --dataset deep --warm 2>&1 | tee build_hnsw_deep.txt
```


# Before experiments:

1. have 10G+ swapspace
``` bash
# check:
sudo swapon --show
free -h

# new 10G swapfile:
sudo fallocate -l 10G /swapfile
# secure to only root 
sudo chmod 600 /swapfile
# mark file as swap space
sudo mkswap /swapfile
# enable swapfile
sudo swapon /swapfile
# make change permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
#verify ( run check again/step 1)
```

2. disable hyperthreading
```bash
# disable
echo off | sudo tee /sys/devices/system/cpu/smt/control
# enable
echo on | sudo tee /sys/devices/system/cpu/smt/control
```

3. create cgroup
```bash
mkdir -p /sys/fs/cgroup/postgres_test
```

4. disable turboboost?
```bash
# Intel uses the intel_pstate driver. Note that for Intel, the file is called no_turbo, meaning you write 1 to disable it and 0 to enable it.

# disable
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# enable
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```


# Run Experiments

Manually play with the DB:
```bash
sudo su - postgres -s /bin/bash
psql -d mydb
```


## QPS - Recall
Assume: (.venv) is active <---

- [ ] double check this is the tuned config for DB
- def: cold/warm == where data starts from ( 1k query benchmark for gist, 10k for deep )

```bash
# build ground truth
cd /local/home/vmageirakos/projects/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/exact_results

sudo python3 compute_exact_results.py --dataset deep --k 100

# run QPS-Recall from "best" index "snapshot"
cd /local/home/vmageirakos/projects/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/recall_qps_plots

# ALL ( run & plot )

# Individually:
## ALL {ivfflat, parallel ivfflat, hnsw}
sudo python3 recall_qps_diagram.py --dataset gist --mode warm 2>&1 | tee qps_recall_warm.txt
sudo python3 recall_qps_diagram.py --dataset gist --mode cold 2>&1 | tee qps_recall_cold.txt

## warm
sudo python3 recall_qps_diagram.py --dataset gist --mode warm --index ivfflat 2>&1 | tee qps_recall_ivf.txt

# TODO: cold
# plot:



```

## Parallel IVF Speedup
```bash

```


## Redudant Computations Optimization
```bash

```


## If we have inconsistent state

```bash
pigz -dc /local/home/vmageirakos/projects/vdb_tuning/saved_indexes/gist_ivfflat_lists2048_pgdata.tar.gz | tar -xf - -C /local/home/postgres/pgdata
rm -f /local/home/postgres/pgdata/postmaster.pid
su - postgres -c "export PATH=/usr/local/pgsql/bin:\$PATH && pg_ctl -D /local/home/postgres/pgdata -l /tmp/logfile start"
```

echo 'export PATH=/usr/local/pgsql/bin:$PATH' >> ~/.bashrc
echo 'export PGDATA=/local/home/postgres/pgdata' >> ~/.bashrc
source ~/.bashrc

    pg_ctl restart
  pg_ctl status
  pg_ctl stop



# TODO:
- [] do the tune/or copy in intel-01
- [] qps/recall  ivf, parallel ivf, hnasw
- [] parallel ivf scalability
- [] duplicate elimination ivf/hnsw