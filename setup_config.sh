#!/bin/bash
# ============================================================================
# Central machine-specific configuration for the vdb_tuning setup scripts.
#
# EDIT THIS FILE per machine, then run setup_postgres.sh / setup_tpch.sh.
# Both scripts `source` this file, so all machine-specific paths/flags live here.
#
# This config is for the SHELL setup scripts only. The Python frontier code
# uses TPCH/code/experiment_config.json separately (cores / index params).
# ============================================================================

# ---- Project / source locations ----
# Root of the postgres source tree that gets compiled.
#   clnode265 (CloudLab):  /users/mangelis/vdb_tuning/postgres/postgres
#   vmageirakos box:       /local/home/vmageirakos/projects/vdb_tuning/postgres/postgres
PG_SRC=/users/mangelis/vdb_tuning/postgres/postgres

# pgvector source dirs (derived from PG_SRC; usually no need to change).
PGVECTOR_DEFAULT=${PG_SRC}/pgvector
PGVECTOR_PARALLEL=${PG_SRC}/pgvector_modified_final

# Where PostgreSQL gets installed (make install --prefix).
PG_INSTALL=/usr/local/pgsql

# PostgreSQL data directory (initdb target). Put it on fast storage.
#   clnode265:        /mnt/nvme/pgdata
#   vmageirakos box:  /local/home/postgres/pgdata
PG_DATA=/mnt/nvme/pgdata

# Repo postgresql.conf to copy into PG_DATA.
#   clnode265:        /users/mangelis/vdb_tuning/postgres_configurations/postgresql.conf
#   vmageirakos box:  /local/home/vmageirakos/projects/vdb_tuning/postgres_configurations/postgresql.conf
PG_CONF_SRC=/users/mangelis/vdb_tuning/postgres_configurations/postgresql.conf

# Database name created by the setup.
DB_NAME=mydb

# ---- configure flags ----
# Extra flags passed to ./configure (e.g. "--without-icu"). Leave empty if not needed.
PG_CONFIGURE_FLAGS="--without-icu"

# ---- NVMe / storage ----
# SETUP_NVME=true  -> mkfs + mount NVME_DEV at NVME_MOUNT (CloudLab fresh node).
# SETUP_NVME=false -> skip; assume PG_DATA already lives on usable storage.
SETUP_NVME=true
NVME_DEV=/dev/nvme0n1
NVME_MOUNT=/mnt/nvme

# ---- TPC-H ----
# tpch-dbgen location and scale factor for setup_tpch.sh.
#   clnode265:        /users/mangelis/tpch-dbgen
#   vmageirakos box:  /local/home/vmageirakos/tpch-dbgen   (adjust)
DBGEN_DIR=/users/mangelis/tpch-dbgen
TPCH_SCALE=10
