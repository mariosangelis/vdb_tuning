# vdb_tuning — containerized build & experiments

Clean, reproducible setup that runs everything inside an **Ubuntu 24.04**
container, on **x86_64 or arm64** (AWS Graviton, CloudLab AMD). Three phases:

1. **`install.sh`** — one-shot: Docker + container + build PostgreSQL 18
   (from the repo source) + pgvector, load TPC-H SF10, download + load
   GIST-960 (1M vectors), build the base IVFFlat index.
2. **`tune.sh`** — build + tune each index type (IVFFlat / HNSW / parallel
   IVFFlat); writes best-params JSON + reusable PGDATA snapshots.
3. **`run-experiment.sh`** — run a specific experiment category.

## Quick start (fresh machine)

```bash
cd vdb_tuning/build

# Phase 1: install everything (Docker, build, data). ~60-90 min; downloads ~1GB.
./install.sh
#   (Docker group change needs re-login, or install.sh uses sudo docker throughout.)

# Phase 2: tune the indexes (builds + snapshots each). ~30-45 min.
./tune.sh                      # gist, all three index types

# Phase 3: run experiments
./run-experiment.sh recall-qps --index ivfflat --mode warm
./run-experiment.sh recall-qps --index hnsw --mode warm
./run-experiment.sh parallel-vs-default --mode warm
./run-experiment.sh topk-opt
./run-experiment.sh tpch-interference
```

## Files

| File | Purpose |
|---|---|
| `install.sh` | **Phase 1** orchestrator (host → container → build → data) |
| `tune.sh` | **Phase 2** orchestrator (tune ivfflat/hnsw/parallel) |
| `run-experiment.sh` | **Phase 3** dispatcher (one category per invocation) |
| `apply-fixes.sh` | Patches repo files for this environment (called by install.sh) |
| `install-docker-host.sh` | Installs Docker CE + compose plugin (dnf or apt) |
| `install-prerequisites.sh` | Verify/install build deps + Python venv (in container) |
| `build-postgres.sh` | Build PG + pgvector, initdb, auth, extensions |
| `load-tpch.sh` | dbgen SF10 + load + indexes |
| `load-gist.sh` | Download + load GIST-960 into `gist_data` |
| `build-indexes.sh` | Base IVFFlat/HNSW index on `gist_data` |
| `setup-all.sh` | Runs prerequisites→build→tpch→gist→indexes (used by install.sh) |
| `Dockerfile` | Ubuntu 24.04 image (build deps, python venv, locales, pigz, pyyaml) |
| `docker-compose.yml` | Container: privileged, 48G /dev/shm, bind mounts |
| `config.container.yaml` | Container-path config (selected via `VDB_CONFIG`) |
| `_common.sh` | Shared config reader + helpers |

## What was fixed to run on this environment (baked in)

- **Never reformats a disk**: `config.yaml` `setup_nvme: false` (via `apply-fixes.sh`).
- **TCP auth**: `postgres`/`secret` + `pg_hba` md5 (in `build-postgres.sh`); experiment
  scripts are launched with `PGPASSWORD=secret`.
- **ARM portability**: the parallel pgvector's stray `#include <immintrin.h>` in
  `vector.c` is guarded to x86 (`apply-fixes.sh`); no `-march` forced.
- **Locale**: `en_US.UTF-8` generated in the image (repo's `postgresql.conf` needs it).
- **/dev/shm = 48G**: parallel index builds allocate their work area there
  (docker-compose `shm_size`), else HNSW/IVFFlat builds fail with "No space left".
- **Single NUMA node**: `experiment_config.json` `postgres_numa_nodes: "0"`.
- **Memory sanity**: tune scripts' `maintenance_work_mem` 400GB→15GB, workers 64→16.
- **Persistence**: `/usr/local/pgsql` bind-mounted to host `/mnt/nvme/pgsql`, so a
  container recreate does not lose the compiled PostgreSQL. Image bakes in
  locales/pigz/pyyaml so recreates keep all tooling.
- **Fast snapshots**: `pigz` installed (parallel gzip) for PGDATA snapshots.

## Fresh VM: is anything manual? (ARM or x86)

`install.sh` is designed to be run on a brand-new VM with **no manual mount or
config edits** in the common case. A `[0/6]` preflight handles the variability:

- **Architecture (ARM/x86):** auto. The `ubuntu:24.04` image is multi-arch, the
  Docker install detects arch, and `apply-fixes.sh` guards the x86-only SIMD.
- **Config paths:** auto. Inside the container everything uses
  `config.container.yaml` (via `VDB_CONFIG`), which has container paths. The
  repo's top-level `config.yaml` (CloudLab paths) is not used by the container;
  `apply-fixes.sh` still flips its `setup_nvme` to false for safety.
- **`/dev/shm` size:** auto. Preflight sizes it to ~75% of host RAM (capped 48G,
  min 4G) and rewrites `docker-compose.yml`, so parallel index builds work on
  any RAM size.
- **Mount point `/mnt/nvme`:** auto **if the root disk has enough space**
  (preflight requires >= 150 GB free: 33GB PGDATA + ~45GB snapshots + build).
  `/mnt/nvme` is created as a directory on the root filesystem and bind-mounted
  into the container.

**The only manual step, and only if you want it:** to put the data on a
**dedicated disk** (separate NVMe) instead of the root filesystem, mount that
disk at `/mnt/nvme` on the host **before** running `install.sh`:

```bash
# mount the dedicated disk at /mnt/nvme BEFORE running install.sh
sudo mount /dev/nvme1n1 /mnt/nvme
```

The preflight detects an existing mount and uses it as-is. If they don't care
(root disk >= 150 GB), they do nothing.

So the manager's flow on a fresh VM (ARM or x86):

```bash
cd vdb_tuning/build
cat README.md
./install.sh
./tune.sh
./run-experiment.sh recall-qps --index ivfflat --mode warm
```

## Benchmark-grade isolation (for the TPCH frontier experiment)

The TPCH interference experiment is timing-sensitive and pins Postgres vs client
threads to separate cores. Two host-side helpers prepare a VM for it. These are
**optional** for the recall/QPS and parallel-vs-default experiments, but
**recommended** before the TPCH frontier on the real (bigger) VM.

- **`tune-host-os.sh`** — run ONCE on the **host** (settings are host-kernel-wide
  and don't apply from inside a container; they reset on reboot):
  ```bash
  sudo ./tune-host-os.sh            # apply: governor, turbo off, SMT off, THP off,
                                    #        NUMA balancing off, C-states off, swap
  sudo ./tune-host-os.sh --verify   # just print current state
  ```
  Portable + idempotent: it probes each knob and **skips** whatever is absent.
  On AWS Graviton (ARM) there is no turbo and no SMT, so those steps report
  "not present" — expected. On Intel/AMD they engage.

- **`partition-cores.sh`** — auto-generates the CPU split in
  `experiment_config.json` from the machine's actual CPUs/NUMA layout, so it
  adapts to any VM size (no manual editing of core lists):
  ```bash
  ./partition-cores.sh              # single-NUMA: PG = first half, producers = rest
  ./partition-cores.sh --pg-frac 0.75
  ```
  - Multi-NUMA host: PostgreSQL -> NUMA node 0, producers -> the other node(s).
  - Single-NUMA host: split that node's cores by `--pg-frac` (default 0.5).

  (The committed config targets a 64-core CloudLab box — producers on cores
  32-63 — which would crash on a 32-core VM. This helper fixes that automatically.)

## Data & persistence

Bind mounts (survive container recreate):
- repo → `/vdb_tuning`
- host `/mnt/nvme` → `/mnt/nvme` (PGDATA + `saved_indexes/` snapshots)
- host `/mnt/nvme/pgsql` → `/usr/local/pgsql` (compiled Postgres install)

## Notes

- Everything is idempotent; re-running skips completed work. Tune scripts reuse an
  existing snapshot if present.
- On **arm64**, the parallel pgvector's hand-vectorized distance kernels fall back
  to scalar (AVX is x86-only), so parallel-vs-default *speedup* numbers differ from
  the x86 paper; correctness and recall are unaffected.
- Connect: `sudo docker exec -it vdb-tuning psql -h 127.0.0.1 -U postgres -d mydb`
  (password `secret`).
- Logs: `~/vdb_tuning/build/logs/` on the host.
