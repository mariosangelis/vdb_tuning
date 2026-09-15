# CLAUDE.md — session knowledge & modifications log

This file documents everything set up / modified to make the `vdb_tuning`
experiments build and run inside a container on a fresh VM (validated on AWS
Graviton / **arm64**, designed to also work on **x86_64** / CloudLab). It is
meant so a future session (or a new engineer) can pick up with full context.

---

## 1. What this project is

Research framework studying **OLAP (TPC-H) vs ANN (pgvector vector search)
workload interference** in a **custom-built PostgreSQL 18** with a **modified
pgvector** (parallel IVFFlat) and a **top-K executor optimization**.

PostgreSQL is **built from the in-repo source tree** (`postgres/postgres/`), NOT
from a package — because the source contains custom modifications:
- `postgres/postgres/src/backend/executor/execExpr.c` — `enable_topk_search_optimizations`
  flag (top-K heap optimization; prints "TOPK OPTIMIZATION TRIGGERED!").
- `postgres/postgres/pgvector_modified_final/` — parallel IVFFlat fork (v0.8.0),
  with `#define NUM_WORKERS` in `src/ivfscan.c`.
- `postgres/postgres/pgvector/` — stock pgvector (v0.8.1), used as the default baseline.

Datasets: **TPC-H SF10** (~60M lineitem rows) + **GIST-960** (1M × 960-dim vectors,
loaded into table `gist_data`). Query workload: `gist_test_queries.txt` (1000 vectors).

---

## 2. The containerized build system (everything is under `build/`)

A clean 3-phase flow was created in `build/`. Read `build/README.md` for usage.

**Phase 1 — `build/install.sh`** (run on HOST): installs Docker CE + compose
plugin (arch-aware), preps `/mnt/nvme` + `/mnt/nvme/pgsql`, builds the Ubuntu
image, starts the container, runs `apply-fixes.sh`, then `setup-all.sh`
(build PG + pgvector, load TPC-H SF10, download+load GIST-960, build base index).

**Phase 2 — `build/tune.sh`**: tunes IVFFlat / HNSW / parallel IVFFlat; writes
best-param JSONs + reusable PGDATA snapshots to `/mnt/nvme/saved_indexes/`.

**Phase 3 — `build/run-experiment.sh <category>`**: `recall-qps`,
`parallel-vs-default`, `topk-opt`, `tpch-interference`.

Entry points you actually invoke: `install.sh`, `tune.sh`, `run-experiment.sh`,
and (host, optional) `tune-host-os.sh`. Everything else in `build/` is machinery
those call.

### build/ file inventory
| File | Role |
|---|---|
| install.sh | Phase-1 orchestrator (host) |
| tune.sh | Phase-2 orchestrator |
| run-experiment.sh | Phase-3 dispatcher |
| tune-host-os.sh | HOST OS tuning (turbo/SMT/THP/governor/swap) — run once |
| partition-cores.sh | auto CPU-core split for TPCH frontier (NUMA-aware) |
| apply-fixes.sh | patches repo files for this environment (idempotent) |
| install-docker-host.sh | Docker CE + compose plugin (dnf or apt) |
| install-prerequisites.sh | verify/install build deps + venv (in container) |
| build-postgres.sh | build PG + pgvector, initdb, auth, extensions |
| load-tpch.sh | dbgen SF10 + load + indexes |
| load-gist.sh | download + load GIST-960 into gist_data |
| build-indexes.sh | base IVFFlat/HNSW index on gist_data |
| setup-all.sh | chains prerequisites→build→tpch→gist→indexes |
| _common.sh | shared config reader + helpers |
| Dockerfile | Ubuntu 24.04 image (build deps + venv + locales + pigz + pyyaml) |
| docker-compose.yml | container (privileged, 48G /dev/shm, bind mounts) |
| config.container.yaml | container-path config (selected via VDB_CONFIG) |
| README.md | usage docs |

---

## 3. Every fix made, and WHY (this is the important part)

These were discovered incrementally while getting the CloudLab-x86 repo to run
in a container on Graviton-arm64. All are encoded in `build/` so a fresh run
applies them automatically.

### Baked into the image (`build/Dockerfile`)
1. **UTF-8 locale** — the repo's `postgres_configurations/postgresql.conf` sets
   `lc_messages`/`lc_monetary`/etc. to `en_US.UTF-8`. Ubuntu ships only
   `C.UTF-8`, so PG failed to start ("invalid value for parameter lc_messages").
   Fix: `locale-gen en_US.UTF-8` + `ENV LANG/LC_ALL=en_US.UTF-8` in the Dockerfile.
2. **pigz** — snapshots are ~15GB (full PGDATA incl. TPC-H). Single-thread
   `tar -czf` was painfully slow; `pigz` (parallel gzip) makes save/load fast.
3. **pyyaml** — `config_loader.py` requires it; not in the base venv.

### Container settings (`build/docker-compose.yml`)
4. **`shm_size: 48gb`** — PostgreSQL **parallel index builds** (IVFFlat/HNSW)
   allocate their work area in `/dev/shm`. Docker defaults shm to 64MB; the HNSW
   build needed ~15GB and failed with "No space left on device" (that error means
   `/dev/shm`, NOT disk). `install.sh` auto-sizes this to ~75% of host RAM.
5. **Persist `/usr/local/pgsql`** — bind-mounted to host `/mnt/nvme/pgsql`.
   Without this, the compiled PostgreSQL lived only in the container layer and was
   LOST on every `docker compose up` recreate (e.g. after changing shm_size),
   forcing a full ~10-min rebuild. Now it survives recreates.
6. **`privileged: true` + `cgroup: host`** — the TPCH frontier writes cgroup v2
   `memory.max` and moves PG into `/sys/fs/cgroup/postgres_test`.

### Repo file patches (`build/apply-fixes.sh`, idempotent, keeps .orig on first change)
7. **`config.yaml`: `setup_nvme: true` → `false`** — the original would run
   `mkfs.ext4` on `/dev/nvme0n1` = **the root disk** = destroys the VM. Never
   reformat. `/mnt/nvme` is a directory (or a disk you mount yourself).
8. **`config.yaml`: source paths `/users/mangelis/...` → `/vdb_tuning/...`** —
   the TPCH bash scripts (`run_all_frontiers.sh`) read the top-level `config.yaml`
   DIRECTLY (they ignore `$VDB_CONFIG`), so its `pg_src`/`pgvector_*` must resolve
   inside the container. (Python scripts use `config.container.yaml` via VDB_CONFIG.)
9. **`experiment_config.json`: `postgres_numa_nodes` `0-7` → `0`** — this box has
   1 NUMA node; `numactl --membind=0-7` would fail. (`partition-cores.sh` also
   rewrites the core lists — see below.)
10. **`tune_ivfflat.py` & `tune_parallel_ivfflat.py`: `maintenance_work_mem`
    `400GB` → `15GB`, `max_parallel_*` `64` → `16`** — 400GB is impossible on a
    61GB box; the index build would fail.
11. **`pgvector_modified_final/src/vector.c`: guard `#include <immintrin.h>`** —
    the parallel fork had an UNCONDITIONAL x86 AVX include (`immintrin.h`), which
    does not exist on ARM → compile error `fatal error: immintrin.h: No such file`.
    Wrapped in `#if defined(__x86_64__) || defined(_M_AMD64)`. (The AVX kernels in
    `bitutils.c`/`halfutils.c` were already guarded behind `USE_DISPATCH`/
    `HALFVEC_DISPATCH`, which `halfvec.h` only defines for x86 — so on ARM they
    correctly fall back to scalar. Only the stray `vector.c` include was the bug.)

### PostgreSQL auth (in `build/build-postgres.sh`)
12. **postgres role password = `secret` + TCP md5 in `pg_hba.conf`** — the Python
    loaders/experiments connect `user=postgres host=localhost` over TCP. The
    original setup only had local trust. Added password + `host all all
    127.0.0.1/32 md5` + `listen_addresses='localhost'`.
13. **Idempotent server start** — detects an already-running server (reload
    instead of failing on "postmaster.pid already exists").

### Runtime, NOT a file edit
14. **`PGPASSWORD=secret` in the environment** — the experiment scripts have
    `DB_CONFIG` with NO password (they assumed local trust). psycopg2 honors the
    `PGPASSWORD` env var, so all experiments are launched with it exported. This
    is why the experiment `.py` files did NOT need editing for auth.

### Host OS tuning (`build/tune-host-os.sh`, run once on host)
15. Consolidated the repo's `apply_all_optimizations.sh` + `SETUP_SGS.md` steps
    (CPU governor→performance, turbo off, SMT/hyperthreading off, THP off, NUMA
    balancing off, C-states off, ≥10GB swap), made **portable** (Intel
    `intel_pstate/no_turbo` vs AMD `cpufreq/boost` vs ARM which has neither) and
    **idempotent** (probes each knob, skips if absent). On Graviton most are
    skipped (no turbo, no SMT — 1 thread/core); on x86 they engage.

### Core partitioning (`build/partition-cores.sh`)
16. The committed `experiment_config.json` targets a **64-core CloudLab box**:
    producers on cores 32-63, which DON'T EXIST on a 32-core VM →
    `os.sched_setaffinity` would crash. `partition-cores.sh` auto-generates the
    split from the actual machine: multi-NUMA → PG on node 0, producers on other
    node(s); single-NUMA → split cores by `--pg-frac` (default half/half; our box
    got PG 0-15, producers 16-31).

---

## 4. Experiments — what runs, and results obtained on ARM

Pipeline: **exact ground-truth (shipped in repo) → tune (build+snapshot) →
recall/QPS sweep / comparisons / frontier**.

- **Tuning** (all produced param JSONs + PGDATA snapshots in `/mnt/nvme/saved_indexes/`):
  - IVFFlat: best `lists=8192, probes=150` → recall 0.907, 26.8ms (build ~522s)
  - HNSW: best `m=32, ef_construction=256, ef_search=120` → recall 0.909, 6.9ms (build ~1155s, 7.7GB)
  - Parallel IVFFlat: `lists=8192` snapshot built (recompiles the parallel fork,
    edits `NUM_WORKERS` in `ivfscan.c`)
- **Recall/QPS (IVFFlat, warm):** 88.5 QPS @0.73 recall → 16.1 QPS @0.98. Textbook curve.
- **Parallel vs Default (warm):** parallel gives up to **3.9× speedup at recall
  0.98** (12 workers). Speedup grows with recall. NOTE: on ARM the per-vector AVX
  distance kernels fall back to scalar, but the parallelism is at the scan/worker
  level (arch-independent), so speedups are real and even exceeded the old x86
  numbers. Absolute numbers still differ from x86.
- **Top-K optimization (execExpr.c flag on/off):** ~2× speedup at mid/high K
  (e.g. K=100: 26.8ms with vs 65.4ms without). Toggling the flag recompiles PG.
- **TPCH frontier:** machinery VALIDATED (cgroup memory limit set, core pinning
  PG 0-15 / clients 16-31, snapshot load, real interference data produced) but NOT
  run to completion — the full grid is ~44-50h PER index phase (300s × 528 points
  × 3 indexes). It is designed for the bigger dedicated VM. Partial run was stopped
  and its output deleted; cgroup reset to `max`; execExpr.c flag restored to `true`.

### Frontier worker facts (important for sizing)
- OLAP workers ramp is HARDCODED in `find_saturation_points.py`
  (`OLAP_WORKERS_TO_TEST = [5,8,10,12,14,16,18,20,22,24]`, max 24) — NOT in
  `experiment_config.json` (config only controls cores + K + workload file).
- Each OLAP worker runs ONE TPC-H query at a time, but each query inherits
  `max_parallel_workers_per_gather = 4` from postgresql.conf → up to
  24×5 ≈ 120 backend processes. On 16 PG cores this over-subscribes; fits the
  intended big VM.

---

## 5. Environment specifics of the validation VM

- AWS EC2, **Amazon Linux 2023**, **aarch64 (Graviton)**, 32 cores (1 thread/core,
  no SMT), 61 GB RAM, 1 NUMA node, single 500GB NVMe mounted at `/` (no separate
  data disk — `/mnt/nvme` is a directory on root).
- No Docker/Podman/GPU initially. cgroup v2 active.
- Reached via jump host: `ssh -J ec2-user@<bastion> ec2-user@<private-ip>`.

### GitHub access (for clone/push)
The repo `git@github.com:mariosangelis/vdb_tuning.git` is accessed via the deploy
key **`~/.ssh/id_ed25519_vdb`** (authenticates as `mariosangelis/vdb_tuning`).
The forwarded agent authenticates as a DIFFERENT account (`marios-angelis`) which
canNOT see the repo → "Repository not found". To push/pull, force the key:
```bash
git config core.sshCommand "ssh -i ~/.ssh/id_ed25519_vdb -o IdentitiesOnly=yes"
```
(A deploy key needs "Allow write access" enabled to push.)

---

## 6. State at end of session (what's committed vs not)

- **Kept as modifications** (needed for the repo to run): `config.yaml`,
  `postgres/postgres/pgvector_modified_final/src/vector.c` (+ `ivfscan.c` left at
  last NUM_WORKERS value), `vdb-tuning-main/TPCH/code/experiment_config.json`,
  `tune_ivfflat.py`, `tune_parallel_ivfflat.py`.
- **All experiment result/data files were restored to their original committed
  (x86) values** — no ARM result files were left in the repo (the runs were for
  validation only).
- **`build/` is the new, self-contained build system** (was untracked; being
  committed).
- Container `vdb-tuning` is running; PG healthy; data intact (gist 1M, lineitem 60M);
  snapshots in `/mnt/nvme/saved_indexes/` (ivfflat/hnsw/parallel, ~15GB each).

---

## 7. IMPORTANT caveat for the next session

The `build/` pipeline encodes every fix, but a **single clean end-to-end
`install.sh` run from zero has NOT been executed** — it was built incrementally
as issues were found. Before trusting it on a fresh VM, do one cold validation:
tear down (`docker compose down`, wipe `/mnt/nvme`) and run `./install.sh` from
scratch, or test on a brand-new VM. Watch for: step ordering in install.sh, the
docker-group/sudo interaction, and that `apply-fixes.sh` runs before paths are used.
