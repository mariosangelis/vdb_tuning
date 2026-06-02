# Parallel IVFFlat vs Default IVFFlat Comparison

Benchmarks comparing the default pgvector IVFFlat scan against a parallelized
implementation with configurable worker thread counts.

## Scripts

### `run_comparison_workers.py`

Main benchmark script. Compares default pgvector vs parallel pgvector (4, 8, 12 workers)
across multiple recall targets.

```bash
# Cold mode: restarts PostgreSQL + drops OS page cache before each measurement
sudo python3 run_comparison_workers.py --dataset gist --mode cold

# Warm mode: prewarmes index+table into shared buffers, no restart between probes
sudo python3 run_comparison_workers.py --dataset gist --mode warm

# Skip default (only run parallel variants):
sudo python3 run_comparison_workers.py --dataset gist --mode cold --skip-default

# Reuse existing index (don't load snapshot or rebuild):
sudo python3 run_comparison_workers.py --dataset deep --mode warm --no-rebuild
```

**What it does for each implementation (default, parallel w4, w8, w12):**

1. Sets `NUM_WORKERS` in `ivfscan.c` (parallel only)
2. Compiles and installs the pgvector extension
3. Loads the PGDATA snapshot from `/mnt/nvme/saved_indexes/` (or rebuilds the index)
4. For each recall target (probes value):
   - **Cold mode**: restarts PostgreSQL, drops OS caches, then runs 1000 queries
   - **Warm mode**: keeps everything cached in shared buffers, runs 1000 queries
5. Saves results incrementally (can resume if interrupted)

**Inputs required:**
- `compute_and_compare_recall/results_from_tune_scripts/tune_ivfflat_results_{dataset}.json`
  (provides the optimal `lists` parameter)
- `compute_and_compare_recall/recall_qps_plots/results_recall_qps/recall_qps_{dataset}_ivfflat_cold.json`
  (provides the probe values for each recall target)
- `{dataset}_test_queries.txt` in the project root (1000 test query vectors)

**Output:**
- `comparison_results_workers_{dataset}_{mode}_K_100.json`

### `plot_comparison_workers.py`

Generates a Recall vs Latency plot from the benchmark results.

```bash
python3 plot_comparison_workers.py --dataset gist --mode cold
python3 plot_comparison_workers.py --dataset gist --mode warm
```

**Output:**
- `comparison_workers_{dataset}_{mode}_K_100.png`

## Cold vs Warm Mode

| | Cold | Warm |
|---|---|---|
| Before each probe | Restart PostgreSQL + `echo 3 > /proc/sys/vm/drop_caches` | Nothing |
| After index load | Just restart | `pg_prewarm` on index + table |
| Measures | Full I/O path (NVMe reads + computation) | Pure computation (everything in shared buffers) |
| Use case | Realistic worst-case latency | Best-case latency, isolates CPU parallelism benefit |

## Directory Structure

```
parallel_implementation_vs_default/
├── run_comparison_workers.py              # Main benchmark script
├── plot_comparison_workers.py             # Plot generator
├── comparison_results_workers_gist_cold_K_100.json   # Results (generated)
├── comparison_results_workers_gist_warm_K_100.json   # Results (generated)
├── comparison_workers_gist_cold_K_100.png            # Plot (generated)
└── README.md
```
