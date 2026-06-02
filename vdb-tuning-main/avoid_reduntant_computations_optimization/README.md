# Compiler Optimization: Avoid Redundant Computations (Top-K Heap)

This experiment measures the performance gain of the top-K heap optimization
in PostgreSQL's executor. The optimization avoids redundant distance computations
during vector similarity search by maintaining a max-heap and skipping vectors
that can't possibly be in the top-K results.

## How It Works

The optimization is controlled by a flag in PostgreSQL source:
```c
// src/backend/executor/execExpr.c
bool enable_topk_search_optimizations = true;  // or false
```

The script toggles this flag, recompiles PostgreSQL, and measures execution
time for different K values (1, 10, 100, 200, 400, 800, 1000) at recall=0.9.

## Usage

```bash
# Run with optimization enabled (cold):
sudo python3 compute_optimization_results_for_different_K.py ivfflat with gist cold
sudo python3 compute_optimization_results_for_different_K.py hnsw with gist cold

# Run without optimization (cold):
sudo python3 compute_optimization_results_for_different_K.py ivfflat without gist cold
sudo python3 compute_optimization_results_for_different_K.py hnsw without gist cold

# Warm mode:
sudo python3 compute_optimization_results_for_different_K.py ivfflat with gist warm
sudo python3 compute_optimization_results_for_different_K.py ivfflat without gist warm

# Generate comparison plot:
python3 plot_optimization_results.py --dataset gist
```

## What the Script Does

1. Compiles the **default pgvector** extension (ensures parallel extension is not loaded)
2. Toggles `enable_topk_search_optimizations` in `execExpr.c` and recompiles PostgreSQL
3. Loads the appropriate PGDATA snapshot from `/mnt/nvme/saved_indexes/`
4. Finds optimal probes/ef_search for each K (recall >= 0.9):
   - Uses all 1000 queries (warm, full sweep)
   - Saves results to `data/optimal_params_{index}_{dataset}_recall_0.9.json`
   - On subsequent runs, loads from cache (delete the JSON to recompute)
5. For each K: measures total execution time with the optimal parameter
   - **Cold mode**: restarts PostgreSQL + drops OS caches before each K
   - **Warm mode**: prewarmes index+table once, no restart between K values

## Prerequisites

- Exact results for all K values must exist:
  `compute_and_compare_recall/exact_results/exact_results_{dataset}/exact_search_results_1000_queries_{dataset}_K_{k}.json`
  (Run `compute_exact_results.py --dataset gist --k <K>` for each K if missing)

- Tuning results must exist:
  `compute_and_compare_recall/results_from_tune_scripts/tune_{ivfflat|hnsw}_results_{dataset}.json`
  (Needed to find the correct snapshot)

- Index snapshots in `/mnt/nvme/saved_indexes/`

## Output

```
data/
├── results_with_optimization_IVFFLAT_{dataset}_{mode}.txt
├── results_without_optimization_IVFFLAT_{dataset}_{mode}.txt
├── results_with_optimization_HNSW_{dataset}_{mode}.txt
├── results_without_optimization_HNSW_{dataset}_{mode}.txt
├── optimal_params_ivfflat_{dataset}_recall_0.9.json    (cached, shared across runs)
├── optimal_params_hnsw_{dataset}_recall_0.9.json       (cached, shared across runs)
└── compiler_optimization_{dataset}.pdf
```

## Cold vs Warm Mode

| | Cold | Warm |
|---|---|---|
| Before each K | Restart PostgreSQL + drop OS caches | Nothing (data stays cached) |
| After snapshot load | Start postgres | Start postgres + pg_prewarm |
| Measures | Full I/O + computation | Pure computation (in-memory) |
| Use case | Realistic worst-case | Isolates CPU optimization benefit |
