# Compute and Compare Recall

This directory contains scripts for tuning vector index parameters and measuring
recall-vs-QPS tradeoffs for pgvector indexes (IVFFlat, HNSW, Parallel IVFFlat).

## Workflow

### Step 1: Compute Exact Results (Ground Truth)

Before tuning, we need brute-force exact search results to measure recall against.

```bash
sudo python3 exact_results/compute_exact_results.py --dataset gist --k 100
sudo python3 exact_results/compute_exact_results.py --dataset deep --k 100
```

This runs sequential scans (no index) for all 1000 test queries and saves the
true top-K IDs. Only needs to be done once per dataset/K combination.

Output: `exact_results/exact_results_{dataset}/exact_search_results_1000_queries_{dataset}_K_{k}.json`

### Step 2: Tune Indexes (Find Best Build Parameters)

Tune each index type to find the build parameters that achieve a target recall
with the fastest query time.

```bash
# IVFFlat: finds best number of lists (centroids)
sudo python3 tuning/tune_ivfflat.py --dataset gist --recall 0.9

# HNSW: finds best m and ef_construction
sudo python3 tuning/tune_hnsw.py --dataset gist --recall 0.9

# Parallel IVFFlat: builds index with best lists from IVFFlat tuning, saves snapshot
sudo python3 tuning/tune_parallel_ivfflat.py --dataset gist
```

Output: `results_from_tune_scripts/tune_ivfflat_results_{dataset}.json`
        `results_from_tune_scripts/tune_hnsw_results_{dataset}.json`

These JSON files contain the best configuration (e.g., lists=8192 for gist IVFFlat)
and the probe/ef_search value needed to reach the target recall.

The tuning scripts also save PGDATA snapshots to `/mnt/nvme/saved_indexes/` so
indexes don't need to be rebuilt from scratch in future runs.

### Step 3: Measure Recall vs QPS (Runtime Parameter Sweep)

With the tuned indexes built and snapshotted, sweep runtime parameters
(probes for IVFFlat, ef_search for HNSW) to measure the full recall-vs-QPS curve.

```bash
# Warm cache (prewarm index into shared buffers):
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode warm --index ivfflat
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode warm --index hnsw
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode warm --index parallel_ivfflat

# Cold cache (restart + drop caches before each measurement):
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode cold --index ivfflat
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode cold --index hnsw
sudo python3 recall_qps_plots/recall_qps_diagram.py --dataset gist --mode cold --index parallel_ivfflat
```

Output: `recall_qps_plots/results_recall_qps/recall_qps_{dataset}_{index}_{mode}.json`

Each JSON contains the full probe/ef_search sweep with actual recall, QPS, and
latency at each point, plus a `recall_qps_points` array mapping each recall
target to the optimal runtime parameter.

### Step 4: Generate Plots

```bash
python3 recall_qps_plots/plot_recall_qps.py --dataset gist --mode warm
python3 recall_qps_plots/plot_recall_qps.py --dataset gist --mode cold
```

Output: `recall_qps_plots/recall_qps_{dataset}_{mode}.png`

### All-in-One Script

Run the entire pipeline for a dataset:

```bash
sudo bash tune_and_compute_recall_plots.sh gist
sudo bash tune_and_compute_recall_plots.sh deep
```

## Directory Structure

```
compute_and_compare_recall/
├── README.md
├── tune_and_compute_recall_plots.sh       # All-in-one pipeline script
├── exact_results/
│   ├── compute_exact_results.py           # Brute-force ground truth computation
│   ├── exact_results_gist/                # Cached exact results for GIST
│   └── exact_results_deep/                # Cached exact results for Deep
├── tuning/
│   ├── tune_ivfflat.py                    # Find best lists for IVFFlat
│   ├── tune_hnsw.py                       # Find best m, ef_construction for HNSW
│   ├── tune_parallel_ivfflat.py           # Build parallel index, save snapshot
│   └── tune_diskann.py                    # Tune DiskANN (pgvectorscale)
├── results_from_tune_scripts/
│   ├── tune_ivfflat_results_gist.json     # Best IVFFlat config for GIST
│   ├── tune_ivfflat_results_deep.json     # Best IVFFlat config for Deep
│   ├── tune_hnsw_results_gist.json        # Best HNSW config for GIST
│   └── tune_hnsw_results_deep.json        # Best HNSW config for Deep
└── recall_qps_plots/
    ├── recall_qps_diagram.py              # Sweep probes/ef_search, measure recall+QPS
    ├── plot_recall_qps.py                 # Generate recall-vs-QPS plots
    └── results_recall_qps/                # JSON results per index/dataset/mode
```

## Supported Datasets

| Dataset | Table | Dimensions | Distance | Queries File |
|---------|-------|-----------|----------|--------------|
| gist    | gist_data | 960 | L2 (`<->`) | gist_test_queries.txt |
| deep    | deep_data | 96  | Cosine (`<=>`) | deep_test_queries.txt |
| sift    | sift_data | 128 | L2 (`<->`) | sift_test_queries.txt |

Query files are located in the project root (`vdb-tuning-main/`).
