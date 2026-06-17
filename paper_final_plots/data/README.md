# final_data — frozen raw data behind the paper figures

This directory is the **single source of truth for the raw data plotted in the
paper**. `scripts/paper_plots.py` reads from here by default (`--data_dir`).

It is a curated copy of the raw experiment outputs (the `.txt` / `.json` files
the plots consume — no rendered images). The working/scratch data lives
elsewhere under `data/`; only data that has been promoted to "final" is copied
here.

## RULE (keep in sync)

**Whenever the data behind a plot changes, update the copy here in the same
change.** A figure in `scripts/final_plots/` must always be reproducible from
the files in `data/final_data/`. If they drift, the figure is stale.

## Layout (mirrors what the script expects)

```
final_data/
├── avoid_reduntant_computations_optimization/data/   # plot1: results_{with,without}_optimization_*.txt
├── parallel_ivf/                                      # plot2: comparison_results_workers_gist_{cold,warm}_K_100.json
├── recall_qps_plots/results_recall_qps/               # plot3: recall_qps_{gist,deep}_*.json
├── throughput_frontier_10_15/{ivfflat,hnsw,parallel_ivfflat}/  # plot4: grid_graph.json + saturation_points_{ann,olap}.json
└── throughput_frontier_45_65/{ivfflat,hnsw}/          # plot5: same files, 45/65 scenario
```

Plots 6 and 7 (HTAP fixed-R / fixed-V line families, per index) read the **same**
`throughput_frontier_{10_15,45_65}/` grids as plots 4/5 — no extra data. plot6 ↔
plot4 (10/15), plot7 ↔ plot5 (45/65).

Plots 4 and 5 are the same throughput frontier at two memory scenarios (same
plotting code, different source dir):

- **`throughput_frontier_10_15/`** (plot4) — warm, memory-constrained: 10GB
  shared buffers / 15GB cgroup. All three indexes.
- **`throughput_frontier_45_65/`** (plot5) — warm: 45GB shared buffers / 65GB
  cgroup. At SF=10 the dataset fits in memory, so this is effectively
  unconstrained. `ivfflat` + `hnsw` only; `parallel_ivfflat` was still being
  collected when this snapshot was taken, so plot5 omits it.

Both were copied from `data/17-june/` (raw results pulled from
`vdb_tuning@daa66f4`) and renamed per index.

`scripts/final_plots/SOURCE.txt` records, per run, exactly which files here fed
each figure.

## Promotions

- **2026-06-17 (latest) — 45/65 HNSW grid refreshed from `vdb_tuning@4e71705`.**
  Re-copied `throughput_frontier_45_65/hnsw/grid_graph.json` from the upstream raw
  `vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_45GB_sb_65GB_cg/`.
  The run grew from 55 → **132 measurements** (last timestamp 19:12). Saturation
  files unchanged. Feeds plots 5 and 7 (45/65 HNSW). Plots regenerated.

- **2026-06-17 (later) — refreshed all frontier data from `vdb_tuning@f722866`.**
  Re-copied the 10/15 (ivfflat, hnsw, parallel_ivfflat) and 45/65 (ivfflat, hnsw)
  grids + saturation files from the latest upstream raw `results_frontier_*` dirs.
  The 45/65 HNSW run grew from 16 → **55 measurements** (ann_workers now
  {1,4,8,12}); still mid-run (more ANN levels to come). Feeds plots 4–7.

- **2026-06-17 — plots 4 & 5 (throughput frontier), two memory scenarios.**
  Split the single `throughput_frontier/` (old 10th-june 10/15 run) into
  `throughput_frontier_10_15/` (plot4) and added `throughput_frontier_45_65/`
  (plot5), both from `data/17-june/` (`vdb_tuning@daa66f4`). The 45/65 `hnsw`
  grid was still being collected at snapshot time (sparse, 16 points); its
  frontier shows a long diagonal to the lone pure-ANN saturation point. No
  `parallel_ivfflat` for 45/65 yet.

- **2026-06-12 — plot 2 (parallel) + plot 3 (recall/QPS), GIST.** The GIST
  `parallel_ivf/` files and the GIST `recall_qps_plots/` series (hnsw, ivfflat,
  parallel_ivfflat_w{4,8,12} × cold/warm) were replaced with the run from
  `vdb_tuning@21390ee`, staged in `data/12-june-26/`. Pre-promotion copies are
  in `data/final_data_backup_2026-06-12/`.
  - **GIST recall/QPS is now a sparse 6-point sweep** (recalls ~0.73, 0.82,
    0.91, 0.96, 0.97, 0.98), versus the previous dense ~21-point sweep. Plot 3
    GIST curves are coarser by design.
  - **`deep` was not re-run**, so the `deep` recall/QPS files are unchanged and
    still dense. Plot 3's `deep` variants therefore mix with the new sparse GIST
    main figure.
  - The stale generic `recall_qps_gist_parallel_ivfflat_{cold,warm}.json` (no
    worker suffix) are left in place but are **not read** by `paper_plots.py`.
