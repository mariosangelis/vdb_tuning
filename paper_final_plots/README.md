# paper_final_plots — figure bundle for the VecDB-pgtune paper

Self-contained bundle to regenerate every figure in the paper from frozen data.
Copied here (2026-06-17) from the paper's wrapper repo so the figures live next
to the experiment code on this branch.

## Contents

```
paper_final_plots/
├── paper_plots.py     # single generator: one make_plotN() per figure
├── data/              # frozen source data the figures consume (one dir per plot)
│   ├── avoid_reduntant_computations_optimization/  # plot1
│   ├── parallel_ivf/                               # plot2
│   ├── recall_qps_plots/                           # plot3
│   ├── throughput_frontier_10_15/                  # plot4, plot6 (10GB sb / 15GB cg)
│   └── throughput_frontier_45_65/                  # plot5, plot7 (45GB sb / 65GB cg)
└── final_plots/       # generated output: pdf (vector) + png (preview) + csv/ + alt/ variants
```

## Regenerate

From inside this folder:

```bash
python3 paper_plots.py --data_dir data --out_dir final_plots          # all plots
python3 paper_plots.py --data_dir data --out_dir final_plots --plots 4,5   # subset
```

`--data_dir data` points the script at this bundle's own data (its built-in
default expects the wrapper repo layout). Each main figure also writes an
auditable `final_plots/csv/plot_N.csv`; `final_plots/SOURCE.txt` records which
data file fed each figure.

## Plot ↔ meaning

| plot | figure | memory |
|------|--------|--------|
| 1 | redundant-computation optimization | — |
| 2 | parallel IVF | — |
| 3 | recall vs QPS | — |
| 4 | interference throughput frontier | 10GB sb / 15GB cg (constrained) |
| 5 | interference throughput frontier | 45GB sb / 65GB cg (fits in memory) |
| 6 | HTAP fixed-R / fixed-V line families, per index (→ plot4) | 10/15 |
| 7 | HTAP fixed-R / fixed-V line families, per index (→ plot5) | 45/65 |

All throughput-frontier runs are GIST + TPC-H SF-10 on NVMe.

## Caveats (carried over from the source data)

- The 45/65 runs are mid-collection; HNSW in particular has only a partial set of
  ANN-worker levels, so plot5/plot7 will fill in on future data refreshes.
- The per-index pure-ANN saturation endpoint on the frontier can come from a
  separate (earlier) saturation run than the grid, so the ANN-axis endpoint and
  the grid are not always from the same system state. Treat the absolute ANN
  peak with care until saturation and grid are re-measured together.
