#!/usr/bin/env python3
"""
Plot all OLAP combinations for a single fixed ANN-worker value, from a frontier
grid_graph.json.

Workflow:
  1. You give the index type (ivfflat / hnsw / parallel_ivfflat). The script maps
     it to the correct results_frontier_* folder and loads grid_graph.json.
  2. It lists the ANN-worker values that exist in the grid.
  3. You pick one interactively.
  4. It plots every measured OLAP combination for that ANN value:
       X axis = ANN throughput (qps)
       Y axis = OLAP throughput (qps)
     Each point is annotated with its olap_workers count.
  5. Saves a PNG (headless-safe; no display needed).

Usage:
    python3 plot_ann_slice.py --index ivfflat
    python3 plot_ann_slice.py --index hnsw --mem 10:15
    python3 plot_ann_slice.py --index ivfflat --ann 20        # skip prompt
    python3 plot_ann_slice.py --grid /path/to/grid_graph.json # explicit file

Options:
    --index TYPE   ivfflat (default) | hnsw | parallel_ivfflat
    --mem SB:CG    memory tag used in the results dir name (default 10:15)
    --ann N        ANN worker value to plot (skips the interactive prompt)
    --grid PATH    use an explicit grid_graph.json instead of auto-locating
    --out PATH     output PNG path (default: figures/ann_slice_<index>_ann<N>.png)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / SSH-safe
import matplotlib.pyplot as plt


# ---- locate the TPCH dir relative to this script ----
SCRIPT_DIR = Path(__file__).resolve().parent          # frontier_analysis/
TPCH_DIR = SCRIPT_DIR.parent.parent                   # TPCH/
FIGURES_DIR = TPCH_DIR / "figures"


def results_dir_for(index_type, mem):
    """Map index type + memory tag to the results_frontier_* directory name.

    Naming convention (from build_throughput_frontier.py):
      ivfflat:          results_frontier_warm_constrained_<SB>GB_sb_<CG>GB_cg
      hnsw:             results_frontier_hnsw_warm_constrained_<SB>GB_sb_<CG>GB_cg
      parallel_ivfflat: results_frontier_pivfflat_warm_constrained_<SB>GB_sb_<CG>GB_cg
    """
    sb, cg = mem.split(":")
    tag = f"warm_constrained_{sb}GB_sb_{cg}GB_cg"
    if index_type == "ivfflat":
        name = f"results_frontier_{tag}"
    elif index_type == "hnsw":
        name = f"results_frontier_hnsw_{tag}"
    elif index_type == "parallel_ivfflat":
        name = f"results_frontier_pivfflat_{tag}"
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    return TPCH_DIR / name


def load_grid(grid_path):
    with open(grid_path) as f:
        data = json.load(f)
    measurements = data.get("measurements", [])
    if not measurements:
        print(f"ERROR: no measurements found in {grid_path}")
        sys.exit(1)
    return data, measurements


def main():
    ap = argparse.ArgumentParser(description="Plot OLAP combinations for a fixed ANN value.")
    ap.add_argument("--index", default="ivfflat",
                    choices=["ivfflat", "hnsw", "parallel_ivfflat"])
    ap.add_argument("--mem", default="10:15", help="memory tag SB:CG (default 10:15)")
    ap.add_argument("--ann", type=int, default=None, help="ANN worker value (skip prompt)")
    ap.add_argument("--grid", default=None, help="explicit path to grid_graph.json")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    # Locate grid_graph.json
    if args.grid:
        grid_path = Path(args.grid)
    else:
        rdir = results_dir_for(args.index, args.mem)
        grid_path = rdir / "grid_graph.json"

    if not grid_path.exists():
        print(f"ERROR: grid file not found: {grid_path}")
        # be helpful: show what frontier dirs do exist
        print("\nAvailable results_frontier_* dirs:")
        for p in sorted(TPCH_DIR.glob("results_frontier_*")):
            has = (p / "grid_graph.json").exists()
            print(f"  {p.name}   {'[grid_graph.json present]' if has else '[no grid]'}")
        sys.exit(1)

    data, measurements = load_grid(grid_path)
    cfg = data.get("config", {})
    print(f"Loaded: {grid_path}")
    print(f"  index_type = {cfg.get('index_type')}, tau_max = {cfg.get('tau_max')}, "
          f"alpha_max = {cfg.get('alpha_max')}, points = {len(measurements)}")

    # Distinct ANN values present
    ann_values = sorted(set(m["ann_workers"] for m in measurements))
    print(f"\nANN worker values present in the grid ({len(ann_values)}):")
    print(f"  {ann_values}")

    # Pick the ANN value
    if args.ann is not None:
        ann_pick = args.ann
    else:
        try:
            raw = input("\nEnter the ANN worker value to plot: ").strip()
            ann_pick = int(raw)
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nNo valid value entered. Exiting.")
            sys.exit(1)

    if ann_pick not in ann_values:
        print(f"ERROR: ANN value {ann_pick} not in grid. Choose from: {ann_values}")
        sys.exit(1)

    # Collect all OLAP combinations for that ANN value
    slice_pts = [m for m in measurements if m["ann_workers"] == ann_pick]
    # sort by olap_workers so the connecting line is monotone in load
    slice_pts.sort(key=lambda m: m["olap_workers"])

    if not slice_pts:
        print(f"ERROR: no points for ann_workers={ann_pick}")
        sys.exit(1)

    xs = [m["ann_throughput_qps"] for m in slice_pts]     # X = ANN throughput
    ys = [m["olap_throughput_qps"] for m in slice_pts]    # Y = OLAP throughput
    olap_w = [m["olap_workers"] for m in slice_pts]

    print(f"\nPlotting {len(slice_pts)} OLAP combinations for ANN={ann_pick} workers:")
    for m in slice_pts:
        print(f"  olap_workers={m['olap_workers']:>3}  "
              f"ANN={m['ann_throughput_qps']:>8.2f} qps  "
              f"OLAP={m['olap_throughput_qps']:>8.4f} qps")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, "-o", color="tab:blue", markersize=7, linewidth=1.5, zorder=2)

    # annotate each point with its olap_workers count
    for x, y, w in zip(xs, ys, olap_w):
        ax.annotate(f"olap={w}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="dimgray")

    ax.set_xlabel("ANN throughput (qps)")
    ax.set_ylabel("OLAP throughput (qps)")
    ax.set_title(f"{cfg.get('index_type', args.index)} | fixed ANN = {ann_pick} workers\n"
                 f"(each point = one OLAP-worker level; mem={args.mem})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(0.1)

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"ann_slice_{cfg.get('index_type', args.index)}_ann{ann_pick}_mem{args.mem.replace(':','-')}.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
