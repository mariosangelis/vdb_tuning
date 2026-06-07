#!/usr/bin/env python3
"""
Plot Recall vs QPS diagrams from JSON result files.

Reads recall_qps result JSONs (produced by recall_qps_diagram.py) and generates
publication-quality plots showing recall-vs-QPS tradeoff curves for multiple
index types, including parallel IVFFlat with multiple worker counts.

Usage:
    python3 plot_recall_qps.py --dataset sift --mode warm
    python3 plot_recall_qps.py --dataset gist --mode cold
    python3 plot_recall_qps.py --dataset sift --mode warm --results-dir /path/to/results_recall_qps
    python3 plot_recall_qps.py --dataset sift --mode warm --output my_plot.png

The script looks for JSON files matching these patterns in the results directory:
    recall_qps_{dataset}_ivfflat_{mode}.json
    recall_qps_{dataset}_hnsw_{mode}.json
    recall_qps_{dataset}_parallel_ivfflat_{mode}.json         (legacy single-worker)
    recall_qps_{dataset}_parallel_ivfflat_w{N}_{mode}.json    (per-worker-count)

Output:
    recall_qps_{dataset}_{mode}.png
"""

import json
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# === DATASET DISPLAY INFO ===
DATASET_INFO = {
    "sift": {
        "name": "SIFT-128",
        "vectors": "1M vectors",
        "distance": "L2 Distance",
    },
    "gist": {
        "name": "GIST-960",
        "vectors": "1M vectors",
        "distance": "L2 Distance",
    },
    "deep": {
        "name": "Deep-96",
        "vectors": "1M vectors",
        "distance": "Cosine Distance",
    },
}

# === STYLE DEFINITIONS ===
# Colors and markers for each series
STYLES = {
    "ivfflat": {
        "color": "#1f77b4",  # blue
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "hnsw": {
        "color": "#2ca02c",  # green
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "parallel_ivfflat": {
        "color": "#ff7f0e",  # orange
        "marker": "^",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "parallel_ivfflat_w4": {
        "color": "#ff7f0e",  # orange
        "marker": "^",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "parallel_ivfflat_w8": {
        "color": "#d62728",  # red
        "marker": "v",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "parallel_ivfflat_w12": {
        "color": "#9467bd",  # purple
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 7,
    },
}


def load_results(results_dir, dataset, mode):
    """Load all available result JSON files for a given dataset and mode.

    Returns dict of {series_key: data_dict}.
    """
    results = {}
    results_dir = Path(results_dir)

    # Standard index types
    for idx_type in ["ivfflat", "hnsw"]:
        fpath = results_dir / f"recall_qps_{dataset}_{idx_type}_{mode}.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
            # The JSON has the index type as key
            if idx_type in data:
                results[idx_type] = data[idx_type]
            else:
                # Try first key
                first_key = next(iter(data))
                results[idx_type] = data[first_key]

    # Parallel IVFFlat - check for per-worker-count files first
    worker_files_found = False
    for nw in [4, 8, 12, 16, 24, 32]:
        fpath = results_dir / f"recall_qps_{dataset}_parallel_ivfflat_w{nw}_{mode}.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
            key = f"parallel_ivfflat_w{nw}"
            if key in data:
                results[key] = data[key]
            else:
                first_key = next(iter(data))
                results[key] = data[first_key]
            worker_files_found = True

    # Fallback: legacy single parallel_ivfflat file (no worker count suffix)
    if not worker_files_found:
        fpath = results_dir / f"recall_qps_{dataset}_parallel_ivfflat_{mode}.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
            if "parallel_ivfflat" in data:
                results["parallel_ivfflat"] = data["parallel_ivfflat"]
            else:
                first_key = next(iter(data))
                results["parallel_ivfflat"] = data[first_key]

    # Also check for combined "all" file
    all_path = results_dir / f"recall_qps_{dataset}_all_{mode}.json"
    if all_path.exists() and not results:
        with open(all_path, "r") as f:
            data = json.load(f)
        results.update(data)

    return results


def get_label(series_key, data):
    """Generate legend label for a series."""
    build_params = data.get("build_params", {})
    num_workers = data.get("num_workers")

    if series_key == "ivfflat":
        lists = build_params.get("lists", "?")
        return f"IVFFlat (lists={lists})"
    elif series_key == "hnsw":
        m = build_params.get("m", "?")
        efc = build_params.get("ef_construction", "?")
        return f"HNSW (m={m}, efc={efc})"
    elif series_key.startswith("parallel_ivfflat_w"):
        nw = series_key.split("_w")[1]
        lists = build_params.get("lists", "?")
        return f"Parallel IVFFlat (lists={lists}, w={nw})"
    elif series_key == "parallel_ivfflat":
        lists = build_params.get("lists", "?")
        if num_workers:
            return f"Parallel IVFFlat (lists={lists}, w={num_workers})"
        return f"Parallel IVFFlat (lists={lists})"
    else:
        return series_key


def get_style(series_key):
    """Get plot style for a series."""
    if series_key in STYLES:
        return STYLES[series_key]
    # Default style for unknown series
    return {
        "color": "gray",
        "marker": "x",
        "linewidth": 1.5,
        "markersize": 6,
    }


def plot_recall_qps(results, dataset, mode, output_path):
    """Generate the recall vs QPS plot."""
    info = DATASET_INFO.get(dataset, {"name": dataset.upper(), "vectors": "", "distance": ""})
    mode_label = "Warm Cache" if mode == "warm" else "Cold Cache"
    k = 100  # default

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Determine plot order: ivfflat, hnsw, then parallel variants sorted by worker count
    plot_order = []
    if "ivfflat" in results:
        plot_order.append("ivfflat")
    if "hnsw" in results:
        plot_order.append("hnsw")
    # Legacy single parallel
    if "parallel_ivfflat" in results:
        plot_order.append("parallel_ivfflat")
    # Per-worker-count parallel (sorted)
    parallel_keys = sorted([k for k in results if k.startswith("parallel_ivfflat_w")],
                           key=lambda x: int(x.split("_w")[1]))
    plot_order.extend(parallel_keys)

    for series_key in plot_order:
        data = results[series_key]
        k = data.get("k", 100)

        # Use full_sweep data for smooth curves
        sweep = data.get("full_sweep", [])
        if not sweep:
            # Fallback to recall_qps_points
            sweep = data.get("recall_qps_points", [])

        if not sweep:
            continue

        # Filter to only include points within the visible x-range
        x_min, x_max = 0.65, 1.0
        recalls = []
        qps_vals = []
        for pt in sweep:
            r = pt["recall"] if "recall" in pt else pt.get("actual_recall", 0)
            if r >= x_min:
                recalls.append(r)
                qps_vals.append(pt["qps"])

        if not recalls:
            continue

        style = get_style(series_key)
        label = get_label(series_key, data)

        ax.plot(recalls, qps_vals,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markeredgewidth=0.5,
                markeredgecolor='black',
                label=label,
                zorder=3)

    # Labels and title
    ax.set_xlabel(f"Recall@{k}", fontsize=13)
    ax.set_ylabel("Queries Per Second (QPS)", fontsize=13)

    title_line1 = f"Recall vs QPS - {info['name']} ({info['vectors']}, {mode_label})"
    title_line2 = f"pgvector on PostgreSQL 18.1, K={k}, {info['distance']}"
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(left=0.65, right=1.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Recall vs QPS diagrams")
    parser.add_argument("--dataset", required=True, choices=["sift", "gist", "deep"],
                        help="Dataset name")
    parser.add_argument("--mode", required=True, choices=["warm", "cold"],
                        help="Cache mode")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing recall_qps JSON files (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: recall_qps_{dataset}_{mode}.png)")
    args = parser.parse_args()

    # Auto-detect results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Try common locations
        candidates = [
            Path(__file__).resolve().parent / "results_recall_qps",
            Path(__file__).resolve().parent / "vdb-tuning-main" / "compute_and_compare_recall" / "results_recall_qps",
            Path("/users/mangelis/vdb_tuning/vdb-tuning-main/compute_and_compare_recall/results_recall_qps"),
        ]
        results_dir = None
        for c in candidates:
            if c.exists():
                results_dir = c
                break
        if results_dir is None:
            print("ERROR: Could not find results_recall_qps directory.")
            print("       Use --results-dir to specify the path.")
            sys.exit(1)

    print(f"Results directory: {results_dir}")

    # Load results
    results = load_results(results_dir, args.dataset, args.mode)

    if not results:
        print(f"ERROR: No result files found for dataset={args.dataset}, mode={args.mode}")
        print(f"       Looked in: {results_dir}")
        print(f"       Expected files like: recall_qps_{args.dataset}_ivfflat_{args.mode}.json")
        sys.exit(1)

    print(f"Loaded {len(results)} series: {list(results.keys())}")

    # Output path
    if args.output:
        output_path = args.output
    else:
        figures_dir = Path(__file__).resolve().parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(figures_dir / f"recall_qps_{args.dataset}_{args.mode}.png")

    # Plot
    plot_recall_qps(results, args.dataset, args.mode, output_path)


if __name__ == "__main__":
    main()
