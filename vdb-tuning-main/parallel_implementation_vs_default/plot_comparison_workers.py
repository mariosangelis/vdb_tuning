#!/usr/bin/env python3
"""
Plot Default IVFFlat vs Parallel IVFFlat (multiple worker counts) comparison.

Reads the output of run_comparison_workers.py and generates a Recall vs Latency
plot showing the speedup of parallel implementations over the default.

Usage:
    python3 plot_comparison_workers.py --dataset gist --mode cold
    python3 plot_comparison_workers.py --dataset gist --mode warm
    python3 plot_comparison_workers.py --dataset deep --mode cold --output my_plot.png
"""

import json
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Style definitions
STYLES = {
    "default": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "label": "Default IVFFlat",
    },
    "4": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "label": "Parallel (4 workers)",
    },
    "8": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "label": "Parallel (8 workers)",
    },
    "12": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": ":",
        "label": "Parallel (12 workers)",
    },
}

DATASET_INFO = {
    "gist": {"name": "GIST-960", "distance": "L2 Distance"},
    "deep": {"name": "Deep-96", "distance": "Cosine Distance"},
    "sift": {"name": "SIFT-128", "distance": "L2 Distance"},
}


def main():
    parser = argparse.ArgumentParser(description="Plot comparison results")
    parser.add_argument("--dataset", required=True, choices=["gist", "deep", "sift"],
                        help="Dataset name")
    parser.add_argument("--mode", required=True, choices=["cold", "warm"],
                        help="Which mode results to plot")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode
    input_file = SCRIPT_DIR / f"comparison_results_workers_{dataset}_{mode}_K_100.json"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run run_comparison_workers.py --mode {mode} first.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        data = json.load(f)

    K = data["k"]
    lists = data["lists"]
    ds_info = DATASET_INFO.get(dataset, {"name": dataset, "distance": "Unknown"})
    mode_label = "Cold Cache" if mode == "cold" else "Warm Cache"

    # Extract data points
    default_results = data.get("default", None)
    parallel_by_workers = data.get("parallel_by_workers", {})

    if not default_results and not parallel_by_workers:
        print("ERROR: No results found in the file.")
        sys.exit(1)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plot default
    if default_results:
        recalls = [r["target_recall"] for r in default_results]
        latencies = [r["avg_time_ms"] for r in default_results]
        style = STYLES["default"]
        ax.plot(recalls, latencies,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2, markersize=8,
                label=style["label"], zorder=5)

    # Plot each worker count
    for worker_key in sorted(parallel_by_workers.keys(), key=int):
        results = parallel_by_workers[worker_key]
        recalls = [r["target_recall"] for r in results]
        latencies = [r["avg_time_ms"] for r in results]
        style = STYLES.get(worker_key, STYLES["4"])
        ax.plot(recalls, latencies,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2, markersize=8,
                label=style["label"], zorder=5)

    # Formatting
    ax.set_xlabel("Recall@100", fontsize=13)
    ax.set_ylabel("Average Query Latency (ms)", fontsize=13)
    ax.set_title(
        f"Default vs Parallel IVFFlat — {ds_info['name']} ({mode_label})\n"
        f"{ds_info['distance']}, K={K}, lists={lists}",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.65)

    # Add speedup annotations
    if default_results and parallel_by_workers:
        best_worker = min(parallel_by_workers.keys(),
                          key=lambda k: parallel_by_workers[k][-1]["avg_time_ms"])
        best_results = parallel_by_workers[best_worker]

        for d, p in zip(default_results, best_results):
            if d["avg_time_ms"] > 0 and p["avg_time_ms"] > 0:
                speedup = d["avg_time_ms"] / p["avg_time_ms"]
                if speedup > 1.05:
                    ax.annotate(f"{speedup:.1f}x",
                                xy=(p["target_recall"], p["avg_time_ms"]),
                                xytext=(0, -15), textcoords='offset points',
                                fontsize=9, ha='center', color=STYLES[best_worker]["color"],
                                fontweight='bold')

    plt.tight_layout()

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = SCRIPT_DIR / f"comparison_workers_{dataset}_{mode}_K_{K}.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
