#!/usr/bin/env python3
"""
Plot comparison of ANN vs OLAP throughput from test_single_point.py results.

Overlays results from multiple indexes (IVFFlat, HNSW, Parallel IVFFlat)
on a single plot with X=ANN throughput (QPS), Y=OLAP throughput (QPS).

Usage:
    python3 plot_single_point_comparison.py \
        "IVFFlat":/tmp/measurement_ivfflat_16sb_25cg.json \
        "HNSW":/tmp/measurement_hnsw_16sb_25cg.json \
        "Parallel IVFFlat":/tmp/measurement_parallel_ivfflat_16sb_25cg.json \
        [--output figure.png] [--title "My Title"]

    # Or auto-detect from a directory:
    python3 plot_single_point_comparison.py --dir /tmp --memory 16:25 [--output figure.png]
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "ivfflat": {"color": "#2ca02c", "marker": "o", "label": "IVFFlat (nlist=8192)"},
    "hnsw": {"color": "#1f77b4", "marker": "s", "label": "HNSW (m=16, efc=128)"},
    "parallel_ivfflat": {"color": "#d62728", "marker": "^", "label": "Parallel IVFFlat (nlist=8192, 8w)"},
}


def load_results(json_path):
    """Load measurement results from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def plot_comparison(datasets, output_path, title=None, annotate=True):
    """
    Plot ANN QPS (x-axis) vs OLAP QPS (y-axis) for multiple indexes.

    datasets: list of (label, data_dict) where data_dict has 'results' list
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for label, data in datasets:
        results = data["results"]
        index_type = data.get("index_type", label.lower().replace(" ", "_"))
        style = STYLES.get(index_type, {"color": "gray", "marker": "x", "label": label})

        ann_qps = [r["ann_qps"] for r in results]
        olap_qps = [r["olap_qps"] for r in results]

        # Plot all points
        ax.scatter(ann_qps, olap_qps, color=style["color"], marker=style["marker"],
                   s=60, alpha=0.7, zorder=5)

        # Group by ann_workers to draw lines (each fixed ann_w is a curve)
        by_ann_w = defaultdict(list)
        for r in results:
            by_ann_w[r["ann_w"]].append(r)

        # Sort each group by olap_w and plot as connected line
        first_line = True
        for ann_w in sorted(by_ann_w.keys()):
            pts = sorted(by_ann_w[ann_w], key=lambda r: r["olap_w"])
            x = [p["ann_qps"] for p in pts]
            y = [p["olap_qps"] for p in pts]
            lbl = style["label"] if first_line else None
            ax.plot(x, y, color=style["color"], linewidth=1.8, alpha=0.8,
                    marker=style["marker"], markersize=7, label=lbl, zorder=4)
            first_line = False

            # Annotate points with (ann_w, olap_w)
            if annotate:
                for p in pts:
                    ax.annotate(f"{p['ann_w']},{p['olap_w']}",
                                xy=(p["ann_qps"], p["olap_qps"]),
                                xytext=(3, 5), textcoords='offset points',
                                fontsize=7, alpha=0.6, color=style["color"])

    # Formatting
    ax.set_xlabel("ANN Throughput (qps)", fontsize=13)
    ax.set_ylabel("OLAP Throughput (qps)", fontsize=13)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        # Auto-generate title from first dataset's config
        d0 = datasets[0][1]
        sb = d0.get("shared_buffers_gb", "?")
        cg = d0.get("cgroup_gb", "?")
        ax.set_title(f"ANN vs OLAP Throughput (Shared Buffers: {sb}GB, Cgroup: {cg}GB, warm)",
                     fontsize=13, fontweight='bold')

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-index throughput comparison")
    parser.add_argument("files", nargs="*",
                        help="label:path pairs (e.g., 'IVFFlat:/tmp/measurement_ivfflat_16sb_25cg.json')")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory to auto-find measurement JSONs")
    parser.add_argument("--memory", type=str, default=None,
                        help="Memory config to match (e.g., 16:25) when using --dir")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Don't annotate points with (ann_w, olap_w)")
    args = parser.parse_args()

    datasets = []

    if args.dir:
        # Auto-find measurement files
        search_dir = Path(args.dir)
        for index_type in ["ivfflat", "hnsw", "parallel_ivfflat"]:
            if args.memory:
                sb, cg = args.memory.split(":")
                pattern = f"measurement_{index_type}_{sb}sb_{cg}cg.json"
            else:
                pattern = f"measurement_{index_type}_*.json"

            matches = list(search_dir.glob(pattern))
            if matches:
                data = load_results(matches[0])
                label = STYLES.get(index_type, {}).get("label", index_type)
                datasets.append((label, data))
                print(f"  Loaded: {matches[0].name} ({len(data['results'])} points)")
    else:
        # Parse label:path pairs
        for arg in args.files:
            if ":" in arg:
                label, path = arg.split(":", 1)
                data = load_results(path)
                datasets.append((label, data))
                print(f"  Loaded: {path} ({len(data['results'])} points)")

    if not datasets:
        print("ERROR: No datasets found.")
        print("Usage: python3 plot_single_point_comparison.py 'IVFFlat':path1.json 'HNSW':path2.json ...")
        print("  or:  python3 plot_single_point_comparison.py --dir /tmp --memory 16:25")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.memory:
            sb, cg = args.memory.split(":")
            output_path = f"comparison_{sb}sb_{cg}cg.png"
        else:
            output_path = "comparison_throughput.png"

    plot_comparison(datasets, output_path, title=args.title, annotate=not args.no_annotate)


if __name__ == "__main__":
    main()
