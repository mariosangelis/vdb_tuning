#!/usr/bin/env python3
"""
Plot throughput frontier comparison across multiple indexes.

Supports:
  - Absolute frontier plot (ANN QPS vs OLAP QPS)
  - Normalized frontier plot (each index normalized by its own saturation)
  - Zoomed view (limit X-axis range)
  - Reading from test_single_point.py JSON outputs or grid_graph.json files

Usage:
    # From test_single_point.py results (files in /tmp):
    python3 plot_frontier_comparison.py --dir /tmp --memory 14:20 --output frontier.png
    python3 plot_frontier_comparison.py --dir /tmp --memory 14:20 --normalize --output frontier_norm.png
    python3 plot_frontier_comparison.py --dir /tmp --memory 14:20 --xlim 100 --output frontier_zoomed.png

    # With nonuma suffix:
    python3 plot_frontier_comparison.py --dir /tmp --memory 14:20 --suffix nonuma --output frontier_fullcores.png

    # From grid_graph.json files (frontier analysis results):
    python3 plot_frontier_comparison.py --grid \
        "IVFFlat":path/to/ivfflat/grid_graph.json \
        "HNSW":path/to/hnsw/grid_graph.json \
        "Parallel IVFFlat":path/to/pivfflat/grid_graph.json \
        --output frontier.png

    # Explicit file mode:
    python3 plot_frontier_comparison.py \
        "IVFFlat":/tmp/measurement_ivfflat_14sb_20cg.json \
        "HNSW":/tmp/measurement_hnsw_14sb_20cg.json \
        "Parallel IVFFlat":/tmp/measurement_parallel_ivfflat_14sb_20cg.json \
        --output frontier.png
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

QUERIES_PER_BATCH = 22


def compute_frontier(points):
    """Correct Pareto frontier: keep only non-dominated points."""
    points = list(set(points))
    frontier = []
    for i, (ann_i, olap_i) in enumerate(points):
        dominated = False
        for j, (ann_j, olap_j) in enumerate(points):
            if i == j:
                continue
            if ann_j >= ann_i and olap_j >= olap_i and (ann_j > ann_i or olap_j > olap_i):
                dominated = True
                break
        if not dominated:
            frontier.append((ann_i, olap_i))
    frontier.sort(key=lambda p: p[0])
    return frontier


def load_single_point_json(path):
    """Load test_single_point.py output JSON."""
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    points = [(r["ann_qps"], r["olap_qps"]) for r in results]
    annotations = [(r["ann_qps"], r["olap_qps"], f"{r['ann_w']},{r['olap_w']}") for r in results]
    return points, annotations


def load_grid_graph_json(path):
    """Load grid_graph.json from frontier analysis."""
    with open(path) as f:
        data = json.load(f)
    measurements = data["measurements"]
    points = []
    annotations = []
    for m in measurements:
        ann = m["ann_throughput_qps"]
        olap = m.get("olap_throughput_qps", m.get("olap_throughput_tps", 0) * QUERIES_PER_BATCH)
        points.append((ann, olap))
        annotations.append((ann, olap, f"{m['ann_workers']},{m['olap_workers']}"))
    return points, annotations


def plot_frontier(datasets, output_path, title=None, normalize=False, xlim=None, ylim=None, annotate=True):
    """
    Plot Pareto frontiers for multiple indexes.

    datasets: list of (label, index_key, points, annotations)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for label, idx_key, points, annotations in datasets:
        style = STYLES.get(idx_key, {"color": "gray", "marker": "x", "label": label})

        # Filter by xlim if specified
        if xlim and not normalize:
            points = [(a, o) for a, o in points if a <= xlim]
            annotations = [(a, o, t) for a, o, t in annotations if a <= xlim]

        if not points:
            continue

        # Normalize if requested
        max_ann = max(p[0] for p in points) or 1
        max_olap = max(p[1] for p in points) or 1

        if normalize:
            plot_points = [(a / max_ann, o / max_olap) for a, o in points]
            plot_annotations = [(a / max_ann, o / max_olap, t) for a, o, t in annotations]
        else:
            plot_points = points
            plot_annotations = annotations

        # Scatter all points (faint)
        px = [p[0] for p in plot_points]
        py = [p[1] for p in plot_points]
        ax.scatter(px, py, color=style["color"], marker=style["marker"],
                   s=40, alpha=0.35, zorder=3)

        # Annotate
        if annotate:
            for a, o, txt in plot_annotations:
                ax.annotate(txt, xy=(a, o), xytext=(3, 5), textcoords="offset points",
                            fontsize=6.5, alpha=0.6, color=style["color"])

        # Compute and plot frontier
        frontier = compute_frontier(plot_points)
        if frontier:
            fa = [p[0] for p in frontier]
            fo = [p[1] for p in frontier]
            if normalize:
                auc = np.trapezoid(fo, fa) if len(frontier) > 1 else 0
                lbl = f"{style['label']} (AUC={auc:.3f})"
            else:
                lbl = style["label"]
            ax.plot(fa, fo, color=style["color"], linewidth=2.5, marker=style["marker"],
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                    label=lbl, zorder=5)
            ax.fill_between(fa, fo, alpha=0.08, color=style["color"])

        if normalize:
            print(f"  {label}: max_ann={max_ann:.1f}, max_olap={max_olap:.4f}, AUC={auc:.3f}")

    # Proportional line for normalized plots
    if normalize:
        prop_x = np.linspace(0, 1, 50)
        ax.plot(prop_x, 1 - prop_x, "k--", alpha=0.4, linewidth=1.2, label="Proportional (AUC=0.5)")

    # Labels
    if normalize:
        ax.set_xlabel("Normalized ANN Throughput", fontsize=13)
        ax.set_ylabel("Normalized OLAP Throughput", fontsize=13)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.12)
    else:
        ax.set_xlabel("ANN Throughput (qps)", fontsize=13)
        ax.set_ylabel("OLAP Throughput (qps)", fontsize=13)
        ax.set_xlim(left=0, right=xlim * 1.05 if xlim else None)
        ax.set_ylim(bottom=0)

    ax.set_title(title or "Throughput Frontier: ANN vs OLAP", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot throughput frontier comparison")
    parser.add_argument("files", nargs="*", help="label:path pairs")
    parser.add_argument("--dir", type=str, default=None, help="Directory with measurement JSONs")
    parser.add_argument("--memory", type=str, default=None, help="Memory config (e.g., 14:20)")
    parser.add_argument("--suffix", type=str, default="", help="File suffix (e.g., nonuma)")
    parser.add_argument("--grid", nargs="*", default=None, help="label:path pairs for grid_graph.json files")
    parser.add_argument("--output", type=str, default="frontier_comparison.png")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", help="Normalize by each index saturation")
    parser.add_argument("--xlim", type=float, default=None, help="Limit X-axis (ANN QPS)")
    parser.add_argument("--ylim", type=float, default=None, help="Limit Y-axis (OLAP QPS)")
    parser.add_argument("--no-annotate", action="store_true", help="Skip point annotations")
    args = parser.parse_args()

    datasets = []

    if args.grid:
        # Load from grid_graph.json files
        for arg in args.grid:
            label, path = arg.split(":", 1)
            points, annotations = load_grid_graph_json(path)
            idx_key = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if "hnsw" in idx_key:
                idx_key = "hnsw"
            elif "parallel" in idx_key:
                idx_key = "parallel_ivfflat"
            else:
                idx_key = "ivfflat"
            datasets.append((label, idx_key, points, annotations))
            print(f"  Loaded grid: {path} ({len(points)} points)")

    elif args.dir:
        # Auto-find from directory
        search_dir = Path(args.dir)
        suffix = f"_{args.suffix}" if args.suffix else ""
        for idx_key in ["ivfflat", "hnsw", "parallel_ivfflat"]:
            if args.memory:
                sb, cg = args.memory.split(":")
                pattern = f"measurement_{idx_key}_{sb}sb_{cg}cg{suffix}.json"
            else:
                pattern = f"measurement_{idx_key}_*{suffix}.json"

            matches = list(search_dir.glob(pattern))
            if matches:
                points, annotations = load_single_point_json(matches[0])
                label = STYLES[idx_key]["label"]
                datasets.append((label, idx_key, points, annotations))
                print(f"  Loaded: {matches[0].name} ({len(points)} points)")

    else:
        # Explicit label:path pairs
        for arg in args.files:
            if ":" in arg:
                label, path = arg.split(":", 1)
                points, annotations = load_single_point_json(path)
                idx_key = label.lower().replace(" ", "_")
                if "hnsw" in idx_key:
                    idx_key = "hnsw"
                elif "parallel" in idx_key:
                    idx_key = "parallel_ivfflat"
                else:
                    idx_key = "ivfflat"
                datasets.append((label, idx_key, points, annotations))
                print(f"  Loaded: {path} ({len(points)} points)")

    if not datasets:
        print("ERROR: No datasets found.")
        parser.print_help()
        sys.exit(1)

    # Auto-generate title if not provided
    if not args.title:
        if args.memory:
            sb, cg = args.memory.split(":")
            mode = "full_cores" if args.suffix == "nonuma" else "NUMA pinned"
            args.title = f"Throughput Frontier: ANN vs OLAP\n(SB: {sb}GB, CG: {cg}GB, {mode})"
            if args.normalize:
                args.title = f"Normalized Throughput Frontier\n(SB: {sb}GB, CG: {cg}GB, {mode})"

    plot_frontier(datasets, args.output, title=args.title, normalize=args.normalize,
                  xlim=args.xlim, ylim=args.ylim, annotate=not args.no_annotate)


if __name__ == "__main__":
    main()
