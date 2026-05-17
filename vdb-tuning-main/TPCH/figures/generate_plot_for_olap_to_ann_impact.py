#!/usr/bin/env python3
"""
Generate throughput frontier plots for OLAP→ANN interference experiments.

Reads JSON-lines result files from results_olap_impact_to_ann/ and produces
a 2x2 grid plot showing:
  X-axis: Vector Search Throughput (qps)
  Y-axis: Analytical Throughput (tps)
  Labels: number of OLAP workers (1-6)

Each panel is a different memory pressure level (No Pressure, Relaxed, Medium, Tight).
Each index type is a different color/marker.
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============== AUTO-DETECT PATHS ==============
SCRIPT_DIR = Path(__file__).resolve().parent            # .../TPCH/figures/
TPCH_DIR = SCRIPT_DIR.parent                            # .../TPCH/
RESULTS_DIR = TPCH_DIR / "results_olap_impact_to_ann"
OUTPUT_DIR = SCRIPT_DIR / "figure_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== STYLING ==============
colors = {
    "HNSW": "#e74c3c",                      # Red
    "Parallel IVFFlat (io=1000)": "#3498db", # Blue
    "Default IVFFlat (io=1000)": "#2ecc71",  # Green
    "Default IVFFlat (io=0)": "#9b59b6",     # Purple
}

markers = {
    "HNSW": "o",
    "Parallel IVFFlat (io=1000)": "s",
    "Default IVFFlat (io=1000)": "^",
    "Default IVFFlat (io=0)": "D",
}

# Memory config mapping: (shared_buffers_gb, cgroup_gb) → display name
MEMORY_LABELS = {
    (40, 60): "No Pressure (40GB)",
    (16, 20): "Relaxed (16GB)",
    (12, 16): "Medium (12GB)",
    (8, 12):  "Tight (8GB)",
}

# Ordered configs for 2x2 grid
MEMORY_ORDER = [
    "No Pressure (40GB)",
    "Relaxed (16GB)",
    "Medium (12GB)",
    "Tight (8GB)",
]


def parse_filename(filename):
    """Extract index type, io_depth, shared_buffers, cgroup from filename.

    Example: olap_impact_on_ann_IVFFLAT_io_0_32_cores_for_postgres_40GB_shared_buffers_60GB_RAM_GIST.txt
    """
    m = re.match(
        r'olap_impact_on_ann_(\w+?)(?:_PARALLEL)?_io_(\d+)_\d+_cores_for_postgres_(\d+)GB_shared_buffers_(\d+)GB_RAM',
        filename
    )
    if not m:
        return None

    index_type = m.group(1)   # IVFFLAT or HNSW
    io_depth = int(m.group(2))
    shared_gb = int(m.group(3))
    cgroup_gb = int(m.group(4))

    # Check if PARALLEL is in filename
    is_parallel = "_PARALLEL_" in filename

    # Build display name
    if index_type == "HNSW":
        display_name = "HNSW"
    elif is_parallel:
        display_name = f"Parallel IVFFlat (io={io_depth})"
    else:
        display_name = f"Default IVFFlat (io={io_depth})"

    mem_label = MEMORY_LABELS.get((shared_gb, cgroup_gb))
    if not mem_label:
        mem_label = f"Custom ({shared_gb}GB/{cgroup_gb}GB)"

    return {
        "index_display": display_name,
        "mem_label": mem_label,
        "shared_gb": shared_gb,
        "cgroup_gb": cgroup_gb,
        "io_depth": io_depth,
    }


def load_results():
    """Load all result files and organize by memory config and index type.

    Returns: {mem_label: {index_display: [(olap_workers, topk_qps, olap_tps), ...]}}
    """
    data = {}

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    result_files = sorted(RESULTS_DIR.glob("*.txt"))
    if not result_files:
        print(f"ERROR: No result files found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(result_files)} result file(s) in {RESULTS_DIR}")

    for fpath in result_files:
        info = parse_filename(fpath.name)
        if not info:
            print(f"  Skipping unrecognized file: {fpath.name}")
            continue

        mem_label = info["mem_label"]
        index_display = info["index_display"]

        print(f"  Loading: {fpath.name}")
        print(f"    -> Memory: {mem_label}, Index: {index_display}")

        if mem_label not in data:
            data[mem_label] = {}
        if index_display not in data[mem_label]:
            data[mem_label][index_display] = []

        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                olap_workers = entry.get("olap_workers", 0)
                topk_throughput = entry.get("topk_throughput", 0)
                olap_tps = entry.get("olap_throughput_tps", 0)

                # Skip baseline (0 OLAP workers) for the frontier plot
                # since it has no OLAP throughput
                if olap_workers == 0:
                    continue

                # Skip entries where OLAP workers didn't complete (tps=0)
                if olap_tps == 0 and olap_workers > 0:
                    print(f"    WARNING: olap_workers={olap_workers} has tps=0, skipping")
                    continue

                data[mem_label][index_display].append(
                    (olap_workers, topk_throughput, olap_tps)
                )

        count = len(data[mem_label][index_display])
        print(f"    Loaded {count} data points")

    return data


def plot_throughput_frontier_grid(data):
    """Create 2x2 grid of throughput frontiers matching the paper style."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        'Throughput Frontiers: OLAP vs Vector Search Performance\n'
        'Under Different Memory Pressure Levels',
        fontsize=14, fontweight='bold', y=0.98
    )

    all_handles = []
    all_labels = []

    for idx, config in enumerate(MEMORY_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        if config not in data or not data[config]:
            ax.set_title(config, fontsize=12, fontweight='bold')
            ax.text(0.5, 0.5, 'No data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            if row == 1:
                ax.set_xlabel('Vector Search Throughput (qps)', fontsize=10)
            if col == 0:
                ax.set_ylabel('Analytical Throughput (tps)', fontsize=10)
            continue

        data_dict = data[config]

        for index_type in sorted(data_dict.keys()):
            points = data_dict[index_type]
            if not points:
                continue

            # Sort by worker count (1→6) for logical progression
            sorted_points = sorted(points, key=lambda x: x[0])

            topk_qps = [p[1] for p in sorted_points]
            olap_tps = [p[2] for p in sorted_points]
            workers = [p[0] for p in sorted_points]

            color = colors.get(index_type, "#333333")
            marker = markers.get(index_type, "o")

            # Plot scatter points
            scatter = ax.scatter(
                topk_qps, olap_tps,
                color=color,
                marker=marker,
                s=100,
                label=index_type,
                zorder=5,
                edgecolors='black',
                linewidths=0.5,
            )

            # Track for legend
            if index_type not in all_labels:
                all_handles.append(scatter)
                all_labels.append(index_type)

            # Annotate with OLAP worker count
            for qps, tps, w in zip(topk_qps, olap_tps, workers):
                ax.annotate(
                    str(w), (qps, tps),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=color,
                )

        ax.set_title(config, fontsize=12, fontweight='bold')
        if row == 1:
            ax.set_xlabel('Vector Search Throughput (qps)', fontsize=10)
        if col == 0:
            ax.set_ylabel('Analytical Throughput (tps)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Common legend at bottom
    if all_handles:
        fig.legend(all_handles, all_labels, loc='lower center', ncol=len(all_labels),
                   fontsize=10, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    out_png = OUTPUT_DIR / "throughput_frontier_grid.png"
    out_pdf = OUTPUT_DIR / "throughput_frontier_grid.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close()


def main():
    print("=" * 60)
    print("OLAP→ANN Throughput Frontier Plot Generator")
    print("=" * 60)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print()

    data = load_results()

    if not data:
        print("ERROR: No data loaded. Check results directory.")
        sys.exit(1)

    print(f"\nLoaded data for {len(data)} memory config(s):")
    for config, idx_data in data.items():
        for idx_name, points in idx_data.items():
            print(f"  {config} / {idx_name}: {len(points)} points")

    plot_throughput_frontier_grid(data)

    print("\nDone!")


if __name__ == "__main__":
    main()
