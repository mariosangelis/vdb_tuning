#!/usr/bin/env python3
"""
Publication-quality plots for Wisconsin machine experiment results.
Reads JSON-lines result files and generates figures showing:
1. TPC-H execution time vs ANN workers (one subplot per memory config, lines per index)
2. TopK throughput vs ANN workers
3. Buffer miss analysis (shared_read totals)
4. Per-query breakdown under interference
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os
import re
from pathlib import Path
from collections import defaultdict

# ============== CONFIGURATION ==============
# Set this to your results directory
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "../results_ann_impact_to_olap/"))
OUTPUT_DIR = Path("figure_plots/")
OUTPUT_DIR.mkdir(exist_ok=True)

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Memory config display names
MEM_CONFIGS = {
    '40GB_shared_buffers_60GB_RAM': ('no_pressure', 'No Pressure\n(40GB/60GB)'),
    '16GB_shared_buffers_20GB_RAM': ('relaxed', 'Relaxed\n(16GB/20GB)'),
    '12GB_shared_buffers_16GB_RAM': ('medium', 'Medium\n(12GB/16GB)'),
    '8GB_shared_buffers_12GB_RAM': ('tight', 'Tight\n(8GB/12GB)'),
}

# Ordered list for consistent subplot placement
MEM_ORDER = [
    '40GB_shared_buffers_60GB_RAM',
    '16GB_shared_buffers_20GB_RAM',
    '12GB_shared_buffers_16GB_RAM',
    '8GB_shared_buffers_12GB_RAM',
]

# Index styles
INDEX_STYLES = {
    'IVFFLAT_io_1000': {'color': '#2196F3', 'marker': 'o', 'label': 'IVFFlat (io=1000)', 'linestyle': '-'},
    'IVFFLAT_io_0': {'color': '#9C27B0', 'marker': 'D', 'label': 'IVFFlat (io=0)', 'linestyle': '--'},
    'HNSW_io_1000': {'color': '#4CAF50', 'marker': '^', 'label': 'HNSW', 'linestyle': '-.'},
}


# ============== DATA LOADING ==============

def parse_filename(filename):
    """Extract index type, io_depth, and memory config from filename."""
    name = filename.stem
    # Example: mixed_workload_tpch_and_topk_IVFFLAT_io_1000_16_cores_for_postgres_40GB_shared_buffers_60GB_RAM_GIST

    # Extract index + io
    for idx_key in ['HNSW_io_1000', 'IVFFLAT_io_1000', 'IVFFLAT_io_0']:
        if idx_key in name:
            index_type = idx_key
            break
    else:
        return None, None

    # Extract memory config
    for mem_key in MEM_CONFIGS:
        if mem_key in name:
            return index_type, mem_key

    return index_type, None


def load_all_results():
    """Load all result files into structured data."""
    # Structure: data[index_type][mem_config] = [(ann_workers_inferred, record), ...]
    data = defaultdict(lambda: defaultdict(list))

    for filepath in sorted(RESULTS_DIR.glob("mixed_workload_*.txt")):
        index_type, mem_config = parse_filename(filepath)
        if not index_type or not mem_config:
            print(f"  Skipping: {filepath.name}")
            continue

        with open(filepath) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Infer ANN workers from topk_query_count
                    ann_workers = 0
                    if record['topk_query_count'] > 0:
                        # Estimate from throughput pattern
                        throughput = record['topk_throughput']
                        if throughput > 0:
                            # Rough mapping: line 0 = 0 workers, line 1 = 5, line 2 = 10
                            ann_workers = [0, 5, 10][min(line_num, 2)]

                    data[index_type][mem_config].append({
                        'ann_workers': ann_workers,
                        'tpch_time': record['tpch_execution_time'],
                        'topk_throughput': record['topk_throughput'],
                        'topk_count': record['topk_query_count'],
                        'query_times': record.get('tpch_query_times', []),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Error in {filepath.name} line {line_num}: {e}")

    return data


# ============== FIGURE 1: TPC-H Time vs ANN Workers (2x2 grid) ==============

def plot_tpch_time_vs_ann_workers(data):
    """One subplot per memory config, one line per index type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('TPC-H Execution Time vs ANN Worker Count\n(Impact of Vector Search on OLAP Performance)',
                 fontsize=15, fontweight='bold')

    for idx, mem_key in enumerate(MEM_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        _, title = MEM_CONFIGS[mem_key]

        for index_type, style in INDEX_STYLES.items():
            if mem_key not in data[index_type]:
                continue

            records = sorted(data[index_type][mem_key], key=lambda x: x['ann_workers'])
            workers = [r['ann_workers'] for r in records]
            times = [r['tpch_time'] for r in records]

            ax.plot(workers, times,
                   marker=style['marker'], color=style['color'],
                   linestyle=style['linestyle'], linewidth=2.5,
                   markersize=9, label=style['label'])

            # Annotate values
            for w, t in zip(workers, times):
                ax.annotate(f'{t:.0f}s', (w, t),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=style['color'])

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('ANN Workers')
        ax.set_ylabel('TPC-H Time (seconds)')
        ax.set_xticks([0, 5, 10])
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(bottom=0)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01),
              fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(OUTPUT_DIR / 'tpch_time_vs_ann_workers.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'tpch_time_vs_ann_workers.pdf')
    print("Saved: tpch_time_vs_ann_workers.png/pdf")
    plt.close()


# ============== FIGURE 2: TopK Throughput vs ANN Workers (2x2 grid) ==============

def plot_topk_throughput_vs_ann_workers(data):
    """TopK throughput for each config."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('ANN (TopK) Throughput vs Worker Count\n(Under Different Memory Pressure Levels)',
                 fontsize=15, fontweight='bold')

    for idx, mem_key in enumerate(MEM_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        _, title = MEM_CONFIGS[mem_key]

        for index_type, style in INDEX_STYLES.items():
            if mem_key not in data[index_type]:
                continue

            records = sorted(data[index_type][mem_key], key=lambda x: x['ann_workers'])
            # Skip 0-worker entries (no TopK throughput)
            records = [r for r in records if r['ann_workers'] > 0]
            if not records:
                continue

            workers = [r['ann_workers'] for r in records]
            throughput = [r['topk_throughput'] for r in records]

            ax.plot(workers, throughput,
                   marker=style['marker'], color=style['color'],
                   linestyle=style['linestyle'], linewidth=2.5,
                   markersize=9, label=style['label'])

            for w, t in zip(workers, throughput):
                ax.annotate(f'{t:.0f}', (w, t),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=style['color'])

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('ANN Workers')
        ax.set_ylabel('TopK Throughput (qps)')
        ax.set_xticks([5, 10])
        ax.set_xlim(3, 12)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01),
              fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(OUTPUT_DIR / 'topk_throughput_vs_ann_workers.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'topk_throughput_vs_ann_workers.pdf')
    print("Saved: topk_throughput_vs_ann_workers.png/pdf")
    plt.close()


# ============== FIGURE 3: ANN Impact on OLAP Frontier (Paper Style) ==============

def plot_throughput_frontier(data):
    """
    Paper-style frontier plot matching the original Wisconsin figure.
    X-axis: Vector Search Throughput (qps)
    Y-axis: OLAP Execution Time (s) — log scale, higher = worse
    Annotated with ANN worker count.
    Same axis limits across all subplots for comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Impact of ANN Pressure on OLAP Performance\n(1 OLAP worker, varying ANN workers)',
                 fontsize=15, fontweight='bold')

    # Memory config display names (shorter for subplot titles)
    mem_titles = {
        '40GB_shared_buffers_60GB_RAM': 'No Pressure (40GB)',
        '16GB_shared_buffers_20GB_RAM': 'Relaxed (16GB)',
        '12GB_shared_buffers_16GB_RAM': 'Medium (12GB)',
        '8GB_shared_buffers_12GB_RAM': 'Tight (8GB)',
    }

    # Find global max for consistent axes
    max_topk = 0
    max_tpch = 0
    for index_type in data:
        for mem_key in data[index_type]:
            for r in data[index_type][mem_key]:
                max_topk = max(max_topk, r['topk_throughput'])
                max_tpch = max(max_tpch, r['tpch_time'])

    for idx, mem_key in enumerate(MEM_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        title = mem_titles.get(mem_key, mem_key)

        for index_type, style in INDEX_STYLES.items():
            if mem_key not in data[index_type]:
                continue

            records = sorted(data[index_type][mem_key], key=lambda x: x['ann_workers'])

            topk_tput = [r['topk_throughput'] for r in records]
            tpch_time = [r['tpch_time'] for r in records]
            workers = [r['ann_workers'] for r in records]

            ax.plot(topk_tput, tpch_time,
                   marker=style['marker'], color=style['color'],
                   linestyle=style['linestyle'], linewidth=2.5,
                   markersize=10, label=style['label'])

            # Annotate with worker count
            for w, x_val, y_val in zip(workers, topk_tput, tpch_time):
                ax.annotate(str(w), (x_val, y_val),
                           textcoords="offset points", xytext=(6, 6),
                           fontsize=9, fontweight='bold', color=style['color'])

        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('Vector Search Throughput (qps)')
        ax.set_ylabel('OLAP Execution Time (s)')
        ax.set_yscale('log')
        ax.set_xlim(-10, max_topk * 1.1)
        ax.set_ylim(80, max_tpch * 1.3)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticks([100, 200, 300, 500, 700])
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01),
              fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(OUTPUT_DIR / 'ann_impact_on_olap_frontier.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'ann_impact_on_olap_frontier.pdf')
    print("Saved: ann_impact_on_olap_frontier.png/pdf")
    plt.close()


# ============== FIGURE 4: Buffer Miss Analysis ==============

def plot_buffer_misses(data):
    """Total shared_read per TPC-H pass for each config, grouped by index type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('TPC-H Buffer Pool Misses (shared_read) vs ANN Workers\n(Higher = More Disk I/O Due to Buffer Contention)',
                 fontsize=15, fontweight='bold')

    for idx, mem_key in enumerate(MEM_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        _, title = MEM_CONFIGS[mem_key]

        for index_type, style in INDEX_STYLES.items():
            if mem_key not in data[index_type]:
                continue

            records = sorted(data[index_type][mem_key], key=lambda x: x['ann_workers'])
            workers = [r['ann_workers'] for r in records]

            total_reads = []
            for r in records:
                total = sum(q.get('shared_read', 0) for q in r['query_times'])
                total_reads.append(total / 1e6)  # millions

            ax.plot(workers, total_reads,
                   marker=style['marker'], color=style['color'],
                   linestyle=style['linestyle'], linewidth=2.5,
                   markersize=9, label=style['label'])

            for w, t in zip(workers, total_reads):
                ax.annotate(f'{t:.1f}M', (w, t),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=style['color'])

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('ANN Workers')
        ax.set_ylabel('Total Buffer Misses (millions of pages)')
        ax.set_xticks([0, 5, 10])
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01),
              fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(OUTPUT_DIR / 'buffer_misses.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'buffer_misses.pdf')
    print("Saved: buffer_misses.png/pdf")
    plt.close()


# ============== FIGURE 5: Per-Query Breakdown (selected queries) ==============

def plot_per_query_breakdown(data):
    """Bar chart showing per-query time for 0 vs 5 ANN workers, tight config, IVFFlat vs HNSW."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Per-Query TPC-H Time: 0 vs 5 ANN Workers (Tight Config, 8GB/12GB)',
                 fontsize=14, fontweight='bold')

    mem_key = '8GB_shared_buffers_12GB_RAM'
    queries_to_show = ['Q1', 'Q3', 'Q6', 'Q7', 'Q9', 'Q12', 'Q18', 'Q20', 'Q21']

    for ax_idx, (index_type, idx_label) in enumerate([
        ('IVFFLAT_io_1000', 'IVFFlat (io=1000)'),
        ('HNSW_io_1000', 'HNSW')
    ]):
        ax = axes[ax_idx]

        if mem_key not in data[index_type]:
            ax.set_title(f'{idx_label}\n(No data)')
            continue

        records = sorted(data[index_type][mem_key], key=lambda x: x['ann_workers'])
        rec_0 = [r for r in records if r['ann_workers'] == 0]
        rec_5 = [r for r in records if r['ann_workers'] == 5]

        if not rec_0 or not rec_5:
            continue

        times_0 = {q['query']: q['time_sec'] for q in rec_0[0]['query_times']}
        times_5 = {q['query']: q['time_sec'] for q in rec_5[0]['query_times']}

        x = np.arange(len(queries_to_show))
        width = 0.35

        vals_0 = [times_0.get(q, 0) for q in queries_to_show]
        vals_5 = [times_5.get(q, 0) for q in queries_to_show]

        bars1 = ax.bar(x - width/2, vals_0, width, label='0 ANN workers', color='#64B5F6')
        bars2 = ax.bar(x + width/2, vals_5, width, label='5 ANN workers', color='#EF5350')

        ax.set_title(idx_label, fontweight='bold')
        ax.set_xlabel('TPC-H Query')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_xticks(x)
        ax.set_xticklabels(queries_to_show, rotation=45)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUTPUT_DIR / 'per_query_breakdown.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'per_query_breakdown.pdf')
    print("Saved: per_query_breakdown.png/pdf")
    plt.close()


# ============== FIGURE 6: Summary Bar Chart (all configs, fixed 5 ANN) ==============

def plot_summary_bars(data):
    """Grouped bar chart: TPC-H time across memory configs, grouped by index type, for 0 and 5 ANN workers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mem_labels = ['No Pressure\n(40/60)', 'Relaxed\n(16/20)', 'Medium\n(12/16)', 'Tight\n(8/12)']
    x = np.arange(len(MEM_ORDER))
    width = 0.25

    for ax_idx, ann_count in enumerate([0, 5]):
        ax = axes[ax_idx]
        ax.set_title(f'TPC-H Execution Time ({ann_count} ANN Workers)', fontweight='bold')

        for i, (index_type, style) in enumerate(INDEX_STYLES.items()):
            times = []
            for mem_key in MEM_ORDER:
                if mem_key in data[index_type]:
                    records = data[index_type][mem_key]
                    matching = [r for r in records if r['ann_workers'] == ann_count]
                    if matching:
                        times.append(matching[0]['tpch_time'])
                    else:
                        times.append(0)
                else:
                    times.append(0)

            if any(t > 0 for t in times):
                bars = ax.bar(x + i * width, times, width,
                             label=style['label'], color=style['color'], alpha=0.85)
                # Value labels
                for bar, val in zip(bars, times):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Memory Configuration')
        ax.set_ylabel('TPC-H Time (seconds)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(mem_labels)
        ax.legend()
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_bars.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'summary_bars.pdf')
    print("Saved: summary_bars.png/pdf")
    plt.close()


# ============== MAIN ==============

if __name__ == "__main__":
    print(f"Loading results from: {RESULTS_DIR}")
    data = load_all_results()

    print(f"\nLoaded data for:")
    for idx_type in sorted(data.keys()):
        configs = sorted(data[idx_type].keys())
        for cfg in configs:
            records = data[idx_type][cfg]
            workers = [r['ann_workers'] for r in records]
            print(f"  {idx_type} / {cfg}: workers={workers}")

    print(f"\nGenerating figures to: {OUTPUT_DIR}")

    print("\n1. TPC-H Time vs ANN Workers...")
    plot_tpch_time_vs_ann_workers(data)

    print("\n2. TopK Throughput vs ANN Workers...")
    plot_topk_throughput_vs_ann_workers(data)

    print("\n3. Throughput Frontier...")
    plot_throughput_frontier(data)

    print("\n4. Buffer Miss Analysis...")
    plot_buffer_misses(data)

    print("\n5. Per-Query Breakdown...")
    plot_per_query_breakdown(data)

    print("\n6. Summary Bar Chart...")
    plot_summary_bars(data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
