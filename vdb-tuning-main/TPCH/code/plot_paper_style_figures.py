#!/usr/bin/env python3
"""
Publication-quality plots for mixed workload experiments.
Inspired by:
1. "How Good is My HTAP System?" (Chronis et al.) - Throughput frontier style
2. "An In-Depth Study of Filter-Agnostic Vector Search" - QPS vs selectivity style

Creates multiple figure types from the experimental data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

BASE_DIR = Path("/home/mariosangelis/thesis/wisconsin_machine/last_results_wisconsin/vdb-tuning-main/TPCH")

# ============== DATA LOADING ==============

def load_ann_impact_data(filepath):
    """Load ANN impact on OLAP data (varying ANN workers, fixed OLAP worker=1)."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append({
                        'ann_workers': record.get('topk_query_count', 0) > 0,  # infer from count
                        'topk_throughput': record.get('topk_throughput', 0),
                        'tpch_execution_time': record.get('tpch_execution_time', 0),
                        'tpch_throughput': record.get('summary', {}).get('Throughput (requests/second)', 0),
                    })
                except json.JSONDecodeError:
                    continue
    return data


def load_olap_impact_data(filepath):
    """Load OLAP impact on ANN data (varying OLAP workers, fixed ANN worker=1)."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append({
                        'tpch_workers': record.get('tpch_workers', 0),
                        'topk_workers': record.get('topk_workers', 1),
                        'topk_throughput_mean': record.get('topk_throughput_mean', 0),
                        'topk_throughput_std': record.get('topk_throughput_std', 0),
                        'tpch_throughput_mean': record.get('tpch_throughput_mean', 0),
                        'tpch_throughput_std': record.get('tpch_throughput_std', 0),
                        'tpch_execution_time_mean': record.get('tpch_execution_time_mean', 0),
                        'topk_latency_p50_mean': record.get('topk_latency_p50_mean', 0),
                        'topk_latency_p95_mean': record.get('topk_latency_p95_mean', 0),
                        'topk_latency_p99_mean': record.get('topk_latency_p99_mean', 0),
                    })
                except json.JSONDecodeError:
                    continue
    return data


# ============== FIGURE 1: THROUGHPUT FRONTIER (HTAP Style) ==============

def plot_throughput_frontier():
    """
    Create throughput frontier plot similar to Chronis et al.
    X-axis: Vector Search Throughput (qps)
    Y-axis: OLAP Throughput (tps)
    Shows the Pareto frontier of achievable throughputs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Throughput Frontiers: Mixed Workload Performance\n(OLAP Impact on Vector Search)',
                 fontsize=14, fontweight='bold')

    # Memory configurations
    configs = [
        ('60GB_RAM_40', 'No Pressure (40GB)'),
        ('20GB_RAM_16', 'Relaxed (16GB)'),
        ('14GB_RAM_12', 'Medium (12GB)'),
        ('12GB_RAM_8', 'Tight (8GB)'),
    ]

    # Index types and their styles
    index_styles = {
        'IVFFLAT_io_1000': {'color': '#3498db', 'marker': 'o', 'label': 'IVFFlat (io=1000)'},
        'IVFFLAT_io_0': {'color': '#9b59b6', 'marker': 'D', 'label': 'IVFFlat (io=0)'},
        'IVFFLAT_PARALLEL_io_1000': {'color': '#e74c3c', 'marker': 's', 'label': 'Parallel IVFFlat'},
        'HNSW_io_1000': {'color': '#2ecc71', 'marker': '^', 'label': 'HNSW'},
    }

    for idx, (mem_pattern, title) in enumerate(configs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Find matching files
        for index_type, style in index_styles.items():
            # Try to find OLAP_IMPACT files
            pattern = f"OLAP_IMPACT_*{index_type}*{mem_pattern}*"
            matching_files = list(BASE_DIR.glob(pattern))

            if not matching_files:
                continue

            for filepath in matching_files:
                data = load_olap_impact_data(filepath)
                if not data:
                    continue

                # Sort by tpch_workers
                data.sort(key=lambda x: x['tpch_workers'])

                tpch_workers = [d['tpch_workers'] for d in data]
                topk_throughput = [d['topk_throughput_mean'] for d in data]
                tpch_throughput = [d['tpch_throughput_mean'] for d in data]
                topk_std = [d['topk_throughput_std'] for d in data]

                # Plot with error bars
                ax.errorbar(topk_throughput, tpch_throughput,
                           xerr=topk_std,
                           marker=style['marker'],
                           color=style['color'],
                           linewidth=2, markersize=8,
                           label=style['label'], capsize=3)

                # Annotate with worker count
                for w, x, y in zip(tpch_workers, topk_throughput, tpch_throughput):
                    ax.annotate(str(w), (x, y), textcoords="offset points",
                               xytext=(5, 5), fontsize=8, color=style['color'])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Vector Search Throughput (qps)')
        ax.set_ylabel('OLAP Throughput (tps)')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(BASE_DIR / 'throughput_frontier_htap_style.png', dpi=300)
    plt.savefig(BASE_DIR / 'throughput_frontier_htap_style.pdf')
    print("Saved: throughput_frontier_htap_style.png/pdf")
    plt.close()


# ============== FIGURE 2: QPS vs Memory Pressure (Bar Chart) ==============

def plot_qps_vs_memory_pressure():
    """
    Bar chart comparing throughput across memory configurations.
    Similar to paper Figure 6 style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data structure: {index_type: {memory_config: throughput}}
    ann_throughput = defaultdict(dict)
    olap_throughput = defaultdict(dict)

    memory_configs = ['60GB', '20GB', '14GB', '12GB']
    memory_labels = ['No Pressure\n(40GB)', 'Relaxed\n(16GB)', 'Medium\n(12GB)', 'Tight\n(8GB)']

    # Load data from OLAP_IMPACT files (tpch_workers=1, baseline)
    index_types = ['IVFFLAT_io_1000', 'IVFFLAT_io_0', 'HNSW_io_1000']

    for index_type in index_types:
        for mem_cfg in memory_configs:
            pattern = f"OLAP_IMPACT_*{index_type}*{mem_cfg}*"
            matching_files = list(BASE_DIR.glob(pattern))

            for filepath in matching_files:
                data = load_olap_impact_data(filepath)
                # Get baseline (tpch_workers=1)
                baseline = [d for d in data if d['tpch_workers'] == 1]
                if baseline:
                    ann_throughput[index_type][mem_cfg] = baseline[0]['topk_throughput_mean']
                    olap_throughput[index_type][mem_cfg] = baseline[0]['tpch_throughput_mean']

    # Plot ANN throughput
    ax1 = axes[0]
    x = np.arange(len(memory_configs))
    width = 0.25

    colors = {'IVFFLAT_io_1000': '#3498db', 'IVFFLAT_io_0': '#9b59b6', 'HNSW_io_1000': '#2ecc71'}
    labels = {'IVFFLAT_io_1000': 'IVFFlat (io=1000)', 'IVFFLAT_io_0': 'IVFFlat (io=0)', 'HNSW_io_1000': 'HNSW'}

    for i, index_type in enumerate(index_types):
        values = [ann_throughput[index_type].get(mem, 0) for mem in memory_configs]
        if any(v > 0 for v in values):
            ax1.bar(x + i*width, values, width, label=labels[index_type], color=colors[index_type])

    ax1.set_xlabel('Memory Configuration')
    ax1.set_ylabel('Vector Search Throughput (qps)')
    ax1.set_title('Vector Search Performance vs Memory Pressure\n(with 1 OLAP worker)', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(memory_labels)
    ax1.legend()

    # Plot OLAP throughput
    ax2 = axes[1]
    for i, index_type in enumerate(index_types):
        values = [olap_throughput[index_type].get(mem, 0) for mem in memory_configs]
        if any(v > 0 for v in values):
            ax2.bar(x + i*width, values, width, label=labels[index_type], color=colors[index_type])

    ax2.set_xlabel('Memory Configuration')
    ax2.set_ylabel('OLAP Throughput (tps)')
    ax2.set_title('OLAP Performance vs Memory Pressure\n(with 1 Vector Search worker)', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(memory_labels)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'throughput_vs_memory_pressure.png', dpi=300)
    plt.savefig(BASE_DIR / 'throughput_vs_memory_pressure.pdf')
    print("Saved: throughput_vs_memory_pressure.png/pdf")
    plt.close()


# ============== FIGURE 3: Latency Distribution (Box Plot Style) ==============

def plot_latency_distribution():
    """
    Box plot or percentile chart showing ANN latency under different OLAP loads.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vector Search Latency Distribution Under OLAP Pressure\n(IVFFlat io=1000)',
                 fontsize=14, fontweight='bold')

    configs = [
        ('60GB_RAM_40', 'No Pressure (40GB)'),
        ('20GB_RAM_16', 'Relaxed (16GB)'),
        ('14GB_RAM_12', 'Medium (12GB)'),
        ('12GB_RAM_8', 'Tight (8GB)'),
    ]

    for idx, (mem_pattern, title) in enumerate(configs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Find IVFFLAT io=1000 file
        pattern = f"OLAP_IMPACT_*IVFFLAT_io_1000*{mem_pattern}*"
        matching_files = list(BASE_DIR.glob(pattern))

        if not matching_files:
            ax.set_title(f"{title}\n(No data)")
            continue

        data = load_olap_impact_data(matching_files[0])
        data.sort(key=lambda x: x['tpch_workers'])

        tpch_workers = [d['tpch_workers'] for d in data]
        p50 = [d['topk_latency_p50_mean'] for d in data]
        p95 = [d['topk_latency_p95_mean'] for d in data]
        p99 = [d['topk_latency_p99_mean'] for d in data]

        x = np.arange(len(tpch_workers))
        width = 0.25

        ax.bar(x - width, p50, width, label='P50', color='#3498db')
        ax.bar(x, p95, width, label='P95', color='#e74c3c')
        ax.bar(x + width, p99, width, label='P99', color='#2c3e50')

        ax.set_xlabel('OLAP Workers')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tpch_workers)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(BASE_DIR / 'latency_distribution.png', dpi=300)
    plt.savefig(BASE_DIR / 'latency_distribution.pdf')
    print("Saved: latency_distribution.png/pdf")
    plt.close()


# ============== FIGURE 4: Normalized Degradation Plot ==============

def plot_normalized_degradation():
    """
    Show throughput degradation as percentage of baseline.
    Similar to paper's relative performance plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ANN degradation under OLAP pressure
    ax1 = axes[0]
    ax1.set_title('Vector Search Degradation Under OLAP Pressure\n(Normalized to 1 OLAP worker)', fontweight='bold')

    configs = [
        ('60GB_RAM_40', 'No Pressure (40GB)', '#2ecc71'),
        ('20GB_RAM_16', 'Relaxed (16GB)', '#3498db'),
        ('14GB_RAM_12', 'Medium (12GB)', '#f39c12'),
        ('12GB_RAM_8', 'Tight (8GB)', '#e74c3c'),
    ]

    for mem_pattern, label, color in configs:
        pattern = f"OLAP_IMPACT_*IVFFLAT_io_1000*{mem_pattern}*"
        matching_files = list(BASE_DIR.glob(pattern))

        if not matching_files:
            continue

        data = load_olap_impact_data(matching_files[0])
        data.sort(key=lambda x: x['tpch_workers'])

        # Normalize to baseline (first point or workers=1)
        baseline_data = [d for d in data if d['tpch_workers'] == 1]
        if not baseline_data:
            baseline_data = [data[0]] if data else None

        if not baseline_data:
            continue

        baseline = baseline_data[0]['topk_throughput_mean']

        tpch_workers = [d['tpch_workers'] for d in data]
        normalized = [100 * d['topk_throughput_mean'] / baseline if baseline > 0 else 0 for d in data]

        ax1.plot(tpch_workers, normalized, marker='o', linewidth=2,
                label=label, color=color, markersize=8)

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xlabel('Number of OLAP Workers')
    ax1.set_ylabel('Vector Search Throughput (% of baseline)')
    ax1.legend()
    ax1.set_ylim(0, 120)

    # OLAP degradation (showing scaling)
    ax2 = axes[1]
    ax2.set_title('OLAP Throughput Scaling\n(Normalized to 1 OLAP worker)', fontweight='bold')

    for mem_pattern, label, color in configs:
        pattern = f"OLAP_IMPACT_*IVFFLAT_io_1000*{mem_pattern}*"
        matching_files = list(BASE_DIR.glob(pattern))

        if not matching_files:
            continue

        data = load_olap_impact_data(matching_files[0])
        data.sort(key=lambda x: x['tpch_workers'])

        baseline_data = [d for d in data if d['tpch_workers'] == 1]
        if not baseline_data:
            baseline_data = [data[0]] if data else None

        if not baseline_data:
            continue

        baseline = baseline_data[0]['tpch_throughput_mean']

        tpch_workers = [d['tpch_workers'] for d in data]
        # Calculate actual scaling vs ideal scaling
        actual = [d['tpch_throughput_mean'] for d in data]
        ideal = [baseline * w for w in tpch_workers]
        efficiency = [100 * a / i if i > 0 else 0 for a, i in zip(actual, ideal)]

        ax2.plot(tpch_workers, efficiency, marker='s', linewidth=2,
                label=label, color=color, markersize=8)

    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal Scaling')
    ax2.set_xlabel('Number of OLAP Workers')
    ax2.set_ylabel('Scaling Efficiency (%)')
    ax2.legend()
    ax2.set_ylim(0, 120)

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'normalized_degradation.png', dpi=300)
    plt.savefig(BASE_DIR / 'normalized_degradation.pdf')
    print("Saved: normalized_degradation.png/pdf")
    plt.close()


# ============== FIGURE 5: io_depth Effect Comparison ==============

def plot_io_depth_effect():
    """
    Compare io=0 vs io=1000 for IVFFlat.
    Shows the benefit of prefetching.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ('60GB', 'No Pressure\n(40GB)'),
        ('20GB', 'Relaxed\n(16GB)'),
        ('14GB', 'Medium\n(12GB)'),
        ('12GB', 'Tight\n(8GB)'),
    ]

    io_1000_throughput = []
    io_0_throughput = []

    for mem_pattern, _ in configs:
        # io=1000
        pattern1 = f"OLAP_IMPACT_*IVFFLAT_io_1000*{mem_pattern}*"
        files1 = list(BASE_DIR.glob(pattern1))
        if files1:
            data = load_olap_impact_data(files1[0])
            baseline = [d for d in data if d['tpch_workers'] == 1]
            io_1000_throughput.append(baseline[0]['topk_throughput_mean'] if baseline else 0)
        else:
            io_1000_throughput.append(0)

        # io=0
        pattern0 = f"OLAP_IMPACT_*IVFFLAT_io_0*{mem_pattern}*"
        files0 = list(BASE_DIR.glob(pattern0))
        if files0:
            data = load_olap_impact_data(files0[0])
            baseline = [d for d in data if d['tpch_workers'] == 1]
            io_0_throughput.append(baseline[0]['topk_throughput_mean'] if baseline else 0)
        else:
            io_0_throughput.append(0)

    x = np.arange(len(configs))
    width = 0.35

    # Throughput comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, io_1000_throughput, width, label='io_depth=1000', color='#3498db')
    bars2 = ax1.bar(x + width/2, io_0_throughput, width, label='io_depth=0', color='#e74c3c')

    ax1.set_xlabel('Memory Configuration')
    ax1.set_ylabel('Vector Search Throughput (qps)')
    ax1.set_title('Effect of Prefetching (io_depth)\non Vector Search Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[1] for c in configs])
    ax1.legend()

    # Speedup
    ax2 = axes[1]
    speedup = [io_1000_throughput[i] / io_0_throughput[i] if io_0_throughput[i] > 0 else 0
               for i in range(len(configs))]

    bars = ax2.bar(x, speedup, color='#2ecc71')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Memory Configuration')
    ax2.set_ylabel('Speedup (io=1000 / io=0)')
    ax2.set_title('Prefetching Speedup Factor', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[1] for c in configs])

    # Add value labels
    for bar, val in zip(bars, speedup):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'io_depth_effect.png', dpi=300)
    plt.savefig(BASE_DIR / 'io_depth_effect.pdf')
    print("Saved: io_depth_effect.png/pdf")
    plt.close()


# ============== FIGURE 6: Index Type Comparison ==============

def plot_index_comparison():
    """
    Compare IVFFlat vs HNSW performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Index Type Comparison: IVFFlat vs HNSW\n(Under varying OLAP pressure)',
                 fontsize=14, fontweight='bold')

    configs = [
        ('60GB_RAM_40', 'No Pressure (40GB)'),
        ('20GB_RAM_16', 'Relaxed (16GB)'),
        ('14GB_RAM_12', 'Medium (12GB)'),
        ('12GB_RAM_8', 'Tight (8GB)'),
    ]

    for idx, (mem_pattern, title) in enumerate(configs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # IVFFlat
        pattern_ivf = f"OLAP_IMPACT_*IVFFLAT_io_1000*{mem_pattern}*"
        files_ivf = list(BASE_DIR.glob(pattern_ivf))

        # HNSW
        pattern_hnsw = f"OLAP_IMPACT_*HNSW*{mem_pattern}*"
        files_hnsw = list(BASE_DIR.glob(pattern_hnsw))

        if files_ivf:
            data = load_olap_impact_data(files_ivf[0])
            data.sort(key=lambda x: x['tpch_workers'])
            workers = [d['tpch_workers'] for d in data]
            throughput = [d['topk_throughput_mean'] for d in data]
            std = [d['topk_throughput_std'] for d in data]
            ax.errorbar(workers, throughput, yerr=std, marker='o', linewidth=2,
                       label='IVFFlat (io=1000)', color='#3498db', capsize=3)

        if files_hnsw:
            data = load_olap_impact_data(files_hnsw[0])
            data.sort(key=lambda x: x['tpch_workers'])
            workers = [d['tpch_workers'] for d in data]
            throughput = [d['topk_throughput_mean'] for d in data]
            std = [d['topk_throughput_std'] for d in data]
            ax.errorbar(workers, throughput, yerr=std, marker='^', linewidth=2,
                       label='HNSW', color='#2ecc71', capsize=3)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('OLAP Workers')
        ax.set_ylabel('Vector Search Throughput (qps)')
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(BASE_DIR / 'index_comparison.png', dpi=300)
    plt.savefig(BASE_DIR / 'index_comparison.pdf')
    print("Saved: index_comparison.png/pdf")
    plt.close()


# ============== FIGURE 7: Combined Summary Plot ==============

def plot_summary_heatmap():
    """
    Create a summary heatmap showing throughput across all configurations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    memory_configs = ['60GB', '20GB', '14GB', '12GB']
    memory_labels = ['No Pressure (40GB)', 'Relaxed (16GB)', 'Medium (12GB)', 'Tight (8GB)']
    index_types = ['IVFFLAT_io_1000', 'IVFFLAT_io_0', 'HNSW_io_1000', 'IVFFLAT_PARALLEL_io_1000']
    index_labels = ['IVFFlat\n(io=1000)', 'IVFFlat\n(io=0)', 'HNSW', 'Parallel\nIVFFlat']

    # Create matrices
    ann_matrix = np.zeros((len(index_types), len(memory_configs)))
    olap_matrix = np.zeros((len(index_types), len(memory_configs)))

    for i, index_type in enumerate(index_types):
        for j, mem_cfg in enumerate(memory_configs):
            pattern = f"OLAP_IMPACT_*{index_type}*{mem_cfg}*"
            matching_files = list(BASE_DIR.glob(pattern))

            for filepath in matching_files:
                data = load_olap_impact_data(filepath)
                baseline = [d for d in data if d['tpch_workers'] == 1]
                if baseline:
                    ann_matrix[i, j] = baseline[0]['topk_throughput_mean']
                    olap_matrix[i, j] = baseline[0]['tpch_throughput_mean']

    # Plot ANN heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(ann_matrix, cmap='YlGnBu', aspect='auto')
    ax1.set_xticks(np.arange(len(memory_configs)))
    ax1.set_yticks(np.arange(len(index_types)))
    ax1.set_xticklabels(memory_labels, rotation=45, ha='right')
    ax1.set_yticklabels(index_labels)
    ax1.set_title('Vector Search Throughput (qps)\n(with 1 OLAP worker)', fontweight='bold')

    # Add text annotations
    for i in range(len(index_types)):
        for j in range(len(memory_configs)):
            if ann_matrix[i, j] > 0:
                ax1.text(j, i, f'{ann_matrix[i, j]:.1f}',
                        ha='center', va='center', color='black', fontsize=9)

    plt.colorbar(im1, ax=ax1)

    # Plot OLAP heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(olap_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(np.arange(len(memory_configs)))
    ax2.set_yticks(np.arange(len(index_types)))
    ax2.set_xticklabels(memory_labels, rotation=45, ha='right')
    ax2.set_yticklabels(index_labels)
    ax2.set_title('OLAP Throughput (tps)\n(with 1 Vector Search worker)', fontweight='bold')

    for i in range(len(index_types)):
        for j in range(len(memory_configs)):
            if olap_matrix[i, j] > 0:
                ax2.text(j, i, f'{olap_matrix[i, j]:.3f}',
                        ha='center', va='center', color='black', fontsize=9)

    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'summary_heatmap.png', dpi=300)
    plt.savefig(BASE_DIR / 'summary_heatmap.pdf')
    print("Saved: summary_heatmap.png/pdf")
    plt.close()


# ============== MAIN ==============

if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print(f"Data directory: {BASE_DIR}")
    print()

    # List available files
    olap_impact_files = list(BASE_DIR.glob("OLAP_IMPACT_*.txt"))
    print(f"Found {len(olap_impact_files)} OLAP_IMPACT files")

    # Generate all figures
    print("\n1. Throughput Frontier (HTAP style)...")
    plot_throughput_frontier()

    print("\n2. Throughput vs Memory Pressure...")
    plot_qps_vs_memory_pressure()

    print("\n3. Latency Distribution...")
    plot_latency_distribution()

    print("\n4. Normalized Degradation...")
    plot_normalized_degradation()

    print("\n5. io_depth Effect...")
    plot_io_depth_effect()

    print("\n6. Index Comparison...")
    plot_index_comparison()

    print("\n7. Summary Heatmap...")
    plot_summary_heatmap()

    print("\nAll figures generated!")
