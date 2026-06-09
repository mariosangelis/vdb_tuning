#!/usr/bin/env python3
"""
Plot compiler optimization results (with vs without) for IVFFlat and HNSW.

Generates two PNG files:
  - compiler_optimization_ivfflat_{dataset}.png  (cold and warm side by side)
  - compiler_optimization_hnsw_{dataset}.png     (cold and warm side by side)

Usage:
    python3 plot_optimization_results.py --dataset gist
    python3 plot_optimization_results.py --dataset deep
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_results(filename):
    k_vals = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            k = int(parts[0].split('=')[1])
            time_val = float(parts[1].split('=')[1])
            k_vals.append(k)
            times.append(time_val)
    return k_vals, times


def plot_index(index_type, dataset, data_dir, output_path):
    """Plot cold and warm side by side for one index type."""
    idx_upper = index_type.upper()

    # Try files with mode suffix
    cold_without = data_dir / f"results_without_optimization_{idx_upper}_{dataset}_cold.txt"
    cold_with = data_dir / f"results_with_optimization_{idx_upper}_{dataset}_cold.txt"
    warm_without = data_dir / f"results_without_optimization_{idx_upper}_{dataset}_warm.txt"
    warm_with = data_dir / f"results_with_optimization_{idx_upper}_{dataset}_warm.txt"

    has_cold = cold_without.exists() and cold_with.exists()
    has_warm = warm_without.exists() and warm_with.exists()

    if not has_cold and not has_warm:
        print(f"  No results found for {idx_upper} ({dataset})")
        return False

    num_plots = int(has_cold) + int(has_warm)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_cold:
        k_no, t_no = load_results(cold_without)
        k_opt, t_opt = load_results(cold_with)
        gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_no, t_opt)]

        ax = axes[plot_idx]
        ax.plot(k_no, t_no, 'r^--', label='Without Optimization', markersize=8)
        ax.plot(k_opt, t_opt, 'gs-', label='With Optimization', markersize=8)

        for k, t, gain in zip(k_opt, t_opt, gains):
            if gain > 0:
                ax.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                            xytext=(0, -18), ha='center', fontsize=8, color='green')

        ax.set_xlabel('K (Top-K)', fontsize=11)
        ax.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
        ax.set_title(f'{idx_upper} — Cold Cache (recall=0.9)', fontsize=13)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    if has_warm:
        k_no, t_no = load_results(warm_without)
        k_opt, t_opt = load_results(warm_with)
        gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_no, t_opt)]

        ax = axes[plot_idx]
        ax.plot(k_no, t_no, 'r^--', label='Without Optimization', markersize=8)
        ax.plot(k_opt, t_opt, 'gs-', label='With Optimization', markersize=8)

        for k, t, gain in zip(k_opt, t_opt, gains):
            if gain > 0:
                ax.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                            xytext=(0, -18), ha='center', fontsize=8, color='green')

        ax.set_xlabel('K (Top-K)', fontsize=11)
        ax.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
        ax.set_title(f'{idx_upper} — Warm Cache (recall=0.9)', fontsize=13)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{idx_upper} Compiler Optimization — {dataset.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    # Changed from PDF to PNG with higher DPI for better quality
    fig.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot compiler optimization results")
    parser.add_argument("--dataset", required=True, choices=["gist", "deep", "sift"],
                        help="Dataset name")
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = SCRIPT_DIR / "data"

    print(f"Generating compiler optimization plots for {dataset}...")

    # Plot IVFFlat - changed to .png
    ivf_output = data_dir / f"compiler_optimization_ivfflat_{dataset}.png"
    plot_index("ivfflat", dataset, data_dir, ivf_output)

    # Plot HNSW - changed to .png
    hnsw_output = data_dir / f"compiler_optimization_hnsw_{dataset}.png"
    plot_index("hnsw", dataset, data_dir, hnsw_output)

    # Print summary table
    print(f"\n% Performance Gains ({dataset}):")
    for idx_type in ["IVFFLAT", "HNSW"]:
        for mode in ["cold", "warm"]:
            without_f = data_dir / f"results_without_optimization_{idx_type}_{dataset}_{mode}.txt"
            with_f = data_dir / f"results_with_optimization_{idx_type}_{dataset}_{mode}.txt"
            if without_f.exists() and with_f.exists():
                k_no, t_no = load_results(without_f)
                k_opt, t_opt = load_results(with_f)
                print(f"\n  {idx_type} ({mode}):")
                print(f"  {'K':<8} {'Without':>10} {'With':>10} {'Gain':>8}")
                print(f"  {'-'*38}")
                for k, t1, t2 in zip(k_no, t_no, t_opt):
                    gain = ((t1 - t2) / t1 * 100) if t1 > 0 else 0
                    print(f"  {k:<8} {t1:>9.2f}s {t2:>9.2f}s {gain:>7.1f}%")


if __name__ == "__main__":
    main()