#!/usr/bin/env python3
"""
Plot compiler optimization results (with vs without) for IVFFlat and HNSW.

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


def main():
    parser = argparse.ArgumentParser(description="Plot compiler optimization results")
    parser.add_argument("--dataset", required=True, choices=["gist", "deep", "sift"],
                        help="Dataset name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: data/compiler_optimization_{dataset}.pdf)")
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = SCRIPT_DIR / "data"

    # Load results
    ivf_without = data_dir / f"results_without_optimization_IVFFLAT_{dataset}.txt"
    ivf_with = data_dir / f"results_with_optimization_IVFFLAT_{dataset}.txt"
    hnsw_without = data_dir / f"results_without_optimization_HNSW_{dataset}.txt"
    hnsw_with = data_dir / f"results_with_optimization_HNSW_{dataset}.txt"

    has_ivfflat = ivf_without.exists() and ivf_with.exists()
    has_hnsw = hnsw_without.exists() and hnsw_with.exists()

    if not has_ivfflat and not has_hnsw:
        # Try legacy naming (without dataset suffix)
        ivf_without = data_dir / "results_without_optimization_IVFFLAT.txt"
        ivf_with = data_dir / "results_with_optimization_IVFFLAT.txt"
        hnsw_without = data_dir / "results_without_optimization_HNSW.txt"
        hnsw_with = data_dir / "results_with_optimization_HNSW.txt"
        has_ivfflat = ivf_without.exists() and ivf_with.exists()
        has_hnsw = hnsw_without.exists() and hnsw_with.exists()

    if not has_ivfflat and not has_hnsw:
        print("ERROR: No result files found.")
        print(f"  Looked in: {data_dir}")
        print(f"  Expected: results_{{with|without}}_optimization_{{IVFFLAT|HNSW}}_{dataset}.txt")
        sys.exit(1)

    num_plots = int(has_ivfflat) + int(has_hnsw)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_ivfflat:
        k_no, t_no = load_results(ivf_without)
        k_opt, t_opt = load_results(ivf_with)
        gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_no, t_opt)]

        ax = axes[plot_idx]
        ax.plot(k_no, t_no, 'r^--', label='Without Optimization', markersize=8, color='red')
        ax.plot(k_opt, t_opt, 'gs-', label='With Optimization', markersize=8, color='green')

        for k, t, gain in zip(k_opt, t_opt, gains):
            if gain > 0:
                ax.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                            xytext=(0, -18), ha='center', fontsize=8, color='green')

        ax.set_xlabel('K (Top-K)', fontsize=11)
        ax.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
        ax.set_title(f'IVFFlat (recall=0.9) — {dataset.upper()}', fontsize=13)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    if has_hnsw:
        k_no, t_no = load_results(hnsw_without)
        k_opt, t_opt = load_results(hnsw_with)
        gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_no, t_opt)]

        ax = axes[plot_idx]
        ax.plot(k_no, t_no, 'r^--', label='Without Optimization', markersize=8, color='red')
        ax.plot(k_opt, t_opt, 'gs-', label='With Optimization', markersize=8, color='green')

        for k, t, gain in zip(k_opt, t_opt, gains):
            if gain > 0:
                ax.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                            xytext=(0, -18), ha='center', fontsize=8, color='green')

        ax.set_xlabel('K (Top-K)', fontsize=11)
        ax.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
        ax.set_title(f'HNSW (recall=0.9) — {dataset.upper()}', fontsize=13)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"compiler_optimization_{dataset}.pdf"

    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print percentage gains
    print(f"\n% Performance Gains by K ({dataset}):")
    if has_ivfflat and has_hnsw:
        k_ivf, t_ivf_no = load_results(ivf_without)
        _, t_ivf_opt = load_results(ivf_with)
        k_hnsw, t_hnsw_no = load_results(hnsw_without)
        _, t_hnsw_opt = load_results(hnsw_with)
        ivf_gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_ivf_no, t_ivf_opt)]
        hnsw_gains = [((t1 - t2) / t1) * 100 if t1 > 0 else 0 for t1, t2 in zip(t_hnsw_no, t_hnsw_opt)]
        print(f"{'K':<8} {'IVFFlat Gain':<15} {'HNSW Gain':<15}")
        print("-" * 38)
        for k, ig, hg in zip(k_ivf, ivf_gains, hnsw_gains):
            print(f"{k:<8} {ig:<15.2f} {hg:<15.2f}")


if __name__ == "__main__":
    main()
