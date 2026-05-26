import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Auto-detect paths
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# Function to load the results from a text file
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

# Load all result sets
k_ivf_no, t_ivf_no = load_results('data/results_without_optimization_IVFFLAT.txt')
k_ivf_opt, t_ivf_opt = load_results('data/results_with_optimization_IVFFLAT.txt')
k_hnsw_no, t_hnsw_no = load_results('data/results_without_optimization_HNSW.txt')
k_hnsw_opt, t_hnsw_opt = load_results('data/results_with_optimization_HNSW.txt')

# Compute percentage gains
ivf_gains = [((t1 - t2) / t1) * 100 for t1, t2 in zip(t_ivf_no, t_ivf_opt)]
hnsw_gains = [((t1 - t2) / t1) * 100 for t1, t2 in zip(t_hnsw_no, t_hnsw_opt)]

# Create 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- IVFFlat subplot ---
ax1.plot(k_ivf_no, t_ivf_no, 'r^--', label='Without Optimization', markersize=8, color='red')
ax1.plot(k_ivf_opt, t_ivf_opt, 'gs-', label='With Optimization', markersize=8, color='green')

for k, t, gain in zip(k_ivf_opt, t_ivf_opt, ivf_gains):
    ax1.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                xytext=(0, -18), ha='center', fontsize=8, color='green')

ax1.set_xlabel('K (Top-K)', fontsize=11)
ax1.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
ax1.set_title('IVFFlat (recall=0.9)', fontsize=13)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)

# --- HNSW subplot ---
ax2.plot(k_hnsw_no, t_hnsw_no, 'r^--', label='Without Optimization', markersize=8, color='red')
ax2.plot(k_hnsw_opt, t_hnsw_opt, 'gs-', label='With Optimization', markersize=8, color='green')

for k, t, gain in zip(k_hnsw_opt, t_hnsw_opt, hnsw_gains):
    ax2.annotate(f"{gain:.1f}%", (k, t), textcoords="offset points",
                xytext=(0, -18), ha='center', fontsize=8, color='green')

ax2.set_xlabel('K (Top-K)', fontsize=11)
ax2.set_ylabel('Total Execution Time (seconds) - 1000 queries', fontsize=10)
ax2.set_title('HNSW (recall=0.9)', fontsize=13)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

output_path = SCRIPT_DIR / 'data' / 'compiler_optimization_all.pdf'
fig.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved to {output_path}")

# Print percentage gains
print("\n% Performance Gains by K:")
print(f"{'K':<8} {'IVFFlat Gain':<15} {'HNSW Gain':<15}")
print("-" * 38)
for k, ig, hg in zip(k_ivf_no, ivf_gains, hnsw_gains):
    print(f"{k:<8} {ig:<15.2f} {hg:<15.2f}")
