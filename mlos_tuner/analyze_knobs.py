#!/usr/bin/env python3
"""
Robust Analysis of PostgreSQL Knob Tuning Results

This script provides multiple analysis methods to understand which knobs
affect performance, without relying on unstable ML metrics.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# LOAD DATA
# =============================================================================
with open("results_mlos_tuner_IVFFLAT_BAS_BULKREAD_index_memory_knobs_GIST_DATASET_all_memory_knobs.jsonl") as f:
    data = [json.loads(line) for line in f]
df = pd.json_normalize(data)

# Extract columns
df['shared_buffers_gb'] = df['config.shared_buffers'] / 1024 / 1024  # Convert to GB
df['io_concurrency'] = df['config.effective_io_concurrency']
df['exec_time'] = df['total_exec_time_topk']

print("=" * 70)
print("POSTGRESQL KNOB ANALYSIS")
print("=" * 70)
print(f"\nDataset: {len(df)} samples")
print(f"\nParameter Ranges:")
print(f"  shared_buffers: {df['shared_buffers_gb'].min():.2f} - {df['shared_buffers_gb'].max():.2f} GB")
print(f"  io_concurrency: {df['io_concurrency'].min():.0f} - {df['io_concurrency'].max():.0f}")
print(f"  exec_time:      {df['exec_time'].min():.1f} - {df['exec_time'].max():.1f} seconds")

# =============================================================================
# METHOD 1: CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 1: CORRELATION ANALYSIS")
print("=" * 70)

# Pearson correlation (linear relationship)
corr_sb, p_sb = stats.pearsonr(df['shared_buffers_gb'], df['exec_time'])
corr_io, p_io = stats.pearsonr(df['io_concurrency'], df['exec_time'])

# Spearman correlation (monotonic relationship, more robust)
spearman_sb, sp_sb = stats.spearmanr(df['shared_buffers_gb'], df['exec_time'])
spearman_io, sp_io = stats.spearmanr(df['io_concurrency'], df['exec_time'])

print(f"\nPearson Correlation (linear):")
print(f"  shared_buffers vs exec_time: r = {corr_sb:.4f} (p = {p_sb:.2e})")
print(f"  io_concurrency vs exec_time: r = {corr_io:.4f} (p = {p_io:.2e})")

print(f"\nSpearman Correlation (monotonic, more robust):")
print(f"  shared_buffers vs exec_time: rho = {spearman_sb:.4f} (p = {sp_sb:.2e})")
print(f"  io_concurrency vs exec_time: rho = {spearman_io:.4f} (p = {sp_io:.2e})")

# Interpretation
print(f"\nInterpretation:")
print(f"  shared_buffers: {'STRONG' if abs(spearman_sb) > 0.7 else 'MODERATE' if abs(spearman_sb) > 0.4 else 'WEAK'} effect")
print(f"  io_concurrency: {'STRONG' if abs(spearman_io) > 0.7 else 'MODERATE' if abs(spearman_io) > 0.4 else 'WEAK'} effect")

# =============================================================================
# METHOD 2: BINNED ANALYSIS (Effect Size)
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 2: BINNED ANALYSIS (Effect Size)")
print("=" * 70)

# Bin shared_buffers into categories
sb_bins = [0, 2, 4, 6, 8, 100]
sb_labels = ['<2GB', '2-4GB', '4-6GB', '6-8GB', '>8GB']
df['sb_bin'] = pd.cut(df['shared_buffers_gb'], bins=sb_bins, labels=sb_labels)

# Bin io_concurrency into categories
io_bins = [0, 100, 200, 300, 400, 1000]
io_labels = ['<100', '100-200', '200-300', '300-400', '>400']
df['io_bin'] = pd.cut(df['io_concurrency'], bins=io_bins, labels=io_labels)

print("\nExec Time by shared_buffers bin:")
sb_stats = df.groupby('sb_bin')['exec_time'].agg(['mean', 'std', 'count'])
for idx, row in sb_stats.iterrows():
    if row['count'] > 0:
        print(f"  {idx}: {row['mean']:.1f}s +/- {row['std']:.1f}s (n={int(row['count'])})")

print("\nExec Time by io_concurrency bin:")
io_stats = df.groupby('io_bin')['exec_time'].agg(['mean', 'std', 'count'])
for idx, row in io_stats.iterrows():
    if row['count'] > 0:
        print(f"  {idx}: {row['mean']:.1f}s +/- {row['std']:.1f}s (n={int(row['count'])})")

# Effect size: difference between best and worst bin
sb_effect = sb_stats['mean'].max() - sb_stats['mean'].min()
io_effect = io_stats['mean'].max() - io_stats['mean'].min()

print(f"\nEffect Size (best vs worst bin):")
print(f"  shared_buffers effect: {sb_effect:.1f}s")
print(f"  io_concurrency effect: {io_effect:.1f}s")
print(f"  Ratio: shared_buffers is {sb_effect/io_effect:.1f}x more impactful")

# =============================================================================
# 2D CROSS-TABULATION: Combined Effect
# =============================================================================
print("\n" + "=" * 70)
print("COMBINED EFFECT: 2D Analysis (shared_buffers x io_concurrency)")
print("=" * 70)

# Create pivot table: mean exec_time for each (sb_bin, io_bin) combination
pivot = df.pivot_table(
    values='exec_time',
    index='sb_bin',
    columns='io_bin',
    aggfunc=['mean', 'count'],
    observed=False
)

print("\nMean Execution Time (seconds) by bin combination:")
print("-" * 70)
print(pivot['mean'].round(1).to_string())

print("\nSample Count by bin combination:")
print("-" * 70)
print(pivot['count'].fillna(0).astype(int).to_string())

# Calculate interaction effect
print("\n" + "-" * 70)
print("KEY INSIGHTS:")
print("-" * 70)

# Compare: low sb with low io vs low sb with high io
low_sb_low_io = df[(df['shared_buffers_gb'] < 2) & (df['io_concurrency'] < 100)]['exec_time']
low_sb_high_io = df[(df['shared_buffers_gb'] < 2) & (df['io_concurrency'] > 300)]['exec_time']
high_sb_low_io = df[(df['shared_buffers_gb'] > 5) & (df['io_concurrency'] < 100)]['exec_time']
high_sb_high_io = df[(df['shared_buffers_gb'] > 5) & (df['io_concurrency'] > 300)]['exec_time']

print(f"\nCorner Analysis (extreme combinations):")
if len(low_sb_low_io) > 0:
    print(f"  Low sb + Low io:   {low_sb_low_io.mean():.1f}s (n={len(low_sb_low_io)})")
if len(low_sb_high_io) > 0:
    print(f"  Low sb + High io:  {low_sb_high_io.mean():.1f}s (n={len(low_sb_high_io)})")
if len(high_sb_low_io) > 0:
    print(f"  High sb + Low io:  {high_sb_low_io.mean():.1f}s (n={len(high_sb_low_io)})")
if len(high_sb_high_io) > 0:
    print(f"  High sb + High io: {high_sb_high_io.mean():.1f}s (n={len(high_sb_high_io)})")

# Effect of io within each sb level
print(f"\nEffect of io_concurrency WITHIN each shared_buffers level:")
for sb_bin in ['<2GB', '2-4GB', '4-6GB', '6-8GB']:
    subset = df[df['sb_bin'] == sb_bin]
    if len(subset) > 1:
        low_io_subset = subset[subset['io_concurrency'] < 150]['exec_time']
        high_io_subset = subset[subset['io_concurrency'] > 300]['exec_time']
        if len(low_io_subset) > 0 and len(high_io_subset) > 0:
            diff = low_io_subset.mean() - high_io_subset.mean()
            print(f"  {sb_bin}: low_io={low_io_subset.mean():.1f}s, high_io={high_io_subset.mean():.1f}s, diff={diff:.1f}s")
        else:
            # Try different thresholds
            io_corr = subset['io_concurrency'].corr(subset['exec_time'])
            print(f"  {sb_bin}: io correlation within bin = {io_corr:.3f} (n={len(subset)})")

# Effect of sb within each io level
print(f"\nEffect of shared_buffers WITHIN each io_concurrency level:")
for io_bin in ['<100', '100-200', '200-300', '300-400', '>400']:
    subset = df[df['io_bin'] == io_bin]
    if len(subset) > 1:
        low_sb_subset = subset[subset['shared_buffers_gb'] < 3]['exec_time']
        high_sb_subset = subset[subset['shared_buffers_gb'] > 5]['exec_time']
        if len(low_sb_subset) > 0 and len(high_sb_subset) > 0:
            diff = low_sb_subset.mean() - high_sb_subset.mean()
            print(f"  {io_bin}: low_sb={low_sb_subset.mean():.1f}s, high_sb={high_sb_subset.mean():.1f}s, diff={diff:.1f}s")
        else:
            sb_corr = subset['shared_buffers_gb'].corr(subset['exec_time'])
            print(f"  {io_bin}: sb correlation within bin = {sb_corr:.3f} (n={len(subset)})")

# =============================================================================
# METHOD 3: CONTROLLED COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 3: CONTROLLED COMPARISON")
print("=" * 70)

# Find configs with similar shared_buffers but different io_concurrency
print("\nHolding shared_buffers constant (~5.5-6GB), varying io_concurrency:")
mask_sb_constant = (df['shared_buffers_gb'] >= 5.5) & (df['shared_buffers_gb'] <= 6.5)
subset_sb = df[mask_sb_constant].copy()
if len(subset_sb) > 5:
    subset_sb_sorted = subset_sb.sort_values('io_concurrency')
    low_io = subset_sb_sorted[subset_sb_sorted['io_concurrency'] < 100]
    high_io = subset_sb_sorted[subset_sb_sorted['io_concurrency'] > 300]

    if len(low_io) > 0 and len(high_io) > 0:
        print(f"  Low io (<100):  {low_io['exec_time'].mean():.1f}s (n={len(low_io)})")
        print(f"  High io (>300): {high_io['exec_time'].mean():.1f}s (n={len(high_io)})")
        print(f"  Difference: {low_io['exec_time'].mean() - high_io['exec_time'].mean():.1f}s")
    else:
        print("  Not enough samples in low/high io bins")
else:
    print("  Not enough samples with sb in 5.5-6.5GB range")

# Find configs with similar io_concurrency but different shared_buffers
print("\nHolding io_concurrency constant (~150-250), varying shared_buffers:")
mask_io_constant = (df['io_concurrency'] >= 150) & (df['io_concurrency'] <= 250)
subset_io = df[mask_io_constant].copy()
if len(subset_io) > 5:
    low_sb = subset_io[subset_io['shared_buffers_gb'] < 3]
    high_sb = subset_io[subset_io['shared_buffers_gb'] > 5]

    if len(low_sb) > 0 and len(high_sb) > 0:
        print(f"  Low sb (<3GB):  {low_sb['exec_time'].mean():.1f}s (n={len(low_sb)})")
        print(f"  High sb (>5GB): {high_sb['exec_time'].mean():.1f}s (n={len(high_sb)})")
        print(f"  Difference: {low_sb['exec_time'].mean() - high_sb['exec_time'].mean():.1f}s")
    else:
        print("  Not enough samples in low/high sb bins")
else:
    print("  Not enough samples with io in 150-250 range")

# =============================================================================
# METHOD 4: EXTREME POINTS ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 4: EXTREME POINTS ANALYSIS")
print("=" * 70)

# Best and worst configs
best_idx = df['exec_time'].idxmin()
worst_idx = df['exec_time'].idxmax()

print(f"\nBest configuration:")
print(f"  shared_buffers: {df.loc[best_idx, 'shared_buffers_gb']:.2f} GB")
print(f"  io_concurrency: {df.loc[best_idx, 'io_concurrency']:.0f}")
print(f"  exec_time: {df.loc[best_idx, 'exec_time']:.1f}s")

print(f"\nWorst configuration:")
print(f"  shared_buffers: {df.loc[worst_idx, 'shared_buffers_gb']:.2f} GB")
print(f"  io_concurrency: {df.loc[worst_idx, 'io_concurrency']:.0f}")
print(f"  exec_time: {df.loc[worst_idx, 'exec_time']:.1f}s")

print(f"\nPerformance range: {df['exec_time'].max() - df['exec_time'].min():.1f}s ({(df['exec_time'].max()/df['exec_time'].min() - 1)*100:.1f}% difference)")

# =============================================================================
# METHOD 5: REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 5: REGRESSION ANALYSIS")
print("=" * 70)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Standardize features so coefficients are comparable
scaler = StandardScaler()
X = df[['shared_buffers_gb', 'io_concurrency']].values
X_scaled = scaler.fit_transform(X)
y = df['exec_time'].values

# Fit linear regression
lr = LinearRegression()
lr.fit(X_scaled, y)

print(f"\nLinear Regression (standardized coefficients):")
print(f"  shared_buffers coefficient: {lr.coef_[0]:.4f}")
print(f"  io_concurrency coefficient: {lr.coef_[1]:.4f}")
print(f"  R-squared: {lr.score(X_scaled, y):.4f}")

# Relative importance from standardized coefficients
total_coef = abs(lr.coef_[0]) + abs(lr.coef_[1])
sb_pct = abs(lr.coef_[0]) / total_coef * 100
io_pct = abs(lr.coef_[1]) / total_coef * 100

print(f"\nRelative Importance (from standardized coefficients):")
print(f"  shared_buffers: {sb_pct:.1f}%")
print(f"  io_concurrency: {io_pct:.1f}%")

# =============================================================================
# METHOD 6: VARIANCE DECOMPOSITION (ANOVA-style)
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 6: VARIANCE DECOMPOSITION")
print("=" * 70)

# Total variance
total_var = df['exec_time'].var()

# Variance explained by shared_buffers alone
from sklearn.linear_model import LinearRegression
lr_sb = LinearRegression()
lr_sb.fit(df[['shared_buffers_gb']], df['exec_time'])
var_explained_sb = lr_sb.score(df[['shared_buffers_gb']], df['exec_time']) * total_var

# Variance explained by io_concurrency alone
lr_io = LinearRegression()
lr_io.fit(df[['io_concurrency']], df['exec_time'])
var_explained_io = lr_io.score(df[['io_concurrency']], df['exec_time']) * total_var

# Variance explained by both
lr_both = LinearRegression()
lr_both.fit(df[['shared_buffers_gb', 'io_concurrency']], df['exec_time'])
var_explained_both = lr_both.score(df[['shared_buffers_gb', 'io_concurrency']], df['exec_time']) * total_var

print(f"\nTotal variance: {total_var:.2f}")
print(f"Variance explained by shared_buffers alone: {var_explained_sb:.2f} ({var_explained_sb/total_var*100:.1f}%)")
print(f"Variance explained by io_concurrency alone: {var_explained_io:.2f} ({var_explained_io/total_var*100:.1f}%)")
print(f"Variance explained by both: {var_explained_both:.2f} ({var_explained_both/total_var*100:.1f}%)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
Based on {len(df)} MLOS tuning samples:

1. SHARED_BUFFERS is the PRIMARY factor:
   - Spearman correlation: {spearman_sb:.3f}
   - Variance explained: {lr_sb.score(df[['shared_buffers_gb']], df['exec_time'])*100:.1f}%
   - Effect size: {sb_effect:.1f}s between best/worst bins

2. IO_CONCURRENCY has a SECONDARY effect:
   - Spearman correlation: {spearman_io:.3f}
   - Variance explained: {lr_io.score(df[['io_concurrency']], df['exec_time'])*100:.1f}%
   - Effect size: {io_effect:.1f}s between best/worst bins

3. OPTIMAL CONFIGURATION:
   - shared_buffers: {df.loc[best_idx, 'shared_buffers_gb']:.1f} GB (>5GB recommended for 5.3GB index)
   - io_concurrency: {df.loc[best_idx, 'io_concurrency']:.0f} (>100 recommended)

4. PERFORMANCE GAIN from tuning: {(df['exec_time'].max()/df['exec_time'].min() - 1)*100:.1f}%
""")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Scatter plot: shared_buffers vs exec_time
ax1 = axes[0, 0]
ax1.scatter(df['shared_buffers_gb'], df['exec_time'], alpha=0.6, c='blue')
ax1.set_xlabel('shared_buffers (GB)')
ax1.set_ylabel('Execution Time (s)')
ax1.set_title(f'shared_buffers vs exec_time (r={spearman_sb:.3f})')

# Scatter plot: io_concurrency vs exec_time
ax2 = axes[0, 1]
ax2.scatter(df['io_concurrency'], df['exec_time'], alpha=0.6, c='green')
ax2.set_xlabel('effective_io_concurrency')
ax2.set_ylabel('Execution Time (s)')
ax2.set_title(f'io_concurrency vs exec_time (r={spearman_io:.3f})')

# Box plot by shared_buffers bins
ax3 = axes[1, 0]
df.boxplot(column='exec_time', by='sb_bin', ax=ax3)
ax3.set_xlabel('shared_buffers bin')
ax3.set_ylabel('Execution Time (s)')
ax3.set_title('Exec Time by shared_buffers')
plt.suptitle('')

# Box plot by io_concurrency bins
ax4 = axes[1, 1]
df.boxplot(column='exec_time', by='io_bin', ax=ax4)
ax4.set_xlabel('io_concurrency bin')
ax4.set_ylabel('Execution Time (s)')
ax4.set_title('Exec Time by io_concurrency')
plt.suptitle('')

# 2D Scatter: colored by exec_time
ax5 = axes[0, 2]
scatter = ax5.scatter(df['shared_buffers_gb'], df['io_concurrency'],
                      c=df['exec_time'], cmap='RdYlGn_r', alpha=0.7, s=50)
ax5.set_xlabel('shared_buffers (GB)')
ax5.set_ylabel('io_concurrency')
ax5.set_title('Combined Effect (color = exec_time)')
plt.colorbar(scatter, ax=ax5, label='Exec Time (s)')

# Heatmap: mean exec_time by (sb_bin, io_bin)
ax6 = axes[1, 2]
heatmap_data = df.pivot_table(values='exec_time', index='sb_bin', columns='io_bin',
                               aggfunc='mean', observed=False)
im = ax6.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto')
ax6.set_xticks(range(len(heatmap_data.columns)))
ax6.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
ax6.set_yticks(range(len(heatmap_data.index)))
ax6.set_yticklabels(heatmap_data.index)
ax6.set_xlabel('io_concurrency bin')
ax6.set_ylabel('shared_buffers bin')
ax6.set_title('Heatmap: Mean Exec Time')
plt.colorbar(im, ax=ax6, label='Exec Time (s)')

# Add text annotations to heatmap
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax6.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if val > 170 else 'black')

plt.tight_layout()
plt.savefig('knob_analysis.png', dpi=150)
print("\nVisualization saved to: knob_analysis.png")
plt.show()
