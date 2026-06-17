#!/usr/bin/env python3
"""
paper_plots.py — single source for the paper's figures.

Modeled on maxvec's scripts/final_paper_plots.py: each figure has its own
make_plotN() function, a thin main() driver selects which to render with
--plots, and the save helpers encode the main-vs-variant convention.

Output convention (see CLAUDE.md "Final Plots"):
  MAIN figures    -> {out_dir}/{stem}.pdf  (paper-quality vector)
                   + {out_dir}/{stem}.png  (quick raster preview)
  VARIANT figures -> {out_dir}/alt/{plotN}/{stem}.jpeg  (jpeg only)

Naming:
  plotN_<descriptive>_<dataset/index/scenario>   for main figures
  plotN<letter>_<descriptive>                     for variants (a, b, c, ...)

plotN <-> meaning:
  plot1 = redundant computation optimization
  plot2 = parallel IVF
  plot3 = recall vs QPS
  plot4 = interference throughput frontier, 10GB SB / 15GB CG (constrained)
  plot5 = interference throughput frontier, 45GB SB / 65GB CG (fits in memory)
  plot6 = HTAP fixed-R / fixed-V line families per index, 10/15 (-> plot4)
  plot7 = HTAP fixed-R / fixed-V line families per index, 45/65 (-> plot5)

Usage:
  python3 scripts/paper_plots.py                 # builds 1..7 into scripts/final_plots/
  python3 scripts/paper_plots.py --plot 1        # plot 1 main + all its variants
  python3 scripts/paper_plots.py --plot 2a,3b    # only those two variants
  python3 scripts/paper_plots.py --out_dir /tmp/figs --plots 1,3
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paper-friendly defaults. Keep these few so the file stays easy to tweak.
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType, no Type-3 (paper requirement)
plt.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Frozen copy of the raw data behind the figures. See data/final_data/README.md:
# whenever the data changes, that copy must be updated so figures stay
# reproducible. Point --data_dir elsewhere only for one-off experiments.
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "final_data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "final_plots"
DEFAULT_DPI = 300   # paper-grade raster resolution (PNG/JPEG); PDF stays vector

# Near-lossless JPEG settings for the variant previews (paper-worthy quality):
# top quality, no chroma subsampling so thin coloured lines stay crisp.
JPEG_PIL_KWARGS = {"quality": 95, "subsampling": 0}

_PLOT_NUM_RE = re.compile(r"^(plot\d+)")

# ---------------------------------------------------------------------------
# Shared style config — edit here to restyle every plot at once.
#
# Okabe-Ito colourblind-safe palette (copied from maxvec final_plots.py).
# ---------------------------------------------------------------------------

# Okabe-Ito colourblind-safe palette. The (R,G,B) of each hex is in the comment
# for future reference. The three we use for the paper's index/parallel scheme:
#   green = IVFFlat, blue = HNSW, orange = Parallel IVFFlat (canonical/main).
OKABE_ITO = {
    "blue":       "#0072B2",  # (0, 114, 178)    <- HNSW
    "orange":     "#E69F00",  # (230, 159, 0)    <- Parallel IVFFlat (main)
    "green":      "#009E73",  # (0, 158, 115)    <- IVFFlat
    "purple":     "#CC79A7",  # (204, 121, 167)
    "yellow":     "#F0E442",  # (240, 228, 66)
    "vermillion": "#D55E00",  # (213, 94, 0)
    "sky":        "#56B4E9",  # (86, 180, 233)
    "black":      "#000000",  # (0, 0, 0)
}

# Global index/parallel palette: IVFFlat green, HNSW blue, parallel IVFFlat an
# orange family. One hue per concept so each reads as a group across every figure.
INDEX_COLORS = {"ivfflat": OKABE_ITO["green"], "hnsw": OKABE_ITO["blue"]}
INDEX_MARKERS = {"ivfflat": "o", "hnsw": "s"}
INDEX_LABELS = {"ivfflat": "IVFFlat", "hnsw": "HNSW"}

# Parallel-IVFFlat orange family. The base/default parallel colour is Okabe-Ito
# orange #E69F00 = (230, 159, 0); the worker sweep starts there and gets DARKER
# as the worker count rises (w=4 -> w=8 -> w=12), so the counts read as one
# "parallel" group, distinguished by shade + marker (+ linestyle in plot3).
PARALLEL_ORANGES = {
    "4":  OKABE_ITO["orange"],  # (230, 159, 0)  Okabe-Ito orange (base)
    "8":  "#B57A00",            # (181, 122, 0)  darker
    "12": "#804D00",            # (128, 77, 0)   darkest
}

# Parallel-worker colours, consistent across all plots (plot2, plot3).
# "default" = single worker = the plain IVFFlat baseline, so it shares IVFFlat's
# green; the parallel worker counts take the orange family.
WORKER_COLORS = {
    "default": INDEX_COLORS["ivfflat"],
    "4":       PARALLEL_ORANGES["4"],
    "8":       PARALLEL_ORANGES["8"],
    "12":      PARALLEL_ORANGES["12"],
}
WORKER_MARKERS = {"default": "o", "4": "^", "8": "v", "12": "D"}
WORKER_LABELS = {"default": "1 worker", "4": "4 workers", "8": "8 workers",
                 "12": "12 workers"}

# Cache-mode panel labels, shared across every plot: cold = served from disk,
# warm = served from memory.
MODE_LABELS = {"cold": "cold (disk)", "warm": "warm (in-mem)"}

# Index colour for the throughput-frontier plot (plot4): three indexes share the
# global hues. Parallel IVFFlat is a single series here, so it takes the
# canonical Okabe-Ito orange #E69F00 = (230, 159, 0).
FRONTIER_COLORS = {
    "ivfflat":          INDEX_COLORS["ivfflat"],
    "hnsw":             INDEX_COLORS["hnsw"],
    "parallel_ivfflat": OKABE_ITO["orange"],
}
FRONTIER_MARKERS = {"ivfflat": "o", "hnsw": "s", "parallel_ivfflat": "^"}
FRONTIER_LABELS = {"ivfflat": "IVFFlat", "hnsw": "HNSW",
                   "parallel_ivfflat": "Parallel IVFFlat"}

# With/without optimization (plot1): colour by index, distinguish opt by hatch.
OPT_HATCH = {"without": "", "with": "//"}

# ---------------------------------------------------------------------------
# Figure sizing for the PVLDB two-column layout (inches). Every figure must
# either fit one column (COL_WIDTH) or span both columns (FULL_WIDTH), so it
# drops into the paper without rescaling (which would shrink the fonts).
#   - side-by-side panels  -> FULL_WIDTH (spans two columns)
#   - stacked panels       -> COL_WIDTH  (fits one column)
# ---------------------------------------------------------------------------
COL_WIDTH = 3.33
FULL_WIDTH = 7.0

# Shared font sizes (absolute pt) + line/marker styling, applied across every
# figure via rcParams so one edit here restyles all plots. Because figures are
# drawn at their true target width, these pt sizes render identically whether a
# figure is one column or two. Annotations have no rcParam -> read ANNOT_FS.
SUPTITLE_FS = 9
TITLE_FS = 8
LABEL_FS = 8
TICK_FS = 7
LEGEND_FS = 6.5
ANNOT_FS = 7.5          # speedup / gain-% text
ANNOT_WEIGHT = "normal"

LINE_WIDTH = 1.6
MARKER_SIZE = 4.5
BAR_EDGE_WIDTH = 0.5

# Common-legend + title band spacing, tuned per layout (see _common_legend).
# legend_y / suptitle_y are figure fractions (>1 allowed; bbox_inches="tight"
# keeps overflow). rect_top is the top of the axes region: raise it to shrink
# the gap below the legend, lower it to add room.
LEGEND_BAND = {
    "side":    {"legend_y": 0.965, "suptitle_y": 1.02, "rect_top": 0.90},
    "stacked": {"legend_y": 0.965, "suptitle_y": 1.01, "rect_top": 0.93},
    # Recall-vs-QPS stacked plots carry a 3-row legend (5 series, 2 cols), so
    # they need a deeper top band than the shared "stacked" so the legend clears
    # the first panel's "cold cache" title.
    "recall_stacked": {"legend_y": 0.97, "suptitle_y": 1.02, "rect_top": 0.895},
}

plt.rcParams["hatch.linewidth"] = 1.0   # bolder hatch so 'w/ opt.' pattern reads
plt.rcParams.update({
    "axes.titlesize":   TITLE_FS,
    "axes.labelsize":   LABEL_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "legend.fontsize":  LEGEND_FS,
    "figure.titlesize": SUPTITLE_FS,
})


def _speedup_label(ax, x, y, speed, *, dy=3, below=False):
    """Horizontal speedup tag '<n>x' near a point/bar (shared style). Set
    below=True to drop it under the marker instead of above."""
    yoff = -dy if below else dy
    ax.annotate(f"{speed:.1f}x", (x, y), textcoords="offset points",
                xytext=(0, yoff), ha="center", va=("top" if below else "bottom"),
                fontsize=ANNOT_FS, fontweight=ANNOT_WEIGHT, rotation=0)


def _common_legend(fig, handles, labels, title, *, ncol, layout):
    """One frameless figure-level legend at top + suptitle, with axes packed
    under it. Spacing comes from LEGEND_BAND[layout] so it is tunable per plot."""
    band = LEGEND_BAND[layout]
    # Lay out the axes first (so tight_layout doesn't also reserve space for the
    # suptitle and over-shrink), then overlay title + legend in the top margin.
    fig.tight_layout(rect=(0, 0, 1, band["rect_top"]))
    if title:
        fig.suptitle(title, fontweight="bold", y=band["suptitle_y"])
    # Center the legend over the plot box, not the figure: the y-axis label and
    # tick labels push the axes right of figure-center, so anchoring at x=0.5
    # would leave the legend visibly left of the plots.
    boxes = [ax.get_position() for ax in fig.axes if ax.get_visible()]
    legend_x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(legend_x, band["legend_y"]), ncol=ncol, frameon=False,
               fontsize=LEGEND_FS, handlelength=1.6, handletextpad=0.4,
               columnspacing=1.2)


# ---------------------------------------------------------------------------
# Save helpers — the main vs variant mechanism.
#
# Granular selection: the driver may set _SELECTION to a set of tokens so a
# partial run emits only some figures. A token is a plot number ("1") meaning
# "the main + every variant of that plot", or number+letter ("2a") meaning just
# that one variant. Each save_* call derives its stem's token and skips when the
# selection excludes it, so makers can run whole but emit only what was asked.
# _SELECTION = None means emit everything (full run).
# ---------------------------------------------------------------------------

_SELECTION = None
# Optional override for plot4's frontier source dir (set from --frontier_dir).
# plot5 always reads its default scenario dir.
_FRONTIER_DIR = None
_STEM_TOKEN_RE = re.compile(r"^plot(\d+)([a-z]?)")


def _should_emit(stem):
    """True if `stem` is wanted under the current _SELECTION. A figure passes
    when its whole-plot number is selected (e.g. '1' wants plot1 + variants) or
    its exact variant token is selected (e.g. '2a')."""
    if _SELECTION is None:
        return True
    m = _STEM_TOKEN_RE.match(stem)
    if not m:
        return True
    num, token = m.group(1), m.group(1) + m.group(2)
    return num in _SELECTION or token in _SELECTION


def save_main(fig, out_dir, stem, *, dpi=DEFAULT_DPI):
    """A main paper figure: emit BOTH a vector .pdf and a raster .png."""
    if not _should_emit(stem):
        plt.close(fig)
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = out_dir / f"{stem}{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"  -> {path}")
    plt.close(fig)


def save_variant(fig, out_dir, stem, *, dpi=DEFAULT_DPI):
    """An alternate figure: emit a single .jpeg under alt/{plotN}/."""
    if not _should_emit(stem):
        plt.close(fig)
        return
    m = _PLOT_NUM_RE.match(stem)
    plot_dir = m.group(1) if m else "misc"
    alt_dir = Path(out_dir) / "alt" / plot_dir
    alt_dir.mkdir(parents=True, exist_ok=True)
    path = alt_dir / f"{stem}.jpeg"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs=JPEG_PIL_KWARGS)
    print(f"  -> {path}")
    plt.close(fig)


def write_plot_csv(out_dir, n, header, rows):
    """Write the exact plotted values to {out_dir}/csv/plot_{n}.csv.

    Light, pandas-free companion so every main plot has an auditable table."""
    csv_dir = Path(out_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"plot_{n}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {path}")


# ===========================================================================
# Plot 1 — redundant computation optimization.
#
# Source: avoid_reduntant_computations_optimization/data/
#   results_{with,without}_optimization_{IVFFLAT,HNSW}_{dataset}_{cold,warm}.txt
# Each line: "K=<k>, exec_time=<seconds>"  (1000 queries at recall 0.9, GIST).
#
# MAIN = single 2x2 figure: IVFFlat / HNSW (rows) x cold / warm (cols).
# ===========================================================================

PLOT1_INDEXES = ["ivfflat", "hnsw"]
PLOT1_MODES = ["cold", "warm"]


def _load_opt_results(path):
    """Parse a results_*.txt file -> (k_vals, times)."""
    k_vals, times = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            k_vals.append(int(parts[0].split("=")[1]))
            times.append(float(parts[1].split("=")[1]))
    return k_vals, times


def _draw_opt_panel(ax, index_type, dataset, mode, data_dir):
    """Draw one (index, cache-mode) panel. Returns (csv_rows, sources)."""
    idx = index_type.upper()
    without = data_dir / f"results_without_optimization_{idx}_{dataset}_{mode}.txt"
    with_ = data_dir / f"results_with_optimization_{idx}_{dataset}_{mode}.txt"
    if not (without.exists() and with_.exists()):
        ax.set_visible(False)
        return [], []

    k_no, t_no = _load_opt_results(without)
    k_opt, t_opt = _load_opt_results(with_)
    gains = [((a - b) / a * 100) if a > 0 else 0 for a, b in zip(t_no, t_opt)]

    ax.plot(k_no, t_no, "^--", color="r", label="Without optimization",
            markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    ax.plot(k_opt, t_opt, "s-", color="g", label="With optimization",
            markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    for k, t, g in zip(k_opt, t_opt, gains):
        if g > 0:
            ax.annotate(f"{g:.0f}%", (k, t), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=ANNOT_FS - 1.5,
                        color="green")
    ax.set_xlabel("top-K")
    ax.set_ylabel("Exec. Time (s)")
    ax.set_title(f"{idx} — {MODE_LABELS[mode]}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    csv_rows = [[idx, dataset, mode, k, a, b, round(g, 2)]
                for k, a, b, g in zip(k_no, t_no, t_opt, gains)]
    return csv_rows, [without, with_]


def make_plot1(data_dir, out_dir, dpi=DEFAULT_DPI):
    """Render plot 1 (main: stacked barplot, single-column) + csv/plot_1.csv +
    variants: 2x2 line panel (plot1a) and side-by-side barplot (plot1b)."""
    opt_dir = Path(data_dir) / "avoid_reduntant_computations_optimization" / "data"
    dataset = "gist"
    csv_rows, sources = [], []

    # Variant plot1a: 2x2 line panel (IVFFlat/HNSW rows x cold/warm cols),
    # spans two columns. Also the source of csv/plot_1.csv (layout-independent).
    fig, axes = plt.subplots(len(PLOT1_INDEXES), len(PLOT1_MODES),
                             figsize=(FULL_WIDTH, 4.2), squeeze=False)
    for i, idx in enumerate(PLOT1_INDEXES):
        for j, mode in enumerate(PLOT1_MODES):
            rows, srcs = _draw_opt_panel(axes[i][j], idx, dataset, mode, opt_dir)
            csv_rows += rows
            sources += srcs

    fig.suptitle(f"Redundant computation optimization ({dataset.upper()})",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save_variant(fig, out_dir,
                 f"plot1a_redundant_computation_optimization_lineplot_{dataset}", dpi=dpi)

    if csv_rows:
        write_plot_csv(out_dir, 1,
                       ["index", "dataset", "mode", "k",
                        "time_without_s", "time_with_s", "gain_pct"],
                       csv_rows)

    # Main: stacked barplot fits one column. Dataset + recall live in the name
    # (the in-figure title and recall annotation are dropped).
    _render_plot1_barplot(opt_dir, dataset, "stacked", out_dir, dpi,
                          f"plot1_redundant_computation_optimization_barplot_stacked_{dataset}_recall0.9",
                          is_main=True)
    # Variant plot1b: side-by-side barplot spans two columns.
    _render_plot1_barplot(opt_dir, dataset, "side", out_dir, dpi,
                          f"plot1b_redundant_computation_optimization_barplot_{dataset}")
    return sources


# Per K, 4 bars = {IVFFlat, HNSW} x {without, with optimization}.
# Colour = index, hatch = optimization. Bar label = optimization speedup.
PLOT1_BAR_ORDER = [("ivfflat", "without"), ("ivfflat", "with"),
                   ("hnsw", "without"), ("hnsw", "with")]


def _render_plot1_barplot(opt_dir, dataset, layout, out_dir, dpi, stem, is_main=False):
    """Grouped-bar rendering of plot 1. layout='side' -> cold|warm two-column;
    layout='stacked' -> cold/warm stacked single-column. y starts at 0; the
    'with' bars carry the optimization speedup (without/with). is_main selects
    save_main (vector PDF + PNG) vs save_variant (JPEG)."""
    modes = ["cold", "warm"]
    if layout == "side":
        fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9), squeeze=False)
        flat = list(axes[0])
    else:
        fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.6), squeeze=False)
        flat = [axes[0][0], axes[1][0]]
    # Cluster the 4 bars as two per-index pairs (w/o + w/ opt touching) with a
    # gap between the pairs; offsets indexed by PLOT1_BAR_ORDER position.
    width = 0.16
    pair_gap = 0.12
    bar_off = {0: -(1.5 * width + pair_gap / 2), 1: -(0.5 * width + pair_gap / 2),
               2:  (0.5 * width + pair_gap / 2), 3:  (1.5 * width + pair_gap / 2)}
    drew = False
    for ax, mode in zip(flat, modes):
        series, k_ref = {}, None
        for idx, opt in PLOT1_BAR_ORDER:
            tag = "with_optimization" if opt == "with" else "without_optimization"
            fpath = opt_dir / f"results_{tag}_{idx.upper()}_{dataset}_{mode}.txt"
            if fpath.exists():
                ks, ts = _load_opt_results(fpath)
                series[(idx, opt)] = ts
                k_ref = ks
        if not series:
            ax.set_visible(False)
            continue
        drew = True
        x = np.arange(len(k_ref))
        for b, (idx, opt) in enumerate(PLOT1_BAR_ORDER):
            if (idx, opt) not in series:
                continue
            offset = bar_off[b]
            # 'with opt' bars are slightly transparent (and hatched) to read as
            # the optimized counterpart of the same-colour 'without' bar.
            ax.bar(x + offset, series[(idx, opt)], width,
                   color=INDEX_COLORS[idx], hatch=OPT_HATCH[opt],
                   alpha=(0.6 if opt == "with" else 1.0),
                   edgecolor="black", linewidth=BAR_EDGE_WIDTH)
            # Speedup (without/with) above the taller w/o-opt bar of the pair,
            # where there is open space, rather than on the short optimized bar.
            if opt == "without" and (idx, "with") in series:
                # In warm cache HNSW bars are short and sit right of the tall
                # IVFFlat bars, so nudge their label right to clear them.
                nudge = 0.85 * width if (mode == "warm" and idx == "hnsw") else 0.0
                for xi, (t_wo, t_w) in enumerate(zip(series[(idx, "without")],
                                                     series[(idx, "with")])):
                    if t_w:
                        _speedup_label(ax, x[xi] + offset + nudge, t_wo,
                                       t_wo / t_w, dy=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_ref])
        ax.set_xlabel("top-K")
        ax.set_ylabel("Exec. Time (s)")
        ax.set_title(MODE_LABELS[mode])
        # Headroom above the tallest bar so its speedup label stays inside the
        # axes (the labels sit on top of the w/o-opt bars).
        maxh = max((max(v) for v in series.values()), default=1)
        ax.set_ylim(0, maxh * 1.22)
        ax.grid(True, axis="y", alpha=0.3)

    if not drew:
        plt.close(fig)
        return
    from matplotlib.patches import Patch
    # Single-row legend: colour = index; the hatched swatch flags 'w/ opt.'
    # (solid bars are the unoptimized baseline). Dense hatch so the pattern reads.
    handles = [
        Patch(facecolor=INDEX_COLORS["ivfflat"], edgecolor="black", label="IVFFlat"),
        Patch(facecolor=INDEX_COLORS["hnsw"], edgecolor="black", label="HNSW"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="w/ opt."),
    ]
    # Drop the title on the main figure (it moves to the filename).
    title = "" if is_main else f"Redundant computation optimization ({dataset.upper()})"
    _common_legend(fig, handles, [h.get_label() for h in handles],
                   title, ncol=3, layout=layout)
    if is_main:
        save_main(fig, out_dir, stem, dpi=dpi)
    else:
        save_variant(fig, out_dir, stem, dpi=dpi)


# ===========================================================================
# Plot 2 — parallel IVF.
#
# Source: parallel_ivf/comparison_results_workers_{dataset}_{mode}_K_100.json
#   { "default": [{target_recall, probes, total_time_s, avg_time_ms}, ...],
#     "parallel_by_workers": {"4": [...], "8": [...], "12": [...]} }
#
# MAIN = single figure, cold + warm side by side, avg latency vs recall,
#        one line per worker count plus the single-worker default.
# ===========================================================================

PLOT2_SERIES = ["default", "4", "8", "12"]  # single worker + parallel worker counts


def _plot2_series_points(data, key):
    """Latency points for one worker series -> list[{target_recall, avg_time_ms}]."""
    return data["default"] if key == "default" \
        else data.get("parallel_by_workers", {}).get(key)


def make_plot2(data_dir, out_dir, dpi=DEFAULT_DPI):
    """Render plot 2 (main: stacked barplot, single-column) + csv/plot_2.csv +
    variants: latency-vs-recall lineplot (plot2a) and side-by-side barplot
    (plot2b)."""
    pdir = Path(data_dir) / "parallel_ivf"
    dataset = "gist"
    panels = []  # (mode, data)
    sources = []
    for mode in ("cold", "warm"):
        fpath = pdir / f"comparison_results_workers_{dataset}_{mode}_K_100.json"
        if fpath.exists():
            with open(fpath) as f:
                panels.append((mode, json.load(f)))
            sources.append(fpath)
    if not panels:
        print("  [plot2] no parallel_ivf data found, skipping")
        return []

    lists = panels[0][1].get("lists", "NA")

    # Variant plot2a: cold|warm lineplot spans two columns, one common legend.
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 2.9),
                             squeeze=False, sharey=True)
    csv_rows = []
    handles, labels = [], []
    for ax, (mode, data) in zip(axes[0], panels):
        base = {p["target_recall"]: p["avg_time_ms"] for p in data["default"]}
        for key in PLOT2_SERIES:
            points = _plot2_series_points(data, key)
            if not points:
                continue
            recalls = [p["target_recall"] for p in points]
            latency = [p["avg_time_ms"] for p in points]
            line, = ax.plot(recalls, latency, color=WORKER_COLORS[key],
                    marker=WORKER_MARKERS[key], linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, markeredgewidth=0.4,
                    markeredgecolor="black", label=WORKER_LABELS[key])
            if WORKER_LABELS[key] not in labels:
                handles.append(line)
                labels.append(WORKER_LABELS[key])
            for r, lat in zip(recalls, latency):
                csv_rows.append([dataset, mode, WORKER_LABELS[key], r, lat])
            # Speedup label below the fastest series (w=12), vs single worker.
            if key == "12":
                for r, lat in zip(recalls, latency):
                    if base.get(r):
                        _speedup_label(ax, r, lat, base[r] / lat, below=True, dy=5)
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Avg. query latency (ms)")
        ax.set_title(MODE_LABELS[mode])
        ax.set_ylim(bottom=0)
        ax.margins(y=0.10)
        ax.grid(True, alpha=0.3)

    _common_legend(fig, handles, labels,
                   f"Parallel IVFFlat: latency vs recall ({dataset.upper()})",
                   ncol=len(labels), layout="side")
    save_variant(fig, out_dir, f"plot2a_parallel_ivf_latency_vs_recall_{dataset}", dpi=dpi)

    write_plot_csv(out_dir, 2,
                   ["dataset", "mode", "series", "target_recall", "avg_latency_ms"],
                   csv_rows)

    # Main: stacked barplot fits one column. Dataset + lists live in the name
    # (the in-figure title and lists annotation are dropped).
    _render_plot2_barplot(panels, dataset, "stacked", out_dir, dpi,
                          f"plot2_parallel_ivf_latency_barplot_stacked_{dataset}_lists{lists}",
                          is_main=True)
    # Variant plot2b: side-by-side barplot spans two columns.
    _render_plot2_barplot(panels, dataset, "side", out_dir, dpi,
                          f"plot2b_parallel_ivf_latency_barplot_{dataset}")
    return sources


def _render_plot2_barplot(panels, dataset, layout, out_dir, dpi, stem, is_main=False):
    """Grouped-bar rendering of plot 2. layout='side' -> cold|warm two-column;
    layout='stacked' -> cold/warm stacked single-column. Bars per recall =
    {1 worker, w=4, w=8, w=12}; each bar carries its speedup vs 1 worker
    (horizontal, shared style). y starts at 0. is_main selects save_main
    (vector PDF + PNG) vs save_variant (JPEG)."""
    if layout == "side":
        fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 2.9),
                                 squeeze=False, sharey=True)
        flat = list(axes[0])
    else:
        fig, axes = plt.subplots(len(panels), 1, figsize=(COL_WIDTH, 4.6),
                                 squeeze=False)
        flat = [axes[r][0] for r in range(len(panels))]
    width = 0.2
    for ax, (mode, data) in zip(flat, panels):
        recalls = [p["target_recall"] for p in data["default"]]
        base = [p["avg_time_ms"] for p in data["default"]]
        x = np.arange(len(recalls))
        for b, key in enumerate(PLOT2_SERIES):
            points = _plot2_series_points(data, key)
            if not points:
                continue
            lat = [p["avg_time_ms"] for p in points]
            offset = (b - (len(PLOT2_SERIES) - 1) / 2) * width
            ax.bar(x + offset, lat, width, color=WORKER_COLORS[key],
                   edgecolor="black", linewidth=BAR_EDGE_WIDTH,
                   label=WORKER_LABELS[key])
            # Speedup vs 1 worker (skip the trivial 1.0x baseline). The side
            # (two-column) layout has room to label every parallel bar; the
            # stacked single column only fits the fastest (w=12) cleanly.
            label_this = (key == "12") if layout == "stacked" else (key != "default")
            if label_this:
                # Stacked (main): only w=12 is labelled; sit it a bit higher
                # above the bar. Side variant keeps its per-series spacing.
                dy = 7 if layout == "stacked" else (9 if key == "8" else 2)
                for xi, (l, bl) in enumerate(zip(lat, base)):
                    if l:
                        _speedup_label(ax, x[xi] + offset, l, bl / l, dy=dy)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in recalls])
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Avg. query latency (ms)")
        ax.set_title(MODE_LABELS[mode])
        ax.set_ylim(bottom=0)
        ax.margins(y=0.14)
        ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=WORKER_COLORS[k], edgecolor="black",
                     label=WORKER_LABELS[k]) for k in PLOT2_SERIES]
    title = "" if is_main else f"Parallel IVFFlat: latency vs recall ({dataset.upper()})"
    # One row (all worker counts on a single line).
    _common_legend(fig, handles, [h.get_label() for h in handles],
                   title, ncol=len(handles), layout=layout)
    if is_main:
        save_main(fig, out_dir, stem, dpi=dpi)
    else:
        save_variant(fig, out_dir, stem, dpi=dpi)


# ===========================================================================
# Plot 3 — recall vs QPS.
#
# Source: recall_qps_plots/results_recall_qps/
#   recall_qps_{dataset}_{ivfflat,hnsw}_{mode}.json
#   recall_qps_{dataset}_parallel_ivfflat_w{N}_{mode}.json
# Each series JSON has a "full_sweep" list of {recall, qps, ...}.
#
# MAIN     = GIST, cold + warm side by side (two-column).
# VARIANTS = GIST stacked (one-column), Deep side, Deep stacked.
# All share one common (figure-level) legend.
# ===========================================================================

# Colours/markers reuse the shared config so IVF/HNSW and worker counts match
# across every plot. The parallel-IVFFlat sweep is one orange hue distinguished
# by shade + marker + linestyle (4th field) so the three worker counts read as
# one "parallel" group; index series are solid.
PLOT3_STYLES = {
    "ivfflat":              (INDEX_COLORS["ivfflat"], INDEX_MARKERS["ivfflat"], "IVFFlat", "-"),
    "hnsw":                 (INDEX_COLORS["hnsw"], INDEX_MARKERS["hnsw"], "HNSW", "-"),
    "parallel_ivfflat_w4":  (WORKER_COLORS["4"], WORKER_MARKERS["4"], "Parallel IVFFlat (w=4)", "-"),
    "parallel_ivfflat_w8":  (WORKER_COLORS["8"], WORKER_MARKERS["8"], "Parallel IVFFlat (w=8)", "--"),
    "parallel_ivfflat_w12": (WORKER_COLORS["12"], WORKER_MARKERS["12"], "Parallel IVFFlat (w=12)", ":"),
}
PLOT3_ORDER = ["ivfflat", "hnsw",
               "parallel_ivfflat_w4", "parallel_ivfflat_w8", "parallel_ivfflat_w12"]

# full_sweep includes very-low-recall points (huge QPS) that would inflate the
# y-axis. We only show recall >= this, so the axis tracks the real values.
PLOT3_RECALL_MIN = 0.65


def _load_recall_qps(results_dir, dataset, mode):
    """Load all available series for a dataset/mode -> ({series_key: data}, sources)."""
    results, sources = {}, []
    for idx in ("ivfflat", "hnsw"):
        fpath = results_dir / f"recall_qps_{dataset}_{idx}_{mode}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            results[idx] = data.get(idx, next(iter(data.values())))
            sources.append(fpath)
    for nw in (4, 8, 12):
        key = f"parallel_ivfflat_w{nw}"
        fpath = results_dir / f"recall_qps_{dataset}_{key}_{mode}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            results[key] = data.get(key, next(iter(data.values())))
            sources.append(fpath)
    return results, sources


def _render_recall_qps_dataset(results_dir, dataset, layout, styles=PLOT3_STYLES,
                               is_main=False):
    """Build (fig, csv_rows, sources) for one dataset. layout='side' -> cold|warm
    two-column; layout='stacked' -> cold/warm stacked one-column. One common
    figure-level legend (no per-panel legends). `styles` selects the palette.
    is_main drops the in-figure title (it moves to the filename)."""
    sources = []
    panels = []
    for m in ("cold", "warm"):
        results, srcs = _load_recall_qps(results_dir, dataset, m)
        if results:
            panels.append((m, results))
            sources += srcs
    if not panels:
        return None, [], []

    if layout == "side":
        fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 2.9),
                                 squeeze=False)
        flat = list(axes[0])
    else:
        fig, axes = plt.subplots(len(panels), 1, figsize=(COL_WIDTH, 4.6),
                                 squeeze=False)
        flat = [axes[r][0] for r in range(len(panels))]

    csv_rows = []
    handles, labels = [], []   # captured once for the common legend
    for ax, (mode, results) in zip(flat, panels):
        for key in PLOT3_ORDER:
            if key not in results:
                continue
            style = styles[key]
            color, marker, label = style[0], style[1], style[2]
            linestyle = style[3] if len(style) > 3 else "-"
            sweep = results[key].get("full_sweep") or results[key].get("recall_qps_points", [])
            if not sweep:
                continue
            pts = [(p.get("recall", p.get("actual_recall", 0)), p["qps"]) for p in sweep]
            pts = [(r, q) for r, q in pts if r >= PLOT3_RECALL_MIN]
            if not pts:
                continue
            recalls = [r for r, _ in pts]
            qps = [q for _, q in pts]
            line, = ax.plot(recalls, qps, color=color, marker=marker,
                            linestyle=linestyle,
                            linewidth=LINE_WIDTH, markersize=MARKER_SIZE - 1,
                            markeredgewidth=0.3, markeredgecolor="black", label=label)
            if label not in labels:
                handles.append(line)
                labels.append(label)
            for r, q in zip(recalls, qps):
                csv_rows.append([dataset, mode, label, r, q])
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Queries per second (QPS)")
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlim(PLOT3_RECALL_MIN, 1.0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Side = single row; stacked = natural order wrapped over 2 rows (no grouping
    # or gap, entries fall where they fall).
    ncol = len(labels) if layout == "side" else (len(labels) + 1) // 2

    band_key = "side" if layout == "side" else "recall_stacked"
    # Drop the title on the main figure (it moves to the filename).
    title = "" if is_main else f"Recall vs QPS — {dataset.upper()}"
    _common_legend(fig, handles, labels, title, ncol=ncol, layout=band_key)
    return fig, csv_rows, sources


def make_plot3(data_dir, out_dir, dpi=DEFAULT_DPI):
    """Render plot 3: GIST stacked main (one-column) + GIST side, Deep side,
    Deep stacked variants (all one common legend) + csv/plot_3.csv."""
    results_dir = Path(data_dir) / "recall_qps_plots" / "results_recall_qps"
    csv_rows, sources = [], []

    # (dataset, layout, is_main, stem)
    targets = [
        ("gist", "stacked", True,  "plot3_recall_qps_gist_stacked"),
        ("gist", "side",    False, "plot3a_recall_qps_gist"),
        ("deep", "side",    False, "plot3b_recall_qps_deep"),
        ("deep", "stacked", False, "plot3c_recall_qps_deep_stacked"),
    ]
    for dataset, layout, is_main, stem in targets:
        fig, rows, srcs = _render_recall_qps_dataset(results_dir, dataset, layout,
                                                     is_main=is_main)
        if fig is None:
            continue
        if is_main:
            save_main(fig, out_dir, stem, dpi=dpi)
        else:
            save_variant(fig, out_dir, stem, dpi=dpi)
        # CSV/sources once per dataset (from the 'side' render; stacked is identical data).
        if layout == "side":
            csv_rows += rows
            sources += srcs

    if csv_rows:
        write_plot_csv(out_dir, 3,
                       ["dataset", "mode", "series", "recall", "qps"], csv_rows)
    return sources


# ===========================================================================
# Plots 4 & 5 — interference throughput frontier (ANN-vs-OLAP).
#
# Two memory scenarios, same plotting machinery, different source dir:
#   plot4 -> throughput_frontier_10_15/  (10GB SB / 15GB CG, memory-constrained)
#   plot5 -> throughput_frontier_45_65/  (45GB SB / 65GB CG; at SF=10 everything
#                                         fits in memory, so effectively unconstrained)
#
# Source layout (per scenario): one index per directory,
#   <scenario>/{ivfflat,hnsw,parallel_ivfflat}/
#   grid_graph.json           (mixed-workload grid measurements)
#   saturation_points_ann.json / saturation_points_olap.json  (axis endpoints)
# All warm. The 45/65 scenario may lack parallel_ivfflat while still collecting;
# missing indexes are simply skipped.
#
# Methodology (How Good is My HTAP System?, SIGMOD'22 §3.3): each measured point
# is (ANN throughput, OLAP throughput); the frontier is the Pareto-optimal
# (non-dominated) boundary, extended to the axes by the pure-ANN and pure-OLAP
# saturation points. OLAP throughput is in per-query QPS (olap_throughput_qps),
# the resolution at which the frontier is non-degenerate: the batch-level (tps)
# frontier in throughput_frontier.json collapses to the two axis endpoints
# because a full 22-query TPC-H batch almost never completes under interference.
#
# MAIN     = absolute frontier, single column.
# VARIANTS = log-x (ANN spans ~1 to 800+ QPS), normalized (1 = each index's max).
# ===========================================================================

QUERIES_PER_BATCH = 22
PLOT4_ORDER = ["ivfflat", "hnsw", "parallel_ivfflat"]

# Default per-scenario source subdirs under <data_dir> for the two frontier
# plots. Each holds the clean per-index layout (see _resolve_frontier_dir).
FRONTIER_SCENARIO_DIRS = {
    4: "throughput_frontier_10_15",   # 10GB SB / 15GB CG (constrained)
    5: "throughput_frontier_45_65",   # 45GB SB / 65GB CG (fits in memory)
}

# A log x-axis cannot place ANN=0, so the pure-OLAP saturation point (0, X_OLAP)
# is pinned to this left-edge value in the log variants. Every index shares the
# same X_OLAP, so all curves then attach to the same y-axis saturation.
LOGX_FLOOR = 1.0




def _pareto_frontier(points):
    """Non-dominated frontier (maximize both ANN and OLAP), sorted by ANN, with
    axis endpoints so the curve connects to both axes. A point is dominated when
    another has >= on both coords and > on at least one."""
    pts = sorted(set(points))
    frontier = []
    for i, (ai, oi) in enumerate(pts):
        dominated = any(aj >= ai and oj >= oi and (aj > ai or oj > oi)
                        for j, (aj, oj) in enumerate(pts) if j != i)
        if not dominated:
            frontier.append((ai, oi))
    # Connect to the y-axis (pure-OLAP point at ann=0) and x-axis (max ann, 0).
    for p in pts:
        if p[0] == 0 and p not in frontier:
            frontier.append(p)
    max_ann = max((p[0] for p in pts), default=0)
    if (max_ann, 0) not in frontier:
        frontier.append((max_ann, 0))
    frontier.sort(key=lambda p: p[0])
    return frontier


def _load_frontier_index(idx_dir):
    """Load one index's frontier points -> (points, sources). points are
    (ann_qps, olap_qps); OLAP uses per-query QPS (see header), plus the pure-ANN
    and pure-OLAP saturation endpoints."""
    idx_dir = Path(idx_dir)
    grid = idx_dir / "grid_graph.json"
    if not grid.exists():
        return [], []
    sources = [grid]
    with open(grid) as f:
        data = json.load(f)
    points = []
    for m in data["measurements"]:
        ann = m["ann_throughput_qps"]
        olap = m.get("olap_throughput_qps",
                     m.get("olap_throughput_tps", 0) * QUERIES_PER_BATCH)
        points.append((ann, olap))
    ann_sat = idx_dir / "saturation_points_ann.json"
    olap_sat = idx_dir / "saturation_points_olap.json"
    if ann_sat.exists():
        with open(ann_sat) as f:
            points.append((json.load(f)["ann_saturation"]["X_ANN"], 0.0))
        sources.append(ann_sat)
    if olap_sat.exists():
        with open(olap_sat) as f:
            points.append((0.0, json.load(f)["olap_saturation"]["X_OLAP"]))
        sources.append(olap_sat)
    return points, sources


def _render_frontier(index_points, out_dir, dpi, stem, title, *,
                     normalize=False, xscale="linear", is_main=False):
    """Draw the throughput frontier (single column). normalize -> each index
    scaled so 1 = its own max ANN / max OLAP. xscale: 'linear' or 'log'. On a
    log axis the ANN=0 pure-OLAP saturation is pinned to LOGX_FLOOR so every
    curve attaches to the same shared OLAP saturation on the y-axis."""
    log_x = xscale == "log"
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))
    for key in PLOT4_ORDER:
        pts = index_points.get(key)
        if not pts:
            continue
        color = FRONTIER_COLORS[key]
        marker = FRONTIER_MARKERS[key]
        max_ann = max((p[0] for p in pts), default=1) or 1
        max_olap = max((p[1] for p in pts), default=1) or 1
        if normalize:
            pts = [(a / max_ann, o / max_olap) for a, o in pts]
        frontier = _pareto_frontier(pts)
        if log_x:  # pin the ann=0 saturation point to the left-edge floor
            pts = [(a if a > 0 else LOGX_FLOOR, o) for a, o in pts]
            frontier = [(a if a > 0 else LOGX_FLOOR, o) for a, o in frontier]
        ax.scatter([a for a, _ in pts], [o for _, o in pts], color=color,
                   marker=marker, s=12, alpha=0.30, linewidths=0, zorder=2)
        fa = [a for a, _ in frontier]
        fo = [o for _, o in frontier]
        ax.plot(fa, fo, color=color, marker=marker, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, markeredgewidth=0.3, markeredgecolor="black",
                label=FRONTIER_LABELS[key], zorder=4)
        ax.fill_between(fa, fo, alpha=0.07, color=color, zorder=1)

    if normalize:
        prop = np.linspace(0, 1, 50)
        ax.plot(prop, 1 - prop, "k--", alpha=0.4, linewidth=1.0,
                label="Proportional")
        ax.set_xlabel("Normalized ANN throughput")
        ax.set_ylabel("Normalized OLAP throughput")
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.05)
    else:
        ax.set_xlabel("ANN throughput (QPS)")
        ax.set_ylabel("OLAP throughput (QPS)")
        if xscale == "log":
            ax.set_xscale("log")
            ax.set_xlim(left=LOGX_FLOOR)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FS,
              handlelength=1.6, handletextpad=0.4)
    fig.tight_layout()
    if is_main:
        save_main(fig, out_dir, stem, dpi=dpi)
    else:
        save_variant(fig, out_dir, stem, dpi=dpi)


# Fixed properties of the throughput-frontier runs that the data files don't
# record, so they live here (update if the experiment setup changes): storage
# medium, the ANN vector dataset, and the TPC-H scale factor. All these runs are
# GIST + SF-10 on NVMe; these tokens go into the figure filenames.
FRONTIER_STORAGE = "nvme"
FRONTIER_DATASET = "gist"
FRONTIER_SCALE = "sf10"

# Per-index directory discovery for plot4. The frozen copy uses clean names
# (throughput_frontier/<key>/), but a raw results dir (e.g. data/10th-june/)
# uses these globs instead, so --frontier_dir can point straight at one.
FRONTIER_RAW_GLOBS = {
    "ivfflat":          "results_frontier_warm_constrained_*",
    "hnsw":             "results_frontier_hnsw_*",
    "parallel_ivfflat": "results_frontier_pivfflat_*",
}


def _resolve_frontier_dir(base, key):
    """Find one index's frontier directory under `base`: the clean
    throughput_frontier/<key>/ name first, else the raw results_frontier_*
    directory matched by FRONTIER_RAW_GLOBS. Returns a Path (may not exist)."""
    clean = base / key
    if (clean / "grid_graph.json").exists():
        return clean
    for m in sorted(base.glob(FRONTIER_RAW_GLOBS[key])):
        if (m / "grid_graph.json").exists():
            return m
    return clean


def _frontier_scenario_suffix(index_dirs, constrained=True):
    """Build the frontier filename scenario suffix from the data's own config so
    the name tracks where the data came from. 'gist'/'sf10' = ANN dataset and
    TPC-H scale factor (see FRONTIER_DATASET / FRONTIER_SCALE); 'olap' = the
    relational (OLAP) throughput on the y-axis; 'nvme' = storage (see
    FRONTIER_STORAGE); '<constrained|unconstrained>_<sb>sb_<cg>cg' = the
    shared_buffers / cgroup memory limits read from a saturation file. The
    constrained/unconstrained word is the scenario's own (caller-supplied): at
    SF-10 the 45/65 run fits in memory, so it is unconstrained despite the
    cgroup limit."""
    mem_word = "constrained" if constrained else "unconstrained"
    parts = [FRONTIER_DATASET, FRONTIER_SCALE, "olap", FRONTIER_STORAGE]
    for d in index_dirs.values():
        for fname in ("saturation_points_ann.json", "saturation_points_olap.json"):
            sat = d / fname
            if not sat.exists():
                continue
            mem = json.load(open(sat)).get("config", {}).get("memory_config")
            if mem:
                parts += [mem_word, f"{mem['shared_buffers_gb']}sb",
                          f"{mem['cgroup_gb']}cg"]
                return "_".join(parts)
    return "_".join(parts)


def _make_frontier_plot(data_dir, out_dir, dpi, n, fdir, constrained=True):
    """Render one throughput-frontier figure (plot `n`): main (absolute,
    single-column) + csv/plot_{n}.csv + variants log-x (plot{n}a) and normalized
    (plot{n}b). `fdir` is the source dir; it may use the clean per-index names or
    raw results_frontier_* dirs (see _resolve_frontier_dir). `constrained` sets
    the memory word in the filename (False = fits in memory). The scenario suffix
    is derived from the data config; titles are dropped (scenario lives in the
    name). plot4 and plot5 share this code, differing only in source dir."""
    fdir = Path(fdir)
    index_dirs = {key: _resolve_frontier_dir(fdir, key) for key in PLOT4_ORDER}
    index_points, sources = {}, []
    for key in PLOT4_ORDER:
        pts, srcs = _load_frontier_index(index_dirs[key])
        if pts:
            index_points[key] = pts
            sources += srcs
    if not index_points:
        print(f"  [plot{n}] no throughput_frontier data found under {fdir}, skipping")
        return []

    scn = _frontier_scenario_suffix(index_dirs, constrained=constrained)
    _render_frontier(index_points, out_dir, dpi,
                     f"plot{n}_throughput_frontier_{scn}", "",
                     is_main=True)
    _render_frontier(index_points, out_dir, dpi,
                     f"plot{n}a_throughput_frontier_logx_{scn}", "", xscale="log")
    _render_frontier(index_points, out_dir, dpi,
                     f"plot{n}b_throughput_frontier_normalized_{scn}", "",
                     normalize=True)

    csv_rows = []
    for key in PLOT4_ORDER:
        if key not in index_points:
            continue
        for a, o in _pareto_frontier(index_points[key]):
            csv_rows.append([FRONTIER_LABELS[key], round(a, 4), round(o, 4)])
    write_plot_csv(out_dir, n, ["index", "ann_qps", "olap_qps"], csv_rows)
    return sources


def make_plot4(data_dir, out_dir, dpi=DEFAULT_DPI):
    """Throughput frontier, 10GB SB / 15GB CG (memory-constrained). Reads
    --frontier_dir if set, else <data_dir>/throughput_frontier_10_15."""
    fdir = _FRONTIER_DIR or Path(data_dir) / FRONTIER_SCENARIO_DIRS[4]
    return _make_frontier_plot(data_dir, out_dir, dpi, 4, fdir, constrained=True)


def make_plot5(data_dir, out_dir, dpi=DEFAULT_DPI):
    """Throughput frontier, 45GB SB / 65GB CG (fits in memory at SF=10). Reads
    <data_dir>/throughput_frontier_45_65. Same plotting as plot4."""
    fdir = Path(data_dir) / FRONTIER_SCENARIO_DIRS[5]
    return _make_frontier_plot(data_dir, out_dir, dpi, 5, fdir, constrained=False)


# ===========================================================================
# Plots 6 & 7 — HTAP-style fixed-workload line families (the decomposition
# behind the frontier in plots 4/5).
#
# Mirrors "How Good is My HTAP System?" (SIGMOD'22): for one index, on the same
# (ANN throughput, OLAP throughput) axes as the frontier, draw two line families:
#   Fixed-R : hold the relational (TPC-H / OLAP) worker count, sweep the vector
#             (GIST / ANN) workers. One line per OLAP level; as ANN load rises the
#             line moves down-right (ANN throughput up, OLAP throughput down).
#   Fixed-V : hold the vector (ANN) worker count, sweep the relational (OLAP)
#             workers. One line per ANN level; moves up-left as OLAP load rises.
# Their upper envelope is the throughput frontier (plots 4/5). HTAP's T/A
# (transactions/analytics) map to our R/V (relational/vector).
#
# Lines are colour-graded by the held worker count (sequential colormap +
# colorbar) so each family reads as a sweep. The grid is finely sampled, so a
# first cut shows many lines; thin later by subsampling the held values.
#   plot6 -> 10/15 (constrained):  ivfflat, hnsw, parallel_ivfflat
#   plot7 -> 45/65 (fits in mem):  ivfflat, hnsw
# One figure PER INDEX, two panels [Fixed-R | Fixed-V] -> FULL_WIDTH.
# ===========================================================================

# Sequential colormaps for the two families (distinct so R vs V never confuse).
HTAP_CMAPS = {"R": "viridis", "V": "plasma"}


def _grade_lines(ax, groups, cmap_name):
    """Draw one colour-graded line per held-worker level. `groups` maps the held
    worker count -> list of (x, y) already sorted along the swept axis. Colour
    runs along the held value via `cmap_name`. Returns a ScalarMappable so the
    caller can attach a colorbar."""
    keys = sorted(groups)
    lo, hi = (keys[0], keys[-1]) if keys else (0, 1)
    norm = plt.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
    cmap = plt.get_cmap(cmap_name)
    for k in keys:
        pts = groups[k]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                color=cmap(norm(k)), linewidth=1.0, markersize=2.5,
                markeredgewidth=0, alpha=0.9, zorder=2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def _olap_qps(m):
    """OLAP per-query QPS for one grid measurement (same convention as the
    frontier: fall back to tps * QUERIES_PER_BATCH if qps is absent)."""
    return m.get("olap_throughput_qps",
                 m.get("olap_throughput_tps", 0) * QUERIES_PER_BATCH)


def _render_htap_index(measurements, index_key, out_dir, dpi, n, scn, mem_label):
    """One per-index figure: [Fixed-R | Fixed-V] colour-graded line families.
    `mem_label` (e.g. '10:15') is appended to the title after the index name.
    Returns CSV rows (family, fixed_workers, ann_workers, olap_workers, qps...)."""
    # Group into the two families. Each line keeps (sweep_key, ann_qps, olap_qps)
    # so we can sort along the swept axis and emit a faithful CSV.
    fixedR, fixedV = {}, {}   # held OLAP -> rows ; held ANN -> rows
    for m in measurements:
        a, o = m["ann_workers"], m["olap_workers"]
        rec = (m["ann_throughput_qps"], _olap_qps(m))
        fixedR.setdefault(o, []).append((a, rec))   # hold OLAP, sweep ANN
        fixedV.setdefault(a, []).append((o, rec))   # hold ANN, sweep OLAP
    R = {o: [rec for _, rec in sorted(v)] for o, v in fixedR.items()}
    V = {a: [rec for _, rec in sorted(v)] for a, v in fixedV.items()}

    fig, (axR, axV) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))
    smR = _grade_lines(axR, R, HTAP_CMAPS["R"])
    smV = _grade_lines(axV, V, HTAP_CMAPS["V"])
    for ax in (axR, axV):
        ax.set_xlabel("ANN throughput (QPS)")
        ax.set_ylabel("OLAP throughput (QPS)")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    axR.set_title("Fixed-R: hold TPC-H workers, sweep ANN")
    axV.set_title("Fixed-V: hold ANN workers, sweep TPC-H")
    fig.colorbar(smR, ax=axR, fraction=0.046, pad=0.02
                 ).set_label("TPC-H (OLAP) workers", fontsize=LABEL_FS)
    fig.colorbar(smV, ax=axV, fraction=0.046, pad=0.02
                 ).set_label("ANN (vector) workers", fontsize=LABEL_FS)
    fig.suptitle(f"{FRONTIER_LABELS[index_key]} - {mem_label}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_main(fig, out_dir, f"plot{n}_htap_lines_{index_key}_{scn}", dpi=dpi)

    rows = []
    for fam, groups in (("fixed_R", fixedR), ("fixed_V", fixedV)):
        for held, recs in sorted(groups.items()):
            for sweep, (aq, oq) in sorted(recs):
                ann_w, olap_w = (sweep, held) if fam == "fixed_R" else (held, sweep)
                rows.append([FRONTIER_LABELS[index_key], fam, held,
                             ann_w, olap_w, round(aq, 4), round(oq, 4)])
    return rows


def _make_htap_lines(data_dir, out_dir, dpi, n, scen_dir, constrained, mem_label):
    """Render plot `n`: one HTAP fixed-line figure per index present under
    <data_dir>/<scen_dir>/<index>/grid_graph.json, plus a combined csv/plot_{n}.csv.
    `mem_label` (e.g. '10:15') goes into each figure title."""
    base = Path(data_dir) / scen_dir
    index_dirs = {key: _resolve_frontier_dir(base, key) for key in PLOT4_ORDER}
    scn = _frontier_scenario_suffix(index_dirs, constrained=constrained)
    csv_rows, sources = [], []
    for key in PLOT4_ORDER:
        grid = index_dirs[key] / "grid_graph.json"
        if not grid.exists():
            continue
        with open(grid) as f:
            measurements = json.load(f).get("measurements", [])
        if not measurements:
            continue
        csv_rows += _render_htap_index(measurements, key, out_dir, dpi, n, scn, mem_label)
        sources.append(grid)
    if not sources:
        print(f"  [plot{n}] no grid data found under {base}, skipping")
        return []
    write_plot_csv(out_dir, n,
                   ["index", "family", "fixed_workers", "ann_workers",
                    "olap_workers", "ann_qps", "olap_qps"], csv_rows)
    return sources


def make_plot6(data_dir, out_dir, dpi=DEFAULT_DPI):
    """HTAP fixed-R / fixed-V line families per index, 10/15 (constrained).
    Maps to plot4's frontier. Reads <data_dir>/throughput_frontier_10_15."""
    return _make_htap_lines(data_dir, out_dir, dpi, 6,
                            FRONTIER_SCENARIO_DIRS[4], constrained=True,
                            mem_label="10:15")


def make_plot7(data_dir, out_dir, dpi=DEFAULT_DPI):
    """HTAP fixed-R / fixed-V line families per index, 45/65 (fits in memory).
    Maps to plot5's frontier. Reads <data_dir>/throughput_frontier_45_65."""
    return _make_htap_lines(data_dir, out_dir, dpi, 7,
                            FRONTIER_SCENARIO_DIRS[5], constrained=False,
                            mem_label="45:65")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

MAKERS = {
    "1": ("plot1_redundant_computation_optimization", make_plot1),
    "2": ("plot2_parallel_ivf", make_plot2),
    "3": ("plot3_recall_qps", make_plot3),
    "4": ("plot4_throughput_frontier_10_15", make_plot4),
    "5": ("plot5_throughput_frontier_45_65", make_plot5),
    "6": ("plot6_htap_lines_10_15", make_plot6),
    "7": ("plot7_htap_lines_45_65", make_plot7),
}


def _parse_selection(tokens_str):
    """Parse a '--plot'/'--plots' value into (run_numbers, selection).

    Tokens are a plot number ('1' = main + all its variants) or number+letter
    ('2a' = just that variant). Returns the ordered unique plot numbers whose
    makers must run, and the raw token set used to gate save_* (None if every
    token is a bare number, i.e. a normal full-plot run with no gating needed)."""
    run_numbers, selection, only_full = [], set(), True
    for raw in tokens_str.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        m = re.match(r"^(\d+)([a-z]?)$", tok)
        if not m or m.group(1) not in MAKERS:
            print(f"[skip] unknown plot selector '{raw.strip()}'")
            continue
        num = m.group(1)
        if num not in run_numbers:
            run_numbers.append(num)
        selection.add(tok)
        if m.group(2):
            only_full = False
    return run_numbers, (None if only_full else selection)


def main():
    global _SELECTION, _FRONTIER_DIR
    ap = argparse.ArgumentParser(description="Build the paper figures.")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--plots", default=None,
                    help="Comma-separated selectors. A number '1' renders plot 1 "
                         "and all its variants; a number+letter '2a' renders only "
                         "that variant. E.g. '1,3', '2a,3b'. Default: all.")
    ap.add_argument("--plot", default=None,
                    help="Alias for --plots (takes precedence if both given).")
    ap.add_argument("--frontier_dir", default=None,
                    help="Override plot4's frontier source dir (plot5 is "
                         "unaffected). Accepts either the clean <scenario>/<index>/ "
                         "layout or a raw results_frontier_* dir (e.g. "
                         "data/10th-june). Filename scenario suffix is derived "
                         "from that data's config.")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = ap.parse_args()

    _FRONTIER_DIR = str(Path(args.frontier_dir).resolve()) if args.frontier_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    tokens_str = args.plot or args.plots or "1,2,3,4,5,6,7"
    run_numbers, _SELECTION = _parse_selection(tokens_str)
    print(f"[data_dir] {data_dir}")
    if _SELECTION is not None:
        print(f"[select] {','.join(sorted(_SELECTION))}")

    manifest = {}  # plot key -> list of source files used
    for key in run_numbers:
        name, maker = MAKERS[key]
        print(f"[plot{key}] {name}")
        sources = maker(data_dir, out_dir, dpi=args.dpi) or []
        manifest[key] = (name, [Path(s).resolve() for s in sources])

    _write_source_manifest(out_dir, data_dir, manifest)


def _write_source_manifest(out_dir, data_dir, manifest):
    """Write final_plots/SOURCE.txt so the data provenance is crystal clear:
    which raw files under data/final_data/ produced each figure."""
    lines = [
        "# Provenance for the figures in this directory.",
        "# Generated by scripts/paper_plots.py — do not edit by hand.",
        "#",
        "# Source data lives in (frozen copy, see data/final_data/README.md):",
        f"#   {data_dir}",
        "",
    ]
    for key in sorted(manifest):
        name, sources = manifest[key]
        lines.append(f"plot{key}  {name}")
        if not sources:
            lines.append("    (no source files found)")
        for s in sources:
            try:
                rel = s.relative_to(data_dir)
            except ValueError:
                rel = s
            lines.append(f"    {rel}")
        lines.append("")
    path = Path(out_dir) / "SOURCE.txt"
    path.write_text("\n".join(lines))
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
