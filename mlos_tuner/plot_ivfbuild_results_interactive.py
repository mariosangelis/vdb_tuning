import json
import pandas as pd
import plotly.graph_objects as go

# Load JSONL data
entries = []
with open("ivfflat_build_memory_knobs_results.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            timings = entry.get("timings", {})
            config = entry["config"]
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
            total_time = entry.get("total_exec_time") or sum(timings.values())

            data = {
                "config": config,
                "config_str": config_str,
                "kmeans_time": timings.get("kmeans_time", 0.0),
                "detoast_time": timings.get("detoast_time", 0.0),
                "closest_center_time": timings.get("closest_center_time", 0.0),
                "exec_tuple_time": timings.get("exec_tuple_time", 0.0),
                "sort_time": timings.get("sort_time", 0.0),
                "load_time": timings.get("load_time", 0.0),
                "total_exec_time": total_time
            }

            data["other_time"] = max(
                0.0,
                total_time - sum(data[k] for k in [
                    "kmeans_time", "detoast_time", "closest_center_time",
                    "exec_tuple_time", "sort_time", "load_time"
                ])
            )

            entries.append(data)
        except Exception as e:
            print("Skipping line due to error:", e)

# Create DataFrame
df = pd.DataFrame(entries)
df["label"] = [f"C{i+1}" for i in range(len(df))]

# Prettify config for tooltip
df["config_pretty"] = df["config"].apply(lambda cfg: "<br>".join(f"{k}: {v}" for k, v in cfg.items()))

# Sort by total execution time
df = df.sort_values("total_exec_time", ascending=False).reset_index(drop=True)

# Color & labels
colors = {
    "kmeans_time": "#1f77b4",
    "detoast_time": "#ff7f0e",
    "closest_center_time": "#2ca02c",
    "exec_tuple_time": "#d62728",
    "sort_time": "#9467bd",
    "load_time": "#8c564b",
    "other_time": "#e377c2",
}
timing_labels = {
    "kmeans_time": "KMeans Computation",
    "detoast_time": "Detoasting vectors",
    "closest_center_time": "Compute Closest Centroid for each vector",
    "exec_tuple_time": "Store structs to heap",
    "sort_time": "Sort Heap by centroid id",
    "load_time": "Create Index Pages",
    "other_time": "Remaining",
}

# Add bar traces
for i, key in enumerate(colors.keys()):
    fig_data = df[list(colors.keys()) + ["config_pretty"]]
    fig = fig if 'fig' in locals() else go.Figure()

    fig.add_trace(go.Bar(
        x=df["label"],
        y=df[key],
        name=timing_labels[key],
        marker_color=colors[key],
        customdata=fig_data.values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"KMeans: %{{customdata[0]:.3f}}s<br>"
            f"Detoasting: %{{customdata[1]:.3f}}s<br>"
            f"Closest Center: %{{customdata[2]:.3f}}s<br>"
            f"Store Tuples: %{{customdata[3]:.3f}}s<br>"
            f"Sort: %{{customdata[4]:.3f}}s<br>"
            f"Load: %{{customdata[5]:.3f}}s<br>"
            f"Other: %{{customdata[6]:.3f}}s<br><br>"
            "<b>Config:</b><br>%{customdata[7]}<extra></extra>"
        )
    ))

fig.update_layout(
    title="IVFFlat Index Build Time Breakdown by Config (Memory Knobs)",
    xaxis_title="Configuration",
    yaxis_title="Execution Time (s)",
    barmode="stack",
    xaxis_tickangle=-45,
    height=950,
    width=1900,
    showlegend=True  # Set to False to hide legend
)

fig.show()
