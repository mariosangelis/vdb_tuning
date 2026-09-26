import json
import pandas as pd
import plotly.graph_objects as go

# Load JSONL data
entries = []
exclude=0
with open("ivfflat_scan_BAS_NORMAL_memory_metrics_optimizations_OFF.jsonl", "r") as f:
    for line in f:
        try:
            exclude+=1
            if exclude==44:
                continue
            entry = json.loads(line)
            timings = entry.get("timings", {})
            config = entry["config"]
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
            total_time = entry.get("total_exec_time") or sum(timings.values())

            # Build full config and individual timing components
            data = {
                "config": config,
                "config_str": config_str,
                "get_scan_lists_time": timings.get("get_scan_lists_time", 0.0),
                "get_scan_items_time": timings.get("get_scan_items_time", 0.0),
                "total_exec_time": total_time
            }

            # Compute derived timing
            data["other_time"] = max(
                0.0,
                total_time - data["get_scan_lists_time"] - data["get_scan_items_time"]
            )

            entries.append(data)
        except Exception as e:
            print("Skipping line due to error:", e)

# Create DataFrame
df = pd.DataFrame(entries)
df["label"] = [f"C{i+1}" for i in range(len(df))]

# Human-readable config for hover
def format_config(cfg_dict):
    return "<br>".join(f"{k}: {v}" for k, v in cfg_dict.items())

df["config_pretty"] = df["config"].apply(format_config)

# Sort by total execution time
df = df.sort_values("total_exec_time", ascending=False).reset_index(drop=True)

# Plot
fig = go.Figure()
colors = {
    "get_scan_lists_time": "green",
    "get_scan_items_time": "steelblue",
    "other_time": "darkorange",
}
timing_labels = {
    "get_scan_lists_time": "Scan Lists",
    "get_scan_items_time": "Scan Items",
    "other_time": "Order By",
}

for key in ["get_scan_lists_time", "get_scan_items_time", "other_time"]:
    fig.add_trace(go.Bar(
        x=df["label"],
        y=df[key],
        #name=timing_labels[key],
        marker_color=colors[key],
        customdata=df[["get_scan_lists_time", "get_scan_items_time", "other_time", "config_pretty"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Scan Lists: %{customdata[0]:.3f}s<br>"
            "Scan Items: %{customdata[1]:.3f}s<br>"
            "Order By: %{customdata[2]:.3f}s<br><br>"
            "<b>Config:</b><br>%{customdata[3]}<extra></extra>"
        )
    ))

fig.update_layout(
    title="IVFFlat Index Scan Time Breakdown by Config (Memory Knobs) using BAS_NORMAL without topK optimizations",
    xaxis_title="Configuration",
    yaxis_title="Execution Time (s)",
    barmode="stack",
    xaxis_tickangle=-45,
    #legend_title="Timing Components",
    height=950,
    width=1900,
    showlegend=False
)

fig.show()
