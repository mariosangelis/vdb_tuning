import json
import pandas as pd
import plotly.graph_objects as go

import plotly.express as px
colors = px.colors.qualitative.Plotly  # Only 10, so we'll repeat with variation
colors = px.colors.sample_colorscale("Viridis", [i/21 for i in range(22)])

# Load JSONL data
entries = []
with open("results_mlos_tuner_hnsw_index_memory_knobs.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            config = entry["config"]
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
            #tpch_execution_time = entry.get("tpch_execution_time")
            total_exec_time_topk = entry.get("total_exec_time_topk")

            data = {
                "config": config,
                "config_str": config_str,
                #"tpch_execution_time": tpch_execution_time,
                "total_exec_time_topk": total_exec_time_topk
            }

            entries.append(data)
        except Exception as e:
            print("Skipping line due to error:", e)



# Create DataFrame
df = pd.DataFrame(entries)
df["label"] = [f"C{i+1}" for i in range(len(df))]
df["config_pretty"] = df["config"].apply(lambda cfg: "<br>".join(f"{k}: {v}" for k, v in cfg.items()))

# Sort by total execution time
df = df.sort_values("total_exec_time_topk", ascending=False).reset_index(drop=True)

# Plot
fig = go.Figure()

#
# TopK scan items
fig.add_trace(go.Bar(
    x=df["label"],
    y=df["total_exec_time_topk"],
    showlegend=False,
    name="TopK: Scan Items",
    marker_color="steelblue",
    customdata=df[["total_exec_time_topk", "config_pretty"]],
    hovertemplate="<b>%{x}</b><br>Scan Items: %{customdata[0]:.3f}s<br><br><b>Config:</b><br>%{customdata[1]}<extra></extra>",
))

# TPCC other (remaining part)
#fig.add_trace(go.Bar(
    #x=df["label"],
    #y=df["tpch_execution_time"],
    #showlegend=False,
    #name="tpch_execution_time",
    #marker_color="darkorange",
    #customdata=df[["tpch_execution_time", "config_pretty"]],
    #hovertemplate="<b>%{x}</b><br>TopK Batch: %{customdata[0]:.3f}s<br><br><b>Config:</b><br>%{customdata[1]}<extra></extra>",
#))



fig.show()
