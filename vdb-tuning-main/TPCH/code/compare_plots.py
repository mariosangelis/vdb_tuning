import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Files and labels to compare
configs = [
    ("mixed_workload_tpch_and_topk_IVFFLAT_io_1000_NO_HYPERTHREADING_16_cores_for_postgres_20GB_RAM_GIST.txt", "IVFFLAT"),
    ("mixed_workload_tpch_and_topk_HNSW_WITH_M_16_four_GB_size_NO_HYPERTHREADING_16_cores_for_postgres_20GB_RAM_GIST.txt", "HNSW"),
]

# Initialize data containers
results = {}

for filename, label in configs:
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
        data =data[0:11]

    results[label] = {
        "topk_qps": [d["topk_throughput"] for d in data],
        "tpch_exec_time": [d["tpch_execution_time"] for d in data],
        "tpch_avg_latency": [d["summary"]["Latency Distribution"]["Average Latency (microseconds)"] / 1e6 for d in data],
        "tpch_95p_latency": [d["summary"]["Latency Distribution"]["95th Percentile Latency (microseconds)"] / 1e6 for d in data],
        "tpch_99p_latency": [d["summary"]["Latency Distribution"]["99th Percentile Latency (microseconds)"] / 1e6 for d in data],
        "tpch_median": [d["summary"]["Latency Distribution"]["Median Latency (microseconds)"] / 1e6 for d in data],
        "tpch_75p": [d["summary"]["Latency Distribution"]["75th Percentile Latency (microseconds)"] / 1e6 for d in data],
        "topk_avg_exec_time": [d["topk_avg_exec_time_sec"] for d in data],
        "topk_threads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(data)],
    }

# Create subplots with more spacing
fig = make_subplots(
    rows=2, cols=3,
    horizontal_spacing=0.15,  # wider horizontal space between columns
    vertical_spacing=0.2,     # more vertical space between rows
    subplot_titles=[
        "TopK QPS vs TPC-H Execution Time (all TPCH queries)",
        "TopK Query Producers vs Max TopK QPS Achieved",
        "",
        "TopK QPS vs TPC-H Avg Query Latency",
        "TopK QPS vs TopK Avg Execution Time",
        ""
    ]
)

#for label in results:
    #df = pd.DataFrame({
        #"TopK_QPS": results[label]["topk_qps"],
        #"TPCH_Exec_Time": results[label]["tpch_exec_time"],
        #"TPCH_Avg_Latency": results[label]["tpch_avg_latency"],
        #"TopK_Avg_Exec_Time": results[label]["topk_avg_exec_time"],
        #"TopK_QPS": results[label]["topk_qps"],
        #"TopK_Threads": results[label]["topk_threads"]
    #})
    #df.to_csv(f"{label}_metrics_30GB_RAM.csv", index=False)

# 1. Execution Time (Row 1, Col 1)
for label in results:
    fig.add_trace(go.Scatter(
        x=results[label]["topk_qps"],
        y=results[label]["tpch_exec_time"],
        mode="lines+markers",
        name=f"{label} - TPCH Exec Time"
    ), row=1, col=1)

# 2. QPS by Threads (Row 1, Col 2)
for label in results:
    fig.add_trace(go.Bar(
        x=results[label]["topk_threads"],
        y=results[label]["topk_qps"],
        name=f"{label} - Topk QPS"
    ), row=1, col=2)

# 3. Average Latency (Row 2, Col 1)
for label in results:
    fig.add_trace(go.Scatter(
        x=results[label]["topk_qps"],
        y=results[label]["tpch_avg_latency"],
        mode="lines+markers",
        name=f"{label} - Topk Query Avg Latency"
    ), row=2, col=1)

# 5. TopK Avg Execution Time (Row 2, Col 3)
for label in results:
    fig.add_trace(go.Scatter(
        x=results[label]["topk_qps"],
        y=results[label]["topk_avg_exec_time"],
        mode="lines+markers",
        name=f"{label} - TopK Avg Exec Time"
    ), row=2, col=2)

# Axis labels
fig.update_xaxes(title_text="TopK QPS", row=1, col=1)
fig.update_yaxes(title_text="Time (s)", row=1, col=1)

fig.update_xaxes(title_text="TopK Query Producers", row=1, col=2)
fig.update_yaxes(title_text="TopK QPS", row=1, col=2)

fig.update_xaxes(title_text="TopK QPS", row=2, col=1)
fig.update_yaxes(title_text="Latency (s)", row=2, col=1)

fig.update_xaxes(title_text="TopK QPS", row=2, col=2)
fig.update_yaxes(title_text="Time (s)", row=2, col=2)

# Final layout
fig.update_layout(
    title="Comparison: Mixed Workload TPC-H Under TopK Load, RAM: 20GB, Cores:16, NO HYPERTHREADING, Index: Both",
    height=800,
    width=1400,
    showlegend=True,
    legend=dict(
        orientation="v",       # vertical legend
        x=0.71,                # left-aligned
        y=1.0,                 # top-aligned
        font=dict(size=11)
    ),
    template="plotly_white"
)

fig.show()
