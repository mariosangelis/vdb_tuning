import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Load data
with open("mixed_workload_tpch_and_topk_HNSW_WITH_M_16_four_GB_size_NO_HYPERTHREADING_16_cores_for_postgres_64GB_RAM_GIST.txt", "r") as f:
    data = [json.loads(line) for line in f]

# Extract data
topk_qps = [d["topk_throughput"] for d in data]
tpch_exec_time = [d["tpch_execution_time"] for d in data]
tpch_avg_latency = [d["summary"]["Latency Distribution"]["Average Latency (microseconds)"] / 1e6 for d in data]
tpch_95p_latency = [d["summary"]["Latency Distribution"]["95th Percentile Latency (microseconds)"] / 1e6 for d in data]
tpch_99p_latency = [d["summary"]["Latency Distribution"]["99th Percentile Latency (microseconds)"] / 1e6 for d in data]
tpch_median = [d["summary"]["Latency Distribution"]["Median Latency (microseconds)"] / 1e6 for d in data]
tpch_75p = [d["summary"]["Latency Distribution"]["75th Percentile Latency (microseconds)"] / 1e6 for d in data]
topk_avg_exec_time = [d["topk_avg_exec_time_sec"] for d in data]

topk_threads_used = list(range(len(data)))
topk_threads_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]



## ✅ SAVE to CSV here
#df = pd.DataFrame({
    #"TopK_QPS": topk_qps,
    #"TPCH_Exec_Time": tpch_exec_time,
    #"TPCH_Avg_Latency": tpch_avg_latency,
    #"TPCH_Median_Latency": tpch_median,
    #"TPCH_75p_Latency": tpch_75p,
    #"TPCH_95p_Latency": tpch_95p_latency,
    #"TPCH_99p_Latency": tpch_99p_latency,
    #"TopK_Avg_Exec_Time": topk_avg_exec_time,
    #"TopK_Threads": topk_threads_used
#})
#df.to_csv("HNSW_metrics_percentiles_30GB_RAM.csv", index=False)

# Create a 2x3 grid of subplots
fig = make_subplots(rows=2, cols=3, subplot_titles=[
    "TopK QPS vs TPC-H Execution Time",
    "TopK Query Producers vs Max TopK QPS Achieved",
    "",
    "TopK QPS vs TPC-H Avg Query Latency",
    "TopK QPS vs TPC-H Latency Percentiles",
    "TopK QPS vs TopK Avg Execution Time",
])

# Row 1, Col 1: TPC-H Execution Time
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_exec_time, mode='lines+markers',
                         name="TPC-H Execution Time", marker=dict(size=8)),
              row=1, col=1)

# Row 1, Col 2: TopK Threads vs TopK QPS
fig.add_trace(go.Bar(
    x=topk_threads_used,
    y=topk_qps,
    name="TopK QPS (Observed)",
    marker_color="steelblue"
), row=1, col=2)

# Force all thread numbers to show as x-axis ticks
fig.update_xaxes(
    title_text="TopK Threads",
    tickmode='array',
    tickvals=topk_threads_used,
    ticktext=[str(t) for t in topk_threads_used],
    row=1, col=2
)


#static_thread_qps = {
    #1: 20,
    #2: 30,
    #3: 44,
    #4: 60
#}

## Observed TopK QPS
#fig.add_trace(go.Bar(
    #x=topk_threads_used,
    #y=topk_qps,
    #name="Observed TopK QPS",
    #marker_color="steelblue"
#), row=1, col=2)

## Static theoretical TopK QPS
#fig.add_trace(go.Bar(
    #x=list(static_thread_qps.keys()),
    #y=list(static_thread_qps.values()),
    #name="Theoretical QPS Limit",
    #marker_color="indianred",
    #opacity=0.6
#), row=1, col=2)


# Row 2, Col 1: TPC-H Average Latency
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_avg_latency, mode='lines+markers',
                         name="TPC-H Avg Latency", marker=dict(size=8)),
              row=2, col=1)

# Row 2, Col 2: TPC-H Latency Percentiles
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_median, name="Median", mode='lines+markers'), row=2, col=2)
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_75p, name="75th", mode='lines+markers'), row=2, col=2)
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_95p_latency, name="95th", mode='lines+markers'), row=2, col=2)
fig.add_trace(go.Scatter(x=topk_qps, y=tpch_99p_latency, name="99th", mode='lines+markers'), row=2, col=2)

# Row 2, Col 3: TopK Avg Execution Time
fig.add_trace(go.Scatter(x=topk_qps, y=topk_avg_exec_time, mode='lines+markers',
                         name="TopK Avg Exec Time", marker=dict(size=8)),
              row=2, col=3)

# Axis labels
fig.update_xaxes(title_text="TopK QPS", row=1, col=1)
fig.update_yaxes(title_text="Time (s)", row=1, col=1)

fig.update_xaxes(title_text="TopK Query Producers", row=1, col=2)
fig.update_yaxes(title_text="Max TopK QPS Achieved", row=1, col=2)

fig.update_xaxes(title_text="TopK QPS", row=2, col=1)
fig.update_yaxes(title_text="Latency (s)", row=2, col=1)

fig.update_xaxes(title_text="TopK QPS", row=2, col=2)
fig.update_yaxes(title_text="Latency (s)", row=2, col=2)

fig.update_xaxes(title_text="TopK QPS", row=2, col=3)
fig.update_yaxes(title_text="Time (s)", row=2, col=3)

# Final layout
fig.update_layout(
    title="Mixed Workload Benchmark: TPC-H Under TopK Load (HNSW 4GB & GIST) RAM=64 GB, 16 cores, NO HYPERTHREADING",
    height=750,
    width=1400,
    showlegend=True,
    template="plotly_white"
)

fig.show()
