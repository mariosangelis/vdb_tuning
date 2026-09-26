import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, LeaveOneOut
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import numpy as np


# Load JSONL
with open("vm_results_pinned_cores/results_mlos_tuner_IVFFLAT_GIST_DATASET_all_knobs_16KB_page_size_PLAIN_storage_PINNED.jsonl") as f:
    data = [json.loads(line) for line in f]
df = pd.json_normalize(data)

# Select relevant config parameters
#X = df[['config.work_mem', 'config.shared_buffers', 'config.effective_cache_size']]
#         'config.cpu_index_tuple_cost', 'config.cpu_operator_cost', 'config.parallel_setup_cost', 'config.parallel_tuple_cost']]


#X = df[['config.shared_buffers', 'config.effective_io_concurrency', 'config.io_combine_limit', 'config.temp_buffers', 'config.work_mem','config.io_max_combine_limit','config.effective_cache_size', 'config.random_page_cost', 'config.cpu_tuple_cost','config.cpu_index_tuple_cost', 'config.cpu_operator_cost', 'config.parallel_setup_cost', 'config.parallel_tuple_cost']]



X = df[['config.shared_buffers','config.effective_io_concurrency', 'config.io_combine_limit', 'config.temp_buffers', 'config.work_mem', 'config.random_page_cost', 'config.cpu_tuple_cost', 'config.cpu_index_tuple_cost', 'config.cpu_operator_cost','config.parallel_setup_cost','config.parallel_tuple_cost'        ,'config.effective_cache_size', 'config.max_parallel_workers_per_gather']]


#X = df[['config.shared_buffers', 'config.effective_io_concurrency', 'config.io_combine_limit', 'config.work_mem', 'config.temp_buffers']]
#X = df[['config.random_page_cost', 'config.cpu_tuple_cost', 'config.cpu_index_tuple_cost', 'config.cpu_operator_cost','config.parallel_setup_cost','config.parallel_tuple_cost'        ,'config.effective_cache_size', 'config.max_parallel_workers_per_gather']]

y = df['total_exec_time_topk']


# Optional: Convert to numeric (if any are strings like '512MB')
X = X.apply(pd.to_numeric, errors='coerce')

X['config.shared_buffers'] = np.log1p(X['config.shared_buffers'])
X['config.work_mem'] = np.log1p(X['config.work_mem'])
X['config.temp_buffers'] = np.log1p(X['config.temp_buffers'])
X['config.effective_io_concurrency'] = np.log1p(X['config.effective_io_concurrency'])
X['config.io_combine_limit'] = np.log1p(X['config.io_combine_limit'])


X['config.random_page_cost'] = np.log1p(X['config.random_page_cost'])
X['config.cpu_tuple_cost'] = np.log1p(X['config.cpu_tuple_cost'])
X['config.cpu_index_tuple_cost'] = np.log1p(X['config.cpu_index_tuple_cost'])
X['config.cpu_operator_cost'] = np.log1p(X['config.cpu_operator_cost'])
X['config.parallel_setup_cost'] = np.log1p(X['config.parallel_setup_cost'])
X['config.parallel_tuple_cost'] = np.log1p(X['config.parallel_tuple_cost'])
X['config.effective_cache_size'] = np.log1p(X['config.effective_cache_size'])
X['config.max_parallel_workers_per_gather'] = np.log1p(X['config.max_parallel_workers_per_gather'])


# Drop rows with NaNs
X = X.dropna()
y = y.loc[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and compute R²
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² score (single split): {r2:.4f}")

# Cross-validation for robust R² estimate
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"R² cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"R² per fold: {[f'{s:.4f}' for s in cv_scores]}")

# Feature importances
importances = rf.feature_importances_
feature_names = X.columns


# Feature importances
importances = rf.feature_importances_
feature_names = np.array(X.columns)

# Sort by importance (optional but useful)
sorted_indices = importances.argsort()[::-1]
feature_names = feature_names[sorted_indices]
importances = importances[sorted_indices]

import plotly.graph_objects as go


# Colors
ivfflat_color = 'rgb(31, 119, 180)'  # soft blue

# Create the bar chart
fig = go.Figure(data=[
    go.Bar(
        name='IVFFLAT',
        x=feature_names,
        y=importances,
        marker=dict(
            color=ivfflat_color,
            pattern_shape="/"
        ),
        #text=[f"{imp:.4f}" for imp in importances],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>"
    )
])

# Style the axes and layout
fig.update_yaxes(
    title='Importance',
    range=[0, 1],
    dtick=0.1,
    showgrid=True
)

fig.update_xaxes(
    title='Configuration Parameter',
    tickangle=-45
)

fig.update_layout(
    title="Feature Importance for Execution Time (Random Forest) - IVFFLAT ALL Knobs 16KB Page Size PLAIN ",
    template='plotly_white',
    barmode='group',
    width=800,
    height=500,
    margin=dict(t=80),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

fig.show()



#import plotly.graph_objects as go

## Interactive bar chart with Plotly
#fig = go.Figure(go.Bar(
    #x=feature_names,
    #y=importances,
    #marker_color='teal',
    #text=[f"{imp:.4f}" for imp in importances],
    #textposition="auto",
    #hovertemplate="<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>"
#))

#fig.update_layout(
    #title="Feature Importance for Execution Time (Random Forest) TopK Batch, IVFFLAT memory knobs",
    #xaxis_title="Configuration Parameter",
    #yaxis_title="Importance",
    #xaxis_tickangle=-45,
    #height=600,
    #width=1000,
    #margin=dict(t=80),
    #template="plotly_white"
#)

#fig.show()
