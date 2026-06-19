######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
######## # DEFAULT UN-CONSTRAINED - 45:60
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
# HNSW + IVF
# python3 plot_frontier_comparison.py --grid  \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       --output frontier-HNSW-IVFFlat-16-June-26-unconstrained.png

# # # IVF
# python3 plot_frontier_comparison.py --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       --output frontier-IVFFlat-16-June-26-unconstrained.png
      
# # HNSW
# python3 plot_frontier_comparison.py --grid \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       --output frontier-HNSW-16-June-26-unconstrained.png

# # # # All indexes 
# python3 plot_frontier_comparison.py --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       "Parallel IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       --output frontier-all-16-June-26-unconstrained.png

# python3 plot_frontier_comparison.py --logx --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       "Parallel IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_45GB_sb_65GB_cg/grid_graph.json \
#       --output frontier-all-16-June-26-LOG-unconstrained.png


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
######## # DEFAULT CONSTRAINED - 10:15
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
# # # PIVF w=4
python3 plot_frontier_comparison.py --grid \
      "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
      "Parallel IVFFlat (w=8)":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_10GB_sb_15GB_cg_w8/grid_graph.json \
      "Parallel IVFFlat (w=4)":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
      --output frontier-IVFFlat-19-June-26.png

# HNSW + IVF
# python3 plot_frontier_comparison.py --grid  \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       --output frontier-HNSW-IVFFlat-13-June-26.png


# # # IVF
# python3 plot_frontier_comparison.py --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       --output frontier-IVFFlat-13-June-26.png
#       # "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       # "Parallel IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \

# # HNSW
# python3 plot_frontier_comparison.py --grid \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       --output frontier-HNSW-13-June-26.png


# # All indexes 
# python3 plot_frontier_comparison.py --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       "Parallel IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       --output frontier-all-13-June-26.png

# python3 plot_frontier_comparison.py --logx --grid \
#       "IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       "HNSW":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_hnsw_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       "Parallel IVFFlat":/users/vmageira/vdb_tuning/vdb-tuning-main/TPCH/results_frontier_pivfflat_warm_constrained_10GB_sb_15GB_cg/grid_graph.json \
#       --output frontier-all-13-June-26-LOG.png
