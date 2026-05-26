=============================================================================
COMPILER OPTIMIZATION EXPERIMENT
=============================================================================

The postgres version we provide contains the modifications we performed 
regarding the compiler optimization (top-k heap optimization).

To measure the performance gain, follow the steps below IN ORDER:

=============================================================================
PREREQUISITES (run these first, from compute_and_compare_recall/)
=============================================================================

1. Tune index build parameters (find optimal lists/m for recall=0.9):

   sudo python3 tune_ivfflat.py --recall 0.9
   sudo python3 tune_hnsw.py --recall 0.9

   Output: tune_ivfflat_results.json, tune_hnsw_results.json

2. Calculate optimal probes/ef_search per K (uses tuned build params):

   sudo python3 calculate_index_parameters_for_compiler_optimization.py ivfflat 0.9
   sudo python3 calculate_index_parameters_for_compiler_optimization.py hnsw 0.9

   Output: optimal_ivfflat_params_recall_0.9.json, optimal_hnsw_params_recall_0.9.json

=============================================================================
RUNNING THE COMPILER OPTIMIZATION EXPERIMENT
=============================================================================

3. Enable the optimization flag:
   Set enable_topk_search_optimizations to true in:
   postgresql/src/backend/executor/execExpr.c

4. Recompile postgres and restart.

5. Run the experiment for both indexes:
   sudo python3 compute_optimization_results_for_different_K.py

   Output: results_with_optimization_IVFFLAT.txt
           results_with_optimization_HNSW.txt

6. Disable the optimization flag, recompile postgres and restart.

7. Run the experiment again (default version):
   sudo python3 compute_optimization_results_for_different_K.py

   Output: results_without_optimization_IVFFLAT.txt
           results_without_optimization_HNSW.txt

8. Plot the results:
   sudo python3 plot_optimization_results.py

=============================================================================
FULL EXPERIMENT PIPELINE (all experiments)
=============================================================================

Step 1: Index Tuning          -> compute_and_compare_recall/tune_*.py
Step 2: Index Parameters      -> compute_and_compare_recall/calculate_*.py
Step 3: Compiler Optimization -> compiler_optimization/ (this directory)
Step 4: Parallel IVFFlat      -> parallel_implementation_vs_default/run_comparison_workers.py
Step 5: Throughput Frontier   -> TPCH/code/find_saturation_points.py + build_throughput_frontier.py
Step 6: Memory Constrained    -> Same as Step 5 with --memory 12:16 flag
