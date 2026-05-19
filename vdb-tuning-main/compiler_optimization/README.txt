

The postgres version we provide contains the modifications we performed regarding the ompiler optimization.
To measure the perfromance gain, first make sure that the "enable_topk_search_optimizations" flag is set to true inside postgresql/src/backend/executor/execExpr.c

Then, run the compute_optimization_results_for_different_K.py skript for both indexes.

Check the compute_optimization_results_for_different_K.py. Go to the end, change the name of the file where you write the results based on the index you are testing:
For example, when testing the optimized version, set the file name to "results_with_optimization_DESKTOP_HNSW.txt" and "results_with_optimization_DESKTOP_IVFFLAT.txt" respectively.

Then, disable the "enable_topk_search_optimizations" flag, recompile postgres and run the experiment again to test the default version.
In total, you will have 4 files, namely "results_with_optimization_DESKTOP_HNSW.txt", "results_without_optimization_DESKTOP_HNSW.txt", "results_with_optimization_DESKTOP_IVFFLAT.txt", "results_without_optimization_DESKTOP_IVFFLAT.txt"

Finally, to plot the results, run the plot_optimization_results.py skript.
