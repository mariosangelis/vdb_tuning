#!/usr/bin/env bash
# Run compiler optimization experiments for vector indexes
# Usage: ./run_optimization_experiments.sh [--dataset DATASET] [--index INDEX] [--iterations N]

set -euo pipefail

# Default values
DATASET="gist"
INDEX="all"
ITERATIONS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --index)
            INDEX="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset DATASET] [--index INDEX] [--iterations N]"
            echo "  --dataset: gist, deep, or sift (default: gist)"
            echo "  --index: ivfflat, hnsw, or all (default: all)"
            echo "  --iterations: number of iterations per experiment (default: 1)"
            exit 1
            ;;
    esac
done

# Determine which indexes to run
if [ "$INDEX" = "all" ]; then
    INDEXES=("ivfflat" "hnsw")
elif [ "$INDEX" = "ivfflat" ] || [ "$INDEX" = "hnsw" ]; then
    INDEXES=("$INDEX")
else
    echo "Error: Invalid index type '$INDEX'. Must be ivfflat, hnsw, or all"
    exit 1
fi

# Validate dataset
if [ "$DATASET" != "gist" ] && [ "$DATASET" != "deep" ] && [ "$DATASET" != "sift" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be gist, deep, or sift"
    exit 1
fi

# Print configuration
echo "========================================"
echo "Compiler Optimization Experiments"
echo "========================================"
echo "Dataset:    $DATASET"
echo "Index(es):  ${INDEXES[*]}"
echo "Iterations: $ITERATIONS"
echo "========================================"
echo ""

# Track failures
FAILED_COMMANDS=()

# Function to run a command and track failures
run_experiment() {
    local cmd="$1"
    echo "Running: $cmd"
    if eval "$cmd"; then
        echo "✓ Success"
    else
        echo "✗ Failed"
        FAILED_COMMANDS+=("$cmd")
    fi
    echo ""
}

# Run experiments for each index type
for idx in "${INDEXES[@]}"; do
    echo "========================================"
    echo "Running experiments for $idx"
    echo "========================================"

    # Warm cache experiments
    echo "--- Warm Cache ---"
    run_experiment "sudo python3 compute_optimization_results_for_different_K.py $idx with $DATASET warm --iterations $ITERATIONS"
    run_experiment "sudo python3 compute_optimization_results_for_different_K.py $idx without $DATASET warm --iterations $ITERATIONS"

    # Cold cache experiments
    echo "--- Cold Cache ---"
    run_experiment "sudo python3 compute_optimization_results_for_different_K.py $idx with $DATASET cold --iterations $ITERATIONS"
    run_experiment "sudo python3 compute_optimization_results_for_different_K.py $idx without $DATASET cold --iterations $ITERATIONS"
done

# Summary
echo ""
echo "========================================"
echo "EXPERIMENT SUMMARY"
echo "========================================"

if [ ${#FAILED_COMMANDS[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!"
else
    echo "✗ ${#FAILED_COMMANDS[@]} experiment(s) failed:"
    for cmd in "${FAILED_COMMANDS[@]}"; do
        echo "  - $cmd"
    done
fi

echo ""
echo "Results are saved in: data/"
echo "To generate plots, run: python3 plot_optimization_results.py --dataset $DATASET"
echo "========================================"