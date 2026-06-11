#!/usr/bin/env python3
import psycopg2
import time
import subprocess
import json
import os
import csv
import argparse
import numpy as np
import pandas as pd

# === CONFIG ===
K = 100
QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> %s::vector LIMIT %s"
METRICS_PATH = "/tmp/ivfflat_metrics.json"

# Load configuration
def load_config():
    config_path = "TPCH/code/experiment_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        print(f"Warning: Could not load {config_path}, using defaults")
        config = {"hardware": {"postgres_numa_nodes": "0-7", "postgres_cores": "0-31"}}

    # Extract hardware configuration
    hardware = config.get('hardware', {})
    postgres_numa_nodes = hardware.get('postgres_numa_nodes', '0-7')
    postgres_cores = hardware.get('postgres_cores', '0-31')

    # Determine NUMA nodes from cores
    if '-' in postgres_cores:
        start, end = map(int, postgres_cores.split('-'))
        start_numa = start // 4
        end_numa = end // 4
        if start_numa == end_numa:
            numa_nodes = str(start_numa)
        else:
            numa_nodes = f"{start_numa}-{end_numa}"
    else:
        core = int(postgres_cores)
        numa_nodes = str(core // 4)

    return config, numa_nodes, postgres_cores

def restart_postgres_with_pinning(numa_nodes, cpu_cores):
    """Restart PostgreSQL with CPU and NUMA pinning"""
    print(f"Restarting PostgreSQL with NUMA nodes: {numa_nodes}, CPU cores: {cpu_cores}")

    # First stop PostgreSQL if running
    stop_cmd = (
        f"su - postgres -s /bin/bash -c '"
        f"export PGDATA=/mnt/nvme/pgdata && "
        f"export PATH=/usr/local/pgsql/bin:$PATH && "
        f"pg_ctl -D $PGDATA stop -m fast'"
    )
    subprocess.run(stop_cmd, shell=True, check=False)  # Ignore if not running
    time.sleep(2)

    # Start with specific CPU pinning
    # For single core, we need to ensure all processes stay on that core
    start_cmd = (
        f"su - postgres -s /bin/bash -c '"
        f"export PGDATA=/mnt/nvme/pgdata && "
        f"export PATH=/usr/local/pgsql/bin:$PATH && "
        f"taskset -c {cpu_cores} numactl --cpunodebind={numa_nodes} --membind={numa_nodes} "
        f"postgres -D $PGDATA' > /tmp/postgres_single_core.log 2>&1 &"
    )

    # Use postgres command directly instead of pg_ctl to ensure taskset applies
    subprocess.run(start_cmd, shell=True, check=True)
    time.sleep(3)  # Wait for startup

    # Set CPU affinity for all PostgreSQL processes
    if cpu_cores == "0" or not "-" in cpu_cores:
        # Single core - force all postgres processes to it
        print(f"Forcing all PostgreSQL processes to CPU {cpu_cores}...")
        force_cmd = f"for pid in $(pgrep postgres); do taskset -pc {cpu_cores} $pid 2>/dev/null; done"
        subprocess.run(force_cmd, shell=True)

def drop_caches():
    """Drop system caches"""
    print("Dropping system caches...")
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

def prewarm_table_and_index(table_name="gist_data"):
    """Prewarm table and index data into shared buffers"""
    print(f"Prewarming table {table_name} and its indexes...")

    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="secret",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Enable pg_prewarm extension if not already enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")

    # Prewarm the table
    cur.execute(f"SELECT pg_prewarm('{table_name}')")
    pages_prewarmed_table = cur.fetchone()[0]
    print(f"  Prewarmed {pages_prewarmed_table} pages from table {table_name}")

    # Get all indexes on the table
    cur.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = %s
    """, (table_name,))

    indexes = cur.fetchall()

    # Prewarm each index
    for (index_name,) in indexes:
        cur.execute(f"SELECT pg_prewarm('{index_name}')")
        pages_prewarmed = cur.fetchone()[0]
        print(f"  Prewarmed {pages_prewarmed} pages from index {index_name}")

    conn.commit()
    cur.close()
    conn.close()

def wait_for_postgres():
    """Wait for PostgreSQL to be ready"""
    for attempt in range(100):
        try:
            conn = psycopg2.connect(
                dbname="mydb",
                user="postgres",
                password="secret",
                host="localhost",
                port="5432"
            )
            conn.close()
            print("✅ PostgreSQL is ready.")
            return
        except psycopg2.OperationalError:
            print(f"⏳ Waiting for PostgreSQL... attempt {attempt + 1}")
            time.sleep(2)

    raise Exception("❌ PostgreSQL did not become ready in time.")

def print_statistics(elapsed_times, iteration):
    """Print detailed statistics for query times"""
    times_ms = [t * 1000 for t in elapsed_times]  # Convert to milliseconds

    print(f"\n📊 Iteration {iteration} Statistics:")
    print(f"  Queries executed: {len(times_ms)}")
    print(f"  Mean: {np.mean(times_ms):.2f} ms")
    print(f"  Median: {np.median(times_ms):.2f} ms")
    print(f"  Std Dev: {np.std(times_ms):.2f} ms")
    print(f"  Min: {np.min(times_ms):.2f} ms")
    print(f"  Max: {np.max(times_ms):.2f} ms")
    print(f"  Percentiles:")
    print(f"    P50: {np.percentile(times_ms, 50):.2f} ms")
    print(f"    P90: {np.percentile(times_ms, 90):.2f} ms")
    print(f"    P95: {np.percentile(times_ms, 95):.2f} ms")
    print(f"    P99: {np.percentile(times_ms, 99):.2f} ms")

    return {
        'mean': np.mean(times_ms),
        'median': np.median(times_ms),
        'std': np.std(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms),
        'p50': np.percentile(times_ms, 50),
        'p90': np.percentile(times_ms, 90),
        'p95': np.percentile(times_ms, 95),
        'p99': np.percentile(times_ms, 99)
    }

def main():
    parser = argparse.ArgumentParser(description='Execute workload with statistics')
    parser.add_argument('--warm', action='store_true', help='Warm mode: prewarm data after restart')
    parser.add_argument('--cold', action='store_true', help='Cold mode: drop caches and restart')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to run')
    parser.add_argument('--output-prefix', type=str, default='workload_results', help='Prefix for output files')
    parser.add_argument('--cores', type=str, help='Override CPU cores (e.g., "0" or "0-3")')
    args = parser.parse_args()

    # Default to cold mode if neither specified
    if not args.warm and not args.cold:
        args.cold = True

    # Load configuration
    config, numa_nodes, cpu_cores = load_config()

    # Override cores if specified
    if args.cores:
        cpu_cores = args.cores
        # Recalculate NUMA nodes
        if '-' in cpu_cores:
            start, end = map(int, cpu_cores.split('-'))
            start_numa = start // 4
            end_numa = end // 4
            if start_numa == end_numa:
                numa_nodes = str(start_numa)
            else:
                numa_nodes = f"{start_numa}-{end_numa}"
        else:
            core = int(cpu_cores)
            numa_nodes = str(core // 4)
        print(f"Override: Using CPU cores={cpu_cores}, NUMA nodes={numa_nodes}")

    # === LOAD WORKLOAD ===
    try:
        with open('gist_test_queries.txt', 'r') as f:
            workload = eval(f.read())  # WARNING: Make sure the input is safe
    except Exception as e:
        print(f"Error loading workload: {e}")
        return

    all_iterations_stats = []
    all_query_times = []

    # === MAIN LOOP ===
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"🚀 ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")

        elapsed_time_per_query = []

        # Handle cold vs warm mode
        if args.cold:
            print("\n🧊 COLD MODE: Dropping caches and restarting PostgreSQL...")
            drop_caches()
            restart_postgres_with_pinning(numa_nodes, cpu_cores)
        else:
            print("\n🔥 WARM MODE: Restarting PostgreSQL (keeping caches)...")
            restart_postgres_with_pinning(numa_nodes, cpu_cores)

        # Clean metrics file
        subprocess.run(f"rm -f {METRICS_PATH}", shell=True, check=True)

        # Wait for PostgreSQL
        wait_for_postgres()

        # If warm mode, prewarm the data
        if args.warm:
            print("🔥 Prewarming data...")
            prewarm_table_and_index()

        # === DB CONNECTION ===
        conn = psycopg2.connect(
            dbname="mydb",
            user="postgres",
            password="secret",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        # Set search parameters
        cur.execute("SET enable_seqscan = off")

        total_exec_time = 0.0
        shared_hit = 0
        shared_read = 0

        print(f"\n📝 Executing {len(workload)} queries...")

        # Progress reporting
        report_interval = max(1, len(workload) // 10)

        for i in range(len(workload)):
            if i > 0 and i % report_interval == 0:
                print(f"  Progress: {i}/{len(workload)} queries ({i*100//len(workload)}%)")

            cur.execute(
                f"EXPLAIN (ANALYZE, BUFFERS) {QUERY}",
                (str(workload[i]), K)
            )
            explain_result = cur.fetchall()
            found_main_buffers = False

            for row in explain_result:
                line = row[0].strip()

                # First Buffers line (outside Planning section)
                if 'Buffers:' in line and not found_main_buffers:
                    if 'Planning:' not in line:
                        parts = line.split()
                        for part in parts:
                            if part.startswith('hit='):
                                shared_hit += int(part.split('=')[1].rstrip(','))
                            elif part.startswith('read=') and 'temp' not in line.split('read=')[0][-10:]:
                                shared_read += int(part.split('=')[1].rstrip(','))
                        found_main_buffers = True

                # Execution Time
                if 'Execution Time:' in line:
                    parts = line.split()
                    elapsed_time = float(parts[2]) / 1000.0  # ms -> sec
                    total_exec_time += elapsed_time
                    elapsed_time_per_query.append(elapsed_time)

        # Print statistics for this iteration
        stats = print_statistics(elapsed_time_per_query, iteration)
        stats['iteration'] = iteration
        all_iterations_stats.append(stats)
        all_query_times.extend(elapsed_time_per_query)

        # === OUTPUT RESULTS ===
        mode_str = "WARM" if args.warm else "COLD"
        print(f"\n✅ Iteration {iteration} complete ({mode_str} mode)")
        print(f"📄 shared_hit={shared_hit}, shared_read={shared_read}")
        print(f"⏱️  Total query exec time: {total_exec_time:.3f} sec")

        # Calculate cache hit ratio
        if shared_hit + shared_read > 0:
            cache_hit_ratio = shared_hit / (shared_hit + shared_read) * 100
            print(f"💾 Cache hit ratio: {cache_hit_ratio:.1f}%")

        # Save per-iteration results
        iteration_file = f"{args.output_prefix}_iter{iteration}_query_times.csv"
        with open(iteration_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query Number', 'Execution Time (s)', 'Execution Time (ms)'])
            for idx, time_sec in enumerate(elapsed_time_per_query):
                writer.writerow([idx + 1, time_sec, time_sec * 1000])
        print(f"📁 Saved query times to {iteration_file}")

        cur.close()
        conn.close()

    # === OVERALL STATISTICS ===
    if args.iterations > 1:
        print(f"\n{'='*60}")
        print("📊 OVERALL STATISTICS ACROSS ALL ITERATIONS")
        print(f"{'='*60}")

        all_times_ms = [t * 1000 for t in all_query_times]
        print(f"Total queries executed: {len(all_times_ms)}")
        print(f"Overall Mean: {np.mean(all_times_ms):.2f} ms")
        print(f"Overall Median: {np.median(all_times_ms):.2f} ms")
        print(f"Overall Std Dev: {np.std(all_times_ms):.2f} ms")
        print(f"Overall Min: {np.min(all_times_ms):.2f} ms")
        print(f"Overall Max: {np.max(all_times_ms):.2f} ms")
        print(f"\nPercentiles across all queries:")
        print(f"  P50: {np.percentile(all_times_ms, 50):.2f} ms")
        print(f"  P90: {np.percentile(all_times_ms, 90):.2f} ms")
        print(f"  P95: {np.percentile(all_times_ms, 95):.2f} ms")
        print(f"  P99: {np.percentile(all_times_ms, 99):.2f} ms")

        # === SUMMARY LISTS ===
        print(f"\n{'='*60}")
        print("📋 SUMMARY LISTS BY ITERATION")
        print(f"{'='*60}")

        # Collect per-iteration statistics
        sums = []
        means = []
        medians = []
        p50s = []
        p90s = []
        p95s = []
        p99s = []
        std_devs = []

        # Load each iteration's data and compute stats
        for i in range(1, args.iterations + 1):
            try:
                iteration_file = f"{args.output_prefix}_iter{i}_query_times.csv"
                df = pd.read_csv(iteration_file)
                times_ms = df['Execution Time (ms)'].values

                sums.append(np.sum(times_ms))
                means.append(np.mean(times_ms))
                medians.append(np.median(times_ms))
                p50s.append(np.percentile(times_ms, 50))
                p90s.append(np.percentile(times_ms, 90))
                p95s.append(np.percentile(times_ms, 95))
                p99s.append(np.percentile(times_ms, 99))
                std_devs.append(np.std(times_ms))
            except:
                print(f"Warning: Could not load data for iteration {i}")

        # Print lists
        print(f"\nSums (total ms per iteration):")
        print(f"  {[f'{s:.2f}' for s in sums]}")

        print(f"\nMeans (ms):")
        print(f"  {[f'{m:.2f}' for m in means]}")

        print(f"\nMedians (ms):")
        print(f"  {[f'{m:.2f}' for m in medians]}")

        print(f"\nStandard Deviations (ms):")
        print(f"  {[f'{s:.2f}' for s in std_devs]}")

        print(f"\nP50 (ms):")
        print(f"  {[f'{p:.2f}' for p in p50s]}")

        print(f"\nP90 (ms):")
        print(f"  {[f'{p:.2f}' for p in p90s]}")

        print(f"\nP95 (ms):")
        print(f"  {[f'{p:.2f}' for p in p95s]}")

        print(f"\nP99 (ms):")
        print(f"  {[f'{p:.2f}' for p in p99s]}")

        # Calculate variance metrics
        if len(means) > 1:
            mean_cv = (np.std(means) / np.mean(means)) * 100
            median_cv = (np.std(medians) / np.mean(medians)) * 100
            print(f"\n📊 Consistency Metrics:")
            print(f"  Mean CV across iterations: {mean_cv:.2f}%")
            print(f"  Median CV across iterations: {median_cv:.2f}%")
            print(f"  Max-Min mean difference: {max(means) - min(means):.2f} ms")
            print(f"  Max-Min median difference: {max(medians) - min(medians):.2f} ms")

        # Save summary statistics
        summary_file = f"{args.output_prefix}_summary_stats.json"
        summary_data = {
            'mode': 'WARM' if args.warm else 'COLD',
            'iterations': args.iterations,
            'cpu_cores': cpu_cores,
            'numa_nodes': numa_nodes,
            'queries_per_iteration': len(workload),
            'total_queries': len(all_times_ms),
            'overall_stats': {
                'mean_ms': np.mean(all_times_ms),
                'median_ms': np.median(all_times_ms),
                'std_ms': np.std(all_times_ms),
                'min_ms': np.min(all_times_ms),
                'max_ms': np.max(all_times_ms),
                'p50_ms': np.percentile(all_times_ms, 50),
                'p90_ms': np.percentile(all_times_ms, 90),
                'p95_ms': np.percentile(all_times_ms, 95),
                'p99_ms': np.percentile(all_times_ms, 99)
            },
            'per_iteration_stats': all_iterations_stats,
            'summary_lists': {
                'sums': sums,
                'means': means,
                'medians': medians,
                'std_devs': std_devs,
                'p50s': p50s,
                'p90s': p90s,
                'p95s': p95s,
                'p99s': p99s
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📁 Saved summary statistics to {summary_file}")

if __name__ == "__main__":
    main()