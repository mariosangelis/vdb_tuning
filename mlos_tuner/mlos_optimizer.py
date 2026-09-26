import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlos_core.data_classes import Suggestion
import ConfigSpace as CS
import mlos_core.optimizers
import psycopg2
import time
import subprocess
import re
import json


#Open the sample vectors, they are stored in the sampled_vectors.txt file in the form of a list
workload=[]
with open('sampled_vectors.txt', 'r') as f:
    workload = eval(f.read())


results=[]
accurate_results=[]

K = 100
TOPK_QUERY = "SELECT key FROM image_data ORDER BY embeddings <=> %s LIMIT %s"
TPC_H_QUERIES_FILE = '../laion/tpch_queries.sql'  # One query per line or separated clearly
CONTAINER_NAME = "mypostgres2"
METRICS_PATH = "/tmp/ivfflat_metrics.json"


with open(TPC_H_QUERIES_FILE, "r") as f:
    content = f.read()
tpch_queries = [q.strip() for q in content.split('-- QUERY') if q.strip()]


def score_config(config: pd.Series) -> pd.Series:
    for key, value in config.items():
        if key in ["max_parallel_maintenance_workers", "maintenance_io_concurrency", "effective_io_concurrency", "io_combine_limit", "random_page_cost" , "cpu_tuple_cost","cpu_index_tuple_cost", "cpu_operator_cost", "parallel_setup_cost", "parallel_tuple_cost"]:
            exec_command = f"sudo sed -i 's/^{key} = .*/{key} = {value}/' /home/postgres/pgdata/postgresql.conf"
        else:
            exec_command = f"sudo sed -i 's/^{key} = .*/{key} = {value}kB/' /home/postgres/pgdata/postgresql.conf"

        subprocess.run(exec_command, shell=True, check=True)

    # Corrected drop caches:
    subprocess.run('sudo bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"', shell=True, check=True)

    # Restart postgres
    subprocess.run(
        "sudo su - postgres -s /bin/bash -c 'export PGDATA=$HOME/pgdata && export PATH=/usr/local/pgsql/bin:$PATH && pg_ctl -D $PGDATA -l logfile restart'",
        shell=True,
        check=True
    )

    subprocess.run(
        f"sudo rm -f {METRICS_PATH}",
        shell=True,
        check=True
    )

    for attempt in range(1000):
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
            break
        except psycopg2.OperationalError:
            print(f"⏳ Waiting for PostgreSQL to accept connections... attempt {attempt + 1}")
            time.sleep(2)
    else:
        raise Exception("❌ PostgreSQL did not become ready in time.")


    # Connect to Postgres and run the index creation
    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="secret",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    #cur.execute("DROP INDEX IF EXISTS image_data_embeddings_idx", ())
    #conn.commit()
    print("WORKLOAD QUERY EXECUTION STARTED");

    # === DB CONNECTION ===
    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="secret",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    total_exec_time_topk = 0.0
    total_exec_time_tpch = 0.0
    timing_json = {
        "get_scan_lists_time": 0.0,
        "get_scan_items_time": 0.0,
        "get_next_tuple_time":0.0,
        "total_time": 0.0
    }

    shared_hit = 0
    shared_read = 0
    next_query_index=0

    for i in range(len(workload)):
        # === 1. Run TopK query ===
        #cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {TOPK_QUERY}", (workload[i], K))
        #explain_result = cur.fetchall()

        #found_main_buffers = False
        #for row in explain_result:
        #    line = row[0].strip()
        #    if 'Execution Time:' in line:
        #        total_exec_time_topk += float(line.split()[2]) / 1000.0


        # === 2. Run TPC-H query ===
        tpch_query = tpch_queries[next_query_index]
        #print("execute query ",next_query_index)
        next_query_index+=1
        if(next_query_index==22):
            next_query_index=0
        try:
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {tpch_query}")
            explain_result = cur.fetchall()

            found_main_buffers = False
            for row in explain_result:
                line = row[0].strip()
                if 'Execution Time:' in line:
                    total_exec_time_tpch += float(line.split()[2]) / 1000.0

        except Exception as e:
            print(f"⚠️ Error running TPC-H query: {e}")


    print("ALL QUERIES EXECUTED");
    try:
        # Read the full file content from inside the container
        result = subprocess.check_output(
            f"sudo cat {METRICS_PATH}",
            shell=True,
            text=True
        )

        block = ""
        files=0
        for line in result.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            block += line
            if line.endswith("}"):
                files+=1
                try:
                    entry = json.loads(block)
                    for key in timing_json:
                        timing_json[key] += entry.get(key, 0.0)
                except json.JSONDecodeError as err:
                    print(f"❌ Failed to parse: {block} — {err}")
                block = ""
        print("files are ",files)

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to read file from container: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    total_exec_time = total_exec_time_tpch + total_exec_time_topk

    cur.close()
    conn.close()

    print("ANALYSIS ENDED");

    result = {
        "config": config.to_dict(),
        "total_exec_time": total_exec_time_tpch+total_exec_time_topk,
        "total_exec_time_tpch": total_exec_time_tpch,
        "total_exec_time_topk": total_exec_time_topk,
        "timings": timing_json
    }

    # Save to a results file for later analysis
    with open("mixed_workload_TPCH_ONLY_BAS_BULKREAD_FULL_metrics_eith_topk_optimizations_ON.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    print("✅ Config tested:", config.to_dict())
    print("⏱️  Total exec time:", total_exec_time)
    print("🧪  Timings:", timing_json)
    return pd.Series({"score": total_exec_time})

# Start defining a ConfigurationSpace for the Optimizer to search.
input_space = CS.ConfigurationSpace(seed=1234)

#Memory Knobs For ivfflat build
#input_space.add(CS.UniformIntegerHyperparameter(name='shared_buffers', lower=300000, upper=3000000))
#input_space.add(CS.UniformIntegerHyperparameter(name='max_wal_size', lower=1000000, upper=4000000))
#input_space.add(CS.UniformIntegerHyperparameter(name='maintenance_work_mem', lower=300000, upper=3000000))

#System Knobs For ivfflat build
#input_space.add(CS.UniformIntegerHyperparameter(name='max_parallel_maintenance_workers', lower=1, upper=16))
#input_space.add(CS.UniformIntegerHyperparameter(name='maintenance_io_concurrency', lower=1, upper=1000))


#Memory Knobs For ivfflat scan
input_space.add(CS.UniformIntegerHyperparameter(name='shared_buffers', lower=30000, upper=4000000))
input_space.add(CS.UniformIntegerHyperparameter(name='work_mem', lower=64, upper=3000000))
input_space.add(CS.UniformIntegerHyperparameter(name='temp_buffers', lower=64, upper=3000000))
input_space.add(CS.UniformIntegerHyperparameter(name='effective_io_concurrency', lower=1, upper=1000))
input_space.add(CS.UniformFloatHyperparameter('random_page_cost', 1.0, 4.0))
input_space.add(CS.UniformFloatHyperparameter('cpu_tuple_cost', 0.001, 0.05))
input_space.add(CS.UniformFloatHyperparameter('cpu_index_tuple_cost', 0.001, 0.03))
input_space.add(CS.UniformFloatHyperparameter('cpu_operator_cost', 0.0005, 0.01))
input_space.add(CS.UniformFloatHyperparameter('parallel_setup_cost', 100, 1000))
input_space.add(CS.UniformFloatHyperparameter('parallel_tuple_cost', 0.01, 0.2))
input_space.add(CS.UniformIntegerHyperparameter('effective_cache_size', 30000, 8000000))

#System Knobs For ivfflat scan
#input_space.add(CS.UniformIntegerHyperparameter(name='shared_buffers', lower=30000, upper=4000000))
#input_space.add(CS.UniformIntegerHyperparameter(name='effective_io_concurrency', lower=1, upper=1000))
#input_space.add(CS.UniformIntegerHyperparameter(name='io_combine_limit', lower=8, upper=256))



optimizer = mlos_core.optimizers.SmacOptimizer(parameter_space=input_space, optimization_targets=['score']) # , seed=42, n_random_init=20)


def run_optimization(optimizer: mlos_core.optimizers.BaseOptimizer):
    # get a new config value suggestion to try from the optimizer.
    suggestion = optimizer.suggest()
    # suggested value are dictionary-like, keys are input space parameter names
    # evaluate target function
    scores = score_config(suggestion.config)
    # print(suggested_value, "\n", scores)
    optimizer.register(suggestion.complete(scores))

# run for some iterations
n_iterations = 200
for i in range(n_iterations):
    run_optimization(optimizer)

observations = optimizer.get_observations()
best_suggestion = min(observations, key=lambda x: x.score.item())

print("Best Configuration:", best_suggestion.config)
print("Best Score:", best_suggestion.score)

worse_suggestion = max(observations, key=lambda x: x.score.item())
print("worse Configuration:", worse_suggestion.config)
print("worse Score:", worse_suggestion.score)
