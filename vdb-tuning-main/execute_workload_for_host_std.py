import psycopg2
import time
import subprocess
import json
import shutil
import os
import csv

# === CONFIG ===
K = 100
#QUERY = "SELECT key FROM image_data ORDER BY embeddings <=> %s LIMIT %s"

QUERY = "SELECT id FROM gist_data ORDER BY embedding <-> %s LIMIT %s"

#QUERY = "SELECT key, (embeddings <=> %s) as distance  FROM image_data ORDER BY distance LIMIT %s"
CONTAINER_NAME = "mypostgres2"
METRICS_PATH = "/tmp/ivfflat_metrics.json"
RUNS = 1

# === LOAD WORKLOAD ===
with open('sampled_vectors_gist.txt', 'r') as f:
    workload = eval(f.read())  # WARNING: Make sure the input is safe

per_query_accessed_pages = [] 
calculations_per_query= []

# === MAIN LOOP ===
for run in range(RUNS):

    elapsed_time_per_query=[]

    # Requires running as root (or use sudo)
    subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, check=True)

    subprocess.run(
        "su - postgres -s /bin/bash -c 'export PGDATA=/mnt/nvme/pgdata && export PATH=/usr/local/pgsql/bin:$PATH && pg_ctl -D $PGDATA -l logfile restart'",
        shell=True,
        check=True
    )
    subprocess.run(
            f"rm -f {METRICS_PATH}",
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




    # === DB CONNECTION ===
    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="secret",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    total_exec_time = 0.0
    timing_json = {
        "get_scan_lists_time": 0.0,
        "get_scan_items_time": 0.0,
        "get_next_tuple_time":0.0,
        "read_buffer_extended_overhead": 0.0,
        "calculate_distances_from_page_overhead": 0.0,
        "process_tuples_overhead": 0.0,
        "sort_overhead": 0.0,
        "total_time": 0.0,
        "calculations": 0.0
    }


    print(f"\n🚀 Run #{run+1} starting...")
    shared_hit = 0
    shared_read = 0

    for i in range(len(workload)):
        #shared_hit = 0
        #shared_read = 0


        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS) {QUERY}",
            (workload[i], K)
        )
        explain_result = cur.fetchall()
        print("execute query %d",i)
        found_main_buffers = False

        for row in explain_result:
            line = row[0].strip()

            # First Buffers line (outside Planning section)
            if 'Buffers:' in line and not found_main_buffers:
                if 'Planning:' not in line:  # Make sure not inside Planning
                    parts = line.split()
                    for part in parts:
                        if part.startswith('hit='):
                            shared_hit += int(part.split('=')[1].rstrip(','))
                        elif part.startswith('read=') and 'temp' not in line.split('read=')[0][-10:]:
                            # This is tricky, better use regex above
                            shared_read += int(part.split('=')[1].rstrip(','))

                    found_main_buffers = True  # Only take first occurrence during execution

            # Execution Time
            if 'Execution Time:' in line:
                parts = line.split()
                elapsed_time = float(parts[2]) / 1000.0  # ms -> sec
                total_exec_time += elapsed_time
                elapsed_time_per_query.append(elapsed_time);
        per_query_accessed_pages.append(shared_read+shared_hit)

        # 4. Print or store results
    try:
        # Read the full file content from inside the container
        result = subprocess.check_output(
            f"cat {METRICS_PATH}",
            shell=True,
            text=True
        )

        block = ""
        files=0
        query=0
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
                        value=entry.get(key, 0.0)
                        timing_json[key] += value
                        if(key=="calculations"):
                            calculations_per_query.append(value)
                            
                except json.JSONDecodeError as err:
                    print(f"❌ Failed to parse: {block} — {err}")
                block = ""
        print("files are ",files)

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to read file from container: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # === OUTPUT RESULTS ===
    print(f"✅ Run #{run+1} complete")
    print(f"📄 shared_hit={shared_hit}, shared_read={shared_read}")
    print(f"⏱️  Total query exec time:       {total_exec_time:.3f} sec")
    print(f"📦 get_scan_lists_time:         {timing_json['get_scan_lists_time']:.3f} ms")
    print(f"📦 get_scan_items_time:         {timing_json['get_scan_items_time']:.3f} ms")
    print(f"📦 get_next_tuple_time:         {timing_json['get_next_tuple_time']:.3f} ms")
    print(f"📦 read_buffer_extended_overhead: {timing_json['read_buffer_extended_overhead']:.3f} ms")
    print(f"📦 calculate_distances_from_page_overhead:            {timing_json['calculate_distances_from_page_overhead']:.3f} ms")
    print(f"📦 process_tuples_overhead:            {timing_json['process_tuples_overhead']:.3f} ms")
    print(f"📦 sort_overhead:            {timing_json['sort_overhead']:.3f} ms")
    print(f"📊 total_time (combined):  {timing_json['total_time']:.3f} ms")
    print(f"📦 calculations:            {timing_json['calculations']:.3f} ")

    # with open("hnsw_reads.json", "w") as f:
    #     json.dump(per_query_accessed_pages, f)

    # with open("accessed_pages_per_query_ivfflat.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Query Number', 'Accessed Pages'])  # Header
    #     for i, pages in enumerate(per_query_accessed_pages):
    #         writer.writerow([i + 1, pages]) 

    # with open("calculations_per_query_ivfflat.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Query Number', 'Calculations'])  # Header
    #     for i, pages in enumerate(calculations_per_query):
    #         writer.writerow([i + 1, calculations_per_query[i]]) 

    # with open("elapsed_time_per_query_ivfflat_normal.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Query Number', 'Execution Time (s)'])  # Header
    #     for i, time in enumerate(elapsed_time_per_query):
    #         writer.writerow([i + 1, time]) 

    cur.close()
    conn.close()
